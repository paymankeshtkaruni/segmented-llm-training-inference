#!/usr/bin/env python
"""X1 — training exactness for ALL SIX training modes, at the paper's model size (0.84B).

Dials: dropout x recomputation x streaming (streaming => recomputation), giving:

  T1_drop_resident       dropout 0.1 | recompute off | streaming off
  T2_drop_recompute      dropout 0.1 | recompute on  | streaming off
  T3_drop_streaming      dropout 0.1 | recompute on  | streaming on
  T4_nodrop_resident     dropout 0.0 | recompute off | streaming off
  T5_nodrop_recompute    dropout 0.0 | recompute on  | streaming off
  T6_nodrop_streaming    dropout 0.0 | recompute on  | streaming on

Checks:
  dropout OFF (T4-T6): full identity vs the reference model — eval forward hidden,
    gradient slices, one AdamW step compared over EVERY parameter via reassembly,
    then after --steps total steps.
  dropout ON (T1-T3): (a) eval-forward identity vs reference (dropout inactive in
    eval — verifies weights/wiring); (b) DETERMINISM: the same seeded backward run
    twice must give bit-identical gradients (recompute must replay its recorded
    masks exactly); (c) CROSS-MODE: gradients vs the T1 full-graph anchor under the
    same seed — agreement means the record/recompute path draws the same masks the
    plain forward does.

Store binding follows the paper: pass --store cpu_ram on GPU, --store disk on CPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace, fields
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from techniques import Tech                                              # noqa: E402
from config import get_preset                                            # noqa: E402
from modules import ReferenceGPTDecoder, causal_lm_cross_entropy_loss    # noqa: E402
from forward_engine import populate_from_reference, SegmentedForwardEngine  # noqa: E402
from backward_engine import SegmentedBackwardEngine                      # noqa: E402
from optimizer import SegmentwiseAdamW                                   # noqa: E402
from loader import StrictSegmentLoader                                   # noqa: E402
from stores import make_store, SegmentKey                                # noqa: E402
from export import reassemble_state_dict                                 # noqa: E402
from segments import head_range, hidden_range                            # noqa: E402

# tech-code flag order: [sdpa, mlp_sum, ce, recompute, stream, records, park, adam,
#                        segment_wo, free_device, no_kv_cache]
# (name, tech_code, dropout, update_style). Update style is a protocol dial:
# "after_full" = separate optimizer sweep with GLOBAL clip (reference uses clip too);
# "immediate"  = per-segment update as gradients finalize -> NO global clip is
# possible, so its reference protocol is AdamW without clipping. Value-wise the
# within-step update timing is irrelevant (no updated weight is re-read in the
# same step), so exactness compares the no-clip update math.
TRAIN_MODES = [
    ("T1_drop_resident_after",     "11100111000", 0.1, "after_full"),
    ("T2_drop_recompute_after",    "11110111000", 0.1, "after_full"),
    ("T3_drop_streaming_after",    "11111111100", 0.1, "after_full"),
    ("T4_nodrop_resident_after",   "11100111000", 0.0, "after_full"),
    ("T5_nodrop_recompute_after",  "11110111000", 0.0, "after_full"),
    ("T6_nodrop_streaming_after",  "11111111100", 0.0, "after_full"),
    ("T7_drop_resident_immed",     "11100111000", 0.1, "immediate"),
    ("T8_drop_recompute_immed",    "11110111000", 0.1, "immediate"),
    ("T9_drop_streaming_immed",    "11111111100", 0.1, "immediate"),
    ("T10_nodrop_resident_immed",  "11100111000", 0.0, "immediate"),
    ("T11_nodrop_recompute_immed", "11110111000", 0.0, "immediate"),
    ("T12_nodrop_streaming_immed", "11111111100", 0.0, "immediate"),
]
LR, WD, BETAS, EPS, CLIP = 3e-4, 0.1, (0.9, 0.95), 1e-8, 1.0
SEED_DATA, SEED_BWD = 1000, 123


def tech_from_code(code: str) -> Tech:
    names = [f.name for f in fields(Tech)]
    return Tech(**{n: c == "1" for n, c in zip(names, code)})


def _batch(m, step: int, batch: int, seq: int):
    g = torch.Generator().manual_seed(SEED_DATA + step)
    x = torch.randint(0, m.vocab_size, (batch, seq), generator=g)
    lab = x.clone()
    lab[:, :5] = -100
    return x, lab


def _shared_state(shared):
    return {k: v.detach().to("cpu") for k, v in shared.state_dict().items()}


def _max_delta_all_params(ref, store, shared, m, s):
    """Worst |delta| over every reassembled parameter, plus the number of
    elements above 1e-5 and 1e-6 over ALL parameters -- the same counts
    control_fp32_floor.py reports, so the two are comparable like for like."""
    sd = reassemble_state_dict(store, _shared_state(shared), m, s)
    worst, worst_name = 0.0, ""
    n5 = n6 = ntot = 0
    rsd = ref.state_dict()
    for k, v in sd.items():
        diff = (rsd[k].detach().to("cpu") - v.to("cpu")).abs()
        d = diff.max().item()
        if d > worst:
            worst, worst_name = d, k
        n5 += int((diff > 1e-5).sum().item())
        n6 += int((diff > 1e-6).sum().item())
        ntot += diff.numel()
    return worst, worst_name, {"n_elements_gt_1e-5": n5, "n_elements_gt_1e-6": n6,
                               "n_elements": ntot}


def _grad_slices(gstore, shared_grads, m, s):
    """Representative gradient slices as CPU tensors, keyed by name."""
    out = {}
    out["mlp_chunk0_L0"] = gstore.get(SegmentKey(0, "mlp", 0))["input_projection.weight"].to("cpu").clone()
    out["attn_q_seg0_L0"] = gstore.get(SegmentKey(0, "attention", 0))["q_proj.weight"].to("cpu").clone()
    out["output_head_seg0"] = gstore.get(SegmentKey(-1, "output_head", 0))["projection.weight"].to("cpu").clone()
    out["final_norm_w"] = shared_grads["final_norm.weight"].to("cpu").clone()
    return out


def _slice_delta(a: dict, b: dict) -> dict:
    return {k: (a[k] - b[k]).abs().max().item() for k in a}


def _build(mode_code, dropout, device, store_kind, store_root: Path, preset_name,
           update_style="after_full"):
    tech = tech_from_code(mode_code)
    preset = get_preset(preset_name)
    m = replace(preset["model"], dropout=dropout)
    s = preset["seg"]
    torch.manual_seed(0)
    ref = ReferenceGPTDecoder(m)                       # built on CPU, identical every call
    if store_root.exists():
        shutil.rmtree(store_root)

    def mk(name):
        return make_store(store_kind) if store_kind == "cpu_ram" else make_store(store_kind, store_root / name)

    store = mk("params")
    shared = populate_from_reference(ref, m, s, store).to(device)
    ref = ref.to(device)
    loader = StrictSegmentLoader(m, s, store, device, tech=tech)
    fwd = SegmentedForwardEngine(m, s, loader, shared, device)
    gstore, ostore, rstore = mk("grads"), mk("opt"), mk("records")
    bwd = SegmentedBackwardEngine(fwd, gstore, rstore)
    opt = SegmentwiseAdamW(m, s, loader, shared, gstore, ostore, device,
                           lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    if update_style == "immediate":
        bwd.set_immediate_optimizer(opt)   # REAL immediate scheduling, not a proxy
    return m, s, ref, store, shared, loader, fwd, gstore, bwd, opt


def _seg_backward(bwd, fwd, x, lab):
    torch.manual_seed(SEED_BWD)
    fwd.train()
    return bwd.backward(x, lab, pad_token_id=None)


def verify_nodrop(name, code, device, store_kind, store_root, preset_name, batch, seq,
                  n_steps, clip, style="after_full"):
    m, s, ref, store, shared, loader, fwd, gstore, bwd, opt = _build(
        code, 0.0, device, store_kind, store_root, preset_name, update_style=style)
    decay = [p for p in ref.parameters() if p.dim() >= 2]
    nodecay = [p for p in ref.parameters() if p.dim() < 2]
    ref_opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                                 {"params": nodecay, "weight_decay": 0.0}],
                                lr=LR, betas=BETAS, eps=EPS)
    out = {"mode": name, "tech_code": code, "dropout": 0.0,
           "update_style": style, "clip": clip}

    x, lab = _batch(m, 0, batch, seq)
    x, lab = x.to(device), lab.to(device)
    ref.eval(); fwd.eval()
    with torch.no_grad():
        _, h_ref = ref(x, pad_token_id=None)
        h_seg = fwd.forward_hidden(x, pad_token_id=None)
    out["forward_hidden_max_delta"] = (h_ref - h_seg).abs().max().item()
    del h_ref, h_seg

    ref.train()
    ref.zero_grad(set_to_none=False)
    logits, _ = ref(x, pad_token_id=None)
    causal_lm_cross_entropy_loss(logits, lab).backward()
    del logits
    shared_grads = _seg_backward(bwd, fwd, x, lab)   # immediate: segments update HERE

    if style == "after_full":                        # immediate stores no gradients
        seg_sl = _grad_slices(gstore, shared_grads, m, s)
        f0, f1 = hidden_range(m.d_ff, s.mlp_chunks, 0)
        h0, h1 = head_range(m.n_heads, s.attention_segments, 0)
        hd = m.d_model // m.n_heads
        oh_rows = seg_sl["output_head_seg0"].shape[0]
        ref_sl = {
            "mlp_chunk0_L0": ref.blocks[0].mlp.input_projection.weight.grad[f0:f1].to("cpu"),
            "attn_q_seg0_L0": ref.blocks[0].attention.qkv_projection.weight.grad[:m.d_model][h0 * hd:h1 * hd].to("cpu"),
            "output_head_seg0": ref.output_projection.weight.grad[:oh_rows].to("cpu"),
            "final_norm_w": ref.final_norm.weight.grad.to("cpu"),
        }
        out["grad_slice_deltas"] = _slice_delta(ref_sl, seg_sl)
        out["grad_max_delta"] = max(out["grad_slice_deltas"].values())

    if clip is not None:
        torch.nn.utils.clip_grad_norm_(ref.parameters(), clip)
    ref_opt.step()
    if style == "immediate":
        opt.step_shared(shared_grads)                # segments already updated in-backward
    else:
        opt.step(shared_grads, clip_norm=clip)
    d1, n1, c1 = _max_delta_all_params(ref, store, shared, m, s)
    out["step1_all_params_max_delta"], out["step1_worst_param"] = d1, n1
    out["step1_all_params_counts"] = c1

    for step in range(1, n_steps):
        x, lab = _batch(m, step, batch, seq)
        x, lab = x.to(device), lab.to(device)
        ref.zero_grad(set_to_none=False)
        logits, _ = ref(x, pad_token_id=None)
        causal_lm_cross_entropy_loss(logits, lab).backward()
        del logits
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(ref.parameters(), clip)
        ref_opt.step()
        sg = _seg_backward(bwd, fwd, x, lab)
        if style == "immediate":
            opt.step_shared(sg)
        else:
            opt.step(sg, clip_norm=clip)
    dn, nn_, cn = _max_delta_all_params(ref, store, shared, m, s)
    out[f"step{n_steps}_all_params_max_delta"], out[f"step{n_steps}_worst_param"] = dn, nn_
    out[f"step{n_steps}_all_params_counts"] = cn
    return out


def verify_drop(name, code, device, store_kind, store_root, preset_name, batch, seq,
                anchor, style):
    m, s, ref, store, shared, loader, fwd, gstore, bwd, opt = _build(
        code, 0.1, device, store_kind, store_root, preset_name)
    out = {"mode": name, "tech_code": code, "dropout": 0.1, "update_style": style,
           "note": "gradient-level checks; update style cannot alter gradients"}

    x, lab = _batch(m, 0, batch, seq)
    x, lab = x.to(device), lab.to(device)
    ref.eval(); fwd.eval()
    with torch.no_grad():
        _, h_ref = ref(x, pad_token_id=None)
        h_seg = fwd.forward_hidden(x, pad_token_id=None)
    out["eval_forward_hidden_max_delta"] = (h_ref - h_seg).abs().max().item()
    del h_ref, h_seg

    # (b) determinism: same seeded backward twice -> bit-identical grads
    sg1 = _seg_backward(bwd, fwd, x, lab)
    sl1 = _grad_slices(gstore, sg1, m, s)
    sg2 = _seg_backward(bwd, fwd, x, lab)
    sl2 = _grad_slices(gstore, sg2, m, s)
    out["determinism_deltas"] = _slice_delta(sl1, sl2)
    out["determinism_max_delta"] = max(out["determinism_deltas"].values())

    # (c) cross-mode: vs the T1 full-graph anchor (same weights, same seed)
    if anchor is not None:
        out["vs_T1_anchor_deltas"] = _slice_delta(anchor, sl1)
        out["vs_T1_anchor_max_delta"] = max(out["vs_T1_anchor_deltas"].values())
    return out, sl1


def _diff_state_dicts(a, b):
    worst, worst_name = 0.0, ""
    for k in a:
        d = (a[k].to("cpu") - b[k].to("cpu")).abs().max().item()
        if d > worst:
            worst, worst_name = d, k
    return worst, worst_name


def verify_drop_imm(name, code, device, store_kind, store_root, preset_name, batch, seq):
    """Dropout + IMMEDIATE: gradients are consumed in-backward, so grad-level checks
    cannot exist. Instead run one seeded step through the REAL immediate path and
    through its after_full twin (clip None, same weights, same masks) — the two must
    yield the same parameters (bit-exact on CPU; summation-order rounding on GPU)."""
    out = {"mode": name, "tech_code": code, "dropout": 0.1, "update_style": "immediate"}
    x = lab = None
    sds = {}
    for style in ("immediate", "after_full"):
        m, s, ref, store, shared, loader, fwd, gstore, bwd, opt = _build(
            code, 0.1, device, store_kind, store_root, preset_name, update_style=style)
        if x is None:
            xt, labt = _batch(m, 0, batch, seq)
            x, lab = xt.to(device), labt.to(device)
            ref.eval(); fwd.eval()
            with torch.no_grad():
                _, h_ref = ref(x, pad_token_id=None)
                h_seg = fwd.forward_hidden(x, pad_token_id=None)
            out["eval_forward_hidden_max_delta"] = (h_ref - h_seg).abs().max().item()
            del h_ref, h_seg
        sg = _seg_backward(bwd, fwd, x, lab)
        if style == "immediate":
            opt.step_shared(sg)
        else:
            opt.step(sg, clip_norm=None)
        sds[style] = reassemble_state_dict(store, _shared_state(shared), m, s)
        del ref, store, shared, loader, fwd, gstore, bwd, opt
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    d, n = _diff_state_dicts(sds["immediate"], sds["after_full"])
    out["imm_vs_afterfull_all_params_max_delta"] = d
    out["worst_param"] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--store", required=True, choices=["cpu_ram", "disk"])
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "verify_modes")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    store_root = a.out_dir / f"_stores_{'gpu' if a.device.startswith('cuda') else 'cpu'}"

    results, anchor = [], None
    for name, code, dropout, style in TRAIN_MODES:
        clip = CLIP if style == "after_full" else None
        print(f"\n===== X1 {name} on {a.device} store={a.store} style={style} "
              f"({a.preset}) =====", flush=True)
        if dropout == 0.0:
            r = verify_nodrop(name, code, a.device, a.store, store_root, a.preset,
                              a.batch, a.seq, a.steps, clip, style)
            print(f"  eval forward   max|D| = {r['forward_hidden_max_delta']:.3e}")
            if "grad_max_delta" in r:
                print(f"  gradients      max|D| = {r['grad_max_delta']:.3e}")
            print(f"  1 step (all p) max|D| = {r['step1_all_params_max_delta']:.3e} ({r['step1_worst_param']})")
            k = f"step{a.steps}_all_params_max_delta"
            print(f"  {a.steps} steps max|D| = {r[k]:.3e} ({r[f'step{a.steps}_worst_param']})", flush=True)
        elif style == "immediate":
            r = verify_drop_imm(name, code, a.device, a.store, store_root, a.preset,
                                a.batch, a.seq)
            print(f"  eval forward   max|D| = {r['eval_forward_hidden_max_delta']:.3e}")
            print(f"  imm vs after-full (all p) max|D| = "
                  f"{r['imm_vs_afterfull_all_params_max_delta']:.3e} ({r['worst_param']})", flush=True)
        else:
            r, slices = verify_drop(name, code, a.device, a.store, store_root, a.preset,
                                    a.batch, a.seq, anchor, style)
            if anchor is None:                      # first dropout mode = the anchor
                anchor = slices
            print(f"  eval forward   max|D| = {r['eval_forward_hidden_max_delta']:.3e}")
            print(f"  determinism    max|D| = {r['determinism_max_delta']:.3e}")
            if "vs_T1_anchor_max_delta" in r:
                print(f"  vs T1 anchor   max|D| = {r['vs_T1_anchor_max_delta']:.3e}", flush=True)
        results.append(r)
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()

    tag = "gpu" if a.device.startswith("cuda") else "cpu"
    out = a.out_dir / f"x1_train_exact_{tag}.json"
    json.dump({"run": "x1_training_exactness", "preset": a.preset, "device": a.device,
               "store": a.store, "batch": a.batch, "seq": a.seq,
               "note": "12 modes: dropout x recompute x streaming x update style. "
                       "Dropout rows: determinism + cross-mode grads vs T1 anchor. "
                       "No-dropout rows: identity vs reference under the mode's own "
                       "protocol (after_full: global clip 1.0; immediate: no clip), "
                       "incl. all-parameter step checks",
               "results": results}, open(out, "w"), indent=2)
    print(f"\n-> {out}")
    if store_root.exists():
        shutil.rmtree(store_root)


if __name__ == "__main__":
    main()

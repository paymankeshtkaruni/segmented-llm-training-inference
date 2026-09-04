#!/usr/bin/env python
"""Why do all-parameter step checks show ~1e-5 worst elements when gradients agree at ~1e-8?

Hypothesis to PROVE or refute: Adam's normalizer. At t=1 the update is
lr * g/(|g|+eps) (sign-like); where |g| is at the float-noise floor, a ~1e-8
gradient difference flips the ratio and moves the update by up to 2*lr.

Method (mode T5 = base+recompute, dropout 0):
  1. one backward on reference and segmented engines; capture the FULL gradient
     of the X1 worst-offender segment (attn output projection of layer L) from
     both sides, plus each side's global clip scale;
  2. one optimizer step on both; capture measured per-element parameter deltas;
  3. reconstruct both sides analytically with the exact _adamw_ kernel applied to
     the captured gradients; if reconstruction matches measurement elementwise
     (residual ~1e-9), the update rule fully explains the deltas — no bug.

Outputs: residuals, the |g| of every element whose delta exceeds thresholds, and
the 2*lr bound check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from verify_modes import _build, _batch, LR, WD, BETAS, EPS, CLIP   # noqa: E402
from modules import causal_lm_cross_entropy_loss                    # noqa: E402
from stores import SegmentKey                                       # noqa: E402
from optimizer import _adamw_                                       # noqa: E402

T5_CODE = "11110111000"   # base + recompute, no streaming (dropout 0 passed in)


def run(device: str, store_kind: str, layer: int, out_dir: Path, preset: str,
        batch: int, seq: int) -> dict:
    store_root = out_dir / f"_stores_delta_{device.replace(':', '_')}"
    m, s, ref, store, shared, loader, fwd, gstore, bwd, opt = _build(
        T5_CODE, 0.0, device, store_kind, store_root, preset)
    key = SegmentKey(layer, "attn_out_proj", 0)
    pname = f"blocks.{layer}.attention.output_projection.weight"

    decay = [p for p in ref.parameters() if p.dim() >= 2]
    nodecay = [p for p in ref.parameters() if p.dim() < 2]
    ref_opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                                 {"params": nodecay, "weight_decay": 0.0}],
                                lr=LR, betas=BETAS, eps=EPS)

    x, lab = _batch(m, 0, batch, seq)
    x, lab = x.to(device), lab.to(device)

    # --- backward on both sides -------------------------------------------------
    ref.train()
    ref.zero_grad(set_to_none=False)
    logits, _ = ref(x, pad_token_id=None)
    causal_lm_cross_entropy_loss(logits, lab).backward()
    del logits
    fwd.train()
    shared_grads = bwd.backward(x, lab, pad_token_id=None)

    ref_param = dict(ref.named_parameters())[pname]
    g_ref = ref_param.grad.detach().clone()
    g_seg = gstore.get(key)["weight"].to(device).clone()
    p0 = ref_param.detach().clone()
    grad_agreement = (g_ref - g_seg).abs().max().item()

    # --- clip scales (both sides compute a GLOBAL norm over all grads) ----------
    total_norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), CLIP)
    scale_ref = min(1.0, CLIP / (float(total_norm) + 1e-6))
    scale_seg = float(opt._clip_scale(shared_grads, CLIP))

    # --- the real steps ---------------------------------------------------------
    ref_opt.step()
    opt.step(shared_grads, clip_norm=CLIP)
    p_ref1 = dict(ref.named_parameters())[pname].detach().clone()
    p_seg1 = store.get(key)["weight"].to(device).clone()
    measured = (p_ref1 - p_seg1).abs()

    # --- analytic reconstruction with the exact _adamw_ kernel ------------------
    def predict(p_start, g, scale):
        p = p_start.clone()
        st = {}
        _adamw_(p, g * scale, st, 1, LR, BETAS[0], BETAS[1], EPS, WD)
        return p

    p_ref_pred = predict(p0, g_ref, scale_ref)
    p_seg_pred = predict(p0, g_seg, scale_seg)
    resid_ref = (p_ref_pred - p_ref1).abs().max().item()      # kernel == torch.AdamW?
    resid_seg = (p_seg_pred - p_seg1).abs().max().item()      # kernel == engine step?
    predicted = (p_ref_pred - p_seg_pred).abs()
    resid_delta = (predicted - measured).abs().max().item()   # explanation residual

    # --- where do the big deltas live? ------------------------------------------
    def bucket(th):
        mask = measured > th
        n = int(mask.sum().item())
        gmax = float(g_ref.abs()[mask].max().item()) if n else None
        return {"threshold": th, "n_elements": n, "max_|g|_at_those": gmax}

    flat = measured.flatten()
    top = torch.topk(flat, 10)
    idx = top.indices
    top10 = [{"delta": float(top.values[i]),
              "g_ref": float(g_ref.flatten()[idx[i]]),
              "g_seg": float(g_seg.flatten()[idx[i]])} for i in range(10)]

    r = {
        "device": device, "param": pname, "n_elements": int(measured.numel()),
        "grad_agreement_max_delta": grad_agreement,
        "clip_scale_ref": scale_ref, "clip_scale_seg": scale_seg,
        "clip_scale_diff": abs(scale_ref - scale_seg),
        "measured_step_delta_max": float(measured.max().item()),
        "bound_2lr": 2 * LR,
        "residual_kernel_vs_torch_adamw": resid_ref,
        "residual_kernel_vs_engine_step": resid_seg,
        "residual_explanation": resid_delta,
        "buckets": [bucket(t) for t in (1e-5, 1e-6, 1e-7)],
        "top10_deltas_with_grads": top10,
    }
    import shutil
    if store_root.exists():
        shutil.rmtree(store_root)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--store", required=True, choices=["cpu_ram", "disk"])
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "verify_modes")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    r = run(a.device, a.store, a.layer, a.out_dir, a.preset, a.batch, a.seq)
    print(json.dumps(r, indent=2))
    tag = "gpu" if a.device.startswith("cuda") else "cpu"
    out = a.out_dir / f"step_delta_analysis_{tag}.json"
    json.dump(r, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

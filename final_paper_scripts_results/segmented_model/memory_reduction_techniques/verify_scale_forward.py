#!/usr/bin/env python
"""Forward-state identity at SCALE (0.84B / 3.1B / 6.9B) — sequential, one model
on the device at a time.

X1 (`verify_modes.py`) and A2 (`verify_infer_modes.py`) prove exactness at 0.84B by
holding the reference model AND a segmented engine on the same device. That does not
scale: at 6.9B the fp32 reference alone is 27.4 GB, so reference + engine cannot both
sit on a 40 GB A100. This script makes the same forward-state comparison possible at
any size by SEQUENCING it:

    1. build the reference on CPU and populate the segment store from it;
    2. move the reference to the device, compute EVERY reference output it will ever
       be compared against (training-forward hidden states + greedy decode tokens),
       and copy those outputs to the host;
    3. DELETE the reference and empty the allocator;
    4. only then build each segmented mode from the store and compare against the
       saved outputs.

The reference and the segmented engine therefore never share the device, and the peak
of each phase is reported separately (`ref_peak_reserved_mb` vs per-mode
`peak_reserved_mb`) so the sequencing is visible in the artifact.

WHY only forward state (no gradients/optimizer): the backward+optimizer identity
needs the reference's own autograd graph and optimizer state resident alongside the
comparison, which is exactly what does not fit at scale. Forward state is the part
that can be checked size-independently, and it is what the referee asked for.

Tech-code flag order: [sdpa, mlp_sum, ce, recompute, stream, records, park, adam,
segment_wo, free_device, no_kv_cache].
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from config import get_preset                                            # noqa: E402
from modules import ReferenceGPTDecoder                                  # noqa: E402
from forward_engine import populate_from_reference, SegmentedForwardEngine  # noqa: E402
from loader import StrictSegmentLoader                                   # noqa: E402
from stores import make_store                                            # noqa: E402
from inference import SegmentedGenerator                                 # noqa: E402

# reuse the two 0.84B verification scripts verbatim where they are import-safe, so
# this script's batch, prompts and mode definitions are the SAME objects those
# scripts use — no second copy that could silently drift.
from verify_modes import _batch, tech_from_code                          # noqa: E402
from verify_infer_modes import reference_greedy, _Tok, MODES as INFER_MODES  # noqa: E402

# the three distinct TRAINING forward configurations (dropout 0). The update-style
# dial of verify_modes.py cannot change a forward, so the 12 training modes collapse
# to these three on the forward path.
TRAIN_FWD_MODES = [
    ("resident",  "11100111000"),
    ("recompute", "11110111000"),
    ("streaming", "11111111100"),
]

_MB = 1024 ** 2


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _reset_peak(device) -> None:
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device) -> float:
    return round(torch.cuda.max_memory_reserved() / _MB, 1) if _is_cuda(device) else 0.0


def _free(device) -> None:
    gc.collect()
    if _is_cuda(device):
        torch.cuda.empty_cache()


def _mk_store(store_kind: str, store_root: Path, name: str):
    return (make_store("cpu_ram") if store_kind == "cpu_ram"
            else make_store("disk", store_root / name))


def _build_engine(m, s, store, shared, device, code):
    """Segmented forward engine from an ALREADY populated store — the reference-free
    twin of verify_modes.py's `_build` (no ref, no grad/opt stores)."""
    loader = StrictSegmentLoader(m, s, store, device, tech=tech_from_code(code))
    fwd = SegmentedForwardEngine(m, s, loader, shared, device).eval()
    loader.training = False
    return loader, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=["cuda", "cpu"])
    ap.add_argument("--store", default="cpu_ram", choices=["cpu_ram", "disk"])
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "verify_modes")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dev = a.device
    store_root = a.out_dir / f"_store_{a.preset}"

    preset = get_preset(a.preset)
    m = replace(preset["model"], dropout=0.0)       # forward-only: dropout must be off
    s = preset["seg"]

    # ---------------- phase 1: reference on CPU, store populated from it ----------
    torch.manual_seed(0)                            # same init as verify_modes/_build
    ref = ReferenceGPTDecoder(m)
    params = sum(p.numel() for p in ref.parameters())
    print(f"[scale-forward] {a.preset} on {dev} store={a.store} "
          f"params={params/1e6:.1f}M seg={s.code}", flush=True)

    if store_root.exists():
        shutil.rmtree(store_root)
    store = _mk_store(a.store, store_root, "params")
    shared = populate_from_reference(ref, m, s, store)      # stays on CPU for now

    # ---------------- phase 2: every reference output, then free the reference ----
    _reset_peak(dev)
    ref = ref.to(dev).eval()

    x, _lab = _batch(m, 0, a.batch, a.seq)          # == verify_modes.py's step-0 batch
    x = x.to(dev)
    with torch.no_grad():
        _, h_ref = ref(x, pad_token_id=None)
    h_ref = h_ref.to("cpu")

    g = torch.Generator().manual_seed(7)            # == verify_infer_modes.py prompts
    prompts = [torch.randint(0, m.vocab_size, (a.prompt_len,), generator=g).tolist()
               for _ in range(a.n_prompts)]
    ref_tokens = [reference_greedy(ref, p, a.new_tokens, m.max_seq_len, dev)
                  for p in prompts]
    xp = torch.tensor([prompts[0]], dtype=torch.long, device=dev)
    with torch.no_grad():
        _, h_ref_prefill = ref(xp, pad_token_id=None)
    h_ref_prefill = h_ref_prefill.to("cpu")

    ref_peak = _peak_mb(dev)
    del ref, xp
    _free(dev)
    after_free = round(torch.cuda.memory_reserved() / _MB, 1) if _is_cuda(dev) else 0.0
    print(f"  reference freed: peak reserved {ref_peak} MB -> {after_free} MB resident",
          flush=True)

    shared = shared.to(dev)

    # ---------------- phase 3: segmented training forward, mode by mode -----------
    train_rows = []
    for name, code in TRAIN_FWD_MODES:
        _reset_peak(dev)
        loader, fwd = _build_engine(m, s, store, shared, dev, code)
        with torch.no_grad():
            h_seg = fwd.forward_hidden(x, pad_token_id=None)
        d = (h_ref - h_seg.to("cpu")).abs().max().item()
        peak = _peak_mb(dev)
        train_rows.append({"mode": name, "tech_code": code, "dropout": 0.0,
                           "forward_hidden_max_delta": d,
                           "peak_reserved_mb": peak})
        print(f"  train/{name:10} forward_hidden max|D|={d:.3e}  peak={peak} MB",
              flush=True)
        del h_seg, fwd, loader
        _free(dev)

    del x, h_ref
    _free(dev)

    # ---------------- phase 4: segmented greedy decode, mode by mode --------------
    infer_rows = []
    for name, code in INFER_MODES.items():
        _reset_peak(dev)
        loader, fwd = _build_engine(m, s, store, shared, dev, code)
        xp = torch.tensor([prompts[0]], dtype=torch.long, device=dev)
        with torch.no_grad():
            h_seg = fwd.forward_hidden(xp, pad_token_id=None)
        d_pre = (h_ref_prefill - h_seg.to("cpu")).abs().max().item()
        del h_seg, xp

        gen = SegmentedGenerator(fwd, _Tok())
        agree, first_div = True, None
        for i, p in enumerate(prompts):
            seg_out = gen.generate(p, max_new_tokens=a.new_tokens)
            if seg_out != ref_tokens[i]:
                agree = False
                for j, (sa, ra) in enumerate(zip(seg_out, ref_tokens[i])):
                    if sa != ra:
                        first_div = {"prompt": i, "pos": j, "seg": sa, "ref": ra}
                        break
                else:
                    first_div = {"prompt": i,
                                 "pos": min(len(seg_out), len(ref_tokens[i])),
                                 "note": "length mismatch"}
                break
        peak = _peak_mb(dev)
        infer_rows.append({"mode": name, "tech_code": code,
                           "prefill_hidden_max_delta": d_pre,
                           "greedy_tokens_identical": agree,
                           "first_divergence": first_div,
                           "peak_reserved_mb": peak})
        print(f"  infer/{name:20} prefill max|D|={d_pre:.3e}  tokens identical: {agree}"
              f"  peak={peak} MB"
              + (f"  first divergence: {first_div}" if first_div else ""), flush=True)
        del gen, fwd, loader
        _free(dev)

    # ---------------- artifact ----------------------------------------------------
    tag = "cuda" if _is_cuda(dev) else "cpu"
    out = a.out_dir / f"scale_forward_{a.preset}_{tag}.json"
    json.dump({"run": "scale_forward_identity", "preset": a.preset, "device": dev,
               "store": a.store, "batch": a.batch, "seq": a.seq,
               "n_prompts": a.n_prompts, "prompt_len": a.prompt_len,
               "new_tokens": a.new_tokens,
               "params_billion": round(params / 1e9, 4),
               "note": "reference forward computed first and freed; segmented modes "
                       "built afterwards from the same store, so reference and engine "
                       "never share the device",
               "ref_peak_reserved_mb": ref_peak,
               "reserved_after_ref_freed_mb": after_free,
               "train_forward": train_rows,
               "inference": infer_rows}, open(out, "w"), indent=2)
    print(f"\n-> {out}")

    if store_root.exists():
        shutil.rmtree(store_root)

    worst = max([r["forward_hidden_max_delta"] for r in train_rows]
                + [r["prefill_hidden_max_delta"] for r in infer_rows])
    ok = all(r["greedy_tokens_identical"] for r in infer_rows) and worst < 1e-2
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — worst forward max|D| = {worst:.3e}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Leave-one-out MARGINAL ablation driver — runs the LOO for train and/or inference on one
device (fixed large 8x2x2x8). Order-independent per-technique value (memory saved / time
cost vs the all-ON reference). Results -> results/{gpu,cpu}_{train,inference}_loo/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from loo_runner import run_train_loo, run_infer_loo   # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kinds", default="train,infer", help="comma list: train,infer")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--out-suffix", default="",
                    help="appended to output dir names, e.g. _rep1 -> results/gpu_train_loo_rep1")
    a = ap.parse_args()
    tag = "gpu" if a.device.startswith("cuda") else "cpu"
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    if "train" in kinds:
        run_train_loo(a.preset, a.device, HERE / "results" / f"{tag}_train_loo{a.out_suffix}",
                      batch=a.batch, n_steps=a.n_steps)
    if "infer" in kinds:
        run_infer_loo(a.preset, a.device, HERE / "results" / f"{tag}_inference_loo{a.out_suffix}",
                      prompt_len=a.prompt_len, gen_tokens=a.gen_tokens)

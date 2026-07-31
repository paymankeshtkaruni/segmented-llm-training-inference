#!/usr/bin/env python
"""
GPU TRAINING — incremental (cumulative) ablation of the memory-reduction TECHNIQUES.

Subject = the FIXED large segmented cost model (large_8x2x2x8). The segmentation is NOT
varied. We start from that model with every technique OFF (rung r0) and switch the
techniques ON one at a time (order = techniques.TRAIN_LADDER), recording at each rung, on
one MemFlow clock: peak VRAM (no-miss) and step time (+ fwd/bwd/opt split). Output: the
memory-down / time-up staircase -> results/gpu_train/ (ladder_train.json + per-rung JSON).

Self-contained: imports the LOCAL techniques.py / seg_cost_lib.py / engine copies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from ladder_runner import run_train_ladder   # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")   # segmentation FIXED at 8x2x2x8
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "gpu_train")
    a = ap.parse_args()
    run_train_ladder(a.preset, a.device, a.out_dir, batch=a.batch, n_steps=a.n_steps, seq_len=a.seq_len)

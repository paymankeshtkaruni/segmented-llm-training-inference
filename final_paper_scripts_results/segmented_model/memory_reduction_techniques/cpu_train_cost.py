#!/usr/bin/env python
"""
CPU TRAINING — incremental (cumulative) ablation of the memory-reduction TECHNIQUES.

Same ladder as gpu_train_cost.py but on CPU: the constrained resource is host RAM (RSS),
and the device->store rule parks segments on DISK. Peak RSS goes DOWN and step time goes
UP as techniques accumulate. Output -> results/cpu_train/ (ladder_train.json + per-rung).

NOTE: rung r0 (all techniques OFF: all segments resident + full autograd graph + full
logits) approaches full-model training memory — on the large model that can be many GB of
RSS; run on a node with enough RAM. Later rungs shrink it drastically.
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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "cpu_train")
    a = ap.parse_args()
    run_train_ladder(a.preset, a.device, a.out_dir, batch=a.batch, n_steps=a.n_steps, seq_len=a.seq_len)

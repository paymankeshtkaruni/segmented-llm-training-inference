#!/usr/bin/env python
"""
GPU INFERENCE — incremental (cumulative) ablation of the memory-reduction TECHNIQUES.

Subject = the FIXED large segmented model (large_8x2x2x8). Greedy generation (prefill +
gen_tokens decode). Techniques toggled ON one at a time (techniques.INFER_LADDER: no
backward/optimizer rungs; ends with no_kv_cache). Records peak VRAM + per-token time per
rung -> results/gpu_inference/ (ladder_infer.json + per-rung JSON).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from ladder_runner import run_infer_ladder   # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")   # segmentation FIXED at 8x2x2x8
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "gpu_inference")
    a = ap.parse_args()
    run_infer_ladder(a.preset, a.device, a.out_dir, prompt_len=a.prompt_len, gen_tokens=a.gen_tokens)

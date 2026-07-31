#!/usr/bin/env python
"""
A1 — Accuracy training (GPU variant).

Identical to full_model_train_cpu.py but defaults to CUDA compute and the
outputs/train_gpu/ directory. All logic lives in _train_lib (shared with the CPU
twin) so the two can never drift. Launched on a GPU node via sbatch.

Outputs (default outputs/train_gpu/):
  checkpoints/best.pt, checkpoints/last.pt, metrics.json
"""

from __future__ import annotations

import argparse

import _common as C
import _train_lib as T

DEFAULT_OUT = C.OUTPUTS_DIR / "train_gpu"


def main() -> None:
    p = argparse.ArgumentParser(description="A1 accuracy training (GPU)")
    T.add_train_args(p, default_device="cuda", default_out=DEFAULT_OUT)
    T.run(p.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
A1 — Accuracy training (CPU variant).

Trains the full GPTTransformer (GPTDecoder) baseline on the log-line -> label
generative task and tracks training loss, validation loss, validation accuracy.
Saves best/last checkpoints and metrics.json.

Device: CPU compute. Per the plan it is still LAUNCHED on a GPU node via sbatch,
never the login node. All logic lives in _train_lib (shared with the GPU twin).

Outputs (default outputs/train_cpu/):
  checkpoints/best.pt, checkpoints/last.pt, metrics.json
"""

from __future__ import annotations

import argparse

import _common as C
import _train_lib as T

DEFAULT_OUT = C.OUTPUTS_DIR / "train_cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="A1 accuracy training (CPU)")
    T.add_train_args(p, default_device="cpu", default_out=DEFAULT_OUT)
    T.run(p.parse_args())


if __name__ == "__main__":
    main()

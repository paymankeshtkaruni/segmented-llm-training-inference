#!/usr/bin/env python
"""B1 real inference (GPU) — small model. Thin wrapper over _infer_lib."""
from __future__ import annotations
import argparse
import _common as C
import _infer_lib as I

DEFAULT_OUT = C.OUTPUTS_DIR / "infer_gpu"
DEFAULT_CKPT = C.OUTPUTS_DIR / "train_gpu" / "checkpoints" / "best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description="B1 real inference (GPU)")
    I.add_infer_args(p, default_device="cuda", default_out=DEFAULT_OUT, default_ckpt=DEFAULT_CKPT)
    I.run_infer(p.parse_args())


if __name__ == "__main__":
    main()

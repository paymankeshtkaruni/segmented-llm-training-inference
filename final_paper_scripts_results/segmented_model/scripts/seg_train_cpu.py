#!/usr/bin/env python
"""Segmented A1 — accuracy training (CPU), small model, 8x2x2x8.
Thin entrypoint over segmentation_management.trainer.SegmentedTrainer. Same
hyperparameters as the full-model A1 baseline; only the execution is segmented."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segmentation_management"))
from trainer import SegmentedTrainer  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "seg_train_cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="Segmented A1 accuracy training (CPU)")
    p.add_argument("--preset", default="small_8x2x2x8")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--store-kind", default=None)  # None -> device default (gpu:cpu_ram, cpu:disk)
    # smoke knobs
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    tr = SegmentedTrainer(a.preset, a.device, a.out_dir, lr=a.lr, weight_decay=a.weight_decay,
                          grad_clip=a.grad_clip, warmup=a.warmup, seed=a.seed, store_kind=a.store_kind)
    tr.fit(a.epochs, a.batch_size, workers=a.workers, log_every=a.log_every,
           max_train_steps=a.max_train_steps, max_val_steps=a.max_val_steps, max_rows=a.max_rows)


if __name__ == "__main__":
    main()

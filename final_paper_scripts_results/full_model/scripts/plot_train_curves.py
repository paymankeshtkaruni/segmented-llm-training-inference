#!/usr/bin/env python
"""Generate A1 accuracy-training learning-curve figures into full_model/figures/.

For each device whose outputs/train_<dev>/metrics.json exists, writes:
  train_curves_<dev>.png   train/val loss (log) + val token-accuracy vs epoch

Promoted from temp/test_gpu_train_plots.py after review.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL_MODEL = Path(__file__).resolve().parent.parent
OUT = FULL_MODEL / "outputs"
FIG = FULL_MODEL / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def plot_device(dev: str) -> bool:
    mpath = OUT / f"train_{dev}" / "metrics.json"
    if not mpath.exists():
        print(f"  skip {dev}: {mpath} not found")
        return False
    d = json.load(open(mpath))
    ep = d["epochs_detail"]
    epochs = [e["epoch"] for e in ep]
    train_loss = [e["train_loss"] for e in ep]
    val_loss = [e["val_loss"] for e in ep]
    val_acc = [e.get("val_acc_global", e["val_acc"]) for e in ep]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epochs, train_loss, "o-", color="tab:red", label="train loss")
    ax1.plot(epochs, val_loss, "s-", color="tab:orange", label="val loss")
    ax1.set_yscale("log"); ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (log)", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, "^-", color="tab:blue", label="val token-acc (global)")
    ax2.set_ylabel("val token accuracy", color="tab:blue")
    ax1.set_title(f"A1 {dev.upper()} — learning curve "
                  f"({d['params_million']:.2f}M params, test_acc {d['test_acc']:.4f})")
    ax1.set_xticks(epochs); ax1.grid(True, alpha=0.3)
    L = ax1.get_lines() + ax2.get_lines()
    ax1.legend(L, [x.get_label() for x in L], loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"train_curves_{dev}.png", dpi=130)
    print(f"  wrote train_curves_{dev}.png  (test_acc {d['test_acc']:.4f})")
    return True


def main():
    for dev in ("gpu", "cpu"):
        plot_device(dev)
    print(f"A1 figures -> {FIG}/")


if __name__ == "__main__":
    main()

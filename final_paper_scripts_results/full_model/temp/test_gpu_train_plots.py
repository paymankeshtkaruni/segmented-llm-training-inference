"""Temp: plot the GPU A1 training learning curves from metrics.json.

Reads outputs/train_gpu/metrics.json and writes PNGs into this temp folder.
Plots: (1) train/val loss vs epoch, (2) train/val accuracy vs epoch.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FULL_MODEL = HERE.parent
METRICS = FULL_MODEL / "outputs" / "train_gpu" / "metrics.json"

d = json.load(open(METRICS))
ep = d["epochs_detail"]
epochs = [e["epoch"] for e in ep]
train_loss = [e["train_loss"] for e in ep]
val_loss = [e["val_loss"] for e in ep]
train_acc = [e["train_acc"] for e in ep]
val_acc = [e.get("val_acc_global", e["val_acc"]) for e in ep]

print("epochs:", epochs)
print("train_loss:", [round(x, 4) for x in train_loss])
print("val_loss  :", [round(x, 4) for x in val_loss])
print("train_acc :", [round(x, 4) for x in train_acc])
print("val_acc   :", [round(x, 4) for x in val_acc])

# 1) Loss
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, train_loss, "o-", label="train loss")
ax.plot(epochs, val_loss, "s-", label="val loss")
ax.set_yscale("log")
ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)")
ax.set_title("A1 GPU — training vs validation loss")
ax.set_xticks(epochs); ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "gpu_train_loss.png", dpi=130)

# 2) Accuracy
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, train_acc, "o-", label="train token-acc")
ax.plot(epochs, val_acc, "s-", label="val token-acc (global)")
ax.set_xlabel("epoch"); ax.set_ylabel("token accuracy")
ax.set_title("A1 GPU — training vs validation token accuracy")
ax.set_xticks(epochs); ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "gpu_train_acc.png", dpi=130)

# 3) Combined (loss + acc, twin axis)
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(epochs, train_loss, "o-", color="tab:red", label="train loss")
ax1.plot(epochs, val_loss, "s-", color="tab:orange", label="val loss")
ax1.set_yscale("log"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss (log)", color="tab:red")
ax2 = ax1.twinx()
ax2.plot(epochs, val_acc, "^-", color="tab:blue", label="val token-acc")
ax2.set_ylabel("val token accuracy", color="tab:blue")
ax1.set_title("A1 GPU — loss & val accuracy")
ax1.set_xticks(epochs); ax1.grid(True, alpha=0.3)
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
fig.tight_layout(); fig.savefig(HERE / "gpu_train_combined.png", dpi=130)

print("\nwrote:")
for p in ["gpu_train_loss.png", "gpu_train_acc.png", "gpu_train_combined.png"]:
    print("  ", HERE / p)

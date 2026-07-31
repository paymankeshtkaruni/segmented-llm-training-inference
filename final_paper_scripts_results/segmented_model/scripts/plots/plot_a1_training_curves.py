#!/usr/bin/env python
"""A1 — segmented GPU TRAINING curves: train/validation LOSS and ACCURACY per epoch.
Reads outputs/seg_train_gpu_pathb/metrics.json (epochs_detail). -> figures/plots/:
  seg_a1_training_curves.png   (left: loss train vs val; right: accuracy train vs val)
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG = Path(__file__).resolve().parents[2]
O = SEG / "outputs"; FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)
MET = O / "seg_train_gpu_pathb" / "metrics.json"
if not MET.exists():
    print(f"MISSING {MET}"); raise SystemExit(1)

d = json.load(open(MET)); ed = d["epochs_detail"]
ep = [r["epoch"] for r in ed]
tr_loss = [r["train_loss"] for r in ed]; va_loss = [r["val_loss"] for r in ed]
tr_acc = [r["train_acc"] for r in ed]; va_acc = [r["val_acc"] for r in ed]
test_acc = d.get("test_acc_global", d.get("test_acc"))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.plot(ep, tr_loss, "o-", color="#4C72B0", label="train loss", lw=1.8, ms=6)
a1.plot(ep, va_loss, "s-", color="#C44E52", label="val loss", lw=1.8, ms=6)
a1.set_xlabel("epoch"); a1.set_ylabel("cross-entropy loss"); a1.set_title("Loss")
a1.set_xticks(ep); a1.legend(fontsize=9); a1.grid(True, alpha=0.25)

a2.plot(ep, tr_acc, "o-", color="#4C72B0", label="train acc", lw=1.8, ms=6)
a2.plot(ep, va_acc, "s-", color="#55A868", label="val acc", lw=1.8, ms=6)
if test_acc is not None:
    a2.axhline(test_acc, color="#888", ls="--", lw=1, label=f"test acc ({test_acc:.4f})")
a2.set_xlabel("epoch"); a2.set_ylabel("token accuracy"); a2.set_title("Accuracy")
a2.set_xticks(ep); a2.legend(fontsize=9); a2.grid(True, alpha=0.25)

fig.suptitle(f"Segmented GPU training (A1) — loss & accuracy  "
             f"[{d['preset']}  {d['epochs']} epochs  bs={d['batch_size']}]", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG / "seg_a1_training_curves.png", dpi=140)
print(f"  wrote {FIG/'seg_a1_training_curves.png'}")

#!/usr/bin/env python
"""Cross-task INFERENCE comparison: the 4 segmented inference cost runs — torch (B2) vs ONNX (C),
GPU vs CPU — on peak VRAM, peak host-RAM, and time/token. -> figures/plots/seg_infer_compare.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG = Path(__file__).resolve().parents[2]
O = SEG / "outputs"; FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)


def b2(p):
    d = json.load(open(p))
    return (d["overall"].get("vram_hw_total_peak_mb", 0.0),
            max(d["per_phase"][x]["cpu_ram"]["rss_peak_mb"] for x in d["per_phase"]),
            d["avg_token_time_s"])


def c(p):
    d = json.load(open(p))
    return d["peak"]["vram_mb"], d["peak"]["rss_mb"], d["avg_token_s"]


specs = [("torch · GPU", "#DD8452", b2, O / "seg_b2_gpu/seg_b2_metrics.json"),
         ("ONNX · GPU",  "#4C72B0", c,  O / "seg_c_gpu/seg_c_metrics.json"),
         ("torch · CPU", "#E8B894", b2, O / "seg_b2_cpu/seg_b2_metrics.json"),
         ("ONNX · CPU",  "#9DBBD6", c,  O / "seg_c_cpu/seg_c_metrics.json")]
tasks = []
for lab, col, fn, p in specs:
    if p.exists():
        v, r, t = fn(p); tasks.append((lab, col, v, r, t))
    else:
        print("missing", p)

labels = [t[0] for t in tasks]; cols = [t[1] for t in tasks]
vram = [t[2] for t in tasks]; rss = [t[3] for t in tasks]; tok = [t[4] for t in tasks]
x = np.arange(len(tasks))
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, vals, title, ylab in [(a1, vram, "Peak VRAM (GPU only)", "MB"),
                              (a2, rss, "Peak host-RAM (RSS)", "MB"),
                              (a3, tok, "Time per token", "s")]:
    ax.bar(x, vals, color=cols)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.0f}" if ylab == "MB" else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.25)
a1.text(0.5, -0.22, "CPU has no VRAM (bars=0)", transform=a1.transAxes, ha="center", fontsize=8, color="#888")
fig.suptitle("Segmented inference cost — torch (B2) vs ONNX (C), GPU vs CPU  [large 838M, bs=4 seq=512]", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG / "seg_infer_compare.png", dpi=110)
print("wrote", FIG / "seg_infer_compare.png")
for lab, col, v, r, t in tasks:
    print(f"  {lab:12} VRAM={v:5.0f}MB  RSS={r:5.0f}MB  tok={t:.1f}s")

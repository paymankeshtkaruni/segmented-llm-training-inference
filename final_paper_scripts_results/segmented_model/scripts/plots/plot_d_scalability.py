#!/usr/bin/env python
"""D — granularity SCALABILITY figure: coarse 8x2x2x8 vs fine 16x4x4x16 (large 838M train cost),
GPU and CPU, showing the memory-down / time-up dial. -> figures/plots/seg_d_scalability.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG = Path(__file__).resolve().parents[2]
O = SEG / "outputs"; FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)
runs = {
    ("GPU", "8x2x2x8"):   O / "seg_cost_gpu/seg_cost_metrics.json",
    ("GPU", "16x4x4x16"): O / "seg_cost_gpu_16x4x4x16/seg_cost_metrics.json",
    ("CPU", "8x2x2x8"):   O / "seg_cost_cpu/seg_cost_metrics.json",
    ("CPU", "16x4x4x16"): O / "seg_cost_cpu_16x4x4x16/seg_cost_metrics.json",
}
g = lambda p: json.load(open(p))


def mem(dev, cfg):
    d = g(runs[(dev, cfg)])
    if dev == "GPU":
        return d["overall"]["vram_hw_total_peak_mb"]
    return max(d["per_phase"][x]["cpu_ram"]["rss_peak_mb"] for x in d["per_phase"])


def tm(dev, cfg):
    return g(runs[(dev, cfg)])["avg_step_time_s"]


fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
devs = ["GPU", "CPU"]; x = np.arange(2); w = 0.36; colA, colB = "#4C72B0", "#C44E52"
for ax, fn, title, yl in [(a1, mem, "Peak memory (GPU=VRAM, CPU=RSS)", "MB"),
                          (a2, tm, "Avg train step time", "s")]:
    v8 = [fn(dv, "8x2x2x8") for dv in devs]; v16 = [fn(dv, "16x4x4x16") for dv in devs]
    ax.bar(x - w / 2, v8, w, color=colA, label="8x2x2x8 (coarser)")
    ax.bar(x + w / 2, v16, w, color=colB, label="16x4x4x16 (finer)")
    for xi, (a, b) in enumerate(zip(v8, v16)):
        ax.text(xi - w / 2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(devs); ax.set_ylabel(yl); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("D — granularity scaling (large 838M train cost): finer = less memory, more time", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG / "seg_d_scalability.png", dpi=110)
print("wrote", FIG / "seg_d_scalability.png")

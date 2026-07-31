#!/usr/bin/env python
"""
Plot the incremental memory-reduction-technique staircases.

Reads results/{gpu_train,cpu_train,gpu_inference,cpu_inference}/ladder_*.json and draws, for
each available category, a dual-axis figure: peak memory (bars, left axis — goes DOWN as
techniques accumulate) and step/per-token time (line, right axis — goes UP). Missing
categories are skipped. Figures -> results/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)

# (subdir, ladder json, is_cuda, time key, time label)
CATS = [
    ("gpu_train",     "ladder_train.json", True,  "step_time_s", "step time (s)"),
    ("cpu_train",     "ladder_train.json", False, "step_time_s", "step time (s)"),
    ("gpu_inference", "ladder_infer.json", True,  "per_token_s", "per-token time (s)"),
    ("cpu_inference", "ladder_infer.json", False, "per_token_s", "per-token time (s)"),
]


def plot_one(subdir, jname, is_cuda, tkey, tlabel):
    p = RES / subdir / jname
    if not p.exists():
        print(f"  skip {subdir}: {p.name} not found"); return
    d = json.load(open(p)); rows = d["ladder"]
    memk = "vram_peak_mb" if is_cuda else "rss_peak_mb"
    memlabel = "peak VRAM (MB)" if is_cuda else "peak RSS (MB)"
    labels = [r["rung"].replace("_", "\n", 1) for r in rows]
    mem = [r[memk] for r in rows]; t = [r[tkey] for r in rows]
    x = list(range(len(rows)))

    fig, axm = plt.subplots(figsize=(max(9, 1.1 * len(rows)), 5.2))
    axm.bar(x, mem, color="#4C72B0", alpha=0.85, label=memlabel)
    axm.set_ylabel(memlabel, color="#4C72B0"); axm.tick_params(axis="y", labelcolor="#4C72B0")
    axm.set_xticks(x); axm.set_xticklabels(labels, fontsize=8)
    axt = axm.twinx()
    axt.plot(x, t, "o-", color="#C44E52", lw=2, ms=6, label=tlabel)
    axt.set_ylabel(tlabel, color="#C44E52"); axt.tick_params(axis="y", labelcolor="#C44E52")
    tag = d.get("preset", d.get("subdir", subdir))
    axm.set_title(f"Incremental memory-reduction techniques — {subdir}  "
                  f"[{tag}]  (memory ↓ / time ↑)", fontsize=11)
    axm.set_xlabel("cumulative technique (each rung = previous + one)")
    fig.tight_layout()
    out = FIG / f"ladder_{subdir}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("plotting ladders ->", FIG)
    for c in CATS:
        plot_one(*c)

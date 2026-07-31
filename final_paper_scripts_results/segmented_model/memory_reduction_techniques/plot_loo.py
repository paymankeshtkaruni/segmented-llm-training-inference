#!/usr/bin/env python
"""
Plot the leave-one-out MARGINAL per-technique value (order-independent).

Reads results/{gpu,cpu}_{train,inference}_loo/loo_{train,infer}.json and draws, per
category, a horizontal bar of memory each technique SAVES (vs all-on), ranked by
efficiency (MB saved per second added); each bar annotated with the time it costs and its
MB/s. Figures -> results/figures/loo_*.png
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

CATS = [("gpu_train_loo", "loo_train.json", True), ("cpu_train_loo", "loo_train.json", False),
        ("gpu_inference_loo", "loo_infer.json", True), ("cpu_inference_loo", "loo_infer.json", False)]


def plot_one(subdir, jname, is_cuda):
    p = RES / subdir / jname
    if not p.exists():
        print(f"  skip {subdir}: no json"); return
    d = json.load(open(p))
    rows = sorted(d["techniques"], key=lambda r: r["efficiency_mb_per_s"])   # low->high (top=best)
    names = [r["technique"] for r in rows]
    saves = [r["saves_mem_mb"] for r in rows]
    costs = [r["costs_time_s"] for r in rows]
    eff = [r["efficiency_mb_per_s"] for r in rows]
    y = list(range(len(rows)))
    colors = ["#C44E52" if c > 5 else ("#DD8452" if c > 0.5 else "#55A868") for c in costs]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(y, saves, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    for k in y:
        ax.annotate(f"+{costs[k]:.1f}s  ({eff[k]:.0f} MB/s)", (max(saves[k], 0), k), fontsize=7,
                    va="center", xytext=(3, 0), textcoords="offset points",
                    color="#C44E52" if costs[k] > 5 else "#333")
    ax.axvline(0, color="k", lw=0.6)
    memlbl = "VRAM" if is_cuda else "RSS"
    ref = d["reference_all_on"]
    ax.set_xlabel(f"memory this technique SAVES ({memlbl} MB, vs all-ON) · annot: time added & MB/s")
    ax.set_title(f"{subdir}: leave-one-out MARGINAL value (order-independent)\n"
                 f"all-ON = {ref['mem_mb']:.0f} MB / {ref['time_s']:.2f}s · top = best MB/s "
                 f"(recompute excluded: prerequisite for streaming)", fontsize=10)
    fig.tight_layout()
    out = FIG / f"loo_{subdir.replace('_loo','')}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    print("LOO plots ->", FIG)
    for c in CATS:
        plot_one(*c)

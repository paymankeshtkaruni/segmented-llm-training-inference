#!/usr/bin/env python
"""
DECISION plots — "given my memory limit and time budget, which techniques do I apply,
and in what priority?"  Built from the ladder_*.json of each category.

Per category, one figure with two panels:
  LEFT  — memory<->time TRADEOFF (Pareto) curve across the cumulative rungs. Drop a
          vertical line at your per-node memory budget: the left-most point under it is
          the technique set you need, and its y is the resulting step/token time.
  RIGHT — per-technique BANG-FOR-BUCK: memory SAVED (bar, MB) with time ADDED annotated,
          sorted so the cheap big-memory wins are on top (apply first) and the costly /
          low-value ones at the bottom (apply last, only if memory-bound).

Also prints a text recommendation per category. Figures -> results/figures/decision_*.png
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

CATS = [
    ("gpu_train",     "ladder_train.json", True,  "step_time_s", "step time (s)"),
    ("cpu_train",     "ladder_train.json", False, "step_time_s", "step time (s)"),
    ("gpu_inference", "ladder_infer.json", True,  "per_token_s", "per-token time (s)"),
    ("cpu_inference", "ladder_infer.json", False, "per_token_s", "per-token time (s)"),
]


def load(subdir, jname, is_cuda, tkey):
    p = RES / subdir / jname
    if not p.exists():
        return None
    rows = json.load(open(p))["ladder"]
    memk = "vram_peak_mb" if is_cuda else "rss_peak_mb"
    mem = [r[memk] for r in rows]; t = [r[tkey] for r in rows]
    names = [r["rung"] for r in rows]; adds = [r["adds"] for r in rows]
    return rows, mem, t, names, adds, memk


def plot_one(subdir, jname, is_cuda, tkey, tlabel):
    d = load(subdir, jname, is_cuda, tkey)
    if d is None:
        print(f"  skip {subdir}: no json"); return
    rows, mem, t, names, adds, memk = d
    memlabel = "peak VRAM (MB)" if is_cuda else "peak RSS (MB)"
    # per-technique deltas (rung i vs i-1): memory SAVED, time ADDED
    dmem = [mem[i - 1] - mem[i] for i in range(1, len(mem))]     # +ve = saved
    dt = [t[i] - t[i - 1] for i in range(1, len(t))]             # +ve = added
    tech = [adds[i] for i in range(1, len(adds))]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- LEFT: memory<->time tradeoff (Pareto) ----
    axL.plot(mem, t, "-o", color="#4C72B0", lw=1.8, ms=6)
    for i, nm in enumerate(names):
        axL.annotate(nm.split("_", 1)[-1], (mem[i], t[i]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    axL.set_xlabel(memlabel + "  (← less memory = more techniques on)")
    axL.set_ylabel(tlabel)
    axL.invert_xaxis()                                          # less memory to the right->left
    axL.grid(True, alpha=0.25)
    axL.set_title(f"{subdir}: memory ↔ time tradeoff\n(drop a vertical line at your memory budget)")
    # example budget guides
    lo, hi = min(mem), max(mem)
    for frac, lab in [(0.5, "½ baseline"), (0.25, "¼"), (0.1, "10%")]:
        b = hi * frac
        if lo <= b <= hi:
            axL.axvline(b, color="#999", ls="--", lw=0.8)
            axL.annotate(lab, (b, max(t)), fontsize=7, color="#666", rotation=90, va="top")

    # ---- RIGHT: bang-for-buck ranking by EFFICIENCY = memory saved per second added ----
    # (this is the greedy-knapsack priority: to hit a memory target at least time cost,
    #  apply the most memory-per-time-efficient techniques first.)
    eff = [dmem[i] / max(dt[i], 0.05) for i in range(len(dmem))]  # MB saved per second
    order = sorted(range(len(dmem)), key=lambda i: eff[i])        # ascending -> most efficient on top
    y = list(range(len(order)))
    colors = ["#C44E52" if dt[i] > 5 else ("#DD8452" if dt[i] > 0.5 else "#55A868") for i in order]
    axR.barh(y, [dmem[i] for i in order], color=colors)
    axR.set_yticks(y)
    axR.set_yticklabels([tech[i].split(" (")[0] for i in order], fontsize=8)
    for k, i in enumerate(order):
        axR.annotate(f"+{dt[i]:.1f}s  ({eff[i]:.0f} MB/s)", (max(dmem[i], 0), k), fontsize=7, va="center",
                     xytext=(3, 0), textcoords="offset points",
                     color="#C44E52" if dt[i] > 5 else "#333")
    axR.axvline(0, color="k", lw=0.6)
    axR.set_xlabel("memory SAVED (MB) · annot = time added & efficiency (MB/s) · green=cheap orange=some red=slow")
    axR.set_title(f"{subdir}: apply-FIRST priority\n(top = most memory saved per second of time)")
    fig.tight_layout()
    out = FIG / f"decision_{subdir}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out.name}")

    # ---- text recommendation, ranked by efficiency ----
    rank = sorted(range(len(dmem)), key=lambda i: -eff[i])
    print(f"    {subdir}: baseline {max(mem):.0f} {memk[:3].upper()}/{min(t):.2f}s  ->  all-on {min(mem):.0f}/{max(t):.2f}s")
    first = [i for i in rank if dmem[i] > 50][:4]
    last = [i for i in rank if dt[i] > 5 and eff[i] < 100][-3:]
    if first:
        print("      apply FIRST (best MB/s): " + " | ".join(
            f"{tech[i].split(' (')[0]} -{dmem[i]:.0f}MB/+{dt[i]:.1f}s = {eff[i]:.0f}MB/s" for i in first))
    if last:
        print("      apply LAST (poor MB/s):  " + " | ".join(
            f"{tech[i].split(' (')[0]} -{dmem[i]:.0f}MB/+{dt[i]:.0f}s = {eff[i]:.0f}MB/s" for i in last))


if __name__ == "__main__":
    print("decision plots ->", FIG)
    for c in CATS:
        plot_one(*c)

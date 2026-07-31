#!/usr/bin/env python
"""
Extra paper artifacts from the finished ladder results:
  master_staircase.png  — all 4 cumulative ladders in one 2x2 grid.
  phase_time.png        — forward/backward/optimizer split at all-ON (recompute-dominated bwd).
  summary_table.{png,md}— baseline->final memory (reduction x) + time (x) per category.
  recipe_gpu_train.{png,md} — for each memory budget, minimal technique set + resulting time.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)

CATS = [("gpu_train", "ladder_train.json", True, "step_time_s", "step time (s)"),
        ("cpu_train", "ladder_train.json", False, "step_time_s", "step time (s)"),
        ("gpu_inference", "ladder_infer.json", True, "per_token_s", "per-token time (s)"),
        ("cpu_inference", "ladder_infer.json", False, "per_token_s", "per-token time (s)")]


def _load(sub, jn):
    p = RES / sub / jn
    return json.load(open(p))["ladder"] if p.exists() else None


def master_staircase():
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (sub, jn, is_cuda, tk, tl) in zip(axs.flat, CATS):
        rows = _load(sub, jn)
        if not rows:
            ax.set_title(f"{sub} (missing)"); continue
        memk = "vram_peak_mb" if is_cuda else "rss_peak_mb"
        x = range(len(rows)); mem = [r[memk] for r in rows]; t = [r[tk] for r in rows]
        ax.bar(x, mem, color="#4C72B0", alpha=0.85)
        ax.set_ylabel(("VRAM" if is_cuda else "RSS") + " MB", color="#4C72B0")
        ax.set_xticks(list(x)); ax.set_xticklabels([r["rung"].split("_", 1)[-1] for r in rows],
                                                    rotation=40, ha="right", fontsize=7)
        a2 = ax.twinx(); a2.plot(list(x), t, "o-", color="#C44E52", lw=1.8, ms=4)
        a2.set_ylabel(tl, color="#C44E52")
        ax.set_title(f"{sub}: {mem[0]:.0f}→{mem[-1]:.0f} MB ({mem[0]/max(mem[-1],1):.1f}×),  "
                     f"{t[0]:.2f}→{t[-1]:.2f}s", fontsize=10)
    fig.suptitle("Incremental memory-reduction techniques on the large 8×2×2×8 model "
                 "(memory ↓ bars, time ↑ line)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "master_staircase.png", dpi=140); plt.close(fig)
    print("  wrote master_staircase.png")


def phase_time():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, sub in zip(axs, ["gpu_train", "cpu_train"]):
        rows = _load(sub, "ladder_train.json")
        if not rows:
            continue
        r = rows[-1]  # all-ON
        parts = [("forward", r.get("fwd_ms") or 0), ("backward", r.get("bwd_ms") or 0),
                 ("optimizer", r.get("opt_ms") or 0)]
        vals = [v / 1000 for _, v in parts]
        cols = ["#55A868", "#C44E52", "#DD8452"]
        ax.bar([p for p, _ in parts], vals, color=cols)
        tot = sum(vals)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}s\n{100*v/max(tot,1e-9):.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{sub} (all-ON): backward = recompute-dominated", fontsize=10)
        ax.set_ylabel("time (s)")
    fig.suptitle("Where the step time goes: the backward pass (recomputation) dominates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "phase_time.png", dpi=140); plt.close(fig)
    print("  wrote phase_time.png")


def summary_table():
    lines = ["| category | metric | baseline | all-ON | reduction | time baseline→all-ON |",
             "|---|---|---|---|---|---|"]
    rowsfig = []
    for sub, jn, is_cuda, tk, tl in CATS:
        rows = _load(sub, jn)
        if not rows:
            continue
        memk = "vram_peak_mb" if is_cuda else "rss_peak_mb"
        m0, m1 = rows[0][memk], rows[-1][memk]; t0, t1 = rows[0][tk], rows[-1][tk]
        metric = "VRAM" if is_cuda else "RSS"
        lines.append(f"| {sub} | {metric} | {m0:.0f} MB | {m1:.0f} MB | "
                     f"**{m0/max(m1,1):.1f}×** | {t0:.2f} → {t1:.2f} s ({t1/max(t0,1e-9):.0f}×) |")
        rowsfig.append([sub, metric, f"{m0:.0f}", f"{m1:.0f}", f"{m0/max(m1,1):.1f}×",
                        f"{t0:.2f}→{t1:.2f}s"])
    (RES / "figures" / "summary_table.md").write_text("\n".join(lines) + "\n")
    # png table
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.5 * len(rowsfig))); ax.axis("off")
    tbl = ax.table(cellText=rowsfig,
                   colLabels=["category", "metric", "baseline", "all-ON", "reduction", "time (b→all-ON)"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(6):
        tbl[0, j].set_facecolor("#4C72B0"); tbl[0, j].set_text_props(color="w", weight="bold")
    ax.set_title("Peak memory reduction and time cost, all-ON vs baseline (large 8×2×2×8)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "summary_table.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  wrote summary_table.{png,md}")


def recipe_table():
    rows = _load("gpu_train", "ladder_train.json")
    if not rows:
        print("  skip recipe: no gpu_train"); return
    budgets = [16000, 8000, 4000, 2000, 1000]
    out = ["| memory budget | minimal techniques (cumulative) | peak VRAM | step time |",
           "|---|---|---|---|"]
    rowsfig = []
    for b in budgets:
        pick = None
        for r in rows:
            if r["vram_peak_mb"] <= b:
                pick = r; break
        if pick is None:
            out.append(f"| ≤ {b/1000:.0f} GB | (not reachable) | — | — |")
            rowsfig.append([f"≤{b/1000:.0f}GB", "not reachable", "—", "—"]); continue
        out.append(f"| ≤ {b/1000:.0f} GB | up to +{pick['rung'].split('_',1)[-1]} | "
                   f"{pick['vram_peak_mb']:.0f} MB | {pick['step_time_s']:.1f} s |")
        rowsfig.append([f"≤{b/1000:.0f}GB", "up to +" + pick["rung"].split("_", 1)[-1],
                        f"{pick['vram_peak_mb']:.0f}MB", f"{pick['step_time_s']:.0f}s"])
    (RES / "figures" / "recipe_gpu_train.md").write_text("\n".join(out) + "\n")
    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.5 * len(rowsfig))); ax.axis("off")
    tbl = ax.table(cellText=rowsfig, colLabels=["memory budget", "minimal technique set",
                                                "peak VRAM", "step time"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(4):
        tbl[0, j].set_facecolor("#55A868"); tbl[0, j].set_text_props(color="w", weight="bold")
    ax.set_title("Deployment recipe (GPU training): fit a memory budget at minimum time", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "recipe_gpu_train.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  wrote recipe_gpu_train.{png,md}")


if __name__ == "__main__":
    print("paper extras ->", FIG)
    master_staircase(); phase_time(); summary_table(); recipe_table()

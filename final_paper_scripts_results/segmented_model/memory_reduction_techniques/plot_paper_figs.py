#!/usr/bin/env python
"""
Schematic + breakdown figures for the paper.
  F1  segmentation schematic: the 4 axes (E×A×M×H) + one-segment-at-a-time execution.
  F2  memory WATERFALL (real GPU-train data): baseline peak -> each technique's reclaimed
      slice -> irreducible residual (context + resident + one segment).
Figures -> results/figures/fig_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
FIG = HERE / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)


def fig1_schematic():
    fig, ax = plt.subplots(figsize=(11, 5.4)); ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.4)
    ax.text(5.5, 5.15, "Segmented execution: one segment resident at a time",
            ha="center", fontsize=13, weight="bold")
    # LEFT: the model with 4 axes
    ax.text(2.4, 4.55, "GPT decoder — 4 segmentation axes (E×A×M×H)", ha="center", fontsize=10, weight="bold")
    axes = [("E  embedding", "split d_model → concat", "#4C72B0"),
            ("A  attention", "head-groups → concat → Wₒ", "#55A868"),
            ("M  MLP", "d_ff chunks → running-sum", "#DD8452"),
            ("H  output head", "vocab slices → streamed CE", "#C44E52")]
    for i, (name, how, c) in enumerate(axes):
        yy = 3.9 - i * 0.82
        ax.add_patch(FancyBboxPatch((0.5, yy), 3.8, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=c, ec="k", alpha=0.85, lw=1))
        ax.text(0.68, yy + 0.31, name, va="center", fontsize=9.5, weight="bold", color="w")
        ax.text(2.55, yy + 0.31, how, va="center", ha="center", fontsize=8, color="w")
    # MIDDLE: the constrained device holding ONE segment
    ax.add_patch(FancyBboxPatch((5.1, 1.5), 2.5, 2.4, boxstyle="round,pad=0.03,rounding_size=0.1",
                                fc="none", ec="k", lw=2))
    ax.text(6.35, 3.72, "constrained device", ha="center", fontsize=9.5, weight="bold")
    ax.text(6.35, 3.42, "(GPU VRAM / CPU RAM)", ha="center", fontsize=7.5, color="#555")
    ax.add_patch(FancyBboxPatch((5.35, 2.35), 2.0, 0.75, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#4C72B0", ec="k", alpha=0.9))
    ax.text(6.35, 2.72, "1 active segment", ha="center", va="center", fontsize=9, color="w", weight="bold")
    ax.text(6.35, 1.72, "+ tiny resident norms/bias", ha="center", fontsize=7.5, color="#555")
    # RIGHT: the store
    ax.add_patch(FancyBboxPatch((8.6, 1.5), 2.1, 2.4, boxstyle="round,pad=0.03,rounding_size=0.1",
                                fc="#eee", ec="k", lw=1.5))
    ax.text(9.65, 3.72, "backing store", ha="center", fontsize=9.5, weight="bold")
    ax.text(9.65, 3.44, "GPU→host RAM\nCPU→disk", ha="center", fontsize=7.5, color="#555")
    for k in range(5):
        ax.add_patch(FancyBboxPatch((8.85 + (k % 3) * 0.55, 1.75 + (k // 3) * 0.5), 0.45, 0.38,
                                    boxstyle="round,pad=0.01", fc="#bbb", ec="#666"))
    ax.text(9.65, 1.55, "all other segments,\ngrads, optimizer state", ha="center", fontsize=7, color="#555")
    # arrows: model -> device, device <-> store (stream)
    ax.add_patch(FancyArrowPatch((4.4, 2.7), (5.05, 2.7), arrowstyle="-|>", mutation_scale=14, lw=1.5, color="#333"))
    ax.add_patch(FancyArrowPatch((7.65, 2.95), (8.55, 2.95), arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#C44E52"))
    ax.add_patch(FancyArrowPatch((8.55, 2.35), (7.65, 2.35), arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#55A868"))
    ax.text(8.1, 3.15, "evict", ha="center", fontsize=7.5, color="#C44E52")
    ax.text(8.1, 2.08, "load", ha="center", fontsize=7.5, color="#55A868")
    ax.text(6.35, 0.95, "backward by RECOMPUTATION (no retained graph)  ·  streamed segment-wise AdamW",
            ha="center", fontsize=8.5, style="italic", color="#333")
    ax.text(6.35, 0.55, "peak memory ≈ one segment, not the whole model  →  identical weights, different schedule",
            ha="center", fontsize=8.5, color="#333", weight="bold")
    fig.savefig(FIG / "fig1_segmentation_schematic.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  wrote fig1_segmentation_schematic.png")


def fig2_waterfall():
    p = HERE / "results" / "gpu_train" / "ladder_train.json"
    if not p.exists():
        print("  skip fig2: gpu_train ladder not found"); return
    rows = json.load(open(p))["ladder"]
    mem = [r["vram_peak_mb"] for r in rows]
    labels = [r["rung"].split("_", 1)[-1] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    # baseline bar
    ax.bar(0, mem[0], width=0.7, color="#B0B0B0", ec="k")
    ax.text(0, mem[0] + 250, f"{mem[0]:.0f}", ha="center", fontsize=8, weight="bold")
    ax.text(0, -900, "baseline\n(all OFF)", ha="center", fontsize=8)
    running = mem[0]
    for i in range(1, len(mem)):
        drop = mem[i - 1] - mem[i]
        bottom = mem[i]
        col = "#C44E52" if drop < 0 else "#4C72B0"
        ax.bar(i, abs(drop), bottom=min(mem[i - 1], mem[i]), width=0.7, color=col, ec="k", alpha=0.85)
        ax.text(i, max(mem[i - 1], mem[i]) + 250, f"{-drop:+.0f}", ha="center", fontsize=7.5,
                color="#C44E52" if drop < 0 else "#333")
        ax.text(i, -900, "+" + labels[i], ha="center", fontsize=7.5, rotation=30)
    # residual bar
    ax.bar(len(mem), mem[-1], width=0.7, color="#55A868", ec="k")
    ax.text(len(mem), mem[-1] + 250, f"{mem[-1]:.0f}", ha="center", fontsize=8, weight="bold")
    ax.text(len(mem), -900, "residual\n(all ON)", ha="center", fontsize=8)
    ax.set_ylabel("peak VRAM (MB)")
    ax.set_title("Where the 18 GB goes: memory reclaimed by each technique (GPU training, large 8×2×2×8)\n"
                 "blue = memory reclaimed · green = irreducible residual (context + resident + one segment)",
                 fontsize=11)
    ax.set_xticks([]); ax.set_ylim(-1400, mem[0] * 1.08)
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_memory_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  wrote fig2_memory_waterfall.png")


if __name__ == "__main__":
    print("paper figures ->", FIG)
    fig1_schematic()
    fig2_waterfall()

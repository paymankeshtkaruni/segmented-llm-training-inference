#!/usr/bin/env python
"""
Scale-experiment figures (from results/scale_summary.json — run scale_summary.py first):
  comp_scale_memory.png : full vs segmented peak TRAINING memory at 0.84B/3B/7B,
                          GPU + CPU; GPU full at 3B/7B = recorded OOM, drawn at its
                          estimated requirement against the 40 GB device line.
  comp_scale_model.png  : the performance model — measured vs fitted step time
                          (T = overhead + slope*P) and peak memory
                          (M = ctx + 2*s_max + c_act*d_model), GPU segmented training.
Same design system as comparison_figures.py (validated palette, no dual axis,
thin marks, direct labels, recessive axes).
"""
from __future__ import annotations

import json
from pathlib import Path

from comparison_figures import (SEG, FULL_, WARM, INK, INK2, MUTED, GRID, SURF,
                                grid, OUT)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "results" / "scale_summary.json"))
SC = D["scales"]
FIT = D["performance_model_fit"]
GB = 1000.0


def _label(ax, x, mb, color, weight="normal", dy=5):
    txt = f"{mb/GB:.1f} GB" if mb >= GB else f"{mb:.0f} MB"
    ax.annotate(txt, (x, mb), xytext=(0, dy), textcoords="offset points",
                ha="center", fontsize=9.5, color=color, weight=weight)


def fig_scale_memory():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
    w = 0.34
    for ax, dev, title in [(axes[0], "gpu", "GPU training (A100, 40 GB)"),
                           (axes[1], "cpu", "CPU training (host RAM)")]:
        grid(ax); ax.set_yscale("log")
        xs = range(len(SC))
        for i, e in enumerate(SC):
            ft, st = e[dev]["full_train"], e[dev]["seg_train"]
            if ft.get("oom"):
                # HATCHED BAR = ESTIMATED requirement (always), so the hatch means one
                # thing throughout the figure and matches the caption. Where the
                # requirement was independently MEASURED on a 94 GB H100, that
                # measurement is drawn as a separate marker on top of the same bar.
                est = ft["estimated_required_mb"]
                h100 = e[dev].get("full_train_h100") or {}
                ax.bar(i - w/2, est, w, color="none", edgecolor=FULL_, lw=1.4,
                       hatch="///", zorder=2)
                ax.annotate(f"est. {est/GB:.0f} GB\nOOM at 40 GB", (i - w/2, est),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=9, color=INK2, style="italic")
                mv, mlab = None, ""
                if h100 and not h100.get("oom"):      # requirement MEASURED on the H100
                    mv = h100["mem_mb"]; mlab = f"H100 measured {mv/GB:.1f} GB"
                elif h100:                            # aborts even on the H100
                    mv = h100.get("device_total_mb") or 95330.0
                    mlab = f"H100: OOM $>${mv/GB:.0f} GB"
                if mv:
                    ax.plot([i - w, i], [mv, mv], "-", color=WARM, lw=2.2, zorder=5,
                            solid_capstyle="butt")
                    ax.annotate(mlab, (i - w/2, mv), xytext=(0, -3),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=8.5, color=WARM, zorder=6,
                                bbox=dict(fc=SURF, ec="none", pad=1.2))
            else:
                ax.bar(i - w/2, ft["mem_mb"], w, color=FULL_, zorder=2)
                _label(ax, i - w/2, ft["mem_mb"], INK2)
            ax.bar(i + w/2, st["mem_mb"], w, color=SEG, zorder=3)
            _label(ax, i + w/2, st["mem_mb"], SEG, weight="bold")
            if not ft.get("oom"):
                # in the empty space above the segmented bar, at the geometric midpoint of
                # the pair — above the taller bar it collides with the 40 GB device line
                ax.annotate(f"{ft['mem_mb']/st['mem_mb']:.0f}$\\times$",
                            (i + w/2, (ft["mem_mb"] * st["mem_mb"]) ** 0.5),
                            ha="center", va="center", fontsize=11.5,
                            color=SEG, weight="bold")
        if dev == "gpu":
            lim = 40442.4
            ax.axhline(lim, color=WARM, lw=1.2, ls=(0, (5, 4)), zorder=1)
            ax.annotate("device limit (40 GB)", (0.012, lim),
                        xycoords=ax.get_yaxis_transform(), xytext=(0, 5),
                        textcoords="offset points", ha="left", fontsize=9.5, color=WARM)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{e['params_billion']:.2f}B" for e in SC], fontsize=11)
        ax.set_title(title, color=INK, loc="left")
        ax.set_ylim(400, 300000)
    axes[0].set_ylabel("peak memory (log)")
    axes[0].set_yticks([1000, 10000, 40000, 100000])
    axes[0].set_yticklabels(["1 GB", "10 GB", "40 GB", "100 GB"])
    # one figure-level legend (below the panels) — inside either panel it collides with
    # the ratio labels and the estimated-requirement annotations
    fig.legend(handles=[Patch(facecolor=FULL_, label="full model (measured)"),
                        Patch(facecolor="none", edgecolor=FULL_, hatch="///",
                              label="full model (estimated requirement; OOM on 40 GB)"),
                        Line2D([0], [0], color=WARM, lw=2.2,
                               label="that requirement checked on a 94 GB H100"),
                        Patch(facecolor=SEG, label="segmented")],
               loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Scaling up: full-model training stops fitting; segmented peak tracks one segment",
                 fontsize=13.5, weight="bold", color=INK, x=0.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUT / "comp_scale_memory.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("  comp_scale_memory")


def fig_scale_model():
    P = [pt["params_billion"] for pt in FIT["points"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    # ---- time: measured vs fitted lines, per node class ----
    grid(a1)
    a, b = FIT["time_overhead_s"], FIT["time_slope_s_per_bparam"]
    px = [0, 8]
    a1.plot(px, [a + b * x for x in px], "-", color=MUTED, lw=1.8, zorder=2,
            label=f"A100-40GB fit:  $T \\approx {a:.0f} + {b:.0f}\\,\\mathrm{{s}} \\times P$")
    f80 = FIT.get("time_fit_a100_80gb")
    if f80:
        a8, b8 = f80["time_overhead_s"], f80["time_slope_s_per_bparam"]
        a1.plot(px, [a8 + b8 * x for x in px], "--", color=MUTED, lw=1.4, zorder=2,
                label=f"A100-80GB:  $T \\approx {a8:.0f} + {b8:.0f}\\,\\mathrm{{s}} \\times P$")
    for node, mk in [("A100-40GB", "o"), ("A100-80GB", "s")]:
        pts = [pt for pt in FIT["points"] if pt["node"] == node]
        if not pts:
            continue
        a1.plot([pt["params_billion"] for pt in pts], [pt["time_measured_s"] for pt in pts],
                mk, ls="none", ms=10 if mk == "o" else 9, color=SEG,
                mfc=SEG if mk == "o" else "none", mec=SURF if mk == "o" else SEG,
                mew=1.6 if mk == "o" else 2.0, zorder=3, label=f"measured ({node})")
    for pt in FIT["points"]:
        a1.annotate(f"{pt['time_measured_s']:.0f} s", (pt["params_billion"], pt["time_measured_s"]),
                    xytext=(8, -4), textcoords="offset points", fontsize=9.5, color=INK)
    a1.annotate("same intercept on both node classes:\nthe overhead term is hardware-invariant",
                (0.25, 655), fontsize=9, color=INK2, style="italic", va="top")
    a1.set_xlabel("model size $P$ (B params)"); a1.set_ylabel("step time (s)")
    a1.set_title("Step time is linear in model size", color=INK, loc="left")
    a1.set_xlim(0, 7.6); a1.set_ylim(0, 700)
    a1.legend(loc="lower right", frameon=False, fontsize=9)
    # ---- memory: measured vs formula ----
    grid(a2)
    a2.plot(P, [pt["mem_predicted_mb"] for pt in FIT["points"]], "--D", color=MUTED,
            lw=1.6, ms=8, mfc="none", mec=MUTED, mew=1.8, zorder=2,
            label="formula:  $M \\approx \\mathrm{ctx} + 2\\,s_{max} + c\\,d_{model}$")
    a2.plot(P, [pt["mem_measured_mb"] for pt in FIT["points"]], "o", ms=10, color=SEG,
            mec=SURF, mew=1.6, zorder=3, label="measured")
    for pt in FIT["points"]:
        a2.annotate(f"{pt['mem_measured_mb']:.0f} MB", (pt["params_billion"], pt["mem_measured_mb"]),
                    xytext=(8, -4), textcoords="offset points", fontsize=9.5, color=INK)
    a2.annotate("total $P$ does not appear in the formula", (0.4, 2600), fontsize=10,
                color=INK2, style="italic")
    a2.set_xlabel("model size $P$ (B params)"); a2.set_ylabel("peak VRAM (MB)")
    a2.set_title("Peak memory tracks one segment, not the model", color=INK, loc="left")
    a2.set_xlim(0, 7.6); a2.set_ylim(0, 3000)
    a2.legend(loc="lower right", frameon=False)
    fig.suptitle("The memory–time model, fitted on GPU segmented training (8$\\times$2$\\times$2$\\times$8, all techniques on)",
                 fontsize=13.5, weight="bold", color=INK, x=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "comp_scale_model.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("  comp_scale_model")


if __name__ == "__main__":
    fig_scale_memory()
    fig_scale_model()
    print(f"-> {OUT}")

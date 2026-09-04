#!/usr/bin/env python
"""Generate all figures for paper_v4_tpds from the committed result files.

Reads ONLY files under memory_reduction_techniques/results/ (summaries,
traces, quality logs). Writes PDF figures to paper_v4_tpds/figures/.
Run via Slurm (slurm/figures.sbatch); nothing runs on the login node.
"""
from __future__ import annotations

import gzip
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent                      # paper_v4_tpds/
R = HERE.parent / "memory_reduction_techniques" / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "figure.dpi": 200, "savefig.bbox": "tight",
})
COL = {"retained": "#c44e52", "recomp": "#4c72b0", "stream": "#55a868",
       "full": "#111111", "zero": "#8172b2", "naive": "#937860",
       "onnx": "#dd8452"}


def med(xs):
    return statistics.median(xs)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- F1: curves
def fig_learning_curves():
    def epochs(log):
        pat = re.compile(r"\[validation\] loss=([\d.]+) acc\(global\)=([\d.]+)")
        return [(float(m.group(1)), float(m.group(2)))
                for m in map(pat.search, open(log)) if m]
    t1 = epochs(R / "quality_fresh_pair" / "segmented_T1_10ep.log")
    t2 = epochs(R / "quality_fresh_pair" / "segmented_T2_10ep.log")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.1))
    ep = range(1, len(t1) + 1)
    a.plot(ep, [x[0] for x in t1], "o-", ms=3, color=COL["retained"],
           label="segmented, retained graph")
    a.plot(ep, [x[0] for x in t2], "s--", ms=3, color=COL["recomp"],
           label="segmented, recomputation")
    a.set_yscale("log"); a.set_xlabel("epoch"); a.set_ylabel("validation loss")
    a.legend(); a.grid(alpha=.3)
    b.plot(ep, [x[1] for x in t1], "o-", ms=3, color=COL["retained"],
           label="segmented, retained graph")
    b.plot(ep, [x[1] for x in t2], "s--", ms=3, color=COL["recomp"],
           label="segmented, recomputation")
    b.axhline(0.9954, color=COL["full"], lw=1, ls=":",
              label="full model, final test acc. (0.9954)")
    b.set_xlabel("epoch"); b.set_ylabel("validation token accuracy")
    b.set_ylim(0.90, 1.0); b.legend(loc="lower right"); b.grid(alpha=.3)
    save(fig, "learning_curves.pdf")


# ------------------------------------------------------------- F2: frontier
def fig_frontier():
    grid = json.load(open(R / "exp3_grid" / "grid_summary.json"))["modes"]
    # style per mode: color by memory regime, marker by update schedule
    def style(i):
        graph = ["retained", "recomp", "stream"][i % 3]
        imm = i >= 6
        return COL[graph], ("^" if imm else "o")
    order = ["T1", "T2", "T3", "T4", "T5", "T6",
             "T7", "T8", "T9", "T10", "T11", "T12"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for i, m in enumerate(order):
        g, c = grid[m]["gpu"], grid[m]["cpu"]
        col, mk = style(i)
        a.scatter(g["step_s"], g["vram_mb"] / 1024, c=col, marker=mk, s=22, zorder=3)
        b.scatter(c["step_s"], c["rss_mb"] / 1024, c=col, marker=mk, s=22, zorder=3)
    # anchors
    a.scatter(0.72, 25.406, c=COL["full"], marker="*", s=90, zorder=4)
    a.annotate("full model", (0.72, 25.4), textcoords="offset points",
               xytext=(4, 3), fontsize=6.5)
    a.scatter(6.2, 18.244, c=COL["naive"], marker="D", s=25, zorder=3)
    a.annotate("naive segmented", (6.2, 18.2), textcoords="offset points",
               xytext=(4, 3), fontsize=6.5)
    a.scatter(2.12, 21.478, c=COL["zero"], marker="P", s=40, zorder=3)
    a.scatter(2.93, 23.622, c=COL["zero"], marker="P", s=40, zorder=3)
    a.annotate("ZeRO-Offload 2/3", (2.12, 21.5), textcoords="offset points",
               xytext=(4, 4), fontsize=6.5)
    b.scatter(33.0, 26.928, c=COL["full"], marker="*", s=90, zorder=4)
    b.annotate("full model", (33.0, 26.9), textcoords="offset points",
               xytext=(4, 3), fontsize=6.5)
    b.scatter(77.4, 20.545, c=COL["naive"], marker="D", s=25, zorder=3)
    b.annotate("naive segmented", (77.4, 20.5), textcoords="offset points",
               xytext=(4, 3), fontsize=6.5)
    for ax, ttl, xl in [(a, "GPU (A100-40GB)", "step time (s)"),
                        (b, "CPU", "step time (s)")]:
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_title(ttl)
        ax.grid(alpha=.3, which="both")
    a.set_ylabel("peak device memory (GB)")
    b.set_ylabel("peak RAM (GB)")
    import matplotlib.lines as ml
    handles = [
        ml.Line2D([], [], color=COL["retained"], marker="s", ls="", label="retained graph"),
        ml.Line2D([], [], color=COL["recomp"], marker="s", ls="", label="recomputation"),
        ml.Line2D([], [], color=COL["stream"], marker="s", ls="", label="recomp. + streaming"),
        ml.Line2D([], [], color="gray", marker="o", ls="", label="deferred update"),
        ml.Line2D([], [], color="gray", marker="^", ls="", label="optimizer-in-backward"),
    ]
    a.legend(handles=handles, loc="lower left", ncol=1)
    save(fig, "frontier_train.pdf")


# --------------------------------------------------------------- F3: traces
def fig_traces():
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 4.0), constrained_layout=True)
    names = [("trace_gpu_T2.json.gz", "recomputation, resident, deferred"),
             ("trace_gpu_T3.json.gz", "recomputation, streamed, deferred"),
             ("trace_gpu_T9.json.gz", "recomp., streamed, in-backward")]
    for ax, (fn, ttl) in zip(axes, names):
        d = json.load(gzip.open(R / "traces" / fn))
        tl = d["vram_timeline"]
        t = [x[0] for x in tl]; res = [x[2] / 1024 for x in tl]
        ax.plot(t, res, lw=0.7, color=COL["recomp"])
        ax.fill_between(t, res, color=COL["recomp"], alpha=.25, lw=0)
        phases = [(m[0], m[1]) for m in d["moves"] if m[2] == "<phase>"]
        seen = set()
        heights = [0.10, 0.40, 0.65, 0.15]
        for pt, ph in phases:
            if ph in ("build", "warmup") or ph in seen:
                continue
            ax.axvline(pt, color="k", lw=0.4, ls=":")
            ax.text(pt, max(res) * heights[len(seen) % 4], " " + ph,
                    rotation=90, fontsize=5, va="bottom", color="0.25")
            seen.add(ph)
        ax.set_ylabel("GB"); ax.set_title(ttl, fontsize=7)
        ax.grid(alpha=.3)
    axes[-1].set_xlabel("time (s)")
    save(fig, "traces_gpu.pdf")


# ---------------------------------------------------------------- F4: scale
def fig_scale():
    cells = json.load(open(R / "exp5_scale" / "scale_summary.json"))["cells"]
    grid = json.load(open(R / "exp3_grid" / "grid_summary.json"))["modes"]

    def cell(name, key="vram_mb"):
        reps = cells.get(name, [])
        vals = [r[key] for r in reps if r.get(key) is not None]
        return med(vals) / 1024 if vals else None

    sizes = ["0.84B", "1.5B", "3.1B", "5.1B", "6.9B"]
    full = [25.406, cell("gpu40_xl15b_full_train"), None, None, None]
    full_note = [None, None, "67.6 (H100)", ">80", ">94 (H100)"]
    resident = [grid["T2"]["gpu"]["vram_mb"] / 1024, None,
                cell("gpu40_xl3b_T2"), None, cell("gpu40_xxl7b_T2")]
    t3 = [grid["T3"]["gpu"]["vram_mb"] / 1024, cell("gpu40_xl15b_T3"),
          cell("gpu40_xl3b_T3"), cell("gpu40_xl5b_T3"), cell("gpu40_xxl7b_T3")]
    t9 = [grid["T9"]["gpu"]["vram_mb"] / 1024, cell("gpu40_xl15b_T9"),
          cell("gpu40_xl3b_T9"), cell("gpu40_xl5b_T9"), cell("gpu40_xxl7b_T9")]

    import numpy as np
    x = np.arange(len(sizes)); w = 0.2
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    for off, vals, lab, col in [
            (-1.5 * w, full, "full model", COL["full"]),
            (-0.5 * w, resident, "resident segmented", COL["recomp"]),
            (0.5 * w, t3, "streamed, deferred", COL["stream"]),
            (1.5 * w, t9, "streamed, in-backward", "#88c999")]:
        xs = [xi + off for xi, v in zip(x, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.bar(xs, ys, w, label=lab, color=col)
    for xi, note in zip(x, full_note):
        if note:
            ax.bar([xi - 1.5 * w], [40], w, color="none", edgecolor=COL["full"],
                   hatch="////", lw=0.6)
            ax.text(xi - 1.5 * w, 41, "OOM\n" + note, ha="center", fontsize=5)
    ax.axhline(40, color="k", lw=0.7, ls="--")
    ax.text(-0.45, 41.0, "40 GB device", fontsize=6, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel("peak device memory (GB)")
    ax.set_ylim(0, 48)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02),
              fontsize=5.8, frameon=False)
    save(fig, "scale_train.pdf")


# ------------------------------------------------------------ F5: inference
def fig_inference():
    d = json.load(open(R / "exp4_infer" / "infer_summary.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3))
    tor = {"I1": "resident + cache", "I2": "resident, recompute",
           "I3": "streamed + cache", "I4": "streamed, recompute"}
    onx = {"O1_full": "full-session", "O3_preload": "preloaded",
           "O4_stream": "disk-streamed"}
    # per-point label offsets (points), tuned to avoid collisions
    off_g = {"I1": (5, 5), "I2": (5, -9), "I3": (-2, 7), "I4": (2, -11),
             "O1_full": (-10, -12), "O3_preload": (-12, 8), "O4_stream": (5, -3)}
    off_c = {"I1": (-24, 9), "I2": (5, -10), "I3": (-8, 7), "I4": (2, -11),
             "O1_full": (-14, -12), "O3_preload": (5, 3), "O4_stream": (-30, 6)}
    for m, lab in tor.items():
        g, c = d["torch"][m]["gpu"], d["torch"][m]["cpu"]
        a.scatter(g["per_token_s"], g["vram_mb"] / 1024, c=COL["recomp"], s=22)
        a.annotate(lab, (g["per_token_s"], g["vram_mb"] / 1024),
                   textcoords="offset points", xytext=off_g[m], fontsize=6)
        b.scatter(c["per_token_s"], c["rss_mb"] / 1024, c=COL["recomp"], s=22)
        b.annotate(lab, (c["per_token_s"], c["rss_mb"] / 1024),
                   textcoords="offset points", xytext=off_c[m], fontsize=6)
    for m, lab in onx.items():
        g, c = d["onnx"][m]["gpu"], d["onnx"][m]["cpu"]
        a.scatter(g["per_token_s"], g["vram_mb"] / 1024, c=COL["onnx"],
                  marker="^", s=24)
        a.annotate(lab, (g["per_token_s"], g["vram_mb"] / 1024),
                   textcoords="offset points", xytext=off_g[m], fontsize=6)
        b.scatter(c["per_token_s"], c["rss_mb"] / 1024, c=COL["onnx"],
                  marker="^", s=24)
        b.annotate(lab, (c["per_token_s"], c["rss_mb"] / 1024),
                   textcoords="offset points", xytext=off_c[m], fontsize=6)
    a.set_xlim(0.009, 30)
    fa = d["full_anchor"]
    fg = ((fa["gpu"]["vram_reserved_mb"] + 490) / 1024, fa["gpu"]["per_token_s"])
    fc = (fa["cpu"]["rss_mb"] / 1024, fa["cpu"]["per_token_s"])
    a.scatter(fg[1], fg[0], c=COL["full"], marker="*", s=80)
    a.annotate("full model", (fg[1], fg[0]), textcoords="offset points",
               xytext=(4, 6), fontsize=6)
    b.scatter(fc[1], fc[0], c=COL["full"], marker="*", s=80)
    b.annotate("full model", (fc[1], fc[0]), textcoords="offset points",
               xytext=(5, 6), fontsize=6)
    import matplotlib.lines as ml
    handles = [ml.Line2D([], [], color=COL["recomp"], marker="o", ls="",
                         label="training runtime (PyTorch)"),
               ml.Line2D([], [], color=COL["onnx"], marker="^", ls="",
                         label="serving runtime (ONNX)")]
    for ax, ttl, yl in [(a, "GPU (A100-40GB)", "peak device memory (GB)"),
                        (b, "CPU", "peak RAM (GB)")]:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("seconds per generated token"); ax.set_ylabel(yl)
        ax.set_title(ttl); ax.grid(alpha=.3, which="both")
    a.legend(handles=handles, loc="lower left")
    save(fig, "inference_frontier.pdf")


if __name__ == "__main__":
    fig_learning_curves()
    fig_frontier()
    fig_traces()
    fig_scale()
    fig_inference()

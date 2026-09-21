#!/usr/bin/env python
"""Generate all figures for paper_jsa from the committed result files.

Reads ONLY files under memory_reduction_techniques/results/ (summaries,
traces, quality logs). Writes PDF figures to paper_jsa/figures/.
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

HERE = Path(__file__).resolve().parent                      # paper_jsa/
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


# the paper writes MB for mebibytes and GB for 1,000 of them (Sec. V-C)
GB = 1000.0


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
        a.scatter(g["step_s"], g["vram_mb"] / GB, c=col, marker=mk, s=22, zorder=3)
        b.scatter(c["step_s"], c["rss_mb"] / GB, c=col, marker=mk, s=22, zorder=3)
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
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
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
        t = [x[0] for x in tl]; res = [x[2] / GB for x in tl]
        ax.plot(t, res, lw=0.7, color=COL["recomp"])
        ax.fill_between(t, res, color=COL["recomp"], alpha=.25, lw=0)
        phases = [(m[0], m[1]) for m in d["moves"] if m[2] == "<phase>"]
        seen = set()
        # Phase boundaries can fall within a label width of each other (0.27 s
        # in T2, 0.11 s in T9), which overprinted the rotated labels.  Draw the
        # rule at the true boundary and slide the label right until it clears
        # every label already placed; sep is a constant in points because it
        # scales with the panel's own time range.
        sep = 0.032 * (max(t) - min(t))
        placed = []
        for pt, ph in phases:
            if ph in ("build", "warmup") or ph in seen:
                continue
            ax.axvline(pt, color="k", lw=0.4, ls=":")
            xl = pt
            while any(abs(xl - q) < sep for q in placed):
                xl += sep
            placed.append(xl)
            ax.annotate(ph, xy=(xl, 0.03), xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points",
                        rotation=90, fontsize=5, va="bottom", ha="center",
                        color="0.25")
            seen.add(ph)
        ax.set_ylabel("GB"); ax.set_title(ttl, fontsize=7)
        ax.grid(alpha=.3)
    axes[-1].set_xlabel("time (s)")
    save(fig, "traces_gpu.pdf")


# ---------------------------------------------------------------- F4: scale
def fig_scale():
    cells = json.load(open(R / "exp5_scale" / "scale_summary.json"))["cells"]
    fast = json.load(open(R / "exp6_fast" / "exp6_summary.json"))["cells"]
    grid = json.load(open(R / "exp3_grid" / "grid_summary.json"))["modes"]

    def cell(name, key="vram_mb", src=None):
        reps = (cells if src is None else src).get(name, [])
        vals = [r[key] for r in reps if r.get(key) is not None]
        return med(vals) / GB if vals else None

    sizes = ["0.84B", "1.5B", "3.1B", "5.1B", "6.9B"]
    full = [25.406, cell("gpu40_xl15b_full_train"), None, None, None]
    full_note = [None, None, "67.6\n(H100)", ">80\n(A100-80GB)", ">94\n(H100)"]
    # 1.5B has no recomputation-resident cell; the resident mode shown there is the
    # retained-graph in-backward one, exactly as in the scale table (hatched bar).
    resident = [grid["T2"]["gpu"]["vram_mb"] / GB, cell("gpu40_xl15b_T7", src=fast),
                cell("gpu40_xl3b_T2"), cell("gpu40_xl5b_T2"), cell("gpu40_xxl7b_T2")]
    t3 = [grid["T3"]["gpu"]["vram_mb"] / GB, cell("gpu40_xl15b_T3"),
          cell("gpu40_xl3b_T3"), cell("gpu40_xl5b_T3"), cell("gpu40_xxl7b_T3")]
    t9 = [grid["T9"]["gpu"]["vram_mb"] / GB, cell("gpu40_xl15b_T9"),
          cell("gpu40_xl3b_T9"), cell("gpu40_xl5b_T9"), cell("gpu40_xxl7b_T9")]

    import numpy as np
    x = np.arange(len(sizes)); w = 0.2
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    for off, vals, lab, col in [
            (-1.5 * w, full, "full model", COL["full"]),
            (-0.5 * w, resident, "resident segmented", COL["recomp"]),
            (0.5 * w, t3, "streamed, deferred", COL["stream"]),
            (1.5 * w, t9, "streamed, in-backward", "#88c999")]:
        first = True
        for xi, sz, v in zip(x, sizes, vals):
            if v is None:
                continue
            hatched = (lab == "resident segmented" and sz == "1.5B")
            ax.bar([xi + off], [v], w, label=(lab if first else None), color=col,
                   hatch=("///" if hatched else None),
                   edgecolor=("white" if hatched else "none"), linewidth=0.0)
            first = False
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 2.3))
    tor = {"I1": "resident + cache", "I2": "resident, recompute",
           "I3": "streamed + cache", "I4": "streamed, recompute"}
    onx = {"O1_full": "full-session", "O3_preload": "preloaded",
           "O4_stream": "disk-streamed"}
    # Per-point label placement (offset in points, plus alignment).  The GPU
    # panel packs four points into ~8 pt of height at the top and four more
    # into ~15 pt at the bottom right, so its labels are aligned away from the
    # frame and, where a label cannot sit beside its marker, tied to it with a
    # leader line.  The CPU panel is sparse enough for plain offsets.
    off_g = {"I1": (0, 7, "center", "bottom"),
             "I2": (0, -7, "center", "top"),
             "I3": (-24, 5, "right", "bottom"),
             "I4": (-24, 22, "right", "bottom"),
             "O1_full": (0, 7, "center", "bottom"),
             "O3_preload": (-6, 2, "right", "bottom"),
             "O4_stream": (6, -6, "right", "top")}
    lead_g = {"I4"}                      # label too far from its marker to read
    # the three CPU points nearest the right frame carry their labels leftwards
    off_c = {"I1": (-2, 5, "left", "baseline"),
             "I2": (6, -11, "left", "baseline"),
             "I3": (-6, 8, "right", "baseline"),
             "I4": (-4, -12, "right", "baseline"),
             "O1_full": (-18, -13, "left", "baseline"),
             "O3_preload": (-6, 5, "right", "baseline"),
             "O4_stream": (-34, 7, "left", "baseline")}

    def label(ax, off, m, lab, xy):
        dx, dy, ha, va = off[m]
        kw = {}
        if ax is a and m in lead_g:
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.4, color="0.55",
                                    shrinkA=1, shrinkB=3)
        ax.annotate(lab, xy, textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=6, **kw)

    def label_g(m, lab, xy):
        label(a, off_g, m, lab, xy)

    for m, lab in tor.items():
        g, c = d["torch"][m]["gpu"], d["torch"][m]["cpu"]
        a.scatter(g["per_token_s"], g["vram_mb"] / GB, c=COL["recomp"], s=22)
        label_g(m, lab, (g["per_token_s"], g["vram_mb"] / GB))
        b.scatter(c["per_token_s"], c["rss_mb"] / GB, c=COL["recomp"], s=22)
        label(b, off_c, m, lab, (c["per_token_s"], c["rss_mb"] / GB))
    for m, lab in onx.items():
        g, c = d["onnx"][m]["gpu"], d["onnx"][m]["cpu"]
        a.scatter(g["per_token_s"], g["vram_mb"] / GB, c=COL["onnx"],
                  marker="^", s=24)
        label_g(m, lab, (g["per_token_s"], g["vram_mb"] / GB))
        b.scatter(c["per_token_s"], c["rss_mb"] / GB, c=COL["onnx"],
                  marker="^", s=24)
        label(b, off_c, m, lab, (c["per_token_s"], c["rss_mb"] / GB))
    a.set_xlim(0.007, 32)
    a.set_ylim(0.26, 24)                 # headroom for the two label bands
    b.set_ylim(0.33, 7.5)                # keeps the CPU anchor label off the title
    fa = d["full_anchor"]
    fg = ((fa["gpu"]["vram_reserved_mb"] + 490) / GB, fa["gpu"]["per_token_s"])
    fc = (fa["cpu"]["rss_mb"] / GB, fa["cpu"]["per_token_s"])
    a.scatter(fg[1], fg[0], c=COL["full"], marker="*", s=80)
    a.annotate("full model", (fg[1], fg[0]), textcoords="offset points",
               xytext=(-2, -8), ha="right", va="top", fontsize=6)
    b.scatter(fc[1], fc[0], c=COL["full"], marker="*", s=80)
    b.annotate("full model", (fc[1], fc[0]), textcoords="offset points",
               xytext=(6, 8), ha="left", fontsize=6)
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
    a.legend(handles=handles, loc="upper left")
    save(fig, "inference_frontier.pdf")




def fig_schematic():
    """Concept figure: four segmentation axes; one segment EXECUTING at a
    time in every mode; residency differs by regime (streamed vs resident).
    Honest to the corrected invariant of Sec. III-B."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, fc, ec="black", lw=1.0, r=1.6):
        b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.4,rounding_size={r}",
                           fc=fc, ec=ec, lw=lw)
        ax.add_patch(b)

    def txt(x, y, t, size=7.5, c="black", w="normal", ha="center", style="normal"):
        ax.text(x, y, t, fontsize=size, color=c, ha=ha, va="center",
                fontweight=w, fontstyle=style)

    txt(50, 97, "Segmented execution: one segment executing at a time",
        11, w="bold")

    # ---- left: four axes
    txt(14, 89, "GPT decoder — four segmentation axes", 7.5, w="bold")
    axes_boxes = [
        ("E — embedding", "split $d_{model}$, concat", "#4c72b0"),
        ("A — attention", "head groups; $W_o$ its own segment", "#55a868"),
        ("M — MLP", "$d_{ff}$ chunks, running-sum", "#dd8452"),
        ("H — output head", "vocab slices, streamed cross-entropy", "#c44e52"),
    ]
    y = 76
    for name, sub, col in axes_boxes:
        box(2, y, 24, 9.5, col)
        txt(14, y + 6.4, name, 7.5, "white", "bold")
        txt(14, y + 2.6, sub, 5.6, "white")
        y -= 13.5
    ax.add_patch(FancyArrowPatch((27.5, 55), (31.5, 55), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.4, color="black"))

    # ---- middle top: streamed modes (box 33..67)
    box(33, 56, 34, 32, "#f4f4f4", lw=1.6)
    txt(50, 84.8, "STREAMED modes — constrained device", 6.8, w="bold")
    box(40, 69, 20, 9, "#4c72b0")
    txt(50, 74.6, "1 active segment", 7.5, "white", "bold")
    txt(50, 71.2, "the only weights on device", 5.4, "white")
    txt(50, 64.5, "+ tiny shared norms / biases", 5.8)
    txt(50, 60.5, "peak $\\approx$ one segment (memory floor)", 5.8, style="italic")

    # ---- middle bottom: resident modes
    box(33, 20, 34, 30, "#f4f4f4", lw=1.6)
    txt(50, 46.8, "RESIDENT modes — constrained device", 6.8, w="bold")
    blue_cx = None
    for i2 in range(8):
        cx = 37.5 + (i2 % 4) * 6.4
        cy = 35.5 - (i2 // 4) * 6.8
        col = "#4c72b0" if i2 == 1 else "#bfbfbf"
        if i2 == 1:
            blue_cx = cx + 2.5
        box(cx, cy, 5.0, 4.6, col, lw=0.7)
    txt(blue_cx, 43.0, "executing", 5.2, "#4c72b0", "bold")
    txt(50, 26.0, "all segment weights stay on device;", 5.8)
    txt(50, 23.0, "activations of one segment at a time", 5.8)

    # ---- right: backing store (box 72..98)
    box(72, 34, 26, 42, "#eaeef3", lw=1.6)
    txt(85, 72.6, "backing store", 8, w="bold")
    txt(85, 69.2, "GPU: host RAM  |  CPU: disk", 5.8)
    for i2 in range(6):
        cx = 75.5 + (i2 % 3) * 7.2
        cy = 57 - (i2 // 3) * 7.5
        box(cx, cy, 5.8, 5.2, "#bfbfbf", lw=0.7)
    txt(85, 44.0, "streamed: all other segments,", 5.6)
    txt(85, 41.4, "gradients, Adam state, records", 5.6)
    txt(85, 37.6, "resident: gradients + Adam state", 5.6)

    ax.add_patch(FancyArrowPatch((68, 76), (71, 71), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#c44e52"))
    txt(69.5, 77.5, "evict", 6, "#c44e52", "bold")
    ax.add_patch(FancyArrowPatch((71, 63), (68, 68), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#55a868"))
    txt(69.5, 61.5, "load", 6, "#55a868", "bold")
    ax.add_patch(FancyArrowPatch((68, 38), (71, 42), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="0.45"))
    txt(69.3, 36.2, "park", 6, "0.35", "bold")

    # ---- footer
    txt(50, 13.5, "dials: recomputation  ·  weight streaming  ·  update schedule  ·  dropout  ·  KV cache"
        "   $\\Rightarrow$   12 training + 4 inference modes", 6.8)
    txt(50, 9.2, "every mode computes the same model — identical weights, "
        "only the memory schedule changes", 7.2, "#4c72b0", "bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_schematic.pdf")
    fig.savefig(OUT / "fig_schematic.png", dpi=220)
    plt.close(fig)
    print("wrote", OUT / "fig_schematic.pdf")


if __name__ == "__main__":
    fig_learning_curves()
    fig_frontier()
    fig_traces()
    fig_scale()
    fig_inference()
    fig_schematic()

#!/usr/bin/env python
"""
Beautiful segmented-vs-full comparison figures, per the data-viz design system:
validated colorblind-safe palette, NO dual-axis, emphasis form (segmented = blue accent,
full = gray context), thin marks, surface rings, selective direct labels, recessive axes.
-> results/figures/paper/comp_*.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]        # final_paper_scripts_results
FULL = ROOT / "full_model" / "outputs"
SEGO = ROOT / "segmented_model" / "outputs"
OUT = Path(__file__).resolve().parent / "results" / "figures" / "paper"; OUT.mkdir(parents=True, exist_ok=True)

# ---- design tokens (validated palette) ----
SEG   = "#2a78d6"   # accent  (the thing we propose)
FULL_ = "#9a9a96"   # de-emphasis gray (the baseline / context)
WARM  = "#eb6834"   # time
INK   = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a86"; GRID = "#ececec"; SURF = "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": "#cfcfcf", "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 13, "axes.labelsize": 12, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
def L(p):
    try: return json.load(open(p))
    except Exception: return None
def grid(ax):
    ax.grid(True, color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def fig_learning():
    """Fig A: full vs segmented training/validation loss + accuracy (they coincide)."""
    fd = L(FULL / "train_gpu" / "metrics.json"); sd = L(SEGO / "seg_train_gpu_pathb" / "metrics.json")
    fe, se = fd["epochs_detail" if "epochs_detail" in fd else "epochs"], sd["epochs_detail"]
    ep = [r["epoch"] for r in se]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.7))
    # LOSS
    for ax, key_tr, key_va, ylab, title in [(a1, "train_loss", "val_loss", "cross-entropy loss", "Loss"),
                                             (a2, "train_acc", "val_acc", "token accuracy", "Accuracy")]:
        grid(ax)
        # the two curves COINCIDE — draw so BOTH stay visible: full = solid line,
        # segmented = OPEN circles + dashed line (the line shows through the hollow
        # markers and between the dashes -> exact overlap is itself visible)
        ax.plot(ep, [r[key_va] for r in fe], "-", color=FULL_, lw=3.2, alpha=0.9, solid_capstyle="round",
                zorder=2, label="full model")
        ax.plot(ep, [r[key_va] for r in se], "--o", color=SEG, lw=1.6, dashes=(4, 3),
                ms=9, mfc="none", mec=SEG, mew=2.0, zorder=3, label="segmented")
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab); ax.set_title(title, color=INK, loc="left")
        ax.set_xticks(ep)
    a1.legend(loc="upper right", frameon=False)
    # direct-label final accuracy
    fa, sa = fe[-1]["val_acc"], se[-1]["val_acc"]
    a2.annotate(f"{sa:.3f}", (ep[-1], sa), xytext=(-4, 10), textcoords="offset points",
                fontsize=10, color=INK, ha="right", weight="bold")
    fig.suptitle("Validation curves coincide — segmented training learns the same model as the full baseline",
                 fontsize=13.5, weight="bold", color=INK, x=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "comp_learning.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("  comp_learning")


# --------------------------------------------------------------------------- #
SS = L(Path(__file__).resolve().parent / "results" / "scale_summary.json")
_ref = [e for e in SS["scales"] if e["preset"] == "large_8x2x2x8"][0]


def _full_decode(dev):
    """Like-for-like full-model decode timing (full_decode_cost.py); None if not run yet."""
    f = Path(__file__).resolve().parent / "results" / f"full_decode_{dev}" / "full_decode_metrics.json"
    return L(f)["per_token_s"] if f.exists() else None


# (full time, segmented time, unit) per dumbbell row — training on the step protocol,
# inference on the per-decoded-token protocol (identical prompt/decode on both sides)
TIMES = {
    "GPU train": (round(_ref["gpu"]["full_train"]["time_s"], 2),
                  _ref["gpu"]["seg_train"]["time_s"], "s/step"),
    "CPU train": (round(_ref["cpu"]["full_train"]["time_s"]),
                  _ref["cpu"]["seg_train"]["time_s"], "s/step"),
    "GPU infer": (_full_decode("gpu"), _ref["gpu"]["seg_infer"]["per_token_s"], "s/token"),
    "CPU infer": (_full_decode("cpu"), _ref["cpu"]["seg_infer"]["per_token_s"], "s/token"),
}


def _tlab(v, unit):
    if v is None:
        return ""
    txt = f"{v:g}" if v < 10 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")
    return "\n" + txt + " " + unit


def fig_cost_dumbbell():
    """Fig B: peak-memory collapse (full -> segmented) per setting, as a dumbbell (log x).
    Segmented = the clean ablation all-on peak (no one-time build spike); full = profiler peak."""
    # BOTH endpoints of every row come from the ONE 0.84B reference block of
    # results/scale_summary.json, so memory and time on this figure are the same
    # cells as Table "scale" in the paper (CPU cells: results/scale_cpu_large/,
    # GPU segmented inference: results/gpu_inference_a100_40/).
    d = {k: {"full_mem": _ref[dev][f"full_{ph}"]["mem_mb"],
             "seg_mem": _ref[dev][f"seg_{ph}"]["mem_mb"]}
         for k, dev, ph in [("GPU train", "gpu", "train"), ("CPU train", "cpu", "train"),
                            ("GPU infer", "gpu", "infer"), ("CPU infer", "cpu", "infer")]}
    order = ["GPU train", "CPU train", "GPU infer", "CPU infer"]
    order = [k for k in order if d.get(k, {}).get("full_mem") and d[k].get("seg_mem")]
    y = list(range(len(order)))[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.6)); grid(ax); ax.set_xscale("log")
    for yi, k in zip(y, order):
        f, s = d[k]["full_mem"], d[k]["seg_mem"]
        ax.plot([s, f], [yi, yi], "-", color="#d7d7d4", lw=3, zorder=2, solid_capstyle="round")
        ax.plot(f, yi, "o", ms=13, color=FULL_, mec=SURF, mew=2, zorder=3)
        ax.plot(s, yi, "o", ms=13, color=SEG, mec=SURF, mew=2, zorder=4)
        ft, st, unit = TIMES.get(k, (None, None, ""))
        ax.annotate(f"{f/1000:.1f} GB" + _tlab(ft, unit), (f, yi), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=10, color=INK2)
        ax.annotate(f"{s:.0f} MB" + _tlab(st, unit), (s, yi), xytext=(-10, 0),
                    textcoords="offset points", va="center", ha="right", fontsize=10,
                    color=INK, weight="bold")
        ax.annotate(f"full = {f/s:.0f}$\\times$ segmented", ((s*f)**0.5, yi), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10.5, color=SEG, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=11.5, color=INK)
    ax.set_xlabel("peak memory (log scale)\neach point: peak memory + time \u2014 "
                  "s/step (training) or s/decoded-token (inference), same protocol on both sides",
                  fontsize=10)
    ax.set_xlim(280, 60000)
    ax.set_xticks([500, 1000, 5000, 10000, 50000]); ax.set_xticklabels(["500 MB", "1 GB", "5 GB", "10 GB", "50 GB"])
    ratios = sorted(d[k]["full_mem"] / d[k]["seg_mem"] for k in order)
    ax.set_title(f"Peak memory: the full model needs {ratios[0]:.0f}–{ratios[-1]:.0f}$\\times$ "
                 f"the segmented footprint, at identical quality",
                 color=INK, weight="bold", loc="left")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=FULL_, ms=11, label="full model"),
                       Line2D([0], [0], marker="o", color="none", markerfacecolor=SEG, ms=11, label="segmented")],
              loc="lower right", frameon=False)
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.tight_layout(); fig.savefig(OUT / "comp_cost_dumbbell.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("  comp_cost_dumbbell")


# --------------------------------------------------------------------------- #
def fig_onnx():
    """Fig E: inference peak memory across BOTH levers — full->segmented and torch->ONNX.
    Constrained resource per device (GPU: VRAM, CPU: RSS). Log y so every step is visible."""
    FT, FO, ST, SO = "#6b6b67", "#c2c2be", "#17539c", "#3987e5"   # full-torch/onnx, seg-torch/onnx
    def nest(d, *ks):
        for k in ks: d = d[k]
        return d
    # read every value from its source metrics file (full = full_model profiler; seg = B2 torch / C ONNX)
    ft_g = nest(L(FULL/"infer_cost_profiler_gpu"/"infer_cost_profiler_metrics.json"), "memory","vram","nvidia_smi_peak_mb")
    fo_g = nest(L(FULL/"onnx_infer_cost_profiler_gpu"/"onnx_infer_cost_profiler_metrics.json"), "memory","vram","nvidia_smi_peak_mb")
    ft_c = nest(L(FULL/"infer_cost_profiler_cpu"/"infer_cost_profiler_metrics.json"), "memory","host_cpu_ram","peak_mb")
    fo_c = nest(L(FULL/"onnx_infer_cost_profiler_cpu"/"onnx_infer_cost_profiler_metrics.json"), "memory","host_cpu_ram","peak_mb")
    st_g = L(SEGO/"seg_b2_gpu"/"seg_b2_metrics.json")["overall"]["vram_hw_total_peak_mb"]
    so_g = L(SEGO/"seg_c_gpu"/"seg_c_metrics.json")["peak"]["vram_mb"]
    so_c = L(SEGO/"seg_c_cpu"/"seg_c_metrics.json")["peak"]["rss_mb"]
    # seg torch CPU: clean from_scratch rerun (no model-build spike) -> directly measured
    st_c = L(Path(__file__).resolve().parent/"results"/"seg_b2_cpu_clean"/"seg_b2_clean_metrics.json")["overall"]["rss_peak_sampled_mb"]
    data = {"GPU  (peak VRAM)": [("full\ntorch", ft_g, FT), ("full\nONNX", fo_g, FO),
                                 ("seg\ntorch", st_g, ST), ("seg\nONNX", so_g, SO)],
            "CPU  (peak RSS)":  [("full\ntorch", ft_c, FT), ("full\nONNX", fo_c, FO),
                                 ("seg\ntorch", st_c, ST), ("seg\nONNX", so_c, SO)]}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    for ax, (title, bars) in zip(axs, data.items()):
        grid(ax); ax.set_yscale("log")
        for i, (lab, v, c) in enumerate(bars):
            ax.bar(i, v, 0.72, color=c, zorder=3)
            ax.annotate(f"{v/1000:.1f} GB" if v >= 1000 else f"{v:.0f} MB", (i, v),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color=INK2)
        ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in bars], fontsize=10)
        ax.set_title(title, color=INK, loc="left")
        ax.set_ylim(300, 9000)
    axs[0].set_ylabel("peak memory, log scale")
    # annotate the two levers on the GPU panel
    fig.suptitle("Two levers on inference memory: segmentation (large drop) then torch-free ONNX",
                 fontsize=13.5, weight="bold", color=INK)
    fig.text(0.5, -0.02, "gray = full model · blue = segmented · darker = PyTorch, lighter = torch-free ONNX",
             ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT / "comp_onnx.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("  comp_onnx")


def fig_granularity():
    """E x A x M x H granularity -> memory & time, one trajectory (memory x, time y), no dual axis.
    The code is a 4-tuple (segments per component), NOT a product."""
    RES = Path(__file__).resolve().parent / "results"
    g = L(RES / "gpu_granularity" / "granularity_train.json")["grid"]
    mem = [r["vram_peak_mb"] for r in g]; t = [r["step_time_s"] for r in g]
    code = [r["code"] for r in g]
    fig, ax = plt.subplots(figsize=(8.8, 5.4)); grid(ax)
    ax.plot(mem, t, "-", color="#cfd3d6", lw=2.5, zorder=2, solid_capstyle="round")
    ax.scatter(mem, t, s=150, color=SEG, zorder=4, edgecolors=SURF, linewidths=2)
    offs = [(14, -3), (14, -2), (-6, 17), (16, 0), (16, 2)]
    has  = ["left", "left", "center", "left", "left"]
    for i2, (m, tt, c) in enumerate(zip(mem, t, code)):
        ax.annotate(c.replace("x", "$\\times$"), (m, tt), xytext=offs[i2], textcoords="offset points",
                    fontsize=11, color=INK, va="center", ha=has[i2], weight="bold")
    ax.annotate("coarser split", (mem[0], t[0]), xytext=(-12, -22),
                textcoords="offset points", ha="right", fontsize=9.5, color=MUTED)
    ax.annotate("finer split", (mem[-1], t[-1]), xytext=(34, 16),
                textcoords="offset points", ha="left", fontsize=9.5, color=MUTED)
    ax.text(0.5, -0.155, "label = segments per component:  embedding $\\times$ attention $\\times$ MLP $\\times$ output-head",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK2)
    ax.set_xlabel("peak VRAM (MB)"); ax.set_ylabel("step time (s)")
    ax.set_xlim(680, 2120); ax.margins(y=0.18)
    ax.set_title("Finer segmentation trades memory for time — with diminishing memory returns",
                 color=INK, weight="bold", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "comp_granularity.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("  comp_granularity")


def fig_memflow():
    """Memory over time (2x2): the segmented sawtooth (one segment at a time) stays bounded,
    while the full model would be flat and off-scale. Real MemFlow timelines."""
    RES = Path(__file__).resolve().parent / "results"
    # off-scale full-model reference peaks: the SAME 0.84B cells as the dumbbell and the
    # scale table (results/scale_summary.json), not hard-coded literals.
    fullpk = {"GPU training": _ref["gpu"]["full_train"]["mem_mb"],
              "CPU training": _ref["cpu"]["full_train"]["mem_mb"],
              "GPU inference": _ref["gpu"]["full_infer"]["mem_mb"],
              "CPU inference": _ref["cpu"]["full_infer"]["mem_mb"]}
    # CPU training trace = the cpu_ram_timeline of the 0.84B CPU scale run itself
    # (results/scale_cpu_large/), i.e. the very run whose 997 MB peak the text quotes.
    panels = [("GPU training", SEGO/"seg_cost_gpu"/"seg_cost_metrics.json", True),
              ("CPU training", RES/"scale_cpu_large"/"seg_train_metrics.json", False),
              ("GPU inference", SEGO/"seg_b2_gpu"/"seg_b2_metrics.json", True),
              ("CPU inference", SEGO/"seg_b2_cpu"/"seg_b2_metrics.json", False)]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.6))
    ratios = []
    for ax, (name, path, cuda) in zip(axs.flat, panels):
        fpk = fullpk[name]
        d = L(path); grid(ax)
        if cuda:
            ctx = d.get("baseline", {}).get("cuda_context_mb", 0.0)
            tl = d["vram_timeline"]; t = [r[0] for r in tl]; y = [ctx + r[2] for r in tl]   # context+reserved
            ylab = "VRAM (MB)"
        else:
            tl = d["cpu_ram_timeline"]; t = [r[0] for r in tl]; y = [r[1] for r in tl]
            ylab = "RSS (MB)"
        t0 = t[0]; t = [x - t0 for x in t]
        # steady-state window: middle 45% of the run (skip build/prefill and tail)
        n = len(t); a, b = int(n * 0.35), int(n * 0.80)
        t, y = t[a:b], y[a:b]
        t = [x - t[0] for x in t]
        step = max(1, len(t) // 2500)                      # downsample for a clean render
        t, y = t[::step], y[::step]
        pk = max(y)
        ax.fill_between(t, 0, y, color=SEG, alpha=0.13, zorder=2)
        ax.plot(t, y, color=SEG, lw=1.1, zorder=3)
        ax.set_ylim(0, pk * 1.35); ax.set_xlim(0, t[-1])
        ax.set_title(f"{name}   ·   peak {pk:.0f} MB", color=INK, loc="left", fontsize=12)
        ax.set_ylabel(ylab); ax.set_xlabel("elapsed time (s)")
        ratios.append((name, fpk / pk))
        ax.annotate(f"full model ≈ {fpk/1000:.0f} GB  ($\\approx${fpk/pk:.0f}$\\times$ higher, off-scale)",
                    (0.5, 0.9), xycoords="axes fraction", ha="center", fontsize=9.5, color=MUTED)
    tr = sorted(r for n, r in ratios if "training" in n)
    print("    memflow ratios:", [(n, round(r, 1)) for n, r in ratios],
          f"-> training range {tr[0]:.0f}–{tr[-1]:.0f}x")
    fig.suptitle("Memory through a step: one segment at a time — a bounded sawtooth, never the whole model",
                 fontsize=14, weight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "comp_memflow.png", bbox_inches="tight", facecolor=SURF)
    plt.close(fig); print("  comp_memflow")


if __name__ == "__main__":
    print("comparison figures ->", OUT)
    fig_learning(); fig_cost_dumbbell(); fig_onnx(); fig_granularity(); fig_memflow()

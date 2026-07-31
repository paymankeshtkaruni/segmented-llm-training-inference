#!/usr/bin/env python
"""
Publication-quality figures for the paper. One clean figure per concept, consistent style,
no overlapping labels. Reads the committed result JSONs. -> results/figures/paper/*.png
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "figures" / "paper"; OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12, "axes.linewidth": 0.8,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
BLUE, RED, GREEN, ORANGE, GREY = "#2a78d6", "#c0392b", "#1baf7a", "#eb6834", "#8a8a86"

RUNG_SHORT = {  # compact technique labels
    "r0_baseline": "baseline", "r1_sdpa": "SDPA", "r2_mlp_sum": "MLP-sum",
    "r3_chunked_ce": "chunked-CE", "r4_recompute": "recompute", "r5_stream": "stream",
    "r6_records": "records", "r7_park_grads": "park-grads", "r8_offload_adam": "offload-Adam",
    "r9_segment_wo": "seg-$W_o$", "r10_free_device": "free-dev",
    # inference
    "r4_stream": "stream", "r5_segment_wo": "seg-$W_o$", "r6_free_device": "free-dev",
    "r7_no_kv_cache": "no-KV",
}


def load(sub, jn):
    p = RES / sub / jn
    return json.load(open(p))["ladder"] if p.exists() else None


# --------------------------------------------------------------------------- #
def fig_schematic():
    fig, ax = plt.subplots(figsize=(12, 5.6)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 10)
    ax.text(6, 9.5, "Segmented execution: one segment resident at a time",
            ha="center", fontsize=15, weight="bold")
    # left: four axis boxes, two lines each (no horizontal collision)
    ax.text(2.4, 8.95, "GPT decoder — four segmentation axes", ha="center", fontsize=11.5, weight="bold")
    axes = [("E — embedding", "split $d_{model}$, reassemble by concat", BLUE),
            ("A — attention", "head-groups, concat then shared $W_o$", GREEN),
            ("M — MLP", "$d_{ff}$ chunks, reassemble by running-sum", ORANGE),
            ("H — output head", "vocab slices, streamed cross-entropy", RED)]
    for i, (name, how, c) in enumerate(axes):
        y = 7.5 - i * 1.65
        ax.add_patch(FancyBboxPatch((0.4, y), 4.0, 1.25, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=c, ec="black", lw=1.1, alpha=0.92))
        ax.text(2.4, y + 0.82, name, ha="center", va="center", fontsize=12, weight="bold", color="white")
        ax.text(2.4, y + 0.36, how, ha="center", va="center", fontsize=8.8, color="white")
    # middle: constrained device
    dx, dy, dw, dh = 5.9, 2.7, 2.7, 4.9
    ax.add_patch(FancyBboxPatch((dx, dy), dw, dh, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc="#F4F6F7", ec="black", lw=2))
    ax.text(dx + dw/2, dy + dh - 0.45, "constrained device", ha="center", fontsize=12, weight="bold")
    ax.text(dx + dw/2, dy + dh - 0.95, "GPU VRAM / CPU RAM", ha="center", fontsize=9, color="#555")
    ax.add_patch(FancyBboxPatch((dx + 0.35, dy + dh/2 - 0.55), dw - 0.7, 1.1,
                                boxstyle="round,pad=0.02,rounding_size=0.1", fc=BLUE, ec="black", alpha=0.92))
    ax.text(dx + dw/2, dy + dh/2, "1 active segment", ha="center", va="center", fontsize=11.5,
            weight="bold", color="white")
    ax.text(dx + dw/2, dy + 0.5, "+ tiny resident\nnorms / biases", ha="center", va="center",
            fontsize=8.8, color="#555")
    # right: backing store
    sx, sy, sw, sh = 9.4, 2.7, 2.3, 4.9
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc="#ECEFF1", ec="black", lw=1.4))
    ax.text(sx + sw/2, sy + sh - 0.45, "backing store", ha="center", fontsize=12, weight="bold")
    ax.text(sx + sw/2, sy + sh - 0.95, "GPU: host RAM\nCPU: disk", ha="center", fontsize=8.8, color="#555")
    for k in range(6):
        ax.add_patch(FancyBboxPatch((sx + 0.35 + (k % 3) * 0.62, sy + 1.75 + (k // 3) * 0.7), 0.5, 0.5,
                                    boxstyle="round,pad=0.01", fc=GREY, ec="#5D6D7E", alpha=0.7))
    ax.text(sx + sw/2, sy + 1.25, "all other segments,\ngrads, optimizer state",
            ha="center", va="center", fontsize=8.2, color="#555")
    # arrows
    ax.add_patch(FancyArrowPatch((4.6, dy + dh/2), (dx - 0.05, dy + dh/2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, color="#333"))
    ax.add_patch(FancyArrowPatch((dx + dw + 0.05, dy + dh/2 + 0.45), (sx - 0.05, dy + dh/2 + 0.45),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.5, color=RED))
    ax.add_patch(FancyArrowPatch((sx - 0.05, dy + dh/2 - 0.45), (dx + dw + 0.05, dy + dh/2 - 0.45),
                                 arrowstyle="-|>", mutation_scale=14, lw=1.5, color=GREEN))
    ax.text((dx + dw + sx) / 2, dy + dh/2 + 0.75, "evict", ha="center", fontsize=9, color=RED)
    ax.text((dx + dw + sx) / 2, dy + dh/2 - 0.9, "load", ha="center", fontsize=9, color=GREEN)
    ax.text(6, 1.75, "backward by recomputation (no retained graph)   ·   streamed segment-wise AdamW",
            ha="center", fontsize=10, style="italic", color="#333")
    ax.text(6, 1.15, "peak memory ≈ one segment, not the whole model", ha="center", fontsize=11,
            weight="bold", color="#333")
    ax.text(6, 0.65, "identical weights — only the memory schedule changes", ha="center",
            fontsize=10, color=BLUE)
    fig.savefig(OUT / "fig1_schematic.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  fig1_schematic")


# --------------------------------------------------------------------------- #
def fig_waterfall():
    rows = load("gpu_train", "ladder_train.json")
    mem = [r["vram_peak_mb"] / 1000 for r in rows]
    t = [r["step_time_s"] for r in rows]
    fig, (ax, at) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    ax.bar(0, mem[0], width=0.62, color=GREY, ec="black", lw=0.6)
    ax.annotate(f"{mem[0]:.1f}", (0, mem[0]), ha="center", va="bottom", fontsize=9, weight="bold")
    labels = ["baseline"]
    for i in range(1, len(mem)):
        drop = mem[i-1] - mem[i]
        lo, hi = min(mem[i-1], mem[i]), max(mem[i-1], mem[i])
        col = BLUE if drop >= 0 else ORANGE
        ax.bar(i, hi - lo, bottom=lo, width=0.62, color=col, ec="black", lw=0.5, alpha=0.9)
        if abs(drop) >= 0.3:
            ax.annotate(f"$-${drop:.1f}" if drop >= 0 else f"$+${-drop:.1f}", (i, hi),
                        ha="center", va="bottom", fontsize=8, color="#333")
        labels.append("+" + RUNG_SHORT.get(rows[i]["rung"], ""))
    ax.bar(len(mem), mem[-1], width=0.62, color=GREEN, ec="black", lw=0.6)
    ax.annotate(f"{mem[-1]:.2f}", (len(mem), mem[-1]), ha="center", va="bottom", fontsize=9, weight="bold")
    labels.append("all-on")
    ax.set_ylabel("peak VRAM (GB)")
    ax.set_title("Where the 18 GB goes: memory reclaimed and time paid, per technique (GPU training)")
    ax.margins(x=0.02)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=GREY, ec="k", label="baseline peak"),
                       Patch(fc=BLUE, ec="k", label="reclaimed"),
                       Patch(fc=GREEN, ec="k", label="residual (context+1 segment)")],
              loc="upper right", frameon=True)
    # bottom row: the step time each cumulative technique set pays (all-on = last rung)
    tt = t + [t[-1]]
    at.bar(range(len(tt)), tt, width=0.62, color=RED, ec="black", lw=0.5, alpha=0.85)
    at.set_yscale("log"); at.set_ylim(3, 500)
    for i, v in enumerate(tt):
        at.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=8.5, color="#333")
    at.set_ylabel("step time (s, log)")
    at.set_xticks(range(len(tt))); at.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    at.margins(x=0.02)
    fig.savefig(OUT / "fig2_waterfall.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  fig2_waterfall")


# --------------------------------------------------------------------------- #
CATS = [("gpu_train", "ladder_train.json", True, "step_time_s", "step time (s)", "GPU training"),
        ("cpu_train", "ladder_train.json", False, "step_time_s", "step time (s)", "CPU training"),
        ("gpu_inference", "ladder_infer.json", True, "per_token_s", "per-token time (s)", "GPU inference"),
        ("cpu_inference", "ladder_infer.json", False, "per_token_s", "per-token time (s)", "CPU inference")]


def fig_master():
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.4))
    for ax, (sub, jn, cuda, tk, tl, title) in zip(axs.flat, CATS):
        rows = load(sub, jn); memk = "vram_peak_mb" if cuda else "rss_peak_mb"
        mem = [r[memk] / 1000 for r in rows]; t = [r[tk] for r in rows]
        x = range(len(rows))
        b = ax.bar(x, mem, color=BLUE, alpha=0.85, label="peak memory")
        ax.set_ylabel(("VRAM" if cuda else "RSS") + " (GB)", color=BLUE)
        ax.tick_params(axis="y", labelcolor=BLUE)
        ax.set_xticks(list(x)); ax.set_xticklabels([str(i) for i in x], fontsize=9)
        ax.set_xlabel("rung (cumulative techniques)")
        a2 = ax.twinx(); a2.spines["top"].set_visible(False)
        ln, = a2.plot(list(x), t, "o-", color=RED, lw=1.8, ms=4, label=tl)
        a2.set_ylabel(tl, color=RED); a2.tick_params(axis="y", labelcolor=RED)
        ax.set_title(f"{title}: {mem[0]:.1f}→{mem[-1]*1000:.0f} MB "
                     f"({mem[0]*1000/max(mem[-1]*1000,1):.0f}×),  {t[0]:.1f}→{t[-1]:.0f} s",
                     fontsize=11)
    fig.suptitle("Cumulative technique staircase — peak memory (bars) ↓, time (line) ↑",
                 fontsize=14, weight="bold")
    # shared rung legend
    fig.text(0.5, 0.005, "rungs 0–10 add: 1 SDPA · 2 MLP-sum · 3 chunked-CE · 4 recompute · "
             "5 stream · 6 records · 7 park-grads · 8 offload-Adam · 9 seg-$W_o$ · 10 free-device "
             "  (inference omits 4/6/7/8; adds no-KV last)", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / "fig3_master.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  fig3_master")


# --------------------------------------------------------------------------- #
def fig_pareto():
    rows = load("gpu_train", "ladder_train.json")
    mem = [r["vram_peak_mb"] / 1000 for r in rows]; t = [r["step_time_s"] for r in rows]
    # reference points on the same protocol: eager full model + DeepSpeed ZeRO-Offload
    full = json.load(open(RES / "comparison" / "cost_compare.json"))["GPU train"]
    # DeepSpeed ZeRO-Offload: the 16-pinned-thread runs, replicated on THREE distinct
    # A100-40GB nodes. The peak was bit-identical across nodes; the step time was not, so
    # each configuration is drawn as its median with a vertical bar spanning the node
    # spread (min-max) rather than as a single number from one node.
    DS_DIRS = ["deepspeed_baseline_16t", "deepspeed_baseline_16t_rep1",
               "deepspeed_baseline_16t_rep2"]
    zero = {}
    for zm in ("zero2", "zero3"):
        runs = [json.load(open(RES / dd / f"{zm}_offload_metrics.json")) for dd in DS_DIRS]
        ts = sorted(r["avg_step_time_s"] for r in runs)
        mems = {round(r["overall"]["vram_hw_total_peak_mb"], 1) for r in runs}
        zero[zm] = {"mem_gb": max(mems) / 1000, "med": ts[len(ts) // 2],
                    "lo": ts[0], "hi": ts[-1], "n": len(ts)}
    z2, z3 = zero["zero2"], zero["zero3"]
    refs = [("eager full model", full["full_mem"] / 1000, full["full_time"], GREY),
            ("zero2", z2["mem_gb"], z2["med"], ORANGE),
            ("zero3", z3["mem_gb"], z3["med"], ORANGE)]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(mem, t, "-", color=GREY, lw=1.4, zorder=1)
    ax.scatter(mem, t, s=42, color=BLUE, zorder=3, ec="black", lw=0.5)
    for z in (z2, z3):   # node-to-node spread, drawn behind the median marker
        ax.plot([z["mem_gb"], z["mem_gb"]], [z["lo"], z["hi"]], "-", color=ORANGE,
                lw=1.6, solid_capstyle="butt", zorder=3)
        for cap in (z["lo"], z["hi"]):
            ax.plot([z["mem_gb"] * 0.965, z["mem_gb"] * 1.035], [cap, cap], "-",
                    color=ORANGE, lw=1.4, zorder=3)
    for _, m_, t_, c in refs:
        ax.scatter([m_], [t_], s=70, marker="D", color=c, ec="black", lw=0.7, zorder=4)
    ax.annotate(f"eager full model\n{refs[0][1]:.1f} GB, {refs[0][2]:.2f} s",
                (refs[0][1], refs[0][2]), textcoords="offset points",
                xytext=(-12, -6), fontsize=8.5, ha="right", va="top", color="#444")
    ax.annotate(f"ZeRO-2 / ZeRO-3 offload   {z2['mem_gb']:.1f} / {z3['mem_gb']:.1f} GB\n"
                f"{z2['med']:.1f} / {z3['med']:.1f} s median "
                f"(bar = {z2['n']}-node spread {z2['lo']:.1f}–{z2['hi']:.1f} / "
                f"{z3['lo']:.1f}–{z3['hi']:.1f} s)",
                (z2["mem_gb"], z2["lo"]), textcoords="offset points",
                xytext=(-14, -12), fontsize=8.5, ha="right", va="top", color="#8a4a10")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.75, 48); ax.set_ylim(0.4, 320)
    ax.set_xlabel("peak VRAM (GB, log)   \u2190 more techniques"); ax.set_ylabel("step time (s, log)")
    ax.set_title("Memory\u2013time frontier (GPU training): pick a budget, read the time")
    # label only key operating points, with offsets that don't collide
    key = {0: (10, 8, "baseline 18\u2009GB"), 1: (0, 12, "+SDPA"), 5: (-6, 12, "+stream"),
           6: (-4, -16, "+records 1.3\u2009GB"), 10: (10, 6, "all-on 956\u2009MB")}
    for i, (dx, dy, lab) in key.items():
        ax.annotate(lab, (mem[i], t[i]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9, ha="center", color="#222",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#999"))
    for gb in [1, 2, 4, 8, 16]:
        ax.axvline(gb, color="#ddd", lw=0.8, zorder=0)
        if gb > 1:  # the 1 GB label would collide with the all-on annotation
            ax.text(gb, ax.get_ylim()[1]*0.80, f"{gb}GB", rotation=90, va="top", ha="right",
                    fontsize=7.5, color="#999")
    ax.grid(True, which="both", alpha=0.15)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=BLUE, mec="k", ms=7, label="segmented (cumulative sets)"),
                       Line2D([0], [0], marker="D", ls="", mfc=GREY, mec="k", ms=7, label="eager full model"),
                       Line2D([0], [0], marker="D", ls="", mfc=ORANGE, mec="k", ms=7, label="DeepSpeed ZeRO-Offload")],
              loc="lower left", frameon=True, fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "fig4_pareto.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  fig4_pareto")


# --------------------------------------------------------------------------- #
def fig_loo():
    o = json.load(open(RES / "gpu_train_loo" / "loo_train.json"))
    techs = sorted(o["techniques"], key=lambda r: r["saves_mem_mb"])
    names = {"sdpa": "SDPA", "mlp_running_sum": "MLP running-sum", "chunked_ce": "chunked-CE",
             "stream_segments": "stream segments", "offload_records": "offload records",
             "park_grads_host": "park grads", "offload_adam": "offload Adam",
             "segment_wo": "stream $W_o$", "free_device": "free-device"}
    lbl = [names.get(t["technique"], t["technique"]) for t in techs]
    save_mb = [t["saves_mem_mb"] for t in techs]; cost = [t["costs_time_s"] for t in techs]
    y = range(len(techs))
    cols = [RED if c > 5 else GREEN for c in cost]      # red = costs real time; green = ~free
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.barh(list(y), [s / 1000 for s in save_mb], color=cols, ec="black", lw=0.4)
    ax.set_yticks(list(y)); ax.set_yticklabels(lbl)
    for i in y:  # annotate memory saved; append time ONLY for the time-costly (reliable) ones
        txt = f"{save_mb[i]:.0f} MB" + (f",  +{cost[i]:.0f}s" if cost[i] > 5 else "")
        ax.annotate(txt, (max(save_mb[i] / 1000, 0), i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.6, color=(RED if cost[i] > 5 else "#333"))
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("memory saved vs all-on (GB)")
    ax.set_title("Leave-one-out marginal value (GPU training):\nstreaming dominates; the rest are small but time-free")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=GREEN, ec="k", label="time-free"),
                       Patch(fc=RED, ec="k", label="costs real time (>5 s)")],
              loc="lower right", frameon=True, title="time cost")
    ax.margins(x=0.26)
    fig.tight_layout(); fig.savefig(OUT / "fig5_loo.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  fig5_loo")


# --------------------------------------------------------------------------- #
def fig_granularity():
    g = json.load(open(RES / "gpu_granularity" / "granularity_train.json"))["grid"]
    codes = [r["code"] for r in g]; mem = [r["vram_peak_mb"]/1000 for r in g]; t = [r["step_time_s"] for r in g]
    fig, ax = plt.subplots(figsize=(8.2, 5)); x = range(len(g))
    ax.bar(x, mem, color=BLUE, alpha=0.85, width=0.6)
    ax.set_ylabel("peak VRAM (GB)", color=BLUE); ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_xticks(list(x)); ax.set_xticklabels(codes, fontsize=10)
    ax.set_xlabel("granularity  $E{\\times}A{\\times}M{\\times}H$  (coarse → fine)")
    for i in x:
        ax.annotate(f"{mem[i]*1000:.0f}", (i, mem[i]), ha="center", va="bottom", fontsize=8.5, color=BLUE)
    a2 = ax.twinx(); a2.spines["top"].set_visible(False)
    a2.plot(list(x), t, "o-", color=RED, lw=2, ms=6); a2.set_ylabel("step time (s)", color=RED)
    a2.tick_params(axis="y", labelcolor=RED)
    ax.set_title("Granularity dial (GPU training, all techniques on):\nfiner → less memory, more time; diminishing returns past 16×4×4×16")
    fig.tight_layout(); fig.savefig(OUT / "fig6_granularity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  fig6_granularity")


# --------------------------------------------------------------------------- #
def fig_phase():
    """Decomposition of the WHOLE step. Read from the all-on run's per_phase block, which
    carries a fourth phase (validation) that the earlier ladder-based version dropped, so
    the bars used to sum to only 133 s of the 164.5 s GPU step."""
    PANELS = [("gpu_train/r10_free_device_metrics.json", "GPU training"),
              ("scale_cpu_large/seg_train_metrics.json", "CPU training")]
    KEYS = [("forward", "forward", GREEN), ("backward", "backward\n(recompute)", RED),
            ("optimizer", "optimizer", ORANGE), ("validation", "validation", BLUE)]
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for ax, (rel, title) in zip(axs, PANELS):
        m = json.load(open(RES / rel))
        ph = m["per_phase"]
        vals = [ph[k]["time_ms"] / 1000 for k, _, _ in KEYS]
        tot = sum(vals)
        step = m.get("avg_step_time_s") or tot
        ax.bar([lab for _, lab, _ in KEYS], vals, color=[c for *_, c in KEYS],
               ec="black", lw=0.5, width=0.62)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}s\n{100*v/max(tot,1e-9):.0f}%", (i, v), ha="center",
                        va="bottom", fontsize=9)
        ax.set_title(f"{title}  ·  {step:.1f} s step", fontsize=12)
        ax.set_ylabel("time (s)"); ax.margins(y=0.22)
        ax.tick_params(axis="x", labelsize=9.5)
        print(f"  phase {title}: " + ", ".join(f"{lab.splitlines()[0]} {v:.1f}s"
              for (_, lab, _), v in zip(KEYS, vals)) + f"  sum {tot:.1f}s / step {step:.1f}s")
    fig.suptitle("Where the step time goes: all four phases of the step, backward dominated by recomputation",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig7_phase.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  fig7_phase")


if __name__ == "__main__":
    print("clean paper figures ->", OUT)
    fig_schematic(); fig_waterfall(); fig_master(); fig_pareto(); fig_loo(); fig_granularity(); fig_phase()

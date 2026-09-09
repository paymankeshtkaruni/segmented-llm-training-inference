#!/usr/bin/env python
"""exp8: per-step / per-request evolution plots for every long-horizon job.

For each training job (g1-g9): two stacked panels — per-step peak memory (MiB,
printed as MB like every other table; raw rows are decimal MB, converted here)
and per-step wall time (s) with the running average overlaid — from step 1 to
the last step. Same per request for the serving jobs (ia-id). These are the
"how do time and memory change throughout the steps" plots; the running
average visibly converging answers where the long-run average settles.

Writes results/exp8_sustained/plots/exp8_training_evolution.pdf and
exp8_serving_evolution.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results" / "exp8_sustained"
OUT = R / "plots"
MIB_PER_MB = 1e6 / 2**20          # decimal MB -> MiB, the paper-wide unit
OUT.mkdir(exist_ok=True)


NAMES = {
    "g1_084b_resident_imm":      "0.84B GPU  |  retained graph, resident, in-backward (fastest)",
    "g2_084b_resident_recomp":   "0.84B GPU  |  recomputation, resident, deferred",
    "g3_084b_streamed_def":      "0.84B GPU  |  recomputation, streamed, deferred (memory floor)",
    "g4_084b_streamed_imm":      "0.84B GPU  |  recomputation, streamed, in-backward",
    "g5_69b_resident_recomp":    "6.9B GPU  |  recomputation, resident, deferred",
    "g6_69b_streamed_imm":       "6.9B GPU  |  recomputation, streamed, in-backward",
    "g7_084b_cpu_streamed_def":  "0.84B CPU  |  recomputation, streamed, deferred",
    "g8_084b_cpu_resident_imm":  "0.84B CPU  |  retained graph, resident, in-backward",
    "g9_69b_cpu_streamed_imm":   "6.9B CPU  |  recomputation, streamed, in-backward",
    "ia_084b_cache":             "0.84B GPU serving  |  resident, KV cache",
    "ib_084b_streamed":          "0.84B GPU serving  |  streamed, recompute",
    "ic_084b_onnx_disk":         "0.84B CPU serving  |  ONNX runtime, disk-streamed weights",
    "id_69b_cache":              "6.9B GPU serving  |  resident, KV cache",
}

plt.rcParams.update({"font.size": 7, "font.family": "serif",
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def load(kind):
    jobs = []
    for d in sorted(R.iterdir()):
        if not d.is_dir() or d.name in ("pilot_30step", "plots"):
            continue
        fs = list(d.glob(f"*_{kind}.json"))
        if fs:
            jobs.append((d.name, json.load(open(fs[0]))))
    return jobs


def running_mean(xs):
    out, s = [], 0.0
    for i, x in enumerate(xs, 1):
        s += x
        out.append(s / i)
    return out


def plot(kind, series_key, xlabel, fname, title):
    jobs = load(kind)
    n = len(jobs)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 1.35 * n), squeeze=False)
    for row, (name, d) in enumerate(jobs):
        rows = d[series_key]
        x = [r.get("step") or r.get("req") for r in rows]
        mem = [r["peak_mb"] * MIB_PER_MB if r["peak_mb"] is not None else None for r in rows]
        t = [r["s"] for r in rows]
        am, at = axes[row][0], axes[row][1]
        if all(m is not None for m in mem):
            am.plot(x, mem, lw=0.5, color="#4c72b0")
            lo, hi = min(mem), max(mem)
            pad = max(5.0, (hi - lo) * 0.3)
            am.set_ylim(lo - pad, hi + pad)
            am.set_ylabel("MB", fontsize=6)
            am.annotate(f"band {hi - lo:.0f} MB", xy=(0.99, 0.9),
                        xycoords="axes fraction", ha="right", fontsize=6)
        else:
            am.text(0.5, 0.5, "n/a", ha="center", transform=am.transAxes)
        at.plot(x, t, lw=0.4, color="0.6", label="per step")
        at.plot(x, running_mean(t), lw=1.0, color="#c44e52",
                label="running avg")
        at.set_ylabel("s", fontsize=6)
        at.annotate(f"avg {sum(t)/len(t):.2f} s", xy=(0.99, 0.9),
                    xycoords="axes fraction", ha="right", fontsize=6)
        if row == 0:
            at.legend(fontsize=5.5, loc="lower right")
        pretty = NAMES.get(name, name)
        am.set_title(f"{pretty} — memory", fontsize=6.5, loc="left")
        at.set_title(f"{pretty} — time", fontsize=6.5, loc="left")
        for ax in (am, at):
            ax.grid(alpha=.3)
            if row == n - 1:
                ax.set_xlabel(xlabel, fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(OUT / fname)
    plt.close(fig)
    print("wrote", OUT / fname)






def fig_validation_cost():
    """exp9 figure candidate: per configuration, the training-only step time
    (long-run average, exp8) with the directly timed per-step validation cost
    (exp9) stacked on top -- what the four-phase measurement protocol adds."""
    bridge = json.load(open(HERE / "results" / "exp9_validation_bridge" / "bridge_summary.json"))["jobs"]
    order = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"]
    prefix2dir = {k.split("_", 1)[0]: k for k in NAMES}
    labels, train, val = [], [], []
    for g in order:
        r = bridge.get(g, {})
        if "exp8_full_avg_step_s" not in r:
            continue
        labels.append(NAMES[prefix2dir[g]].replace("  |  ", "\n"))
        train.append(r["exp8_full_avg_step_s"])
        val.append(r["exp9_validation_phase_s_direct"] or 0)
    if not labels:
        print("fig_validation_cost: no bridge rows available, skipping")
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = range(len(labels))
    ax.barh(y, train, color="#4c72b0", label="training step (long-run average)")
    ax.barh(y, val, left=train, color="#c44e52",
            label="per-step validation pass (directly timed)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("seconds per step (log scale)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for i, (t, v) in enumerate(zip(train, val)):
        ax.annotate(f"+{v:.1f}s ({100*v/(t+v):.0f}%)", xy=(t + v, i),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_xlim(right=max(t + v for t, v in zip(train, val)) * 1.8)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp9_validation_cost.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_exp9_validation_cost.pdf")

if __name__ == "__main__":
    plot("long", "per_step", "step", "fig_exp8_training.pdf",
         "Long-horizon training: per-step peak memory and whole-step time (running average)")
    plot("serve", "per_request", "request", "fig_exp8_serving.pdf",
         "Long-horizon serving: per-request peak memory and latency (running average)")
    fig_validation_cost()

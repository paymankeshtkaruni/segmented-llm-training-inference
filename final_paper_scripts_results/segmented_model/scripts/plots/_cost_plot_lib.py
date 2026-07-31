#!/usr/bin/env python
"""Shared plotting engine for the segmented COST experiments (A2 train, B2 torch infer,
D granularity train). Each figure function takes (metrics_dict, out_dir, prefix) and writes
`<prefix>_<kind>.png`, so the SAME engine produces correctly-named files for every experiment
(seg_a2_*, seg_a2_cpu_*, seg_b2_*, seg_b2_cpu_*, seg_d16_*, seg_d16_cpu_*).

Figures:
  <prefix>_memory_flow.png      VRAM + CPU-RAM timelines over the whole run (two panels; CPU = 1)
  <prefix>_phase_breakdown.png  per-phase VRAM-by-category + CPU-RAM floor/peak
  <prefix>_round_stability.png  cross-round drift (TRAINING only; needs 'forward' round markers)
  <prefix>_timing.png           per-phase time, split by segment-kind
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE_COLORS = {"forward": "#4C72B0", "backward": "#C44E52",
                "optimizer": "#55A868", "validation": "#DD8452",
                "build": "#7f7f7f", "warmup": "#9467bd",
                "prefill": "#4C72B0", "decode": "#C44E52"}


def load(path) -> dict:
    """Robust load — tolerate a stray trailing object/whitespace (returns the first object)."""
    s = open(path).read()
    return json.JSONDecoder().raw_decode(s.lstrip())[0] if s.strip() else {}


def phase_spans(moves, t_end):
    marks = sorted((t, ph) for t, ph, lbl, *_ in moves if lbl == "<phase>")
    spans = []
    for i, (t, ph) in enumerate(marks):
        t1 = marks[i + 1][0] if i + 1 < len(marks) else t_end
        spans.append((ph, max(0.0, t), t1))
    return spans


def shade(ax, spans, ymax, label=True):
    seen = set()
    for ph, t0, t1 in spans:
        ax.axvspan(t0, t1, color=PHASE_COLORS.get(ph, "#cccccc"), alpha=0.08, zorder=0)
        ax.axvline(t0, color=PHASE_COLORS.get(ph, "#cccccc"), lw=0.6, alpha=0.5, zorder=1)
        if label and ph not in seen:
            ax.text((t0 + t1) / 2, ymax * 0.97, ph, ha="center", va="top",
                    fontsize=8, color=PHASE_COLORS.get(ph, "#666"))
            seen.add(ph)


def _is_cuda(d):
    return str(d.get("device", "")).startswith("cuda") and any(r[2] > 0 for r in d["vram_timeline"][:200])


def _kind(d):
    return "inference" if "infer" in str(d.get("run", "")) else "training"


def memory_flow_fig(d, out: Path, prefix: str):
    ctx = d["baseline"]["cuda_context_mb"]
    vt = d["vram_timeline"]; ct = d["cpu_ram_timeline"]
    t = [r[0] for r in vt]
    alloc_top = [ctx + r[1] for r in vt]; res_top = [ctx + r[2] for r in vt]
    t_end = t[-1] if t else 1.0
    spans = phase_spans(d["moves"], t_end)

    if not _is_cuda(d):
        fig, axc = plt.subplots(1, 1, figsize=(13, 5))
        rss = [r[1] for r in ct]
        axc.fill_between([r[0] for r in ct], 0, rss, color="#55A868", alpha=0.8,
                         label="process RSS (CPU-RAM)")
        if rss:
            axc.set_ylim(min(rss) * 0.97, max(rss) * 1.02); axc.set_xlim(0, t_end)
            shade(axc, spans, max(rss) * 1.02)
        axc.set_ylabel("CPU-RAM RSS (MB)"); axc.set_xlabel(f"time over the {_kind(d)} run (s)")
        axc.set_title(f"Segmented {_kind(d)} — CPU-RAM flow per operation  "
                      f"[{d['preset']}  bs={d['batch_size']} seq={d['seq_len']}  on cpu  (VRAM: N/A)]")
        axc.legend(loc="upper left", fontsize=8); axc.grid(True, axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(out / f"{prefix}_memory_flow.png", dpi=140); plt.close(fig)
        print(f"  wrote {out/(prefix+'_memory_flow.png')} (CPU: single RSS panel)")
        return

    fig, (axv, axc) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    axv.fill_between(t, 0, ctx, color="#bbbbbb", label="CUDA context (non-torch)", zorder=2)
    axv.fill_between(t, ctx, alloc_top, color="#4C72B0", alpha=0.85,
                     label="torch allocated (live tensors)", zorder=3)
    axv.fill_between(t, alloc_top, res_top, color="#9DBBD6", alpha=0.7,
                     label="torch cache (reserved slack)", zorder=2)
    axv.plot(t, res_top, color="#1f3b57", lw=0.8, label="total VRAM (context+reserved)", zorder=4)
    ymaxv = max(res_top) * 1.12 if res_top else 1
    axv.set_ylim(0, ymaxv); axv.set_xlim(0, t_end)
    shade(axv, spans, ymaxv)
    axv.set_ylabel("VRAM (MB)")
    axv.set_title(f"Segmented {_kind(d)} — memory flow per operation  "
                  f"[{d['preset']}  bs={d['batch_size']} seq={d['seq_len']}  on {d['device']}]")
    axv.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    axv.grid(True, axis="y", alpha=0.25)
    rss = [r[1] for r in ct]
    axc.fill_between([r[0] for r in ct], 0, rss, color="#55A868", alpha=0.8,
                     label="process RSS (CPU-RAM)")
    if rss:
        meas = [r[1] for r in ct if 0 <= r[0] <= t_end] or rss
        axc.set_ylim(min(meas) * 0.99, max(meas) * 1.01); axc.set_xlim(0, t_end)
        shade(axc, spans, max(meas) * 1.01)
    axc.set_ylabel("CPU-RAM RSS (MB)"); axc.set_xlabel(f"time over the {_kind(d)} run (s)")
    axc.legend(loc="upper left", fontsize=8, framealpha=0.9); axc.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / f"{prefix}_memory_flow.png", dpi=140); plt.close(fig)
    print(f"  wrote {out/(prefix+'_memory_flow.png')}")


def phase_breakdown_fig(d, out: Path, prefix: str):
    import numpy as np
    phases = list(d["per_phase"].keys())
    ctx = [d["per_phase"][p]["vram"]["context_mb"] for p in phases]
    res = [d["per_phase"][p]["vram"]["alloc_resident_mb"] for p in phases]
    opv = [d["per_phase"][p]["vram"]["alloc_operation_mb"] for p in phases]
    cache = [d["per_phase"][p]["vram"]["torch_cache_mb"] for p in phases]
    rss_floor = [d["per_phase"][p]["cpu_ram"]["rss_floor_mb"] for p in phases]
    rss_op = [d["per_phase"][p]["cpu_ram"]["rss_operation_mb"] for p in phases]
    x = np.arange(len(phases))
    if _is_cuda(d):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        b = np.zeros(len(phases))
        for vals, col, lab in [(ctx, "#bbbbbb", "CUDA context"),
                               (res, "#4C72B0", "allocated: resident (params+grads)"),
                               (opv, "#C44E52", "allocated: operation (transient)"),
                               (cache, "#9DBBD6", "torch cache (reserved slack)")]:
            a1.bar(x, vals, bottom=b, color=col, label=lab); b = b + np.array(vals)
        for xi, tot in zip(x, b):
            a1.text(xi, tot + b.max() * 0.01, f"{tot:.0f}", ha="center", va="bottom", fontsize=8)
        a1.set_xticks(x); a1.set_xticklabels(phases)
        a1.set_ylabel("VRAM (MB)"); a1.set_title("VRAM by category, per phase (not blended)")
        a1.legend(fontsize=8); a1.grid(True, axis="y", alpha=0.25)
    else:
        fig, a2 = plt.subplots(1, 1, figsize=(7, 5))
    a2.bar(x, rss_floor, color="#55A868", label="RSS floor (resident)")
    a2.bar(x, rss_op, bottom=rss_floor, color="#DD8452", label="RSS operation (Δ in phase)")
    a2.set_xticks(x); a2.set_xticklabels(phases)
    a2.set_ylabel("CPU-RAM RSS (MB)"); a2.set_title("CPU-RAM by phase")
    a2.legend(fontsize=8); a2.grid(True, axis="y", alpha=0.25)
    a2.set_ylim(min(rss_floor) * 0.97, (max(np.array(rss_floor) + np.array(rss_op))) * 1.03)
    fig.tight_layout(); fig.savefig(out / f"{prefix}_phase_breakdown.png", dpi=140); plt.close(fig)
    print(f"  wrote {out/(prefix+'_phase_breakdown.png')}")


def _round_windows(moves, t_end):
    marks = sorted((t, ph) for t, ph, lbl, *_ in moves if lbl == "<phase>")
    rounds = []; cur = None
    for i, (t, ph) in enumerate(marks):
        t1 = marks[i + 1][0] if i + 1 < len(marks) else t_end
        if ph == "forward":
            cur = {}; rounds.append(cur)
        if cur is not None and ph in ("forward", "backward", "optimizer", "validation"):
            cur[ph] = (max(0.0, t), t1)
    return rounds


def round_stability_fig(d, out: Path, prefix: str):
    ctx = d["baseline"]["cuda_context_mb"]
    vt = d["vram_timeline"]; ct = d["cpu_ram_timeline"]
    t_end = vt[-1][0] if vt else 1.0
    rounds = _round_windows(d["moves"], t_end)
    if not rounds:
        print(f"  (no training rounds; skipping {prefix}_round_stability.png)"); return
    phases = ["forward", "backward", "optimizer", "validation"]
    peak_v = lambda t0, t1: max([ctx + r[2] for r in vt if t0 <= r[0] < t1] or [float("nan")])
    peak_r = lambda t0, t1: max([r[1] for r in ct if t0 <= r[0] < t1] or [float("nan")])
    xs = list(range(1, len(rounds) + 1))
    vram = {ph: [peak_v(*r[ph]) if ph in r else float("nan") for r in rounds] for ph in phases}
    rss = {ph: [peak_r(*r[ph]) if ph in r else float("nan") for r in rounds] for ph in phases}
    times = {ph: [(r[ph][1] - r[ph][0]) if ph in r else float("nan") for r in rounds] for ph in phases}
    step_total = [sum(t for t in (times[ph][i] for ph in phases) if t == t) for i in range(len(rounds))]
    title = (f"Cross-round stability — drift between training rounds?  "
             f"[{d['preset']}  bs={d['batch_size']} seq={d['seq_len']}  {d['device']}]")
    if _is_cuda(d):
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        mem_axes = [(a1, vram, "peak VRAM (MB)"), (a2, rss, "peak CPU-RAM RSS (MB)")]; atime = a3
    else:
        fig, (a2, a3) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        mem_axes = [(a2, rss, "peak CPU-RAM RSS (MB)")]; atime = a3
    for ax, series, ylab in mem_axes:
        for ph in phases:
            ax.plot(xs, series[ph], "o-", color=PHASE_COLORS[ph], label=ph, lw=1.6, ms=5)
        ax.set_ylabel(ylab)
    for ph in phases:
        atime.plot(xs, times[ph], "o-", color=PHASE_COLORS[ph], label=ph, lw=1.6, ms=5)
    atime.plot(xs, step_total, "s--", color="black", label="step total", lw=1.4, ms=5)
    atime.set_ylabel("phase time (s)"); atime.set_xlabel("training round (step #)")
    mem_axes[0][0].set_title(title)
    for ax in [a for a, _, _ in mem_axes] + [atime]:
        ax.grid(True, alpha=0.25); ax.set_xticks(xs); ax.legend(fontsize=8, ncol=5, loc="best")
    fig.tight_layout(); fig.savefig(out / f"{prefix}_round_stability.png", dpi=140); plt.close(fig)
    print(f"  wrote {out/(prefix+'_round_stability.png')}  ({len(rounds)} rounds)")


def timing_fig(d, out: Path, prefix: str):
    import numpy as np
    from collections import defaultdict
    phases = list(d["per_phase"].keys())
    kinds = ["embedding", "attention", "mlp", "output_head"]
    seg_t = defaultdict(float); pend = None
    for t, ph, lbl, *_ in d["moves"]:
        if lbl.startswith("load|"):
            pend = (t, ph, lbl.split("|")[1])
        elif lbl.startswith("release|") and pend:
            t0, ph0, kind = pend; pend = None
            if ph0 in phases:
                seg_t[(ph0, kind)] += (t - t0)
    phase_total = {p: d["per_phase"][p]["time_ms"] / 1000.0 for p in phases}
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(phases)); bottom = np.zeros(len(phases))
    kcol = {"embedding": "#8c564b", "attention": "#4C72B0", "mlp": "#55A868", "output_head": "#C44E52"}
    for k in kinds:
        vals = [seg_t.get((p, k), 0.0) for p in phases]
        ax.bar(x, vals, bottom=bottom, color=kcol[k], label=f"{k} segments"); bottom = bottom + np.array(vals)
    other = [max(0.0, phase_total[p] - bottom[i]) for i, p in enumerate(phases)]
    ax.bar(x, other, bottom=bottom, color="#cccccc", label="other (norms/adds/CE/grad)")
    for xi, p in zip(x, phases):
        ax.text(xi, phase_total[p] + max(phase_total.values()) * 0.01,
                f"{phase_total[p]:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(phases); ax.set_ylabel("time (s)")
    _avg = d.get("avg_step_time_s") or d.get("avg_token_time_s") or 0.0
    _unit = "step" if "avg_step_time_s" in d else "token"
    ax.set_title(f"Time per phase, by segment-kind  "
                 f"[{d['preset']}  bs={d['batch_size']} seq={d['seq_len']}  {d['device']}  avg {_unit}={_avg:.1f}s]")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / f"{prefix}_timing.png", dpi=140); plt.close(fig)
    print(f"  wrote {out/(prefix+'_timing.png')}")

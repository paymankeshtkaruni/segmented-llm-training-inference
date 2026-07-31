#!/usr/bin/env python
"""Generate the B2 inference-cost figures (large model, forward-only) into figures/.

Reads outputs/{infer_cost_profiler,infer_cost}_{gpu,cpu}/*.json and writes:
  b2_memory_timeline.png   memory over the few steps (GPU VRAM+host; CPU host)
  b2_decomposition.png     framework/context vs model+data (GPU | CPU subplots)
  b2_timing.png            avg step time / throughput, 4 variants (log)

Promoted from temp/test_b2_cost_plots.py after review.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL_MODEL = Path(__file__).resolve().parent.parent
OUT = FULL_MODEL / "outputs"
FIG = FULL_MODEL / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


gpu_prof = load(OUT / "infer_cost_profiler_gpu" / "infer_cost_profiler_metrics.json")
cpu_prof = load(OUT / "infer_cost_profiler_cpu" / "infer_cost_profiler_metrics.json")
gpu_light = load(OUT / "infer_cost_gpu" / "infer_cost_metrics.json")
cpu_light = load(OUT / "infer_cost_cpu" / "infer_cost_metrics.json")

# ---- 1) memory timeline ----
fig, axes = plt.subplots(2, 1, figsize=(8, 7))
ax = axes[0]
v = gpu_prof["memory"]["vram"]
tl = v["nvidia_smi_timeline"]
ax.plot([p[0] for p in tl], [p[1] for p in tl], "-", color="tab:red", lw=2,
        label="process VRAM (nvidia-smi)")
ax.axhline(v["cuda_context_mb"], ls="--", color="gray",
           label=f"CUDA context baseline ({v['cuda_context_mb']:.0f} MB)")
ax.set_ylabel("VRAM (MB)", color="tab:red"); ax.set_xlabel("time (s)")
ax.set_title(f"GPU inference-cost timeline — {gpu_prof['params_million']:.0f}M params, "
             f"bs {gpu_prof['batch_size']}, seq {gpu_prof['seq_len']}, {gpu_prof['n_steps']} steps")
axh = ax.twinx()
htl = gpu_prof["memory"]["host_cpu_ram"]["timeline"]
axh.plot([p[0] for p in htl], [p[1] for p in htl], "-", color="tab:blue", alpha=0.6,
         label="host RAM (RSS)")
axh.set_ylabel("host RAM (MB)", color="tab:blue")
L = ax.get_lines() + axh.get_lines()
ax.legend(L, [x.get_label() for x in L], loc="center right", fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
htl = cpu_prof["memory"]["host_cpu_ram"]["timeline"]
ax.plot([p[0] for p in htl], [p[1] for p in htl], "-", color="tab:cyan", label="host RAM (RSS)")
ax.axhline(cpu_prof["memory"]["host_cpu_ram"]["framework_baseline_mb"], ls="--", color="gray",
           label=f"framework baseline ({cpu_prof['memory']['host_cpu_ram']['framework_baseline_mb']:.0f} MB)")
ax.set_ylabel("host RAM (MB)"); ax.set_xlabel("time (s)")
ax.set_title("CPU inference-cost timeline — host RAM"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "b2_memory_timeline.png", dpi=130)

# ---- 2) decomposition: GPU | CPU subplots ----
gv = gpu_prof["memory"]["vram"]; gh = gpu_prof["memory"]["host_cpu_ram"]
ch = cpu_prof["memory"]["host_cpu_ram"]
fig, (axg, axc) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)


def _stack(ax, labels, base, net, title):
    ax.bar(labels, base, label="framework / CUDA context (before inference)", color="tab:gray")
    ax.bar(labels, net, bottom=base, label="model + data (params + activations)", color="tab:orange")
    for i, (bb, nn) in enumerate(zip(base, net)):
        if nn > 0:
            ax.text(i, bb + nn / 2, f"{nn:.0f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)


_stack(axg, ["VRAM", "host RAM"],
       [gv["cuda_context_mb"], gh["framework_baseline_mb"]],
       [gv["nvidia_smi_net_model_data_mb"], gh["net_model_data_mb"]], "GPU inference cost")
axg.set_ylabel("memory (MB)")
_stack(axc, ["VRAM", "host RAM"],
       [0.0, ch["framework_baseline_mb"]],
       [0.0, ch["net_model_data_mb"]], "CPU inference cost  (VRAM = 0, no GPU)")
axg.legend(fontsize=8, loc="upper left")
fig.suptitle("B2 inference cost — memory decomposition: framework vs model+data")
fig.tight_layout(); fig.savefig(FIG / "b2_decomposition.png", dpi=130)

# ---- 3) timing / throughput (log) ----
names = ["GPU light", "GPU profiler", "CPU light", "CPU profiler"]
ms = [gpu_light["avg_step_time_s"] * 1000, gpu_prof["avg_step_time_s"] * 1000,
      cpu_light["avg_step_time_s"] * 1000, cpu_prof["avg_step_time_s"] * 1000]
thr = [gpu_light["throughput_samples_per_s"], gpu_prof["throughput_samples_per_s"],
       cpu_light["throughput_samples_per_s"], cpu_prof["throughput_samples_per_s"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
bb = a1.bar(names, ms, color="tab:purple"); a1.bar_label(bb, fmt="%.0f")
a1.set_yscale("log"); a1.set_ylabel("avg step time (ms, log)"); a1.set_title("B2 — avg step time")
a1.tick_params(axis="x", rotation=20)
bb = a2.bar(names, thr, color="tab:green"); a2.bar_label(bb, fmt="%.1f")
a2.set_yscale("log"); a2.set_ylabel("samples / s (log)"); a2.set_title("B2 — throughput")
a2.tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(FIG / "b2_timing.png", dpi=130)

print(f"wrote B2 figures to {FIG}/  (b2_memory_timeline, b2_decomposition, b2_timing)")

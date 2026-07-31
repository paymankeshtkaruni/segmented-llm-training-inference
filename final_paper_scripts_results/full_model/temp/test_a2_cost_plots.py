"""Temp (for review): plot A2 large-model cost results into this temp folder.

Reads outputs/{cost_profiler,training_cost}_{gpu,cpu}/*.json and produces:
  a2_memory_timeline.png   continuous memory over the few steps (VRAM + host RAM)
  a2_decomposition.png     framework/context vs model+data (stacked bars)
  a2_timing.png            avg step time / throughput for the 4 variants

Once confirmed, this graduates into scripts/ as the real figure generator.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


gpu_prof = load(OUT / "cost_profiler_gpu" / "cost_profiler_metrics.json")
cpu_prof = load(OUT / "cost_profiler_cpu" / "cost_profiler_metrics.json")
gpu_light = load(OUT / "training_cost_gpu" / "training_cost_metrics.json")
cpu_light = load(OUT / "training_cost_cpu" / "training_cost_metrics.json")

# ---- 1) memory timeline over the few steps ----
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
# GPU: VRAM timeline (+context baseline) and host RSS on twin axis
ax = axes[0]
v = gpu_prof["memory"]["vram"]
tl = v["nvidia_smi_timeline"]
ts, vram = [p[0] for p in tl], [p[1] for p in tl]
ax.plot(ts, vram, "-", color="tab:red", label="process VRAM (nvidia-smi)")
ax.axhline(v["cuda_context_mb"], ls="--", color="gray",
           label=f"CUDA context baseline ({v['cuda_context_mb']:.0f} MB)")
ax.set_ylabel("VRAM (MB)", color="tab:red"); ax.set_xlabel("time (s)")
ax.set_title(f"GPU training-cost timeline — {gpu_prof['params_million']:.0f}M params, "
             f"bs {gpu_prof['batch_size']}, seq {gpu_prof['seq_len']}, {gpu_prof['n_steps']} steps")
axh = ax.twinx()
htl = gpu_prof["memory"]["host_cpu_ram"]["timeline"]
axh.plot([p[0] for p in htl], [p[1] for p in htl], "-", color="tab:blue", alpha=0.6,
         label="host RAM (RSS)")
axh.set_ylabel("host RAM (MB)", color="tab:blue")
l1 = ax.get_lines() + axh.get_lines()
ax.legend(l1, [x.get_label() for x in l1], loc="center right", fontsize=8)
ax.grid(True, alpha=0.3)
# CPU: host RSS timeline
ax = axes[1]
htl = cpu_prof["memory"]["host_cpu_ram"]["timeline"]
ax.plot([p[0] for p in htl], [p[1] for p in htl], "-", color="tab:cyan", label="host RAM (RSS)")
ax.axhline(cpu_prof["memory"]["host_cpu_ram"]["framework_baseline_mb"], ls="--", color="gray",
           label=f"framework baseline ({cpu_prof['memory']['host_cpu_ram']['framework_baseline_mb']:.0f} MB)")
ax.set_ylabel("host RAM (MB)"); ax.set_xlabel("time (s)")
ax.set_title("CPU training-cost timeline — host RAM"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "a2_memory_timeline.png", dpi=130)

# ---- 2) decomposition: framework/context vs model+data ----
fig, ax = plt.subplots(figsize=(7.5, 4.8))
gv = gpu_prof["memory"]["vram"]; gh = gpu_prof["memory"]["host_cpu_ram"]
ch = cpu_prof["memory"]["host_cpu_ram"]
labels = ["GPU VRAM", "GPU host RAM", "CPU host RAM"]
base = [gv["cuda_context_mb"], gh["framework_baseline_mb"], ch["framework_baseline_mb"]]
net = [gv["nvidia_smi_net_model_data_mb"], gh["net_model_data_mb"], ch["net_model_data_mb"]]
b1 = ax.bar(labels, base, label="framework / CUDA context (before training)", color="tab:gray")
b2 = ax.bar(labels, net, bottom=base, label="model + data (params/grads/opt/acts)", color="tab:orange")
for i, (bb, nn) in enumerate(zip(base, net)):
    ax.text(i, bb/2, f"{bb:.0f}", ha="center", va="center", fontsize=8)
    ax.text(i, bb+nn/2, f"{nn:.0f}", ha="center", va="center", fontsize=8, color="white")
ax.set_ylabel("memory (MB)")
ax.set_title("A2 cost — memory decomposition: framework vs model+data")
ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "a2_decomposition.png", dpi=130)

# ---- 3) timing / throughput ----
names = ["GPU light", "GPU profiler", "CPU light", "CPU profiler"]
ms = [gpu_light["avg_step_time_s"]*1000, gpu_prof["avg_step_time_s"]*1000,
      cpu_light["avg_step_time_s"]*1000, cpu_prof["avg_step_time_s"]*1000]
thr = [gpu_light["throughput_samples_per_s"], gpu_prof["throughput_samples_per_s"],
       cpu_light["throughput_samples_per_s"], cpu_prof["throughput_samples_per_s"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
bb = a1.bar(names, ms, color="tab:purple"); a1.bar_label(bb, fmt="%.0f"); a1.set_ylabel("avg step time (ms, log)")
a1.set_yscale("log"); a1.set_title("A2 — avg step time"); a1.tick_params(axis="x", rotation=20)
bb = a2.bar(names, thr, color="tab:green"); a2.bar_label(bb, fmt="%.1f"); a2.set_ylabel("samples / s (log)")
a2.set_yscale("log"); a2.set_title("A2 — throughput"); a2.tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(HERE / "a2_timing.png", dpi=130)

print("A2 large-model summary:")
print(f"  params {gpu_prof['params_million']:.0f}M  bs {gpu_prof['batch_size']}  seq {gpu_prof['seq_len']}")
print(f"  GPU VRAM peak {gv['nvidia_smi_peak_mb']:.0f} MB = context {gv['cuda_context_mb']:.0f} "
      f"+ model+data {gv['nvidia_smi_net_model_data_mb']:.0f}")
print(f"  GPU host {gh['peak_mb']:.0f} = framework {gh['framework_baseline_mb']:.0f} + model+data {gh['net_model_data_mb']:.0f}")
print(f"  CPU host {ch['peak_mb']:.0f} = framework {ch['framework_baseline_mb']:.0f} + model+data {ch['net_model_data_mb']:.0f}")
print("wrote: a2_memory_timeline.png, a2_decomposition.png, a2_timing.png")

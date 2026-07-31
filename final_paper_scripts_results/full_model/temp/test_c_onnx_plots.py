"""Temp (for review): C torch-free ONNX inference-cost figures (large model).

Reads outputs/{onnx_infer_cost_profiler,onnx_infer_cost}_{gpu,cpu}/*.json and writes:
  c_memory_timeline.png   memory over the few steps (GPU VRAM+host; CPU host)
  c_decomposition.png     framework vs model+data (GPU | CPU subplots)
  c_timing.png            avg step time, 4 variants (log)

ONNX has no torch sub-categories; onnxruntime creates context + loads weights at
session, so the GPU 'baseline' (before session) is tiny.
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


gpu_prof = load(OUT / "onnx_infer_cost_profiler_gpu" / "onnx_infer_cost_profiler_metrics.json")
cpu_prof = load(OUT / "onnx_infer_cost_profiler_cpu" / "onnx_infer_cost_profiler_metrics.json")
gpu_light = load(OUT / "onnx_infer_cost_gpu" / "onnx_infer_cost_metrics.json")
cpu_light = load(OUT / "onnx_infer_cost_cpu" / "onnx_infer_cost_metrics.json")

# ---- 1) timeline ----
fig, axes = plt.subplots(2, 1, figsize=(8, 7))
ax = axes[0]
v = gpu_prof["memory"]["vram"]
tl = v["nvidia_smi_timeline"]
ax.plot([p[0] for p in tl], [p[1] for p in tl], "-", color="tab:red", lw=2,
        label="process VRAM (nvidia-smi)")
ax.set_ylabel("VRAM (MB)", color="tab:red"); ax.set_xlabel("time (s)")
ax.set_title(f"C: GPU ONNX inference-cost timeline (torch-free) — "
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
ax.set_title("C: CPU ONNX inference-cost timeline — host RAM"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(HERE / "c_memory_timeline.png", dpi=130)

# ---- 2) decomposition (GPU | CPU) ----
gv = gpu_prof["memory"]["vram"]; gh = gpu_prof["memory"]["host_cpu_ram"]
ch = cpu_prof["memory"]["host_cpu_ram"]
fig, (axg, axc) = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)


def _stack(ax, labels, base, net, title):
    ax.bar(labels, base, label="framework (python + onnxruntime, NO torch)", color="tab:gray")
    ax.bar(labels, net, bottom=base, label="model + data (weights + activations)", color="tab:green")
    for i, (bb, nn) in enumerate(zip(base, net)):
        if nn > 0:
            ax.text(i, bb + nn / 2, f"{nn:.0f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)


_stack(axg, ["VRAM", "host RAM"],
       [gv["before_session_mb"], gh["framework_baseline_mb"]],
       [gv["net_model_data_mb"], gh["net_model_data_mb"]], "GPU ONNX inference cost")
axg.set_ylabel("memory (MB)")
_stack(axc, ["VRAM", "host RAM"],
       [0.0, ch["framework_baseline_mb"]],
       [0.0, ch["net_model_data_mb"]], "CPU ONNX inference cost  (VRAM = 0, no GPU)")
axg.legend(fontsize=8, loc="upper left")
fig.suptitle("C torch-free ONNX inference cost — framework vs model+data")
fig.tight_layout(); fig.savefig(HERE / "c_decomposition.png", dpi=130)

# ---- 3) timing ----
names = ["GPU light", "GPU profiler", "CPU light", "CPU profiler"]
ms = [gpu_light["avg_step_time_s"] * 1000, gpu_prof["avg_step_time_s"] * 1000,
      cpu_light["avg_step_time_s"] * 1000, cpu_prof["avg_step_time_s"] * 1000]
fig, ax = plt.subplots(figsize=(7, 4))
bb = ax.bar(names, ms, color="tab:purple"); ax.bar_label(bb, fmt="%.0f")
ax.set_yscale("log"); ax.set_ylabel("avg step time (ms, log)"); ax.set_title("C ONNX — avg step time")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(HERE / "c_timing.png", dpi=130)

print("C ONNX summary:")
print(f"  GPU VRAM peak {gv['nvidia_smi_peak_mb']:.0f} (weights+ctx {gv['after_session_mb']:.0f}, activations {gv['activations_mb']:.0f})")
print(f"  GPU host {gh['peak_mb']:.0f} = framework {gh['framework_baseline_mb']:.0f} + model+data {gh['net_model_data_mb']:.0f}")
print(f"  CPU host {ch['peak_mb']:.0f} = framework {ch['framework_baseline_mb']:.0f} + model+data {ch['net_model_data_mb']:.0f}")
print("wrote: c_memory_timeline.png, c_decomposition.png, c_timing.png")

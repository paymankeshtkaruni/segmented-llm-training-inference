"""Temp (for review): cost-comparison table — large model, bs 4, seq 512.

Compares: A2 training cost (torch), B2 inference cost (torch), C inference cost
(ONNX, torch-free). Memory (GPU VRAM, GPU host framework, CPU host) + step times.
Writes Markdown + LaTeX.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def vram_peak(m):
    v = m.get("memory", {}).get("vram")
    return v["nvidia_smi_peak_mb"] if v else float("nan")


def host_framework(m):
    h = m.get("memory", {}).get("host_cpu_ram", {})
    return h.get("framework_baseline_mb", float("nan"))


def host_peak(m):
    h = m.get("memory", {}).get("host_cpu_ram", {})
    return h.get("peak_mb", float("nan"))


rows = []
for name, runtime, gpu_p, cpu_p in [
    ("A2 training", "torch", "cost_profiler_gpu", "cost_profiler_cpu"),
    ("B2 inference", "torch", "infer_cost_profiler_gpu", "infer_cost_profiler_cpu"),
    ("C inference", "ONNX (torch-free)", "onnx_infer_cost_profiler_gpu", "onnx_infer_cost_profiler_cpu"),
]:
    pre_g = gpu_p.replace("_gpu", "")
    pre_c = cpu_p.replace("_cpu", "")
    g = load(OUT / gpu_p / f"{pre_g}_metrics.json")
    c = load(OUT / cpu_p / f"{pre_c}_metrics.json")
    if g is None or c is None:
        print(f"skip {name}: metrics missing")
        continue
    rows.append({
        "run": name, "runtime": runtime,
        "gpu_vram": vram_peak(g), "gpu_host_fw": host_framework(g),
        "cpu_host": host_peak(c),
        "gpu_step_ms": g["avg_step_time_s"] * 1000,
        "cpu_step_ms": c["avg_step_time_s"] * 1000,
    })

md = ["### Cost comparison — large model (838M), batch 4, seq 512, n_steps 3\n",
      "Memory is the *peak*; GPU host framework = python+torch(+transformers) or "
      "python+onnxruntime (C, no torch).\n",
      "| Run | Runtime | GPU VRAM peak (MB) | GPU host framework (MB) | CPU host peak (MB) | GPU step | CPU step |",
      "|-----|---------|-------------------:|------------------------:|-------------------:|---------:|---------:|"]
for r in rows:
    md.append(f"| {r['run']} | {r['runtime']} | {r['gpu_vram']:.0f} | {r['gpu_host_fw']:.0f} | "
              f"{r['cpu_host']:.0f} | {r['gpu_step_ms']:.0f} ms | {r['cpu_step_ms']/1000:.1f} s |")
md_text = "\n".join(md) + "\n"

tex = [r"\begin{table}[t]\centering",
       r"\caption{Cost comparison (large model, batch 4, seq 512): training vs inference, "
       r"torch vs torch-free ONNX. Memory = peak.}",
       r"\begin{tabular}{llrrrrr}", r"\hline",
       r"Run & Runtime & GPU VRAM (MB) & GPU host fw (MB) & CPU host (MB) & GPU step & CPU step \\", r"\hline"]
for r in rows:
    tex.append(f"{r['run']} & {r['runtime']} & {r['gpu_vram']:.0f} & {r['gpu_host_fw']:.0f} & "
               f"{r['cpu_host']:.0f} & {r['gpu_step_ms']:.0f} ms & {r['cpu_step_ms']/1000:.1f} s \\\\")
tex += [r"\hline", r"\end{tabular}", r"\end{table}"]

(HERE / "cost_comparison_table.md").write_text(md_text)
(HERE / "cost_comparison_table.tex").write_text("\n".join(tex) + "\n")
print(md_text)
print("wrote cost_comparison_table.md and .tex")

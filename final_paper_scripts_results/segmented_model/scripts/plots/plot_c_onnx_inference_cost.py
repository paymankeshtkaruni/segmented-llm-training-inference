#!/usr/bin/env python
"""C — segmented ONNX TORCH-FREE INFERENCE cost figures (large 838M model), GPU and CPU.
Generates into figures/plots/:
  seg_c_memory_flow_cuda, seg_c_compare_cuda   (ONNX vs B2 torch peak)
  seg_c_memory_flow_cpu,  seg_c_compare_cpu
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG = Path(__file__).resolve().parents[2]
O = SEG / "outputs"; FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)


def memory_flow(d, tag):
    tl = d["timeline"]; t = [r[0] for r in tl]; vram = [r[1] for r in tl]; rss = [r[2] for r in tl]
    dev = d["provider"]; is_cuda = dev == "cuda"
    n = 2 if is_cuda else 1
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
    axes = axes if n == 2 else [axes]
    if is_cuda:
        ax = axes[0]
        ax.fill_between(t, 0, vram, color="#4C72B0", alpha=0.8, label="process VRAM (nvidia-smi)")
        ax.axhline(d["after_session"]["vram_mb"], color="#888", ls="--", lw=1,
                   label=f"after 1st session ({d['after_session']['vram_mb']:.0f})")
        ax.axhline(d["peak"]["vram_mb"], color="#C44E52", ls=":", lw=1,
                   label=f"peak ({d['peak']['vram_mb']:.0f})")
        ax.set_ylabel("VRAM (MB)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_title(f"Segmented ONNX inference (torch-free) — memory flow  "
                     f"[{tag}  {dev}  {d['unique_sessions']} sigs, one CUDA session at a time]")
    axc = axes[-1]
    axc.fill_between(t, 0 if not is_cuda else min(rss) * 0.9, rss, color="#55A868", alpha=0.8,
                     label="process RSS (CPU-RAM)")
    axc.set_ylabel("CPU-RAM RSS (MB)"); axc.set_xlabel("time over the inference run (s)")
    axc.legend(fontsize=8); axc.grid(True, alpha=0.25)
    if not is_cuda:
        axc.set_title(f"Segmented ONNX inference (torch-free) — CPU-RAM flow  [{tag}  cpu  (VRAM: N/A)]")
    fig.tight_layout(); fig.savefig(FIG / f"seg_c_memory_flow_{dev}.png", dpi=140); plt.close(fig)
    print(f"  wrote {FIG/f'seg_c_memory_flow_{dev}.png'}")


def compare(d, b2_path, tag):
    dev = d["provider"]
    rows = [("C ONNX\n(torch-free)", d["peak"]["vram_mb"], d["peak"]["rss_mb"], "#4C72B0")]
    if b2_path.exists():
        b2 = json.load(open(b2_path))
        rows.append(("B2 torch", b2["overall"].get("vram_hw_total_peak_mb", 0.0),
                     b2["overall"].get("rss_peak_sampled_mb", 0.0), "#DD8452"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5)); x = np.arange(len(rows))
    a1.bar(x, [r[1] for r in rows], color=[r[3] for r in rows])
    for xi, r in zip(x, rows): a1.text(xi, r[1], f"{r[1]:.0f}", ha="center", va="bottom")
    a1.set_xticks(x); a1.set_xticklabels([r[0] for r in rows]); a1.set_ylabel("peak VRAM (MB)")
    a1.set_title(f"Peak VRAM — {dev}"); a1.grid(True, axis="y", alpha=0.25)
    a2.bar(x, [r[2] for r in rows], color=[r[3] for r in rows])
    for xi, r in zip(x, rows): a2.text(xi, r[2], f"{r[2]:.0f}", ha="center", va="bottom")
    a2.set_xticks(x); a2.set_xticklabels([r[0] for r in rows]); a2.set_ylabel("peak RSS (MB)")
    a2.set_title(f"Peak CPU-RAM — {dev}"); a2.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Segmented inference cost: ONNX torch-free vs torch  [{tag}  {dev}]")
    fig.tight_layout(); fig.savefig(FIG / f"seg_c_compare_{dev}.png", dpi=140); plt.close(fig)
    print(f"  wrote {FIG/f'seg_c_compare_{dev}.png'}")


for dev, cpath, b2path in [("cuda", O / "seg_c_gpu/seg_c_metrics.json", O / "seg_b2_gpu/seg_b2_metrics.json"),
                           ("cpu",  O / "seg_c_cpu/seg_c_metrics.json", O / "seg_b2_cpu/seg_b2_metrics.json")]:
    if not cpath.exists():
        print(f"MISSING {cpath}"); continue
    d = json.load(open(cpath)); tag = Path(d["onnx_dir"]).name
    print(f"[C] {cpath} ({dev})")
    memory_flow(d, tag)
    compare(d, b2path, tag)

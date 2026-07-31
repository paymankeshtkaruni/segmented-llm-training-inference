#!/usr/bin/env python
"""D — segmented TRAINING cost at finer granularity 16x4x4x16 (large 838M model), GPU and CPU.
Generates into figures/plots/:
  GPU: seg_d16_memory_flow, seg_d16_phase_breakdown, seg_d16_round_stability, seg_d16_timing
  CPU: seg_d16_cpu_memory_flow, seg_d16_cpu_phase_breakdown, seg_d16_cpu_round_stability, seg_d16_cpu_timing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cost_plot_lib import load, memory_flow_fig, phase_breakdown_fig, round_stability_fig, timing_fig

SEG = Path(__file__).resolve().parents[2]
O = SEG / "outputs"; FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)

RUNS = [("seg_d16",     O / "seg_cost_gpu_16x4x4x16/seg_cost_metrics.json"),
        ("seg_d16_cpu", O / "seg_cost_cpu_16x4x4x16/seg_cost_metrics.json")]

for prefix, mpath in RUNS:
    if not mpath.exists():
        print(f"MISSING {mpath}"); continue
    print(f"[D] {mpath} -> {prefix}_*")
    d = load(mpath)
    memory_flow_fig(d, FIG, prefix)
    phase_breakdown_fig(d, FIG, prefix)
    round_stability_fig(d, FIG, prefix)
    timing_fig(d, FIG, prefix)

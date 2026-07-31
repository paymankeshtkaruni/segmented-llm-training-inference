#!/usr/bin/env python
"""A2 — segmented TRAINING cost figures (large 838M model), GPU and CPU.
Generates into figures/plots/:
  GPU: seg_a2_memory_flow, seg_a2_phase_breakdown, seg_a2_round_stability, seg_a2_timing
  CPU: seg_a2_cpu_memory_flow, seg_a2_cpu_phase_breakdown, seg_a2_cpu_round_stability, seg_a2_cpu_timing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cost_plot_lib import load, memory_flow_fig, phase_breakdown_fig, round_stability_fig, timing_fig

SEG = Path(__file__).resolve().parents[2]
O = SEG / "outputs"; FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)

RUNS = [("seg_a2",     O / "seg_cost_gpu/seg_cost_metrics.json"),
        ("seg_a2_cpu", O / "seg_cost_cpu/seg_cost_metrics.json")]

for prefix, mpath in RUNS:
    if not mpath.exists():
        print(f"MISSING {mpath}"); continue
    print(f"[A2] {mpath} -> {prefix}_*")
    d = load(mpath)
    memory_flow_fig(d, FIG, prefix)
    phase_breakdown_fig(d, FIG, prefix)
    round_stability_fig(d, FIG, prefix)
    timing_fig(d, FIG, prefix)

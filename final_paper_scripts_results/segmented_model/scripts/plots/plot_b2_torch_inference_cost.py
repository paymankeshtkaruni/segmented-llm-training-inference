#!/usr/bin/env python
"""B2 — segmented TORCH INFERENCE cost figures (large 838M model), GPU and CPU.
Generates into figures/plots/:
  GPU: seg_b2_memory_flow, seg_b2_phase_breakdown, seg_b2_timing
  CPU: seg_b2_cpu_memory_flow, seg_b2_cpu_phase_breakdown, seg_b2_cpu_timing
(No round_stability — inference has prefill/decode phases, not training rounds.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cost_plot_lib import load, memory_flow_fig, phase_breakdown_fig, timing_fig

SEG = Path(__file__).resolve().parents[2]
O = SEG / "outputs"; FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)

RUNS = [("seg_b2",     O / "seg_b2_gpu/seg_b2_metrics.json"),
        ("seg_b2_cpu", O / "seg_b2_cpu/seg_b2_metrics.json")]

for prefix, mpath in RUNS:
    if not mpath.exists():
        print(f"MISSING {mpath}"); continue
    print(f"[B2] {mpath} -> {prefix}_*")
    d = load(mpath)
    memory_flow_fig(d, FIG, prefix)
    phase_breakdown_fig(d, FIG, prefix)
    timing_fig(d, FIG, prefix)

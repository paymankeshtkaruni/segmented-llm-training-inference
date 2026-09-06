#!/usr/bin/env python
"""exp9 bridge analysis: the cost of the per-step validation pass, per mode.

Design (user-specified): exp9 re-ran every exp8 training configuration for
5 steps under the EXACT cost protocol (which times a validation pass inside
every step). Then, per configuration:

    validation cost per step  =  avg_step(exp9)  -  avg_step(exp8, first 5)

cross-checked against the cost protocol's own per-phase validation timing
recorded inside the exp9 metrics (steady-state step). The two estimates
agreeing validates the bridge; the residual after subtracting validation is
the remaining protocol/node difference.

Writes results/exp9_validation_bridge/bridge_summary.json.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
E9 = HERE / "results" / "exp9_validation_bridge"
E8 = HERE / "results" / "exp8_sustained"

JOBS = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"]


def main():
    rows = {}
    for g in JOBS:
        met = list((E9 / g).glob("*_met.json"))
        e8dir = next((d for d in E8.iterdir() if d.name.startswith(g + "_")), None)
        if not met or e8dir is None:
            rows[g] = {"error": "missing data"}
            continue
        d9 = json.load(open(met[0]))
        steps9 = None
        def find(dd):
            nonlocal steps9
            if isinstance(dd, dict):
                for k, v in dd.items():
                    if k == "avg_step_time_s":
                        steps9 = v
                    else:
                        find(v)
        find(d9)
        val_phase_s = ((d9.get("per_phase") or {}).get("validation") or {}).get("time_ms")
        val_phase_s = val_phase_s / 1000 if val_phase_s else None
        d8 = json.load(open(next((E8 / e8dir.name).glob("*_long.json"))))
        first5 = [r["s"] for r in d8["per_step"][:5]]
        avg8_5 = statistics.mean(first5)
        avg8_all = d8["avg_step_s"]
        bridge = round(steps9 - avg8_5, 2) if steps9 else None
        rows[g] = {
            "exp9_avg_step_s_with_validation": round(steps9, 3) if steps9 else None,
            "exp9_validation_phase_s_direct": round(val_phase_s, 3) if val_phase_s else None,
            "exp8_first5_avg_step_s": round(avg8_5, 3),
            "exp8_full_avg_step_s": avg8_all,
            "validation_cost_per_step_s_bridge": bridge,
            "bridge_vs_direct_gap_s": (round(bridge - val_phase_s, 2)
                                       if bridge is not None and val_phase_s else None),
        }
    out = {"run": "exp9_validation_bridge_summary",
           "method": "avg_step(exp9, cost protocol incl. validation, 5 steps) minus "
                     "avg_step(exp8, first 5 steps, no validation); cross-checked "
                     "against exp9's directly timed validation phase",
           "jobs": rows}
    dst = E9 / "bridge_summary.json"
    json.dump(out, open(dst, "w"), indent=1)
    for g, r in rows.items():
        print(g, json.dumps(r))
    print("->", dst)


if __name__ == "__main__":
    main()

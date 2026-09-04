#!/usr/bin/env python
"""Per-step peak-memory report for the sustained 6.9B resident run (exp8).

Reads the MemFlow trace in results/exp8_sustained/rep1/T2sus_met.json, splits
the timeline at each training step's forward-phase mark, and reports the peak
reserved device memory of every step. The claim under test: the peak is FLAT
across 50 consecutive steps (no allocator creep), so "trains on a 40 GB card"
holds for sustained training, not just the 2-step measurement cell.

Writes results/exp8_sustained/sustained_summary.json (small, committed).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results" / "exp8_sustained"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "rep1" / "T2sus_met.json"
    if not src.exists() and src.with_suffix(src.suffix + ".gz").exists():
        src = src.with_suffix(src.suffix + ".gz")
    opener = gzip.open if src.suffix == ".gz" else open
    d = json.load(opener(src, "rt"))
    tl = d["vram_timeline"]                     # [t, alloc_mb, reserved_mb]
    marks = [m[0] for m in d["moves"]
             if m[2] == "<phase>" and m[1] == "forward"]
    if not marks:
        raise SystemExit("no forward phase marks in trace")
    bounds = marks + [tl[-1][0] + 1]
    peaks = []
    j = 0
    for k in range(len(marks)):
        lo, hi = bounds[k], bounds[k + 1]
        peak = 0.0
        while j < len(tl) and tl[j][0] < hi:
            if tl[j][0] >= lo:
                peak = max(peak, tl[j][2])
            j += 1
        peaks.append(round(peak, 1))
    ctx = 490.0                                  # CUDA context on the pinned node class
    out = {
        "run": "exp8_sustained_6p9b_resident",
        "source": str(src.name),
        "n_steps": len(peaks),
        "per_step_reserved_peak_mb": peaks,
        "first_step_mb": peaks[0], "last_step_mb": peaks[-1],
        "max_mb": max(peaks), "min_after_step1_mb": min(peaks[1:]) if len(peaks) > 1 else None,
        "drift_last_minus_first_mb": round(peaks[-1] - peaks[0], 1),
        "note": "reserved-memory peaks per training step; add %.0f MB CUDA "
                "context for the no-miss total" % ctx,
    }
    dst = R / "sustained_summary.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "per_step_reserved_peak_mb"}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()

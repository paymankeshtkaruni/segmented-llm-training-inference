#!/usr/bin/env python
"""exp8 aggregator: long-horizon training and serving stability.

exp8 (redefined 2026-09-05) is the long-horizon experiment family: training
runs of 128-2,048 consecutive steps and serving runs of 500-2,000 requests
at the cost scale, each recording per-step/per-request wall time and peak
memory (runners: exp8_long_run.py, exp8_long_serve.py; jobs:
slurm/exp8_g*.sbatch, slurm/exp8_i*.sbatch). The earlier 30-step run is
kept as the sizing pilot under results/exp8_sustained/pilot_30step/.

This script aggregates every completed job under results/exp8_sustained/
into one committed summary: results/exp8_sustained/sustained_summary.json.

Units: the runners record peak memory in decimal megabytes (bytes/1e6). The
four-phase cost protocol (seg_cost_lib.py) reports mebibytes (bytes/2**20),
and the paper prints every memory figure on that basis. This summary
therefore converts every peak to MiB (factor 1e6/2**20 = 0.95367) so the
long-horizon tables share the unit of every other table.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIB_PER_MB = 1e6 / 2**20          # decimal MB -> MiB
MEM_KEYS = ("peak_mb_first", "peak_mb_max")
R = HERE / "results" / "exp8_sustained"

KEEP = ["n_steps_target", "n_steps_done", "n_requests_target", "n_requests_done",
        "complete", "wall_total_s", "avg_step_s", "median_step_s", "step_s_p5_p95",
        "avg_request_s", "avg_per_token_s", "request_s_p5_p95",
        "peak_mb_first", "peak_mb_max", "peak_mb_max_at_step",
        "peak_mb_max_at_request", "peak_mb_band",
        "preset", "device", "engine", "tech_code", "update_style", "batch",
        "seq_len", "prompt_len", "gen_tokens", "memory_note", "protocol"]


def main():
    jobs = {}
    for f in sorted(R.glob("*/*_long.json")) + sorted(R.glob("*/*_serve.json")):
        if "pilot_30step" in f.parts:
            continue
        d = json.load(open(f))
        row = {k: d[k] for k in KEEP if k in d}
        for k in MEM_KEYS:
            if row.get(k) is not None:
                row[k] = round(row[k] * MIB_PER_MB, 1)
        if row.get("peak_mb_band"):
            row["peak_mb_band"] = [round(v * MIB_PER_MB, 1) for v in row["peak_mb_band"]]
        # stationarity: how far does the last decile's peak sit from the first's?
        series = d.get("per_step") or d.get("per_request") or []
        peaks = [r["peak_mb"] * MIB_PER_MB for r in series if r.get("peak_mb") is not None]
        if len(peaks) >= 20:
            n = max(1, len(peaks) // 10)
            first_decile = max(peaks[:n])
            last_decile = max(peaks[-n:])
            row["peak_mb_first_decile_max"] = round(first_decile, 1)
            row["peak_mb_last_decile_max"] = round(last_decile, 1)
            row["peak_mb_last_minus_first_decile"] = round(last_decile - first_decile, 1)
        # stationarity of time: median step (or request) time of the last decile
        # against the first decile, as a percentage of the first
        times = [r["s"] for r in series if r.get("s") is not None]
        if len(times) >= 20:
            n = max(1, len(times) // 10)
            first_med = statistics.median(times[:n])
            last_med = statistics.median(times[-n:])
            row["time_s_first_decile_median"] = round(first_med, 3)
            row["time_s_last_decile_median"] = round(last_med, 3)
            row["time_last_vs_first_decile_pct"] = round((last_med / first_med - 1) * 100, 1)
        jobs[f.parent.name] = row
    out = {"run": "exp8_long_horizon_summary",
           "note": "per-job long-horizon stability; pilot_30step/ holds the "
                   "30-step sizing pilot (superseded)",
           "memory_unit": "MiB (raw per-step rows are decimal MB; converted here, "
                          "factor 1e6/2**20); GPU values are reserved peaks without "
                          "the CUDA context, which table_enrichment_report.py adds",
           "jobs": jobs}
    dst = R / "sustained_summary.json"
    json.dump(out, open(dst, "w"), indent=1)
    for name, row in jobs.items():
        done = row.get("n_steps_done") or row.get("n_requests_done")
        print(f"{name:24s} done={done} avg_step={row.get('avg_step_s') or row.get('avg_request_s')} "
              f"peak_band={row.get('peak_mb_band')} drift_decile={row.get('peak_mb_last_minus_first_decile')} "
              f"time_last_vs_first_decile={row.get('time_last_vs_first_decile_pct')}%")
    print("->", dst)


if __name__ == "__main__":
    main()

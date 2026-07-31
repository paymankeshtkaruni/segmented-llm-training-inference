#!/usr/bin/env python
"""Aggregate every outputs/*/**_metrics.json into one summary JSON.

Writes jsons/full_model_summary.json keyed by run directory. Lets the table/figure
scripts (and the paper) read all results from one place. Torch-free.
"""
import json
from pathlib import Path

FULL_MODEL = Path(__file__).resolve().parent.parent
OUT = FULL_MODEL / "outputs"
JSONS = FULL_MODEL / "jsons"
JSONS.mkdir(parents=True, exist_ok=True)

summary = {}
for metrics_path in sorted(OUT.glob("*/*_metrics.json")):
    run_dir = metrics_path.parent.name
    if run_dir.startswith("smoke") or run_dir.startswith("verify"):
        continue
    try:
        summary[run_dir] = json.load(open(metrics_path))
    except Exception as e:
        summary[run_dir] = {"error": str(e)}

out = JSONS / "full_model_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"aggregated {len(summary)} runs -> {out}")
for k in sorted(summary):
    print("  -", k)

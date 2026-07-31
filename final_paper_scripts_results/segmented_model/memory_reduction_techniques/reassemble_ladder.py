#!/usr/bin/env python
"""
Rebuild ladder_{train,infer}.json for each category from its per-rung *_metrics.json.

Uses the correct PER-RUNG memory:
  * GPU -> vram_hw_total_peak_mb  (no-miss high-water, reset per phase -> per-rung)
  * CPU -> rss_peak_sampled_mb    (per-rung sampled peak; ru_maxrss is process-LIFETIME
                                   monotonic and wrong when all rungs share one process)

Lets us correct an already-finished run without re-running it (the per-rung JSONs already
hold both fields). Run:  python reassemble_ladder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))
from techniques import cumulative, TRAIN_LADDER, INFER_LADDER   # noqa: E402

CATS = [("gpu_train", "train"), ("cpu_train", "train"),
        ("gpu_inference", "infer"), ("cpu_inference", "infer")]


def rebuild(subdir: str, kind: str):
    d = HERE / "results" / subdir
    if not d.exists():
        return
    ladder = TRAIN_LADDER if kind == "train" else INFER_LADDER
    rows = []
    for name, desc, tech in cumulative(ladder):
        p = d / f"{name}_metrics.json"
        if not p.exists():
            print(f"  {subdir}: missing {p.name} (skipped)"); continue
        m = json.load(open(p)); o = m["overall"]
        row = {"rung": name, "adds": desc, "tech_code": tech.code(),
               "vram_peak_mb": o.get("vram_hw_total_peak_mb", 0.0),
               "rss_peak_mb":  o.get("rss_peak_sampled_mb", 0.0)}
        if kind == "train":
            pp = m.get("per_phase", {})
            row["step_time_s"] = round(m["avg_step_time_s"], 2)
            row["fwd_ms"] = pp.get("forward", {}).get("time_ms")
            row["bwd_ms"] = pp.get("backward", {}).get("time_ms")
            row["opt_ms"] = pp.get("optimizer", {}).get("time_ms")
        else:
            row["per_token_s"] = m["per_token_s"]
        rows.append(row)
    out = d / f"ladder_{kind}.json"
    json.dump({"run": f"{kind}_incremental_ablation", "subdir": subdir,
               "memory_note": "GPU headline = vram_peak_mb (no-miss high-water); "
                              "CPU headline = rss_peak_mb (PER-RUNG sampled, not lifetime ru_maxrss)",
               "ladder": rows}, open(out, "w"), indent=2)
    print(f"  rebuilt {out.relative_to(HERE)}  ({len(rows)} rungs)")


if __name__ == "__main__":
    print("reassembling ladders from per-rung metrics ...")
    for sub, kind in CATS:
        rebuild(sub, kind)

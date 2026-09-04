#!/usr/bin/env python
"""Rebuild the four committed experiment summaries from the raw rep dirs.

    results/exp3_grid/grid_summary.json    (12 training modes x GPU/CPU)
    results/exp4_infer/infer_summary.json  (torch I-modes, ONNX O-modes,
                                            full-decode anchors)
    results/exp5_scale/scale_summary.json  (scale cells incl. OOM records)
    results/exp6_fast/exp6_summary.json    (fastest-mode + granularity cells)

Every value is extracted from the per-rep metrics JSONs written by
measure_one.py / scale_cost.py / full_decode_cost.py / onnx_resident_cost.py,
so reviewers can regenerate the summaries (and the paper tables/figures built
on them) from the raw measurements with this one script. Medians over reps.

Field mapping (see measure_one.py / seg_cost_lib.py):
    vram_mb      <- overall.vram_hw_total_peak_mb   (CUDA context + peak reserved)
    rss_mb       <- overall.rss_peak_sampled_mb for CPU cells,
                    overall.rss_hw_peak_mb for GPU cells (lifetime peak)
    step_s       <- avg_step_time_s
    per_token_s  <- per_token_s
    oom          <- top-level "oom": true records written on allocation failure
Full-decode anchors record vram_hw_reserved_peak_mb (no context term); the
CUDA context on the pinned node class is 490 MB (see any infer cell's
overall.cuda_context_mb), which the paper adds when quoting a total.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results" if (HERE / "results").exists() else HERE  # run from pkg or results/


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def find_key(d, key):
    hits = []
    def go(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == key:
                    hits.append(v)
                else:
                    go(v)
    go(d)
    return hits[0] if hits else None


def rep_record(rep_dir: Path, cpu: bool):
    js = sorted(rep_dir.glob("*_met.json")) or sorted(rep_dir.glob("*_metrics.json"))
    if not js:
        return None
    d = json.load(open(js[0]))
    if d.get("run") == "segmented_onnx_infer_cost":   # weights-as-inputs runtime
        pk = d.get("peak", {})
        return {"vram_mb": pk.get("vram_mb"), "rss_mb": pk.get("rss_mb"),
                "step_s": None, "per_token_s": d.get("avg_token_s"),
                "oom": False, "phases_ms": None}
    if d.get("oom"):
        return {"vram_mb": find_key(d, "vram_hw_total_peak_mb") or find_key(d, "vram_peak_mb"),
                "rss_mb": None, "step_s": None, "per_token_s": None, "oom": True}
    o = d.get("overall", {})
    rss = o.get("rss_peak_sampled_mb") if cpu else o.get("rss_hw_peak_mb")
    if rss is None:
        rss = find_key(d, "rss_hw_peak_mb") or d.get("rss_peak_mb")
    vram = o.get("vram_hw_total_peak_mb")
    if vram is None:
        vram = find_key(d, "vram_hw_total_peak_mb")
    if vram is None:  # ONNX runtimes (sampler keys)
        vram = d.get("vram_peak_mb")
    if vram is None:  # full-decode anchors: reserved only (context not measured)
        vram = find_key(d, "vram_hw_reserved_peak_mb")
    return {"vram_mb": vram, "rss_mb": rss,
            "step_s": find_key(d, "avg_step_time_s"),
            "per_token_s": d.get("per_token_s"),
            "oom": False,
            "phases_ms": d.get("per_phase")}


def scan_cells(root: Path):
    cells = {}
    for rep_dir in sorted(root.iterdir()):
        m = re.match(r"^(.+)_rep(\d+)$", rep_dir.name)
        if not m or not rep_dir.is_dir():
            continue
        rec = rep_record(rep_dir, cpu="cpu" in m.group(1))
        if rec is not None:
            rec.pop("phases_ms", None)
            cells.setdefault(m.group(1), []).append(rec)
    return cells


def build_grid():
    cells = {}
    root = R / "exp3_grid"
    for rep_dir in sorted(root.iterdir()):
        m = re.match(r"^(gpu|cpu)_(T\d+)_rep(\d+)$", rep_dir.name)
        if not m:
            continue
        dev, mode = m.group(1), m.group(2)
        rec = rep_record(rep_dir, cpu=(dev == "cpu"))
        if rec is None:
            continue
        raw = rec.pop("phases_ms", None) or {}
        ph = {k: v.get("time_ms") for k, v in raw.items() if isinstance(v, dict)}
        cells.setdefault(mode, {}).setdefault(dev, []).append(
            {"vram": rec["vram_mb"], "rss": rec["rss_mb"],
             "step_s": rec["step_s"], "phases_ms": ph})
    modes = {}
    for mode in sorted(cells, key=lambda t: int(t[1:])):
        modes[mode] = {}
        for dev, reps in cells[mode].items():
            modes[mode][dev] = {
                "vram_mb": med([r["vram"] for r in reps]) or 0,
                "rss_mb": med([r["rss"] for r in reps]),
                "step_s": med([r["step_s"] for r in reps]),
                "opt_phase_s": med([(r["phases_ms"].get("optimizer") or 0) / 1000
                                    for r in reps]),
                "n": len(reps), "reps": reps,
            }
    out = {"run": "exp3_12mode_grid",
           "protocol": "batch4 seq512 3steps, medians of 3 reps, pinned A100-40 / corrected CPU",
           "modes": modes}
    json.dump(out, open(root / "grid_summary.json", "w"), indent=1)
    print("wrote", root / "grid_summary.json")


def build_infer():
    root = R / "exp4_infer"
    torch_cells, onnx_cells, anchor = {}, {}, {}
    for rep_dir in sorted(root.iterdir()):
        m = re.match(r"^(?:onnx_)?(gpu|cpu)_(I\d+|O\d+_\w+)_rep(\d+)$", rep_dir.name)
        fa = re.match(r"^full_anchor_(cuda|cpu)_rep(\d+)$", rep_dir.name)
        if m:
            dev, mode = m.group(1), m.group(2)
            rec = rep_record(rep_dir, cpu=(dev == "cpu"))
            if rec is None:
                continue
            rec.pop("phases_ms", None)
            tgt = torch_cells if mode.startswith("I") else onnx_cells
            tgt.setdefault(mode, {}).setdefault(dev, []).append(
                {"vram": rec["vram_mb"], "rss": rec["rss_mb"],
                 "per_token_s": rec["per_token_s"]})
        elif fa:
            dev = "gpu" if fa.group(1) == "cuda" else "cpu"
            rec = rep_record(rep_dir, cpu=(dev == "cpu"))
            if rec is None:
                continue
            anchor.setdefault(dev, []).append(
                {"vram_reserved": rec["vram_mb"], "rss": rec["rss_mb"],
                 "per_token_s": rec["per_token_s"]})

    def agg(cells, keys=("vram", "rss", "per_token_s")):
        out = {}
        for mode, devs in sorted(cells.items()):
            out[mode] = {}
            for dev, reps in devs.items():
                row = {("%s_mb" % k if k != "per_token_s" else k):
                       med([r[k] for r in reps]) for k in keys}
                row["reps"] = reps
                out[mode][dev] = row
        return out

    out = {"run": "exp4_inference_modes",
           "protocol": "prompt256 decode32 batch1, medians of 3",
           "torch": agg(torch_cells), "onnx": agg(onnx_cells),
           "full_anchor": {dev: {"vram_reserved_mb": med([r["vram_reserved"] for r in reps]),
                                 "rss_mb": med([r["rss"] for r in reps]),
                                 "per_token_s": med([r["per_token_s"] for r in reps]),
                                 "reps": reps}
                           for dev, reps in anchor.items()},
           "note_full_anchor": "vram_reserved excludes the 490 MB CUDA context"}
    json.dump(out, open(root / "infer_summary.json", "w"), indent=1)
    print("wrote", root / "infer_summary.json")


def build_scale(root_name, run_name, note):
    root = R / root_name
    if not root.exists():
        return
    cells = scan_cells(root)
    out = {"run": run_name, "note": note, "cells": cells}
    name = {"exp5_scale": "scale_summary.json",
            "exp6_fast": "exp6_summary.json"}.get(
        root_name, root_name.split("_", 1)[1] + "_summary.json")
    json.dump(out, open(root / name, "w"), indent=1)
    print("wrote", root / name)


if __name__ == "__main__":
    build_grid()
    build_infer()
    build_scale("exp5_scale", "exp5_scale",
                "medians of reps; OOM cells recorded; n-steps 2 at 7B")
    build_scale("exp6_fast", "exp6_fastest_modes",
                "fastest-mode (T7/T8, I1) at scale + granularity cells")
    build_scale("exp7_sensitivity", "exp7_batch_seq_sensitivity",
                "streamed deferred (dropout on) 0.84B GPU, batch x seq sweep")

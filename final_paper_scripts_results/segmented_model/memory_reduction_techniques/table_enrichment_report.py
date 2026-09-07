#!/usr/bin/env python3
"""Numbers for the enriched paper tables (12-mode grid and long-horizon
training): full-model anchors, per-mode ratios vs. the full model, and
long-run deltas vs. the short cost protocol.

Inputs (all committed results):
  results/exp1_compare/full_{drop,nodrop}_rep*/full_train_metrics.json
      -> GPU full-model anchors (vram_hw_total_peak_mb, avg_step_time_s)
  results/exp2_anchor/cpu_full_rep*/full_train_metrics.json
      -> CPU full-model anchor (rss_peak_sampled_mb, avg_step_time_s)
  results/exp3_grid/grid_summary.json          -> 12-mode protocol cells
  results/exp5_scale/scale_summary.json        -> 6.9B protocol cells
  results/exp8_sustained/sustained_summary.json-> long-run peaks and step times
  results/exp9_validation_bridge/bridge_summary.json -> direct validation cost

Output: results/table_enrichment.json

Ratio conventions: GPU rows divide by the dropout-matched GPU full anchor;
all CPU rows divide by the single (dropout) CPU full anchor - stated in the
paper caption. Long-run GPU peaks add the 490 MB CUDA context so both sides
of the delta use the same basis as the grid (hw-total).

Delta conventions: delta_time_pct compares (long-run step + directly timed
per-step validation) against the protocol step, i.e. protocol-fair after the
exp9 bridge; the residual contains the tracer/trim overhead and node
co-tenancy priced in the paper. delta_peak_pct compares peaks on the shared
basis; the grid side additionally contains warmup/validation allocations.
"""
import json
import statistics as st
from glob import glob
from pathlib import Path

RES = Path("results")
CUDA_CTX_MB = 490.0


def med(xs):
    return round(st.median(xs), 3)


def full_anchor(pattern, mem_key):
    reps = []
    for f in sorted(glob(str(pattern))):
        d = json.load(open(f))
        reps.append((d["overall"][mem_key], d["avg_step_time_s"]))
    assert reps, f"no reps for {pattern}"
    return {"peak_mb": med([r[0] for r in reps]),
            "step_s": med([r[1] for r in reps]),
            "n_reps": len(reps)}


def main():
    out = {"run": "table_enrichment",
           "note": __doc__.strip().splitlines()[0]}

    # ---- 1. full-model anchors -------------------------------------------
    anchors = {
        "gpu_dropout": full_anchor(RES / "exp1_compare/full_drop_rep*/full_train_metrics.json",
                                   "vram_hw_total_peak_mb"),
        "gpu_nodropout": full_anchor(RES / "exp1_compare/full_nodrop_rep*/full_train_metrics.json",
                                     "vram_hw_total_peak_mb"),
        "cpu_dropout": full_anchor(RES / "exp2_anchor/cpu_full_rep*/full_train_metrics.json",
                                   "rss_peak_sampled_mb"),
    }
    out["full_anchors"] = anchors

    # ---- 2. grid ratios vs. full model -----------------------------------
    grid = json.load(open(RES / "exp3_grid/grid_summary.json"))["modes"]
    DROPOUT_MODES = {"T1", "T2", "T3", "T7", "T8", "T9"}
    ratios = {}
    for t, cells in grid.items():
        ga = anchors["gpu_dropout" if t in DROPOUT_MODES else "gpu_nodropout"]
        ca = anchors["cpu_dropout"]
        ratios[t] = {
            "gpu_mem_ratio": round(cells["gpu"]["vram_mb"] / ga["peak_mb"], 4),
            "gpu_time_ratio": round(cells["gpu"]["step_s"] / ga["step_s"], 2),
            "cpu_mem_ratio": round(cells["cpu"]["rss_mb"] / ca["peak_mb"], 4),
            "cpu_time_ratio": round(cells["cpu"]["step_s"] / ca["step_s"], 2),
        }
    out["grid_ratios_vs_full"] = ratios

    # ---- 3. long-run deltas vs. the short cost protocol ------------------
    scale = json.load(open(RES / "exp5_scale/scale_summary.json"))["cells"]
    sust = json.load(open(RES / "exp8_sustained/sustained_summary.json"))["jobs"]
    bridge = json.load(open(RES / "exp9_validation_bridge/bridge_summary.json"))["jobs"]

    def grid_cell(t, dev):
        c = grid[t][dev]
        mem = c["vram_mb"] if dev == "gpu" else c["rss_mb"]
        return mem, c["step_s"]

    def scale_cell(name):
        reps = scale[name]
        mem_key = "vram_mb" if reps[0]["vram_mb"] else "rss_mb"
        return (med([r[mem_key] for r in reps]),
                med([r["step_s"] for r in reps]))

    # job prefix -> (protocol source, device) ; explicit, documented mapping
    MAPPING = {
        "g1": ("grid", "T7", "gpu"), "g2": ("grid", "T2", "gpu"),
        "g3": ("grid", "T3", "gpu"), "g4": ("grid", "T9", "gpu"),
        "g5": ("scale", "gpu40_xxl7b_T2", "gpu"),
        "g6": ("scale", "gpu40_xxl7b_T9", "gpu"),
        "g7": ("grid", "T3", "cpu"), "g8": ("grid", "T7", "cpu"),
        "g9": ("scale", "cpu_xxl7b_T9", "cpu"),
    }
    deltas = {}
    for gid, (src, cell, dev) in MAPPING.items():
        job_key = next(k for k in sust if k.startswith(gid + "_"))
        job = sust[job_key]
        proto_mem, proto_step = (grid_cell(cell, dev) if src == "grid"
                                 else scale_cell(cell))
        long_peak = job["peak_mb_max"] + (CUDA_CTX_MB if dev == "gpu" else 0.0)
        val = bridge[gid]["exp9_validation_phase_s_direct"]
        long_step_fair = job["avg_step_s"] + val
        deltas[gid] = {
            "job": job_key,
            "protocol_cell": f"{src}:{cell}:{dev}",
            "protocol_peak_mb": proto_mem,
            "longrun_peak_mb": round(long_peak, 1),
            "delta_peak_pct": round((long_peak - proto_mem) / proto_mem * 100, 1),
            "protocol_step_s": proto_step,
            "longrun_step_s": job["avg_step_s"],
            "validation_direct_s": val,
            "longrun_step_plus_val_s": round(long_step_fair, 3),
            "delta_time_pct": round((long_step_fair - proto_step) / proto_step * 100, 1),
        }
    out["longrun_vs_protocol"] = deltas

    dst = RES / "table_enrichment.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)
    for gid in sorted(deltas):
        d = deltas[gid]
        print(f"{gid}: peak {d['protocol_peak_mb']:.0f} -> {d['longrun_peak_mb']:.0f} MB "
              f"({d['delta_peak_pct']:+.1f}%), step {d['protocol_step_s']:.2f} -> "
              f"{d['longrun_step_plus_val_s']:.2f} s ({d['delta_time_pct']:+.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Numbers for the enriched paper tables (12-mode grid and long-horizon
training): full-model anchors, per-mode ratios vs. the full model, and
long-run deltas vs. the short cost protocol.

Inputs (all committed results):
  results/exp1_compare/full_{drop,nodrop}_rep*/full_train_metrics.json
      -> GPU full-model anchors (vram_hw_total_peak_mb, avg_step_time_s)
  results/exp2_anchor/cpu_full_rep*/full_train_metrics.json
      -> CPU full-model anchor (rss_peak_sampled_mb, avg_step_time_s)
  results/exp2_anchor/{gpu,cpu}_rep*/naive_anchor_met.json
      -> naive-segmented anchors, every technique off (tech code 00000000000,
         dropout 0.1); GPU vram_hw_total_peak_mb, CPU rss_peak_sampled_mb
  results/exp3_grid/grid_summary.json          -> 12-mode protocol cells
  results/exp5_scale/scale_summary.json        -> 6.9B protocol cells
  results/exp8_sustained/sustained_summary.json-> long-run peaks and step times
  results/exp9_validation_bridge/bridge_summary.json -> direct validation cost

Output: results/table_enrichment.json

Ratio conventions: GPU rows divide by the dropout-matched GPU full anchor;
all CPU rows divide by the single (dropout) CPU full anchor - stated in the
paper caption. Long-run GPU peaks (MiB, from sustained_summary.json) add the
CUDA context calibrated by the four-phase protocol itself (median of
hw_total - reserved over the committed GPU reps: one value for the training
cells, one for the serving cells) so both sides of the delta use the same
basis and the same unit as the grid (hw-total, MiB).

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
MIB_PER_MB = 1e6 / 2**20   # raw exp8 per-step rows are decimal MB


def calibrated_ctx(patterns, total_key, reserved_key):
    """CUDA context (MiB) as the four-phase protocol measured it: hw_total minus
    peak reserved, median over every committed GPU rep matching the patterns."""
    vals = []
    for pat in patterns:
        for f in glob(str(pat)):
            o = json.load(open(f)).get("overall", {})
            if o.get(total_key) and o.get(reserved_key):
                vals.append(o[total_key] - o[reserved_key])
    assert vals, f"no reps for {patterns}"
    return {"ctx_mib": med(vals), "n_reps": len(vals), "min": min(vals), "max": max(vals)}


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

    # ---- 1b. naive-segmented anchors: partition only, every technique off --
    # Same conventions as the grid cells: GPU host = rss_hw_peak_mb, CPU RAM =
    # rss_peak_sampled_mb, optimizer phase = per_phase.optimizer.time_ms / 1000.
    def naive_anchor(pattern, mem_key, host_key):
        reps = [json.load(open(f)) for f in sorted(glob(str(pattern)))]
        assert reps, f"no reps for {pattern}"
        return {"peak_mb": med([d["overall"][mem_key] for d in reps]),
                "host_rss_mb": med([d["overall"][host_key] for d in reps]) if host_key else None,
                "step_s": med([d["avg_step_time_s"] for d in reps]),
                "opt_phase_s": med([d["per_phase"]["optimizer"]["time_ms"] / 1000 for d in reps]),
                "n_reps": len(reps)}
    naive = {
        "gpu_dropout": naive_anchor(RES / "exp2_anchor/gpu_rep*/naive_anchor_met.json",
                                    "vram_hw_total_peak_mb", "rss_hw_peak_mb"),
        "cpu_dropout": naive_anchor(RES / "exp2_anchor/cpu_rep*/naive_anchor_met.json",
                                    "rss_peak_sampled_mb", None),
    }
    for k, v in naive.items():
        v["naive_over_full_peak_pct"] = round(100 * v["peak_mb"] / anchors[k]["peak_mb"], 1)
        v["full_over_naive_peak_x"] = round(anchors[k]["peak_mb"] / v["peak_mb"], 3)
        v["naive_over_full_step_x"] = round(v["step_s"] / anchors[k]["step_s"], 3)
    out["naive_anchors"] = naive

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
    ctx_train = calibrated_ctx([RES / "exp3_grid/gpu_*_rep*/*_met.json",
                                RES / "exp5_scale/gpu40_*_rep*/*_met.json"],
                               "vram_hw_total_peak_mb", "vram_hw_reserved_peak_mb")
    ctx_serve = calibrated_ctx([RES / "exp4_infer/gpu_I*_rep*/*.json"],
                               "vram_hw_total_peak_mb", "vram_reserved_peak_sampled_mb")
    out["cuda_context_mib"] = {"training": ctx_train, "serving": ctx_serve}
    deltas = {}
    for gid, (src, cell, dev) in MAPPING.items():
        job_key = next(k for k in sust if k.startswith(gid + "_"))
        job = sust[job_key]
        proto_mem, proto_step = (grid_cell(cell, dev) if src == "grid"
                                 else scale_cell(cell))
        long_peak = job["peak_mb_max"] + (ctx_train["ctx_mib"] if dev == "gpu" else 0.0)
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

    # ---- 3b. long-run serving footprints on the same basis (Table VII) -----
    serving = {}
    for job_key, job in sust.items():
        if not job_key[0] == "i":
            continue
        gpu = job.get("device") == "cuda"
        band = job.get("peak_mb_band") or [job["peak_mb_max"], job["peak_mb_max"]]
        ctx = ctx_serve["ctx_mib"] if gpu else 0.0
        serving[job_key] = {
            "requests": job.get("n_requests_done"),
            "footprint_mib": round(job["peak_mb_max"] + ctx, 1),
            "footprint_band_mib": [round(band[0] + ctx, 1), round(band[1] + ctx, 1)],
            "band_mib": round(band[1] - band[0], 1),
            "per_token_s": job.get("avg_per_token_s"),
            "basis": "reserved + calibrated CUDA context" if gpu else "resident set",
        }
    out["longrun_serving"] = serving

    # ---- 4. within-run spread: max vs min over the whole run --------------
    # The stability question the long-horizon table answers: over the run
    # itself, how far apart are the largest and smallest per-step peak
    # memory and step time, as a percentage of the minimum.
    drift = {}
    for gid, (_, _, _) in MAPPING.items():
        job_dir = next((RES / "exp8_sustained").glob(gid + "_*"))
        raw = json.load(open(next(job_dir.glob("*_long.json"))))
        ps = raw["per_step"]
        mems = [r["peak_mb"] * MIB_PER_MB for r in ps]   # decimal MB -> MiB
        ts = [r["s"] for r in ps]
        drift[gid] = {
            "job": job_dir.name, "n_steps": len(ps),
            "peak_mb_min": round(min(mems), 1),
            "peak_mb_max": round(max(mems), 1),
            "spread_mem_pct": round((max(mems) - min(mems)) / min(mems) * 100, 1),
            "step_s_min": round(min(ts), 3),
            "step_s_max": round(max(ts), 3),
            "spread_time_pct": round((max(ts) - min(ts)) / min(ts) * 100, 1),
        }
    out["longrun_drift"] = drift

    dst = RES / "table_enrichment.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)
    for gid in sorted(deltas):
        d = deltas[gid]
        print(f"{gid}: peak {d['protocol_peak_mb']:.0f} -> {d['longrun_peak_mb']:.0f} MB "
              f"({d['delta_peak_pct']:+.1f}%), step {d['protocol_step_s']:.2f} -> "
              f"{d['longrun_step_plus_val_s']:.2f} s ({d['delta_time_pct']:+.1f}%)")
    for gid in sorted(drift):
        d = drift[gid]
        print(f"{gid} spread: mem {d['spread_mem_pct']:.1f}%, time {d['spread_time_pct']:.1f}% "
              f"(max vs min over {d['n_steps']} steps)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Granularity sweep (paper S5) — the COARSE memory<->time dial.

Fix the technique set (ALL enabled) and sweep the partition from coarse to fine on the large
model. Finer granularity -> smaller resident working set -> lower peak memory, at the cost of
more load/offload + recompute boundaries -> more time. Produces the granularity curve.

Granularities must divide the large model (d_model=1280, n_heads=20, d_ff=5120).
Results -> results/{gpu,cpu}_granularity/ (granularity_train.json + figure).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SegmentationConfig   # noqa: E402
from techniques import Tech             # noqa: E402
from seg_cost_lib import run_cost       # noqa: E402

# coarse -> fine (E,A,M,H); all divide the large model
GRID = [(2, 1, 1, 2), (4, 2, 2, 4), (8, 2, 2, 8), (16, 4, 4, 16), (20, 5, 5, 20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")   # model; seg overridden per point
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    a = ap.parse_args()
    is_cuda = a.device.startswith("cuda")
    tag = "gpu" if is_cuda else "cpu"
    out_dir = HERE / "results" / f"{tag}_granularity"; out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for e, am, mm, h in GRID:
        seg = SegmentationConfig(embedding_segments=e, attention_segments=am,
                                 mlp_chunks=mm, output_head_segments=h)
        print(f"\n===== granularity {seg.code} (all techniques ON) =====")
        met = run_cost(a.preset, a.device, out_dir, prefix=f"g_{seg.code}", batch=a.batch,
                       n_steps=a.n_steps, from_scratch=True, seg_override=seg, tech=Tech())
        o = met["overall"]
        rows.append({"code": seg.code, "n_segments": e * am * mm * h,
                     "vram_peak_mb": o["vram_hw_total_peak_mb"],
                     "rss_peak_mb": o["rss_peak_sampled_mb"],
                     "step_time_s": round(met["avg_step_time_s"], 2)})
    out = out_dir / "granularity_train.json"
    json.dump({"run": "granularity_sweep", "preset": a.preset, "device": a.device,
               "note": "all techniques ON; seg partition varied coarse->fine", "grid": rows}, open(out, "w"), indent=2)

    memk = "vram_peak_mb" if is_cuda else "rss_peak_mb"
    lbl = "peak VRAM (MB)" if is_cuda else "peak RSS (MB)"
    codes = [r["code"] for r in rows]; mem = [r[memk] for r in rows]; t = [r["step_time_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(rows))
    ax.bar(x, mem, color="#4C72B0", alpha=0.85); ax.set_ylabel(lbl, color="#4C72B0")
    ax.set_xticks(list(x)); ax.set_xticklabels(codes)
    ax.set_xlabel("segmentation granularity  E×A×M×H  (coarse → fine)")
    a2 = ax.twinx(); a2.plot(list(x), t, "o-", color="#C44E52", lw=2, ms=6); a2.set_ylabel("step time (s)", color="#C44E52")
    ax.set_title(f"Granularity dial ({tag} train, all techniques ON): finer → less memory, more time")
    fig.tight_layout(); fig.savefig(HERE / "results" / "figures" / f"granularity_{tag}_train.png", dpi=140)
    print(f"\n  {'code':12}{lbl:>14}{'step s':>9}")
    for r in rows:
        print(f"  {r['code']:12}{r[memk]:14.0f}{r['step_time_s']:9.1f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

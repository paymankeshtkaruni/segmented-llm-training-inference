"""
Shared ladder runner — walk a cumulative technique ladder on the FIXED segmented model,
measuring peak memory + step time at each rung (the memory-down / time-up staircase).

The segmentation config is NOT varied (it is the preset's, e.g. large 8x2x2x8). Only the
memory-reduction TECHNIQUES are toggled, cumulatively, from the all-OFF baseline (rung 0)
via techniques.cumulative(). Each rung runs the verified MemFlow profiler (seg_cost_lib).
"""
from __future__ import annotations

import json
from pathlib import Path

from techniques import cumulative, TRAIN_LADDER, INFER_LADDER
from measure_one import run_isolated   # fresh process per rung: no allocator residue


def run_train_ladder(preset: str, device: str, out_dir: Path,
                     batch: int = 4, n_steps: int = 3, seq_len=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = str(device).startswith("cuda")
    rows = []
    for name, desc, tech in cumulative(TRAIN_LADDER):
        print(f"\n===== {name}  tech={tech.code()}  ({desc}) =====")
        met = run_isolated("train", preset, device, out_dir, name, tech,
                           batch=batch, n_steps=n_steps, seq_len=seq_len)
        o = met["overall"]; pp = met.get("per_phase", {})
        rows.append({
            "rung": name, "adds": desc, "tech_code": tech.code(),
            "vram_peak_mb": o["vram_hw_total_peak_mb"],   # no-miss total VRAM (GPU)
            "rss_peak_mb":  o["rss_peak_sampled_mb"],      # PER-RUNG sampled RSS (ru_maxrss is
                                                           # process-lifetime monotonic -> wrong here)
            "step_time_s":  round(met["avg_step_time_s"], 2),
            "fwd_ms": pp.get("forward", {}).get("time_ms"),
            "bwd_ms": pp.get("backward", {}).get("time_ms"),
            "opt_ms": pp.get("optimizer", {}).get("time_ms"),
        })
    _write(out_dir, "train", preset, device, batch, n_steps, rows, is_cuda)
    return rows


def run_infer_ladder(preset: str, device: str, out_dir: Path,
                     prompt_len: int = 256, gen_tokens: int = 8):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = str(device).startswith("cuda")
    rows = []
    for name, desc, tech in cumulative(INFER_LADDER):
        print(f"\n===== {name}  tech={tech.code()}  ({desc}) =====")
        met = run_isolated("infer", preset, device, out_dir, name, tech,
                           prompt_len=prompt_len, gen_tokens=gen_tokens)
        o = met["overall"]
        rows.append({
            "rung": name, "adds": desc, "tech_code": tech.code(),
            "vram_peak_mb": o["vram_hw_total_peak_mb"],
            "rss_peak_mb":  o["rss_hw_peak_mb"],
            "per_token_s":  met["per_token_s"],
        })
    out = out_dir / "ladder_infer.json"
    json.dump({"run": "infer_incremental_ablation", "preset": preset, "device": device,
               "prompt_len": prompt_len, "gen_tokens": gen_tokens, "ladder": rows}, open(out, "w"), indent=2)
    memk = "vram_peak_mb" if is_cuda else "rss_peak_mb"
    lbl = "VRAM MB" if is_cuda else "RSS MB"
    print(f"\n\n========== INCREMENTAL LADDER (INFER on {device}) ==========")
    hdr = f"{'rung':16} {'code':12} {lbl:>9} {'Δmem':>8} {'tok s':>8} {'Δt':>7}  adds"
    print(hdr); print("-" * len(hdr))
    m0 = t0 = None
    for r in rows:
        mem, t = r[memk], r["per_token_s"]
        dm = "" if m0 is None else f"{mem - m0:+.0f}"
        dt = "" if t0 is None else f"{t - t0:+.2f}"
        print(f"{r['rung']:16} {r['tech_code']:12} {mem:9.0f} {dm:>8} {t:8.3f} {dt:>7}  {r['adds']}")
        m0, t0 = mem, t
    print(f"\n  -> {out}")
    return rows


def _write(out_dir: Path, kind: str, preset, device, batch, n_steps, rows, is_cuda):
    out = out_dir / f"ladder_{kind}.json"
    json.dump({"run": f"{kind}_incremental_ablation", "preset": preset, "device": device,
               "batch": batch, "n_steps": n_steps,
               "metric_notes": "vram_peak_mb = no-miss total VRAM (context+reserved); "
                               "rss_peak_mb = no-miss host RSS; step_time_s = avg over steps",
               "ladder": rows}, open(out, "w"), indent=2)
    memk = "vram_peak_mb" if is_cuda else "rss_peak_mb"
    lbl = "VRAM MB" if is_cuda else "RSS MB"
    print(f"\n\n========== INCREMENTAL LADDER ({kind.upper()} on {device}) ==========")
    hdr = f"{'rung':16} {'code':12} {lbl:>9} {'Δmem':>8} {'step s':>8} {'Δt':>7}  adds"
    print(hdr); print("-" * len(hdr))
    m0 = t0 = None
    for r in rows:
        mem, t = r[memk], r["step_time_s"]
        dm = "" if m0 is None else f"{mem - m0:+.0f}"
        dt = "" if t0 is None else f"{t - t0:+.1f}"
        print(f"{r['rung']:16} {r['tech_code']:12} {mem:9.0f} {dm:>8} {t:8.1f} {dt:>7}  {r['adds']}")
        m0, t0 = mem, t
    print(f"\n  -> {out}")

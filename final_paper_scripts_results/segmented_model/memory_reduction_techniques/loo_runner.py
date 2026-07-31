"""
Leave-one-out (LOO) MARGINAL ablation — order-independent per-technique value.

Start from ALL techniques ON, then drop exactly ONE and measure how much memory it was
saving (mem rises) and how much time it was costing (time falls). Efficiency = MB saved
per second added, computed against the SAME all-on reference for every technique (so it is
not order-dependent, unlike the cumulative ladder).

Coupling note: `recompute` OFF (full-graph backward) requires params resident, i.e.
`stream_segments` OFF — so recompute cannot be dropped alone. It is a structural
PREREQUISITE for streaming during training and is excluded from the LOO set (its value is
read from the cumulative ladder's r3->r4 step instead).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from techniques import Tech
from seg_cost_lib import run_cost

# independently-droppable techniques (recompute excluded — prerequisite for streaming)
TRAIN_LOO = ["sdpa", "mlp_running_sum", "chunked_ce", "stream_segments", "offload_records",
             "park_grads_host", "offload_adam", "segment_wo", "free_device"]
INFER_LOO = ["sdpa", "mlp_running_sum", "chunked_ce", "stream_segments", "segment_wo",
             "free_device", "no_kv_cache"]


def _mem_time(met, is_cuda, kind):
    o = met["overall"]
    mem = o["vram_hw_total_peak_mb"] if is_cuda else o["rss_peak_sampled_mb"]
    t = met["avg_step_time_s"] if kind == "train" else met["per_token_s"]
    return mem, t


def _run_loo(preset, device, out_dir: Path, kind, flags, run_fn, ref_kwargs):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = str(device).startswith("cuda")
    print(f"\n##### LOO {kind} reference: ALL techniques ON #####")
    ref = run_fn(preset, device, out_dir, prefix="all_on", tech=Tech(), **ref_kwargs)
    ref_mem, ref_t = _mem_time(ref, is_cuda, kind)
    rows = []
    for f in flags:
        print(f"\n##### LOO {kind}: drop {f} (rest ON) #####")
        met = run_fn(preset, device, out_dir, prefix=f"drop_{f}", tech=replace(Tech(), **{f: False}), **ref_kwargs)
        mem, t = _mem_time(met, is_cuda, kind)
        d_mem = mem - ref_mem            # memory this technique SAVES (drop -> mem rises)
        d_t = ref_t - t                  # time this technique COSTS (drop -> time falls); +ve = costs
        eff = d_mem / max(d_t, 0.05)
        rows.append({"technique": f, "saves_mem_mb": round(d_mem, 1),
                     "costs_time_s": round(d_t, 3), "efficiency_mb_per_s": round(eff, 1),
                     "mem_without_mb": round(mem, 1), "time_without_s": round(t, 3)})
    rows.sort(key=lambda r: -r["efficiency_mb_per_s"])
    out = out_dir / f"loo_{kind}.json"
    json.dump({"run": f"loo_{kind}", "preset": preset, "device": device,
               "reference_all_on": {"mem_mb": round(ref_mem, 1), "time_s": round(ref_t, 3)},
               "note": "recompute excluded (prerequisite for streaming); saves_mem = memory the "
                       "technique removes vs all-on; costs_time = time it adds; ranked by efficiency",
               "techniques": rows}, open(out, "w"), indent=2)
    memlbl = "VRAM" if is_cuda else "RSS"
    print(f"\n===== LOO {kind} ({device}) — marginal value, ranked by efficiency =====")
    print(f"  all-ON reference: {ref_mem:.0f} {memlbl} MB / {ref_t:.3f} s")
    print(f"  {'technique':18}{'saves MB':>9}{'costs s':>9}{'MB/s':>9}")
    for r in rows:
        print(f"  {r['technique']:18}{r['saves_mem_mb']:9.0f}{r['costs_time_s']:9.2f}{r['efficiency_mb_per_s']:9.0f}")
    print(f"  -> {out}")
    return rows


def run_train_loo(preset, device, out_dir, batch=4, n_steps=3):
    return _run_loo(preset, device, out_dir, "train", TRAIN_LOO, run_cost,
                    {"batch": batch, "n_steps": n_steps, "from_scratch": True})


def run_infer_loo(preset, device, out_dir, prompt_len=256, gen_tokens=8):
    from infer_cost_lib import run_infer_cost
    return _run_loo(preset, device, out_dir, "infer", INFER_LOO, run_infer_cost,
                    {"prompt_len": prompt_len, "gen_tokens": gen_tokens, "from_scratch": True})

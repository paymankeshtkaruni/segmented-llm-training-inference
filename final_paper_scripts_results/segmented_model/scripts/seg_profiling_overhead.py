"""
Measure the REAL wall-time overhead of the cost profiler's collection points.

Runs identical segmented steps (forward+backward+optimizer+validation) under four
configs, adding one collection mechanism at a time, and times each:
  1. none            — no hooks, no threads (pure segmented-method time)
  2. +op-marks       — loader hook + optrace marks (per-op memory reads + append)
  3. +fast-thread    — also the ~1 kHz alloc/reserved/RSS sampler thread
  4. +nvidia-smi     — also the ~33 Hz nvidia-smi subprocess sampler (the full profiler)

The per-mechanism overhead is the step-time delta between consecutive configs. One
trainer is built and warmed once, then reused, so only the profiling config differs.

Writes seg_a2_profiling_overhead.{json,png} to temp/. (Confirms profiling is a minor
contributor and isolates which mechanism costs the most.)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys
SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
import optrace                                          # noqa: E402
from trainer import SegmentedTrainer                   # noqa: E402
from seg_cost_lib import MemFlow, _synth, _fwd, _val    # reuse the real profiler + step


def time_steps(tr, x, y, pad, n, is_cuda):
    """Run n full steps (fwd+bwd+opt+val) and return per-step wall times (s)."""
    times = []
    last_grads = None
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = _fwd(tr, x, pad)
        g = tr.bwd.backward(x, y, pad_token_id=pad)
        tr.opt.step(g, clip_norm=1.0)
        last_grads = g
        _val(tr, x, y, pad)
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def run_config(name, tr, x, y, pad, is_cuda, n, *, marks, fast, slow):
    mf = None
    if marks or fast or slow:
        mf = MemFlow(is_cuda)
        tr.loader.profiler = mf if marks else None
        optrace.set_hook(mf.op_mark if marks else None)
        if fast or slow:
            mf.start(fast=fast, slow=slow)
    else:
        tr.loader.profiler = None
        optrace.set_hook(None)
    times = time_steps(tr, x, y, pad, n, is_cuda)
    if mf is not None:
        mf.stop()
    tr.loader.profiler = None
    optrace.set_hook(None)
    mean = sum(times) / len(times)
    print(f"  {name:16} mean step = {mean*1000:8.1f} ms  (n={n}, per-step {[round(t*1000) for t in times]})")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "temp")
    a = ap.parse_args()
    is_cuda = a.device.startswith("cuda")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    tr = SegmentedTrainer(a.preset, a.device, a.out_dir / "_ovh_work",
                          store_kind=("cpu_ram" if is_cuda else "disk"))
    m = tr.m
    pad = tr.tok.pad_token_id
    x, y = _synth(a.batch, m.max_seq_len, m.vocab_size, pad, a.device, 0)

    print(f"[overhead] {a.preset} on {a.device}  {m.n_params_estimate/1e6:.0f}M  bs={a.batch} steps/config={a.steps}")
    # warm once (build optimizer state, allocator pool) so configs compare at steady state
    run_config("warmup", tr, x, y, pad, is_cuda, 1, marks=False, fast=False, slow=False)

    t_none = run_config("none",        tr, x, y, pad, is_cuda, a.steps, marks=False, fast=False, slow=False)
    t_mark = run_config("+op-marks",   tr, x, y, pad, is_cuda, a.steps, marks=True,  fast=False, slow=False)
    t_fast = run_config("+fast-thread",tr, x, y, pad, is_cuda, a.steps, marks=True,  fast=True,  slow=False)
    t_full = run_config("+nvidia-smi", tr, x, y, pad, is_cuda, a.steps, marks=True,  fast=True,  slow=True)

    base = t_none
    rows = [
        ("op-marks (per-op reads+append)", t_mark - t_none),
        ("fast thread (~1kHz alloc/reserved/RSS)", t_fast - t_mark),
        ("nvidia-smi thread (~33Hz subprocess)", t_full - t_fast),
    ]
    total = t_full - t_none
    res = {
        "preset": a.preset, "device": a.device, "batch": a.batch, "steps_per_config": a.steps,
        "method_only_step_s": round(t_none, 4),
        "full_profiled_step_s": round(t_full, 4),
        "mechanisms": [{"name": n, "overhead_s": round(d, 4), "pct_of_method": round(100 * d / base, 2)}
                       for n, d in rows],
        "total_profiling_overhead_s": round(total, 4),
        "total_profiling_pct_of_method": round(100 * total / base, 2),
    }
    json.dump(res, open(a.out_dir / "seg_a2_profiling_overhead.json", "w"), indent=2)

    print("\n  ===== REAL profiling-overhead table =====")
    print(f"  method-only step:    {t_none*1000:8.1f} ms")
    for n, d in rows:
        print(f"  + {n:42} {d*1000:7.1f} ms  ({100*d/base:5.2f}% of method)")
    print(f"  = full profiled step:{t_full*1000:8.1f} ms   total profiling = {total*1000:.1f} ms ({100*total/base:.2f}%)")
    # Plot is generated separately by scripts/plots/plot_a2_profiling_overhead.py (reads this json).


if __name__ == "__main__":
    main()

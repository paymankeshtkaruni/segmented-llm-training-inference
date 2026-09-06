#!/usr/bin/env python
"""exp9: long-horizon training runs at the cost scale.

Answers "nobody trains for a handful of steps": run N consecutive training
steps (hundreds to a thousand) and record, PER STEP, wall time and peak
memory, so the paper can (a) demonstrate cost stationarity over hours and
(b) quote headline per-step times as long-run averages.

Deliberate differences from the short-protocol cells (seg_cost_lib.run_cost),
each in the direction of faithfulness to real training:
  - no per-step validation phase (real training does not validate every step);
  - no per-step torch.cuda.empty_cache()/gc.collect() (the short protocol
    calls both; a stationarity claim is stronger without them);
  - no sampling timeline (a 1,000-step trace would be GB-scale); instead the
    allocator high-water mark is reset and read every step on GPU, and a
    50 ms RSS sampler tracks the per-step maximum on CPU.

Writes <out-dir>/<prefix>_long.json: one compact row per step + summary.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from techniques import Tech                                # noqa: E402


def tech_from_code(code: str) -> Tech:
    names = [f.name for f in fields(Tech)]
    return Tech(**{n: c == "1" for n, c in zip(names, code)})


class RssSampler:
    """Per-step RSS max via a 50 ms sampling thread (CPU runs)."""

    def __init__(self):
        self._max = 0.0
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _rss(self):
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1e6   # pages -> MB

    def _loop(self):
        while self._run:
            self._max = max(self._max, self._rss())
            time.sleep(0.05)

    def start(self):
        self._t.start()
        return self

    def reset(self):
        self._max = self._rss()

    def peak(self):
        return self._max

    def stop(self):
        self._run = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--tech-code", required=True)
    ap.add_argument("--update-style", default="after_full",
                    choices=["after_full", "immediate"])
    ap.add_argument("--n-steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from seg_cost_lib import _synth
    from trainer import SegmentedTrainer

    is_cuda = str(a.device).startswith("cuda")
    tech = tech_from_code(a.tech_code)
    tr = SegmentedTrainer(a.preset, a.device, a.out_dir / "_work",
                          store_kind=("cpu_ram" if is_cuda else "disk"),
                          from_scratch=True, tech=tech,
                          update_style=a.update_style)
    imm = (a.update_style == "immediate")
    _opt_step = (lambda g: tr.opt.step_shared(g)) if imm \
        else (lambda g: tr.opt.step(g, clip_norm=1.0))
    m = tr.m
    seq = a.seq_len or m.max_seq_len
    pad = tr.tok.pad_token_id

    sampler = None if is_cuda else RssSampler().start()

    # one warm-up step, excluded from every statistic
    xw, yw = _synth(a.batch, seq, m.vocab_size, pad, a.device, 0)
    gw = tr.bwd.backward(xw, yw, pad_token_id=pad)
    _opt_step(gw)
    del xw, yw, gw
    if is_cuda:
        torch.cuda.synchronize()

    rows = []
    t_start = time.perf_counter()
    for step in range(a.n_steps):
        x, y = _synth(a.batch, seq, m.vocab_size, pad, a.device, 1 + step)
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        else:
            sampler.reset()
        t0 = time.perf_counter()
        grads = tr.bwd.backward(x, y, pad_token_id=pad)
        _opt_step(grads)
        if is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_reserved() / 1e6 if is_cuda
                else sampler.peak())
        rows.append({"step": step + 1, "s": round(dt, 3),
                     "peak_mb": round(peak, 1)})
        grads = None
        del x, y
        if (step + 1) % 25 == 0:
            print(f"  step {step+1}/{a.n_steps}  {dt:.2f}s  peak={peak:.0f}MB",
                  flush=True)
    if sampler:
        sampler.stop()

    times = [r["s"] for r in rows]
    peaks = [r["peak_mb"] for r in rows]
    srt = sorted(times)
    out = {
        "run": "exp9_long_horizon",
        "preset": a.preset, "device": a.device, "tech_code": a.tech_code,
        "update_style": a.update_style, "batch": a.batch, "seq_len": seq,
        "n_steps": a.n_steps,
        "protocol": "warm-up step excluded; no per-step validation; no "
                    "per-step empty_cache/gc (see module docstring)",
        "memory_note": ("per-step peak reserved (allocator high-water mark, "
                        "reset each step); add the CUDA context for no-miss "
                        "totals" if is_cuda
                        else "per-step peak RSS (50 ms sampler)"),
        "wall_total_s": round(time.perf_counter() - t_start, 1),
        "avg_step_s": round(statistics.mean(times), 3),
        "median_step_s": round(statistics.median(times), 3),
        "step_s_p5_p95": [round(srt[int(len(srt) * 0.05)], 3),
                          round(srt[min(len(srt) - 1, int(len(srt) * 0.95))], 3)],
        "peak_mb_first": peaks[0], "peak_mb_max": max(peaks),
        "peak_mb_max_at_step": peaks.index(max(peaks)) + 1,
        "peak_mb_band": [min(peaks), max(peaks)],
        "per_step": rows,
    }
    dst = a.out_dir / f"{a.prefix}_long.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_step"}, indent=1))
    print("->", dst)
    import shutil
    work = a.out_dir / "_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
        print("cleaned scratch", work)


if __name__ == "__main__":
    main()

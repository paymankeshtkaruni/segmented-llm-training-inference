#!/usr/bin/env python
"""exp8: long-horizon SERVING runs.

A serving process must hold its memory over thousands of requests, not one
probe: this runner answers many independent greedy-generation requests and
records, PER REQUEST, wall time and peak memory, so the paper can show the
serving footprint is stationary (no per-request residue, no session leaks)
and quote per-token latency as a long-run average.

Engines:
  --engine torch : the segmented PyTorch runtime (any inference tech-code,
                   e.g. resident+KV-cache or streamed+recompute).
  --engine onnx  : the PyTorch-free weights-as-inputs runtime with weights
                   read from disk per use (the smallest serving mode).

Per request: a fresh synthetic prompt, greedy decode of --gen-tokens tokens.
GPU peak = allocator high-water mark reset per request; CPU peak = 50 ms RSS
sampler maximum. Rolling partial writes every 10 requests; --max-hours
graceful deadline. Writes <out-dir>/<prefix>_serve.json.
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
sys.path.insert(0, str(HERE.parent / "scripts"))

from techniques import Tech                                # noqa: E402


def tech_from_code(code: str) -> Tech:
    names = [f.name for f in fields(Tech)]
    return Tech(**{n: c == "1" for n, c in zip(names, code)})


class RssSampler:
    def __init__(self):
        self._max = 0.0
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _rss(self):
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1e6

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


class _NoEos:
    eos_token_id = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["torch", "onnx"], required=True)
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tech-code", default="11100111000",
                    help="torch engine: inference technique code")
    ap.add_argument("--onnx-dir", type=Path, default=None,
                    help="onnx engine: exported weights-as-inputs directory")
    ap.add_argument("--n-requests", type=int, required=True)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=46.0)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = str(a.device).startswith("cuda") and a.engine == "torch"

    if a.engine == "torch":
        import torch
        from trainer import SegmentedTrainer
        from inference import SegmentedGenerator
        tr = SegmentedTrainer(a.preset, a.device, a.out_dir / "_work",
                              store_kind=("cpu_ram" if is_cuda else "disk"),
                              from_scratch=True, tech=tech_from_code(a.tech_code))
        m = tr.m
        pad = tr.tok.pad_token_id
        tr.fwd.eval()
        gen = SegmentedGenerator(tr.fwd, _NoEos())
        vocab, max_seq = m.vocab_size, m.max_seq_len

        def one_request(seed):
            g = torch.Generator().manual_seed(seed)
            plen = min(a.prompt_len, max(1, max_seq - a.gen_tokens))
            ids = torch.randint(0, vocab, (plen,), generator=g).tolist()
            if pad is not None:
                ids = [((t + 1) % vocab) if t == pad else t for t in ids]
            out = gen.generate(ids, max_new_tokens=a.gen_tokens)
            return len(out)

        def sync():
            if is_cuda:
                torch.cuda.synchronize()

        def reset_peak():
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

        def read_peak():
            return torch.cuda.max_memory_reserved() / 1e6 if is_cuda else None
    else:
        import numpy as np
        from seg_c_onnx_cost import SegOnnxRuntime
        rt = SegOnnxRuntime(a.onnx_dir, "cpu", preload_weights=False)
        cfg = rt.cfg
        vocab, max_seq = cfg["vocab_size"], cfg["max_seq_len"]

        def one_request(seed):
            rng = np.random.default_rng(seed)
            plen = min(a.prompt_len, max(1, max_seq - a.gen_tokens))
            ids = rng.integers(0, vocab, (1, plen)).astype(np.int64)
            for _ in range(a.gen_tokens):
                win = ids[:, -max_seq:] if ids.shape[1] > max_seq else ids
                nxt = rt.next_token(win)
                ids = np.concatenate([ids, nxt.reshape(1, 1)], axis=1)
            return a.gen_tokens

        def sync():
            pass

        def reset_peak():
            pass

        def read_peak():
            return None

    sampler = None if is_cuda else RssSampler().start()

    # warm-up request, excluded from every statistic
    one_request(0)
    sync()

    rows = []
    t_start = time.perf_counter()

    def flush(final=False):
        lat = [r["s"] for r in rows]
        peaks = [r["peak_mb"] for r in rows if r["peak_mb"] is not None]
        srt = sorted(lat)
        out = {
            "run": "exp8_long_horizon_serve",
            "engine": a.engine, "preset": a.preset, "device": a.device,
            "tech_code": a.tech_code if a.engine == "torch" else None,
            "onnx_dir": str(a.onnx_dir) if a.onnx_dir else None,
            "prompt_len": a.prompt_len, "gen_tokens": a.gen_tokens,
            "n_requests_target": a.n_requests, "n_requests_done": len(rows),
            "complete": final and len(rows) == a.n_requests,
            "memory_note": ("per-request peak reserved VRAM (reset each request); "
                            "add CUDA context for no-miss totals" if is_cuda
                            else "per-request peak RSS (50 ms sampler)"),
            "wall_total_s": round(time.perf_counter() - t_start, 1),
            "avg_request_s": round(statistics.mean(lat), 3) if lat else None,
            "avg_per_token_s": (round(statistics.mean(lat) / a.gen_tokens, 4)
                                if lat else None),
            "request_s_p5_p95": ([round(srt[int(len(srt) * 0.05)], 3),
                                  round(srt[min(len(srt) - 1, int(len(srt) * 0.95))], 3)]
                                 if lat else None),
            "peak_mb_first": peaks[0] if peaks else None,
            "peak_mb_max": max(peaks) if peaks else None,
            "peak_mb_max_at_request": (peaks.index(max(peaks)) + 1) if peaks else None,
            "peak_mb_band": [min(peaks), max(peaks)] if peaks else None,
            "per_request": rows,
        }
        json.dump(out, open(a.out_dir / f"{a.prefix}_serve.json", "w"), indent=1)
        return out

    deadline = t_start + a.max_hours * 3600
    for req in range(a.n_requests):
        if time.perf_counter() > deadline:
            print(f"[deadline] {a.max_hours}h reached after {len(rows)} requests",
                  flush=True)
            break
        sync()
        reset_peak()
        if sampler:
            sampler.reset()
        t0 = time.perf_counter()
        one_request(1 + req)
        sync()
        dt = time.perf_counter() - t0
        peak = read_peak() if is_cuda else (sampler.peak() if sampler else None)
        rows.append({"req": req + 1, "s": round(dt, 3),
                     "peak_mb": round(peak, 1) if peak is not None else None})
        if (req + 1) % 10 == 0:
            flush()
        if (req + 1) % 50 == 0:
            print(f"  request {req+1}/{a.n_requests}  {dt:.2f}s", flush=True)
    if sampler:
        sampler.stop()

    out = flush(final=True)
    print(json.dumps({k: v for k, v in out.items() if k != "per_request"}, indent=1))
    print("->", a.out_dir / f"{a.prefix}_serve.json")
    import shutil
    work = a.out_dir / "_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
        print("cleaned scratch", work)


if __name__ == "__main__":
    main()

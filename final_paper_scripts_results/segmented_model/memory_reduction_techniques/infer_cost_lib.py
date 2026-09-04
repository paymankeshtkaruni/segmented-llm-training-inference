"""
Segmented INFERENCE cost profiler — peak memory + per-token time of greedy generation,
under the memory-reduction technique toggles (esp. no_kv_cache and stream_segments).

Reuses the verified MemFlow profiler (seg_cost_lib) on one clock. Builds the segmented
forward with a techniques.Tech, runs a warmup token, then measures a prefill + gen_tokens
decode with the SegmentedGenerator (which honors no_kv_cache: recompute vs K,V cache).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import torch

SEG_PKG = Path(__file__).resolve().parent / "segmentation_management"   # LOCAL copy
sys.path.insert(0, str(SEG_PKG))
import optrace                                           # noqa: E402
from trainer import SegmentedTrainer                    # noqa: E402
from inference import SegmentedGenerator                # noqa: E402
from seg_cost_lib import MemFlow                         # reuse the verified profiler

_MB = 1024 ** 2


class _NoEos:                       # force the full gen_tokens (measurement, no early stop)
    eos_token_id = None


def run_infer_cost(preset, device, out_dir: Path, prefix, prompt_len=256, gen_tokens=8,
                   tech=None, from_scratch=True, seg_override=None):
    is_cuda = str(device).startswith("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_cuda:
        torch.zeros(1, device=device); torch.cuda.synchronize()

    mf = MemFlow(is_cuda); mf.start(); mf.set_phase("build")
    tr = SegmentedTrainer(preset, device, out_dir / "_work",
                          store_kind=("cpu_ram" if is_cuda else "disk"),
                          from_scratch=from_scratch, tech=tech,
                          seg_override=seg_override)
    m = tr.m; pad = tr.tok.pad_token_id
    tr.fwd.eval()
    gen = SegmentedGenerator(tr.fwd, _NoEos())
    tr.loader.profiler = mf
    optrace.set_hook(mf.op_mark)

    g = torch.Generator().manual_seed(0)
    # leave room for gen_tokens so KV-cache decode never runs past the context window
    plen = min(prompt_len, max(1, m.max_seq_len - gen_tokens))
    ids = torch.randint(0, m.vocab_size, (plen,), generator=g).tolist()
    if pad is not None:
        ids = [((t + 1) % m.vocab_size) if t == pad else t for t in ids]

    mf.set_phase("warmup")
    gen.generate(ids, max_new_tokens=1)

    mf.set_phase("generate")
    if is_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = gen.generate(ids, max_new_tokens=gen_tokens)
    if is_cuda:
        torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    hw_r = torch.cuda.max_memory_reserved() / _MB if is_cuda else 0.0
    mf.stop(); optrace.set_hook(None)

    ctx = mf.context_mb()
    rss_hw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # KB->MB lifetime peak
    n = max(1, len(out))
    metrics = {
        "run": "segmented_infer_cost", "preset": preset, "device": device,
        "model_config": m.to_dict(), "seg_config": tr.s.to_dict(),
        "prompt_len": len(ids), "gen_tokens": len(out),
        "overall": {
            "vram_hw_total_peak_mb": round(ctx + hw_r, 1),          # no-miss total VRAM
            "vram_reserved_peak_sampled_mb": round(max((r for _, _, r, _ in mf.fast), default=0.0), 1),
            "rss_hw_peak_mb": round(rss_hw, 1),
            "rss_peak_sampled_mb": round(max((h for _, _, _, h in mf.fast), default=0.0), 1),
            "cuda_context_mb": round(ctx, 1),
        },
        "generate_total_s": round(gen_s, 3),
        "per_token_s": round(gen_s / n, 3),
    }
    json.dump(metrics, open(out_dir / f"{prefix}_metrics.json", "w"), indent=2)
    o = metrics["overall"]
    print(f"[infer-cost] {preset} on {device}  prompt={len(ids)} gen={len(out)}  "
          f"| VRAM_hw={o['vram_hw_total_peak_mb']:.0f} RSS_hw={o['rss_hw_peak_mb']:.0f}MB  "
          f"per-token={metrics['per_token_s']:.3f}s  -> {out_dir/(prefix+'_metrics.json')}")
    return metrics

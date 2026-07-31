"""
B2 — segmented torch INFERENCE cost (memory + time), one segment at a time.

Reuses the A2 profiler (MemFlow: continuous VRAM+RSS, per-op marks, GUARANTEED no-miss
high-water peak, categories not blended). Inference is forward-only (eval, no_grad) — no
backward graph / grads / optimizer state — so it must be far cheaper than A2 training.

Applies the T05 (GPU) / T02 (CPU) findings:
  * empty_cache() after each segment release on CUDA  (free_device — else nvidia-smi
    over-reports the freed pool; T05's key fix)
  * storage cpu_ram on GPU / disk on CPU              (device->target rule; T02/T05)
  * honest VRAM = nvidia-smi (not torch.allocated)    (T05: 33 MB alloc vs 544 MB smi)
  * in-memory generation state (no per-token disk I/O) (T02 in_memory=True)
Plus our extras: no-miss peak, per-op category split, attn_out_proj streamed, CPU malloc
env + build-vs-step separation.

Phases: prefill (forward at [B,seq] = the memory bound) and decode (generate tokens, B=1).
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import torch

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for seg_cost_lib
import optrace                                              # noqa: E402
from trainer import SegmentedTrainer                        # noqa: E402
from inference import SegmentedGenerator                    # noqa: E402
from memory import host_rss_mb                              # noqa: E402
from seg_cost_lib import MemFlow, _phase_breakdown, _op_family, _MB   # noqa: E402


def run_infer_cost(preset, device, out_dir: Path, prefix, batch=4, seq=512,
                   gen_tokens=8, prompt_len=64, no_empty_cache=False):
    is_cuda = str(device).startswith("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    if no_empty_cache and is_cuda:
        # ABLATION: neuter the T05 fix so free_device's empty_cache() is a no-op,
        # to measure whether it is what bounds inference VRAM.
        torch.cuda.empty_cache = lambda *a, **k: None
        print("[ablation] torch.cuda.empty_cache DISABLED")
    host_baseline = host_rss_mb()
    if is_cuda:
        torch.zeros(1, device=device); torch.cuda.synchronize()

    mf = MemFlow(is_cuda); mf.start(); mf.set_phase("build")
    tr = SegmentedTrainer(preset, device, out_dir / "_work",
                          store_kind=("cpu_ram" if is_cuda else "disk"))
    tr.fwd.eval()                                           # inference mode (dropout off)
    m = tr.m; pad = tr.tok.pad_token_id
    tr.loader.profiler = mf; optrace.set_hook(mf.op_mark)
    host_after_build = host_rss_mb()

    g = torch.Generator().manual_seed(0)
    windows = []

    # ---- PREFILL: forward over a full [batch, seq] context (the memory bound) ----
    x = torch.randint(0, m.vocab_size, (batch, seq), generator=g)
    if pad is not None:
        x[x == pad] = (int(pad) + 1) % m.vocab_size
    x = x.to(device)
    for rep in range(3):                                    # 3 reps for steady state
        mf.set_phase("prefill")
        if is_cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            h = tr.fwd.forward_hidden(x, pad_token_id=None)
        del h
        if is_cuda:
            torch.cuda.synchronize()
        hw_a = torch.cuda.max_memory_allocated() / _MB if is_cuda else 0.0
        hw_r = torch.cuda.max_memory_reserved() / _MB if is_cuda else 0.0
        windows.append(("prefill", rep, t0, time.perf_counter(), hw_a, hw_r))
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

    # ---- DECODE: autoregressive BATCH generation (B examples together, one segment) ----
    gen = SegmentedGenerator(tr.fwd, tr.tok)
    prompts = torch.randint(0, m.vocab_size, (batch, prompt_len), generator=g)
    if pad is not None:
        prompts[prompts == pad] = (int(pad) + 1) % m.vocab_size
    tok_times = []
    mf.set_phase("decode")
    if is_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t_dec0 = time.perf_counter()
    ids = prompts.to(device)
    for _ in range(gen_tokens):
        tt = time.perf_counter()
        nxt = gen._next_token(ids)                          # [batch,1]
        ids = torch.cat([ids, nxt], dim=1)
        tok_times.append(time.perf_counter() - tt)
    if is_cuda:
        torch.cuda.synchronize()
    hw_a = torch.cuda.max_memory_allocated() / _MB if is_cuda else 0.0
    hw_r = torch.cuda.max_memory_reserved() / _MB if is_cuda else 0.0
    windows.append(("decode", 0, t_dec0, time.perf_counter(), hw_a, hw_r))
    mf.stop(); optrace.set_hook(None)

    ctx = mf.context_mb()
    rss_hw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    per_phase = {}
    for ph in ("prefill", "decode"):
        wins = [(t0, t1, ha, hr) for p, st, t0, t1, ha, hr in windows if p == ph]
        if wins:
            t0, t1, ha, hr = wins[-1]
            bd = _phase_breakdown(mf, ctx, t0, t1)
            if bd:
                bd["vram"]["hw_total_peak_mb"] = round(ctx + hr, 1)
                per_phase[ph] = bd

    st0 = mf.fast[0][0] if mf.fast else 0.0
    metrics = {
        "run": "segmented_infer_cost", "preset": preset, "device": device,
        "model_config": m.to_dict(), "seg_config": s_to(tr),
        "batch_size": batch, "seq_len": seq, "gen_tokens": gen_tokens, "prompt_len": prompt_len,
        "avg_token_time_s": sum(tok_times) / len(tok_times) if tok_times else 0.0,
        "measurement": "torch inference; synchronized VRAM+CPU-RAM per-op; no-miss high-water",
        "baseline": {"host_framework_mb": round(host_baseline, 1),
                     "host_after_build_mb": round(host_after_build, 1),
                     "cuda_context_mb": round(ctx, 1)},
        "overall": {
            "vram_smi_peak_mb": round(max((v for _, v in mf.slow), default=0.0), 1),
            "vram_hw_total_peak_mb": round(ctx + max((w[5] for w in windows), default=0.0), 1),
            "rss_peak_sampled_mb": round(max((h for _, _, _, h in mf.fast), default=0.0), 1),
            "rss_hw_peak_mb": round(rss_hw, 1),
        },
        "per_phase": per_phase,
        "vram_timeline": [[round(t - st0, 4), round(a, 1), round(r, 1)] for t, a, r, _ in mf.fast],
        "cpu_ram_timeline": [[round(t - st0, 4), round(h, 1)] for t, _, _, h in mf.fast],
        "moves": [[round(t - st0, 4), ph, lbl, round(a, 1), round(r, 1), round(h, 1)]
                  for t, ph, lbl, a, r, h in mf.moves],
    }
    json.dump(metrics, open(out_dir / f"{prefix}_metrics.json", "w"), indent=2)

    o = metrics["overall"]
    print(f"[seg-infer] {preset} on {device}  {m.n_params_estimate/1e6:.0f}M  bs={batch} seq={seq} gen={gen_tokens}")
    print(f"  CUDA context={ctx:.0f}MB  fast-samples={len(mf.fast)}  moves={len(mf.moves)}")
    print(f"  OVERALL VRAM: smi={o['vram_smi_peak_mb']:.0f}  HW(no-miss)={o['vram_hw_total_peak_mb']:.0f}  |  "
          f"RSS sampled={o['rss_peak_sampled_mb']:.0f} HW={o['rss_hw_peak_mb']:.0f} MB")
    for ph in ("prefill", "decode"):
        if ph in per_phase:
            v = per_phase[ph]["vram"]; c = per_phase[ph]["cpu_ram"]
            print(f"  {ph:9} VRAM_total={v['hw_total_peak_mb']:.0f} (ctx {v['context_mb']:.0f} + resident {v['alloc_resident_mb']:.0f} + op {v['alloc_operation_mb']:.0f}) "
                  f"| RSS={c['rss_peak_mb']:.0f} time={per_phase[ph]['time_ms']:.0f}ms")
    print(f"  avg token={metrics['avg_token_time_s']*1000:.0f}ms  -> {out_dir/(prefix+'_metrics.json')}")


def s_to(tr):
    try:
        return tr.s.to_dict()
    except Exception:
        return {"code": getattr(tr.s, "code", "?")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="large_8x2x2x8")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default="seg_b2")
    p.add_argument("--no-empty-cache", action="store_true")
    a = p.parse_args()
    run_infer_cost(a.preset, a.device, a.out_dir, a.prefix, batch=a.batch, seq=a.seq,
                   gen_tokens=a.gen_tokens, prompt_len=a.prompt_len, no_empty_cache=a.no_empty_cache)

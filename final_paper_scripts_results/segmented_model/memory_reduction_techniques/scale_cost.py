#!/usr/bin/env python
"""
Scale cost experiment — memory + time of FULL-MODEL vs SEGMENTED execution as the
model grows (0.84B -> ~3.1B -> ~6.9B params), with the partition FIXED at 8x2x2x8
and ALL memory-reduction techniques ON (the segmented operating point of the paper).

Purpose (paper, scale section): the 0.84B setting is the largest scale at which a
full-model reference still fits the GPU; this experiment extends the cost comparison
to 3B/7B, where full-model TRAINING no longer fits a 40 GB A100 at all. There:
  * GPU  : full training  -> the OOM is CAUGHT and RECORDED as the baseline data point;
           full inference -> measured if it fits; segmented -> measured (MemFlow).
  * CPU  : host RAM is large enough that the full model still runs -> a real measured
           full-vs-segmented ratio at 3B and 7B (the professor-facing comparison).
It also validates the time model  T_step ~ T_compute + N*alpha + V/beta:  the
partition (hence N, the number of load/store events) is FIXED across scales, so
step time should grow ~linearly with the bytes moved / FLOPs (model size), while
segmented peak memory tracks ONE segment, not the model.

Each invocation performs exactly ONE run (one process => clean per-run RSS and
ru_maxrss; no cross-run allocator pollution):

  python scale_cost.py --preset xl3b_8x2x2x8  --device cuda --run seg_train
  python scale_cost.py --preset xxl7b_8x2x2x8 --device cpu  --run full_train

Runs: seg_train | seg_infer | full_train | full_infer
  seg_*  : the verified segmented engine (local copy), tech=None (ALL techniques ON),
           from-scratch init (the full model is never materialized), MemFlow profiler
           -> same JSON schema as the ladder rungs.
  full_* : the project's reference full model (src package, UNCHANGED — imported, not
           copied, exactly as the 0.84B baseline scripts do): standard resident
           weights, full autograd graph, torch.optim AdamW; mirrors
           full_model/scripts/_cost_lib.py (batch 4, seq 512, synthetic non-pad
           batches, warmup + n measured steps; infer = forward-only under no_grad).

The FULL runs mirror the baseline measurement conventions so the 3B/7B numbers are
directly comparable with the published 0.84B ones.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # repo root (for src/ imports, full model)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from seg_cost_lib import run_cost, MemFlow, _synth      # noqa: E402
from config import get_preset                            # noqa: E402

_MB = 1024 ** 2


# --------------------------------------------------------------------------- #
# FULL-model cost (mirrors full_model/scripts/_cost_lib.py; OOM caught+recorded)
# --------------------------------------------------------------------------- #
def _build_full_model(m):
    """Reference GPTDecoder from the UNCHANGED src package (same class the 0.84B
    full-model baseline used)."""
    sys.path.insert(0, str(REPO / "src"))
    from sequential_segmented_llm_training_inference.model.full_model import GPTDecoder
    return GPTDecoder(vocab_size=m.vocab_size, d_model=m.d_model, n_heads=m.n_heads,
                      n_layers=m.n_layers, d_ff=m.d_ff, max_seq_len=m.max_seq_len,
                      dropout=m.dropout)


def run_full_cost(preset: str, device: str, out_dir: Path, prefix: str,
                  mode: str, batch: int = 4, n_steps: int = 3, seq_len=None):
    """mode='train': forward+backward+AdamW step (params+grads+opt+full graph).
       mode='infer': forward-only under no_grad (params+activations+full logits).
    On CUDA OOM the failure is CAUGHT and RECORDED as a data point (oom=true)."""
    is_cuda = str(device).startswith("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = get_preset(preset); m = p["model"]
    seq = seq_len or m.max_seq_len
    dev_total_mb = (torch.cuda.get_device_properties(0).total_memory / _MB
                    if is_cuda and torch.cuda.is_available() else None)

    base = {
        "run": "full_cost_scale", "mode": mode, "preset": preset, "device": device,
        "model_config": m.to_dict(), "params_estimate_million": m.n_params_estimate / 1e6,
        "batch_size": batch, "seq_len": seq, "n_steps": n_steps,
        "device_total_mb": round(dev_total_mb, 1) if dev_total_mb else None,
        "measurement": "reference full model (src GPTDecoder), standard resident execution; "
                       "mirrors full_model/scripts/_cost_lib.py conventions",
    }

    def _fail(stage, err, mf=None):
        rec = dict(base)
        rec["oom"] = True
        rec["oom_stage"] = stage
        rec["error"] = f"{type(err).__name__}: {err}"
        if is_cuda:
            rec["vram_alloc_at_oom_mb"] = round(torch.cuda.memory_allocated() / _MB, 1)
            rec["vram_reserved_at_oom_mb"] = round(torch.cuda.max_memory_reserved() / _MB, 1)
        if mf is not None:
            mf.stop()
            rec["rss_peak_sampled_mb"] = round(max((h for *_, h in mf.fast), default=0.0), 1)
        json.dump(rec, open(out_dir / f"{prefix}_metrics.json", "w"), indent=2)
        print(f"[full-cost] {preset} {mode} on {device}: OOM at '{stage}' — RECORDED "
              f"(device_total={dev_total_mb and round(dev_total_mb)} MB)")
        print(f"  {rec['error'][:200]}")
        print(f"  -> {out_dir/(prefix+'_metrics.json')}")
        return rec

    if is_cuda:
        torch.zeros(1, device=device); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    mf = MemFlow(is_cuda); mf.start()
    mf.set_phase("build")

    try:
        torch.manual_seed(42)
        model = _build_full_model(m).to(device)
        from sequential_segmented_llm_training_inference.model.full_model import (
            causal_lm_cross_entropy_loss as ce_loss)
        if is_cuda:
            torch.cuda.synchronize()
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
        if not _is_oom(e):
            raise
        return _fail("build", e, mf)

    optimizer = None
    if mode == "train":
        optimizer = model.configure_optimizers(learning_rate=3e-4, weight_decay=0.1,
                                               betas=(0.9, 0.95))
        model.train()
    else:
        model.eval()

    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"[full-cost] {preset} {mode} on {device}  params={n_params/1e6:.0f}M  bs={batch} seq={seq}")
    pad = None  # GPT-2: no distinct pad; synthetic non-pad ids (full attention)

    def one_step(seed):
        x, y = _synth(batch, seq, m.vocab_size, pad, device, seed)
        if mode == "infer":
            with torch.no_grad():
                logits, _ = model(x, pad_token_id=pad)
                return ce_loss(logits, y)
        optimizer.zero_grad()
        logits, _ = model(x, pad_token_id=pad)
        loss = ce_loss(logits, y)
        loss.backward()
        optimizer.step()
        return loss

    step_times, hw_windows = [], []
    try:
        mf.set_phase("warmup")
        one_step(0)
        if is_cuda:
            torch.cuda.synchronize()
        for step in range(n_steps):
            mf.set_phase(f"step{step}")
            if is_cuda:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            one_step(1 + step)
            if is_cuda:
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            hw_windows.append((torch.cuda.max_memory_allocated() / _MB if is_cuda else 0.0,
                               torch.cuda.max_memory_reserved() / _MB if is_cuda else 0.0))
            gc.collect()
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
        if not _is_oom(e):
            raise
        return _fail("warmup" if not step_times else f"step{len(step_times)}", e, mf)

    mf.stop()
    ctx = mf.context_mb()
    hw_r = max((r for _, r in hw_windows), default=0.0)
    metrics = dict(base)
    metrics.update({
        "oom": False,
        "params_million": round(n_params / 1e6, 1),
        "avg_step_time_s": sum(step_times) / len(step_times),
        "step_times_s": [round(t, 3) for t in step_times],
        "overall": {
            "cuda_context_mb": round(ctx, 1),
            "vram_alloc_peak_mb": round(max((a for _, a, _, _ in mf.fast), default=0.0), 1),
            "vram_reserved_peak_mb": round(max((r for _, _, r, _ in mf.fast), default=0.0), 1),
            "vram_hw_reserved_peak_mb": round(hw_r, 1),
            "vram_hw_total_peak_mb": round(ctx + hw_r, 1),
            "rss_peak_sampled_mb": round(max((h for *_, h in mf.fast), default=0.0), 1),
        },
    })
    json.dump(metrics, open(out_dir / f"{prefix}_metrics.json", "w"), indent=2)
    o = metrics["overall"]
    print(f"  avg_step={metrics['avg_step_time_s']:.1f}s  "
          f"VRAM hw_total={o['vram_hw_total_peak_mb']:.0f} MB  RSS={o['rss_peak_sampled_mb']:.0f} MB")
    print(f"  -> {out_dir/(prefix+'_metrics.json')}")
    return metrics


def _is_oom(e: BaseException) -> bool:
    if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    return "out of memory" in str(e).lower() or "can't allocate" in str(e).lower()


# --------------------------------------------------------------------------- #
# main — exactly one run per process
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True,
                    help="xl3b_8x2x2x8 | xxl7b_8x2x2x8 (or any preset for smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run", required=True,
                    choices=["seg_train", "seg_infer", "full_train", "full_infer"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: results/scale_{gpu|cpu}_{preset}/")
    a = ap.parse_args()

    tag = "gpu" if a.device.startswith("cuda") else "cpu"
    out_dir = a.out_dir or (HERE / "results" / f"scale_{tag}_{a.preset.split('_')[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.run == "seg_train":
        # tech=None => ALL memory-reduction techniques ON (the paper's operating point)
        run_cost(a.preset, a.device, out_dir, prefix="seg_train", batch=a.batch,
                 n_steps=a.n_steps, seq_len=a.seq_len, from_scratch=True, tech=None)
    elif a.run == "seg_infer":
        from infer_cost_lib import run_infer_cost
        run_infer_cost(a.preset, a.device, out_dir, prefix="seg_infer",
                       prompt_len=a.prompt_len, gen_tokens=a.gen_tokens,
                       tech=None, from_scratch=True)
    elif a.run == "full_train":
        run_full_cost(a.preset, a.device, out_dir, prefix="full_train", mode="train",
                      batch=a.batch, n_steps=a.n_steps, seq_len=a.seq_len)
    else:
        run_full_cost(a.preset, a.device, out_dir, prefix="full_infer", mode="infer",
                      batch=a.batch, n_steps=a.n_steps, seq_len=a.seq_len)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

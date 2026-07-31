#!/usr/bin/env python
"""
DeepSpeed ZeRO-Offload baseline (referee comparison) — the SAME 0.84B reference model,
protocol, and measurement conventions as the paper's cost runs (fp32, AdamW, batch 4,
seq 512, warmup + measured steps), trained under DeepSpeed's offload engines on one GPU:

  --mode zero2_offload : ZeRO stage 2, optimizer state offloaded to CPU
                         (classic ZeRO-Offload: params + grads stay on device)
  --mode zero3_offload : ZeRO stage 3, parameters AND optimizer state offloaded
                         (ZeRO-Infinity-style on a single GPU)

Fair-comparison notes:
  * identical model (src GPTDecoder), identical synthetic batches, fp32 throughout
    (fp16/bf16 disabled) — the paper's protocol;
  * torch.optim.AdamW as the client optimizer with zero_force_ds_cpu_optimizer=false,
    so no JIT-compiled DeepSpeed ops are required (portable; DeepSpeedCPUAdam would
    make the offloaded optimizer step faster but does not change peak device memory,
    which is what this comparison measures);
  * DeepSpeed does not retain-free activations: the full autograd graph stays on
    device (no activation checkpointing configured — same as the paper's full-model
    baseline), so the expected device floor is params(+grads) + activations for
    stage 2, activations + per-layer gathered params for stage 3.

Runs standalone in a separate venv (torch + deepspeed + psutil only); does NOT import
the segmented engine. Output JSON mirrors the scale runs' fields.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # repo root (mrt -> segmented_model -> f_p_s_r -> repo)
sys.path.insert(0, str(REPO / "src"))
from sequential_segmented_llm_training_inference.model.full_model import (  # noqa: E402
    GPTDecoder, causal_lm_cross_entropy_loss)

_MB = 1024 ** 2

LARGE = dict(vocab_size=50257, max_seq_len=512, n_layers=36, d_model=1280,
             n_heads=20, d_ff=5120, dropout=0.1)


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def _smi_mb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        me = str(os.getpid())
        for line in out.splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if pid == me:
                return float(mem)
    except Exception:
        pass
    return 0.0


class Sampler:
    """Background peak sampler: torch alloc/reserved + RSS (fast) and smi (slow)."""

    def __init__(self):
        self.stop_ev = threading.Event()
        self.rss_peak = 0.0
        self.smi_peak = 0.0

    def _fast(self):
        while not self.stop_ev.is_set():
            self.rss_peak = max(self.rss_peak, _rss_mb())
            self.stop_ev.wait(0.02)

    def _slow(self):
        while not self.stop_ev.is_set():
            self.smi_peak = max(self.smi_peak, _smi_mb())
            self.stop_ev.wait(0.25)

    def start(self):
        for fn in (self._fast, self._slow):
            threading.Thread(target=fn, daemon=True).start()
        return self

    def stop(self):
        self.stop_ev.set()


def _synth(batch, seq, vocab, device, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, seq), generator=g)
    lab = ids.clone(); lab[:, 0] = -100
    return ids.to(device), lab.to(device)


def ds_config(mode: str, batch: int) -> dict:
    zero = {"stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True}}
    if mode == "zero3_offload":
        zero = {"stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
                "stage3_max_live_parameters": 1e8,
                "stage3_prefetch_bucket_size": 5e7}
    return {"train_batch_size": batch,
            "train_micro_batch_size_per_gpu": batch,
            "gradient_accumulation_steps": 1,
            "zero_optimization": zero,
            "zero_force_ds_cpu_optimizer": False,
            "fp16": {"enabled": False}, "bf16": {"enabled": False},
            "wall_clock_breakdown": False,
            "steps_per_print": 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["zero2_offload", "zero3_offload"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out-dir", type=Path,
                    default=HERE / "results" / "deepspeed_baseline")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # single-process "distributed" setup
    os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    import deepspeed

    device = "cuda"
    torch.zeros(1, device=device); torch.cuda.synchronize()
    ctx0 = _smi_mb() - torch.cuda.memory_reserved() / _MB   # CUDA context estimate
    dev_total = torch.cuda.get_device_properties(0).total_memory / _MB
    sampler = Sampler().start()

    base = {"run": "deepspeed_baseline", "mode": a.mode, "device": device,
            "model_config": LARGE, "batch_size": a.batch, "seq_len": a.seq_len,
            "n_steps": a.n_steps, "fp32": True,
            "deepspeed_version": deepspeed.__version__,
            "torch_version": torch.__version__,
            "device_total_mb": round(dev_total, 1),
            "optimizer": "torch.optim.AdamW (zero_force_ds_cpu_optimizer=false)",
            "measurement": "same conventions as the scale runs: no-miss allocator "
                           "high-water + CUDA context; sampled RSS/smi peaks"}

    def _fail(stage, err):
        rec = dict(base); rec["oom"] = True; rec["oom_stage"] = stage
        rec["error"] = f"{type(err).__name__}: {err}"
        rec["vram_alloc_at_oom_mb"] = round(torch.cuda.memory_allocated() / _MB, 1)
        sampler.stop()
        json.dump(rec, open(a.out_dir / f"{a.mode}_metrics.json", "w"), indent=2)
        print(f"[ds-cost] {a.mode}: OOM at '{stage}' — RECORDED")
        return rec

    torch.manual_seed(42)
    try:
        model = GPTDecoder(**{k: v for k, v in LARGE.items()})
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                weight_decay=0.1)
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt,
                                               config=ds_config(a.mode, a.batch))
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return _fail("build", e)

    n_params = sum(p.numel() for p in model.parameters())
    # under ZeRO-3 params are partitioned; count from the config instead
    if n_params < 1e6:
        n_params = (2 * LARGE["vocab_size"] * LARGE["d_model"]
                    + LARGE["max_seq_len"] * LARGE["d_model"]
                    + LARGE["n_layers"] * (4 * LARGE["d_model"] ** 2
                                           + 2 * LARGE["d_model"] * LARGE["d_ff"]))
    print(f"[ds-cost] {a.mode}  params={n_params/1e6:.0f}M  bs={a.batch} seq={a.seq_len}  "
          f"ds={deepspeed.__version__}")

    def one_step(seed):
        x, y = _synth(a.batch, a.seq_len, LARGE["vocab_size"], device, seed)
        logits, _ = engine(x, pad_token_id=None)
        loss = causal_lm_cross_entropy_loss(logits, y)
        engine.backward(loss)
        engine.step()
        return loss

    step_times, hw = [], []
    try:
        one_step(0); torch.cuda.synchronize()                     # warmup
        for step in range(a.n_steps):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            one_step(1 + step)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            hw.append(torch.cuda.max_memory_reserved() / _MB)
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
        if "out of memory" not in str(e).lower():
            raise
        return _fail("warmup" if not step_times else f"step{len(step_times)}", e)

    sampler.stop()
    metrics = dict(base)
    metrics.update({
        "oom": False, "params_million": round(n_params / 1e6, 1),
        "avg_step_time_s": sum(step_times) / len(step_times),
        "step_times_s": [round(t, 3) for t in step_times],
        "overall": {
            "cuda_context_mb": round(max(0.0, ctx0), 1),
            "vram_hw_reserved_peak_mb": round(max(hw), 1),
            "vram_hw_total_peak_mb": round(max(0.0, ctx0) + max(hw), 1),
            "vram_smi_peak_mb": round(sampler.smi_peak, 1),
            "rss_peak_mb": round(sampler.rss_peak, 1),
        }})
    json.dump(metrics, open(a.out_dir / f"{a.mode}_metrics.json", "w"), indent=2)
    o = metrics["overall"]
    print(f"  avg_step={metrics['avg_step_time_s']:.2f}s  "
          f"VRAM hw_total={o['vram_hw_total_peak_mb']:.0f} MB (smi {o['vram_smi_peak_mb']:.0f})  "
          f"RSS={o['rss_peak_mb']:.0f} MB")
    print(f"  -> {a.out_dir/(a.mode + '_metrics.json')}")
    return metrics


if __name__ == "__main__":
    main()

"""
Shared A2 cost-training logic (CPU + GPU; light timing + memory profiler).

A2 measures how the model uses resources during a FEW training steps. It uses the
LARGE model (GPT-2 vocab, ~838M params) — NOT the small accuracy model — so the
model's own memory dominates the fixed ~500MB CUDA context and full-vs-segmented
differences are clearly measurable (see plan §2). Batch 4, n_steps 3.

To get a stable worst-case memory reading (the real log data is short), cost runs
feed FIXED full-length synthetic batches: random non-pad token ids of shape
(batch_size, max_seq_len) — full causal attention, no padding.

Two flavors, one code path (selected by --profile-memory):
  * light      (training_cost_*)  : clean per-step timing + throughput, NO memory
                                    sampling (so timing isn't perturbed).
  * profiler   (cost_profiler_*)  : timing PLUS detailed memory per plan §5.4:
        CPU runs -> host CPU RAM (RSS): peak + timeline.
        GPU runs -> 4 VRAM categories (torch alloc / peak-alloc / reserved /
                    nvidia-smi process VRAM) AND host CPU RAM, all separate, each
                    with a peak and a continuous TIMELINE over the few steps.

Outputs (in --out-dir):
  <prefix>_metrics.json   (config, timing, throughput, memory peaks + timelines)
  <prefix>_trace.csv      (per-step timing/loss + per-step memory if profiling)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

import _common as C
from sequential_segmented_llm_training_inference.model.full_model import (
    causal_lm_cross_entropy_loss,
)


def add_cost_args(p: argparse.ArgumentParser, default_device: str,
                  default_out: Path, profile_memory: bool) -> None:
    p.add_argument("--device", default=default_device)
    p.add_argument("--batch-size", type=int, default=C.COST_BATCH_SIZE)  # 4
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=None,
                   help="synthetic sequence length; default = LARGE max_seq_len (512)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--profile-memory", dest="profile_memory",
                   action="store_true", default=profile_memory)
    p.add_argument("--no-profile-memory", dest="profile_memory",
                   action="store_false")
    p.add_argument("--poll-interval", type=float, default=0.02)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--prefix", default=None,
                   help="output file prefix; defaults based on profile mode")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synth_batch(batch_size, seq_len, vocab_size, pad_id, device, seed):
    """Full-length synthetic batch of random NON-pad ids (worst-case memory)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    if pad_id is not None:
        ids[ids == pad_id] = (int(pad_id) + 1) % vocab_size  # avoid padding -> full attention
    labels = ids.clone()
    labels[:, 0] = -100  # first position has no target
    return ids.to(device), labels.to(device)


def run_cost(args: argparse.Namespace, run_name: str,
             config=None, mode: str = "train") -> None:
    """mode='train' -> A2: forward+backward+optimizer (params+grads+opt+acts).
       mode='infer' -> B2: forward only under no_grad (params+acts; no grads/opt)."""
    assert mode in ("train", "infer")
    config = config if config is not None else C.LARGE_CONFIG
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    is_cuda = device.type == "cuda"

    if args.prefix:
        prefix = args.prefix
    elif mode == "infer":
        prefix = "infer_cost_profiler" if args.profile_memory else "infer_cost"
    else:
        prefix = "cost_profiler" if args.profile_memory else "training_cost"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = args.seq_len or config["max_seq_len"]

    print(f"[1] {run_name} (LARGE model, mode={mode}) on {device}  bs={args.batch_size}  "
          f"seq_len={seq_len}  n_steps={args.n_steps}  profile_memory={args.profile_memory}")

    # --- baseline 1: framework (python + torch already imported at module load) ---
    host_baseline_mb = C._process_rss_mb()
    tokenizer = C.build_tokenizer(config)               # loads transformers + tokenizer
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    host_after_tok_mb = C._process_rss_mb()

    # start continuous samplers now so the timeline captures context -> model -> steps
    cpu_sampler = gpu_sampler = None
    if args.profile_memory:
        cpu_sampler = C.CpuRamSampler(interval_s=args.poll_interval).start()
        if is_cuda:
            gpu_sampler = C.GpuVramSampler(interval_s=args.poll_interval).start()

    # --- baseline 2: CUDA context (touch the GPU before building the model) ---
    vram_context_mb = 0.0
    if is_cuda:
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)      # torch peak now = model+data only
        vram_context_mb = C._nvidia_smi_process_vram_mb()

    # --- build model (adds params), then capture baseline 3 ---
    model = C.build_model(vocab_size, config).to(device)
    if is_cuda:
        torch.cuda.synchronize(device)
    host_after_model_mb = C._process_rss_mb()
    vram_after_model_mb = C._nvidia_smi_process_vram_mb() if is_cuda else 0.0
    torch_alloc_model_mb = C.torch_vram_mb(device)["torch_alloc_mb"]

    optimizer = None
    if mode == "train":
        optimizer = model.configure_optimizers(
            learning_rate=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
        )
    print(f"    params={C.count_params(model)/1e6:.1f}M  vocab={vocab_size}  "
          f"d_model={config['d_model']}  n_layers={config['n_layers']}")

    model.train() if mode == "train" else model.eval()
    trace_rows = []
    step_times = []

    def one_step(seed):
        ii, lab = _synth_batch(args.batch_size, seq_len, vocab_size, pad_token_id, device, seed)
        if mode == "infer":
            with torch.no_grad():
                logits, _ = model(ii, pad_token_id=pad_token_id)
                return causal_lm_cross_entropy_loss(logits, lab)
        optimizer.zero_grad()
        logits, _ = model(ii, pad_token_id=pad_token_id)
        loss = causal_lm_cross_entropy_loss(logits, lab)
        loss.backward()
        optimizer.step()
        return loss

    # one warmup step (not measured) to stabilize allocator/timing
    one_step(args.seed)
    _sync(device)

    for step in range(args.n_steps):
        _sync(device)
        t0 = time.perf_counter()
        loss = one_step(args.seed + 1 + step)
        _sync(device)
        step_time = time.perf_counter() - t0
        step_times.append(step_time)

        row = {"step": step, "step_time_s": round(step_time, 6),
               "loss": round(loss.item(), 6)}
        if args.profile_memory:
            row["host_rss_mb"] = round(cpu_sampler.current_mb(), 1)
            if is_cuda:
                tv = C.torch_vram_mb(device)
                row["torch_alloc_mb"] = round(tv["torch_alloc_mb"], 1)
                row["torch_reserved_mb"] = round(tv["torch_reserved_mb"], 1)
                row["torch_peak_alloc_mb"] = round(tv["torch_peak_alloc_mb"], 1)
                row["nvidia_smi_vram_mb"] = round(C._nvidia_smi_process_vram_mb(), 1)
        trace_rows.append(row)

    # stop samplers, collect peaks + timelines + the framework/model decomposition
    memory = {}
    if args.profile_memory:
        host_peak = cpu_sampler.stop()
        # baseline = framework (python+torch+transformers) loaded, before the model.
        # net = peak - baseline = the part bound to MODEL + DATA (params/grads/opt/acts).
        memory["baseline"] = {
            "_comment": "memory loaded BEFORE the model (frameworks + CUDA context); "
                        "subtract from peak to get the model+data cost",
            "host_framework_mb": round(host_baseline_mb, 1),
            "host_after_tokenizer_mb": round(host_after_tok_mb, 1),
            "host_after_model_mb": round(host_after_model_mb, 1),
        }
        memory["host_cpu_ram"] = {
            "peak_mb": round(host_peak, 1),
            "framework_baseline_mb": round(host_baseline_mb, 1),
            "net_model_data_mb": round(host_peak - host_baseline_mb, 1),
            "timeline": [[round(t, 4), round(mb, 1)] for t, mb in cpu_sampler.timeline],
        }
        if is_cuda:
            smi_peak = gpu_sampler.stop()
            memory["baseline"]["vram_cuda_context_mb"] = round(vram_context_mb, 1)
            memory["baseline"]["vram_after_model_mb"] = round(vram_after_model_mb, 1)
            memory["baseline"]["vram_model_params_mb"] = round(torch_alloc_model_mb, 1)
            memory["vram"] = {
                # torch-allocated EXCLUDES the CUDA context, so torch_alloc_peak IS
                # essentially the net model+data torch VRAM already.
                "torch_alloc_peak_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
                "torch_reserved_peak_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
                "nvidia_smi_peak_mb": round(smi_peak, 1),
                "cuda_context_mb": round(vram_context_mb, 1),
                "nvidia_smi_net_model_data_mb": round(smi_peak - vram_context_mb, 1),
                "nvidia_smi_timeline": [[round(t, 4), round(mb, 1)] for t, mb in gpu_sampler.timeline],
            }

    measured = step_times[1:] if len(step_times) > 1 else step_times
    avg_step = sum(measured) / len(measured) if measured else float("nan")
    throughput = args.batch_size / avg_step if avg_step and avg_step > 0 else float("nan")

    metrics = {
        "run": run_name,
        "mode": mode,
        "model": "large",
        "device": str(device),
        "model_config": config,
        "vocab_size": vocab_size,
        "params_million": C.count_params(model) / 1e6,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "n_steps": args.n_steps,
        "synthetic_full_length_batches": True,
        "profile_memory": args.profile_memory,
        "step_times_s": [round(t, 6) for t in step_times],
        "avg_step_time_s": avg_step,
        "throughput_samples_per_s": throughput,
        "memory": memory,
    }
    with open(out_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if trace_rows:
        fields = list(trace_rows[0].keys())
        with open(out_dir / f"{prefix}_trace.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(trace_rows)

    print(f"[done] params={C.count_params(model)/1e6:.0f}M  avg_step={avg_step*1000:.1f}ms  "
          f"throughput={throughput:.1f} samp/s")
    if args.profile_memory:
        h = memory["host_cpu_ram"]
        if is_cuda:
            v = memory["vram"]
            print(f"       VRAM: peak(nvidia-smi)={v['nvidia_smi_peak_mb']}MB "
                  f"= CUDA-context {v['cuda_context_mb']}MB + model+data "
                  f"{v['nvidia_smi_net_model_data_mb']}MB  (torch_alloc={v['torch_alloc_peak_mb']}MB)")
            print(f"       host RAM: peak={h['peak_mb']}MB = framework "
                  f"{h['framework_baseline_mb']}MB + model+data {h['net_model_data_mb']}MB")
        else:
            print(f"       host RAM: peak={h['peak_mb']}MB = framework "
                  f"{h['framework_baseline_mb']}MB + model+data {h['net_model_data_mb']}MB  "
                  f"(timeline pts={len(h['timeline'])})")
    print(f"       -> {out_dir/(prefix + '_metrics.json')}")

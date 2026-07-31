#!/usr/bin/env python3
"""
Autoregressive inference using the full-model export produced by FullModelExporter.

The exported model (full_model.pt) is a standard dense transformer reassembled
from segmented checkpoints.  Memory footprint is identical to the normal full
model — all weights are always resident — making this the right comparison point
for confirming that segmented training and inference produce the same model at
the same memory cost as conventional training.

Usage:
    python scripts/infer_exported_model.py \
        --export-pt experiments/outputs_demo/exports/segmented_default_after_full_backward/full_model/full_model.pt \
        --evaluate-test --num-test-examples 50
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

_PAPER_REPORT_DIR = REPO_ROOT / "paper_report_codes_for_segmented_gpt"
if _PAPER_REPORT_DIR.exists() and str(_PAPER_REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_PAPER_REPORT_DIR))

try:
    from paper_report.inference_metrics import InferenceMetricTracker, write_inference_summary_csv
    _HAS_PAPER_REPORT = True
except ImportError:
    _HAS_PAPER_REPORT = False

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.model.embeddings import (
    EmbeddingConfig,
    TokenPositionEmbeddings,
)
from sequential_segmented_llm_training_inference.model.layer_norms import (
    AttentionOutputProjections,
    LayerComponentConfig,
    MLPSharedOutputBiases,
    SegmentedLayerNorms,
)
from sequential_segmented_llm_training_inference.model.output_head import (
    FinalNormLMHead,
    OutputHeadConfig,
)

TOKENIZER_DIR = REPO_ROOT / "gpt2_tokenizer"
DATA_DIR = REPO_ROOT / "log_lines" / "generative_splits"

try:
    import psutil as _psutil
except Exception:
    _psutil = None  # type: ignore


# ---------------------------------------------------------------------------
# Assembled full model (same architecture as FullModelExporter output)
# ---------------------------------------------------------------------------

class _AssembledAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: Tensor, mask: Optional[Tensor]) -> Tensor:
        B, T, _ = hidden.shape
        q = self._split(self.q_proj(hidden), B, T)
        k = self._split(self.k_proj(hidden), B, T)
        v = self._split(self.v_proj(hidden), B, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(T, T, dtype=torch.bool, device=hidden.device).tril()
        scores = scores.masked_fill(~causal.view(1, 1, T, T), float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(~mask.bool().view(B, 1, 1, T), float("-inf"))
        ctx = torch.matmul(torch.softmax(scores, dim=-1), v)
        return ctx.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)

    def _split(self, x: Tensor, B: int, T: int) -> Tensor:
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)


class _AssembledMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AssembledFullModel(nn.Module):
    """Loads the state dict produced by FullModelExporter and runs full-model inference."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        emb_cfg = EmbeddingConfig(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model,
            max_seq_len=cfg.max_seq_len, dropout=0.0,
            pad_token_id=cfg.pad_token_id,
        )
        lc_cfg = LayerComponentConfig(
            n_layers=cfg.n_layers, d_model=cfg.d_model, use_mlp_shared_output_bias=True,
        )
        out_cfg = OutputHeadConfig(d_model=cfg.d_model, vocab_size=cfg.vocab_size)
        self.embeddings = TokenPositionEmbeddings(emb_cfg)
        self.layer_norms = SegmentedLayerNorms(lc_cfg)
        self.attention_output_projections = AttentionOutputProjections(lc_cfg)
        self.mlp_shared_output_biases = MLPSharedOutputBiases(lc_cfg)
        self.output_head = FinalNormLMHead(out_cfg)
        self.layers = nn.ModuleList(
            [_AssembledLayer(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)]
        )

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        h = self.embeddings(input_ids)
        for i, layer in enumerate(self.layers):
            attn_in = self.layer_norms.attention(i, h)
            attn_out = layer.attention(attn_in, attention_mask)
            attn_out = self.attention_output_projections(i, attn_out)
            h = h + attn_out
            mlp_in = self.layer_norms.mlp(i, h)
            mlp_out = layer.mlp(mlp_in)
            mlp_out = self.mlp_shared_output_biases(i, mlp_out)
            h = h + mlp_out
        return self.output_head(h)


class _AssembledLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attention = _AssembledAttention(d_model, n_heads)
        self.mlp = _AssembledMLP(d_model, d_ff)


# ---------------------------------------------------------------------------
# Memory monitoring (same pattern as infer_segmented.py)
# ---------------------------------------------------------------------------

def _gpu_mb_for_pid(pid: int) -> float | None:
    import shutil, subprocess
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
        )
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                try:
                    if int(parts[0]) == pid:
                        return float(parts[1])
                except ValueError:
                    pass
    except Exception:
        pass
    return None


def _monitor(pid: int, stop: Event, interval: float,
             timeseries: list[tuple[float, float, float | None]], t0: float) -> None:
    if _psutil is None:
        return
    try:
        root = _psutil.Process(pid)
    except Exception:
        return
    while not stop.is_set():
        try:
            rss = sum(p.memory_info().rss for p in [root] + root.children(recursive=True))
            cpu_mb = rss / 1024 ** 2
        except Exception:
            cpu_mb = 0.0
        gpu_mb = _gpu_mb_for_pid(pid)
        timeseries.append((time.perf_counter() - t0, cpu_mb, gpu_mb))
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(
    model: AssembledFullModel,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> Tensor:
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if (next_token == eos_token_id).all():
            break
    return generated[:, input_ids.shape[1]:]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference on a FullModelExporter-produced full_model.pt.")
    p.add_argument("--export-pt", type=Path, required=True,
                   help="Path to full_model.pt produced by FullModelExporter.")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--num-test-examples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--trace-output", type=Path, default=None,
                   help="Optional path to write memory_trace.csv.")
    p.add_argument("--summary-output", type=Path, default=None,
                   help="Optional path to write canonical inference_summary.csv (requires paper_report).")
    p.add_argument("--run-name", default="exported_full_model_inference",
                   help="Run name recorded in inference_summary.csv.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("Sequential Segmented LLM — Exported Full Model Inference")
    print("=" * 60)
    print(f"  Device    : {device}")
    print(f"  Export    : {args.export_pt}")

    print("\n[1] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[2] Loading exported full model ...")
    artifact = torch.load(args.export_pt, map_location=device, weights_only=False)
    model_config = ModelConfig(**artifact["model_config"])
    model = AssembledFullModel(model_config)
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_config.n_layers}L × {model_config.d_model}d  ({n_params:,} params)")

    # Start memory monitor
    timeseries: list[tuple[float, float, float | None]] = []
    stop_evt = Event()
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    monitor_thread = Thread(
        target=_monitor, args=(os.getpid(), stop_evt, 0.2, timeseries, t0), daemon=True,
    )
    monitor_thread.start()

    print("\n[3] Running inference ...")

    tracker = None
    if _HAS_PAPER_REPORT and args.summary_output is not None and args.evaluate_test:
        tracker = InferenceMetricTracker(
            run_name=args.run_name,
            method="exported_full",
            device=str(device),
            storage_backend="resident",
        )

    if args.evaluate_test:
        exact_matches = total = 0
        test_csv = DATA_DIR / "test.csv"
        if not test_csv.exists():
            print(f"  test.csv not found at {test_csv}")
        else:
            with test_csv.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if total >= args.num_test_examples:
                        break
                    log_line = row.get("log_line", "").strip()
                    true_label = row.get("label", "").strip()
                    if not log_line or not true_label:
                        continue
                    # Must match the TRAINING prompt format ("{log_line}\nLabel:")
                    # used by the collator and the segmented/full inference paths.
                    # Without the "\nLabel:" cue the model runs out-of-distribution.
                    prompt = f"{log_line}\nLabel:"
                    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=model_config.max_seq_len)
                    input_ids = enc["input_ids"].to(device)
                    t0_gen = time.perf_counter()
                    out_ids = greedy_generate(model, input_ids, args.max_new_tokens,
                                              tokenizer.eos_token_id)
                    elapsed_ms = (time.perf_counter() - t0_gen) * 1000
                    pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
                    match = pred == true_label
                    if match:
                        exact_matches += 1
                    total += 1
                    if tracker is not None:
                        tracker.record_generation(
                            elapsed_ms=elapsed_ms,
                            exact_match=match,
                            generated_tokens=out_ids.shape[1],
                        )
                    status = "✓" if match else "✗"
                    snippet = log_line[:60] + "..." if len(log_line) > 60 else log_line
                    print(f"[{status}] LOG : {snippet}")
                    print(f"     TRUE: {true_label}")
                    print(f"     PRED: {pred}")
                    print()
            acc = exact_matches / total if total > 0 else 0.0
            print(f"Exact-match accuracy: {exact_matches}/{total} = {acc:.1%}")

    elif args.prompt is not None:
        enc = tokenizer(args.prompt, return_tensors="pt", truncation=True,
                        max_length=model_config.max_seq_len)
        input_ids = enc["input_ids"].to(device)
        out_ids = greedy_generate(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id)
        print(f"Generated: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")

    else:
        print("No --prompt or --evaluate-test given. Nothing to generate.")

    stop_evt.set()
    monitor_thread.join(timeout=2)

    # Report memory stats
    if timeseries:
        peak_cpu = max(s[1] for s in timeseries)
        gpu_vals = [s[2] for s in timeseries if s[2] is not None]
        peak_gpu = max(gpu_vals) if gpu_vals else None
        print("\n--- Memory stats (inference) ---")
        print(f"  Peak CPU RAM : {peak_cpu:.1f} MB")
        if peak_gpu is not None:
            print(f"  Peak GPU VRAM (nvidia-smi) : {peak_gpu:.1f} MB")
        if device.type == "cuda":
            alloc_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print(f"  Peak GPU allocated (torch) : {alloc_mb:.1f} MB")

    if args.trace_output is not None and timeseries:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_output.open("w", encoding="utf-8") as f:
            f.write("elapsed_s,cpu_mb,gpu_mb\n")
            for elapsed, cpu_mb, gpu_mb in timeseries:
                gpu_str = f"{gpu_mb:.2f}" if gpu_mb is not None else ""
                f.write(f"{elapsed:.3f},{cpu_mb:.2f},{gpu_str}\n")
        print(f"  Memory trace : {args.trace_output}")

    if tracker is not None and args.summary_output is not None:
        peak_cpu_s = max(s[1] for s in timeseries) if timeseries else None
        gpu_vals_s = [s[2] for s in timeseries if s[2] is not None]
        peak_gpu_s = max(gpu_vals_s) if gpu_vals_s else None
        infer_summary = tracker.summary(peak_cpu_ram_mb=peak_cpu_s, peak_gpu_vram_mb=peak_gpu_s)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        write_inference_summary_csv(args.summary_output, infer_summary)
        print(f"  Inference summary : {args.summary_output}")


if __name__ == "__main__":
    main()

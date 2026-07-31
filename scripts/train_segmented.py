#!/usr/bin/env python3
"""
End-to-end sequential segmented training on log-line-to-label generation.

Implements the full pipeline from sequentia_training_and_inference_of_segmented_model.md:
  - Strict single-segment execution (only one segment on compute device at a time)
  - Both update styles: after_full_backward (default) and immediate_segment_update
  - Recomputation-based segmented backward with RNG state capture/restore for dropout
  - All three optimizer policies: stateless_sgd, segmentwise_sgd_momentum, segmentwise_adamw
  - Both storage backends: cpu_ram_offload (GPU→CPU) and disk_streaming (CPU/GPU→disk)
  - Gradient accumulation (only with after_full_backward)
  - Optional gradient clipping
  - LR scheduler: cosine with linear warmup (default), linear, or none
  - Segmented validation each epoch + best-checkpoint selection
  - Final segmented test on best checkpoint
  - Full model export from trained segments (default: enabled)
  - YAML protocol export for reproducibility (default: enabled)
  - Checkpoint resume support

Architecture defaults (matches full_model.GPTDecoder):
    vocab_size   = 50257   (GPT-2)
    d_model      = 512
    n_heads      = 8
    n_layers     = 6
    d_ff         = 2048
    max_seq_len  = 256
    dropout      = 0.1

Segmentation defaults:
    attention_segments = 4  →  2 heads per segment
    mlp_chunks         = 4  →  512 hidden units per segment

Data:
    log_lines/generative_splits/{train,validation,test}.csv

Tokenizer:
    gpt2_tokenizer/  (GPT-2 BPE, vocab 50257)

Usage:
    # Smoke test (tiny model, few steps)
    python scripts/train_segmented.py \\
        --epochs 1 --batch-size 2 --n-layers 2 --d-model 128 --n-heads 4 --d-ff 256 \\
        --attention-segments 2 --mlp-chunks 2 --device cpu \\
        --max-train-steps 3 --max-val-steps 3

    # Full training (GPU)
    python scripts/train_segmented.py --epochs 5 --batch-size 4 --device cuda

    # Immediate segment update style
    python scripts/train_segmented.py --update-style immediate_segment_update

    # Resume from last checkpoint
    python scripts/train_segmented.py --resume last
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.optimizer_config import normalize_optimizer_type
from sequential_segmented_llm_training_inference.config.segmentation_config import SegmentationConfig
from sequential_segmented_llm_training_inference.data.collator import LogGenerationCollator
from sequential_segmented_llm_training_inference.data.dataset import LogLabelDataset
from sequential_segmented_llm_training_inference.execution.backward_engine import (
    RecomputationCheckConfig,
    SegmentedBackwardEngine,
)
from sequential_segmented_llm_training_inference.execution.composition import (
    concatenate_attention_outputs,
    residual_add,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import SegmentedForwardEngine
from sequential_segmented_llm_training_inference.execution.rng import capture_rng_state, restore_rng_state
from sequential_segmented_llm_training_inference.execution.segment_loader import StrictSegmentLoader
from sequential_segmented_llm_training_inference.optimization.segmentwise_adamw import SegmentwiseAdamW
from sequential_segmented_llm_training_inference.optimization.segmentwise_sgd import SegmentwiseSGD
from sequential_segmented_llm_training_inference.model.embeddings import EmbeddingConfig
from sequential_segmented_llm_training_inference.model.output_head import OutputHeadConfig
from sequential_segmented_llm_training_inference.segments.global_segments import (
    AttentionOutputProjSegmentModule,
    EmbeddingSegmentModule,
    EmbeddingSliceSegmentModule,
    OutputHeadSegmentModule,
    OutputHeadSliceSegmentModule,
)
from sequential_segmented_llm_training_inference.segments.segment_factory import SegmentFactory
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.checkpoint_store import (
    SegmentedCheckpoint,
    SegmentedCheckpointStore,
)
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import CpuRamSegmentStore
from sequential_segmented_llm_training_inference.storage.disk_segment_store import DiskSegmentStore
from sequential_segmented_llm_training_inference.training.losses import CausalCrossEntropyLoss
from sequential_segmented_llm_training_inference.training.metrics import perplexity, token_accuracy
from sequential_segmented_llm_training_inference.training.profiler import SegmentedTrainingProfiler
from sequential_segmented_llm_training_inference.training.trainer import SegmentedTrainer, SegmentTrainBatch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TOKENIZER_DIR = REPO_ROOT / "gpt2_tokenizer"
DATA_DIR = REPO_ROOT / "log_lines" / "generative_splits"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
SEGMENT_CACHE_DIR = REPO_ROOT / "segment_cache"
EXPORT_DIR = REPO_ROOT / "exports"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential segmented GPT-decoder training (full spec implementation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model architecture
    g = p.add_argument_group("Model architecture")
    g.add_argument("--d-model", type=int, default=512)
    g.add_argument("--n-heads", type=int, default=8)
    g.add_argument("--n-layers", type=int, default=6)
    g.add_argument("--d-ff", type=int, default=2048)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--max-seq-len", type=int, default=256)

    # Segmentation
    g = p.add_argument_group("Segmentation")
    g.add_argument("--attention-segments", type=int, default=4,
                   help="Number of attention head-group segments (must be > 1; n_heads must be divisible).")
    g.add_argument("--mlp-chunks", type=int, default=4,
                   help="Number of MLP hidden-dimension chunks (must be > 1; d_ff must be divisible).")
    g.add_argument("--embedding-segments", type=int, default=1,
                   help="Number of d_model slices for the embedding table (1=unsegmented; d_model must be divisible).")
    g.add_argument("--output-head-segments", type=int, default=1,
                   help="Number of d_model slices for the LM head (1=unsegmented; d_model must be divisible).")

    # Storage backend
    g = p.add_argument_group("Segment storage backend")
    g.add_argument("--storage", choices=["cpu_ram", "disk"], default="disk",
                   help="cpu_ram = GPU→CPU RAM offload (requires CUDA). disk = CPU/GPU→disk (default, required on CPU).")
    g.add_argument("--segment-dir", type=Path, default=SEGMENT_CACHE_DIR,
                   help="Root directory for disk-streaming segment files (only used with --storage=disk).")

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=3)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--val-batch-size", type=int, default=None,
                   help="Batch size for validation/test loaders. "
                        "Defaults to --batch-size. Smaller values reduce peak RSS.")
    g.add_argument("--gradient-accumulation", type=int, default=1,
                   help="Gradient accumulation steps. Only valid with after_full_backward update style.")
    g.add_argument("--update-style", choices=["after_full_backward", "immediate_segment_update"],
                   default="after_full_backward",
                   help="after_full_backward: collect all gradients then update (supports grad-accum). "
                        "immediate_segment_update: update each segment immediately during backward.")

    # Optimizer
    g = p.add_argument_group("Optimizer")
    g.add_argument("--optimizer", choices=["segmentwise_adamw", "segmentwise_sgd_momentum", "segmentwise_sgd", "stateless_sgd"],
                   default="segmentwise_adamw",
                   help="Optimizer policy for segment parameters. segmentwise_sgd is accepted as an alias for segmentwise_sgd_momentum. Non-segment params always use AdamW.")
    g.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate.")
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--momentum", type=float, default=0.9,
                   help="Momentum for segmentwise_sgd_momentum (ignored for adamw).")
    g.add_argument("--gradient-clip-norm", type=float, default=None,
                   help="Max gradient norm for non-segment parameters. None = no clipping.")

    # LR scheduler
    g = p.add_argument_group("LR scheduler")
    g.add_argument("--scheduler", choices=["cosine", "linear", "none"], default="cosine",
                   help="LR schedule applied after warmup.")
    g.add_argument("--warmup-steps", type=int, default=200,
                   help="Number of linear warmup steps. Applied to both segment and non-segment optimizers.")

    # Checkpointing
    g = p.add_argument_group("Checkpointing")
    g.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    g.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint name (e.g. 'last' or 'best'). "
                        "Restores model, segment states, and optimizer state.")

    # Export
    g = p.add_argument_group("Export (post-training)")
    g.add_argument("--export-dir", type=Path, default=EXPORT_DIR,
                   help="Root directory for full model and YAML protocol exports.")
    g.add_argument("--export-full-model", action=argparse.BooleanOptionalAction, default=True,
                   help="Assemble trained segments into a full model artifact after training.")
    g.add_argument("--export-yaml", action=argparse.BooleanOptionalAction, default=True,
                   help="Export YAML protocol file describing the complete training run.")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_DIR)
    g.add_argument("--train-split", type=Path, default=DATA_DIR / "train.csv")
    g.add_argument("--validation-split", type=Path, default=DATA_DIR / "validation.csv")
    g.add_argument("--test-split", type=Path, default=DATA_DIR / "test.csv")
    g.add_argument("--input-column", default="log_line")
    g.add_argument("--label-column", default="label")

    # Runtime
    g = p.add_argument_group("Runtime")
    g.add_argument("--device", default="cpu", help="Compute device: cpu or cuda.")
    g.add_argument("--workers", type=int, default=0, help="DataLoader num_workers.")
    g.add_argument("--seed", type=int, default=42, help="Global random seed.")
    g.add_argument("--recomputation-check", action="store_true", default=False,
                   help="Verify recomputed segment outputs match stored outputs (slower, for debugging).")
    g.add_argument("--max-train-steps", type=int, default=None,
                   help="Limit training to this many steps per epoch (for smoke tests).")
    g.add_argument("--max-val-steps", type=int, default=None,
                   help="Limit validation/test to this many steps (for smoke tests).")
    g.add_argument("--max-runtime-hours", type=float, default=None,
                   help="Graceful wall-time limit in hours. After each epoch, if elapsed time "
                        "exceeds this limit the run saves 'last' checkpoint and exits with "
                        "code 0 so a follow-up job can resume with --resume last.")
    g.add_argument("--metrics-output", type=Path, default=None,
                   help="If set, write a JSON file with full profiler timings/counters/"
                        "memory series, per-epoch loss curve, throughput, and disk usage.")
    g.add_argument("--record-backend", default="in_memory", choices=["in_memory", "disk"],
                   help="Backend for storing runtime records: in_memory (default) or disk.")
    g.add_argument("--record-disk-dir", default=None,
                   help="Root directory for per-step disk record files (required when --record-backend=disk).")
    g.add_argument("--gradient-store", default="in_memory", choices=["in_memory", "disk"],
                   help="Backend for storing accumulated gradients: in_memory (default) or disk.")
    g.add_argument("--gradient-store-dir", default=None,
                   help="Root directory for disk gradient files (required when --gradient-store=disk).")
    g.add_argument("--optimizer-state-store", default="in_memory", choices=["in_memory", "disk"],
                   help="Backend for segment optimizer state (Adam moments / SGD momentum): "
                        "in_memory (default) or disk.")
    g.add_argument("--optimizer-state-dir", default=None,
                   help="Root dir for disk optimizer state (default <segment-dir>/optim_state).")
    g.add_argument("--evict-layers-during-backward", action="store_true", default=False,
                   help="Free each layer's shared tensors immediately after that layer's backward pass.")
    g.add_argument("--minimal-records", action="store_true", default=False,
                   help="Do not retain per-segment output tensors (minimal records). "
                        "Incompatible with --recomputation-check.")
    g.add_argument("--lazy-records", action="store_true", default=False,
                   help="Store only layer_X.input per transformer layer (1 tensor vs 10). "
                        "Recompute all intermediates during backward. Reduces forward peak "
                        "memory by ~10x (auto-enabled on CPU).")
    g.add_argument("--output-head-token-chunks", type=int, default=1, metavar="K",
                   help="Split the sequence into K chunks for the output head backward to "
                        "reduce peak logit tensor size (default: 1 = no chunking, "
                        "auto-set to 8 on CPU).")
    g.add_argument("--run-test", action="store_true", default=False,
                   help="Run final evaluation on the test set after training using the best "
                        "checkpoint. Off by default; use infer_segmented.py --evaluate-test "
                        "to evaluate separately without reloading all segment states.")

    return p.parse_args(argv)


def directory_size_bytes(path: Path) -> int:
    """Return the total size in bytes of all files under path (0 if missing)."""

    if not path.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            total += (Path(dirpath) / name).stat().st_size
    return total


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR scheduling
# ---------------------------------------------------------------------------

def get_lr_multiplier(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    scheduler: str,
) -> float:
    """Return LR multiplier (0..1) for the given global step."""
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    if scheduler == "none":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    if scheduler == "linear":
        return max(0.0, 1.0 - progress)
    # cosine
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_lr(
    segment_optimizer: SegmentwiseAdamW | SegmentwiseSGD,
    non_segment_optimizer: torch.optim.Optimizer,
    base_lr: float,
    multiplier: float,
) -> None:
    """Apply LR multiplier to both optimizers."""
    segment_optimizer.lr = base_lr * multiplier
    for pg in non_segment_optimizer.param_groups:
        pg["lr"] = base_lr * multiplier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_dir: Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_segment_store(args: argparse.Namespace) -> CpuRamSegmentStore | DiskSegmentStore:
    if args.storage == "cpu_ram":
        return CpuRamSegmentStore()
    args.segment_dir.mkdir(parents=True, exist_ok=True)
    return DiskSegmentStore(args.segment_dir)


def _make_optimizer_state_store(args: argparse.Namespace):
    from sequential_segmented_llm_training_inference.optimization.optimizer_state_store import (
        InMemoryOptimizerStateStore,
        DiskOptimizerStateStore,
    )

    if getattr(args, "optimizer_state_store", "in_memory") == "disk":
        d = args.optimizer_state_dir or (Path(args.segment_dir) / "optim_state")
        return DiskOptimizerStateStore(d)
    return InMemoryOptimizerStateStore()


def build_segment_optimizer(args: argparse.Namespace) -> SegmentwiseAdamW | SegmentwiseSGD:
    args.optimizer = normalize_optimizer_type(args.optimizer)
    state_store = _make_optimizer_state_store(args)
    if args.optimizer == "segmentwise_adamw":
        return SegmentwiseAdamW(
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            state_store=state_store,
        )
    # stateless_sgd: momentum=0, weight_decay=0
    if args.optimizer == "stateless_sgd":
        return SegmentwiseSGD(lr=args.lr, momentum=0.0, weight_decay=0.0, state_store=state_store)
    # segmentwise_sgd_momentum
    return SegmentwiseSGD(
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        state_store=state_store,
    )


def initialise_segments(
    factory: SegmentFactory,
    store: CpuRamSegmentStore | DiskSegmentStore,
    model_config: ModelConfig | None = None,
    seg_config: "SegmentationConfig | None" = None,
) -> None:
    """Create and save all segment weights one at a time.

    When *model_config* is provided, the three large non-MLP/attention
    components (embedding, output_head, per-layer attention_output_proj) are
    also initialised as loadable segments and saved to the store.
    """
    # Create and save one segment at a time so only one segment's parameters
    # are ever in RAM simultaneously, matching the training-time invariant.
    # (create_all_segments() would hold all segments in RAM at once, pushing
    # the RSS high-water mark to full-model size before training even starts.)
    n_attn = n_mlp = 0
    for segment_id in factory.expected_segment_ids():
        if segment_id.segment_type == "attention":
            module = factory.create_attention_segment(
                layer_id=segment_id.layer_id,
                segment_index=segment_id.segment_id,
            )
            n_attn += 1
        else:
            module = factory.create_mlp_segment(
                layer_id=segment_id.layer_id,
                segment_index=segment_id.segment_id,
            )
            n_mlp += 1
        store.save_segment(segment_id, module.state_dict())
        del module
    print(f"  Initialised {n_attn} attention + {n_mlp} MLP segments.")

    if model_config is not None:
        n_emb_segs = (seg_config.embedding_segments if seg_config is not None else 1)
        n_head_segs = (seg_config.output_head_segments if seg_config is not None else 1)

        # Embedding: single full segment or N slices.
        if n_emb_segs == 1:
            emb_cfg = EmbeddingConfig(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                max_seq_len=model_config.max_seq_len,
                dropout=model_config.dropout,
                pad_token_id=model_config.pad_token_id,
            )
            emb_mod = EmbeddingSegmentModule(emb_cfg)
            store.save_segment(SegmentId(-1, "embedding", 0), emb_mod.state_dict())
            del emb_mod
        else:
            d_slice = model_config.d_model // n_emb_segs
            for i in range(n_emb_segs):
                emb_slice = EmbeddingSliceSegmentModule(
                    vocab_size=model_config.vocab_size,
                    d_slice=d_slice,
                    max_seq_len=model_config.max_seq_len,
                )
                store.save_segment(SegmentId(-1, "embedding", i), emb_slice.state_dict())
                del emb_slice

        # Output head: single full segment or N slices.
        if n_head_segs == 1:
            head_cfg = OutputHeadConfig(
                d_model=model_config.d_model,
                vocab_size=model_config.vocab_size,
            )
            head_mod = OutputHeadSegmentModule(head_cfg)
            store.save_segment(SegmentId(-1, "output_head", 0), head_mod.state_dict())
            del head_mod
        else:
            base = model_config.vocab_size // n_head_segs
            for i in range(n_head_segs):
                vocab_start = i * base
                vocab_end = model_config.vocab_size if i == n_head_segs - 1 else (i + 1) * base
                head_slice = OutputHeadSliceSegmentModule(
                    d_model=model_config.d_model,
                    vocab_slice_size=vocab_end - vocab_start,
                )
                store.save_segment(SegmentId(-1, "output_head", i), head_slice.state_dict())
                del head_slice

        for layer_id in range(model_config.n_layers):
            proj_mod = AttentionOutputProjSegmentModule(model_config.d_model)
            store.save_segment(SegmentId(layer_id, "attention_output_proj", 0), proj_mod.state_dict())
            del proj_mod
        print(
            f"  Initialised {n_emb_segs} embedding + {n_head_segs} output_head + "
            f"{model_config.n_layers} attention_output_proj global segments."
        )


def make_module_factory(
    factory: SegmentFactory,
    model_config: ModelConfig | None = None,
    seg_config: "SegmentationConfig | None" = None,
):
    """Return a callable that creates a fresh nn.Module for any SegmentId.

    When *model_config* is provided the factory also handles the three global
    segment types (``embedding``, ``output_head``, ``attention_output_proj``).
    """
    n_emb_segs = (seg_config.embedding_segments if seg_config is not None else 1)
    n_head_segs = (seg_config.output_head_segments if seg_config is not None else 1)

    def _factory(segment_id: SegmentId) -> torch.nn.Module:
        if segment_id.segment_type == "attention":
            return factory.create_attention_segment(
                layer_id=segment_id.layer_id,
                segment_index=segment_id.segment_id,
            )
        if segment_id.segment_type == "mlp":
            return factory.create_mlp_segment(
                layer_id=segment_id.layer_id,
                segment_index=segment_id.segment_id,
            )
        if model_config is None:
            raise ValueError(
                f"model_config required to instantiate segment_type={segment_id.segment_type!r}."
            )
        if segment_id.segment_type == "embedding":
            if n_emb_segs == 1:
                emb_cfg = EmbeddingConfig(
                    vocab_size=model_config.vocab_size,
                    d_model=model_config.d_model,
                    max_seq_len=model_config.max_seq_len,
                    dropout=model_config.dropout,
                    pad_token_id=model_config.pad_token_id,
                )
                return EmbeddingSegmentModule(emb_cfg)
            d_slice = model_config.d_model // n_emb_segs
            return EmbeddingSliceSegmentModule(
                vocab_size=model_config.vocab_size,
                d_slice=d_slice,
                max_seq_len=model_config.max_seq_len,
            )
        if segment_id.segment_type == "output_head":
            if n_head_segs == 1:
                head_cfg = OutputHeadConfig(
                    d_model=model_config.d_model,
                    vocab_size=model_config.vocab_size,
                )
                return OutputHeadSegmentModule(head_cfg)
            base = model_config.vocab_size // n_head_segs
            i = segment_id.segment_id
            vocab_start = i * base
            vocab_end = model_config.vocab_size if i == n_head_segs - 1 else (i + 1) * base
            return OutputHeadSliceSegmentModule(
                d_model=model_config.d_model,
                vocab_slice_size=vocab_end - vocab_start,
            )
        if segment_id.segment_type == "attention_output_proj":
            return AttentionOutputProjSegmentModule(model_config.d_model)
        raise ValueError(f"Unknown segment_type: {segment_id.segment_type!r}")

    return _factory


def _all_segment_ids(
    factory: SegmentFactory,
    model_config: ModelConfig | None = None,
    seg_config: "SegmentationConfig | None" = None,
) -> tuple[SegmentId, ...]:
    """Return all segment IDs: attention+MLP from factory, plus global ones when model_config given."""
    ids: list[SegmentId] = list(factory.expected_segment_ids())
    if model_config is not None:
        n_emb = (seg_config.embedding_segments if seg_config is not None else 1)
        n_head = (seg_config.output_head_segments if seg_config is not None else 1)
        for i in range(n_emb):
            ids.append(SegmentId(-1, "embedding", i))
        for i in range(n_head):
            ids.append(SegmentId(-1, "output_head", i))
        for layer_id in range(model_config.n_layers):
            ids.append(SegmentId(layer_id, "attention_output_proj", 0))
    return tuple(ids)


class _LazySegmentStates:
    """Dict-like view that loads segment states one at a time from the store.

    The checkpoint store iterates sorted(segment_states) then accesses by key,
    so only one segment state dict is in RAM at any given moment instead of all
    of them simultaneously.
    """

    def __init__(
        self,
        factory: SegmentFactory,
        store: CpuRamSegmentStore | DiskSegmentStore,
        model_config: ModelConfig | None = None,
        seg_config: "SegmentationConfig | None" = None,
    ) -> None:
        self._ids: tuple[SegmentId, ...] = _all_segment_ids(factory, model_config, seg_config)
        self._store = store

    def __bool__(self) -> bool:
        return bool(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)

    def __getitem__(self, key: SegmentId):
        return self._store.load_segment(key)

    def items(self):
        for sid in self._ids:
            yield sid, self._store.load_segment(sid)


def collect_all_segment_states(
    factory: SegmentFactory,
    store: CpuRamSegmentStore | DiskSegmentStore,
    model_config: ModelConfig | None = None,
    seg_config: "SegmentationConfig | None" = None,
) -> _LazySegmentStates:
    return _LazySegmentStates(factory, store, model_config, seg_config)


def collect_non_segment_state(fwd: SegmentedForwardEngine) -> dict[str, Any]:
    """Collect always-resident non-segment state.

    In segment-store mode, only ``layer_norms`` and ``mlp_shared_output_biases``
    remain as always-resident components (plus ``output_slice_final_norm`` when
    output_head_segments > 1).
    """
    if fwd._use_segment_store_for_global:
        state: dict[str, Any] = {
            "layer_norms": fwd.layer_norms.state_dict(),
            "mlp_shared_output_biases": fwd.mlp_shared_output_biases.state_dict(),
        }
        if hasattr(fwd, "output_slice_final_norm"):
            state["output_slice_final_norm"] = fwd.output_slice_final_norm.state_dict()
        return state
    return {
        "embeddings": fwd.embeddings.state_dict(),
        "layer_norms": fwd.layer_norms.state_dict(),
        "attention_output_projections": fwd.attention_output_projections.state_dict(),
        "mlp_shared_output_biases": fwd.mlp_shared_output_biases.state_dict(),
        "output_head": fwd.output_head.state_dict(),
    }


def restore_non_segment_state(fwd: SegmentedForwardEngine, state: dict[str, Any]) -> None:
    """Restore always-resident non-segment state from checkpoint."""
    fwd.layer_norms.load_state_dict(state["layer_norms"])
    fwd.mlp_shared_output_biases.load_state_dict(state["mlp_shared_output_biases"])
    if hasattr(fwd, "output_slice_final_norm") and "output_slice_final_norm" in state:
        fwd.output_slice_final_norm.load_state_dict(state["output_slice_final_norm"])
    if not fwd._use_segment_store_for_global:
        # Legacy mode: restore the large components from non_segment_state.
        if "embeddings" in state:
            fwd.embeddings.load_state_dict(state["embeddings"])
        if "attention_output_projections" in state:
            fwd.attention_output_projections.load_state_dict(state["attention_output_projections"])
        if "output_head" in state:
            fwd.output_head.load_state_dict(state["output_head"])


# ---------------------------------------------------------------------------
# Resumable segment-by-segment eval forward (P3 — one segment in memory at a time)
# ---------------------------------------------------------------------------

def _trim_cpu() -> None:
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def _resumable_forward_for_eval(
    forward_engine: SegmentedForwardEngine,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    device: torch.device,
    state_dir: Path,
) -> Tensor:
    """P3 forward for eval: every segment's activations are saved to disk immediately.

    Memory invariant at any point: only ONE of {hidden, attn_in, h_i, mlp_in,
    mlp_acc} is resident, plus the currently-loaded segment's weights.

    Attention projection uses column-sliced W_o with sequence-chunked matmul so
    no full [B, S, D] temporary tensor is ever allocated during the projection.
    All residual adds are in-place (no third tensor).
    """
    hidden_path  = state_dir / "_h.pt"
    attn_in_path = state_dir / "_attn_in.pt"
    mlp_in_path  = state_dir / "_mlp_in.pt"

    D      = forward_engine.model_config.d_model
    n_attn = forward_engine.segmentation_config.attention_segments
    n_mlp  = forward_engine.segmentation_config.mlp_chunks

    # ── Embedding ────────────────────────────────────────────────────────────
    hidden = forward_engine._run_embeddings(input_ids, None)
    torch.save(hidden.cpu(), hidden_path)
    del hidden; _trim_cpu()

    for layer_id in range(forward_engine.model_config.n_layers):

        # ── Attention layer norm ─────────────────────────────────────────────
        # Save both hidden (for residual later) and attn_in (reloaded per seg).
        hidden = torch.load(hidden_path, map_location=device)
        attn_in = forward_engine.layer_norms.attention(layer_id, hidden)
        torch.save(hidden.cpu(), hidden_path)
        torch.save(attn_in.cpu(), attn_in_path)
        del hidden, attn_in; _trim_cpu()

        # ── Attention segments: one output file per segment, no accumulation ─
        for seg_idx in range(n_attn):
            attn_in = torch.load(attn_in_path, map_location=device)
            seg_id  = SegmentId(layer_id=layer_id, segment_type="attention", segment_id=seg_idx)
            with forward_engine.segment_loader.acquire_segment(seg_id) as seg:
                h_i = seg(attn_in, attention_mask)
            del attn_in; _trim_cpu()
            torch.save(h_i.cpu(), state_dir / f"_ha{seg_idx}.pt")
            del h_i; _trim_cpu()
        attn_in_path.unlink(missing_ok=True)

        # ── Chunked attention projection (no concat of all head outputs) ─────
        # W_o [D, D]: contribution of segment i = h_i @ W_o[:, i*sz:(i+1)*sz].T
        # seq-chunked inner loop avoids a full [B, S, D] intermediate tensor.
        col_size  = D // n_attn
        SEQ_CHUNK = 32
        proj_acc: Tensor | None = None

        if forward_engine._use_segment_store_for_global and forward_engine._attn_proj_seg_ids is not None:
            seg_id = forward_engine._attn_proj_seg_ids[layer_id]
            with forward_engine.segment_loader.acquire_segment(seg_id) as proj_mod:
                W    = proj_mod.proj.weight              # [D, D]
                bias = proj_mod.proj.bias                # [D] or None
                for seg_idx in range(n_attn):
                    h_i      = torch.load(state_dir / f"_ha{seg_idx}.pt", map_location=device)
                    B, S, Dn = h_i.shape
                    if proj_acc is None:
                        proj_acc = torch.zeros(B, S, D, dtype=h_i.dtype, device=device)
                        if bias is not None:
                            proj_acc.add_(bias)          # broadcast [D] → [B,S,D] in-place
                    W_sl = W[:, seg_idx * col_size : (seg_idx + 1) * col_size]  # view, no copy
                    for t0 in range(0, S, SEQ_CHUNK):
                        t1 = min(t0 + SEQ_CHUNK, S)
                        # temp = [B, chunk, D] — small; add_ avoids keeping it
                        proj_acc[:, t0:t1, :].add_(h_i[:, t0:t1, :] @ W_sl.T)
                    del h_i; _trim_cpu()
                    (state_dir / f"_ha{seg_idx}.pt").unlink(missing_ok=True)
        else:
            # Legacy mode: proj weights always in RAM; fallback to concat.
            parts = [
                torch.load(state_dir / f"_ha{i}.pt", map_location=device)
                for i in range(n_attn)
            ]
            proj_acc = forward_engine._run_attention_output_proj(
                layer_id, concatenate_attention_outputs(parts)
            )
            del parts
            for i in range(n_attn):
                (state_dir / f"_ha{i}.pt").unlink(missing_ok=True)

        # ── Attention residual — in-place, no third tensor ───────────────────
        hidden = torch.load(hidden_path, map_location=device)
        hidden.add_(proj_acc)           # hidden += proj_acc (in-place)
        del proj_acc
        torch.save(hidden.cpu(), hidden_path)
        del hidden; _trim_cpu()

        # ── MLP layer norm ───────────────────────────────────────────────────
        # Save mlp_in separately so it can be freed and reloaded per MLP segment.
        hidden = torch.load(hidden_path, map_location=device)
        mlp_in  = forward_engine.layer_norms.mlp(layer_id, hidden)
        torch.save(hidden.cpu(), hidden_path)
        torch.save(mlp_in.cpu(), mlp_in_path)
        del hidden, mlp_in; _trim_cpu()

        # ── MLP segments: reload mlp_in per segment, accumulate in-place ─────
        mlp_acc: Tensor | None = None
        for seg_idx in range(n_mlp):
            mlp_in = torch.load(mlp_in_path, map_location=device)
            seg_id = SegmentId(layer_id=layer_id, segment_type="mlp", segment_id=seg_idx)
            with forward_engine.segment_loader.acquire_segment(seg_id) as seg:
                out = seg(mlp_in)
            del mlp_in; _trim_cpu()
            if mlp_acc is None:
                mlp_acc = out.cpu()
            else:
                mlp_acc.add_(out.cpu())  # in-place: no third tensor
            del out; _trim_cpu()
        mlp_in_path.unlink(missing_ok=True)

        # ── MLP shared bias + residual — in-place ────────────────────────────
        hidden  = torch.load(hidden_path, map_location=device)
        mlp_sum = mlp_acc.to(device)    # type: ignore[union-attr]
        del mlp_acc
        mlp_sum = forward_engine.mlp_shared_output_biases(layer_id, mlp_sum)
        hidden.add_(mlp_sum)            # hidden += mlp_sum (in-place)
        del mlp_sum
        torch.save(hidden.cpu(), hidden_path)
        del hidden; _trim_cpu()

    hidden = torch.load(hidden_path, map_location=device)
    hidden_path.unlink(missing_ok=True)
    return hidden


# ---------------------------------------------------------------------------
# Evaluation (segmented forward only — no backward, no optimizer update)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    forward_engine: SegmentedForwardEngine,
    loss_fn: CausalCrossEntropyLoss,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "validation",
    max_steps: int | None = None,
    profiler: SegmentedTrainingProfiler | None = None,
) -> float:
    """Segmented forward-only evaluation. Returns mean loss.

    Uses _resumable_forward_for_eval so hidden [B, S, D] is saved to disk
    between every segment acquisition — only one segment's weights + current
    hidden are in RAM at any instant (strict P3).
    """
    forward_engine.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct_tokens = 0.0
    total_valid_tokens = 0

    timer_name = "test" if split_name == "test" else "validation"
    use_chunked = (
        forward_engine._use_segment_store_for_global
        and forward_engine._output_head_seg_ids is not None
        and len(forward_engine._output_head_seg_ids) > 1
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tmp_dir = Path(tempfile.mkdtemp(prefix="seg_eval_"))
    try:
        start = time.perf_counter()
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            bsz = input_ids.size(0)

            # P3 resumable forward: every segment's activations saved/freed per step
            hidden = _resumable_forward_for_eval(
                forward_engine, input_ids, attention_mask, device, tmp_dir
            )
            del input_ids, attention_mask
            _trim_cpu()

            if use_chunked:
                loss, acc = forward_engine.compute_chunked_output_head_eval(
                    hidden, labels, ignore_index=loss_fn.ignore_index
                )
            else:
                logits = forward_engine._run_output_head(hidden)
                loss = loss_fn(logits, labels)
                acc = token_accuracy(logits, labels, ignore_index=loss_fn.ignore_index)
                del logits
            del hidden
            _trim_cpu()

            total_loss += loss.item() * bsz
            total_examples += bsz

            valid_mask = labels[:, 1:] != loss_fn.ignore_index
            num_valid = int(valid_mask.sum().item())
            if num_valid > 0:
                total_correct_tokens += acc * num_valid
                total_valid_tokens += num_valid
            del labels, loss
            _trim_cpu()

            if profiler is not None:
                _val_gpu_peak = (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                )
                profiler.record_batch_memory_snapshot(step, phase="validation",
                                                      gpu_peak_bytes=_val_gpu_peak)

        forward_engine.train()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    avg = total_loss / total_examples if total_examples > 0 else float("inf")
    avg_accuracy = total_correct_tokens / total_valid_tokens if total_valid_tokens > 0 else float("nan")
    print(
        f"  [{split_name}] mean loss = {avg:.4f}  "
        f"token_accuracy = {avg_accuracy:.4f}  ({total_examples} examples)"
    )
    if profiler is not None:
        profiler.add_time(timer_name, time.perf_counter() - start)
        profiler.increment(f"{timer_name}_runs")
        if split_name == "test":
            profiler.record_test_loss(avg)
        else:
            profiler.record_validation_loss(avg)
        if not math.isnan(avg_accuracy):
            profiler.record_task_metric(avg_accuracy)
        profiler.set_metadata(f"{split_name}_token_accuracy", avg_accuracy)
        val_rss = [
            s["cpu_rss_bytes"] for s in profiler.batch_memory_snapshots
            if s["phase"] == split_name and s["cpu_rss_bytes"] is not None
        ]
        if val_rss:
            print(
                f"  [{split_name}] peak RSS = {max(val_rss)/1e6:.1f} MB  "
                f"mean RSS = {sum(val_rss)/len(val_rss)/1e6:.1f} MB"
            )
        if torch.cuda.is_available():
            vram_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  [{split_name}] peak VRAM = {vram_peak_mb:.1f} MB")
    return avg


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    ckpt_store: SegmentedCheckpointStore,
    name: str,
    *,
    epoch: int,
    trainer: SegmentedTrainer,
    factory: SegmentFactory,
    store: CpuRamSegmentStore | DiskSegmentStore,
    forward_engine: SegmentedForwardEngine,
    segment_optimizer: SegmentwiseAdamW | SegmentwiseSGD,
    non_segment_optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    seg_config: SegmentationConfig,
    val_loss: float | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> None:
    _mc = model_config if forward_engine._use_segment_store_for_global else None
    segment_states = collect_all_segment_states(factory, store, model_config=_mc, seg_config=seg_config)
    non_segment_state = collect_non_segment_state(forward_engine)
    rng_state = capture_rng_state().to_checkpoint()

    checkpoint = SegmentedCheckpoint(
        epoch=epoch,
        global_step=trainer.global_step,
        segment_states=segment_states,
        non_segment_state=non_segment_state,
        optimizer_state={
            "segment_optimizer": segment_optimizer.state_dict(),
            "non_segment_optimizer": non_segment_optimizer.state_dict(),
        },
        scheduler_state=scheduler_state,
        rng_state=rng_state,
        model_config=model_config.to_dict(),
        segmentation_config=seg_config.to_dict(),
        validation_metadata={"val_loss": val_loss} if val_loss is not None else {},
    )
    ckpt_store.save_checkpoint(checkpoint_name=name, checkpoint=checkpoint)
    print(f"  Checkpoint saved: '{name}'")


def restore_checkpoint(
    ckpt_store: SegmentedCheckpointStore,
    name: str,
    *,
    factory: SegmentFactory,
    store: CpuRamSegmentStore | DiskSegmentStore,
    forward_engine: SegmentedForwardEngine,
    segment_optimizer: SegmentwiseAdamW | SegmentwiseSGD,
    non_segment_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, int, float]:
    """
    Load checkpoint and restore all states.

    Returns:
        (start_epoch, global_step, best_val_loss)
    """
    from sequential_segmented_llm_training_inference.execution.rng import RngState

    loaded = ckpt_store.load_checkpoint(name)
    ckpt = loaded.checkpoint

    # Restore segment states
    for seg_id, state in ckpt.segment_states.items():
        store.save_segment(seg_id, state)

    # Restore non-segment states
    restore_non_segment_state(forward_engine, ckpt.non_segment_state)

    # Restore optimizer state. Older checkpoints stored only segment optimizer
    # state directly; newer checkpoints store both segment and non-segment states.
    if ckpt.optimizer_state:
        try:
            if "segment_optimizer" in ckpt.optimizer_state:
                segment_optimizer.load_state_dict(ckpt.optimizer_state["segment_optimizer"])
                if non_segment_optimizer is not None and ckpt.optimizer_state.get("non_segment_optimizer"):
                    non_segment_optimizer.load_state_dict(ckpt.optimizer_state["non_segment_optimizer"])
            else:
                segment_optimizer.load_state_dict(ckpt.optimizer_state)
        except Exception as e:
            print(f"  Note: optimizer state not restored ({e}), starting fresh.")
            pass

    # Restore RNG state
    if ckpt.rng_state:
        try:
            rng = RngState.from_checkpoint(ckpt.rng_state)
            restore_rng_state(rng, restore_cuda=rng.has_cuda_state)
        except Exception:
            pass

    start_epoch = ckpt.epoch + 1
    global_step = ckpt.global_step
    val_loss = ckpt.validation_metadata.get("val_loss", float("inf"))

    print(f"  Resumed from '{name}': epoch={ckpt.epoch}, step={global_step}, val_loss={val_loss:.4f}")
    return start_epoch, global_step, val_loss


# ---------------------------------------------------------------------------
# Post-training export
# ---------------------------------------------------------------------------

def export_full_model(
    ckpt_store: SegmentedCheckpointStore,
    checkpoint_name: str,
    export_dir: Path,
) -> None:
    from sequential_segmented_llm_training_inference.export.full_model_exporter import (
        export_full_model_from_checkpoint,
    )

    loaded = ckpt_store.load_checkpoint(checkpoint_name)
    ckpt = loaded.checkpoint

    # The exporter expects non_segment_state as a flat {param_name: tensor} dict.
    # Our checkpoint stores it as nested {module_name: state_dict}.
    flat_non_segment: dict[str, Any] = {}
    for module_name, state_dict in ckpt.non_segment_state.items():
        for param_name, tensor in state_dict.items():
            flat_non_segment[f"{module_name}.{param_name}"] = tensor

    # Separate attention/MLP segments (reassembled by the exporter) from the
    # GLOBAL components (embedding, output_head, attention_output_proj).  The
    # global components are handed to the exporter as structured per-segment
    # state dicts under ``global_segments`` so the exporter can consolidate the
    # sliced forms (N>1) into dense full-model weights.  We must NOT flatten
    # multiple embedding / output-head slices into non_segment_state — they
    # share the same parameter names and would collide.
    flat_attn_mlp_segments: dict[str, Any] = {}
    global_segments: dict[str, Any] = {}
    for sid, state in ckpt.segment_states.items():
        if sid.segment_type in ("attention", "mlp"):
            key = f"layer_{sid.layer_id}.{sid.segment_type}.{sid.segment_id}"
            flat_attn_mlp_segments[key] = state
        elif sid.segment_type == "embedding":
            global_segments[f"emb.{sid.segment_id}"] = state
        elif sid.segment_type == "output_head":
            global_segments[f"head.{sid.segment_id}"] = state
        elif sid.segment_type == "attention_output_proj":
            global_segments[f"attn_proj.{sid.layer_id}"] = state

    checkpoint_dict = {
        "model_config": ckpt.model_config,
        "segmentation_config": ckpt.segmentation_config,
        "non_segment_state": flat_non_segment,
        "segments": flat_attn_mlp_segments,
        "global_segments": global_segments,
    }

    out_dir = export_dir / "full_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = export_full_model_from_checkpoint(checkpoint_dict, out_dir)
    print(f"  Full model exported → {result.full_model_path}")
    print(f"    ({result.num_full_state_parameters} parameters total)")


def export_yaml_protocol(
    args: argparse.Namespace,
    model_config: ModelConfig,
    seg_config: SegmentationConfig,
    export_dir: Path,
    *,
    best_val_loss: float,
    test_loss: float,
    total_steps: int,
) -> None:
    from sequential_segmented_llm_training_inference.config import (
        ExecutionConfig,
        ExportConfig,
        OptimizerConfig,
        RuntimeConfig,
        StorageConfig,
        TrainingConfig,
    )
    from sequential_segmented_llm_training_inference.config.protocol_config import ProtocolConfig
    from sequential_segmented_llm_training_inference.export.protocol_exporter import export_protocol_yaml

    # Report the backend that was actually used at runtime.
    backend = "cpu_ram_offload" if args.storage == "cpu_ram" else "disk_streaming"
    storage_cfg = StorageConfig(
        device=args.device,
        backend=backend,
        segment_dir=str(args.segment_dir),
    )
    execution_cfg = ExecutionConfig(mode="strict_single_segment", max_active_segments=1)
    training_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        update_style=args.update_style,
        precision="fp32",
        gradient_clip_norm=args.gradient_clip_norm,
        backward_mode="recomputation",
        gradient_clipping_scope=("true_global" if args.gradient_clip_norm is not None else "disabled"),
    )
    optimizer_cfg = OptimizerConfig(
        type=normalize_optimizer_type(args.optimizer),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    runtime_cfg = RuntimeConfig(
        policy="minimal_segment_records" if args.minimal_records else "full_segment_records",
        store_rng_state=True,
        restore_rng_state_during_backward=True,
        recomputation_check=args.recomputation_check,
        detach_record_tensors=True,
        clone_record_tensors=True,
        record_tensors_to_cpu=True,
    )
    export_cfg = ExportConfig(
        export_full_model=args.export_full_model,
        export_yaml_protocol=args.export_yaml,
        full_model_export_path=str(export_dir / "full_model"),
        yaml_protocol_path=str(export_dir / "segmented_training_protocol.yaml"),
    )
    protocol_config = ProtocolConfig(
        model=model_config,
        segmentation=seg_config,
        storage=storage_cfg,
        execution=execution_cfg,
        training=training_cfg,
        optimizer=optimizer_cfg,
        runtime_records=runtime_cfg,
        export=export_cfg,
    )

    yaml_path = export_dir / "segmented_training_protocol.yaml"
    export_protocol_yaml(
        protocol_config,
        yaml_path,
        data={
            "train_split": str(args.train_split),
            "validation_split": str(args.validation_split),
            "test_split": str(args.test_split),
            "input_column": args.input_column,
            "label_column": args.label_column,
            "tokenizer_path": str(args.tokenizer_path),
        },
        reproducibility={
            "seed": args.seed,
            "save_python_rng_state": True,
            "save_torch_rng_state": True,
            "save_cuda_rng_state": torch.cuda.is_available(),
        },
        results={
            "best_validation_loss": best_val_loss,
            "final_test_loss": test_loss,
            "total_training_steps": total_steps,
        },
    )
    print(f"  YAML protocol exported → {yaml_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    args.optimizer = normalize_optimizer_type(args.optimizer)

    # The spec requires CPU execution to use disk streaming only.
    if args.gradient_clip_norm is not None and args.update_style == "immediate_segment_update":
        raise ValueError(
            "--gradient-clip-norm requires --update-style after_full_backward for exact true-global clipping."
        )
    if args.dropout > 0.0 and not args.recomputation_check:
        # Dropout is supported, but enabling recomputation_check is strongly
        # recommended while debugging new configurations.
        pass
    if args.device == "cpu" and args.storage == "cpu_ram":
        raise ValueError(
            "--storage cpu_ram requires a CUDA device. "
            "Use --storage disk when running on CPU."
        )

    device = torch.device(args.device)

    # CPU training: enforce disk backends for all stores so that only one segment's
    # weights, optimizer state, gradients, and activation records are in RAM at a
    # time. Only override if the user left the default (in_memory); if they
    # explicitly passed --gradient-store=in_memory on CPU we still enforce disk
    # because in_memory on CPU directly violates the single-segment principle.
    if device.type == "cpu":
        if args.gradient_store == "in_memory":
            args.gradient_store = "disk"
        if args.optimizer_state_store == "in_memory":
            args.optimizer_state_store = "disk"
        if args.record_backend == "in_memory":
            args.record_backend = "disk"
        if not args.evict_layers_during_backward:
            args.evict_layers_during_backward = True
        if args.gradient_store_dir is None:
            args.gradient_store_dir = str(args.segment_dir / "grad_store")
        if args.record_disk_dir is None:
            args.record_disk_dir = str(args.segment_dir / "record_store")
        if args.optimizer_state_dir is None:
            args.optimizer_state_dir = str(args.segment_dir / "optim_state")
        # Auto-enable minimal records on CPU: segment output tensors are recomputed
        # during backward anyway, so storing them in RAM is pure waste.
        # Incompatible with --recomputation-check (which needs stored outputs to compare).
        if not args.minimal_records and not args.recomputation_check:
            args.minimal_records = True
        # Auto-enable lazy records: stores only 1 tensor per layer (layer_X.input)
        # instead of 10, recomputing intermediates in backward. Reduces training forward
        # peak by ~10x vs full records.
        if not args.lazy_records:
            args.lazy_records = True
        # Auto-enable token-chunked output head (8 chunks) to reduce peak logit tensor.
        if args.output_head_token_chunks == 1:
            args.output_head_token_chunks = 8
        print(
            "  [CPU] Auto-selected disk backends: gradient_store=disk, "
            "optimizer_state_store=disk, record_backend=disk, evict_layers=True, "
            "minimal_records=True, lazy_records=True, output_head_token_chunks=8"
        )

    # Auto-enable token-chunked output head when using segmented output head (N>1
    # segments), regardless of device.  Without chunking the backward sweep allocates
    # three simultaneous tensors of shape [B, S-1, vocab/N] — at B=128, S=256,
    # vocab=50257, N=8 that is ≈ 782 MB each (≈ 2.3 GB total), which is the dominant
    # source of the GPU training peak.  Chunking by N reduces each tensor to
    # [B, S/N-1, vocab/N] ≈ 97 MB, cutting the peak by ~2 GB.
    # CPU code above already sets this to 8; this block catches GPU (and any CPU path
    # that somehow bypassed the block above).
    if getattr(args, "output_head_segments", 1) > 1 and args.output_head_token_chunks == 1:
        args.output_head_token_chunks = getattr(args, "output_head_segments", 1)
        if device.type != "cpu":
            print(
                f"  [GPU] Auto-set output_head_token_chunks={args.output_head_token_chunks} "
                f"to reduce peak VRAM from segmented output head backward."
            )

    # Capture a FAIR memory baseline BEFORE the model/segments are built, so any
    # resident weights ARE counted in the "from-start" net memory. The profiler's
    # own baseline (captured after setup, below) deliberately EXCLUDES resident
    # state for the back-compat peak_cpu_net_mb; this pre-load reading is the
    # minuend for the fair peak_cpu_net_from_start_mb, keeping the metric
    # symmetric with the full-model training/inference paths.
    try:
        import psutil as _psutil
        _pre_load_baseline_bytes = int(_psutil.Process().memory_info().rss)
    except Exception:
        _pre_load_baseline_bytes = 0

    print("=" * 65)
    print("Sequential Segmented LLM Training")
    print("=" * 65)
    print(f"  Device          : {device}  (storage: {args.storage})")
    print(f"  Update style    : {args.update_style}")
    print(f"  Optimizer       : {args.optimizer}")
    print(f"  Scheduler       : {args.scheduler}  (warmup={args.warmup_steps})")
    print(f"  d_model/n_heads/n_layers/d_ff : "
          f"{args.d_model}/{args.n_heads}/{args.n_layers}/{args.d_ff}")
    print(f"  attention_segments/mlp_chunks : "
          f"{args.attention_segments}/{args.mlp_chunks}")
    print(f"  epochs/batch_size/lr/grad_accum : "
          f"{args.epochs}/{args.batch_size}/{args.lr}/{args.gradient_accumulation}")
    print(f"  Grad clip norm  : {args.gradient_clip_norm}")
    print(f"  Seed            : {args.seed}")

    # ------------------------------------------------------------------
    # 0. Seed
    # ------------------------------------------------------------------
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print("\n[1] Loading tokenizer ...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    print(f"  vocab_size={tokenizer.vocab_size}, pad={pad_token_id}, eos={eos_token_id}")

    # ------------------------------------------------------------------
    # 2. Model and segmentation config  (§31 validation rules applied)
    # ------------------------------------------------------------------
    print("\n[2] Building model and segmentation config ...")
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    seg_config = SegmentationConfig(
        attention_segments=args.attention_segments,
        mlp_chunks=args.mlp_chunks,
        embedding_segments=args.embedding_segments,
        output_head_segments=args.output_head_segments,
    )
    seg_config.validate_against_model(model_config)
    print(f"  {model_config}")
    print(f"  {seg_config}")
    print(f"  heads_per_segment = {seg_config.heads_per_segment(model_config)}, "
          f"mlp_chunk_size = {seg_config.mlp_chunk_size(model_config)}")

    # ------------------------------------------------------------------
    # 3. Segment factory + store
    # ------------------------------------------------------------------
    print("\n[3] Initialising segments ...")
    factory = SegmentFactory(
        model_config=model_config,
        segmentation_config=seg_config,
        attention_dropout=model_config.dropout,
        mlp_activation="gelu",
    )
    store = build_segment_store(args)

    # ------------------------------------------------------------------
    # 4. Segment loader + forward engine (strict single-segment)
    # ------------------------------------------------------------------
    print("\n[4] Building forward engine (strict single-segment) ...")
    profiler = SegmentedTrainingProfiler()
    profiler.set_metadata("device", str(device))
    profiler.set_metadata("storage", args.storage)
    profiler.set_metadata("update_style", args.update_style)
    profiler.set_metadata("optimizer", args.optimizer)
    profiler.set_metadata("batch_size", args.batch_size)
    profiler.set_metadata("gradient_accumulation_steps", args.gradient_accumulation)
    profiler.set_metadata("max_seq_len", args.max_seq_len)
    profiler.set_metadata("seed", args.seed)
    profiler.set_metadata(
        "model",
        {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
        },
    )
    profiler.set_metadata(
        "segmentation",
        {"attention_segments": args.attention_segments, "mlp_chunks": args.mlp_chunks},
    )
    profiler.reset_cuda_peak_memory()
    profiler.capture_cpu_baseline()   # baseline after setup, before any computation
    run_start = time.time()

    module_factory = make_module_factory(factory, model_config, seg_config)
    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device=device,
        profiler=profiler,
    )
    if args.minimal_records and args.recomputation_check:
        raise SystemExit(
            "--minimal-records is incompatible with --recomputation-check: "
            "stored outputs needed for the check are not retained. Disable one."
        )
    recomp_check = RecomputationCheckConfig(
        enabled=args.recomputation_check,
        raise_on_mismatch=args.recomputation_check,
    )
    forward_engine = SegmentedForwardEngine(
        model_config=model_config,
        segmentation_config=seg_config,
        segment_loader=loader,
        residual_dropout=model_config.dropout,
        use_segment_store_for_global=True,
        store_segment_outputs=not args.minimal_records,
    )
    forward_engine.to(device)
    forward_engine.train()

    backward_engine = SegmentedBackwardEngine(
        model_config=model_config,
        segmentation_config=seg_config,
        segment_loader=loader,
        recomputation_check=recomp_check,
    )

    # ------------------------------------------------------------------
    # 5. Optimizers
    # ------------------------------------------------------------------
    print("\n[5] Setting up optimizers ...")
    segment_optimizer = build_segment_optimizer(args)
    # Remove the per-run disk optimizer-state scratch dir on process exit
    # (normal completion or exception). Checkpoints already copy the state out
    # via state_dict(), so the scratch dir is safe to delete afterwards.
    if getattr(args, "optimizer_state_store", "in_memory") == "disk":
        _optim_scratch = Path(
            args.optimizer_state_dir or (Path(args.segment_dir) / "optim_state")
        )
        atexit.register(shutil.rmtree, _optim_scratch, ignore_errors=True)
    non_segment_optimizer = torch.optim.AdamW(
        forward_engine.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    loss_fn = CausalCrossEntropyLoss(ignore_index=-100)
    print(f"  Segment optimizer: {type(segment_optimizer).__name__}")

    trainer = SegmentedTrainer(
        forward_engine=forward_engine,
        backward_engine=backward_engine,
        segment_optimizer=segment_optimizer,
        non_segment_optimizer=non_segment_optimizer,
        loss_fn=loss_fn,
        update_style=args.update_style,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_clip_norm=args.gradient_clip_norm,
        record_backend=args.record_backend,
        record_disk_dir=args.record_disk_dir,
        evict_layers_during_backward=args.evict_layers_during_backward,
        gradient_store=args.gradient_store,
        gradient_store_dir=args.gradient_store_dir,
        store_only_layer_inputs=getattr(args, "lazy_records", False),
        output_head_token_chunks=getattr(args, "output_head_token_chunks", 1),
    )

    # ------------------------------------------------------------------
    # 6. Data loaders
    # ------------------------------------------------------------------
    print("\n[6] Loading data ...")
    collator = LogGenerationCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        prompt_template="{data}\nLabel:",
        target_prefix=" ",
        add_bos=False,
        add_eos=True,
    )
    train_dataset = LogLabelDataset(args.train_split, data_column=args.input_column, label_column=args.label_column)
    val_dataset = LogLabelDataset(args.validation_split, data_column=args.input_column, label_column=args.label_column)
    test_dataset = LogLabelDataset(args.test_split, data_column=args.input_column, label_column=args.label_column)
    print(f"  train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    val_bs = args.val_batch_size if args.val_batch_size is not None else args.batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False,
        collate_fn=collator, num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=val_bs, shuffle=False,
        collate_fn=collator, num_workers=args.workers,
    )

    # ------------------------------------------------------------------
    # 7. Checkpoint store + LR schedule setup
    # ------------------------------------------------------------------
    ckpt_store = SegmentedCheckpointStore(args.checkpoint_dir)
    best_val_loss = float("inf")
    best_epoch = -1
    start_epoch = 0

    steps_per_epoch = min(
        len(train_loader),
        args.max_train_steps if args.max_train_steps is not None else len(train_loader),
    )
    total_steps = args.epochs * steps_per_epoch

    # ------------------------------------------------------------------
    # 8. Resume from checkpoint (optional)
    # ------------------------------------------------------------------
    if args.resume is not None:
        print(f"\n[8-pre] Resuming from checkpoint '{args.resume}' ...")
        start_epoch, resumed_step, best_val_loss = restore_checkpoint(
            ckpt_store, args.resume,
            factory=factory, store=store,
            forward_engine=forward_engine,
            segment_optimizer=segment_optimizer,
            non_segment_optimizer=non_segment_optimizer,
        )
        # Sync trainer global_step
        trainer.global_step = resumed_step
    else:
        # Fresh run — initialise segment weights (including global segments
        # when the forward engine uses segment-store mode).
        _mc = model_config if forward_engine._use_segment_store_for_global else None
        _sc = seg_config if forward_engine._use_segment_store_for_global else None
        initialise_segments(factory, store, model_config=_mc, seg_config=_sc)

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    print("\n[7] Starting training ...")
    print(f"  Total steps (approx): {total_steps}  |  epochs: {args.epochs - start_epoch}")
    print("-" * 65)

    global_step = trainer.global_step
    total_train_examples = 0
    total_train_tokens = 0
    epoch_series: list[dict[str, Any]] = []
    run_deadline = (run_start + args.max_runtime_hours * 3600) if args.max_runtime_hours else None

    # If all epochs already completed (e.g. resumed into a redundant chain job),
    # skip straight to test/export.
    training_complete = (start_epoch >= args.epochs)
    if training_complete:
        print(f"\n[INFO] All {args.epochs} epochs already completed (start_epoch={start_epoch}). Skipping to test/export.")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_steps = 0
        _epoch_train_gpu_peak = 0  # running max of per-step VRAM peaks (training only)

        forward_engine.train()
        for step, batch in enumerate(train_loader):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

            # Apply LR schedule
            lr_mult = get_lr_multiplier(
                global_step,
                warmup_steps=args.warmup_steps,
                total_steps=total_steps,
                scheduler=args.scheduler,
            )
            set_lr(segment_optimizer, non_segment_optimizer, args.lr, lr_mult)

            train_batch = SegmentTrainBatch(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            step_start = time.perf_counter()
            result = trainer.train_step(train_batch)
            profiler.add_time("step", time.perf_counter() - step_start)
            profiler.increment("training_steps")
            profiler.record_training_loss(result.loss)
            profiler.record_memory_snapshot()
            _step_gpu_peak = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            )
            if _step_gpu_peak is not None:
                _epoch_train_gpu_peak = max(_epoch_train_gpu_peak, _step_gpu_peak)
            profiler.record_batch_memory_snapshot(global_step, phase="training",
                                                  gpu_peak_bytes=_step_gpu_peak)

            bsz, seq_len = train_batch.input_ids.shape
            total_train_examples += bsz
            total_train_tokens += int(train_batch.attention_mask.sum().item())

            epoch_loss += result.loss
            num_steps += 1
            global_step += 1

            log_every = max(1, (args.max_train_steps or len(train_loader)) // 10)
            if (step + 1) % log_every == 0:
                elapsed = time.time() - epoch_start
                avg_loss = epoch_loss / num_steps
                print(
                    f"  Epoch {epoch+1}/{args.epochs}  "
                    f"step {step+1}/{len(train_loader)}  "
                    f"loss={avg_loss:.4f}  "
                    f"lr={segment_optimizer.lr:.2e}  "
                    f"elapsed={elapsed:.1f}s"
                )

        avg_train_loss = epoch_loss / num_steps if num_steps > 0 else float("inf")
        elapsed = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} done — train_loss={avg_train_loss:.4f}  time={elapsed:.1f}s")

        # Validation (segmented forward only, no backward)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        val_loss = evaluate(
            forward_engine, loss_fn, val_loader, device, "validation",
            max_steps=args.max_val_steps, profiler=profiler,
        )
        _epoch_val_gpu_peak = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        )

        epoch_series.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "epoch_time_s": elapsed,
                "cpu_peak_memory_bytes": profiler.latest_scalar("cpu_peak_memory_bytes"),
                "cpu_net_peak_memory_bytes": profiler.latest_scalar("cpu_net_peak_memory_bytes"),
                "gpu_peak_memory_bytes": profiler.latest_scalar("gpu_peak_memory_bytes"),
                "gpu_peak_reserved_bytes": profiler.latest_scalar("gpu_peak_reserved_bytes"),
                "train_gpu_peak_bytes": _epoch_train_gpu_peak if _epoch_train_gpu_peak else None,
                "val_gpu_peak_bytes": _epoch_val_gpu_peak,
                "validation_token_accuracy": profiler.metadata.get("validation_token_accuracy"),
            }
        )

        scheduler_state = {
            "scheduler": args.scheduler,
            "warmup_steps": args.warmup_steps,
            "total_steps": total_steps,
            "base_lr": args.lr,
            "current_step": global_step,
        }

        # Save last checkpoint
        with profiler.time_section("checkpoint"):
            save_checkpoint(
                ckpt_store, name="last",
                epoch=epoch, trainer=trainer, factory=factory, store=store,
                forward_engine=forward_engine,
                segment_optimizer=segment_optimizer,
                non_segment_optimizer=non_segment_optimizer,
                model_config=model_config, seg_config=seg_config,
                val_loss=val_loss,
                scheduler_state=scheduler_state,
            )
        profiler.increment("checkpoint_saves")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            with profiler.time_section("checkpoint"):
                save_checkpoint(
                    ckpt_store, name="best",
                    epoch=epoch, trainer=trainer, factory=factory, store=store,
                    forward_engine=forward_engine,
                    segment_optimizer=segment_optimizer,
                    non_segment_optimizer=non_segment_optimizer,
                    model_config=model_config, seg_config=seg_config,
                    val_loss=val_loss,
                    scheduler_state=scheduler_state,
                )
            profiler.increment("checkpoint_saves")
            print(f"  *** New best val_loss={best_val_loss:.4f} at epoch {epoch+1} ***")

        print("-" * 65)

        # Graceful wall-time exit: stop after completing a full epoch so
        # the next job can resume cleanly with --resume last.
        if run_deadline is not None and time.time() >= run_deadline:
            epochs_remaining = args.epochs - (epoch + 1)
            print(
                f"\n[TIME LIMIT] Reached {args.max_runtime_hours:.1f}h wall limit after "
                f"epoch {epoch+1}/{args.epochs}. "
                f"{epochs_remaining} epoch(s) remaining. "
                f"Checkpoint saved as 'last'. Resume with: --resume last --epochs {args.epochs}"
            )
            break
    else:
        training_complete = True

    # ------------------------------------------------------------------
    # 10. Final test using best checkpoint (segmented forward only)
    # ------------------------------------------------------------------
    if not training_complete:
        # Partial run — write minimal metrics and exit so the next job
        # can continue.  Test / export require a finished model.
        print("\nTraining paused (wall-time limit). Partial metrics saved.")
        if args.metrics_output is not None:
            partial_metrics = {
                "run_summary": {
                    "status": "paused",
                    "epochs_completed": epoch + 1,
                    "epochs_requested": args.epochs,
                    "total_runtime_s": time.time() - run_start,
                    "total_train_steps": profiler.counters.get("training_steps", 0),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch + 1,
                },
                "epoch_series": epoch_series,
            }
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.metrics_output, "w") as f:
                json.dump(partial_metrics, f, indent=2, default=str)
            print(f"  Partial metrics → {args.metrics_output}")
        return
    test_loss = float("nan")
    print(f"\nBest model was from epoch {best_epoch + 1} (val_loss={best_val_loss:.4f})")
    if args.run_test:
        print("\n[8] Final test using best checkpoint ...")
        # Load non-segment state (small) eagerly, then reload segment weights
        # one at a time via iter_segment_states to preserve the single-segment
        # invariant — avoids holding all segment dicts in RAM simultaneously.
        restore_non_segment_state(forward_engine, ckpt_store.load_non_segment_state("best"))
        for seg_id, state in ckpt_store.iter_segment_states("best"):
            store.save_segment(seg_id, state)

        test_loss = evaluate(
            forward_engine, loss_fn, test_loader, device, "test",
            max_steps=args.max_val_steps, profiler=profiler,
        )
        print(f"  Final test_loss = {test_loss:.4f}")

    # ------------------------------------------------------------------
    # 11. Full model export (assemble segments → normal model artifact)
    # ------------------------------------------------------------------
    if args.export_full_model:
        print("\n[9] Exporting full model from trained segments ...")
        args.export_dir.mkdir(parents=True, exist_ok=True)
        try:
            with profiler.time_section("full_model_export"):
                export_full_model(ckpt_store, "best", args.export_dir)
            profiler.increment("full_model_exports")
        except Exception as e:
            print(f"  Warning: full model export failed: {e}")

    # ------------------------------------------------------------------
    # 12. YAML protocol export (for reproducibility / paper)
    # ------------------------------------------------------------------
    if args.export_yaml:
        print("\n[10] Exporting YAML training protocol ...")
        args.export_dir.mkdir(parents=True, exist_ok=True)
        try:
            with profiler.time_section("yaml_protocol_export"):
                export_yaml_protocol(
                    args, model_config, seg_config, args.export_dir,
                    best_val_loss=best_val_loss,
                    test_loss=test_loss,
                    total_steps=global_step,
                )
            profiler.increment("yaml_protocol_exports")
        except Exception as e:
            print(f"  Warning: YAML protocol export failed: {e}")

    print("\nTraining complete.")
    print(f"  Checkpoints  → {args.checkpoint_dir}")
    if args.export_full_model or args.export_yaml:
        print(f"  Exports      → {args.export_dir}")

    # ------------------------------------------------------------------
    # 13. Dump full profiler + derived run-level metrics for the paper
    # ------------------------------------------------------------------
    if args.metrics_output is not None:
        total_runtime_s = time.time() - run_start
        checkpoint_bytes = directory_size_bytes(args.checkpoint_dir)
        export_bytes = (
            directory_size_bytes(args.export_dir)
            if (args.export_full_model or args.export_yaml) and args.export_dir.exists()
            else 0
        )
        timers = profiler.timers
        counters = profiler.counters
        total_train_steps = max(counters.get("training_steps", 0), 1)
        # Raw peak process RSS (baseline-independent): includes resident weights.
        _peak_cpu_total_bytes = profiler.cpu_raw_peak_memory_bytes()
        run_summary = {
            "total_runtime_s": total_runtime_s,
            "total_train_steps": counters.get("training_steps", 0),
            "total_train_examples": total_train_examples,
            "total_train_tokens": total_train_tokens,
            "samples_per_s": total_train_examples / total_runtime_s if total_runtime_s > 0 else None,
            "tokens_per_s": total_train_tokens / total_runtime_s if total_runtime_s > 0 else None,
            "avg_step_time_s": timers["step"].average_seconds,
            "avg_epoch_time_s": (
                sum(e["epoch_time_s"] for e in epoch_series) / len(epoch_series)
                if epoch_series
                else None
            ),
            "segment_loads": counters.get("segment_loads", 0),
            "segment_saves": counters.get("segment_saves", 0),
            "segment_unloads": counters.get("segment_unloads", 0),
            "segment_bytes_loaded": counters.get("segment_bytes_loaded", 0),
            "segment_bytes_saved": counters.get("segment_bytes_saved", 0),
            "segment_load_time_s": timers["segment_load"].total_seconds,
            "segment_save_time_s": timers["segment_save"].total_seconds,
            "segment_load_store_overhead_fraction": (
                (timers["segment_load"].total_seconds + timers["segment_save"].total_seconds)
                / total_runtime_s
                if total_runtime_s > 0
                else None
            ),
            "checkpoint_dir_bytes": checkpoint_bytes,
            "export_dir_bytes": export_bytes,
            "best_epoch": best_epoch + 1,
            "best_val_loss": best_val_loss,
            "test_loss": None if math.isnan(test_loss) else test_loss,
            "test_perplexity": None if math.isnan(test_loss) else perplexity(test_loss),
            "segment_type_peak_mb": {
                k: round(v / 1024**2, 3)
                for k, v in profiler.segment_type_peak_bytes.items()
            },
            "per_segment_peak_mb": {
                k: round(v / 1024**2, 3)
                for k, v in profiler.per_segment_peak_bytes.items()
            },
            "peak_model_param_mb": (
                round(max(profiler.segment_type_peak_bytes.values()) / 1024**2, 3)
                if profiler.segment_type_peak_bytes else None
            ),
            # CPU: net = peak(current_rss - baseline); isolates computation from
            # Python/PyTorch overhead. GPU: torch.cuda.max_memory_allocated().
            "peak_cpu_total_mb": round(_peak_cpu_total_bytes / 1024**2, 3),
            "peak_cpu_net_mb": round(profiler.cpu_net_peak_memory_bytes() / 1024**2, 3),
            "peak_cpu_net_from_start_mb": round(
                max(0, _peak_cpu_total_bytes - _pre_load_baseline_bytes) / 1024**2, 3
            ),
            "peak_gpu_torch_mb": (
                round(profiler.latest_scalar("gpu_peak_memory_bytes") / 1024**2, 3)
                if profiler.latest_scalar("gpu_peak_memory_bytes") is not None else None
            ),
            "train_gpu_peak_mb": (
                round(max(
                    e["train_gpu_peak_bytes"] for e in epoch_series
                    if e.get("train_gpu_peak_bytes") is not None
                ) / 1024**2, 3)
                if any(e.get("train_gpu_peak_bytes") for e in epoch_series) else None
            ),
            "val_gpu_peak_mb": (
                round(max(
                    e["val_gpu_peak_bytes"] for e in epoch_series
                    if e.get("val_gpu_peak_bytes") is not None
                ) / 1024**2, 3)
                if any(e.get("val_gpu_peak_bytes") for e in epoch_series) else None
            ),
        }

        metrics_payload = {
            "_field_descriptions": {
                "peak_model_param_mb": "Largest single resident segment's parameter bytes (the model footprint the streaming path must hold at once).",
                "peak_cpu_total_mb": "Raw peak process RSS (baseline-independent, includes Python/torch overhead + resident segment weights).",
                "peak_cpu_net_mb": "peak RSS - baseline captured AFTER setup (EXCLUDES resident state; back-compat).",
                "peak_cpu_net_from_start_mb": "peak RSS - baseline captured BEFORE model/segments built (FAIR: INCLUDES resident weights).",
                "peak_gpu_torch_mb": "Peak torch.cuda.max_memory_allocated (per-step reset; last step's peak).",
                "train_gpu_peak_mb": "Max per-step VRAM peak across all training steps (per-step reset).",
                "val_gpu_peak_mb": "Peak VRAM during validation forward (reset before val loop).",
            },
            "run_summary": run_summary,
            "epoch_series": epoch_series,
            "profiler": profiler.to_dict(),
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "w") as f:
            json.dump(metrics_payload, f, indent=2, default=str)
        print(f"  Metrics JSON → {args.metrics_output}")


if __name__ == "__main__":
    main()

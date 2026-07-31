"""
Shared plumbing for the full-model baseline scripts.

This module is imported by every full_model script. It centralizes:
  * repo paths (tokenizer, data, outputs)
  * the GPTTransformer config (the single source of truth for the model)
  * builders: tokenizer, model, dataset, collator
  * memory instrumentation: CPU RAM (RSS) and GPU VRAM samplers

It ONLY reads the existing `src/` package (GPTDecoder, dataset, collator); it
never modifies anything outside `final_paper_scripts_results/full_model/`.

The conceptual model name in the plan is "GPTTransformer"; the concrete class in
the existing package is `GPTDecoder` (a gpt_decoder architecture).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Paths. REPO_ROOT is 3 parents up from this file:
#   .../full_model/scripts/_common.py
#   parents[0]=scripts  [1]=full_model  [2]=final_paper_scripts_results  [3]=repo
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BPE_TOKENIZER_DIR = REPO_ROOT / "bpe_tokenizer"     # compact 2k byte-level BPE (small model)
GPT2_TOKENIZER_DIR = REPO_ROOT / "gpt2_tokenizer"   # GPT-2 50257-vocab (large model)
TOKENIZER_DIR = BPE_TOKENIZER_DIR                   # default = small
DATA_DIR = REPO_ROOT / "log_lines" / "generative_splits"
FULL_MODEL_DIR = REPO_ROOT / "final_paper_scripts_results" / "full_model"
OUTPUTS_DIR = FULL_MODEL_DIR / "outputs"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

# --------------------------------------------------------------------------- #
# Two GPTTransformer configs (see plan §2). The baseline uses DIFFERENT models
# for accuracy vs cost, on purpose:
#   SMALL  -> accuracy training (A1) + real inference (B1). Small so the task has
#            headroom (the easy 154-label task isn't trivially solved). BPE-2k.
#   LARGE  -> cost runs only (A2 train-cost, B2 infer-cost, C ONNX-cost). Big so
#            the model's OWN memory dominates the ~500MB fixed CUDA context, making
#            full-vs-segmented memory differences clearly measurable. GPT-2 vocab.
# --------------------------------------------------------------------------- #
SMALL_CONFIG: Dict[str, Any] = {
    "architecture": "gpt_decoder",
    "max_seq_len": 128,   # data is short (BPE p95~84, p99~149); 128 covers ~97.5%
    "n_layers": 4,
    "d_model": 64,
    "n_heads": 4,   # head_dim = 16
    "d_ff": 256,    # 4 x d_model
    "dropout": 0.1,
    "tokenizer": "bpe",   # bpe_tokenizer/ (vocab 2000)
}

LARGE_CONFIG: Dict[str, Any] = {
    "architecture": "gpt_decoder",
    "max_seq_len": 512,
    "n_layers": 36,
    "d_model": 1280,
    "n_heads": 20,   # head_dim = 64
    "d_ff": 5120,    # 4 x d_model
    "dropout": 0.1,
    "tokenizer": "gpt2",  # gpt2_tokenizer/ (vocab 50257)  ~838M params
}

# Default config = SMALL (used by A1/B1 scripts unchanged). Cost scripts pass LARGE.
MODEL_CONFIG: Dict[str, Any] = SMALL_CONFIG

# Sequence / objective settings (see plan §4). Matches the project's existing
# train_normal.py / train_segmented.py so the baseline is comparable to the
# segmented runs. NOTE: GPT-2 has bos==eos==pad (50256); BOS is therefore NOT
# prepended (add_bos=False) — otherwise the padding mask would mask position 0
# and the attention softmax would produce NaN. pad_token = eos_token; vocab 50257.
PROMPT_TEMPLATE = "{data}\nLabel:"
TARGET_PREFIX = " "
ADD_BOS = False
ADD_EOS = True

# Batch sizes (see plan §5.3).
TRAIN_BATCH_SIZE = 64        # A1 accuracy training
INFER_BATCH_SIZE = 256       # B1 real inference (large batch = faster; accuracy is batch-independent)
COST_BATCH_SIZE = 4          # A2 / B2 / C cost runs (kept fixed)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_tokenizer(config: Dict[str, Any] = MODEL_CONFIG):
    """Load the tokenizer for `config` (bpe -> bpe_tokenizer/, gpt2 -> gpt2_tokenizer/).

    Defaults to the SMALL config's BPE tokenizer so A1/B1 scripts work unchanged.
    """
    from transformers import AutoTokenizer

    which = config.get("tokenizer", "bpe")
    tok_dir = GPT2_TOKENIZER_DIR if which == "gpt2" else BPE_TOKENIZER_DIR
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token   # GPT-2 has no pad; BPE already has a distinct pad
    return tok


def build_model(vocab_size: int, config: Dict[str, Any] = MODEL_CONFIG):
    """Instantiate the GPTTransformer (GPTDecoder) with `config` (default SMALL)."""
    from sequential_segmented_llm_training_inference.model.full_model import GPTDecoder

    return GPTDecoder(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    )


def build_dataset(csv_path: Path, max_rows: Optional[int] = None):
    """Lazy generative dataset over (log_line, label)."""
    from sequential_segmented_llm_training_inference.data.dataset import LogLabelDataset

    return LogLabelDataset(
        csv_path,
        data_column="log_line",
        label_column="label",
        max_rows=max_rows,
    )


def build_collator(tokenizer, config: Dict[str, Any] = MODEL_CONFIG):
    """Causal-LM collator with prompt masking (loss on label tokens only)."""
    from sequential_segmented_llm_training_inference.data.collator import (
        LogGenerationCollator,
    )

    return LogGenerationCollator(
        tokenizer=tokenizer,
        max_length=config["max_seq_len"],
        prompt_template=PROMPT_TEMPLATE,
        target_prefix=TARGET_PREFIX,
        add_bos=ADD_BOS,
        add_eos=ADD_EOS,
    )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Memory instrumentation (see plan §5.4)
#
# CPU runs  -> host CPU RAM (process RSS) only.
# GPU runs  -> 4 VRAM categories + host CPU RAM, reported separately:
#   1. torch-allocated        torch.cuda.memory_allocated
#   2. torch-peak-allocated   torch.cuda.max_memory_allocated
#   3. torch-reserved         torch.cuda.memory_reserved
#   4. process VRAM           nvidia-smi (headline)
# --------------------------------------------------------------------------- #
def _process_rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1024**2


@dataclass
class CpuRamSampler:
    """Background poller of process RSS (host CPU RAM) in MB."""

    interval_s: float = 0.2
    peak_mb: float = 0.0
    timeline: List[Tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _t0: float = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = _process_rss_mb()
            self.peak_mb = max(self.peak_mb, rss)
            self.timeline.append((time.perf_counter() - self._t0, rss))
            self._stop.wait(self.interval_s)

    def start(self) -> "CpuRamSampler":
        self._t0 = time.perf_counter()
        self.peak_mb = _process_rss_mb()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_mb

    def current_mb(self) -> float:
        return _process_rss_mb()


def _nvidia_smi_process_vram_mb() -> float:
    """VRAM (MB) used by THIS process, queried from nvidia-smi.

    Falls back to 0.0 if nvidia-smi is unavailable.
    """
    import os

    pid = os.getpid()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return 0.0
    total = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                total += float(parts[1])
            except ValueError:
                pass
    return total


@dataclass
class GpuVramSampler:
    """Background poller of process VRAM (nvidia-smi) in MB."""

    interval_s: float = 0.2
    peak_mb: float = 0.0
    timeline: List[Tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _t0: float = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            vram = _nvidia_smi_process_vram_mb()
            self.peak_mb = max(self.peak_mb, vram)
            self.timeline.append((time.perf_counter() - self._t0, vram))
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuVramSampler":
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_mb


def torch_vram_mb(device) -> Dict[str, float]:
    """The 3 torch VRAM categories (MB) for `device`."""
    import torch

    if device is None or torch.device(device).type != "cuda":
        return {"torch_alloc_mb": 0.0, "torch_peak_alloc_mb": 0.0, "torch_reserved_mb": 0.0}
    return {
        "torch_alloc_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "torch_peak_alloc_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "torch_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
    }

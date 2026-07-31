#!/usr/bin/env python3
"""
Autoregressive inference using a trained sequential segmented model.

Loads a segmented checkpoint and generates label text for log lines using
the same strict single-segment forward execution used during training.
No full-model reconstruction is needed — the segmented forward path is used
directly for generation.

Usage:
    # Interactive: read log lines from stdin
    python scripts/infer_segmented.py --checkpoint checkpoints/best

    # Single prompt from command line
    python scripts/infer_segmented.py --checkpoint checkpoints/best \\
        --prompt "{__time=..., instance=NODE[1]} some log message"

    # Batch evaluation on the test CSV
    python scripts/infer_segmented.py --checkpoint checkpoints/best \\
        --evaluate-test
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Union

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

_PAPER_REPORT_DIR = REPO_ROOT / "paper_report_codes_for_segmented_gpt"
if _PAPER_REPORT_DIR.exists() and str(_PAPER_REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_PAPER_REPORT_DIR))

try:
    from paper_report.inference_metrics import InferenceMetricTracker, write_inference_summary_csv
    _HAS_PAPER_REPORT = True
except ImportError:
    _HAS_PAPER_REPORT = False

import torch
from torch import Tensor

from transformers import AutoTokenizer

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import (
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.execution.composition import (
    concatenate_attention_outputs,
    residual_add,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import (
    SegmentedForwardEngine,
)
from sequential_segmented_llm_training_inference.execution.segment_loader import (
    StrictSegmentLoader,
)
from sequential_segmented_llm_training_inference.inference.generation import (
    GenerationConfig,
    SegmentedAutoregressiveGenerator,
)
from sequential_segmented_llm_training_inference.model.embeddings import EmbeddingConfig
from sequential_segmented_llm_training_inference.model.output_head import OutputHeadConfig
from sequential_segmented_llm_training_inference.segments.global_segments import (
    AttentionOutputProjSegmentModule,
    EmbeddingSegmentModule,
    EmbeddingSliceSegmentModule,
    OutputHeadSegmentModule,
    OutputHeadSliceSegmentModule,
)
from sequential_segmented_llm_training_inference.segments.segment_factory import (
    SegmentFactory,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.checkpoint_store import (
    SegmentedCheckpointStore,
)
from sequential_segmented_llm_training_inference.storage.manifest import (
    MANIFEST_FILENAME,
    CheckpointManifest,
)
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import (
    CpuRamSegmentStore,
)
from sequential_segmented_llm_training_inference.storage.disk_segment_store import (
    DiskSegmentStore,
)

SegmentStore = Union[CpuRamSegmentStore, DiskSegmentStore]


# ---------------------------------------------------------------------------
# Resumable segment-by-segment inference
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    if _psutil is None:
        return 0.0
    try:
        return _psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return 0.0


def _trim_cpu(device=None) -> None:
    import ctypes, gc, sys
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    if device is not None and getattr(device, "type", None) == "cuda":
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass


@dataclass
class _ResumableState:
    """All state needed to resume a forward pass at any segment boundary.

    Attention segment outputs (h_i) are stored as individual files in
    state_dir/_ha{layer}_{seg}.pt and are NOT carried in partial_attn.
    partial_attn is kept in the dataclass for API compatibility but is
    always an empty list.
    """
    generated_ids: list           # full token sequence (prompt + generated so far)
    n_to_generate: int            # remaining tokens to emit
    hidden: Tensor | None         # [1, S, D] on CPU — current hidden states
    partial_attn: list            # unused — kept for backward compat; always []
    mlp_acc: Tensor | None        # [1, S, D] on CPU — running MLP chunk sum
    phase: str                    # "embedding"|"attn"|"attn_proj"|"mlp"|"output_head"|"done"
    layer: int                    # current layer id (-1 for embedding/output_head)
    seg: int                      # segment index within current phase


class ResumableSegmentedGenerator:
    """
    Generates tokens one segment at a time.

    After each segment the full state is written to disk.  The process can
    exit and be restarted at any point — calling ``step()`` again picks up
    exactly where it left off.

    Usage::

        gen = ResumableSegmentedGenerator(engine, loader, device, state_dir, eos_id)
        gen.init_generation(prompt_ids, n_to_generate=20)
        while not gen.step():   # step() returns True when done
            pass                # or exit here and re-launch next time
        final_ids = gen.get_generated_ids()
    """

    def __init__(
        self,
        forward_engine: SegmentedForwardEngine,
        loader,
        device: torch.device,
        state_dir: Path,
        eos_token_id: int | None = None,
        profile: bool = False,
        in_memory: bool = False,
    ) -> None:
        self.engine = forward_engine
        self.loader = loader
        self.device = device
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.eos_token_id = eos_token_id
        self._state_path = self.state_dir / "state.pt"
        cfg = forward_engine.segmentation_config
        self._n_layers = forward_engine.model_config.n_layers
        self._n_attn = cfg.attention_segments
        self._n_mlp = cfg.mlp_chunks
        self._profile = profile
        self._profile_events: list = []
        self._t0 = time.perf_counter()
        self._token_step = 0
        # in_memory=True: keep state + attention-head outputs in CPU RAM
        # instead of writing to disk after every segment step. Eliminates
        # Lustre I/O (92k+ file ops for 50 examples × 40 tokens × 46 segs).
        # Disk persistence is kept only when an explicit --state-dir is given.
        self._in_memory = in_memory
        self._mem_state: dict | None = None
        self._ha_cache: dict = {}

    def init_generation(self, prompt_ids: list, n_to_generate: int) -> None:
        """Initialise (or reset) state for a new generation."""
        state = _ResumableState(
            generated_ids=list(prompt_ids),
            n_to_generate=n_to_generate,
            hidden=None,
            partial_attn=[],
            mlp_acc=None,
            phase="embedding",
            layer=0,
            seg=0,
        )
        self._token_step = 0
        self._save(state)

    def step(self) -> bool:
        """Execute one segment, persist state to disk. Returns True when done."""
        state = self._load()
        if state.phase == "done":
            return True

        rss_before = _rss_mb() if self._profile else 0.0
        self._execute_one(state)
        rss_after = _rss_mb() if self._profile else 0.0

        if self._profile:
            self._profile_events.append({
                "elapsed_s": round(time.perf_counter() - self._t0, 4),
                "token_step": self._token_step,
                "phase": state.phase,
                "layer": state.layer,
                "seg": state.seg,
                "rss_before_mb": round(rss_before, 3),
                "rss_after_mb": round(rss_after, 3),
                "delta_mb": round(rss_after - rss_before, 3),
            })

        # Track when a token is emitted (output_head step increments token_step)
        if state.phase == "embedding" and state.layer == 0 and state.seg == 0:
            self._token_step += 1

        self._save(state)
        return state.phase == "done"

    def generate_all(self) -> list:
        """Run all segment steps to completion and return generated_ids."""
        while not self.step():
            pass
        return self.get_generated_ids()

    def get_generated_ids(self) -> list:
        return self._load().generated_ids

    def get_profile_events(self) -> list:
        """Return per-segment RSS events recorded during generation."""
        return list(self._profile_events)

    def reset_profile(self) -> None:
        """Clear per-segment events (call between examples to keep per-example data)."""
        self._profile_events = []
        self._t0 = time.perf_counter()
        self._token_step = 0

    # ------------------------------------------------------------------
    # Single-segment execution
    # ------------------------------------------------------------------

    def _execute_one(self, state: _ResumableState) -> None:
        engine = self.engine
        dev = self.device

        if state.phase == "embedding":
            ids = torch.tensor([state.generated_ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                hidden = engine._run_embeddings(ids, None)
            state.hidden = hidden.cpu()
            del hidden
            state.phase = "attn"
            state.layer = 0
            state.seg = 0
            state.partial_attn = []

        elif state.phase == "attn":
            # P3: compute attn_in fresh each call (cheap for B=1); save h_i to disk.
            # No partial_attn list — each output is its own file in state_dir.
            layer_id, seg_idx = state.layer, state.seg
            hidden = state.hidden.to(dev)
            with torch.no_grad():
                attn_in = engine.layer_norms.attention(layer_id, hidden)
            del hidden
            seg_id = SegmentId(layer_id=layer_id, segment_type="attention", segment_id=seg_idx)
            with self.loader.acquire_segment(seg_id) as seg:
                with torch.no_grad():
                    h_i = seg(attn_in, None)
            del attn_in; _trim_cpu(dev)
            if self._in_memory:
                self._ha_cache[f"{layer_id}_{seg_idx}"] = h_i.cpu()
            else:
                torch.save(h_i.cpu(), self.state_dir / f"_ha{layer_id}_{seg_idx}.pt")
            del h_i
            state.seg += 1
            if state.seg >= self._n_attn:
                state.phase = "attn_proj"
                state.seg = 0

        elif state.phase == "attn_proj":
            # P3: load W_o once; load each h_i from disk one at a time.
            # Chunked seq matmul → no full [B,S,D] temp; in-place residual.
            # Entire phase under no_grad: weight views (W_sl) have requires_grad=True;
            # without no_grad they contaminate proj_acc → hidden via in-place ops.
            layer_id = state.layer
            n_attn   = self._n_attn
            D        = engine.model_config.d_model
            col_size = D // n_attn
            SEQ_CHUNK = 32
            proj_acc: Tensor | None = None

            with torch.no_grad():
                if engine._use_segment_store_for_global and engine._attn_proj_seg_ids is not None:
                    seg_id = engine._attn_proj_seg_ids[layer_id]
                    with self.loader.acquire_segment(seg_id) as proj_mod:
                        W    = proj_mod.proj.weight.detach()    # [D, D]
                        bias = (proj_mod.proj.bias.detach()
                                if proj_mod.proj.bias is not None else None)
                        for seg_idx in range(n_attn):
                            if self._in_memory:
                                h_i = self._ha_cache.pop(f"{layer_id}_{seg_idx}").to(dev)
                            else:
                                h_i = torch.load(
                                    self.state_dir / f"_ha{layer_id}_{seg_idx}.pt",
                                    map_location=dev,
                                )
                            B, S, Dn = h_i.shape
                            if proj_acc is None:
                                proj_acc = torch.zeros(B, S, D, dtype=h_i.dtype, device=dev)
                                if bias is not None:
                                    proj_acc.add_(bias)
                            W_sl = W[:, seg_idx * col_size : (seg_idx + 1) * col_size]
                            for t0 in range(0, S, SEQ_CHUNK):
                                t1 = min(t0 + SEQ_CHUNK, S)
                                proj_acc[:, t0:t1, :].add_(h_i[:, t0:t1, :] @ W_sl.T)
                            del h_i; _trim_cpu(dev)
                            if not self._in_memory:
                                (self.state_dir / f"_ha{layer_id}_{seg_idx}.pt").unlink(missing_ok=True)
                else:
                    if self._in_memory:
                        parts = [self._ha_cache.pop(f"{layer_id}_{i}").to(dev) for i in range(n_attn)]
                    else:
                        parts = [
                            torch.load(self.state_dir / f"_ha{layer_id}_{i}.pt", map_location=dev)
                            for i in range(n_attn)
                        ]
                    proj_acc = engine._run_attention_output_proj(
                        layer_id, concatenate_attention_outputs(parts)
                    )
                    del parts
                    if not self._in_memory:
                        for i in range(n_attn):
                            (self.state_dir / f"_ha{layer_id}_{i}.pt").unlink(missing_ok=True)

                proj_acc = engine.residual_dropout(proj_acc)
                hidden = state.hidden.to(dev)
                hidden.add_(proj_acc)           # in-place residual — no third tensor
                del proj_acc

            state.hidden = hidden.cpu()
            del hidden; _trim_cpu(dev)
            state.mlp_acc = None
            state.phase = "mlp"
            state.seg = 0

        elif state.phase == "mlp":
            # P3: compute mlp_in fresh each call (cheap for B=1).
            # In-place mlp_acc accumulation avoids three-tensor spike.
            # In-place residual at layer end avoids allocation.
            layer_id, seg_idx = state.layer, state.seg
            hidden = state.hidden.to(dev)
            with torch.no_grad():
                mlp_input = engine.layer_norms.mlp(layer_id, hidden)
            del hidden
            seg_id = SegmentId(layer_id=layer_id, segment_type="mlp", segment_id=seg_idx)
            with self.loader.acquire_segment(seg_id) as seg:
                with torch.no_grad():
                    out = seg(mlp_input)
            del mlp_input; _trim_cpu(dev)
            if state.mlp_acc is None:
                state.mlp_acc = out.cpu()
            else:
                state.mlp_acc.add_(out.cpu())  # in-place: no third tensor
            del out
            state.seg += 1

            if state.seg >= self._n_mlp:
                with torch.no_grad():
                    hidden      = state.hidden.to(dev)
                    mlp_acc_dev = state.mlp_acc.to(dev)
                    mlp_sum     = engine.mlp_shared_output_biases(layer_id, mlp_acc_dev)
                    del mlp_acc_dev
                    mlp_sum = engine.residual_dropout(mlp_sum)
                    hidden.add_(mlp_sum)    # in-place residual — no third tensor
                    del mlp_sum
                state.hidden = hidden.cpu()
                del hidden; _trim_cpu(dev)
                state.mlp_acc = None
                state.layer += 1
                if state.layer >= self._n_layers:
                    state.phase = "output_head"
                    state.layer = -1
                else:
                    state.phase = "attn"
                state.seg = 0

        elif state.phase == "output_head":
            hidden = state.hidden.to(dev)
            last_h = hidden[:, -1:, :]  # [1, 1, D] — only last position
            del hidden
            with torch.no_grad():
                logits = engine._run_output_head(last_h)  # [1, 1, vocab]
            del last_h
            if logits.device.type != "cpu":
                logits = logits.cpu()
            next_token = int(logits[0, 0].argmax().item())
            del logits
            _trim_cpu(dev)
            state.generated_ids.append(next_token)
            state.n_to_generate -= 1
            if (
                state.n_to_generate <= 0
                or (self.eos_token_id is not None and next_token == self.eos_token_id)
            ):
                state.phase = "done"
            else:
                # Next token step — reset for a fresh forward pass
                state.hidden = None
                state.partial_attn = []
                state.mlp_acc = None
                state.phase = "embedding"
                state.layer = 0
                state.seg = 0

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save(self, state: _ResumableState) -> None:
        data: dict = {
            "generated_ids": state.generated_ids,
            "n_to_generate": state.n_to_generate,
            "mlp_acc": state.mlp_acc,
            "phase": state.phase,
            "layer": state.layer,
            "seg": state.seg,
        }
        if state.hidden is not None:
            data["hidden"] = state.hidden
        if self._in_memory:
            self._mem_state = data
        else:
            torch.save(data, self._state_path)

    def _load(self) -> _ResumableState:
        if self._in_memory:
            data = self._mem_state
        else:
            data = torch.load(self._state_path, map_location="cpu", weights_only=False)
        return _ResumableState(
            generated_ids=data["generated_ids"],
            n_to_generate=data["n_to_generate"],
            hidden=data.get("hidden"),
            partial_attn=data.get("partial_attn", []),
            mlp_acc=data.get("mlp_acc"),
            phase=data["phase"],
            layer=data["layer"],
            seg=data["seg"],
        )


try:
    import psutil as _psutil
except Exception:
    _psutil = None  # type: ignore


TOKENIZER_DIR = REPO_ROOT / "gpt2_tokenizer"
DATA_DIR = REPO_ROOT / "log_lines" / "generative_splits"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segmented inference for log-line-to-label generation.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Checkpoint store root directory (contains 'best' and 'last' sub-dirs).",
    )
    p.add_argument(
        "--checkpoint-name",
        default="best",
        help="Name of checkpoint to load (default: best).",
    )
    p.add_argument("--prompt", type=str, default=None, help="Single log line to classify.")
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Run generation on test.csv and print accuracy.",
    )
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample", action="store_true", help="Enable sampling instead of greedy.")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--num-test-examples",
        type=int,
        default=50,
        help="Number of test examples to evaluate (--evaluate-test).",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Number of examples processed simultaneously in --evaluate-test "
             "(batch inference via [B,S,D] tensors). Default=1 (sequential).",
    )
    p.add_argument(
        "--storage",
        choices=["cpu_ram", "disk"],
        default="cpu_ram",
        help="Segment storage backend. 'disk' keeps only one segment in RAM/VRAM "
             "at a time; 'cpu_ram' pre-loads all segments into CPU RAM (faster "
             "but higher CPU memory footprint).",
    )
    p.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="Optional path to write a memory_trace.csv (elapsed_s, cpu_mb, gpu_mb).",
    )
    p.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to write canonical inference_summary.csv (requires paper_report).",
    )
    p.add_argument(
        "--run-name",
        default="segmented_inference",
        help="Run name recorded in inference_summary.csv.",
    )
    p.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional path to write a metrics.json with peak memory stats.",
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory for segment-step state files (enables resumable inference). "
             "If not set, a temporary directory is used.",
    )
    p.add_argument(
        "--profile-segments",
        action="store_true",
        help="Record RSS before/after each segment step for per-segment memory analysis.",
    )
    p.add_argument(
        "--segment-profile-output",
        type=Path,
        default=None,
        help="Path to write per-segment RSS profile CSV (requires --profile-segments).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model restoration from checkpoint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Memory monitoring
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
            rss = sum(p.memory_info().rss for p in [root] + root.children(recursive=True)
                      if hasattr(p, "memory_info"))
            cpu_mb = rss / 1024 ** 2
        except Exception:
            cpu_mb = 0.0
        gpu_mb = _gpu_mb_for_pid(pid)
        timeseries.append((time.perf_counter() - t0, cpu_mb, gpu_mb))
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Model restoration from checkpoint
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str,
    device: torch.device,
    storage: str = "cpu_ram",
    _disk_dir: Path | None = None,
) -> tuple[SegmentedForwardEngine, StrictSegmentLoader, SegmentStore, ModelConfig, SegmentationConfig]:
    """
    Load a segmented checkpoint and restore the full forward engine.

    Args:
        storage: 'cpu_ram' keeps all segment weights in CPU RAM (faster);
                 'disk' streams segments one at a time directly from the
                 checkpoint's own segments/ directory (lowest peak memory —
                 only ONE segment is ever resident in RAM, especially for CPU
                 inference). The 'disk' path does a lightweight load that never
                 materialises the whole model.
        _disk_dir: Unused for the 'disk' path (kept for back-compat); segments
                 are streamed from the checkpoint's own segments/ dir, so there
                 is no re-spill to a separate directory.
    Returns:
        forward_engine, loader, store, model_config, seg_config
    """
    ckpt_dir = SegmentedCheckpointStore(checkpoint_dir).checkpoint_dir(checkpoint_name)
    _global_seg_types = {"embedding", "output_head", "attention_output_proj"}

    if storage == "disk":
        # ---- Lightweight (lazy) disk load -------------------------------------
        # Do NOT call load_checkpoint(): that would materialise EVERY segment
        # tensor in RAM (~the whole model) just to re-spill them to a new disk
        # dir. The checkpoint already stores each segment as an individual file
        # under <ckpt_dir>/segments/<segment_id.to_path_name()>.pt — exactly the
        # layout DiskSegmentStore expects. So we read ONLY the small manifest +
        # the tiny non_segment_state file, then point a DiskSegmentStore directly
        # at the existing segments dir. Each segment is then read one at a time
        # during inference, so only ONE segment is ever resident in RAM.
        manifest = CheckpointManifest.load_yaml(ckpt_dir / MANIFEST_FILENAME)

        # Restore architecture from manifest metadata (no tensors loaded).
        model_config = ModelConfig(**manifest.model_config)
        seg_config = SegmentationConfig(**manifest.segmentation_config)

        # Detect global-in-store mode from the segment TYPES in the manifest
        # (matching how the checkpoint was trained) WITHOUT loading any tensors.
        use_segment_store_for_global = any(
            entry.segment_id.segment_type in _global_seg_types
            for entry in manifest.segment_entries
        )

        # Load ONLY the tiny non_segment_state (layer norms / biases / final
        # norm slices) directly from its small file.
        non_segment_state: dict = {}
        if manifest.non_segment_state_path is not None:
            non_segment_state = torch.load(
                ckpt_dir / manifest.non_segment_state_path,
                map_location="cpu",
                weights_only=False,
            )

        # Point at the checkpoint's OWN segments dir — no re-loading, no
        # re-spilling. _disk_dir is intentionally ignored for the disk path: the
        # whole point is to stream segments from where they already live.
        store: SegmentStore = DiskSegmentStore(ckpt_dir / "segments")

        print(f"  Loaded checkpoint '{checkpoint_name}' (lazy disk) — "
              f"epoch={manifest.epoch}, step={manifest.global_step}")
        if manifest.validation_metadata:
            print(f"  Validation metadata: {manifest.validation_metadata}")
    else:
        # ---- cpu_ram: intentionally hold ALL segments in RAM for speed --------
        ckpt_store = SegmentedCheckpointStore(checkpoint_dir)
        loaded = ckpt_store.load_checkpoint(checkpoint_name)
        checkpoint = loaded.checkpoint

        # Restore architecture from checkpoint metadata
        model_config = ModelConfig(**checkpoint.model_config)
        seg_config = SegmentationConfig(**checkpoint.segmentation_config)

        # Detect how the checkpoint was saved. Training (train_segmented.py)
        # builds the engine with use_segment_store_for_global=True, meaning the
        # embedding table, output head and per-layer attention output projections
        # live in the segment store (segment_states), NOT in non_segment_state.
        # Rebuilding the engine in legacy (always-resident) mode would (a)
        # materialise those large components full-size on the compute device all
        # at once and (b) leave them randomly initialised (their weights are
        # absent from non_segment_state) — producing garbage logits AND
        # full-model memory. Detect the mode from the segment types present so
        # inference matches how the checkpoint was trained.
        use_segment_store_for_global = any(
            seg_id.segment_type in _global_seg_types
            for seg_id in checkpoint.segment_states
        )

        store = CpuRamSegmentStore()
        for seg_id, state in checkpoint.segment_states.items():
            store.save_segment(seg_id, state)

        non_segment_state = checkpoint.non_segment_state
        print(f"  Loaded checkpoint '{checkpoint_name}' — epoch={checkpoint.epoch}, "
              f"step={checkpoint.global_step}")
        if checkpoint.validation_metadata:
            print(f"  Validation metadata: {checkpoint.validation_metadata}")

    # Build module factory
    factory = SegmentFactory(
        model_config=model_config,
        segmentation_config=seg_config,
        attention_dropout=0.0,   # no dropout at inference
        mlp_activation="gelu",
    )

    n_emb_segs = seg_config.embedding_segments
    n_head_segs = seg_config.output_head_segments

    def module_factory(segment_id: SegmentId) -> torch.nn.Module:
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
        if segment_id.segment_type == "embedding":
            if n_emb_segs == 1:
                emb_cfg = EmbeddingConfig(
                    vocab_size=model_config.vocab_size,
                    d_model=model_config.d_model,
                    max_seq_len=model_config.max_seq_len,
                    dropout=0.0,
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

    loader = StrictSegmentLoader(
        segment_store=store,
        module_factory=module_factory,
        device=device,
    )

    # Build forward engine and restore non-segment weights.
    # IMPORTANT: match the training-time mode so the embedding/output_head/
    # attention_output_proj are streamed from the segment store (not rebuilt
    # full-size and random in RAM/VRAM).
    forward_engine = SegmentedForwardEngine(
        model_config=model_config,
        segmentation_config=seg_config,
        segment_loader=loader,
        residual_dropout=0.0,   # no dropout at inference
        use_segment_store_for_global=use_segment_store_for_global,
    )
    forward_engine.to(device)

    ns = non_segment_state
    forward_engine.layer_norms.load_state_dict(ns["layer_norms"])
    forward_engine.mlp_shared_output_biases.load_state_dict(ns["mlp_shared_output_biases"])
    if hasattr(forward_engine, "output_slice_final_norm") and "output_slice_final_norm" in ns:
        forward_engine.output_slice_final_norm.load_state_dict(ns["output_slice_final_norm"])
    if not forward_engine._use_segment_store_for_global:
        if "embeddings" in ns:
            forward_engine.embeddings.load_state_dict(ns["embeddings"])
        if "attention_output_projections" in ns:
            forward_engine.attention_output_projections.load_state_dict(ns["attention_output_projections"])
        if "output_head" in ns:
            forward_engine.output_head.load_state_dict(ns["output_head"])

    forward_engine.eval()

    return forward_engine, loader, store, model_config, seg_config


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def make_segmented_forward_callable(forward_engine: SegmentedForwardEngine):
    """
    Return a callable compatible with SegmentedAutoregressiveGenerator.

    The generator calls: callable(input_ids=..., attention_mask=..., labels=None, ...)
    The forward engine expects: forward_engine(input_ids, attention_mask=..., ...)
    """
    @torch.no_grad()
    def forward_callable(
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels=None,
        store_runtime: bool = False,
        **kwargs,
    ):
        return forward_engine(
            input_ids,
            attention_mask=attention_mask,
            store_records=False,
        )

    return forward_callable


def generate_label(
    log_line: str,
    *,
    tokenizer,
    generator: SegmentedAutoregressiveGenerator,
    device: torch.device,
    prompt_template: str = "{log_line}\nLabel:",
) -> str:
    """
    Generate a label for one log line using the segmented forward engine.

    Returns:
        The decoded generated label text (stripped).
    """
    prompt = prompt_template.format(log_line=log_line)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = input_ids.size(1)

    generated = generator.generate(input_ids)

    # Decode only the newly generated tokens (after the prompt)
    new_token_ids = generated[0, prompt_len:].tolist()
    # Strip eos tokens from the end
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        while new_token_ids and new_token_ids[-1] == eos_id:
            new_token_ids.pop()

    label_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return label_text.strip()


def write_segment_profile_csv(events: list, path: Path) -> None:
    """Write per-segment RSS events to CSV."""
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "elapsed_s", "token_step", "phase", "layer", "seg",
            "rss_before_mb", "rss_after_mb", "delta_mb",
        ])
        writer.writeheader()
        writer.writerows(events)


def generate_label_resumable(
    log_line: str,
    *,
    tokenizer,
    resumable_gen: ResumableSegmentedGenerator,
    max_new_tokens: int,
    prompt_template: str = "{log_line}\nLabel:",
) -> str:
    """Generate a label via resumable single-segment-at-a-time execution."""
    prompt = prompt_template.format(log_line=log_line)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    resumable_gen.init_generation(prompt_ids, n_to_generate=max_new_tokens)
    all_ids = resumable_gen.generate_all()
    new_ids = all_ids[len(prompt_ids):]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        while new_ids and new_ids[-1] == eos_id:
            new_ids.pop()
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Batch inference (B examples through segments simultaneously, no state machine)
# ---------------------------------------------------------------------------

def generate_batch_labels(
    log_lines: list,
    *,
    tokenizer,
    engine: SegmentedForwardEngine,
    loader,
    device: torch.device,
    max_new_tokens: int,
    prompt_template: str = "{log_line}\nLabel:",
) -> list:
    """Generate labels for B examples simultaneously using [B,S,D] tensors.

    Bypasses the state machine — not resumable, no disk/mem state files.
    Each token step runs all B examples through every segment together.
    Satisfies P3: only one segment's weights resident at a time.
    """
    B = len(log_lines)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    n_layers = engine.model_config.n_layers
    n_attn = engine.segmentation_config.attention_segments
    n_mlp = engine.segmentation_config.mlp_chunks
    D = engine.model_config.d_model
    col_size = D // n_attn
    SEQ_CHUNK = 32

    prompts = [prompt_template.format(log_line=ll) for ll in log_lines]
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_len = max(len(e) for e in encoded)
    # Left-pad so all prompts end at the same position; last token = real last token
    seqs = [[pad_id] * (max_len - len(e)) + e for e in encoded]

    generated_new = [[] for _ in range(B)]
    done = [False] * B

    for _step in range(max_new_tokens):
        if all(done):
            break

        ids = torch.tensor(seqs, dtype=torch.long, device=device)  # [B, S]

        # Embedding
        with torch.no_grad():
            hidden = engine._run_embeddings(ids, None)  # [B, S, D]
        del ids

        # Transformer layers
        for layer_id in range(n_layers):
            # Attention segments: compute attn_in once, run each head segment
            with torch.no_grad():
                attn_in = engine.layer_norms.attention(layer_id, hidden)
            attn_outputs = []
            for seg_idx in range(n_attn):
                seg_id = SegmentId(layer_id=layer_id, segment_type="attention", segment_id=seg_idx)
                with loader.acquire_segment(seg_id) as seg:
                    with torch.no_grad():
                        h_i = seg(attn_in, None)  # [B, S, Dn]
                attn_outputs.append(h_i.cpu())
                del h_i; _trim_cpu(device)
            del attn_in

            # Attention output projection (chunked W_o; P3-safe: one segment resident)
            with torch.no_grad():
                if engine._use_segment_store_for_global and engine._attn_proj_seg_ids is not None:
                    B_cur, S_cur, _ = attn_outputs[0].shape
                    proj_acc = torch.zeros(B_cur, S_cur, D, dtype=attn_outputs[0].dtype, device=device)
                    seg_id = engine._attn_proj_seg_ids[layer_id]
                    with loader.acquire_segment(seg_id) as proj_mod:
                        W = proj_mod.proj.weight.detach()
                        bias = proj_mod.proj.bias.detach() if proj_mod.proj.bias is not None else None
                        if bias is not None:
                            proj_acc.add_(bias)
                        for si, h_i_cpu in enumerate(attn_outputs):
                            h_i = h_i_cpu.to(device)
                            W_sl = W[:, si * col_size : (si + 1) * col_size]
                            for t0 in range(0, S_cur, SEQ_CHUNK):
                                t1 = min(t0 + SEQ_CHUNK, S_cur)
                                proj_acc[:, t0:t1, :].add_(h_i[:, t0:t1, :] @ W_sl.T)
                            del h_i; _trim_cpu(device)
                else:
                    parts = [h.to(device) for h in attn_outputs]
                    proj_acc = engine._run_attention_output_proj(
                        layer_id, concatenate_attention_outputs(parts)
                    )
                    del parts
                del attn_outputs
                proj_acc = engine.residual_dropout(proj_acc)
                hidden.add_(proj_acc)
                del proj_acc
            _trim_cpu(device)

            # MLP segments
            with torch.no_grad():
                mlp_in = engine.layer_norms.mlp(layer_id, hidden)
            mlp_acc: "Tensor | None" = None
            for seg_idx in range(n_mlp):
                seg_id = SegmentId(layer_id=layer_id, segment_type="mlp", segment_id=seg_idx)
                with loader.acquire_segment(seg_id) as seg:
                    with torch.no_grad():
                        out = seg(mlp_in)
                out_cpu = out.cpu()
                del out; _trim_cpu(device)
                if mlp_acc is None:
                    mlp_acc = out_cpu
                else:
                    mlp_acc.add_(out_cpu)
                del out_cpu
            del mlp_in
            with torch.no_grad():
                mlp_dev = mlp_acc.to(device)
                del mlp_acc
                mlp_sum = engine.mlp_shared_output_biases(layer_id, mlp_dev)
                del mlp_dev
                mlp_sum = engine.residual_dropout(mlp_sum)
                hidden.add_(mlp_sum)
                del mlp_sum
            _trim_cpu(device)

        # Output head: last position only → next token per example
        last_h = hidden[:, -1:, :]  # [B, 1, D]
        del hidden
        with torch.no_grad():
            logits = engine._run_output_head(last_h)  # [B, 1, vocab]
        del last_h
        if logits.device.type != "cpu":
            logits = logits.cpu()
        next_tokens = logits[:, 0, :].argmax(dim=-1).tolist()  # [B]
        del logits; _trim_cpu(device)

        for b in range(B):
            if not done[b]:
                tok = next_tokens[b]
                if eos_id is not None and tok == eos_id:
                    done[b] = True
                    seqs[b].append(pad_id)  # keep lengths uniform even on EOS
                else:
                    generated_new[b].append(tok)
                    seqs[b].append(tok)
            else:
                seqs[b].append(pad_id)  # keep all sequences same length

    results = []
    for b in range(B):
        toks = list(generated_new[b])
        while toks and eos_id is not None and toks[-1] == eos_id:
            toks.pop()
        results.append(tokenizer.decode(toks, skip_special_tokens=True).strip())
    return results


# ---------------------------------------------------------------------------
# Evaluation on test CSV
# ---------------------------------------------------------------------------

def evaluate_test_csv(
    csv_path: Path,
    *,
    tokenizer,
    resumable_gen: "ResumableSegmentedGenerator | None" = None,
    device: torch.device,
    num_examples: int = 50,
    max_new_tokens: int = 40,
    tracker=None,
    eval_batch_size: int = 1,
    engine=None,
    loader=None,
) -> None:
    """Generate labels for test examples and compare with ground truth.

    eval_batch_size=1: sequential mode via resumable_gen (original).
    eval_batch_size>1: batch mode via generate_batch_labels (engine+loader required).
    """
    print(f"\nEvaluating on {csv_path.name} (up to {num_examples} examples, "
          f"batch_size={eval_batch_size}) ...")
    print("-" * 70)

    exact_matches = 0
    total = 0

    if eval_batch_size > 1:
        # ── Batch mode: collect rows, process B at a time ─────────────────────
        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if len(rows) >= num_examples:
                    break
                ll = row.get("log_line", "").strip()
                tl = row.get("label", "").strip()
                if ll and tl:
                    rows.append((ll, tl))

        for b_start in range(0, len(rows), eval_batch_size):
            batch = rows[b_start : b_start + eval_batch_size]
            log_lines_b = [r[0] for r in batch]
            true_labels_b = [r[1] for r in batch]

            t0 = time.perf_counter()
            pred_labels = generate_batch_labels(
                log_lines_b,
                tokenizer=tokenizer,
                engine=engine,
                loader=loader,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            elapsed_ms_total = (time.perf_counter() - t0) * 1000
            per_example_ms = elapsed_ms_total / len(batch)

            for pred_label, true_label, log_line in zip(pred_labels, true_labels_b, log_lines_b):
                match = pred_label.strip() == true_label.strip()
                if match:
                    exact_matches += 1
                total += 1
                if tracker is not None:
                    tracker.record_generation(
                        elapsed_ms=per_example_ms,
                        exact_match=match,
                        generated_tokens=len(tokenizer.encode(pred_label, add_special_tokens=False)),
                    )
                status = "✓" if match else "✗"
                log_snippet = log_line[:60].replace("\n", " ") + ("..." if len(log_line) > 60 else "")
                print(f"[{status}] LOG : {log_snippet}")
                print(f"     TRUE: {true_label}")
                print(f"     PRED: {pred_label}")
                print()

    else:
        # ── Sequential mode (original): one example at a time ─────────────────
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if total >= num_examples:
                    break
                log_line = row.get("log_line", "").strip()
                true_label = row.get("label", "").strip()
                if not log_line or not true_label:
                    continue

                t0 = time.perf_counter()
                pred_label = generate_label_resumable(
                    log_line,
                    tokenizer=tokenizer,
                    resumable_gen=resumable_gen,
                    max_new_tokens=max_new_tokens,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                match = pred_label.strip() == true_label.strip()
                if match:
                    exact_matches += 1
                total += 1

                if tracker is not None:
                    tracker.record_generation(
                        elapsed_ms=elapsed_ms,
                        exact_match=match,
                        generated_tokens=len(tokenizer.encode(pred_label, add_special_tokens=False)),
                    )

                status = "✓" if match else "✗"
                log_snippet = log_line[:60].replace("\n", " ") + "..." if len(log_line) > 60 else log_line
                print(f"[{status}] LOG : {log_snippet}")
                print(f"     TRUE: {true_label}")
                print(f"     PRED: {pred_label}")
                print()

    acc = exact_matches / total if total > 0 else 0.0
    print(f"Exact-match accuracy: {exact_matches}/{total} = {acc:.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # cpu_ram keeps all segment weights resident simultaneously — only valid on CUDA
    # where CPU RAM is not the constrained resource. Force disk on CPU.
    if device.type == "cpu" and args.storage == "cpu_ram":
        print("  [CPU] Forcing --storage=disk (cpu_ram requires CUDA; disk streams one segment at a time).")
        args.storage = "disk"

    # Capture a baseline BEFORE the model/checkpoint is loaded so that the
    # resident model footprint (segment store etc.) IS counted in the
    # "from-start" net memory. This is the fair baseline for comparing against
    # the full-model path, which must hold its weights resident.
    pre_load_baseline_mb = 0.0
    if _psutil is not None:
        try:
            pre_load_baseline_mb = _psutil.Process().memory_info().rss / 1024 ** 2
        except Exception:
            pass

    print("=" * 60)
    print("Sequential Segmented LLM — Inference")
    print("=" * 60)
    print(f"  Device : {device}  |  Storage : {args.storage}")

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print("\n[1] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab_size={tokenizer.vocab_size}, eos={tokenizer.eos_token_id}")

    # ------------------------------------------------------------------
    # 2. Load model from checkpoint
    # ------------------------------------------------------------------
    print(f"\n[2] Loading checkpoint '{args.checkpoint_name}' from {args.checkpoint} ...")
    with tempfile.TemporaryDirectory(prefix="seg_infer_disk_") as _tmp:
        disk_dir = Path(_tmp) if args.storage == "disk" else None
        forward_engine, loader, store, model_config, seg_config = load_model_from_checkpoint(
            checkpoint_dir=args.checkpoint,
            checkpoint_name=args.checkpoint_name,
            device=device,
            storage=args.storage,
            _disk_dir=disk_dir,
        )
        print(f"  Model   : {model_config.n_layers}L × {model_config.d_model}d  "
              f"({seg_config.attention_segments} attn + {seg_config.mlp_chunks} MLP segments/layer)")

        # ------------------------------------------------------------------
        # 3. Build generator (resumable segment-by-segment)
        # ------------------------------------------------------------------
        _state_dir_tmp = None
        if args.state_dir is not None:
            state_dir = args.state_dir
            state_dir.mkdir(parents=True, exist_ok=True)
        else:
            _state_dir_tmp = tempfile.mkdtemp(prefix="seg_infer_state_")
            state_dir = Path(_state_dir_tmp)

        # Use in-memory state when no explicit --state-dir was given:
        # eliminates Lustre I/O (92k+ file ops for 50 examples × 40 tokens).
        # Disk persistence is kept only when --state-dir is explicitly set
        # (fault-tolerant resumable inference).
        _use_in_memory = args.state_dir is None
        resumable_gen = ResumableSegmentedGenerator(
            forward_engine=forward_engine,
            loader=loader,
            device=device,
            state_dir=state_dir,
            eos_token_id=tokenizer.eos_token_id,
            profile=args.profile_segments,
            in_memory=_use_in_memory,
        )

        # ------------------------------------------------------------------
        # 4. Start memory monitor, then generate
        # ------------------------------------------------------------------
        import os
        # Capture baseline RSS now — after all setup, before any computation.
        cpu_baseline_mb = 0.0
        if _psutil is not None:
            try:
                cpu_baseline_mb = _psutil.Process().memory_info().rss / 1024 ** 2
            except Exception:
                pass

        timeseries: list[tuple[float, float, float | None]] = []
        stop_evt = Event()
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        monitor_thread = Thread(
            target=_monitor,
            args=(os.getpid(), stop_evt, 0.2, timeseries, t0),
            daemon=True,
        )
        monitor_thread.start()

        tracker = None
        if _HAS_PAPER_REPORT and args.summary_output is not None and args.evaluate_test:
            tracker = InferenceMetricTracker(
                run_name=args.run_name,
                method="segmented",
                device=str(device),
                storage_backend=args.storage,
            )

        if args.evaluate_test:
            evaluate_test_csv(
                DATA_DIR / "test.csv",
                tokenizer=tokenizer,
                resumable_gen=resumable_gen,
                device=device,
                num_examples=args.num_test_examples,
                max_new_tokens=args.max_new_tokens,
                tracker=tracker,
                eval_batch_size=args.eval_batch_size,
                engine=forward_engine,
                loader=loader,
            )

        elif args.prompt is not None:
            print(f"\nLog line: {args.prompt}")
            label = generate_label_resumable(
                args.prompt,
                tokenizer=tokenizer,
                resumable_gen=resumable_gen,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"Generated label: {label}")

        else:
            # Interactive mode: read log lines from stdin
            print("\nInteractive mode — enter a log line (Ctrl+C to quit):")
            try:
                while True:
                    print()
                    log_line = input("Log line> ").strip()
                    if not log_line:
                        continue
                    label = generate_label_resumable(
                        log_line,
                        tokenizer=tokenizer,
                        resumable_gen=resumable_gen,
                        max_new_tokens=args.max_new_tokens,
                    )
                    print(f"Label   > {label}")
            except (KeyboardInterrupt, EOFError):
                print("\nDone.")

        stop_evt.set()
        monitor_thread.join(timeout=2)

        # ------------------------------------------------------------------
        # 5. Report memory stats
        # ------------------------------------------------------------------
        if timeseries:
            peak_cpu = max(s[1] for s in timeseries)
            peak_cpu_net = max(max(0.0, s[1] - cpu_baseline_mb) for s in timeseries)
            # Fair "from-start" net: baseline taken BEFORE model load, so the
            # resident model/segment-store IS counted.
            peak_cpu_net_from_start = max(
                max(0.0, s[1] - pre_load_baseline_mb) for s in timeseries
            )
            gpu_vals = [s[2] for s in timeseries if s[2] is not None]
            peak_gpu = max(gpu_vals) if gpu_vals else None
            print("\n--- Memory stats (inference) ---")
            print(f"  Peak CPU RAM (raw, total RSS)  : {peak_cpu:.1f} MB")
            print(f"  Peak CPU net (-post-load base) : {peak_cpu_net:.1f} MB  [excludes model]")
            print(f"  Peak CPU net (-pre-load base)  : {peak_cpu_net_from_start:.1f} MB  [FAIR: includes model]")
            if peak_gpu is not None:
                print(f"  Peak GPU VRAM (nvidia-smi)     : {peak_gpu:.1f} MB")
            if device.type == "cuda":
                alloc_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                print(f"  Peak GPU allocated (torch)     : {alloc_mb:.1f} MB")

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

        if args.metrics_output is not None:
            import json as _json
            peak_cpu_m = max(s[1] for s in timeseries) if timeseries else None
            peak_cpu_net_m = (
                max(max(0.0, s[1] - cpu_baseline_mb) for s in timeseries)
                if timeseries else None
            )
            peak_cpu_net_from_start_m = (
                max(max(0.0, s[1] - pre_load_baseline_mb) for s in timeseries)
                if timeseries else None
            )
            gpu_vals_m = [s[2] for s in timeseries if s[2] is not None]
            peak_gpu_m = max(gpu_vals_m) if gpu_vals_m else None
            if device.type == "cuda" and torch.cuda.is_available():
                peak_gpu_torch_m = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            else:
                peak_gpu_torch_m = None
            # Largest single resident segment (streaming working set) AND the
            # total of all segments held in the store (full resident model size).
            peak_seg_mb = 0.0
            total_seg_mb = 0.0
            if hasattr(store, "list_segments"):
                for seg_id in store.list_segments():
                    state = store.load_segment(seg_id)
                    seg_bytes = sum(t.nbytes for t in state.values() if hasattr(t, "nbytes"))
                    peak_seg_mb = max(peak_seg_mb, seg_bytes / 1024 ** 2)
                    total_seg_mb += seg_bytes / 1024 ** 2
                    del state  # release before loading the next segment (P3: one at a time)
            metrics = {
                "_field_descriptions": {
                    "peak_model_param_mb": "Largest single resident segment (streaming working set); cpu_ram backend still holds ALL segments in store_total_param_mb.",
                    "store_total_param_mb": "Sum of all segment param bytes held resident in the store (= full model size for cpu_ram backend).",
                    "peak_cpu_ram_mb": "Raw peak process RSS (baseline-independent, includes Python/torch overhead + model).",
                    "peak_cpu_net_mb": "peak RSS - baseline captured AFTER model load (EXCLUDES model; legacy/back-compat).",
                    "peak_cpu_net_from_start_mb": "peak RSS - baseline captured BEFORE model load (FAIR: INCLUDES resident model/segment store).",
                    "peak_cpu_total_mb": "Alias of peak_cpu_ram_mb (raw peak RSS, no baseline subtraction).",
                    "peak_gpu_vram_mb": "Peak GPU VRAM from nvidia-smi (process-level).",
                    "peak_gpu_torch_mb": "Peak torch.cuda.max_memory_allocated (baseline-independent).",
                },
                "run_summary": {
                    "peak_model_param_mb": peak_seg_mb or None,
                    "store_total_param_mb": total_seg_mb or None,
                    "peak_cpu_ram_mb": peak_cpu_m,
                    "peak_cpu_total_mb": peak_cpu_m,
                    "peak_cpu_net_mb": peak_cpu_net_m,
                    "peak_cpu_net_from_start_mb": peak_cpu_net_from_start_m,
                    "peak_gpu_vram_mb": peak_gpu_m,
                    "peak_gpu_torch_mb": peak_gpu_torch_m,
                }
            }
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(_json.dumps(metrics, indent=2))
            print(f"  Metrics written : {args.metrics_output}")

        if args.profile_segments:
            events = resumable_gen.get_profile_events()
            out_path = args.segment_profile_output or (
                Path(args.trace_output).parent / "segment_profile.csv"
                if args.trace_output else Path("segment_profile.csv")
            )
            write_segment_profile_csv(events, out_path)
            print(f"  Segment profile : {out_path}  ({len(events)} events)")


if __name__ == "__main__":
    main()

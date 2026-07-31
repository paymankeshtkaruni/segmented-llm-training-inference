"""Training loop for true recomputation-based segmented LLM training.

The trainer intentionally does **not** call global ``loss.backward()`` on the
original forward graph. Training forward is executed under ``torch.no_grad()``
and runtime records are detached/offloaded. Backward is driven by a manual
reverse scheduler that locally recomputes one trainable unit at a time and uses
``torch.autograd.grad(...)`` only for that temporary local graph.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.execution.backward_engine import (
    LayerBackwardResult,
    SegmentedBackwardEngine,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import (
    SegmentedForwardEngine,
)
from sequential_segmented_llm_training_inference.execution.rng import restored_rng_state
from sequential_segmented_llm_training_inference.optimization.segment_optimizer import (
    SegmentOptimizer,
    validate_update_style_and_accumulation,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.runtime_records import (
    RuntimeRecordStore,
)
from sequential_segmented_llm_training_inference.training.losses import (
    CausalCrossEntropyLoss,
)
from sequential_segmented_llm_training_inference.config.training_config import UpdateStyle
from sequential_segmented_llm_training_inference.storage.disk_gradient_store import (
    DiskGradientStore,
)


@dataclass(slots=True)
class SegmentTrainBatch:
    """Batch object consumed by :class:`SegmentedTrainer`."""

    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor | None = None
    position_ids: Tensor | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SegmentTrainBatch":
        if "input_ids" not in data:
            raise KeyError("Batch mapping must contain 'input_ids'.")
        if "labels" not in data:
            raise KeyError("Batch mapping must contain 'labels'.")
        return cls(
            input_ids=data["input_ids"],
            labels=data["labels"],
            attention_mask=data.get("attention_mask"),
            position_ids=data.get("position_ids"),
        )

    def validate(self) -> None:
        if not isinstance(self.input_ids, Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if not isinstance(self.labels, Tensor):
            raise TypeError("labels must be a torch.Tensor.")
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len].")
        if self.labels.shape != self.input_ids.shape:
            raise ValueError(
                "labels must have the same [batch_size, seq_len] shape as input_ids."
            )
        if self.attention_mask is not None and self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        if self.position_ids is not None and self.position_ids.shape != self.input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids.")


@dataclass(slots=True)
class TrainingStepResult:
    """Summary of one trainer step."""

    loss: float
    global_step: int
    microbatch_index: int
    optimizer_step_applied: bool
    segment_update_count: int
    non_segment_update_applied: bool
    num_attention_records: int
    num_mlp_records: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_segment_records(self) -> int:
        return self.num_attention_records + self.num_mlp_records


@dataclass(slots=True)
class EpochTrainingResult:
    """Aggregated result for one training epoch."""

    epoch: int
    mean_loss: float
    steps: int
    optimizer_steps: int


def _batch_from_any(batch: SegmentTrainBatch | Mapping[str, Any]) -> SegmentTrainBatch:
    if isinstance(batch, SegmentTrainBatch):
        return batch
    if isinstance(batch, Mapping):
        return SegmentTrainBatch.from_mapping(batch)
    raise TypeError(
        "batch must be SegmentTrainBatch or a mapping with input_ids and labels."
    )


class SegmentedTrainer:
    """Orchestrates true recomputation-based segmented training steps.

    Training memory invariant:
    - the original forward pass is run under ``torch.no_grad()``;
    - stored runtime tensors are detached and can be offloaded to CPU;
    - no global full-model ``loss.backward()`` is called;
    - each backward unit creates only a temporary local autograd graph.
    """

    def __init__(
        self,
        *,
        forward_engine: SegmentedForwardEngine,
        backward_engine: SegmentedBackwardEngine,
        segment_optimizer: SegmentOptimizer,
        non_segment_optimizer: torch.optim.Optimizer | None = None,
        loss_fn: nn.Module | None = None,
        update_style: UpdateStyle = "after_full_backward",
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float | None = None,
        record_tensors_to_cpu: bool = True,
        record_backend: str = "in_memory",
        record_disk_dir: str | None = None,
        evict_layers_during_backward: bool = False,
        gradient_store: str = "in_memory",
        gradient_store_dir: str | None = None,
        store_only_layer_inputs: bool = False,
        output_head_token_chunks: int = 1,
    ) -> None:
        validate_update_style_and_accumulation(
            update_style=update_style,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        if gradient_clip_norm is not None and update_style == "immediate_segment_update":
            raise ValueError(
                "Exact global gradient clipping is only supported with "
                "update_style='after_full_backward'. Disable gradient_clip_norm or "
                "use after_full_backward."
            )

        self.forward_engine = forward_engine
        self.backward_engine = backward_engine
        self.segment_optimizer = segment_optimizer
        self.non_segment_optimizer = non_segment_optimizer or torch.optim.AdamW(
            self.forward_engine.parameters(), lr=1.0e-4
        )
        self.loss_fn = loss_fn or CausalCrossEntropyLoss(ignore_index=-100)
        self.update_style = update_style
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.record_tensors_to_cpu = record_tensors_to_cpu
        self.global_step = 0
        self.microbatch_index = 0
        self._accumulated_segment_gradients: dict[SegmentId, dict[str, Tensor]] = {}
        self._record_backend_type: str = record_backend
        self._record_disk_dir: str | None = record_disk_dir
        self._evict_layers_during_backward: bool = evict_layers_during_backward
        self._store_only_layer_inputs: bool = store_only_layer_inputs
        self._output_head_token_chunks: int = max(1, output_head_token_chunks)
        # Disk gradient store (only created when gradient_store == "disk")
        self._gradient_store_type: str = gradient_store
        if gradient_store == "disk":
            if not gradient_store_dir:
                raise ValueError("gradient_store_dir is required when gradient_store='disk'.")
            from pathlib import Path
            self._disk_grad_store: DiskGradientStore | None = DiskGradientStore(
                Path(gradient_store_dir)
            )
        else:
            self._disk_grad_store = None

    def _make_record_backend(self, step_id: int):
        """Create a fresh per-step record backend."""
        from pathlib import Path
        if self._record_backend_type == "disk":
            from sequential_segmented_llm_training_inference.storage.runtime_record_store_backends import (
                DiskRecordBackend,
            )
            if self._record_disk_dir is None:
                raise ValueError("record_disk_dir is required when record_backend='disk'.")
            step_dir = Path(self._record_disk_dir) / f"step_{step_id}"
            step_dir.mkdir(parents=True, exist_ok=True)
            return DiskRecordBackend(step_dir)
        else:
            from sequential_segmented_llm_training_inference.storage.runtime_record_store_backends import (
                InMemoryRecordBackend,
            )
            return InMemoryRecordBackend()

    @property
    def segment_loader(self):
        return self.forward_engine.segment_loader

    @property
    def device(self) -> torch.device:
        return self.segment_loader.device

    def train_step(self, batch: SegmentTrainBatch | Mapping[str, Any]) -> TrainingStepResult:
        """Run one segmented training step/microbatch without a full forward graph."""

        train_batch = _batch_from_any(batch)
        train_batch.validate()

        self.forward_engine.train()
        if self.microbatch_index % self.gradient_accumulation_steps == 0:
            self.non_segment_optimizer.zero_grad(set_to_none=True)
            if self.update_style == "after_full_backward":
                self._accumulated_segment_gradients.clear()
                if self._disk_grad_store is not None:
                    self._disk_grad_store.clear()

        # For N>1 output-head segments, skip building [B,S,vocab] in the forward
        # pass — the 206 MB logging tensor is recovered cheaply from the backward.
        _use_seg = getattr(self.forward_engine, "_use_segment_store_for_global", False)
        _n_head = len(self.forward_engine._output_head_seg_ids) if _use_seg else 1  # type: ignore[union-attr]
        _skip_output_head = _use_seg and _n_head > 1

        _record_backend = self._make_record_backend(self.global_step)
        _step_record_store = RuntimeRecordStore(backend=_record_backend)
        _step_dir = getattr(_record_backend, "root_dir", None)

        try:
            with torch.no_grad():
                output = self.forward_engine(
                    train_batch.input_ids.to(self.device),
                    attention_mask=(
                        None if train_batch.attention_mask is None
                        else train_batch.attention_mask.to(self.device)
                    ),
                    position_ids=(
                        None if train_batch.position_ids is None
                        else train_batch.position_ids.to(self.device)
                    ),
                    store_records=True,
                    step_id=self.global_step,
                    microbatch_id=self.microbatch_index,
                    detach_record_tensors=True,
                    clone_record_tensors=True,
                    record_tensors_to_cpu=self.record_tensors_to_cpu,
                    return_segment_outputs=False,
                    compute_output_head=not _skip_output_head,
                    runtime_records_store=_step_record_store,
                    store_only_layer_inputs=self._store_only_layer_inputs,
                )
            runtime_records = output.runtime_records
            if runtime_records is None:
                raise RuntimeError("Trainer requires forward runtime records.")
            self._assert_records_detached(runtime_records)

            # N==1 or non-segment path: compute logging loss from forward logits.
            # N>1: logits are empty; loss scalar comes back from the backward sweep.
            raw_loss_value = 0.0
            if not _skip_output_head and output.logits.numel() > 0 and self.loss_fn is not None:
                with torch.no_grad():
                    raw_loss = self.loss_fn(output.logits.detach(), train_batch.labels.to(output.logits.device))
                    if raw_loss.ndim != 0:
                        raise ValueError("Training loss must be scalar.")
                    raw_loss_value = float(raw_loss.cpu().item())

            segment_update_count, bwd_loss_scalar = self._run_true_recomputation_backward(
                runtime_records=runtime_records,
                labels=train_batch.labels,
                grad_hidden_dir=_step_dir,
            )
            if _skip_output_head:
                raw_loss_value = bwd_loss_scalar

            accumulation_boundary = (
                (self.microbatch_index + 1) % self.gradient_accumulation_steps == 0
            )
            non_segment_update_applied = False
            optimizer_step_applied = False

            gradient_global_norm: float | None = None

            if self.update_style == "after_full_backward":
                if accumulation_boundary:
                    if self.gradient_clip_norm is not None:
                        gradient_global_norm = self._clip_all_accumulated_gradients(
                            self.gradient_clip_norm
                        )
                    self._apply_accumulated_segment_updates()
                    self.non_segment_optimizer.step()
                    self.non_segment_optimizer.zero_grad(set_to_none=True)
                    self._accumulated_segment_gradients.clear()
                    if self._disk_grad_store is not None:
                        self._disk_grad_store.clear()
                    non_segment_update_applied = True
                    optimizer_step_applied = True
            else:
                # Segment updates were applied as soon as each segment's local
                # input gradient and parameter gradients were computed. Exact global
                # clipping is intentionally rejected in __init__ for this style.
                self.non_segment_optimizer.step()
                self.non_segment_optimizer.zero_grad(set_to_none=True)
                non_segment_update_applied = True
                optimizer_step_applied = True

            if self.segment_loader.active_segment_count != 0:
                raise RuntimeError("Strict segment loader still has an active segment after training step.")

            result = TrainingStepResult(
                loss=raw_loss_value,
                global_step=self.global_step,
                microbatch_index=self.microbatch_index,
                optimizer_step_applied=optimizer_step_applied,
                segment_update_count=segment_update_count,
                non_segment_update_applied=non_segment_update_applied,
                num_attention_records=len(runtime_records.attention_records),
                num_mlp_records=len(runtime_records.mlp_records),
                metadata={
                    "backward_mode": "true_recomputation_local_autograd",
                    "full_forward_backward_used": False,
                    "record_tensors_to_cpu": self.record_tensors_to_cpu,
                    "global_gradient_norm_before_clip": gradient_global_norm,
                    "gradient_clipping_scope": (
                        "true_global" if self.gradient_clip_norm is not None else "disabled"
                    ),
                },
            )

            runtime_records.clear()
            gc.collect()
            if sys.platform.startswith("linux"):
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
            self.global_step += 1
            self.microbatch_index += 1
            return result
        finally:
            # Clean up per-step disk directory on both success and exception.
            if _step_dir is not None and self._record_backend_type == "disk":
                import shutil as _shutil
                if _step_dir.exists():
                    _shutil.rmtree(_step_dir, ignore_errors=True)

    def train_epoch(
        self,
        batches: Iterable[SegmentTrainBatch | Mapping[str, Any]],
        *,
        epoch: int = 0,
    ) -> EpochTrainingResult:
        """Train over an iterable of batches and return aggregate loss."""

        losses: list[float] = []
        optimizer_steps = 0
        for batch in batches:
            step_result = self.train_step(batch)
            losses.append(step_result.loss)
            if step_result.optimizer_step_applied:
                optimizer_steps += 1

        if not losses:
            raise ValueError("train_epoch received no batches.")

        return EpochTrainingResult(
            epoch=epoch,
            mean_loss=sum(losses) / len(losses),
            steps=len(losses),
            optimizer_steps=optimizer_steps,
        )

    # ------------------------------------------------------------------
    # True recomputation backward scheduler
    # ------------------------------------------------------------------

    def _run_true_recomputation_backward(
        self,
        *,
        runtime_records: RuntimeRecordStore,
        labels: Tensor,
        grad_hidden_dir: "Path | None" = None,
    ) -> tuple[int, float]:
        """Run the full true-recomputation backward pass.

        grad_hidden_dir: when set (CPU disk mode), grad_hidden is saved to disk after
        each layer's backward. This satisfies the design invariant: training can stop at
        any layer boundary and be resumed by loading the file.
        """
        from pathlib import Path as _Path
        grad_hidden, bwd_loss_scalar = self._backward_output_head_from_loss(runtime_records, labels)
        segment_update_count = 0

        for layer_id in reversed(range(self.forward_engine.model_config.n_layers)):
            # Lazy-records: recompute all layer intermediates from the single stored
            # layer_X.input checkpoint before running the backward operations.
            if self._store_only_layer_inputs:
                self._recompute_layer_intermediates(layer_id, runtime_records)

            # MLP residual: y = post_attention_hidden + dropout(shared_bias(sum(chunks)))
            grad_post_attention_direct = grad_hidden.detach().clone()
            grad_mlp_dropout_out = grad_hidden.detach().clone()
            del grad_hidden  # consumed into clones; reassigned at end of loop

            grad_mlp_after_bias = self._backward_dropout(
                input_key=f"layer_{layer_id}.mlp_after_bias_pre_dropout",
                rng_key=f"layer_{layer_id}.mlp_dropout_rng_state",
                grad_output=grad_mlp_dropout_out,
                runtime_records=runtime_records,
            )
            if self._store_only_layer_inputs:
                runtime_records.pop_local_tensor(f"layer_{layer_id}.mlp_after_bias_pre_dropout")
            del grad_mlp_dropout_out

            grad_mlp_sum = self._backward_mlp_shared_bias(
                layer_id=layer_id,
                runtime_records=runtime_records,
                grad_output=grad_mlp_after_bias,
            )
            if self._store_only_layer_inputs:
                runtime_records.pop_local_tensor(f"layer_{layer_id}.mlp_sum_pre_bias")
            del grad_mlp_after_bias

            mlp_result = self.backward_engine.backward_mlp_layer(
                layer_id=layer_id,
                runtime_records=runtime_records,
                grad_mlp_output=grad_mlp_sum.detach().cpu(),
                overwrite_gradient_records=True,
            )
            if self._store_only_layer_inputs:
                runtime_records.pop_local_tensor(f"layer_{layer_id}.mlp_input")
            del grad_mlp_sum
            segment_update_count += self._handle_segment_backward_result(mlp_result)

            grad_post_attention_from_mlp = self._backward_layer_norm(
                module=self.forward_engine.layer_norms.mlp_layer_norms[layer_id],
                input_key=f"layer_{layer_id}.post_attention_hidden",
                grad_output=mlp_result.input_gradient,
                runtime_records=runtime_records,
            )
            if self._store_only_layer_inputs:
                runtime_records.pop_local_tensor(f"layer_{layer_id}.post_attention_hidden")
            del mlp_result

            grad_post_attention = grad_post_attention_direct.to(self.device) + grad_post_attention_from_mlp
            del grad_post_attention_direct, grad_post_attention_from_mlp

            # Attention residual: y = layer_input + dropout(attention_projection(concat(chunks)))
            grad_layer_input_direct = grad_post_attention.detach().clone()
            grad_attention_dropout_out = grad_post_attention.detach().clone()
            del grad_post_attention

            grad_attention_projected = self._backward_dropout(
                input_key=f"layer_{layer_id}.attention_projected_pre_dropout",
                rng_key=f"layer_{layer_id}.attention_dropout_rng_state",
                grad_output=grad_attention_dropout_out,
                runtime_records=runtime_records,
            )
            if self._store_only_layer_inputs:
                runtime_records.pop_local_tensor(f"layer_{layer_id}.attention_projected_pre_dropout")
            del grad_attention_dropout_out

            grad_attention_concat = self._backward_attention_output_projection(
                layer_id=layer_id,
                runtime_records=runtime_records,
                grad_output=grad_attention_projected,
            )
            if self._store_only_layer_inputs:
                runtime_records.pop_local_tensor(f"layer_{layer_id}.attention_concat")
            del grad_attention_projected

            attention_result = self.backward_engine.backward_attention_layer_from_concat_gradient(
                layer_id=layer_id,
                runtime_records=runtime_records,
                grad_attention_concat=grad_attention_concat.detach().cpu(),
                overwrite_gradient_records=True,
            )
            if self._store_only_layer_inputs:
                runtime_records.pop_local_tensor(f"layer_{layer_id}.attention_input")
                runtime_records.pop_local_tensor(f"layer_{layer_id}.attention_mask")
            del grad_attention_concat
            segment_update_count += self._handle_segment_backward_result(attention_result)

            grad_layer_input_from_attention = self._backward_layer_norm(
                module=self.forward_engine.layer_norms.attention_layer_norms[layer_id],
                input_key=f"layer_{layer_id}.input",
                grad_output=attention_result.input_gradient,
                runtime_records=runtime_records,
            )
            del attention_result

            grad_hidden = grad_layer_input_direct.to(self.device) + grad_layer_input_from_attention
            del grad_layer_input_direct, grad_layer_input_from_attention

            # Design invariant: after each layer's backward, persist grad_hidden to disk.
            # Training can stop at any layer boundary and be resumed by loading this file.
            # CPU training: grad_hidden_dir is the per-step record directory.
            if grad_hidden_dir is not None:
                _gh_path = _Path(grad_hidden_dir) / "grad_hidden.pt"
                torch.save(grad_hidden.detach().cpu(), _gh_path)

            # Free this layer's shared tensors and gradient records once consumed.
            runtime_records.clear_gradient_records()
            if self._store_only_layer_inputs:
                runtime_records.clear_local_tensors()
            if self._evict_layers_during_backward:
                runtime_records.evict_layer(layer_id)
            gc.collect()
            if sys.platform.startswith("linux"):
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

        self._backward_embeddings(runtime_records=runtime_records, grad_output=grad_hidden)
        return segment_update_count, bwd_loss_scalar

    def _recompute_layer_intermediates(
        self,
        layer_id: int,
        runtime_records: RuntimeRecordStore,
    ) -> None:
        """Recompute all intermediate tensors for layer_id from the stored layer_X.input.

        Adds results to runtime_records._local_tensors so existing backward operations
        can load them via get_shared_tensor() without any disk I/O.  Call
        clear_local_tensors() (or pop_local_tensor per key) when each tensor is no
        longer needed.
        """
        fe = self.forward_engine
        dev = self.device

        layer_input = runtime_records.get_shared_tensor(f"layer_{layer_id}.input")
        layer_input = layer_input.detach().clone().to(dev)

        with torch.no_grad():
            # Provide attention_mask if present (stored once in metadata by forward).
            attn_mask: Tensor | None = None
            lazy_mask = runtime_records.metadata.get("_lazy_attention_mask")
            if lazy_mask is not None:
                attn_mask = lazy_mask.to(dev)
                runtime_records.add_local_tensor(
                    f"layer_{layer_id}.attention_mask", attn_mask.detach()
                )

            # -- Attention branch --
            attention_input = fe.layer_norms.attention(layer_id, layer_input)
            runtime_records.add_local_tensor(
                f"layer_{layer_id}.attention_input", attention_input.detach()
            )

            layer_attn_outputs: list[Tensor] = []
            for seg_idx in range(fe.segmentation_config.attention_segments):
                seg_id = SegmentId(layer_id, "attention", seg_idx)
                record = runtime_records.get_attention_record(seg_id)
                with self.segment_loader.acquire_segment(seg_id) as segment:
                    if record.rng_state is not None:
                        with restored_rng_state(
                            record.rng_state,
                            restore_cuda=record.rng_state.has_cuda_state,
                        ):
                            seg_out = segment(attention_input, attn_mask)
                    else:
                        seg_out = segment(attention_input, attn_mask)
                layer_attn_outputs.append(seg_out.detach())

            attention_concat = torch.cat(layer_attn_outputs, dim=-1)
            del layer_attn_outputs
            runtime_records.add_local_tensor(
                f"layer_{layer_id}.attention_concat", attention_concat.detach()
            )

            attention_projected = fe._run_attention_output_proj(layer_id, attention_concat)
            del attention_concat
            runtime_records.add_local_tensor(
                f"layer_{layer_id}.attention_projected_pre_dropout",
                attention_projected.detach(),
            )

            attn_rng = runtime_records.metadata.get(
                f"layer_{layer_id}.attention_dropout_rng_state"
            )
            if attn_rng is not None and fe.residual_dropout.p > 0.0:
                with restored_rng_state(attn_rng, restore_cuda=attn_rng.has_cuda_state):
                    attn_dropped = fe.residual_dropout(attention_projected)
            else:
                attn_dropped = attention_projected
            del attention_projected

            post_attention = layer_input + attn_dropped
            del attn_dropped
            runtime_records.add_local_tensor(
                f"layer_{layer_id}.post_attention_hidden", post_attention.detach()
            )

            # -- MLP branch --
            mlp_input = fe.layer_norms.mlp(layer_id, post_attention)
            del post_attention
            runtime_records.add_local_tensor(
                f"layer_{layer_id}.mlp_input", mlp_input.detach()
            )

            mlp_running_sum: Tensor | None = None
            for seg_idx in range(fe.segmentation_config.mlp_chunks):
                seg_id = SegmentId(layer_id, "mlp", seg_idx)
                record = runtime_records.get_mlp_record(seg_id)
                with self.segment_loader.acquire_segment(seg_id) as segment:
                    if record.rng_state is not None:
                        with restored_rng_state(
                            record.rng_state,
                            restore_cuda=record.rng_state.has_cuda_state,
                        ):
                            seg_out = segment(mlp_input)
                    else:
                        seg_out = segment(mlp_input)
                if mlp_running_sum is None:
                    mlp_running_sum = seg_out.detach()
                else:
                    new_sum = mlp_running_sum + seg_out.detach()
                    del mlp_running_sum
                    mlp_running_sum = new_sum
            del mlp_input

            assert mlp_running_sum is not None
            runtime_records.add_local_tensor(
                f"layer_{layer_id}.mlp_sum_pre_bias", mlp_running_sum.detach()
            )

            mlp_after_bias = fe.mlp_shared_output_biases(layer_id, mlp_running_sum)
            del mlp_running_sum
            runtime_records.add_local_tensor(
                f"layer_{layer_id}.mlp_after_bias_pre_dropout", mlp_after_bias.detach()
            )

        gc.collect()

    def _backward_output_head_from_loss(
        self,
        runtime_records: RuntimeRecordStore,
        labels: Tensor,
    ) -> tuple[Tensor, float]:
        hidden = self._floating_shared_tensor(runtime_records, "pre_final_norm").requires_grad_(True)
        if self.forward_engine._use_segment_store_for_global:
            n_head = len(self.forward_engine._output_head_seg_ids)
            if n_head == 1:
                # Original single-segment path.
                seg_id = self.forward_engine._output_head_seg_id
                with self.segment_loader.acquire_segment(seg_id) as head_mod:
                    logits = head_mod(hidden)
                    loss = self.loss_fn(logits, labels.to(logits.device))
                    if loss.ndim != 0:
                        raise ValueError("Training loss must be scalar.")
                    scaled_loss = loss / float(self.gradient_accumulation_steps)
                    params = list(head_mod.named_parameters())
                    trainable_params = [p for _n, p in params if p.requires_grad]
                    grads = torch.autograd.grad(
                        outputs=scaled_loss,
                        inputs=[hidden, *trainable_params],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    input_grad = grads[0]
                    if input_grad is None:
                        input_grad = torch.zeros_like(hidden)
                    param_grads: dict[str, Tensor] = {}
                    p_iter = ((n, p) for n, p in params if p.requires_grad)
                    for (name, _p), grad in zip(p_iter, grads[1:]):
                        if grad is not None:
                            param_grads[name] = grad.detach().cpu()
                if self.update_style == "immediate_segment_update":
                    self._update_one_segment(seg_id, param_grads)
                else:
                    self._accumulate_segment_gradients(seg_id, param_grads)
                return input_grad.detach(), float(loss.detach().cpu().item())

            # N>1 vocab-sliced output head: incremental log-sum-exp.
            # Never materialises full [B, S, vocab] logits.
            final_norm = self.forward_engine.output_slice_final_norm
            ln_params = [p for p in final_norm.parameters() if p.requires_grad]

            # Causal shift: position t predicts token at t+1.
            hidden_shifted = hidden[:, :-1, :]         # [B, S-1, d_model]
            shift_labels = labels[:, 1:].contiguous()  # [B, S-1]

            # hidden_norm retains grad graph back to hidden (needed for LN backward).
            hidden_norm = final_norm(hidden_shifted)    # [B, S-1, d_model]

            seg_ids = self.forward_engine._output_head_seg_ids
            ignore_index: int = getattr(self.loss_fn, "ignore_index", -100)
            dev = hidden.device
            dtype = hidden.dtype
            B, Sm1 = shift_labels.shape

            active_mask = (shift_labels != ignore_index)  # [B, S-1] on CPU/labels device
            n_active = int(active_mask.sum().clamp(min=1).item())

            # ── Forward sweep: running LSE + correct_logit for logging loss ──
            # Seq-chunked: tok_chunk_size positions × one vocab segment at a time so
            # peak logit tensor is [B, tok_chunk_size, vocab/N], not [B, S-1, vocab/N].
            n_tok_chunks = self._output_head_token_chunks
            tok_chunk_size = max(1, (Sm1 + n_tok_chunks - 1) // n_tok_chunks)
            running_lse = torch.full((B, Sm1), float("-inf"), dtype=dtype)
            correct_logit = torch.full((B, Sm1), float("-inf"), dtype=dtype)
            shift_labels_cpu = shift_labels.cpu()
            active_mask_cpu = active_mask.cpu()

            for t0 in range(0, Sm1, tok_chunk_size):
                t1 = min(Sm1, t0 + tok_chunk_size)
                h_chunk = hidden_norm[:, t0:t1, :].detach()  # [B, chunk, D]
                sl_chunk = shift_labels_cpu[:, t0:t1]
                am_chunk = active_mask_cpu[:, t0:t1]
                vocab_start_fwd = 0

                for seg_id in seg_ids:
                    with self.segment_loader.acquire_segment(seg_id) as head_slice:
                        with torch.no_grad():
                            z_i = head_slice(h_chunk)  # [B, chunk, slice_size]

                    z_i_cpu = z_i.cpu()
                    del z_i
                    this_slice_fwd = z_i_cpu.shape[-1]
                    lse_i = torch.logsumexp(z_i_cpu, dim=-1)  # [B, chunk]
                    running_lse[:, t0:t1] = torch.logaddexp(running_lse[:, t0:t1], lse_i)

                    in_range = (
                        am_chunk
                        & (sl_chunk >= vocab_start_fwd)
                        & (sl_chunk < vocab_start_fwd + this_slice_fwd)
                    )
                    if in_range.any():
                        idx = (sl_chunk - vocab_start_fwd).clamp(0, this_slice_fwd - 1)
                        correct_logit[:, t0:t1][in_range] = z_i_cpu.gather(-1, idx.unsqueeze(-1)).squeeze(-1)[in_range]

                    vocab_start_fwd += this_slice_fwd
                    del z_i_cpu

            # Scalar CE loss for logging — no full [B,S,vocab] tensor.
            zero = running_lse.new_zeros(())
            per_token = torch.where(active_mask_cpu, running_lse - correct_logit, zero)
            bwd_loss_scalar = float(per_token.sum().item() / n_active)

            # ── Backward sweep: accumulate grad w.r.t. hidden_norm ──────────
            # Token-chunked path: process C chunks of the sequence per vocab segment to
            # keep the per-segment logit tensor [B, chunk_tokens, slice_size] small.
            # P3: grad accumulator lives on CPU; lse/mask/labels moved to dev only
            # for each tok_chunk, then freed. empty_cache() between segments.
            scale = 1.0 / (n_active * float(self.gradient_accumulation_steps))
            d_model = hidden.shape[-1]

            grad_h_norm_acc = torch.zeros(B, Sm1, d_model, device='cpu', dtype=dtype)
            vocab_start_bwd = 0

            for seg_id in seg_ids:
                this_slice: int | None = None
                seg_param_grads: dict[str, Tensor] = {}

                with self.segment_loader.acquire_segment(seg_id) as head_slice:
                    named_params = list(head_slice.named_parameters())
                    trainable_params = [p for _n, p in named_params if p.requires_grad]

                    for t_start in range(0, Sm1, tok_chunk_size):
                        t_end = min(Sm1, t_start + tok_chunk_size)

                        h_chunk = hidden_norm[:, t_start:t_end, :].detach().requires_grad_(True)
                        z_chunk = head_slice(h_chunk)  # [B, chunk_tokens, slice_size]
                        if this_slice is None:
                            this_slice = z_chunk.shape[-1]

                        # Load only this tok_chunk's lse/mask/labels to device.
                        lse_chunk = running_lse[:, t_start:t_end].to(dev)
                        mask_chunk = active_mask_cpu[:, t_start:t_end].to(dev)
                        labels_chunk = shift_labels_cpu[:, t_start:t_end].to(dev)

                        with torch.no_grad():
                            softmax_chunk = torch.exp(
                                z_chunk.detach() - lse_chunk.unsqueeze(-1)
                            )
                            grad_z_chunk = softmax_chunk * mask_chunk.float().unsqueeze(-1)

                            in_range_chunk = (
                                mask_chunk
                                & (labels_chunk >= vocab_start_bwd)
                                & (labels_chunk < vocab_start_bwd + this_slice)
                            )
                            if in_range_chunk.any():
                                idx_chunk = (labels_chunk - vocab_start_bwd).clamp(
                                    0, this_slice - 1
                                )
                                grad_z_chunk.scatter_add_(
                                    -1,
                                    idx_chunk.unsqueeze(-1),
                                    -in_range_chunk.float().unsqueeze(-1),
                                )

                            grad_z_chunk = grad_z_chunk * scale

                        all_grads = torch.autograd.grad(
                            outputs=z_chunk,
                            inputs=[h_chunk, *trainable_params],
                            grad_outputs=grad_z_chunk,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )

                        gh = all_grads[0]
                        if gh is not None:
                            # Accumulate on CPU — P3: grad accumulator stays off device.
                            grad_h_norm_acc[:, t_start:t_end, :].add_(gh.detach().cpu())

                        p_iter = ((n, p) for n, p in named_params if p.requires_grad)
                        for (name, _p), pg in zip(p_iter, all_grads[1:]):
                            if pg is not None:
                                pg_cpu = pg.detach().cpu()
                                if name in seg_param_grads:
                                    seg_param_grads[name] = seg_param_grads[name] + pg_cpu
                                else:
                                    seg_param_grads[name] = pg_cpu

                        # P3: explicitly free all device tensors created this tok_chunk
                        # so they are not kept alive by Python's loop-variable scoping
                        # until the next iteration. Without this, the last tok_chunk's
                        # tensors (z_chunk, grad_z_chunk, softmax_chunk, h_chunk, gh)
                        # stay in VRAM until the next seg_id iteration assigns over them,
                        # causing large spikes at the segment release event.
                        del all_grads, gh, z_chunk, h_chunk
                        del softmax_chunk, grad_z_chunk, in_range_chunk
                        del lse_chunk, mask_chunk, labels_chunk

                # P3: break autograd reference cycles OUTSIDE the `with` block.
                # gc.collect() must run after Python's `with` bytecode has cleaned
                # up the _GeneratorContextManager — that object is a GC root while
                # the `with` is active, preventing cycle collection. Calling inside
                # the `with` body or inside release_segment() does not work for
                # this reason. Here, immediately after the `with` exits, autograd
                # cycle tensors become unreachable and gc can free them.
                # gc.collect() is unconditional — reference cycles accumulate on
                # CPU just as on CUDA; empty_cache() is CUDA-only.
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

                assert this_slice is not None, "output head segment produced no output"
                param_grads = seg_param_grads
                if self.update_style == "immediate_segment_update":
                    self._update_one_segment(seg_id, param_grads)
                else:
                    self._accumulate_segment_gradients(seg_id, param_grads)
                vocab_start_bwd += this_slice
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            # ── Backprop through always-resident LayerNorm ───────────────────
            ln_grads = torch.autograd.grad(
                outputs=hidden_norm,
                inputs=[hidden, *ln_params],
                grad_outputs=grad_h_norm_acc.to(dev),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            input_grad = ln_grads[0]
            if input_grad is None:
                input_grad = torch.zeros_like(hidden)
            for param, pg in zip(ln_params, ln_grads[1:]):
                if pg is not None:
                    self._accumulate_parameter_grad(param, pg)
            return input_grad.detach(), bwd_loss_scalar
        else:
            logits = self.forward_engine.output_head(hidden)
            loss = self.loss_fn(logits, labels.to(logits.device))
            if loss.ndim != 0:
                raise ValueError("Training loss must be scalar.")
            scaled_loss = loss / float(self.gradient_accumulation_steps)
            params = [p for p in self.forward_engine.output_head.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                outputs=scaled_loss,
                inputs=[hidden, *params],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            input_grad = grads[0]
            if input_grad is None:
                input_grad = torch.zeros_like(hidden)
            for param, grad in zip(params, grads[1:], strict=True):
                self._accumulate_parameter_grad(param, grad)
            return input_grad.detach(), float(loss.detach().cpu().item())

    def _backward_embeddings(
        self,
        *,
        runtime_records: RuntimeRecordStore,
        grad_output: Tensor,
    ) -> None:
        input_ids = runtime_records.metadata.get("input_ids")
        if not isinstance(input_ids, Tensor):
            raise RuntimeError("Runtime records are missing input_ids for embedding recomputation.")
        position_ids = runtime_records.metadata.get("position_ids")
        if position_ids is not None and not isinstance(position_ids, Tensor):
            raise RuntimeError("position_ids metadata must be a Tensor when present.")
        input_ids = input_ids.to(self.device)
        position_ids = None if position_ids is None else position_ids.to(self.device)
        rng_state = runtime_records.metadata.get("embedding_dropout_rng_state")

        if self.forward_engine._use_segment_store_for_global:
            n_emb = len(self.forward_engine._embedding_seg_ids)
            if n_emb == 1:
                # Original single-segment path.
                seg_id = self.forward_engine._embedding_seg_id
                with self.segment_loader.acquire_segment(seg_id) as emb_mod:
                    named_params = list(emb_mod.named_parameters())
                    trainable_params = [p for _n, p in named_params if p.requires_grad]
                    if rng_state is None:
                        output = emb_mod(input_ids, position_ids)
                    else:
                        with restored_rng_state(rng_state, restore_cuda=getattr(rng_state, "has_cuda_state", False)):
                            output = emb_mod(input_ids, position_ids)
                    grads = torch.autograd.grad(
                        outputs=output,
                        inputs=trainable_params,
                        grad_outputs=grad_output.to(output.device),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    param_grads: dict[str, Tensor] = {}
                    p_iter = ((n, p) for n, p in named_params if p.requires_grad)
                    for (name, _p), grad in zip(p_iter, grads):
                        if grad is not None:
                            param_grads[name] = grad.detach().cpu()
                if self.update_style == "immediate_segment_update":
                    self._update_one_segment(seg_id, param_grads)
                else:
                    self._accumulate_segment_gradients(seg_id, param_grads)
                return

            # N>1 sliced embedding path.
            # Step 1: backprop grad_output through the embedding dropout to get
            #         grad_concat using a placeholder (dropout mask depends only
            #         on RNG state, not on input values).
            B, S = input_ids.shape
            d_model = grad_output.shape[-1]
            dev = grad_output.device
            placeholder = torch.ones(B, S, d_model, device=dev, requires_grad=True)
            restore_cuda = getattr(rng_state, "has_cuda_state", False)
            if rng_state is not None and self.forward_engine.embedding_slice_dropout.p > 0.0:
                with restored_rng_state(rng_state, restore_cuda=restore_cuda):
                    dropped = self.forward_engine.embedding_slice_dropout(placeholder)
                (grad_concat,) = torch.autograd.grad(
                    outputs=dropped,
                    inputs=[placeholder],
                    grad_outputs=grad_output.to(dropped.device),
                    retain_graph=False,
                    create_graph=False,
                )
                grad_concat = grad_concat.detach()
            else:
                grad_concat = grad_output.to(dev).detach()

            # Step 2: per-slice gradient computation (no RNG needed — slices are
            #         deterministic, dropout was handled above).
            seg_ids = self.forward_engine._embedding_seg_ids
            d_slice = d_model // n_emb
            for i, seg_id in enumerate(seg_ids):
                grad_slice = grad_concat[:, :, i * d_slice : (i + 1) * d_slice]
                with self.segment_loader.acquire_segment(seg_id) as emb_slice:
                    named_params = list(emb_slice.named_parameters())
                    trainable_params = [p for _n, p in named_params if p.requires_grad]
                    output_i = emb_slice(input_ids, position_ids)
                    grads = torch.autograd.grad(
                        outputs=output_i,
                        inputs=trainable_params,
                        grad_outputs=grad_slice.to(output_i.device),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    param_grads = {}
                    for (name, _p), grad in zip(
                        ((n, p) for n, p in named_params if p.requires_grad), grads
                    ):
                        if grad is not None:
                            param_grads[name] = grad.detach().cpu()
                if self.update_style == "immediate_segment_update":
                    self._update_one_segment(seg_id, param_grads)
                else:
                    self._accumulate_segment_gradients(seg_id, param_grads)
            return
        else:
            params = [p for p in self.forward_engine.embeddings.parameters() if p.requires_grad]
            if rng_state is None:
                output = self.forward_engine.embeddings(input_ids, position_ids=position_ids)
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=params,
                    grad_outputs=grad_output.to(output.device),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            else:
                with restored_rng_state(rng_state, restore_cuda=getattr(rng_state, "has_cuda_state", False)):
                    output = self.forward_engine.embeddings(input_ids, position_ids=position_ids)
                    grads = torch.autograd.grad(
                        outputs=output,
                        inputs=params,
                        grad_outputs=grad_output.to(output.device),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
            for param, grad in zip(params, grads, strict=True):
                self._accumulate_parameter_grad(param, grad)

    def _backward_layer_norm(
        self,
        *,
        module: nn.LayerNorm,
        input_key: str,
        grad_output: Tensor,
        runtime_records: RuntimeRecordStore,
    ) -> Tensor:
        return self._backward_local_module(
            input_tensor=self._floating_shared_tensor(runtime_records, input_key),
            grad_output=grad_output,
            forward_fn=module,
            parameters=[p for p in module.parameters() if p.requires_grad],
        )

    def _backward_attention_output_projection(
        self,
        *,
        layer_id: int,
        runtime_records: RuntimeRecordStore,
        grad_output: Tensor,
    ) -> Tensor:
        if self.forward_engine._use_segment_store_for_global:
            # Load the attention_output_proj segment, recompute, collect
            # gradients, and return the gradient w.r.t. attention_concat.
            # Parameter updates are routed through the usual accumulate/update path.
            seg_id = self.forward_engine._attn_proj_seg_ids[layer_id]  # type: ignore[index]
            input_tensor = self._floating_shared_tensor(
                runtime_records, f"layer_{layer_id}.attention_concat"
            )
            with self.segment_loader.acquire_segment(seg_id) as proj_mod:
                local_input = input_tensor.detach().clone().to(self.device).requires_grad_(True)
                output = proj_mod(local_input)
                grad = grad_output.detach().to(output.device)
                named_params = list(proj_mod.named_parameters())
                trainable_params = [p for _n, p in named_params if p.requires_grad]
                targets: list[Tensor] = [local_input, *trainable_params]
                gradients = torch.autograd.grad(
                    outputs=output,
                    inputs=targets,
                    grad_outputs=grad,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                input_grad = gradients[0]
                if input_grad is None:
                    input_grad = torch.zeros_like(local_input)
                param_grads: dict[str, Tensor] = {}
                p_iter = ((n, p) for n, p in named_params if p.requires_grad)
                for (name, _p), pg in zip(p_iter, gradients[1:]):
                    if pg is not None:
                        param_grads[name] = pg.detach().cpu()
            # Route via accumulate/update path.
            if self.update_style == "immediate_segment_update":
                self._update_one_segment(seg_id, param_grads)
            else:
                self._accumulate_segment_gradients(seg_id, param_grads)
            return input_grad.detach()
        else:
            projection = self.forward_engine.attention_output_projections.projections[layer_id]  # type: ignore[union-attr]
            return self._backward_local_module(
                input_tensor=self._floating_shared_tensor(runtime_records, f"layer_{layer_id}.attention_concat"),
                grad_output=grad_output,
                forward_fn=projection,
                parameters=[p for p in projection.parameters() if p.requires_grad],
            )

    def _backward_mlp_shared_bias(
        self,
        *,
        layer_id: int,
        runtime_records: RuntimeRecordStore,
        grad_output: Tensor,
    ) -> Tensor:
        input_tensor = self._floating_shared_tensor(runtime_records, f"layer_{layer_id}.mlp_sum_pre_bias")
        if not self.forward_engine.mlp_shared_output_biases.enabled:
            return grad_output.detach().to(self.device)
        bias = self.forward_engine.mlp_shared_output_biases.biases[layer_id]
        return self._backward_local_module(
            input_tensor=input_tensor,
            grad_output=grad_output,
            forward_fn=lambda x: x + bias.view(1, 1, -1),
            parameters=[bias],
        )

    def _backward_dropout(
        self,
        *,
        input_key: str,
        rng_key: str,
        grad_output: Tensor,
        runtime_records: RuntimeRecordStore,
    ) -> Tensor:
        input_tensor = self._floating_shared_tensor(runtime_records, input_key).requires_grad_(True)
        grad = grad_output.detach().to(input_tensor.device)
        rng_state = runtime_records.metadata.get(rng_key)
        if rng_state is None or self.forward_engine.residual_dropout.p == 0.0:
            return grad.detach()
        with restored_rng_state(rng_state, restore_cuda=getattr(rng_state, "has_cuda_state", False)):
            output = self.forward_engine.residual_dropout(input_tensor)
            (input_grad,) = torch.autograd.grad(
                outputs=output,
                inputs=[input_tensor],
                grad_outputs=grad,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return input_grad.detach()

    def _backward_local_module(
        self,
        *,
        input_tensor: Tensor,
        grad_output: Tensor,
        forward_fn: Callable[[Tensor], Tensor],
        parameters: Sequence[nn.Parameter],
    ) -> Tensor:
        local_input = input_tensor.detach().clone().to(self.device).requires_grad_(True)
        output = forward_fn(local_input)
        grad = grad_output.detach().to(output.device)
        targets: list[Tensor] = [local_input]
        targets.extend(parameters)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=targets,
            grad_outputs=grad,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        input_grad = gradients[0]
        if input_grad is None:
            input_grad = torch.zeros_like(local_input)
        for parameter, parameter_grad in zip(parameters, gradients[1:], strict=True):
            self._accumulate_parameter_grad(parameter, parameter_grad)
        return input_grad.detach()

    def _handle_segment_backward_result(self, result: LayerBackwardResult) -> int:
        update_count = 0
        for segment_result in result.segment_results:
            if self.update_style == "immediate_segment_update":
                self._update_one_segment(segment_result.segment_id, segment_result.parameter_gradients)
                update_count += 1
            else:
                self._accumulate_segment_gradients(
                    segment_result.segment_id,
                    segment_result.parameter_gradients,
                )
        if self.update_style == "after_full_backward":
            return len(result.segment_results)
        return update_count

    # ------------------------------------------------------------------
    # Gradient storage/update helpers
    # ------------------------------------------------------------------

    def _accumulate_parameter_grad(self, parameter: nn.Parameter, gradient: Tensor | None) -> None:
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        grad = gradient.detach().to(device=parameter.device, dtype=parameter.dtype)
        if parameter.grad is None:
            parameter.grad = grad.clone()
        else:
            parameter.grad = parameter.grad + grad

    def _accumulate_segment_gradients(
        self,
        segment_id: SegmentId,
        gradients: dict[str, Tensor],
    ) -> None:
        if self._disk_grad_store is not None:
            self._disk_grad_store.accumulate(segment_id, gradients)
        else:
            current = self._accumulated_segment_gradients.setdefault(segment_id, {})
            for name, gradient in gradients.items():
                if name in current:
                    current[name] = current[name] + gradient.detach().cpu()
                else:
                    current[name] = gradient.detach().cpu().clone()


    def _clip_all_accumulated_gradients(self, max_norm: float) -> float:
        """Clip non-segment and offloaded segment gradients by one global norm.

        This is only mathematically exact for ``after_full_backward`` because all
        segment gradients are available in ``_accumulated_segment_gradients``
        before any segment update is applied.
        """

        if self._disk_grad_store is None:
            # In-memory path (unchanged).
            gradients: list[Tensor] = []
            for parameter in self.forward_engine.parameters():
                if parameter.grad is not None:
                    gradients.append(parameter.grad.detach())
            for segment_gradients in self._accumulated_segment_gradients.values():
                for gradient in segment_gradients.values():
                    gradients.append(gradient.detach())

            if not gradients:
                return 0.0

            total_sq = torch.zeros((), dtype=torch.float64)
            for gradient in gradients:
                total_sq += gradient.detach().double().pow(2).sum().cpu()
            total_norm = float(total_sq.sqrt().item())

            if total_norm > max_norm:
                scale = max_norm / (total_norm + 1.0e-12)
                for parameter in self.forward_engine.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                for segment_gradients in self._accumulated_segment_gradients.values():
                    for name in list(segment_gradients):
                        segment_gradients[name] = segment_gradients[name] * scale

            return total_norm
        else:
            # Disk path: streaming two-pass (no full gradient list in memory).
            total_sq = torch.zeros((), dtype=torch.float64)
            for parameter in self.forward_engine.parameters():
                if parameter.grad is not None:
                    total_sq += parameter.grad.detach().double().pow(2).sum().cpu()
            for seg_id in self._disk_grad_store:
                g = self._disk_grad_store.load(seg_id)
                for grad in g.values():
                    total_sq += grad.detach().double().pow(2).sum().cpu()
                del g
            total_norm = float(total_sq.sqrt().item())

            if total_norm <= max_norm:
                return total_norm

            scale = max_norm / (total_norm + 1.0e-12)
            for parameter in self.forward_engine.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            for seg_id in self._disk_grad_store:
                g = self._disk_grad_store.load(seg_id)
                scaled = {name: grad * scale for name, grad in g.items()}
                del g
                self._disk_grad_store.save_scaled(seg_id, scaled)
                del scaled

            return total_norm

    def _apply_accumulated_segment_updates(self) -> None:
        if self._disk_grad_store is not None:
            for segment_id in self._disk_grad_store:
                grads = self._disk_grad_store.load(segment_id)
                self._update_one_segment(segment_id, grads)
                self._disk_grad_store.evict(segment_id)
            self._disk_grad_store.clear()
        else:
            # Gradients are already scaled by 1 / gradient_accumulation_steps at the
            # loss root. Do not divide again here.
            for segment_id in sorted(self._accumulated_segment_gradients):
                self._update_one_segment(
                    segment_id, self._accumulated_segment_gradients[segment_id]
                )

    def _update_one_segment(
        self,
        segment_id: SegmentId,
        gradients: dict[str, Tensor],
    ) -> None:
        with self.segment_loader.acquire_segment(segment_id, save_on_exit=True) as module:
            self.segment_optimizer.step_module(
                segment_id=segment_id,
                module=module,
                gradients=gradients,
                strict=True,
            )

    # ------------------------------------------------------------------
    # Runtime-record checks and tensor helpers
    # ------------------------------------------------------------------

    def _floating_shared_tensor(self, runtime_records: RuntimeRecordStore, key: str) -> Tensor:
        tensor = runtime_records.get_shared_tensor(key)
        if not torch.is_floating_point(tensor):
            raise TypeError(f"Shared tensor {key!r} must be floating point for recomputation.")
        return tensor.detach().clone().to(self.device)

    @staticmethod
    def _assert_records_detached(runtime_records: RuntimeRecordStore) -> None:
        # shared_tensors is empty when using DiskRecordBackend (tensors are on disk
        # and were already detached before being written); skip the dict check.
        if runtime_records._backend is None:
            for key, tensor in runtime_records.shared_tensors.items():
                if tensor.grad_fn is not None:
                    raise RuntimeError(f"Shared tensor {key!r} still has a grad_fn.")
        for record in runtime_records.attention_records.values():
            if record.output_tensor.grad_fn is not None:
                raise RuntimeError(
                    f"Attention record {record.segment_id.to_key()} still has a grad_fn."
                )
        for record in runtime_records.mlp_records.values():
            if record.output_tensor.grad_fn is not None:
                raise RuntimeError(
                    f"MLP record {record.segment_id.to_key()} still has a grad_fn."
                )


# ---------------------------------------------------------------------------
# CLI-discoverable aliases (required by cli/train.py dispatch)
# ---------------------------------------------------------------------------

Trainer = SegmentedTrainer


def _deep_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _deep_set(mapping: dict[str, Any], path: str, value: Any) -> None:
    current = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _resolve_config_path(path_value: Any, *, base_dir: Any) -> Any:
    from pathlib import Path

    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def train_from_config(
    config_path=None,
    config: dict | None = None,
    checkpoint=None,
    run_dir=None,
    overrides: Mapping[str, Any] | None = None,
    **kwargs,
) -> Any:
    """Run true-recomputation training from a YAML protocol/config file.

    This is the implementation used by ``segllm train --config ...``. It maps
    the protocol YAML sections into the same assembly/training path used by
    ``scripts/train_segmented.py`` so the config-driven CLI uses the fixed
    detached-forward/local-recompute trainer rather than a stub.
    """

    from pathlib import Path
    import yaml

    if config is None:
        if config_path is None:
            raise ValueError("config_path or config must be provided.")
        with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Training config YAML must contain a mapping at the root.")
        config_data: dict[str, Any] = loaded
    else:
        config_data = dict(config)

    for key, value in dict(overrides or {}).items():
        try:
            parsed_value = yaml.safe_load(str(value))
        except Exception:
            parsed_value = value
        _deep_set(config_data, key, parsed_value)

    # Import lazily because the script depends on transformers, which is only
    # required for real dataset/tokenizer training runs.
    try:
        from scripts import train_segmented as train_script
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        raise RuntimeError(
            "Could not import scripts.train_segmented. Run from the project root "
            "or ensure the repository root is on PYTHONPATH."
        ) from exc

    args = train_script.parse_args([])
    base_dir = Path(config_path).expanduser().resolve().parent if config_path is not None else Path.cwd()
    if run_dir is not None:
        base_dir = Path(run_dir).expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

    model = dict(config_data.get("model", {}))
    segmentation = dict(config_data.get("segmentation", {}))
    storage = dict(config_data.get("segment_storage", {}))
    training = dict(config_data.get("training", {}))
    optimizer = dict(config_data.get("optimizer", {}))
    runtime = dict(config_data.get("runtime_records", {}))
    data = dict(config_data.get("data", {}))
    checkpointing = dict(config_data.get("checkpointing", {}))
    exports = dict(config_data.get("exports", config_data.get("export", {})))
    reproducibility = dict(config_data.get("reproducibility", {}))

    args.device = storage.get("device", config_data.get("device", getattr(args, "device", "cpu")))
    backend = storage.get("backend", "disk_streaming")
    args.storage = "cpu_ram" if backend == "cpu_ram_offload" else "disk"
    args.segment_dir = _resolve_config_path(storage.get("segment_dir", getattr(args, "segment_dir", "segment_cache")), base_dir=base_dir)

    args.d_model = int(model.get("d_model", args.d_model))
    args.n_heads = int(model.get("n_heads", args.n_heads))
    args.n_layers = int(model.get("n_layers", args.n_layers))
    args.d_ff = int(model.get("d_ff", args.d_ff))
    args.dropout = float(model.get("dropout", args.dropout))
    args.max_seq_len = int(model.get("max_seq_len", args.max_seq_len))

    args.attention_segments = int(segmentation.get("attention_segments", args.attention_segments))
    args.mlp_chunks = int(segmentation.get("mlp_chunks", args.mlp_chunks))

    args.epochs = int(training.get("epochs", args.epochs))
    args.batch_size = int(training.get("batch_size", args.batch_size))
    args.gradient_accumulation = int(training.get("gradient_accumulation_steps", args.gradient_accumulation))
    args.update_style = str(training.get("update_style", args.update_style))
    args.gradient_clip_norm = training.get("gradient_clip_norm", args.gradient_clip_norm)
    if args.gradient_clip_norm is not None:
        args.gradient_clip_norm = float(args.gradient_clip_norm)

    args.optimizer = str(optimizer.get("type", args.optimizer))
    args.lr = float(optimizer.get("learning_rate", args.lr))
    args.weight_decay = float(optimizer.get("weight_decay", args.weight_decay))
    args.momentum = float(optimizer.get("momentum", args.momentum))

    args.recomputation_check = bool(runtime.get("recomputation_check", args.recomputation_check))
    args.seed = int(reproducibility.get("seed", args.seed))

    args.tokenizer_path = _resolve_config_path(data.get("tokenizer_path", args.tokenizer_path), base_dir=base_dir)
    args.train_split = _resolve_config_path(data.get("train_split", args.train_split), base_dir=base_dir)
    args.validation_split = _resolve_config_path(data.get("validation_split", args.validation_split), base_dir=base_dir)
    args.test_split = _resolve_config_path(data.get("test_split", args.test_split), base_dir=base_dir)
    args.input_column = str(data.get("input_column", args.input_column))
    args.label_column = str(data.get("label_column", args.label_column))

    args.checkpoint_dir = _resolve_config_path(checkpointing.get("checkpoint_dir", args.checkpoint_dir), base_dir=base_dir)
    yaml_protocol_path = exports.get("yaml_protocol_path")
    full_model_path = exports.get("full_model_export_path")
    if yaml_protocol_path:
        args.export_dir = _resolve_config_path(yaml_protocol_path, base_dir=base_dir).parent
    elif full_model_path:
        args.export_dir = _resolve_config_path(full_model_path, base_dir=base_dir).parent
    else:
        args.export_dir = _resolve_config_path(getattr(args, "export_dir", "exports"), base_dir=base_dir)
    args.export_full_model = bool(exports.get("export_full_model", args.export_full_model))
    args.export_yaml = bool(exports.get("export_yaml_protocol", args.export_yaml))

    if checkpoint is not None:
        # The script accepts checkpoint names such as "last" or "best". If the
        # CLI supplied a path, use its final path component as the checkpoint name.
        args.resume = Path(checkpoint).name
    else:
        args.resume = config_data.get("resume", getattr(args, "resume", None))

    args.max_train_steps = (
        kwargs["max_train_steps"] if "max_train_steps" in kwargs
        else config_data.get("max_train_steps", args.max_train_steps)
    )
    args.max_val_steps = (
        kwargs["max_val_steps"] if "max_val_steps" in kwargs
        else config_data.get("max_val_steps", args.max_val_steps)
    )

    args.record_backend = runtime.get("record_backend", "in_memory")
    args.record_disk_dir = runtime.get("record_disk_dir", None)
    args.evict_layers_during_backward = bool(runtime.get("evict_layers_during_backward", False))
    args.gradient_store = training.get("gradient_store", "in_memory")
    args.gradient_store_dir = training.get("gradient_store_dir", None)

    return train_script.main(args)

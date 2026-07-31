"""Recomputation-based segmented backward engine.

Phase 12 scope:
- Recompute attention and MLP segments from segment-level execution records.
- Compute segment parameter gradients and segment input gradients.
- Store segment-level gradient records in RuntimeRecordStore.
- Validate recomputed outputs against stored forward outputs when configured.
- Preserve strict single-segment loading: every segment is loaded, used, and
  released before the next one is loaded.

This phase intentionally does not implement optimizer updates, checkpointing,
trainer integration, or full end-to-end loss backward. Those are later phases.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import (
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.execution.rng import restored_rng_state
from sequential_segmented_llm_training_inference.execution.segment_loader import (
    StrictSegmentLoader,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.runtime_records import (
    AttentionSegmentExecutionRecord,
    AttentionSegmentGradientRecord,
    MLPSegmentExecutionRecord,
    MLPSegmentGradientRecord,
    RuntimeRecordStore,
)


class RecomputationMismatchError(RuntimeError):
    """Raised when a recomputed segment output does not match the stored output."""


@dataclass(frozen=True, slots=True)
class RecomputationCheckConfig:
    """Controls recomputation-output verification."""

    enabled: bool = True
    atol: float = 1.0e-5
    rtol: float = 1.0e-4
    raise_on_mismatch: bool = True

    def __post_init__(self) -> None:
        if self.atol < 0:
            raise ValueError(f"atol must be non-negative, got {self.atol}.")
        if self.rtol < 0:
            raise ValueError(f"rtol must be non-negative, got {self.rtol}.")


@dataclass(slots=True)
class SegmentBackwardResult:
    """Backward result for one recomputed segment."""

    segment_id: SegmentId
    input_gradient: Tensor
    parameter_gradients: dict[str, Tensor]
    recomputation_matched: bool
    max_abs_diff: float


@dataclass(slots=True)
class LayerBackwardResult:
    """Backward result for all segments of one layer branch."""

    layer_id: int
    branch_type: str
    input_gradient: Tensor
    segment_results: list[SegmentBackwardResult] = field(default_factory=list)


class SegmentedBackwardEngine:
    """Compute segment gradients by recomputing segments from runtime records.

    The engine operates at segment-branch level in Phase 12. Given stored forward
    records and upstream gradients for composed segment outputs, it reloads each
    segment, recomputes its local forward pass, computes parameter/input
    gradients, writes gradient records, and releases the segment.
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        segmentation_config: SegmentationConfig,
        segment_loader: StrictSegmentLoader,
        recomputation_check: RecomputationCheckConfig | None = None,
    ) -> None:
        model_config.validate()
        segmentation_config.validate_against_model(model_config)
        if segment_loader is None:
            raise ValueError("segment_loader must be provided.")

        self.model_config = model_config
        self.segmentation_config = segmentation_config
        self.segment_loader = segment_loader
        self.recomputation_check = recomputation_check or RecomputationCheckConfig()

    @property
    def device(self) -> torch.device:
        return self.segment_loader.device

    def backward_mlp_layer(
        self,
        *,
        layer_id: int,
        runtime_records: RuntimeRecordStore,
        grad_mlp_output: Tensor,
        overwrite_gradient_records: bool = True,
    ) -> LayerBackwardResult:
        """Backward through all MLP segments for one layer.

        Forward MLP composition is summation, so every MLP segment receives the
        same gradient of the composed MLP output.
        """

        self._validate_layer_id(layer_id)
        if grad_mlp_output.ndim != 3:
            raise ValueError(
                "grad_mlp_output must have shape [batch, seq_len, d_model], "
                f"got {tuple(grad_mlp_output.shape)}."
            )
        if grad_mlp_output.shape[-1] != self.model_config.d_model:
            raise ValueError(
                f"grad_mlp_output last dimension must be d_model={self.model_config.d_model}, "
                f"got {grad_mlp_output.shape[-1]}."
            )

        segment_results: list[SegmentBackwardResult] = []
        input_gradients: list[Tensor] = []

        for segment_index in range(self.segmentation_config.mlp_chunks):
            segment_id = SegmentId(layer_id, "mlp", segment_index)
            record = runtime_records.get_mlp_record(segment_id)  # lazy: one at a time
            result = self.backward_mlp_segment(
                record=record,
                runtime_records=runtime_records,
                grad_segment_output=grad_mlp_output,
                overwrite_gradient_records=overwrite_gradient_records,
            )
            del record  # release before loading the next segment record
            segment_results.append(result)
            input_gradients.append(result.input_gradient)

        input_gradient = self._sum_input_gradients(input_gradients)
        return LayerBackwardResult(
            layer_id=layer_id,
            branch_type="mlp",
            input_gradient=input_gradient,
            segment_results=segment_results,
        )

    def backward_attention_layer_from_concat_gradient(
        self,
        *,
        layer_id: int,
        runtime_records: RuntimeRecordStore,
        grad_attention_concat: Tensor,
        overwrite_gradient_records: bool = True,
    ) -> LayerBackwardResult:
        """Backward through all attention segments from concat-output gradient.

        The gradient supplied here is the gradient with respect to the
        concatenated attention segment output, after any later projection/dropout
        operations have already been differentiated by the caller.
        """

        self._validate_layer_id(layer_id)
        if grad_attention_concat.ndim != 3:
            raise ValueError(
                "grad_attention_concat must have shape [batch, seq_len, d_model], "
                f"got {tuple(grad_attention_concat.shape)}."
            )
        if grad_attention_concat.shape[-1] != self.model_config.d_model:
            raise ValueError(
                f"grad_attention_concat last dimension must be d_model={self.model_config.d_model}, "
                f"got {grad_attention_concat.shape[-1]}."
            )

        segment_results: list[SegmentBackwardResult] = []
        input_gradients: list[Tensor] = []
        grad_start = 0  # cumulative offset into grad_attention_concat

        for segment_index in range(self.segmentation_config.attention_segments):
            segment_id = SegmentId(layer_id, "attention", segment_index)
            # Lazy: load one record at a time, process, then release.
            record = runtime_records.get_attention_record(segment_id)
            # record.shape is set in forward (real shape even under minimal
            # records), whereas output_tensor may be an empty placeholder.
            if record.shape is not None:
                seg_dim = int(record.shape[-1])
            else:
                seg_dim = int(record.output_tensor.shape[-1])

            grad_slice = grad_attention_concat[..., grad_start : grad_start + seg_dim]
            grad_start += seg_dim

            result = self.backward_attention_segment(
                record=record,
                runtime_records=runtime_records,
                grad_segment_output=grad_slice,
                overwrite_gradient_records=overwrite_gradient_records,
            )
            del record  # release before loading the next segment record
            segment_results.append(result)
            input_gradients.append(result.input_gradient)

        if grad_start != grad_attention_concat.shape[-1]:
            raise ValueError(
                "Attention segment output dimensions do not sum to the concat gradient "
                f"dimension: consumed={grad_start}, "
                f"grad_dim={grad_attention_concat.shape[-1]}."
            )

        input_gradient = self._sum_input_gradients(input_gradients)
        return LayerBackwardResult(
            layer_id=layer_id,
            branch_type="attention",
            input_gradient=input_gradient,
            segment_results=segment_results,
        )

    def backward_attention_layer_from_segment_gradients(
        self,
        *,
        layer_id: int,
        runtime_records: RuntimeRecordStore,
        grad_segment_outputs: Mapping[SegmentId, Tensor],
        overwrite_gradient_records: bool = True,
    ) -> LayerBackwardResult:
        """Backward through attention segments using explicit per-segment gradients."""

        self._validate_layer_id(layer_id)
        segment_results: list[SegmentBackwardResult] = []
        input_gradients: list[Tensor] = []

        for segment_index in range(self.segmentation_config.attention_segments):
            segment_id = SegmentId(layer_id, "attention", segment_index)
            if segment_id not in grad_segment_outputs:
                raise KeyError(f"Missing gradient for attention segment {segment_id.to_key()}.")
            record = runtime_records.get_attention_record(segment_id)  # lazy: one at a time
            result = self.backward_attention_segment(
                record=record,
                runtime_records=runtime_records,
                grad_segment_output=grad_segment_outputs[segment_id],
                overwrite_gradient_records=overwrite_gradient_records,
            )
            del record  # release before loading the next segment record
            segment_results.append(result)
            input_gradients.append(result.input_gradient)

        input_gradient = self._sum_input_gradients(input_gradients)
        return LayerBackwardResult(
            layer_id=layer_id,
            branch_type="attention",
            input_gradient=input_gradient,
            segment_results=segment_results,
        )

    def backward_mlp_segment(
        self,
        *,
        record: MLPSegmentExecutionRecord,
        runtime_records: RuntimeRecordStore,
        grad_segment_output: Tensor,
        overwrite_gradient_records: bool = True,
    ) -> SegmentBackwardResult:
        """Recompute one MLP segment and compute its gradients."""

        input_tensor = self._resolve_input_tensor(record, runtime_records)
        input_tensor = self._prepare_recompute_input(input_tensor)
        grad_output = grad_segment_output.detach().to(self.device)

        with self.segment_loader.acquire_segment(record.segment_id) as segment:
            if record.rng_state is None:
                recomputed_output = segment(input_tensor)
                match, max_abs_diff = self._check_recomputed_output(
                    recomputed_output,
                    record.output_tensor,
                    segment_id=record.segment_id,
                )
                input_gradient, parameter_gradients = self._compute_gradients(
                    module=segment,
                    input_tensor=input_tensor,
                    output_tensor=recomputed_output,
                    grad_output=grad_output,
                )
            else:
                with restored_rng_state(
                    record.rng_state,
                    restore_cuda=record.rng_state.has_cuda_state,
                ):
                    recomputed_output = segment(input_tensor)
                    match, max_abs_diff = self._check_recomputed_output(
                        recomputed_output,
                        record.output_tensor,
                        segment_id=record.segment_id,
                    )
                    input_gradient, parameter_gradients = self._compute_gradients(
                        module=segment,
                        input_tensor=input_tensor,
                        output_tensor=recomputed_output,
                        grad_output=grad_output,
                    )
            # Free recomputed tensors before the segment is unloaded so no
            # autograd graph nodes hold references to segment parameters.
            del recomputed_output, input_tensor, grad_output
            self._release_memory()

        # gc.collect() runs OUTSIDE the with block so the _GeneratorContextManager
        # is no longer a GC root — Python reference cycles that kept GPU tensors
        # alive inside the recomputation window can now be collected.
        gc.collect()
        torch.cuda.empty_cache()
        result = SegmentBackwardResult(
            segment_id=record.segment_id,
            input_gradient=input_gradient,
            parameter_gradients=parameter_gradients,
            recomputation_matched=match,
            max_abs_diff=max_abs_diff,
        )
        runtime_records.add_mlp_gradient_record(
            MLPSegmentGradientRecord(
                segment_id=record.segment_id,
                parameter_gradients=parameter_gradients,
                input_gradient=input_gradient,
                metadata={
                    "recomputation_matched": match,
                    "max_abs_diff": max_abs_diff,
                },
            ),
            overwrite=overwrite_gradient_records,
        )
        self._require_no_active_segment()
        return result

    def backward_attention_segment(
        self,
        *,
        record: AttentionSegmentExecutionRecord,
        runtime_records: RuntimeRecordStore,
        grad_segment_output: Tensor,
        overwrite_gradient_records: bool = True,
    ) -> SegmentBackwardResult:
        """Recompute one attention segment and compute its gradients."""

        input_tensor = self._resolve_input_tensor(record, runtime_records)
        input_tensor = self._prepare_recompute_input(input_tensor)
        attention_mask = self._resolve_attention_mask(record, runtime_records)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        grad_output = grad_segment_output.detach().to(self.device)

        with self.segment_loader.acquire_segment(record.segment_id) as segment:
            if record.rng_state is None:
                recomputed_output = segment(input_tensor, attention_mask)
                match, max_abs_diff = self._check_recomputed_output(
                    recomputed_output,
                    record.output_tensor,
                    segment_id=record.segment_id,
                )
                input_gradient, parameter_gradients = self._compute_gradients(
                    module=segment,
                    input_tensor=input_tensor,
                    output_tensor=recomputed_output,
                    grad_output=grad_output,
                )
            else:
                with restored_rng_state(
                    record.rng_state,
                    restore_cuda=record.rng_state.has_cuda_state,
                ):
                    recomputed_output = segment(input_tensor, attention_mask)
                    match, max_abs_diff = self._check_recomputed_output(
                        recomputed_output,
                        record.output_tensor,
                        segment_id=record.segment_id,
                    )
                    input_gradient, parameter_gradients = self._compute_gradients(
                        module=segment,
                        input_tensor=input_tensor,
                        output_tensor=recomputed_output,
                        grad_output=grad_output,
                    )
            del recomputed_output, input_tensor, grad_output
            if attention_mask is not None:
                del attention_mask
            self._release_memory()

        # gc.collect() runs OUTSIDE the with block so the _GeneratorContextManager
        # is no longer a GC root — Python reference cycles that kept GPU tensors
        # alive inside the recomputation window can now be collected.
        gc.collect()
        torch.cuda.empty_cache()
        result = SegmentBackwardResult(
            segment_id=record.segment_id,
            input_gradient=input_gradient,
            parameter_gradients=parameter_gradients,
            recomputation_matched=match,
            max_abs_diff=max_abs_diff,
        )
        runtime_records.add_attention_gradient_record(
            AttentionSegmentGradientRecord(
                segment_id=record.segment_id,
                parameter_gradients=parameter_gradients,
                input_gradient=input_gradient,
                metadata={
                    "recomputation_matched": match,
                    "max_abs_diff": max_abs_diff,
                },
            ),
            overwrite=overwrite_gradient_records,
        )
        self._require_no_active_segment()
        return result

    def _compute_gradients(
        self,
        *,
        module: nn.Module,
        input_tensor: Tensor,
        output_tensor: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if output_tensor.shape != grad_output.shape:
            raise ValueError(
                "grad_output shape must match recomputed segment output shape, got "
                f"grad_output={tuple(grad_output.shape)}, "
                f"output={tuple(output_tensor.shape)}."
            )

        named_parameters = [(name, param) for name, param in module.named_parameters()]
        gradient_targets: list[Tensor] = [input_tensor]
        gradient_targets.extend(param for _, param in named_parameters)

        gradients = torch.autograd.grad(
            outputs=output_tensor,
            inputs=gradient_targets,
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        # output_tensor's autograd graph is freed (retain_graph=False).
        # Explicitly release gradient_targets so the graph nodes are dropped now.
        del gradient_targets

        input_grad_raw = gradients[0]
        if input_grad_raw is None:
            input_gradient = torch.zeros_like(input_tensor).cpu()
        else:
            input_gradient = input_grad_raw.detach().cpu().clone()
            del input_grad_raw

        parameter_gradients: dict[str, Tensor] = {}
        for (name, parameter), gradient in zip(named_parameters, gradients[1:], strict=True):
            if gradient is None:
                parameter_gradients[name] = torch.zeros_like(parameter).cpu()
            else:
                parameter_gradients[name] = gradient.detach().cpu().clone()
        del gradients, named_parameters

        return input_gradient, parameter_gradients

    def _check_recomputed_output(
        self,
        recomputed_output: Tensor,
        stored_output: Tensor,
        *,
        segment_id: SegmentId,
    ) -> tuple[bool, float]:
        # When the check is disabled (e.g. minimal records, where stored_output is
        # a zero-element placeholder and not the real output), do not inspect
        # stored_output at all.
        if not self.recomputation_check.enabled:
            return True, 0.0
        stored = stored_output.detach().to(recomputed_output.device, dtype=recomputed_output.dtype)
        if recomputed_output.shape != stored.shape:
            raise RecomputationMismatchError(
                f"Recomputed output shape mismatch for {segment_id.to_key()}: "
                f"recomputed={tuple(recomputed_output.shape)}, stored={tuple(stored.shape)}."
            )

        diff = (recomputed_output.detach() - stored).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
        matched = torch.allclose(
            recomputed_output.detach(),
            stored,
            atol=self.recomputation_check.atol,
            rtol=self.recomputation_check.rtol,
        )
        if not matched and self.recomputation_check.raise_on_mismatch:
            raise RecomputationMismatchError(
                f"Recomputation mismatch for {segment_id.to_key()}: "
                f"max_abs_diff={max_abs_diff}, "
                f"atol={self.recomputation_check.atol}, "
                f"rtol={self.recomputation_check.rtol}."
            )
        return matched, max_abs_diff

    def _resolve_input_tensor(
        self,
        record: AttentionSegmentExecutionRecord | MLPSegmentExecutionRecord,
        runtime_records: RuntimeRecordStore,
    ) -> Tensor:
        if record.input_tensor is not None:
            return record.input_tensor
        if record.input_reference is None:
            raise ValueError(f"No input tensor/reference for {record.segment_id.to_key()}.")
        return runtime_records.get_shared_tensor(record.input_reference)

    def _resolve_attention_mask(
        self,
        record: AttentionSegmentExecutionRecord,
        runtime_records: RuntimeRecordStore,
    ) -> Tensor | None:
        if record.attention_mask_tensor is not None:
            return record.attention_mask_tensor
        if record.attention_mask_reference is None:
            return None
        return runtime_records.get_shared_tensor(record.attention_mask_reference)

    def _prepare_recompute_input(self, tensor: Tensor) -> Tensor:
        if not torch.is_floating_point(tensor):
            raise TypeError("Segment recomputation input must be a floating point tensor.")
        return tensor.detach().clone().to(self.device).requires_grad_(True)

    @staticmethod
    def _sum_input_gradients(input_gradients: Sequence[Tensor]) -> Tensor:
        if not input_gradients:
            raise ValueError("input_gradients must not be empty.")
        total = input_gradients[0]
        for gradient in input_gradients[1:]:
            if gradient.shape != total.shape:
                raise ValueError(
                    "All input gradients must have the same shape, got "
                    f"{tuple(total.shape)} and {tuple(gradient.shape)}."
                )
            total = total + gradient
        return total

    def _validate_layer_id(self, layer_id: int) -> None:
        if not isinstance(layer_id, int):
            raise TypeError(f"layer_id must be an int, got {type(layer_id).__name__}.")
        if layer_id < 0 or layer_id >= self.model_config.n_layers:
            raise ValueError(
                f"layer_id must be in [0, {self.model_config.n_layers - 1}], "
                f"got {layer_id}."
            )

    @staticmethod
    def _release_memory() -> None:
        """Force Python GC and return freed heap pages to the OS.

        gc.collect() drops zero-refcount Python objects and returns their memory
        to the CPython allocator pool, but does NOT return pages to the OS.
        On Linux, malloc_trim(0) tells glibc to release free heap pages back to
        the kernel so they don't show as RSS. This is essential for CPU training
        where the OS-visible RSS is the real constraint.
        """
        gc.collect()
        if sys.platform.startswith("linux"):
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    def _require_no_active_segment(self) -> None:
        if self.segment_loader.active_segment_count != 0:
            raise RuntimeError("Segment loader still has an active segment after backward.")

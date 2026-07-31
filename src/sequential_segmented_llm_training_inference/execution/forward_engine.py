"""Segmented forward engine.

This module executes the transformer in strict single-segment order. For
training under true recomputation, callers can store detached/offloaded runtime
records and then run backward from those records without preserving the original
forward autograd graph.

Memory model
------------
Three groups of parameters are supported:

1. **Always-resident** (tiny, ~2 MB): ``layer_norms`` and
   ``mlp_shared_output_biases`` remain as ``nn.Module`` attributes of this
   engine and are managed by the non-segment optimizer.

2. **Segment-store-managed** (large): ``embedding``, ``output_head``, and
   per-layer ``attention_output_proj`` are stored in the segment store and
   loaded/unloaded on demand, exactly like attention/MLP segments.  These are
   updated by the segment optimizer.  To opt in, set
   ``use_segment_store_for_global=True`` (the default when ``embeddings`` /
   ``output_head`` / ``attention_output_projections`` are *not* passed in).

3. **Legacy always-resident** (backward-compatible): pass explicit
   ``embeddings``, ``layer_norms``, ``attention_output_projections``,
   ``mlp_shared_output_biases``, and ``output_head`` objects to keep the old
   behaviour where all five live in RAM permanently.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import (
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.execution.composition import (
    concatenate_attention_outputs,
    residual_add,
    sum_mlp_outputs,
)
from sequential_segmented_llm_training_inference.execution.segment_loader import (
    StrictSegmentLoader,
)
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
from sequential_segmented_llm_training_inference.execution.rng import capture_rng_state
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.runtime_records import (
    AttentionCompositionRecord,
    AttentionSegmentExecutionRecord,
    MLPCompositionRecord,
    MLPSegmentExecutionRecord,
    ResidualCompositionRecord,
    RuntimeRecordStore,
)


@dataclass(slots=True)
class SegmentedForwardOutput:
    """Output returned by :class:`SegmentedForwardEngine`."""

    logits: Tensor
    hidden_states: Tensor
    runtime_records: RuntimeRecordStore | None = None
    attention_segment_outputs: dict[SegmentId, Tensor] = field(default_factory=dict)
    mlp_segment_outputs: dict[SegmentId, Tensor] = field(default_factory=dict)


class SegmentedForwardEngine(nn.Module):
    """Forward-only strict segmented transformer executor.

    Parameters
    ----------
    use_segment_store_for_global:
        When *True* (default when ``embeddings``/``output_head``/
        ``attention_output_projections`` are not provided), the embedding table,
        output head, and per-layer attention output projections are kept in the
        segment store and loaded on demand.  When *False* (or when the legacy
        module arguments are supplied), those components live permanently in RAM
        as nn.Module attributes on this engine.
    store_segment_outputs:
        When *True* (default), each per-segment execution record retains the
        detached/offloaded segment output tensor.  When *False* (minimal
        records), per-segment output tensors are not retained — only shape
        metadata is kept (the backward scheduler splits the attention concat
        gradient via ``record.shape`` and the recompute check must be disabled).
    """

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        segmentation_config: SegmentationConfig,
        segment_loader: StrictSegmentLoader,
        embeddings: TokenPositionEmbeddings | None = None,
        layer_norms: SegmentedLayerNorms | None = None,
        attention_output_projections: AttentionOutputProjections | None = None,
        mlp_shared_output_biases: MLPSharedOutputBiases | None = None,
        output_head: FinalNormLMHead | None = None,
        residual_dropout: float | None = None,
        use_segment_store_for_global: bool | None = None,
        store_segment_outputs: bool = True,
    ) -> None:
        super().__init__()
        model_config.validate()
        segmentation_config.validate_against_model(model_config)
        if segment_loader is None:
            raise ValueError("segment_loader must be provided.")

        self.model_config = model_config
        self.segmentation_config = segmentation_config
        self.segment_loader = segment_loader
        self.store_segment_outputs = store_segment_outputs

        layer_component_config = LayerComponentConfig(
            n_layers=model_config.n_layers,
            d_model=model_config.d_model,
            use_mlp_shared_output_bias=True,
        )

        # Decide whether to use segment-store offloading for the three large
        # components.  Legacy modules take precedence; if none are provided the
        # caller can opt in via use_segment_store_for_global=True.
        _legacy_provided = any(
            x is not None
            for x in (embeddings, attention_output_projections, output_head)
        )
        if use_segment_store_for_global is None:
            # Default: legacy always-resident mode unless explicitly opted in.
            use_segment_store_for_global = False
        if _legacy_provided and use_segment_store_for_global:
            raise ValueError(
                "Cannot pass explicit embeddings/attention_output_projections/output_head "
                "together with use_segment_store_for_global=True."
            )
        self._use_segment_store_for_global: bool = use_segment_store_for_global

        if self._use_segment_store_for_global:
            # Segment-store mode: do NOT create persistent nn.Module for the
            # large components; they live in the store.
            n_emb = segmentation_config.embedding_segments
            n_head = segmentation_config.output_head_segments
            self._embedding_seg_id: SegmentId | None = SegmentId(-1, "embedding", 0)
            self._embedding_seg_ids: list[SegmentId] = [
                SegmentId(-1, "embedding", i) for i in range(n_emb)
            ]
            self._output_head_seg_id: SegmentId | None = SegmentId(-1, "output_head", 0)
            self._output_head_seg_ids: list[SegmentId] = [
                SegmentId(-1, "output_head", i) for i in range(n_head)
            ]
            self._attn_proj_seg_ids: list[SegmentId] | None = [
                SegmentId(layer_id, "attention_output_proj", 0)
                for layer_id in range(model_config.n_layers)
            ]
            # Always-resident dropout for sliced embedding (N>1 only; N=1 uses
            # dropout inside the segment module itself).
            if n_emb > 1:
                self.embedding_slice_dropout = nn.Dropout(model_config.dropout)
            # Always-resident LayerNorm for sliced output head (N>1 only; N=1
            # keeps LayerNorm inside the segment module).
            if n_head > 1:
                self.output_slice_final_norm = nn.LayerNorm(
                    model_config.d_model, eps=1e-5
                )
            # Keep None sentinels so attribute access raises AttributeError
            # rather than silently returning stale data.
            self.embeddings: TokenPositionEmbeddings | None = None
            self.attention_output_projections: AttentionOutputProjections | None = None
            self.output_head: FinalNormLMHead | None = None
        else:
            # Legacy / explicit-module mode: keep all components always in RAM.
            embedding_config = EmbeddingConfig(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                max_seq_len=model_config.max_seq_len,
                dropout=model_config.dropout,
                pad_token_id=model_config.pad_token_id,
            )
            output_head_config = OutputHeadConfig(
                d_model=model_config.d_model,
                vocab_size=model_config.vocab_size,
            )
            self.embeddings = embeddings or TokenPositionEmbeddings(embedding_config)
            self.attention_output_projections = (
                attention_output_projections or AttentionOutputProjections(layer_component_config)
            )
            self.output_head = output_head or FinalNormLMHead(output_head_config)
            self._embedding_seg_id = None
            self._output_head_seg_id = None
            self._attn_proj_seg_ids = None

        # Always-resident tiny components (~2 MB total).
        self.layer_norms = layer_norms or SegmentedLayerNorms(layer_component_config)
        self.mlp_shared_output_biases = (
            mlp_shared_output_biases or MLPSharedOutputBiases(layer_component_config)
        )

        dropout_probability = model_config.dropout if residual_dropout is None else residual_dropout
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                f"residual_dropout must be in [0, 1), got {dropout_probability}."
            )
        self.residual_dropout = nn.Dropout(dropout_probability)

    # ------------------------------------------------------------------
    # Internal helpers: embedding / attention_proj / output_head access
    # ------------------------------------------------------------------

    def _run_embeddings(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor],
    ) -> Tensor:
        """Run the embedding forward pass.  Loads from segment store if needed."""
        if self._use_segment_store_for_global:
            if len(self._embedding_seg_ids) == 1:
                with self.segment_loader.acquire_segment(self._embedding_seg_id) as emb_mod:
                    return emb_mod(input_ids, position_ids)
            # N>1: run slices sequentially on whatever device input_ids is on,
            # then concat and apply dropout — no CPU round-trip for GPU runs.
            parts = []
            for seg_id in self._embedding_seg_ids:
                with self.segment_loader.acquire_segment(seg_id) as emb_slice:
                    parts.append(emb_slice(input_ids, position_ids).detach())
            return self.embedding_slice_dropout(torch.cat(parts, dim=-1))
        else:
            return self.embeddings(input_ids, position_ids=position_ids)  # type: ignore[union-attr]

    def _run_attention_output_proj(
        self,
        layer_id: int,
        attention_concat: Tensor,
    ) -> Tensor:
        """Run the attention output projection for one layer."""
        if self._use_segment_store_for_global:
            seg_id = self._attn_proj_seg_ids[layer_id]  # type: ignore[index]
            with self.segment_loader.acquire_segment(seg_id) as proj_mod:
                return proj_mod(attention_concat)
        else:
            return self.attention_output_projections(layer_id, attention_concat)  # type: ignore[union-attr]

    def _run_output_head(self, hidden_states: Tensor) -> Tensor:
        """Run the final norm + LM head."""
        if self._use_segment_store_for_global:
            if len(self._output_head_seg_ids) == 1:
                with self.segment_loader.acquire_segment(self._output_head_seg_id) as head_mod:
                    return head_mod(hidden_states)
            # N>1 vocab-sliced: apply LayerNorm once, then each slice projects
            # full hidden → [B, S, vocab/N].  Each partial is immediately moved
            # to CPU so only one segment's output sits in VRAM at a time.
            # Training: return on CPU — backward uses incremental log-sum-exp
            #   and never needs full GPU logits; logging loss computes on CPU.
            # Inference: move concatenated result back to compute device for
            #   argmax/sampling (generation) or loss evaluation.
            hidden_norm = self.output_slice_final_norm(hidden_states)
            dev = hidden_states.device
            parts: list[Tensor] = []
            for seg_id in self._output_head_seg_ids:
                with self.segment_loader.acquire_segment(seg_id) as head_slice:
                    partial = head_slice(hidden_norm)  # [B, S, vocab/N]
                parts.append(partial.detach().cpu())
            full = torch.cat(parts, dim=-1)  # [B, S, vocab] on CPU
            return full if self.training else full.to(dev)
        else:
            return self.output_head(hidden_states)  # type: ignore[union-attr]

    def compute_chunked_output_head_eval(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        ignore_index: int = -100,
        seq_chunk: int = 32,
    ) -> tuple[Tensor, float]:
        """Compute CE loss and token accuracy without materialising full logits.

        Processes one vocab slice × one sequence chunk at a time. Peak memory =
        [B, seq_chunk, vocab/N] per acquisition instead of [B, S-1, vocab/N].

        Only valid when output_head_segments > 1 and _use_segment_store_for_global.

        Returns:
            (scalar_loss, token_accuracy)  — both on CPU.
        """
        assert self._use_segment_store_for_global and len(self._output_head_seg_ids) > 1

        shift_labels = labels[:, 1:].contiguous().cpu()  # [B, S-1]
        B, Sm1 = shift_labels.shape
        active_mask = (shift_labels != ignore_index)      # [B, S-1] CPU
        n_active = int(active_mask.sum().clamp(min=1).item())

        running_lse = torch.full((B, Sm1), float("-inf"), dtype=hidden_states.dtype)
        correct_logit = torch.full((B, Sm1), float("-inf"), dtype=hidden_states.dtype)
        running_max_val = torch.full((B, Sm1), float("-inf"), dtype=hidden_states.dtype)
        running_argmax = torch.zeros((B, Sm1), dtype=torch.long)

        step = seq_chunk if seq_chunk > 0 else Sm1
        for t0 in range(0, Sm1, step):
            t1 = min(t0 + step, Sm1)
            hidden_chunk = hidden_states[:, t0:t1, :]     # [B, chunk, D]
            with torch.no_grad():
                norm_chunk = self.output_slice_final_norm(hidden_chunk)  # [B, chunk, D]

            sl_chunk = shift_labels[:, t0:t1]             # [B, chunk]
            am_chunk = active_mask[:, t0:t1]              # [B, chunk]
            vocab_start = 0

            for seg_id in self._output_head_seg_ids:
                with self.segment_loader.acquire_segment(seg_id) as head_slice:
                    with torch.no_grad():
                        z_i = head_slice(norm_chunk).cpu()  # [B, chunk, V/N]
                this_slice = z_i.shape[-1]

                # Running log-sum-exp for loss
                lse_i = torch.logsumexp(z_i, dim=-1)      # [B, chunk]
                running_lse[:, t0:t1] = torch.logaddexp(running_lse[:, t0:t1], lse_i)

                # Correct logit for labels in this vocab slice's range
                in_range = am_chunk & (sl_chunk >= vocab_start) & (sl_chunk < vocab_start + this_slice)
                if in_range.any():
                    idx = (sl_chunk - vocab_start).clamp(0, this_slice - 1)
                    gathered = z_i.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                    correct_logit[:, t0:t1][in_range] = gathered[in_range]

                # Running argmax for token accuracy
                slice_max_val, slice_argmax = z_i.max(dim=-1)  # [B, chunk]
                rmv_chunk = running_max_val[:, t0:t1]
                better = slice_max_val > rmv_chunk
                running_max_val[:, t0:t1] = torch.where(better, slice_max_val, rmv_chunk)
                running_argmax[:, t0:t1] = torch.where(
                    better, vocab_start + slice_argmax, running_argmax[:, t0:t1]
                )

                vocab_start += this_slice
                del z_i

        # Scalar CE loss — use where() to avoid inf*0=NaN at ignored positions.
        zero = running_lse.new_zeros(())
        per_token = torch.where(active_mask, running_lse - correct_logit, zero)
        loss = per_token.sum() / n_active

        # Token accuracy
        correct_tokens = (running_argmax == shift_labels) & active_mask
        token_acc = correct_tokens.sum().float().item() / n_active

        return loss, token_acc

    def _build_output_record_fields(
        self,
        segment_output: Tensor,
        *,
        detach: bool,
        clone: bool,
        to_cpu: bool,
    ) -> dict[str, object]:
        """Output-tensor record fields honouring ``store_segment_outputs``.

        Minimal records: zero-element placeholder + explicit shape, so the
        backward scheduler can split via ``record.shape`` and skips the recompute
        check.
        """
        if self.store_segment_outputs:
            return {
                "output_tensor": self._record_tensor(
                    segment_output, detach=detach, clone=clone, to_cpu=to_cpu
                ),
                "shape": tuple(int(d) for d in segment_output.shape),
            }
        return {
            "output_tensor": segment_output.detach().new_empty(0),
            "shape": tuple(int(d) for d in segment_output.shape),
        }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        position_ids: Optional[Tensor] = None,
        store_records: bool = True,
        step_id: int = 0,
        microbatch_id: int = 0,
        detach_record_tensors: bool = True,
        clone_record_tensors: bool = True,
        record_tensors_to_cpu: bool = False,
        return_segment_outputs: bool = False,
        compute_output_head: bool = True,
        runtime_records_store: "RuntimeRecordStore | None" = None,
        store_only_layer_inputs: bool = False,
    ) -> SegmentedForwardOutput:
        """Run segmented forward execution.

        ``record_tensors_to_cpu`` is primarily for true recomputation training:
        it keeps detached runtime records outside GPU activation memory while the
        backward scheduler reloads only the currently recomputed unit.
        """

        self._validate_input_ids(input_ids)
        if attention_mask is not None:
            self._validate_attention_mask(attention_mask, input_ids)

        if runtime_records_store is not None:
            runtime_records = runtime_records_store
        elif store_records:
            runtime_records = RuntimeRecordStore()
        else:
            runtime_records = None
        attention_outputs_by_id: dict[SegmentId, Tensor] = {}
        mlp_outputs_by_id: dict[SegmentId, Tensor] = {}

        if runtime_records is not None:
            runtime_records.metadata["input_ids"] = self._record_tensor(
                input_ids,
                detach=True,
                clone=True,
                to_cpu=record_tensors_to_cpu,
            )
            if position_ids is not None:
                runtime_records.metadata["position_ids"] = self._record_tensor(
                    position_ids,
                    detach=True,
                    clone=True,
                    to_cpu=record_tensors_to_cpu,
                )
            runtime_records.metadata["embedding_dropout_rng_state"] = capture_rng_state()

        hidden_states = self._run_embeddings(input_ids, position_ids)

        if runtime_records is not None:
            runtime_records.add_shared_tensor(
                "embedding_output",
                self._record_tensor(
                    hidden_states,
                    detach=detach_record_tensors,
                    clone=clone_record_tensors,
                    to_cpu=record_tensors_to_cpu,
                ),
                detach=False,
                clone=False,
            )

        # Lazy-records mode: store attention_mask once in metadata (all layers share it).
        if store_only_layer_inputs and runtime_records is not None and attention_mask is not None:
            runtime_records.metadata["_lazy_attention_mask"] = self._record_tensor(
                attention_mask,
                detach=True,
                clone=True,
                to_cpu=record_tensors_to_cpu,
            )

        for layer_id in range(self.model_config.n_layers):
            layer_input_key = f"layer_{layer_id}.input"
            if runtime_records is not None:
                runtime_records.add_shared_tensor(
                    layer_input_key,
                    self._record_tensor(
                        hidden_states,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )

            attention_input = self.layer_norms.attention(layer_id, hidden_states)
            attention_input_key = f"layer_{layer_id}.attention_input"
            attention_mask_key = f"layer_{layer_id}.attention_mask" if attention_mask is not None else None

            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    attention_input_key,
                    self._record_tensor(
                        attention_input,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )
                if attention_mask is not None and attention_mask_key is not None:
                    runtime_records.add_shared_tensor(
                        attention_mask_key,
                        self._record_tensor(
                            attention_mask,
                            detach=detach_record_tensors,
                            clone=clone_record_tensors,
                            to_cpu=record_tensors_to_cpu,
                        ),
                        detach=False,
                        clone=False,
                    )

            layer_attention_outputs: list[Tensor] = []
            attention_segment_order: list[int] = []
            for segment_index in range(self.segmentation_config.attention_segments):
                segment_id = SegmentId(
                    layer_id=layer_id,
                    segment_type="attention",
                    segment_id=segment_index,
                )
                with self.segment_loader.acquire_segment(segment_id) as segment:
                    rng_state = capture_rng_state() if store_records else None
                    segment_output = segment(attention_input, attention_mask)

                layer_attention_outputs.append(segment_output)
                attention_segment_order.append(segment_index)
                if return_segment_outputs:
                    attention_outputs_by_id[segment_id] = segment_output

                if runtime_records is not None:
                    _out_fields = self._build_output_record_fields(
                        segment_output,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    )
                    runtime_records.add_attention_record(
                        AttentionSegmentExecutionRecord(
                            segment_id=segment_id,
                            step_id=step_id,
                            microbatch_id=microbatch_id,
                            output_tensor=_out_fields["output_tensor"],
                            shape=_out_fields["shape"],
                            input_reference=attention_input_key,
                            attention_mask_reference=attention_mask_key,
                            rng_state=rng_state,
                            metadata={"layer_id": layer_id, "segment_index": segment_index},
                        )
                    )

            attention_concat = concatenate_attention_outputs(layer_attention_outputs)
            del layer_attention_outputs  # release individual segment outputs; only concat needed now
            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    f"layer_{layer_id}.attention_concat",
                    self._record_tensor(
                        attention_concat,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )

            attention_projected = self._run_attention_output_proj(layer_id, attention_concat)
            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    f"layer_{layer_id}.attention_projected_pre_dropout",
                    self._record_tensor(
                        attention_projected,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )
            if runtime_records is not None:
                runtime_records.metadata[f"layer_{layer_id}.attention_dropout_rng_state"] = capture_rng_state()

            attention_projected = self.residual_dropout(attention_projected)
            hidden_states = residual_add(hidden_states, attention_projected)

            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    f"layer_{layer_id}.post_attention_hidden",
                    self._record_tensor(
                        hidden_states,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )
                runtime_records.add_attention_composition_record(
                    AttentionCompositionRecord(
                        layer_id=layer_id,
                        segment_order=tuple(attention_segment_order),
                        output_projection_applied=True,
                        dropout_applied=self.residual_dropout.p > 0.0,
                    )
                )
                runtime_records.add_residual_composition_record(
                    ResidualCompositionRecord(layer_id=layer_id, branch_type="attention")
                )

            mlp_input = self.layer_norms.mlp(layer_id, hidden_states)
            mlp_input_key = f"layer_{layer_id}.mlp_input"
            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    mlp_input_key,
                    self._record_tensor(
                        mlp_input,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )

            mlp_running_sum: Tensor | None = None
            mlp_segment_order: list[int] = []
            for segment_index in range(self.segmentation_config.mlp_chunks):
                segment_id = SegmentId(
                    layer_id=layer_id,
                    segment_type="mlp",
                    segment_id=segment_index,
                )
                with self.segment_loader.acquire_segment(segment_id) as segment:
                    rng_state = capture_rng_state() if store_records else None
                    segment_output = segment(mlp_input)

                # Running sum: accumulate without holding all chunk outputs simultaneously.
                mlp_running_sum = segment_output if mlp_running_sum is None else mlp_running_sum + segment_output
                mlp_segment_order.append(segment_index)
                if return_segment_outputs:
                    mlp_outputs_by_id[segment_id] = segment_output

                if runtime_records is not None:
                    _out_fields = self._build_output_record_fields(
                        segment_output,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    )
                    runtime_records.add_mlp_record(
                        MLPSegmentExecutionRecord(
                            segment_id=segment_id,
                            step_id=step_id,
                            microbatch_id=microbatch_id,
                            output_tensor=_out_fields["output_tensor"],
                            shape=_out_fields["shape"],
                            input_reference=mlp_input_key,
                            rng_state=rng_state,
                            metadata={"layer_id": layer_id, "segment_index": segment_index},
                        )
                    )
                if not return_segment_outputs:
                    del segment_output

            assert mlp_running_sum is not None
            mlp_sum = mlp_running_sum
            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    f"layer_{layer_id}.mlp_sum_pre_bias",
                    self._record_tensor(
                        mlp_sum,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )

            mlp_sum = self.mlp_shared_output_biases(layer_id, mlp_sum)
            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    f"layer_{layer_id}.mlp_after_bias_pre_dropout",
                    self._record_tensor(
                        mlp_sum,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )
            if runtime_records is not None:
                runtime_records.metadata[f"layer_{layer_id}.mlp_dropout_rng_state"] = capture_rng_state()

            mlp_sum = self.residual_dropout(mlp_sum)
            hidden_states = residual_add(hidden_states, mlp_sum)

            if runtime_records is not None and not store_only_layer_inputs:
                runtime_records.add_shared_tensor(
                    f"layer_{layer_id}.output",
                    self._record_tensor(
                        hidden_states,
                        detach=detach_record_tensors,
                        clone=clone_record_tensors,
                        to_cpu=record_tensors_to_cpu,
                    ),
                    detach=False,
                    clone=False,
                )
                runtime_records.add_mlp_composition_record(
                    MLPCompositionRecord(
                        layer_id=layer_id,
                        segment_order=tuple(mlp_segment_order),
                        shared_output_bias_added_once=self.mlp_shared_output_biases.enabled,
                        dropout_applied=self.residual_dropout.p > 0.0,
                    )
                )
                runtime_records.add_residual_composition_record(
                    ResidualCompositionRecord(layer_id=layer_id, branch_type="mlp")
                )

            if self.segment_loader.active_segment_count != 0:
                raise RuntimeError("Segment loader must not keep active segments after a layer.")

            # Return freed heap pages to the OS after each layer so glibc
            # fragmentation does not inflate the peak RSS at output_head time.
            gc.collect()
            if sys.platform.startswith("linux"):
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

        if runtime_records is not None:
            runtime_records.add_shared_tensor(
                "pre_final_norm",
                self._record_tensor(
                    hidden_states,
                    detach=detach_record_tensors,
                    clone=clone_record_tensors,
                    to_cpu=record_tensors_to_cpu,
                ),
                detach=False,
                clone=False,
            )

        logits = self._run_output_head(hidden_states) if compute_output_head else hidden_states.new_empty(0)
        if self.segment_loader.active_segment_count != 0:
            raise RuntimeError("Segment loader must not keep active segments after forward.")

        return SegmentedForwardOutput(
            logits=logits,
            hidden_states=hidden_states,
            runtime_records=runtime_records,
            attention_segment_outputs=attention_outputs_by_id,
            mlp_segment_outputs=mlp_outputs_by_id,
        )

    def _validate_input_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}."
            )
        if input_ids.shape[1] > self.model_config.max_seq_len:
            raise ValueError(
                f"seq_len={input_ids.shape[1]} exceeds max_seq_len="
                f"{self.model_config.max_seq_len}."
            )
        if input_ids.numel() > 0:
            if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.model_config.vocab_size:
                raise ValueError("input_ids contain token ids outside [0, vocab_size).")

    def _validate_attention_mask(self, attention_mask: Tensor, input_ids: Tensor) -> None:
        if attention_mask.ndim == 2 and attention_mask.shape != input_ids.shape:
            raise ValueError(
                "2-D attention_mask must have the same shape as input_ids, got "
                f"attention_mask={tuple(attention_mask.shape)}, "
                f"input_ids={tuple(input_ids.shape)}."
            )

    @staticmethod
    def _record_tensor(
        tensor: Tensor,
        *,
        detach: bool,
        clone: bool,
        to_cpu: bool = False,
    ) -> Tensor:
        result = tensor.detach() if detach else tensor
        if clone:
            result = result.clone()
        if to_cpu:
            result = result.cpu()
        return result

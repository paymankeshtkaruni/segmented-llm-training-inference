"""Loadable segment wrappers for global (non-per-layer) model components.

These thin nn.Module wrappers allow the embedding table, output head, and
per-layer attention output projections to be stored in the segment store and
loaded/unloaded on demand — exactly like attention/MLP segments.  This
eliminates the large always-resident parameter footprint from those components.

Segment IDs used:
    embedding:              SegmentId(layer_id=-1, segment_type="embedding",            segment_id=0)
    output_head:            SegmentId(layer_id=-1, segment_type="output_head",          segment_id=0)
    attention_output_proj:  SegmentId(layer_id=L,  segment_type="attention_output_proj", segment_id=0)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.model.embeddings import (
    EmbeddingConfig,
    TokenPositionEmbeddings,
)
from sequential_segmented_llm_training_inference.model.output_head import (
    FinalNormLMHead,
    OutputHeadConfig,
)


class EmbeddingSegmentModule(nn.Module):
    """Wraps :class:`TokenPositionEmbeddings` for use as a loadable segment.

    The inner ``TokenPositionEmbeddings`` is exposed via ``self.embeddings`` so
    that ``state_dict()`` / ``load_state_dict()`` use the same nested key paths
    that the original model used (``embeddings.token_embedding.weight``, etc.).
    """

    def __init__(self, embedding_config: EmbeddingConfig) -> None:
        super().__init__()
        self.embeddings = TokenPositionEmbeddings(embedding_config)

    def forward(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        return self.embeddings(input_ids, position_ids=position_ids)

    @property
    def config(self) -> EmbeddingConfig:
        return self.embeddings.config


class OutputHeadSegmentModule(nn.Module):
    """Wraps :class:`FinalNormLMHead` for use as a loadable segment.

    The inner ``FinalNormLMHead`` is exposed via ``self.output_head`` so that
    ``state_dict()`` / ``load_state_dict()`` preserve the original nested paths
    (``output_head.final_norm.weight``, ``output_head.lm_head.weight``, etc.).
    """

    def __init__(self, output_head_config: OutputHeadConfig) -> None:
        super().__init__()
        self.output_head = FinalNormLMHead(output_head_config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.output_head(hidden_states)

    @property
    def config(self) -> OutputHeadConfig:
        return self.output_head.config


class AttentionOutputProjSegmentModule(nn.Module):
    """Single-layer attention output projection [d_model -> d_model].

    After the per-head segments are concatenated, this linear projection maps
    the concatenated output back to ``d_model``.  One module instance is stored
    per transformer layer in the segment store.

    Parameters are named ``proj.weight`` and ``proj.bias`` inside the
    ``state_dict()``.
    """

    def __init__(self, d_model: int, bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Input must have shape [batch, seq_len, d_model], got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Input last dimension must be d_model={self.d_model}, got {x.shape[-1]}."
            )
        return self.proj(x)


class EmbeddingSliceSegmentModule(nn.Module):
    """One d_model slice of the embedding table for use when embedding_segments > 1.

    Handles columns [slice_start:slice_end] of d_model. NO dropout — dropout is
    applied by the forward engine AFTER concatenating all slices.
    """

    def __init__(self, vocab_size: int, d_slice: int, max_seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_slice = d_slice
        self.max_seq_len = max_seq_len
        self.token_embedding_slice = nn.Embedding(vocab_size, d_slice)
        self.position_embedding_slice = nn.Embedding(max_seq_len, d_slice)
        nn.init.normal_(self.token_embedding_slice.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding_slice.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        B, S = input_ids.shape
        ids = input_ids.to(dtype=torch.long)
        if position_ids is None:
            pos = torch.arange(S, device=ids.device, dtype=torch.long).unsqueeze(0).expand(B, S)
        else:
            pos = position_ids.to(device=ids.device, dtype=torch.long)
        return self.token_embedding_slice(ids) + self.position_embedding_slice(pos)


class OutputHeadSliceSegmentModule(nn.Module):
    """One vocabulary slice of the LM head for use when output_head_segments > 1.

    Takes the full hidden state [B, S, d_model] and projects to a contiguous
    vocab range [vocab_start:vocab_end], producing [B, S, vocab_slice_size].
    Slicing over vocabulary (not d_model) lets each slice's logits be processed
    and freed independently, enabling incremental log-sum-exp loss computation
    without ever materialising the full [B, S, vocab] logits tensor.
    """

    def __init__(self, d_model: int, vocab_slice_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_slice_size = vocab_slice_size
        self.linear_slice = nn.Linear(d_model, vocab_slice_size, bias=False)
        nn.init.normal_(self.linear_slice.weight, mean=0.0, std=0.02)

    def forward(self, h: Tensor) -> Tensor:
        return self.linear_slice(h)

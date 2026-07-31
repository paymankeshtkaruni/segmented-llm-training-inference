"""Attention segment implemented by attention-head groups.

Phase 3 scope:
- Implement one self-attention segment that owns a deterministic group of heads.
- All attention segments in a layer receive the same normalized attention input.
- Each segment returns only its own head-group output.
- The attention output projection is intentionally outside this segment because
  projection is applied after concatenating all attention segment outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId


def validate_attention_segmentation(
    *,
    d_model: int,
    n_heads: int,
    attention_segments: int,
) -> None:
    """Validate attention head-group segmentation constraints."""

    if not isinstance(d_model, int):
        raise TypeError(f"d_model must be an int, got {type(d_model).__name__}.")
    if not isinstance(n_heads, int):
        raise TypeError(f"n_heads must be an int, got {type(n_heads).__name__}.")
    if not isinstance(attention_segments, int):
        raise TypeError(
            "attention_segments must be an int, "
            f"got {type(attention_segments).__name__}."
        )
    if d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {d_model}.")
    if n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {n_heads}.")
    if attention_segments <= 1:
        raise ValueError(
            "attention_segments must be > 1 for the segmented attention mode."
        )
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model must be divisible by n_heads, got d_model={d_model}, "
            f"n_heads={n_heads}."
        )
    if n_heads % attention_segments != 0:
        raise ValueError(
            "n_heads must be divisible by attention_segments, got "
            f"n_heads={n_heads}, attention_segments={attention_segments}."
        )


def attention_segment_head_range(
    *,
    n_heads: int,
    attention_segments: int,
    segment_id: int,
) -> tuple[int, int]:
    """Return the inclusive/exclusive head range owned by one segment."""

    if not isinstance(segment_id, int):
        raise TypeError(f"segment_id must be an int, got {type(segment_id).__name__}.")
    if attention_segments <= 1:
        raise ValueError(
            "attention_segments must be > 1 for the segmented attention mode."
        )
    if n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {n_heads}.")
    if n_heads % attention_segments != 0:
        raise ValueError(
            "n_heads must be divisible by attention_segments, got "
            f"n_heads={n_heads}, attention_segments={attention_segments}."
        )
    if segment_id < 0 or segment_id >= attention_segments:
        raise ValueError(
            f"segment_id must be in [0, {attention_segments - 1}], "
            f"got {segment_id}."
        )

    heads_per_segment = n_heads // attention_segments
    start_head = segment_id * heads_per_segment
    end_head = start_head + heads_per_segment
    return start_head, end_head


def attention_segment_output_dim(
    *,
    d_model: int,
    n_heads: int,
    attention_segments: int,
) -> int:
    """Return the output dimension for one attention segment."""

    validate_attention_segmentation(
        d_model=d_model,
        n_heads=n_heads,
        attention_segments=attention_segments,
    )
    head_dim = d_model // n_heads
    heads_per_segment = n_heads // attention_segments
    return head_dim * heads_per_segment


@dataclass(frozen=True, slots=True)
class AttentionSegmentMetadata:
    """Metadata describing deterministic head ownership for one segment."""

    layer_id: int
    segment_id: int
    n_heads: int
    attention_segments: int
    start_head: int
    end_head: int
    head_dim: int
    output_dim: int

    @property
    def heads_per_segment(self) -> int:
        """Number of heads owned by this segment."""
        return self.end_head - self.start_head

    @property
    def owned_heads(self) -> tuple[int, ...]:
        """Exact global head indices owned by this segment."""
        return tuple(range(self.start_head, self.end_head))

    def to_dict(self) -> dict[str, int]:
        """Return YAML/JSON-safe metadata."""
        return {
            "layer_id": self.layer_id,
            "segment_id": self.segment_id,
            "n_heads": self.n_heads,
            "attention_segments": self.attention_segments,
            "start_head": self.start_head,
            "end_head": self.end_head,
            "head_dim": self.head_dim,
            "output_dim": self.output_dim,
        }


class AttentionHeadSegment(nn.Module):
    """Self-attention segment for a deterministic group of attention heads.

    This module computes Q/K/V only for its own head group and returns the
    concatenated output of those heads. The full attention output projection is
    outside the segment and must be applied after all attention segment outputs
    are concatenated in global head order.

    Args:
        segment_id: Logical segment id. ``segment_type`` must be ``"attention"``.
        d_model: Transformer hidden size.
        n_heads: Total number of attention heads in the layer.
        attention_segments: Number of attention head-group segments.
        dropout: Attention probability dropout.
        bias: Whether Q/K/V projections use bias.
    """

    def __init__(
        self,
        *,
        segment_id: SegmentId,
        d_model: int,
        n_heads: int,
        attention_segments: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if segment_id.segment_type != "attention":
            raise ValueError(
                "AttentionHeadSegment requires segment_id.segment_type == 'attention'."
            )
        validate_attention_segmentation(
            d_model=d_model,
            n_heads=n_heads,
            attention_segments=attention_segments,
        )
        if segment_id.segment_id >= attention_segments:
            raise ValueError(
                f"segment_id.segment_id must be < attention_segments, got "
                f"{segment_id.segment_id} and {attention_segments}."
            )
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.segment_id = segment_id
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_segments = attention_segments
        self.head_dim = d_model // n_heads
        self.heads_per_segment = n_heads // attention_segments
        self.output_dim = self.heads_per_segment * self.head_dim
        self.start_head, self.end_head = attention_segment_head_range(
            n_heads=n_heads,
            attention_segments=attention_segments,
            segment_id=segment_id.segment_id,
        )

        self.q_proj = nn.Linear(d_model, self.output_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.output_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.output_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    @property
    def metadata(self) -> AttentionSegmentMetadata:
        """Return deterministic metadata for this attention segment."""

        return AttentionSegmentMetadata(
            layer_id=self.segment_id.layer_id,
            segment_id=self.segment_id.segment_id,
            n_heads=self.n_heads,
            attention_segments=self.attention_segments,
            start_head=self.start_head,
            end_head=self.end_head,
            head_dim=self.head_dim,
            output_dim=self.output_dim,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        causal: bool = True,
    ) -> Tensor:
        """Run this attention segment.

        Args:
            hidden_states: Shared normalized attention input with shape
                ``[batch, seq_len, d_model]``.
            attention_mask: Optional mask. Supported shapes:
                - ``[batch, seq_len]`` with 1/True for valid tokens;
                - ``[batch, 1, 1, seq_len]`` additive/bool mask;
                - ``[batch, 1, seq_len, seq_len]`` additive/bool mask;
                - ``[seq_len, seq_len]`` additive/bool attention mask.
            causal: Whether to apply a causal lower-triangular mask.

        Returns:
            Tensor with shape ``[batch, seq_len, heads_per_segment * head_dim]``.
        """

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, seq_len, d_model], got "
                f"{tuple(hidden_states.shape)}."
            )
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hidden_dim != self.d_model:
            raise ValueError(
                f"hidden_states last dimension must be d_model={self.d_model}, "
                f"got {hidden_dim}."
            )

        q = self._project_to_heads(self.q_proj(hidden_states), batch_size, seq_len)
        k = self._project_to_heads(self.k_proj(hidden_states), batch_size, seq_len)
        v = self._project_to_heads(self.v_proj(hidden_states), batch_size, seq_len)

        dropout_p = self.attn_dropout.p if self.training else 0.0

        if attention_mask is None and causal:
            # Fast path: PyTorch selects Flash Attention when available.
            context = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            sdpa_mask: Optional[Tensor] = None
            if causal or attention_mask is not None:
                sdpa_mask = self._build_sdpa_mask(
                    attention_mask, batch_size, seq_len, q.dtype, q.device, causal
                )
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=dropout_p, is_causal=False
            )

        context = context.transpose(1, 2).contiguous()
        return context.view(batch_size, seq_len, self.output_dim)

    def _project_to_heads(self, projected: Tensor, batch_size: int, seq_len: int) -> Tensor:
        return projected.view(
            batch_size,
            seq_len,
            self.heads_per_segment,
            self.head_dim,
        ).transpose(1, 2)

    def _build_sdpa_mask(
        self,
        attention_mask: Optional[Tensor],
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool,
    ) -> Tensor:
        """Build an additive float mask for F.scaled_dot_product_attention."""
        out = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device)
        if causal:
            upper = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(diagonal=1)
            out = out.masked_fill(upper.view(1, 1, seq_len, seq_len), float("-inf"))

        if attention_mask is None:
            return out

        am = attention_mask.to(device=device)

        if am.dtype == torch.bool:
            if am.ndim == 2:
                if am.shape == (batch_size, seq_len):
                    pad = torch.zeros(batch_size, 1, 1, seq_len, dtype=dtype, device=device)
                    return out + pad.masked_fill(~am.view(batch_size, 1, 1, seq_len), float("-inf"))
                if am.shape == (seq_len, seq_len):
                    blocked = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device)
                    return out + blocked.masked_fill(~am.view(1, 1, seq_len, seq_len), float("-inf"))
            if am.ndim == 3:
                blocked = torch.zeros(am.shape[0], 1, am.shape[1], am.shape[2], dtype=dtype, device=device)
                return out + blocked.masked_fill(~am.unsqueeze(1), float("-inf"))
            if am.ndim == 4:
                blocked = torch.zeros_like(am, dtype=dtype)
                return out + blocked.masked_fill(~am, float("-inf"))
            raise ValueError(f"Unsupported bool attention_mask shape: {tuple(am.shape)}.")

        if am.ndim == 2:
            if am.shape == (batch_size, seq_len):
                pad = torch.zeros(batch_size, 1, 1, seq_len, dtype=dtype, device=device)
                return out + pad.masked_fill(~am.bool().view(batch_size, 1, 1, seq_len), float("-inf"))
            if am.shape == (seq_len, seq_len):
                return out + am.to(dtype=dtype).view(1, 1, seq_len, seq_len)
        if am.ndim == 3:
            return out + am.to(dtype=dtype).unsqueeze(1)
        if am.ndim == 4:
            return out + am.to(dtype=dtype)

        raise ValueError(
            "Unsupported attention_mask shape. Expected [batch, seq_len], "
            "[seq_len, seq_len], [batch, 1, 1, seq_len], or "
            f"[batch, 1, seq_len, seq_len], got {tuple(am.shape)}."
        )

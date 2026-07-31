"""Forward composition utilities for segmented transformer execution.

Phase 10 scope:
- Compose attention segment outputs by deterministic concatenation.
- Compose MLP segment outputs by summation.
- Apply residual addition with shape validation.

These utilities are intentionally small and stateless so the backward engine can
reuse the same composition semantics later.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _require_non_empty_tensors(name: str, tensors: Sequence[Tensor]) -> None:
    if not tensors:
        raise ValueError(f"{name} must contain at least one tensor.")
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name}[{index}] must be a torch.Tensor.")
        if tensor.ndim != 3:
            raise ValueError(
                f"{name}[{index}] must have shape [batch, seq_len, hidden], "
                f"got {tuple(tensor.shape)}."
            )


def concatenate_attention_outputs(segment_outputs: Sequence[Tensor]) -> Tensor:
    """Concatenate attention segment outputs in deterministic head/segment order.

    Args:
        segment_outputs: Sequence of tensors ordered by attention segment index.
            Each tensor must have shape ``[batch, seq_len, segment_dim]``.

    Returns:
        Tensor with shape ``[batch, seq_len, d_model]`` when all segment outputs
        cover the full attention head space.
    """

    _require_non_empty_tensors("segment_outputs", segment_outputs)
    first_shape = segment_outputs[0].shape
    for index, tensor in enumerate(segment_outputs[1:], start=1):
        if tensor.shape[:2] != first_shape[:2]:
            raise ValueError(
                "All attention segment outputs must have the same batch and "
                f"sequence dimensions. segment_outputs[0]={tuple(first_shape)}, "
                f"segment_outputs[{index}]={tuple(tensor.shape)}."
            )
    return torch.cat(tuple(segment_outputs), dim=-1)


def sum_mlp_outputs(segment_outputs: Sequence[Tensor]) -> Tensor:
    """Sum MLP segment outputs.

    Args:
        segment_outputs: Sequence of tensors ordered by MLP segment index.
            Each tensor must have shape ``[batch, seq_len, d_model]``.

    Returns:
        Tensor with shape ``[batch, seq_len, d_model]``.
    """

    _require_non_empty_tensors("segment_outputs", segment_outputs)
    expected_shape = segment_outputs[0].shape
    for index, tensor in enumerate(segment_outputs[1:], start=1):
        if tensor.shape != expected_shape:
            raise ValueError(
                "All MLP segment outputs must have identical shapes. "
                f"segment_outputs[0]={tuple(expected_shape)}, "
                f"segment_outputs[{index}]={tuple(tensor.shape)}."
            )

    total = segment_outputs[0]
    for tensor in segment_outputs[1:]:
        total = total + tensor
    return total


def residual_add(residual_stream: Tensor, branch_output: Tensor) -> Tensor:
    """Apply a residual connection with shape validation."""

    if residual_stream.shape != branch_output.shape:
        raise ValueError(
            "Residual addition requires identical shapes, got "
            f"residual_stream={tuple(residual_stream.shape)}, "
            f"branch_output={tuple(branch_output.shape)}."
        )
    return residual_stream + branch_output

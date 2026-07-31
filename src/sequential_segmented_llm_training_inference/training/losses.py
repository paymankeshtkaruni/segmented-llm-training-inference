"""Causal language-modeling losses.

Phase 11 scope:
- Implement causal cross-entropy with internal shifting.
- Support ignore_index, especially -100 for masked prompt/padding tokens.
- Validate input shapes clearly.
- Return scalar loss for standard training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


LabelShiftPolicy = Literal["internal"]
LossReduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True, slots=True)
class CausalCrossEntropyConfig:
    """Configuration for causal language-modeling cross-entropy."""

    ignore_index: int = -100
    label_shift: LabelShiftPolicy = "internal"
    reduction: LossReduction = "mean"

    def __post_init__(self) -> None:
        if self.label_shift != "internal":
            raise ValueError(
                "Only label_shift='internal' is supported in Phase 11."
            )
        if self.reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be one of {'mean', 'sum', 'none'}, "
                f"got {self.reduction!r}."
            )


def validate_causal_lm_shapes(logits: Tensor, labels: Tensor) -> None:
    """Validate shapes for internal-shift causal LM loss."""

    if not isinstance(logits, Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}.")
    if not isinstance(labels, Tensor):
        raise TypeError(f"labels must be a torch.Tensor, got {type(labels).__name__}.")

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch_size, seq_len, vocab_size], "
            f"got shape {tuple(logits.shape)}."
        )
    if labels.ndim != 2:
        raise ValueError(
            "labels must have shape [batch_size, seq_len], "
            f"got shape {tuple(labels.shape)}."
        )

    batch_size, seq_len, vocab_size = logits.shape

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
    if seq_len < 2:
        raise ValueError(
            "seq_len must be >= 2 for internal causal shifting, "
            f"got seq_len={seq_len}."
        )
    if labels.shape[0] != batch_size or labels.shape[1] != seq_len:
        raise ValueError(
            "labels shape must match logits batch and sequence dimensions. "
            f"Expected {(batch_size, seq_len)}, got {tuple(labels.shape)}."
        )


def shift_logits_and_labels(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Apply internal causal LM shift.

    The token at position t predicts the label at position t+1.

    Returns:
        shift_logits: [batch_size, seq_len - 1, vocab_size]
        shift_labels: [batch_size, seq_len - 1]
    """

    validate_causal_lm_shapes(logits, labels)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return shift_logits, shift_labels


def causal_cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
    reduction: LossReduction = "mean",
) -> Tensor:
    """Compute causal LM cross-entropy with internal shifting.

    Args:
        logits: Tensor with shape [batch_size, seq_len, vocab_size].
        labels: Tensor with shape [batch_size, seq_len].
        ignore_index: Label value ignored by cross-entropy.
        reduction: One of "mean", "sum", or "none".

    Returns:
        Scalar loss for "mean" or "sum"; vector loss for "none".
    """

    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            "reduction must be one of {'mean', 'sum', 'none'}, "
            f"got {reduction!r}."
        )

    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    vocab_size = shift_logits.shape[-1]

    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )


class CausalCrossEntropyLoss(torch.nn.Module):
    """nn.Module wrapper for internal-shift causal cross-entropy."""

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        reduction: LossReduction = "mean",
    ) -> None:
        super().__init__()
        self.config = CausalCrossEntropyConfig(
            ignore_index=ignore_index,
            label_shift="internal",
            reduction=reduction,
        )

    @property
    def ignore_index(self) -> int:
        return self.config.ignore_index

    @property
    def reduction(self) -> LossReduction:
        return self.config.reduction

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return causal_cross_entropy_loss(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

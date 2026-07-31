"""Training, validation, test, and inference quality metrics.

Mirrors the internal causal-shift convention used by
:mod:`training.losses` (token at position t predicts the label at t+1) so
metrics computed here are directly comparable to the reported loss.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from sequential_segmented_llm_training_inference.training.losses import (
    shift_logits_and_labels,
)


def perplexity(loss: float) -> float | None:
    """Convert a mean cross-entropy loss to perplexity.

    Returns None if the loss is large enough that exp() overflows, since
    perplexity is meaningless (effectively infinite) at that point.
    """

    try:
        return math.exp(loss)
    except OverflowError:
        return None


@torch.no_grad()
def token_accuracy(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> float:
    """Return the fraction of non-ignored next-token predictions that are exact matches.

    Args:
        logits: [batch_size, seq_len, vocab_size].
        labels: [batch_size, seq_len].
        ignore_index: label value excluded from both numerator and denominator.
    """

    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    predictions = shift_logits.argmax(dim=-1)
    mask = shift_labels != ignore_index
    num_valid = int(mask.sum().item())
    if num_valid == 0:
        return float("nan")
    num_correct = int(((predictions == shift_labels) & mask).sum().item())
    return num_correct / num_valid


@torch.no_grad()
def top_k_token_accuracy(
    logits: Tensor,
    labels: Tensor,
    *,
    k: int = 5,
    ignore_index: int = -100,
) -> float:
    """Return the fraction of non-ignored next tokens within the top-k logits."""

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    vocab_size = shift_logits.shape[-1]
    k = min(k, vocab_size)
    top_k = shift_logits.topk(k, dim=-1).indices
    mask = shift_labels != ignore_index
    num_valid = int(mask.sum().item())
    if num_valid == 0:
        return float("nan")
    hit = (top_k == shift_labels.unsqueeze(-1)).any(dim=-1)
    num_correct = int((hit & mask).sum().item())
    return num_correct / num_valid

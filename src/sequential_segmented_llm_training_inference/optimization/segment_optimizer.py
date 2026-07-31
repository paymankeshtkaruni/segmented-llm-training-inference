"""Base utilities for segment-wise optimizers.

Phase 14 scope:
- Provide stable optimizer-state handling for loaded/unloaded segments.
- Optimizer state is keyed by logical SegmentParameterKey values, not by
  temporary Python parameter object identity.
- Support validation of update style vs. gradient accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.segments.segment_ids import (
    SegmentId,
    SegmentParameterKey,
)


UpdateStyle = Literal["after_full_backward", "immediate_segment_update"]
OptimizerPolicy = Literal["stateless_sgd", "segmentwise_sgd_momentum", "segmentwise_adamw"]


@dataclass(frozen=True, slots=True)
class SegmentUpdateResult:
    """Summary of one segment optimizer update."""

    segment_id: SegmentId
    optimizer_type: str
    updated_parameter_names: tuple[str, ...]

    @property
    def num_updated_parameters(self) -> int:
        return len(self.updated_parameter_names)


class SegmentOptimizer(Protocol):
    """Protocol implemented by segment-wise optimizers."""

    def step_module(
        self,
        *,
        segment_id: SegmentId,
        module: nn.Module,
        gradients: dict[str, Tensor],
        strict: bool = True,
    ) -> SegmentUpdateResult:
        """Apply gradients to a loaded segment module."""

    def state_dict(self) -> dict[str, object]:
        """Return YAML/checkpoint-safe optimizer metadata plus tensor state."""

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore optimizer state."""


def validate_update_style_and_accumulation(
    *,
    update_style: UpdateStyle,
    gradient_accumulation_steps: int,
    delayed_immediate_updates: bool = False,
) -> None:
    """Validate update-style compatibility with gradient accumulation.

    Immediate segment update is not compatible with normal gradient accumulation
    because parameters would change before all microbatch gradients are known.
    It is only allowed if the caller explicitly delays immediate updates until
    the accumulation boundary.
    """

    if update_style not in {"after_full_backward", "immediate_segment_update"}:
        raise ValueError(
            "update_style must be 'after_full_backward' or "
            f"'immediate_segment_update', got {update_style!r}."
        )
    if not isinstance(gradient_accumulation_steps, int):
        raise TypeError("gradient_accumulation_steps must be an int.")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if (
        update_style == "immediate_segment_update"
        and gradient_accumulation_steps > 1
        and not delayed_immediate_updates
    ):
        raise ValueError(
            "immediate_segment_update is not compatible with "
            "gradient_accumulation_steps > 1 unless updates are internally delayed."
        )


def segment_parameter_key(segment_id: SegmentId, parameter_name: str) -> SegmentParameterKey:
    """Return the logical optimizer-state key for one segment parameter."""

    return SegmentParameterKey.from_segment_id(segment_id, parameter_name)


def named_trainable_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    """Return trainable named parameters for a loaded segment."""

    return {name: param for name, param in module.named_parameters() if param.requires_grad}


def validate_gradient_mapping(
    *,
    segment_id: SegmentId,
    module: nn.Module,
    gradients: dict[str, Tensor],
    strict: bool,
) -> dict[str, nn.Parameter]:
    """Validate that gradient names/shapes match module parameters."""

    if not isinstance(segment_id, SegmentId):
        raise TypeError("segment_id must be a SegmentId.")
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module.")
    if not isinstance(gradients, dict):
        raise TypeError("gradients must be a dictionary mapping parameter name to Tensor.")

    parameters = named_trainable_parameters(module)
    if strict:
        missing = sorted(set(parameters) - set(gradients))
        unexpected = sorted(set(gradients) - set(parameters))
        if missing:
            raise KeyError(f"Missing gradients for {segment_id.to_key()}: {missing}")
        if unexpected:
            raise KeyError(f"Unexpected gradients for {segment_id.to_key()}: {unexpected}")

    for name, gradient in gradients.items():
        if name not in parameters:
            if strict:
                raise KeyError(f"Unexpected gradient {name!r} for {segment_id.to_key()}.")
            continue
        if not isinstance(gradient, Tensor):
            raise TypeError(f"Gradient for parameter {name!r} must be a torch.Tensor.")
        if tuple(gradient.shape) != tuple(parameters[name].shape):
            raise ValueError(
                f"Gradient shape mismatch for {segment_id.to_key()}.{name}: "
                f"expected {tuple(parameters[name].shape)}, got {tuple(gradient.shape)}."
            )

    return parameters


def clone_gradient_to_cpu(gradient: Tensor) -> Tensor:
    """Detach and clone a gradient to CPU for persistent storage."""

    return gradient.detach().cpu().clone()


def clone_state_tensor_to_cpu(tensor: Tensor) -> Tensor:
    """Detach and clone optimizer state tensor to CPU."""

    return tensor.detach().cpu().clone()

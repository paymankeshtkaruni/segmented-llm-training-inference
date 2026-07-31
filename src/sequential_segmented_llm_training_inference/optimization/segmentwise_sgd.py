"""Segment-wise SGD and SGD with momentum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from sequential_segmented_llm_training_inference.segments.segment_ids import (
    SegmentId,
    SegmentParameterKey,
)
from sequential_segmented_llm_training_inference.optimization.segment_optimizer import (
    SegmentUpdateResult,
    clone_state_tensor_to_cpu,
    segment_parameter_key,
    validate_gradient_mapping,
)
from sequential_segmented_llm_training_inference.optimization.optimizer_state_store import (
    InMemoryOptimizerStateStore,
    OptimizerStateStore,
    SegmentOptState,
)


@dataclass(slots=True)
class SegmentwiseSGD:
    """SGD optimizer whose state is keyed by logical segment parameter keys.

    The momentum buffers (when ``momentum > 0``) live in a pluggable
    :class:`OptimizerStateStore` keyed by SegmentId. The default in-memory store
    reproduces the previous behaviour bit-for-bit; the disk store keeps only one
    segment resident at a time. Stateless SGD (``momentum == 0``) stores nothing.
    """

    lr: float
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    state_store: OptimizerStateStore | None = None

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be > 0.")
        if self.momentum < 0:
            raise ValueError("momentum must be >= 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.dampening < 0:
            raise ValueError("dampening must be >= 0.")
        if self.nesterov and self.momentum <= 0:
            raise ValueError("nesterov requires momentum > 0.")
        if self.state_store is None:
            self.state_store = InMemoryOptimizerStateStore()

    @property
    def optimizer_type(self) -> str:
        return "segmentwise_sgd_momentum" if self.momentum > 0 else "stateless_sgd"

    @property
    def state(self) -> dict[SegmentParameterKey, dict[str, Tensor]]:
        """Flat view keyed by SegmentParameterKey.

        Provided for backward compatibility with code/tests that index optimizer
        state by parameter key. Built live from the store on each access.
        """

        flat: dict[SegmentParameterKey, dict[str, Tensor]] = {}
        for segment_id in self.state_store.segment_ids():
            seg_state = self.state_store.get(segment_id)
            for name, pstate in seg_state.items():
                key = segment_parameter_key(segment_id, name)
                flat[key] = pstate  # type: ignore[assignment]
        return flat

    def step_module(
        self,
        *,
        segment_id: SegmentId,
        module: nn.Module,
        gradients: dict[str, Tensor],
        strict: bool = True,
    ) -> SegmentUpdateResult:
        parameters = validate_gradient_mapping(
            segment_id=segment_id, module=module, gradients=gradients, strict=strict
        )
        updated: list[str] = []

        seg_state: SegmentOptState = (
            self.state_store.get(segment_id) if self.momentum != 0 else {}
        )

        with torch.no_grad():
            for name, parameter in parameters.items():
                if name not in gradients:
                    continue
                grad = gradients[name].detach().to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if self.weight_decay != 0:
                    grad = grad.add(parameter, alpha=self.weight_decay)

                if self.momentum != 0:
                    pstate = seg_state.setdefault(name, {})
                    if "momentum_buffer" not in pstate:
                        buffer = grad.detach().clone()
                    else:
                        buffer = pstate["momentum_buffer"].to(  # type: ignore[union-attr]
                            device=parameter.device, dtype=parameter.dtype
                        )
                        buffer.mul_(self.momentum).add_(grad, alpha=1.0 - self.dampening)
                    pstate["momentum_buffer"] = clone_state_tensor_to_cpu(buffer)
                    update = grad.add(buffer, alpha=self.momentum) if self.nesterov else buffer
                else:
                    update = grad

                parameter.add_(update, alpha=-self.lr)
                updated.append(name)

        if self.momentum != 0:
            self.state_store.put(segment_id, seg_state)

        return SegmentUpdateResult(
            segment_id=segment_id,
            optimizer_type=self.optimizer_type,
            updated_parameter_names=tuple(updated),
        )

    def state_dict(self) -> dict[str, Any]:
        flat_state: dict[str, Any] = {}
        for segment_id in self.state_store.segment_ids():
            seg_state = self.state_store.get(segment_id)
            for name, value_dict in seg_state.items():
                key = segment_parameter_key(segment_id, name)
                flat_state[key.to_key()] = {
                    state_name: value.clone()
                    for state_name, value in value_dict.items()
                }
        return {
            "type": self.optimizer_type,
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "dampening": self.dampening,
            "nesterov": self.nesterov,
            "state": flat_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lr = float(state_dict["lr"])
        self.momentum = float(state_dict.get("momentum", 0.0))
        self.weight_decay = float(state_dict.get("weight_decay", 0.0))
        self.dampening = float(state_dict.get("dampening", 0.0))
        self.nesterov = bool(state_dict.get("nesterov", False))
        raw_state = state_dict.get("state", {})
        if not isinstance(raw_state, dict):
            raise TypeError("state must be a dictionary.")

        self.state_store.clear()
        grouped: dict[SegmentId, SegmentOptState] = {}
        for key, value_dict in raw_state.items():
            parameter_key = SegmentParameterKey.from_key(key)
            segment_id = parameter_key.segment_id_object
            grouped.setdefault(segment_id, {})[parameter_key.parameter_name] = {
                name: tensor.detach().cpu().clone()
                for name, tensor in value_dict.items()
            }
        for segment_id, seg_state in grouped.items():
            self.state_store.put(segment_id, seg_state)

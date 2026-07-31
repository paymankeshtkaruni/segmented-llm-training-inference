"""Segment-wise AdamW with logical parameter state keys."""

from __future__ import annotations

import math
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
class SegmentwiseAdamW:
    """AdamW optimizer whose state survives segment unload/load cycles.

    State is stored by SegmentParameterKey, not by Python Parameter object id.
    State tensors are kept on CPU and moved to the active segment device only for
    the update.

    The moment tensors live in a pluggable :class:`OptimizerStateStore` keyed by
    SegmentId (one entry per segment holding all of its parameters' state). The
    default in-memory store reproduces the previous behaviour bit-for-bit; the
    disk store keeps only one segment resident at a time.
    """

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.01
    state_store: OptimizerStateStore | None = None

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be > 0.")
        beta1, beta2 = self.betas
        if not (0 <= beta1 < 1):
            raise ValueError("beta1 must satisfy 0 <= beta1 < 1.")
        if not (0 <= beta2 < 1):
            raise ValueError("beta2 must satisfy 0 <= beta2 < 1.")
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.state_store is None:
            self.state_store = InMemoryOptimizerStateStore()

    @property
    def optimizer_type(self) -> str:
        return "segmentwise_adamw"

    @property
    def state(self) -> dict[SegmentParameterKey, dict[str, Tensor | int]]:
        """Flat view keyed by SegmentParameterKey.

        Provided for backward compatibility with code/tests that index optimizer
        state by parameter key. Built live from the store on each access.
        """

        flat: dict[SegmentParameterKey, dict[str, Tensor | int]] = {}
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
        beta1, beta2 = self.betas
        updated: list[str] = []

        seg_state: SegmentOptState = self.state_store.get(segment_id)

        with torch.no_grad():
            for name, parameter in parameters.items():
                if name not in gradients:
                    continue

                grad = gradients[name].detach().to(
                    device=parameter.device, dtype=parameter.dtype
                )
                pstate = seg_state.get(name)
                if pstate is None:
                    pstate = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(parameter, memory_format=torch.preserve_format).detach().cpu(),
                        "exp_avg_sq": torch.zeros_like(parameter, memory_format=torch.preserve_format).detach().cpu(),
                    }

                step = int(pstate["step"]) + 1
                exp_avg = pstate["exp_avg"].to(device=parameter.device, dtype=parameter.dtype)  # type: ignore[union-attr]
                exp_avg_sq = pstate["exp_avg_sq"].to(device=parameter.device, dtype=parameter.dtype)  # type: ignore[union-attr]

                if self.weight_decay != 0:
                    parameter.mul_(1.0 - self.lr * self.weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = self.lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
                parameter.addcdiv_(exp_avg, denom, value=-step_size)

                pstate["step"] = step
                pstate["exp_avg"] = clone_state_tensor_to_cpu(exp_avg)
                pstate["exp_avg_sq"] = clone_state_tensor_to_cpu(exp_avg_sq)
                seg_state[name] = pstate
                updated.append(name)

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
            for name, value in seg_state.items():
                key = segment_parameter_key(segment_id, name)
                flat_state[key.to_key()] = {
                    "step": int(value["step"]),
                    "exp_avg": value["exp_avg"].clone(),  # type: ignore[union-attr]
                    "exp_avg_sq": value["exp_avg_sq"].clone(),  # type: ignore[union-attr]
                }
        return {
            "type": self.optimizer_type,
            "lr": self.lr,
            "betas": tuple(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": flat_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lr = float(state_dict["lr"])
        self.betas = tuple(state_dict.get("betas", (0.9, 0.999)))  # type: ignore[assignment]
        self.eps = float(state_dict.get("eps", 1.0e-8))
        self.weight_decay = float(state_dict.get("weight_decay", 0.01))
        raw_state = state_dict.get("state", {})
        if not isinstance(raw_state, dict):
            raise TypeError("state must be a dictionary.")

        self.state_store.clear()
        grouped: dict[SegmentId, SegmentOptState] = {}
        for key, value in raw_state.items():
            parameter_key = SegmentParameterKey.from_key(key)
            segment_id = parameter_key.segment_id_object
            grouped.setdefault(segment_id, {})[parameter_key.parameter_name] = {
                "step": int(value["step"]),
                "exp_avg": value["exp_avg"].detach().cpu().clone(),
                "exp_avg_sq": value["exp_avg_sq"].detach().cpu().clone(),
            }
        for segment_id, seg_state in grouped.items():
            self.state_store.put(segment_id, seg_state)

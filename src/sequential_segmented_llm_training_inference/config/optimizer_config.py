"""Optimizer configuration for segmented training."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

OptimizerType = Literal["stateless_sgd", "segmentwise_sgd_momentum", "segmentwise_adamw"]


def normalize_optimizer_type(value: str) -> OptimizerType:
    """Return the canonical optimizer policy name.

    ``segmentwise_sgd`` is accepted as a backward-compatible CLI alias for
    ``segmentwise_sgd_momentum``.
    """

    aliases = {
        "segmentwise_sgd": "segmentwise_sgd_momentum",
        "sgd_momentum": "segmentwise_sgd_momentum",
    }
    normalized = aliases.get(str(value), str(value))
    if normalized not in {"stateless_sgd", "segmentwise_sgd_momentum", "segmentwise_adamw"}:
        raise ValueError(f"Unsupported optimizer type: {value!r}.")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer policy configuration."""

    type: OptimizerType = "segmentwise_adamw"
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    state_key_policy: str = "layer_id.segment_type.segment_id.parameter_name"

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", normalize_optimizer_type(self.type))
        if isinstance(self.betas, list):
            object.__setattr__(self, "betas", tuple(self.betas))
        self.validate()

    def validate(self) -> None:
        if self.type not in {"stateless_sgd", "segmentwise_sgd_momentum", "segmentwise_adamw"}:
            raise ValueError("Unsupported optimizer type.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if len(self.betas) != 2:
            raise ValueError("betas must contain two values.")
        beta1, beta2 = self.betas
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("betas must be in [0, 1).")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if self.state_key_policy != "layer_id.segment_type.segment_id.parameter_name":
            raise ValueError("state_key_policy must be layer_id.segment_type.segment_id.parameter_name.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["betas"] = list(self.betas)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerConfig":
        return cls(**data)

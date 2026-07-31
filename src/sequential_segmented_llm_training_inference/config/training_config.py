"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

UpdateStyle = Literal["after_full_backward", "immediate_segment_update"]
PrecisionName = Literal["fp32", "bf16", "fp16"]
BackwardMode = Literal["recomputation"]


@dataclass(frozen=True)
class TrainingConfig:
    """General training-loop configuration."""

    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    update_style: UpdateStyle = "after_full_backward"
    precision: PrecisionName = "fp32"
    gradient_clip_norm: float | None = None
    backward_mode: BackwardMode = "recomputation"
    gradient_clipping_scope: str = "true_global"
    gradient_store: Literal["in_memory", "disk"] = "in_memory"
    gradient_store_dir: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.update_style not in {"after_full_backward", "immediate_segment_update"}:
            raise ValueError("update_style must be after_full_backward or immediate_segment_update.")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        if self.backward_mode != "recomputation":
            raise ValueError("Only backward_mode='recomputation' is supported.")
        if self.gradient_clipping_scope not in {"true_global", "disabled"}:
            raise ValueError("gradient_clipping_scope must be true_global or disabled.")
        if self.gradient_clip_norm is None and self.gradient_clipping_scope != "disabled":
            object.__setattr__(self, "gradient_clipping_scope", "disabled")
        if self.gradient_clip_norm is not None and self.update_style == "immediate_segment_update":
            raise ValueError(
                "Exact true-global gradient clipping is not compatible with "
                "immediate_segment_update. Use after_full_backward or disable clipping."
            )
        if self.gradient_accumulation_steps > 1 and self.update_style == "immediate_segment_update":
            raise ValueError(
                "immediate_segment_update is incompatible with gradient_accumulation_steps > 1 "
                "unless updates are internally delayed. Use after_full_backward."
            )
        if self.gradient_store not in {"in_memory", "disk"}:
            raise ValueError("gradient_store must be 'in_memory' or 'disk'.")
        if self.gradient_store == "disk" and not self.gradient_store_dir:
            raise ValueError("gradient_store_dir is required when gradient_store='disk'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(**data)

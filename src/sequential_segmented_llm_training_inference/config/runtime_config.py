"""Runtime execution, record, and recomputation configuration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

ExecutionMode = Literal["strict_single_segment"]
RuntimeRecordPolicy = Literal["full_segment_records", "minimal_segment_records"]
PrecisionName = Literal["fp32", "bf16", "fp16"]


@dataclass(frozen=True)
class ExecutionConfig:
    """Strict single-segment execution configuration."""

    mode: ExecutionMode = "strict_single_segment"
    max_active_segments: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode != "strict_single_segment":
            raise ValueError("Only execution mode 'strict_single_segment' is supported.")
        if self.max_active_segments != 1:
            raise ValueError("max_active_segments must be exactly 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionConfig":
        return cls(**data)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime segment-record and recomputation settings."""

    policy: RuntimeRecordPolicy = "full_segment_records"
    store_rng_state: bool = True
    restore_rng_state_during_backward: bool = True
    recomputation_check: bool = False
    recomputation_atol: float = 1.0e-5
    recomputation_rtol: float = 1.0e-4
    detach_record_tensors: bool = True
    clone_record_tensors: bool = True
    record_tensors_to_cpu: bool = True
    record_backend: Literal["in_memory", "disk"] = "in_memory"
    record_disk_dir: str | None = None
    evict_layers_during_backward: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.policy not in {"full_segment_records", "minimal_segment_records"}:
            raise ValueError("runtime record policy must be full_segment_records or minimal_segment_records.")
        if self.recomputation_atol < 0:
            raise ValueError("recomputation_atol must be non-negative.")
        if self.recomputation_rtol < 0:
            raise ValueError("recomputation_rtol must be non-negative.")
        if self.restore_rng_state_during_backward and not self.store_rng_state:
            raise ValueError("restore_rng_state_during_backward requires store_rng_state=true.")
        if not self.detach_record_tensors:
            raise ValueError(
                "True recomputation training requires detach_record_tensors=true; "
                "otherwise the original forward graph may stay alive."
            )
        if self.policy == "minimal_segment_records" and self.recomputation_check:
            raise ValueError(
                "minimal_segment_records cannot perform stored-output recomputation checks. "
                "Use full_segment_records or disable recomputation_check."
            )
        if self.record_backend not in {"in_memory", "disk"}:
            raise ValueError("record_backend must be 'in_memory' or 'disk'.")
        if self.record_backend == "disk" and not self.record_disk_dir:
            raise ValueError("record_disk_dir is required when record_backend='disk'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        return cls(**data)

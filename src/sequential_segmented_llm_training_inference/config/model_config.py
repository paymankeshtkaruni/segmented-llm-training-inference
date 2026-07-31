"""Model architecture configuration for segmented LLM training."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Architecture-level configuration for a GPT-style decoder."""

    architecture: str = "gpt_decoder"
    vocab_size: int = 50_257
    max_seq_len: int = 512
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.architecture != "gpt_decoder":
            raise ValueError("Only architecture='gpt_decoder' is supported in Phase 1.")
        _require_positive("vocab_size", self.vocab_size)
        _require_positive("max_seq_len", self.max_seq_len)
        _require_positive("n_layers", self.n_layers)
        _require_positive("d_model", self.d_model)
        _require_positive("n_heads", self.n_heads)
        _require_positive("d_ff", self.d_ff)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in the interval [0, 1).")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**data)


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

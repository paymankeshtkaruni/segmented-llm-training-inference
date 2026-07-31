"""Token and position embeddings for segmented LLM execution.

These components are trainable non-segment parameters. They remain outside the
attention/MLP segment store, but they participate in training, checkpointing,
validation, testing, inference, and full-model export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for token and position embeddings."""

    vocab_size: int
    d_model: int
    max_seq_len: int
    dropout: float = 0.0
    pad_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}.")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}.")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.pad_token_id is not None:
            if self.pad_token_id < 0 or self.pad_token_id >= self.vocab_size:
                raise ValueError(
                    "pad_token_id must be in [0, vocab_size), got "
                    f"pad_token_id={self.pad_token_id}, vocab_size={self.vocab_size}."
                )


class TokenPositionEmbeddings(nn.Module):
    """Token + position embeddings used before segmented transformer layers.

    Shape contract:
        input_ids: ``[batch_size, seq_len]``
        output: ``[batch_size, seq_len, d_model]``
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_seq_len,
            embedding_dim=config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def max_seq_len(self) -> int:
        return self.config.max_seq_len

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.config.pad_token_id].zero_()

    def forward(self, input_ids: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch_size, seq_len], got {tuple(input_ids.shape)}."
            )
        if not torch.is_floating_point(input_ids) and input_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}.")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.config.max_seq_len}."
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        else:
            if position_ids.shape != input_ids.shape:
                raise ValueError(
                    "position_ids must have the same shape as input_ids, got "
                    f"position_ids={tuple(position_ids.shape)}, input_ids={tuple(input_ids.shape)}."
                )
            position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)

        token_embeddings = self.token_embedding(input_ids.to(dtype=torch.long))
        position_embeddings = self.position_embedding(position_ids)
        return self.dropout(token_embeddings + position_embeddings)

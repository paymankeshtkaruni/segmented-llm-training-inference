"""Final normalization and language-modeling head.

This is a trainable non-segment component. It is used after all segmented
transformer layers have been executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class OutputHeadConfig:
    """Configuration for final normalization and LM head."""

    d_model: int
    vocab_size: int
    layer_norm_eps: float = 1.0e-5
    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}.")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}.")
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be > 0, got {self.layer_norm_eps}."
            )


class FinalNormLMHead(nn.Module):
    """Final LayerNorm followed by an LM projection."""

    def __init__(
        self,
        config: OutputHeadConfig,
        tied_token_embedding: Optional[nn.Embedding] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        if tied_token_embedding is not None:
            self.tie_weights(tied_token_embedding)
        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def tie_weights(self, token_embedding: nn.Embedding) -> None:
        """Tie LM head weights to a token embedding matrix."""

        expected = (self.config.vocab_size, self.config.d_model)
        actual = tuple(token_embedding.weight.shape)
        if actual != expected:
            raise ValueError(
                f"token_embedding weight shape must be {expected}, got {actual}."
            )
        self.lm_head.weight = token_embedding.weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, seq_len, d_model], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states last dimension must be d_model={self.config.d_model}, "
                f"got {hidden_states.shape[-1]}."
            )
        normalized = self.final_norm(hidden_states)
        return self.lm_head(normalized)

    @torch.no_grad()
    def predict_next_token(self, hidden_states: Tensor) -> Tensor:
        """Return argmax token IDs from the last sequence position."""

        logits = self.forward(hidden_states)
        return logits[:, -1, :].argmax(dim=-1)

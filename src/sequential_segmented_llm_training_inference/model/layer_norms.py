"""Trainable non-segment components used inside segmented transformer layers.

Attention/MLP segments are the only true loadable segments. The components here
remain outside the segment store but are still trainable and checkpointed:

- attention LayerNorm per layer;
- MLP LayerNorm per layer;
- attention output projection per layer;
- optional shared MLP output bias per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class LayerComponentConfig:
    """Configuration for per-layer non-segment components."""

    n_layers: int
    d_model: int
    layer_norm_eps: float = 1.0e-5
    use_mlp_shared_output_bias: bool = True

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}.")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}.")
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be > 0, got {self.layer_norm_eps}."
            )


def _validate_layer_id(layer_id: int, n_layers: int) -> None:
    if not isinstance(layer_id, int):
        raise TypeError(f"layer_id must be an int, got {type(layer_id).__name__}.")
    if layer_id < 0 or layer_id >= n_layers:
        raise ValueError(f"layer_id must be in [0, {n_layers - 1}], got {layer_id}.")


def _validate_hidden_states(hidden_states: Tensor, d_model: int, name: str) -> None:
    if hidden_states.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, seq_len, d_model].")
    if hidden_states.shape[-1] != d_model:
        raise ValueError(
            f"{name} last dimension must be d_model={d_model}, "
            f"got {hidden_states.shape[-1]}."
        )


class SegmentedLayerNorms(nn.Module):
    """Attention and MLP LayerNorms for each segmented transformer layer."""

    def __init__(self, config: LayerComponentConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model, eps=config.layer_norm_eps) for _ in range(config.n_layers)]
        )
        self.mlp_layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model, eps=config.layer_norm_eps) for _ in range(config.n_layers)]
        )

    def attention(self, layer_id: int, hidden_states: Tensor) -> Tensor:
        _validate_layer_id(layer_id, self.config.n_layers)
        _validate_hidden_states(hidden_states, self.config.d_model, "hidden_states")
        return self.attention_layer_norms[layer_id](hidden_states)

    def mlp(self, layer_id: int, hidden_states: Tensor) -> Tensor:
        _validate_layer_id(layer_id, self.config.n_layers)
        _validate_hidden_states(hidden_states, self.config.d_model, "hidden_states")
        return self.mlp_layer_norms[layer_id](hidden_states)


class AttentionOutputProjections(nn.Module):
    """Per-layer attention output projection applied after head-segment concat."""

    def __init__(self, config: LayerComponentConfig) -> None:
        super().__init__()
        self.config = config
        self.projections = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model, bias=True) for _ in range(config.n_layers)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, layer_id: int, attention_concat: Tensor) -> Tensor:
        _validate_layer_id(layer_id, self.config.n_layers)
        _validate_hidden_states(attention_concat, self.config.d_model, "attention_concat")
        return self.projections[layer_id](attention_concat)


class MLPSharedOutputBiases(nn.Module):
    """Optional per-layer shared MLP output bias added once after chunk summation."""

    def __init__(self, config: LayerComponentConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_mlp_shared_output_bias:
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(config.d_model)) for _ in range(config.n_layers)]
            )
        else:
            self.biases = nn.ParameterList()

    @property
    def enabled(self) -> bool:
        return self.config.use_mlp_shared_output_bias

    def forward(self, layer_id: int, mlp_sum: Tensor) -> Tensor:
        _validate_layer_id(layer_id, self.config.n_layers)
        _validate_hidden_states(mlp_sum, self.config.d_model, "mlp_sum")
        if not self.enabled:
            return mlp_sum
        return mlp_sum + self.biases[layer_id].view(1, 1, -1)

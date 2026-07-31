"""Segmented autoregressive inference/generation.

Phase 25 scope:
- Provide inference/generation utilities that call the segmented forward path.
- Do not implement a separate full-model inference path.
- Do not implement KV-cache optimization yet; generation recomputes the prefix via
  segmented forward at each step.
- Strict one-segment-at-a-time execution remains the responsibility of the
  forward engine and segment loader used by the supplied forward callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from torch import Tensor


class SegmentedForwardCallable(Protocol):
    """Callable protocol for segmented forward execution."""

    def __call__(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        store_runtime: bool = False,
        **kwargs: Any,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Configuration for autoregressive segmented generation."""

    max_new_tokens: int = 20
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0.")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be None or > 0.")
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be None or in the interval (0, 1].")
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be None or non-negative.")
        if self.pad_token_id is not None and self.pad_token_id < 0:
            raise ValueError("pad_token_id must be None or non-negative.")


def extract_logits(forward_output: Any) -> Tensor:
    """Extract logits from common segmented-forward output formats."""

    if isinstance(forward_output, Tensor):
        logits = forward_output
    elif isinstance(forward_output, dict) and "logits" in forward_output:
        logits = forward_output["logits"]
    elif hasattr(forward_output, "logits"):
        logits = forward_output.logits
    else:
        raise TypeError(
            "forward output must be a Tensor, a dict containing 'logits', "
            "or an object with a .logits attribute."
        )

    if not isinstance(logits, Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}.")
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch_size, seq_len, vocab_size], "
            f"got {tuple(logits.shape)}."
        )
    return logits


def apply_top_k_top_p_filtering(
    logits: Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    """Apply top-k and/or nucleus filtering to next-token logits.

    Args:
        logits: [batch_size, vocab_size] next-token logits.
    """

    if logits.ndim != 2:
        raise ValueError(
            "next-token logits must have shape [batch_size, vocab_size], "
            f"got {tuple(logits.shape)}."
        )

    filtered = logits.clone()

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0 when provided.")
        k = min(top_k, filtered.shape[-1])
        kth_values = torch.topk(filtered, k=k, dim=-1).values[:, -1].unsqueeze(-1)
        filtered = filtered.masked_fill(filtered < kth_values, float("-inf"))

    if top_p is not None:
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the interval (0, 1].")
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False
        remove_mask = torch.zeros_like(sorted_remove)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        filtered = filtered.masked_fill(remove_mask, float("-inf"))

    return filtered


def select_next_token(
    next_token_logits: Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select the next token using greedy decoding or sampling."""

    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must have shape [batch_size, vocab_size], "
            f"got {tuple(next_token_logits.shape)}."
        )

    logits = next_token_logits / config.temperature
    logits = apply_top_k_top_p_filtering(
        logits,
        top_k=config.top_k,
        top_p=config.top_p,
    )

    if not config.do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


class SegmentedAutoregressiveGenerator:
    """Autoregressive generation wrapper around a segmented forward callable.

    The supplied callable must run the segmented forward path. This class does not
    bypass segmentation and does not reconstruct a full model.
    """

    def __init__(
        self,
        forward_callable: SegmentedForwardCallable | Callable[..., Any],
        config: GenerationConfig | None = None,
    ) -> None:
        self.forward_callable = forward_callable
        self.config = config or GenerationConfig()

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        config: GenerationConfig | None = None,
        generator: torch.Generator | None = None,
        **forward_kwargs: Any,
    ) -> Tensor:
        """Generate tokens using repeated segmented forward execution."""

        cfg = config or self.config
        self._validate_inputs(input_ids, attention_mask)

        generated = input_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)

        for _ in range(cfg.max_new_tokens):
            forward_output = self.forward_callable(
                input_ids=generated,
                attention_mask=current_attention_mask,
                labels=None,
                store_runtime=False,
                **forward_kwargs,
            )
            logits = extract_logits(forward_output)
            next_logits = logits[:, -1, :]
            next_token = select_next_token(next_logits, cfg, generator=generator)

            if cfg.eos_token_id is not None:
                eos_tensor = torch.full_like(next_token, cfg.eos_token_id)
                if cfg.pad_token_id is not None:
                    pad_tensor = torch.full_like(next_token, cfg.pad_token_id)
                    next_token = torch.where(finished.unsqueeze(-1), pad_tensor, next_token)
                finished = finished | (next_token.squeeze(-1) == eos_tensor.squeeze(-1))

            generated = torch.cat([generated, next_token], dim=1)

            if current_attention_mask is not None:
                new_mask = torch.ones(
                    (current_attention_mask.shape[0], 1),
                    dtype=current_attention_mask.dtype,
                    device=current_attention_mask.device,
                )
                if cfg.eos_token_id is not None and cfg.pad_token_id is not None:
                    new_mask = torch.where(finished.unsqueeze(-1), torch.zeros_like(new_mask), new_mask)
                current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)

            if cfg.eos_token_id is not None and bool(finished.all()):
                break

        return generated

    @staticmethod
    def _validate_inputs(input_ids: Tensor, attention_mask: Tensor | None) -> None:
        if not isinstance(input_ids, Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}.")
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, seq_len], "
                f"got {tuple(input_ids.shape)}."
            )
        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor):
                raise TypeError(
                    "attention_mask must be a torch.Tensor when provided, "
                    f"got {type(attention_mask).__name__}."
                )
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask shape must match input_ids shape, "
                    f"got {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}."
                )

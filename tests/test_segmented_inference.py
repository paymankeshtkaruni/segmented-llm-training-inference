"""Tests for Phase 25 segmented inference/generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from sequential_segmented_llm_training_inference.inference.generation import (
    GenerationConfig,
    SegmentedAutoregressiveGenerator,
    apply_top_k_top_p_filtering,
    extract_logits,
    select_next_token,
)


@dataclass
class ForwardOutput:
    logits: Tensor


class DummySegmentedForward:
    def __init__(self, vocab_size: int = 10, eos_after_calls: int | None = None) -> None:
        self.vocab_size = vocab_size
        self.eos_after_calls = eos_after_calls
        self.calls = 0
        self.attention_mask_shapes: list[tuple[int, int] | None] = []

    def __call__(self, *, input_ids: Tensor, attention_mask: Tensor | None = None, **_: object) -> ForwardOutput:
        self.calls += 1
        self.attention_mask_shapes.append(None if attention_mask is None else tuple(attention_mask.shape))
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        token = 2
        if self.eos_after_calls is not None and self.calls >= self.eos_after_calls:
            token = 9
        logits[:, -1, token] = 100.0
        return ForwardOutput(logits=logits)


def test_generation_config_validation() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(temperature=0.0)
    with pytest.raises(ValueError, match="top_k"):
        GenerationConfig(top_k=0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(top_p=1.5)


def test_extract_logits_from_tensor_dict_and_object() -> None:
    logits = torch.randn(2, 3, 5)
    assert extract_logits(logits) is logits
    assert extract_logits({"logits": logits}) is logits
    assert extract_logits(ForwardOutput(logits=logits)) is logits


def test_extract_logits_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="logits"):
        extract_logits(torch.randn(2, 5))


def test_greedy_generation_appends_argmax_tokens() -> None:
    forward = DummySegmentedForward(vocab_size=10)
    generator = SegmentedAutoregressiveGenerator(
        forward,
        GenerationConfig(max_new_tokens=3, do_sample=False),
    )
    input_ids = torch.tensor([[1, 4]])

    output = generator.generate(input_ids)

    assert output.tolist() == [[1, 4, 2, 2, 2]]
    assert forward.calls == 3


def test_generation_uses_attention_mask_and_grows_it() -> None:
    forward = DummySegmentedForward(vocab_size=10)
    generator = SegmentedAutoregressiveGenerator(
        forward,
        GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    input_ids = torch.tensor([[1, 4]])
    attention_mask = torch.tensor([[1, 1]])

    output = generator.generate(input_ids, attention_mask=attention_mask)

    assert output.shape == (1, 4)
    assert forward.attention_mask_shapes == [(1, 2), (1, 3)]


def test_generation_stops_when_all_sequences_emit_eos() -> None:
    forward = DummySegmentedForward(vocab_size=10, eos_after_calls=2)
    generator = SegmentedAutoregressiveGenerator(
        forward,
        GenerationConfig(max_new_tokens=5, do_sample=False, eos_token_id=9),
    )

    output = generator.generate(torch.tensor([[1, 4]]))

    assert output.tolist() == [[1, 4, 2, 9]]
    assert forward.calls == 2


def test_select_next_token_greedy() -> None:
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]])
    token = select_next_token(logits, GenerationConfig(max_new_tokens=1, do_sample=False))
    assert token.tolist() == [[2]]


def test_top_k_filtering_keeps_only_k_values() -> None:
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]])
    filtered = apply_top_k_top_p_filtering(logits, top_k=2)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert filtered[0, 2].item() == 3.0
    assert filtered[0, 3].item() == 2.0


def test_input_validation_rejects_wrong_input_shape() -> None:
    forward = DummySegmentedForward(vocab_size=10)
    generator = SegmentedAutoregressiveGenerator(forward)
    with pytest.raises(ValueError, match="input_ids"):
        generator.generate(torch.tensor([1, 2, 3]))


def test_input_validation_rejects_attention_mask_shape_mismatch() -> None:
    forward = DummySegmentedForward(vocab_size=10)
    generator = SegmentedAutoregressiveGenerator(forward)
    with pytest.raises(ValueError, match="attention_mask"):
        generator.generate(torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1]]))

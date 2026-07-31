"""Tests for Phase 5 non-segment model components."""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.model import (
    AttentionOutputProjections,
    EmbeddingConfig,
    FinalNormLMHead,
    LayerComponentConfig,
    MLPSharedOutputBiases,
    OutputHeadConfig,
    SegmentedLayerNorms,
    TokenPositionEmbeddings,
)


def test_token_position_embeddings_output_shape() -> None:
    module = TokenPositionEmbeddings(
        EmbeddingConfig(vocab_size=100, d_model=16, max_seq_len=32, dropout=0.0)
    )
    input_ids = torch.randint(0, 100, (3, 7))
    output = module(input_ids)
    assert output.shape == (3, 7, 16)


def test_token_position_embeddings_default_and_explicit_positions_match() -> None:
    module = TokenPositionEmbeddings(
        EmbeddingConfig(vocab_size=100, d_model=16, max_seq_len=32, dropout=0.0)
    )
    input_ids = torch.randint(0, 100, (2, 5))
    explicit = torch.arange(5).unsqueeze(0).expand(2, 5)
    assert torch.allclose(module(input_ids), module(input_ids, explicit))


def test_token_position_embeddings_rejects_too_long_sequence() -> None:
    module = TokenPositionEmbeddings(
        EmbeddingConfig(vocab_size=100, d_model=16, max_seq_len=4, dropout=0.0)
    )
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        module(torch.randint(0, 100, (2, 5)))


def test_padding_embedding_is_zeroed() -> None:
    module = TokenPositionEmbeddings(
        EmbeddingConfig(vocab_size=100, d_model=16, max_seq_len=32, pad_token_id=0)
    )
    assert torch.count_nonzero(module.token_embedding.weight[0]).item() == 0


def test_embedding_config_validation() -> None:
    with pytest.raises(ValueError):
        EmbeddingConfig(vocab_size=0, d_model=16, max_seq_len=32)
    with pytest.raises(ValueError):
        EmbeddingConfig(vocab_size=10, d_model=0, max_seq_len=32)
    with pytest.raises(ValueError):
        EmbeddingConfig(vocab_size=10, d_model=16, max_seq_len=0)
    with pytest.raises(ValueError):
        EmbeddingConfig(vocab_size=10, d_model=16, max_seq_len=32, dropout=1.0)
    with pytest.raises(ValueError):
        EmbeddingConfig(vocab_size=10, d_model=16, max_seq_len=32, pad_token_id=10)


def test_segmented_layer_norms_shapes() -> None:
    norms = SegmentedLayerNorms(LayerComponentConfig(n_layers=3, d_model=16))
    hidden = torch.randn(2, 5, 16)
    assert norms.attention(0, hidden).shape == hidden.shape
    assert norms.mlp(2, hidden).shape == hidden.shape


def test_segmented_layer_norms_reject_invalid_layer_id() -> None:
    norms = SegmentedLayerNorms(LayerComponentConfig(n_layers=2, d_model=16))
    hidden = torch.randn(2, 5, 16)
    with pytest.raises(ValueError, match="layer_id"):
        norms.attention(2, hidden)


def test_segmented_layer_norms_reject_wrong_hidden_dim() -> None:
    norms = SegmentedLayerNorms(LayerComponentConfig(n_layers=2, d_model=16))
    with pytest.raises(ValueError, match="last dimension"):
        norms.attention(0, torch.randn(2, 5, 15))


def test_attention_output_projection_shape() -> None:
    projections = AttentionOutputProjections(LayerComponentConfig(n_layers=2, d_model=16))
    attention_concat = torch.randn(2, 5, 16)
    output = projections(1, attention_concat)
    assert output.shape == (2, 5, 16)


def test_attention_output_projection_rejects_wrong_dim() -> None:
    projections = AttentionOutputProjections(LayerComponentConfig(n_layers=2, d_model=16))
    with pytest.raises(ValueError, match="last dimension"):
        projections(0, torch.randn(2, 5, 15))


def test_mlp_shared_output_bias_adds_once() -> None:
    biases = MLPSharedOutputBiases(LayerComponentConfig(n_layers=2, d_model=4))
    with torch.no_grad():
        biases.biases[1].copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    mlp_sum = torch.zeros(2, 3, 4)
    output = biases(1, mlp_sum)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4).expand(2, 3, 4)
    assert torch.allclose(output, expected)


def test_mlp_shared_output_bias_can_be_disabled() -> None:
    biases = MLPSharedOutputBiases(
        LayerComponentConfig(n_layers=2, d_model=4, use_mlp_shared_output_bias=False)
    )
    mlp_sum = torch.randn(2, 3, 4)
    assert torch.allclose(biases(0, mlp_sum), mlp_sum)
    assert list(biases.parameters()) == []


def test_output_head_logits_shape() -> None:
    head = FinalNormLMHead(OutputHeadConfig(d_model=16, vocab_size=101))
    logits = head(torch.randn(2, 5, 16))
    assert logits.shape == (2, 5, 101)


def test_output_head_rejects_wrong_hidden_shape() -> None:
    head = FinalNormLMHead(OutputHeadConfig(d_model=16, vocab_size=101))
    with pytest.raises(ValueError, match="shape"):
        head(torch.randn(2, 16))
    with pytest.raises(ValueError, match="last dimension"):
        head(torch.randn(2, 5, 15))


def test_output_head_can_tie_weights_to_embeddings() -> None:
    embeddings = TokenPositionEmbeddings(
        EmbeddingConfig(vocab_size=101, d_model=16, max_seq_len=32)
    )
    head = FinalNormLMHead(
        OutputHeadConfig(d_model=16, vocab_size=101),
        tied_token_embedding=embeddings.token_embedding,
    )
    assert head.lm_head.weight.data_ptr() == embeddings.token_embedding.weight.data_ptr()


def test_output_head_rejects_invalid_tied_embedding_shape() -> None:
    wrong_embedding = torch.nn.Embedding(100, 16)
    with pytest.raises(ValueError, match="weight shape"):
        FinalNormLMHead(
            OutputHeadConfig(d_model=16, vocab_size=101),
            tied_token_embedding=wrong_embedding,
        )


def test_non_segment_components_have_trainable_parameters() -> None:
    embeddings = TokenPositionEmbeddings(
        EmbeddingConfig(vocab_size=100, d_model=16, max_seq_len=32)
    )
    norms = SegmentedLayerNorms(LayerComponentConfig(n_layers=2, d_model=16))
    projections = AttentionOutputProjections(LayerComponentConfig(n_layers=2, d_model=16))
    biases = MLPSharedOutputBiases(LayerComponentConfig(n_layers=2, d_model=16))
    head = FinalNormLMHead(OutputHeadConfig(d_model=16, vocab_size=100))

    total_params = sum(
        p.numel()
        for module in (embeddings, norms, projections, biases, head)
        for p in module.parameters()
    )
    assert total_params > 0

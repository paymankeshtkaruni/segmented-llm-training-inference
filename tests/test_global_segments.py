"""Tests for global segment wrappers and segment-store-mode forward engine.

These tests verify:
1. New SegmentId types (embedding, output_head, attention_output_proj)
2. EmbeddingSegmentModule, OutputHeadSegmentModule, AttentionOutputProjSegmentModule
3. SegmentedForwardEngine in use_segment_store_for_global=True mode
4. Trainer backward pass in segment-store mode (gradient flow, accumulation)
"""

from __future__ import annotations

import pytest
import torch

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import SegmentationConfig
from sequential_segmented_llm_training_inference.execution.backward_engine import SegmentedBackwardEngine
from sequential_segmented_llm_training_inference.execution.forward_engine import SegmentedForwardEngine
from sequential_segmented_llm_training_inference.execution.segment_loader import StrictSegmentLoader
from sequential_segmented_llm_training_inference.model.embeddings import EmbeddingConfig
from sequential_segmented_llm_training_inference.model.output_head import OutputHeadConfig
from sequential_segmented_llm_training_inference.optimization.segmentwise_adamw import SegmentwiseAdamW
from sequential_segmented_llm_training_inference.segments.global_segments import (
    AttentionOutputProjSegmentModule,
    EmbeddingSegmentModule,
    OutputHeadSegmentModule,
)
from sequential_segmented_llm_training_inference.segments.segment_factory import SegmentFactory
from sequential_segmented_llm_training_inference.segments.segment_ids import (
    GLOBAL_SEGMENT_TYPES,
    SegmentId,
    SegmentParameterKey,
    VALID_SEGMENT_TYPES,
)
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import CpuRamSegmentStore
from sequential_segmented_llm_training_inference.training.losses import CausalCrossEntropyLoss
from sequential_segmented_llm_training_inference.training.trainer import SegmentedTrainer


# ---------------------------------------------------------------------------
# SegmentId new types
# ---------------------------------------------------------------------------

def test_embedding_segment_id_uses_layer_minus_one() -> None:
    seg_id = SegmentId(-1, "embedding", 0)
    assert seg_id.layer_id == -1
    assert seg_id.segment_type == "embedding"
    assert seg_id.segment_id == 0


def test_output_head_segment_id_uses_layer_minus_one() -> None:
    seg_id = SegmentId(-1, "output_head", 0)
    assert seg_id.layer_id == -1
    assert seg_id.segment_type == "output_head"


def test_attention_output_proj_segment_id_uses_positive_layer() -> None:
    seg_id = SegmentId(3, "attention_output_proj", 0)
    assert seg_id.layer_id == 3


def test_global_segment_id_rejects_non_minus_one_layer_id() -> None:
    with pytest.raises(ValueError, match="layer_id must be -1"):
        SegmentId(0, "embedding", 0)
    with pytest.raises(ValueError, match="layer_id must be -1"):
        SegmentId(0, "output_head", 0)


def test_non_global_segment_id_rejects_negative_layer_id() -> None:
    with pytest.raises(ValueError):
        SegmentId(-1, "attention", 0)
    with pytest.raises(ValueError):
        SegmentId(-1, "mlp", 0)
    with pytest.raises(ValueError):
        SegmentId(-1, "attention_output_proj", 0)


def test_segment_id_to_key_and_from_key_roundtrip_for_global_types() -> None:
    for seg_type in GLOBAL_SEGMENT_TYPES:
        seg_id = SegmentId(-1, seg_type, 0)  # type: ignore[arg-type]
        key = seg_id.to_key()
        assert "layer_-1" in key
        recovered = SegmentId.from_key(key)
        assert recovered == seg_id


def test_segment_id_to_dict_roundtrip_for_new_types() -> None:
    for seg_type in VALID_SEGMENT_TYPES:
        layer_id = -1 if seg_type in GLOBAL_SEGMENT_TYPES else 2
        seg_id = SegmentId(layer_id, seg_type, 0)  # type: ignore[arg-type]
        assert SegmentId.from_dict(seg_id.to_dict()) == seg_id


def test_segment_parameter_key_works_for_global_segment_types() -> None:
    key = SegmentParameterKey(-1, "embedding", 0, "embeddings.token_embedding.weight")
    assert key.to_key().startswith("layer_-1.embedding.")
    recovered = SegmentParameterKey.from_key(key.to_key())
    assert recovered == key


# ---------------------------------------------------------------------------
# EmbeddingSegmentModule
# ---------------------------------------------------------------------------

def _make_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(vocab_size=64, d_model=16, max_seq_len=32)


def test_embedding_segment_module_forward_shape() -> None:
    cfg = _make_embedding_config()
    mod = EmbeddingSegmentModule(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = mod(input_ids)
    assert out.shape == (2, 8, cfg.d_model)


def test_embedding_segment_module_state_dict_roundtrip() -> None:
    cfg = _make_embedding_config()
    mod = EmbeddingSegmentModule(cfg)
    sd = mod.state_dict()
    assert any("token_embedding" in k for k in sd)
    assert any("position_embedding" in k for k in sd)
    mod2 = EmbeddingSegmentModule(cfg)
    mod2.load_state_dict(sd)


# ---------------------------------------------------------------------------
# OutputHeadSegmentModule
# ---------------------------------------------------------------------------

def _make_output_head_config() -> OutputHeadConfig:
    return OutputHeadConfig(d_model=16, vocab_size=64)


def test_output_head_segment_module_forward_shape() -> None:
    cfg = _make_output_head_config()
    mod = OutputHeadSegmentModule(cfg)
    hidden = torch.randn(2, 8, cfg.d_model)
    out = mod(hidden)
    assert out.shape == (2, 8, cfg.vocab_size)


def test_output_head_segment_module_state_dict_roundtrip() -> None:
    cfg = _make_output_head_config()
    mod = OutputHeadSegmentModule(cfg)
    sd = mod.state_dict()
    assert any("lm_head" in k for k in sd)
    mod2 = OutputHeadSegmentModule(cfg)
    mod2.load_state_dict(sd)


# ---------------------------------------------------------------------------
# AttentionOutputProjSegmentModule
# ---------------------------------------------------------------------------

def test_attention_output_proj_forward_shape() -> None:
    d_model = 16
    mod = AttentionOutputProjSegmentModule(d_model)
    x = torch.randn(2, 8, d_model)
    out = mod(x)
    assert out.shape == (2, 8, d_model)


def test_attention_output_proj_state_dict_has_proj_keys() -> None:
    mod = AttentionOutputProjSegmentModule(16)
    sd = mod.state_dict()
    assert "proj.weight" in sd


# ---------------------------------------------------------------------------
# SegmentedForwardEngine in segment-store mode
# ---------------------------------------------------------------------------

def _build_segment_store_engine(
    *, dropout: float = 0.0
) -> tuple[SegmentedForwardEngine, CpuRamSegmentStore]:
    """Build a SegmentedForwardEngine in segment-store mode for testing."""
    torch.manual_seed(42)
    model_config = ModelConfig(
        vocab_size=64, max_seq_len=16, n_layers=2, d_model=16, n_heads=4, d_ff=32, dropout=dropout
    )
    seg_config = SegmentationConfig(attention_segments=2, mlp_chunks=2)
    factory = SegmentFactory(model_config=model_config, segmentation_config=seg_config)

    store = CpuRamSegmentStore()
    # Attention + MLP segments
    for sid, mod in factory.create_all_segments().iter_segments():
        store.save_segment(sid, mod.state_dict())

    # Global segments
    emb_cfg = EmbeddingConfig(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
    )
    store.save_segment(SegmentId(-1, "embedding", 0), EmbeddingSegmentModule(emb_cfg).state_dict())

    head_cfg = OutputHeadConfig(d_model=model_config.d_model, vocab_size=model_config.vocab_size)
    store.save_segment(SegmentId(-1, "output_head", 0), OutputHeadSegmentModule(head_cfg).state_dict())

    for layer_id in range(model_config.n_layers):
        proj_mod = AttentionOutputProjSegmentModule(model_config.d_model)
        store.save_segment(SegmentId(layer_id, "attention_output_proj", 0), proj_mod.state_dict())

    def module_factory(segment_id: SegmentId):
        if segment_id.segment_type == "attention":
            return factory.create_attention_segment(layer_id=segment_id.layer_id, segment_index=segment_id.segment_id)
        if segment_id.segment_type == "mlp":
            return factory.create_mlp_segment(layer_id=segment_id.layer_id, segment_index=segment_id.segment_id)
        if segment_id.segment_type == "embedding":
            return EmbeddingSegmentModule(emb_cfg)
        if segment_id.segment_type == "output_head":
            return OutputHeadSegmentModule(head_cfg)
        if segment_id.segment_type == "attention_output_proj":
            return AttentionOutputProjSegmentModule(model_config.d_model)
        raise ValueError(f"Unknown segment_type: {segment_id.segment_type!r}")

    loader = StrictSegmentLoader(segment_store=store, module_factory=module_factory, device="cpu")
    engine = SegmentedForwardEngine(
        model_config=model_config,
        segmentation_config=seg_config,
        segment_loader=loader,
        residual_dropout=dropout,
        use_segment_store_for_global=True,
    )
    return engine, store


def test_segment_store_engine_forward_produces_correct_shapes() -> None:
    engine, _ = _build_segment_store_engine()
    input_ids = torch.randint(0, 64, (3, 7))
    out = engine(input_ids, store_records=False)
    assert out.logits.shape == (3, 7, 64)
    assert out.hidden_states.shape == (3, 7, 16)
    assert engine.segment_loader.active_segment_count == 0


def test_segment_store_engine_forward_with_records() -> None:
    engine, _ = _build_segment_store_engine()
    input_ids = torch.randint(0, 64, (2, 5))
    out = engine(input_ids, store_records=True)
    assert out.runtime_records is not None
    assert len(out.runtime_records.attention_records) == 4  # 2 layers * 2 segs
    assert len(out.runtime_records.mlp_records) == 4  # 2 layers * 2 chunks


def test_segment_store_engine_no_large_components_in_parameters() -> None:
    engine, _ = _build_segment_store_engine()
    # In segment-store mode, the three large components should NOT be registered
    # as nn.Module submodules of the forward engine.
    assert engine.embeddings is None
    assert engine.attention_output_projections is None
    assert engine.output_head is None
    # Only layer_norms and mlp_shared_output_biases should be in parameters()
    param_count = sum(p.numel() for p in engine.parameters())
    # layer_norms: 2 layers * 2 norms * (16 weight + 16 bias) = 128
    # mlp_shared_output_biases: 2 layers * 16 = 32
    # Total tiny resident params: 160
    assert param_count < 500, f"Too many resident params: {param_count}"


def test_segment_store_engine_rejects_explicit_modules_together_with_flag() -> None:
    """Passing explicit embeddings + use_segment_store_for_global=True should raise."""
    from sequential_segmented_llm_training_inference.model.embeddings import (
        EmbeddingConfig, TokenPositionEmbeddings,
    )
    model_config = ModelConfig(vocab_size=64, max_seq_len=16, n_layers=2, d_model=16, n_heads=4, d_ff=32)
    seg_config = SegmentationConfig(attention_segments=2, mlp_chunks=2)
    store = CpuRamSegmentStore()

    def module_factory(sid):
        raise NotImplementedError

    loader = StrictSegmentLoader(segment_store=store, module_factory=module_factory, device="cpu")
    emb_cfg = EmbeddingConfig(vocab_size=64, d_model=16, max_seq_len=16)
    with pytest.raises(ValueError, match="use_segment_store_for_global"):
        SegmentedForwardEngine(
            model_config=model_config,
            segmentation_config=seg_config,
            segment_loader=loader,
            embeddings=TokenPositionEmbeddings(emb_cfg),
            use_segment_store_for_global=True,
        )


# ---------------------------------------------------------------------------
# Trainer backward pass in segment-store mode
# ---------------------------------------------------------------------------

def _build_trainer_segment_store_mode() -> tuple[SegmentedTrainer, SegmentedForwardEngine]:
    """Build a trainer using segment-store mode."""
    engine, store = _build_segment_store_engine()
    model_config = engine.model_config
    seg_config = engine.segmentation_config
    loader = engine.segment_loader

    backward_engine = SegmentedBackwardEngine(
        model_config=model_config,
        segmentation_config=seg_config,
        segment_loader=loader,
    )
    segment_optimizer = SegmentwiseAdamW(lr=1e-3)
    non_segment_optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-3)

    trainer = SegmentedTrainer(
        forward_engine=engine,
        backward_engine=backward_engine,
        segment_optimizer=segment_optimizer,
        non_segment_optimizer=non_segment_optimizer,
        loss_fn=CausalCrossEntropyLoss(ignore_index=-100),
        update_style="after_full_backward",
        gradient_accumulation_steps=1,
    )
    return trainer, engine


def test_trainer_segment_store_mode_train_step_runs() -> None:
    """A single training step completes without error in segment-store mode."""
    trainer, engine = _build_trainer_segment_store_mode()
    vocab_size = engine.model_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 8))
    labels = input_ids.clone()

    result = trainer.train_step({"input_ids": input_ids, "labels": labels})
    assert result.loss > 0
    assert result.optimizer_step_applied


def test_trainer_segment_store_mode_global_segments_accumulate_gradients() -> None:
    """After a backward pass, the accumulated segment gradients include the global segments."""
    trainer, engine = _build_trainer_segment_store_mode()
    # Temporarily set update_style to after_full_backward so gradients accumulate
    # before the step is applied.
    vocab_size = engine.model_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 8))
    labels = input_ids.clone()

    # Monkey-patch to skip the apply step so we can inspect accumulated grads.
    original_apply = trainer._apply_accumulated_segment_updates
    trainer._apply_accumulated_segment_updates = lambda: None  # type: ignore[method-assign]

    trainer.non_segment_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        output = engine(input_ids, store_records=True)
    runtime_records = output.runtime_records
    trainer._run_true_recomputation_backward(
        runtime_records=runtime_records,
        labels=labels,
    )

    # The accumulated segment gradients should include the global segment IDs.
    accumulated_ids = set(trainer._accumulated_segment_gradients.keys())
    emb_id = SegmentId(-1, "embedding", 0)
    head_id = SegmentId(-1, "output_head", 0)
    assert emb_id in accumulated_ids, f"embedding not in {accumulated_ids}"
    assert head_id in accumulated_ids, f"output_head not in {accumulated_ids}"
    for layer_id in range(engine.model_config.n_layers):
        proj_id = SegmentId(layer_id, "attention_output_proj", 0)
        assert proj_id in accumulated_ids, f"attn_proj L{layer_id} not in {accumulated_ids}"

    trainer._apply_accumulated_segment_updates = original_apply  # type: ignore[method-assign]

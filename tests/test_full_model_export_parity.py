"""Logit-parity tests for the full-model export + reassembly path.

These tests build a tiny segmented model in *segment-store mode*
(``use_segment_store_for_global=True``), export it through the exporter using the
exact checkpoint-dict shape ``scripts/train_segmented.py::export_full_model``
produces, load it into ``AssembledFullModel`` (the dense reassembled model used
by ``scripts/infer_exported_model.py``), and assert the dense model reproduces
the segmented forward engine's logits to fp32 tolerance.

This proves the global-component consolidation (embedding / output head /
attention output projection / sliced final norm) is numerically correct, not
just that the keys load.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import (
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import (
    SegmentedForwardEngine,
)
from sequential_segmented_llm_training_inference.execution.segment_loader import (
    StrictSegmentLoader,
)
from sequential_segmented_llm_training_inference.export.full_model_exporter import (
    export_full_model_from_checkpoint,
)
from sequential_segmented_llm_training_inference.model.embeddings import EmbeddingConfig
from sequential_segmented_llm_training_inference.model.output_head import OutputHeadConfig
from sequential_segmented_llm_training_inference.segments.global_segments import (
    AttentionOutputProjSegmentModule,
    EmbeddingSegmentModule,
    EmbeddingSliceSegmentModule,
    OutputHeadSegmentModule,
    OutputHeadSliceSegmentModule,
)
from sequential_segmented_llm_training_inference.segments.segment_factory import (
    SegmentFactory,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import (
    CpuRamSegmentStore,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_assembled_full_model_cls():
    """Import AssembledFullModel from scripts/infer_exported_model.py."""
    script_path = REPO_ROOT / "scripts" / "infer_exported_model.py"
    spec = importlib.util.spec_from_file_location("_infer_exported_model", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_infer_exported_model", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.AssembledFullModel


# ---------------------------------------------------------------------------
# Segment-store population (mirrors scripts/train_segmented.py)
# ---------------------------------------------------------------------------

def _module_factory(factory: SegmentFactory, mc: ModelConfig, sc: SegmentationConfig):
    n_emb = sc.embedding_segments
    n_head = sc.output_head_segments

    def _make(seg_id: SegmentId):
        if seg_id.segment_type == "attention":
            return factory.create_attention_segment(
                layer_id=seg_id.layer_id, segment_index=seg_id.segment_id
            )
        if seg_id.segment_type == "mlp":
            return factory.create_mlp_segment(
                layer_id=seg_id.layer_id, segment_index=seg_id.segment_id
            )
        if seg_id.segment_type == "embedding":
            if n_emb == 1:
                return EmbeddingSegmentModule(
                    EmbeddingConfig(
                        vocab_size=mc.vocab_size,
                        d_model=mc.d_model,
                        max_seq_len=mc.max_seq_len,
                        dropout=mc.dropout,
                        pad_token_id=mc.pad_token_id,
                    )
                )
            return EmbeddingSliceSegmentModule(
                vocab_size=mc.vocab_size,
                d_slice=mc.d_model // n_emb,
                max_seq_len=mc.max_seq_len,
            )
        if seg_id.segment_type == "output_head":
            if n_head == 1:
                return OutputHeadSegmentModule(
                    OutputHeadConfig(d_model=mc.d_model, vocab_size=mc.vocab_size)
                )
            base = mc.vocab_size // n_head
            i = seg_id.segment_id
            start = i * base
            end = mc.vocab_size if i == n_head - 1 else (i + 1) * base
            return OutputHeadSliceSegmentModule(
                d_model=mc.d_model, vocab_slice_size=end - start
            )
        if seg_id.segment_type == "attention_output_proj":
            return AttentionOutputProjSegmentModule(mc.d_model)
        raise ValueError(seg_id.segment_type)

    return _make


def _all_global_ids(mc: ModelConfig, sc: SegmentationConfig) -> list[SegmentId]:
    ids: list[SegmentId] = []
    for i in range(sc.embedding_segments):
        ids.append(SegmentId(-1, "embedding", i))
    for i in range(sc.output_head_segments):
        ids.append(SegmentId(-1, "output_head", i))
    for layer_id in range(mc.n_layers):
        ids.append(SegmentId(layer_id, "attention_output_proj", 0))
    return ids


def _build_populated_engine(mc: ModelConfig, sc: SegmentationConfig, seed: int):
    torch.manual_seed(seed)
    factory = SegmentFactory(model_config=mc, segmentation_config=sc, mlp_activation="gelu")
    store = CpuRamSegmentStore()
    make = _module_factory(factory, mc, sc)

    # Initialise + save every segment (attention/mlp + global) with random weights.
    all_ids = list(factory.expected_segment_ids()) + _all_global_ids(mc, sc)
    for seg_id in all_ids:
        module = make(seg_id)
        store.save_segment(seg_id, module.state_dict())
        del module

    loader = StrictSegmentLoader(segment_store=store, module_factory=make, device="cpu")
    engine = SegmentedForwardEngine(
        model_config=mc,
        segmentation_config=sc,
        segment_loader=loader,
        residual_dropout=0.0,
        use_segment_store_for_global=True,
    )
    engine.eval()
    return engine, factory, store


def _collect_non_segment_state(engine: SegmentedForwardEngine) -> dict:
    state: dict = {
        "layer_norms": engine.layer_norms.state_dict(),
        "mlp_shared_output_biases": engine.mlp_shared_output_biases.state_dict(),
    }
    if hasattr(engine, "output_slice_final_norm"):
        state["output_slice_final_norm"] = engine.output_slice_final_norm.state_dict()
    return state


def _build_checkpoint_dict(engine, factory, store, mc, sc) -> dict:
    """Replicate scripts/train_segmented.py::export_full_model checkpoint shape."""
    flat_non_segment: dict = {}
    for module_name, sd in _collect_non_segment_state(engine).items():
        for param_name, tensor in sd.items():
            flat_non_segment[f"{module_name}.{param_name}"] = tensor

    flat_attn_mlp_segments: dict = {}
    global_segments: dict = {}
    all_ids = list(factory.expected_segment_ids()) + _all_global_ids(mc, sc)
    for sid in all_ids:
        state = store.load_segment(sid)
        if sid.segment_type in ("attention", "mlp"):
            flat_attn_mlp_segments[f"layer_{sid.layer_id}.{sid.segment_type}.{sid.segment_id}"] = state
        elif sid.segment_type == "embedding":
            global_segments[f"emb.{sid.segment_id}"] = state
        elif sid.segment_type == "output_head":
            global_segments[f"head.{sid.segment_id}"] = state
        elif sid.segment_type == "attention_output_proj":
            global_segments[f"attn_proj.{sid.layer_id}"] = state

    return {
        "model_config": mc.to_dict(),
        "segmentation_config": sc.to_dict(),
        "non_segment_state": flat_non_segment,
        "segments": flat_attn_mlp_segments,
        "global_segments": global_segments,
    }


def _run_parity(mc: ModelConfig, sc: SegmentationConfig, tmp_path: Path, seed: int) -> float:
    engine, factory, store = _build_populated_engine(mc, sc, seed)

    torch.manual_seed(1234)
    input_ids = torch.randint(0, mc.vocab_size, (2, 7))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        seg_out = engine(input_ids, attention_mask=attention_mask, store_records=False)
    seg_logits = seg_out.logits

    checkpoint_dict = _build_checkpoint_dict(engine, factory, store, mc, sc)
    export_full_model_from_checkpoint(checkpoint_dict, tmp_path)
    artifact = torch.load(tmp_path / "full_model.pt", map_location="cpu", weights_only=False)

    AssembledFullModel = _load_assembled_full_model_cls()
    dense = AssembledFullModel(ModelConfig(**artifact["model_config"]))
    # Must load cleanly (no missing/unexpected keys) with strict=True.
    dense.load_state_dict(artifact["state_dict"])
    dense.eval()

    with torch.no_grad():
        dense_logits = dense(input_ids, attention_mask=attention_mask)

    assert dense_logits.shape == seg_logits.shape
    return (dense_logits - seg_logits).abs().max().item()


def _model_config(vocab_size: int = 40) -> ModelConfig:
    return ModelConfig(
        architecture="gpt_decoder",
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=256,
        vocab_size=vocab_size,
        max_seq_len=16,
        dropout=0.0,
    )


def test_export_logit_parity_sliced(tmp_path: Path) -> None:
    """N_emb>1, N_head>1: sliced embedding + sliced output head + final norm."""
    mc = _model_config(vocab_size=40)
    sc = SegmentationConfig(
        attention_segments=2,
        mlp_chunks=2,
        embedding_segments=2,
        output_head_segments=2,
    )
    max_abs_diff = _run_parity(mc, sc, tmp_path, seed=7)
    assert max_abs_diff < 1e-4, f"max-abs-diff={max_abs_diff}"


def test_export_logit_parity_single_slice(tmp_path: Path) -> None:
    """N_emb==1, N_head==1: whole embedding + whole output head."""
    mc = _model_config(vocab_size=40)
    sc = SegmentationConfig(
        attention_segments=2,
        mlp_chunks=2,
        embedding_segments=1,
        output_head_segments=1,
    )
    max_abs_diff = _run_parity(mc, sc, tmp_path, seed=11)
    assert max_abs_diff < 1e-4, f"max-abs-diff={max_abs_diff}"


def test_export_logit_parity_mixed(tmp_path: Path) -> None:
    """Mixed: sliced embedding, whole output head."""
    mc = _model_config(vocab_size=40)
    sc = SegmentationConfig(
        attention_segments=2,
        mlp_chunks=2,
        embedding_segments=2,
        output_head_segments=1,
    )
    max_abs_diff = _run_parity(mc, sc, tmp_path, seed=23)
    assert max_abs_diff < 1e-4, f"max-abs-diff={max_abs_diff}"

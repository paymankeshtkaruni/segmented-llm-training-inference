"""Tests for the "minimal records" memory technique.

Minimal records: when the recomputation check is OFF, the per-segment
``output_tensor`` is not retained in execution records (only its ``shape``
metadata). The backward scheduler splits the attention concat gradient via
``record.shape`` and skips the recompute check (no stored output to compare).

Guarded contract:
1. Exactness     — training with full records (check ON) and minimal records
                   (check OFF) produces identical losses and post-step weights.
2. Shape kept    — under minimal records each record's output_tensor is empty
                   (numel == 0) but record.shape is the correct full shape.
3. Guard         — minimal records + recompute check is refused by train script.
"""

from __future__ import annotations

import copy

import pytest
import torch

from sequential_segmented_llm_training_inference.config.model_config import ModelConfig
from sequential_segmented_llm_training_inference.config.segmentation_config import (
    SegmentationConfig,
)
from sequential_segmented_llm_training_inference.execution.backward_engine import (
    RecomputationCheckConfig,
    SegmentedBackwardEngine,
)
from sequential_segmented_llm_training_inference.execution.forward_engine import (
    SegmentedForwardEngine,
)
from sequential_segmented_llm_training_inference.execution.segment_loader import (
    StrictSegmentLoader,
)
from sequential_segmented_llm_training_inference.optimization.segmentwise_sgd import (
    SegmentwiseSGD,
)
from sequential_segmented_llm_training_inference.segments.segment_factory import (
    SegmentFactory,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.cpu_ram_segment_store import (
    CpuRamSegmentStore,
)
from sequential_segmented_llm_training_inference.training.trainer import (
    SegmentTrainBatch,
    SegmentedTrainer,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=64,
        dropout=0.0,  # deterministic forward
    )


def _build_trainer(
    *,
    store_segment_outputs: bool,
    recomputation_check: bool,
    initial_states: dict[SegmentId, dict] | None,
):
    """Build a trainer + store; if initial_states is given, load it instead of
    re-initialising random segment weights (so both runs start identically).

    A fixed seed before construction makes the always-resident non-segment
    modules (embeddings, layer norms, output head) init identically across
    builds, so the full-records and minimal-records trainers are byte-for-byte
    comparable when given the same segment ``initial_states``.
    """
    torch.manual_seed(12345)
    model_config = _model_config()
    seg_config = SegmentationConfig(attention_segments=2, mlp_chunks=2)
    factory = SegmentFactory(
        model_config=model_config,
        segmentation_config=seg_config,
        attention_dropout=0.0,
    )
    store = CpuRamSegmentStore()
    if initial_states is None:
        for segment_id, module in factory.create_all_segments().iter_segments():
            store.save_segment(segment_id, module.state_dict())
    else:
        for segment_id, state in initial_states.items():
            store.save_segment(segment_id, copy.deepcopy(state))

    def module_factory(segment_id: SegmentId):
        if segment_id.segment_type == "attention":
            return factory.create_attention_segment(
                layer_id=segment_id.layer_id,
                segment_index=segment_id.segment_id,
            )
        return factory.create_mlp_segment(
            layer_id=segment_id.layer_id,
            segment_index=segment_id.segment_id,
        )

    loader = StrictSegmentLoader(
        segment_store=store, module_factory=module_factory, device="cpu"
    )
    forward_engine = SegmentedForwardEngine(
        model_config=model_config,
        segmentation_config=seg_config,
        segment_loader=loader,
        residual_dropout=0.0,
        store_segment_outputs=store_segment_outputs,
    )
    backward_engine = SegmentedBackwardEngine(
        model_config=model_config,
        segmentation_config=seg_config,
        segment_loader=loader,
        recomputation_check=RecomputationCheckConfig(
            enabled=recomputation_check,
            raise_on_mismatch=recomputation_check,
        ),
    )
    trainer = SegmentedTrainer(
        forward_engine=forward_engine,
        backward_engine=backward_engine,
        segment_optimizer=SegmentwiseSGD(lr=0.05),
        non_segment_optimizer=torch.optim.SGD(forward_engine.parameters(), lr=0.05),
        update_style="after_full_backward",
    )
    return trainer, store, factory, model_config, seg_config


def _snapshot_states(store, factory, model_config, seg_config) -> dict[SegmentId, dict]:
    states: dict[SegmentId, dict] = {}
    for segment_id in factory.expected_segment_ids():
        states[segment_id] = {
            k: v.detach().clone() for k, v in store.load_segment(segment_id).items()
        }
    return states


def _run_two_steps(trainer) -> list[float]:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (2, 6))
    batch = SegmentTrainBatch(
        input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=torch.ones_like(input_ids),
    )
    losses: list[float] = []
    for _ in range(2):
        result = trainer.train_step(batch)
        losses.append(result.loss)
    return losses


# ---------------------------------------------------------------------------
# 1. Exactness
# ---------------------------------------------------------------------------

def test_minimal_records_matches_full_records_loss_and_weights() -> None:
    # Shared starting weights so the two runs are byte-for-byte comparable.
    _, seed_store, seed_factory, mc, sc = _build_trainer(
        store_segment_outputs=True, recomputation_check=False, initial_states=None
    )
    initial_states = _snapshot_states(seed_store, seed_factory, mc, sc)

    full_trainer, full_store, full_factory, mc_f, sc_f = _build_trainer(
        store_segment_outputs=True,
        recomputation_check=True,
        initial_states=initial_states,
    )
    min_trainer, min_store, min_factory, mc_m, sc_m = _build_trainer(
        store_segment_outputs=False,
        recomputation_check=False,
        initial_states=initial_states,
    )

    full_losses = _run_two_steps(full_trainer)
    min_losses = _run_two_steps(min_trainer)

    for lf, lm in zip(full_losses, min_losses, strict=True):
        assert lf == pytest.approx(lm, abs=1e-5)

    full_final = _snapshot_states(full_store, full_factory, mc_f, sc_f)
    min_final = _snapshot_states(min_store, min_factory, mc_m, sc_m)
    for seg_id in full_final:
        for name, w_full in full_final[seg_id].items():
            w_min = min_final[seg_id][name]
            assert torch.allclose(w_full, w_min, atol=1e-5), (
                f"weight mismatch at {seg_id} :: {name}"
            )


# ---------------------------------------------------------------------------
# 2. Shape preserved with empty output tensor
# ---------------------------------------------------------------------------

def test_minimal_records_keep_shape_but_drop_output_tensor() -> None:
    trainer, _store, _factory, _mc, _sc = _build_trainer(
        store_segment_outputs=False, recomputation_check=False, initial_states=None
    )
    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        output = trainer.forward_engine(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            store_records=True,
            detach_record_tensors=True,
            clone_record_tensors=True,
        )
    records = output.runtime_records
    assert records is not None
    assert len(records.attention_records) > 0
    assert len(records.mlp_records) > 0

    for record in records.attention_records.values():
        assert record.output_tensor.numel() == 0
        # full segment output shape = [batch, seq, d_model/attention_segments]
        assert record.shape == (2, 5, 128 // 2)
    for record in records.mlp_records.values():
        assert record.output_tensor.numel() == 0
        # MLP segment output projects back to d_model
        assert record.shape == (2, 5, 128)


def test_full_records_retain_output_tensor() -> None:
    trainer, _store, _factory, _mc, _sc = _build_trainer(
        store_segment_outputs=True, recomputation_check=False, initial_states=None
    )
    torch.manual_seed(2)
    input_ids = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        output = trainer.forward_engine(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            store_records=True,
        )
    records = output.runtime_records
    for record in records.attention_records.values():
        assert record.output_tensor.numel() > 0
        assert tuple(record.output_tensor.shape) == record.shape


# ---------------------------------------------------------------------------
# 3. Guard: minimal records + recompute check refused by train script
# ---------------------------------------------------------------------------

def test_train_script_refuses_minimal_records_with_recomputation_check() -> None:
    """The train script guard fires before any heavy pipeline work runs.

    Loaded from its file path under a unique module name so it never collides
    with the installed package namespace regardless of test ordering.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    # Other tests (test_cli_commands.py) install fake stub modules into
    # sys.modules under real package names without restoring them. Drop any such
    # stubs for the submodules this script imports so the real package loads.
    polluted = [
        name for name in list(sys.modules)
        if name.startswith("sequential_segmented_llm_training_inference")
        and getattr(sys.modules[name], "__file__", None) is None
    ]
    saved = {name: sys.modules.pop(name) for name in polluted}
    try:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "train_segmented.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_train_segmented_guard_under_test", script_path
        )
        train_segmented = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_segmented)
    finally:
        sys.modules.update(saved)

    args = train_segmented.parse_args(
        [
            "--minimal-records",
            "--recomputation-check",
            "--n-layers", "2",
            "--d-model", "128",
            "--n-heads", "4",
            "--d-ff", "64",
            "--attention-segments", "2",
            "--mlp-chunks", "2",
            "--device", "cpu",
            "--max-train-steps", "1",
            "--max-val-steps", "1",
        ]
    )
    with pytest.raises(SystemExit, match="incompatible with --recomputation-check"):
        train_segmented.main(args)


def test_backward_engine_recompute_check_refuses_minimal_records() -> None:
    """Engine-layer guard: a recompute check against minimal records (empty
    placeholder output) must raise rather than silently passing — the stored
    output needed for verification was not retained."""
    from sequential_segmented_llm_training_inference.execution.backward_engine import (
        RecomputationMismatchError,
    )

    trainer, _store, _factory, _mc, _sc = _build_trainer(
        store_segment_outputs=False, recomputation_check=True, initial_states=None
    )
    torch.manual_seed(3)
    input_ids = torch.randint(0, 64, (2, 5))
    batch = SegmentTrainBatch(
        input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=torch.ones_like(input_ids),
    )
    with pytest.raises(RecomputationMismatchError):
        trainer.train_step(batch)

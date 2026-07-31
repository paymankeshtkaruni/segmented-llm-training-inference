"""Tests for the pluggable optimizer-state store (in-memory vs disk).

Covers bit-exactness across multiple steps, checkpoint round-trip with the disk
store, the store unit behaviour, and the backward-compat .state property shim.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from sequential_segmented_llm_training_inference.optimization import (
    DiskOptimizerStateStore,
    InMemoryOptimizerStateStore,
    SegmentwiseAdamW,
    SegmentwiseSGD,
)
from sequential_segmented_llm_training_inference.optimization.segment_optimizer import (
    segment_parameter_key,
)
from sequential_segmented_llm_training_inference.segments.segment_ids import (
    SegmentId,
    SegmentParameterKey,
)


def make_module() -> nn.Linear:
    module = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        module.weight.fill_(0.3)
        module.bias.fill_(0.05)
    return module


def make_gradients(module: nn.Module, value: float) -> dict[str, torch.Tensor]:
    return {name: torch.full_like(p, value) for name, p in module.named_parameters()}


SEG = SegmentId(1, "attention", 0)


# ---------------------------------------------------------------------------
# 1. AdamW exactness across 3 steps (in_memory vs disk)
# ---------------------------------------------------------------------------

def test_adamw_disk_matches_in_memory_across_steps(tmp_path) -> None:
    module_mem = make_module()
    module_disk = copy.deepcopy(module_mem)

    opt_mem = SegmentwiseAdamW(lr=0.01, weight_decay=0.02,
                               state_store=InMemoryOptimizerStateStore())
    opt_disk = SegmentwiseAdamW(lr=0.01, weight_decay=0.02,
                                state_store=DiskOptimizerStateStore(tmp_path / "adam"))

    for step in range(1, 4):
        value = 0.1 * step
        opt_mem.step_module(segment_id=SEG, module=module_mem,
                            gradients=make_gradients(module_mem, value))
        opt_disk.step_module(segment_id=SEG, module=module_disk,
                             gradients=make_gradients(module_disk, value))

        # Params identical after each step.
        for (n_a, p_a), (n_b, p_b) in zip(
            module_mem.named_parameters(), module_disk.named_parameters()
        ):
            assert torch.allclose(p_a, p_b, atol=1e-6), f"param {n_a} diverged at step {step}"

        # Moments + integer step identical after each step.
        for name in ("weight", "bias"):
            key = segment_parameter_key(SEG, name)
            sm = opt_mem.state[key]
            sd = opt_disk.state[key]
            assert int(sm["step"]) == int(sd["step"]) == step
            assert torch.allclose(sm["exp_avg"], sd["exp_avg"], atol=1e-6)
            assert torch.allclose(sm["exp_avg_sq"], sd["exp_avg_sq"], atol=1e-6)


# ---------------------------------------------------------------------------
# 2. SGD-momentum exactness across 3 steps
# ---------------------------------------------------------------------------

def test_sgd_momentum_disk_matches_in_memory_across_steps(tmp_path) -> None:
    module_mem = make_module()
    module_disk = copy.deepcopy(module_mem)

    opt_mem = SegmentwiseSGD(lr=0.1, momentum=0.9, weight_decay=0.01,
                             state_store=InMemoryOptimizerStateStore())
    opt_disk = SegmentwiseSGD(lr=0.1, momentum=0.9, weight_decay=0.01,
                              state_store=DiskOptimizerStateStore(tmp_path / "sgd"))

    for step in range(1, 4):
        value = 0.2 * step
        opt_mem.step_module(segment_id=SEG, module=module_mem,
                            gradients=make_gradients(module_mem, value))
        opt_disk.step_module(segment_id=SEG, module=module_disk,
                             gradients=make_gradients(module_disk, value))

        for (n_a, p_a), (_, p_b) in zip(
            module_mem.named_parameters(), module_disk.named_parameters()
        ):
            assert torch.allclose(p_a, p_b, atol=1e-6), f"param {n_a} diverged at step {step}"

        for name in ("weight", "bias"):
            key = segment_parameter_key(SEG, name)
            bm = opt_mem.state[key]["momentum_buffer"]
            bd = opt_disk.state[key]["momentum_buffer"]
            assert torch.allclose(bm, bd, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Checkpoint round-trip with the disk store
# ---------------------------------------------------------------------------

def test_adamw_checkpoint_roundtrip_disk_store(tmp_path) -> None:
    module = make_module()
    opt = SegmentwiseAdamW(lr=0.01, weight_decay=0.0,
                           state_store=DiskOptimizerStateStore(tmp_path / "src"))
    for step in range(3):
        opt.step_module(segment_id=SEG, module=module,
                        gradients=make_gradients(module, 0.1))

    sd_before = opt.state_dict()

    # Restore into a fresh optimizer with a NEW disk store.
    restored = SegmentwiseAdamW(lr=0.5,
                                state_store=DiskOptimizerStateStore(tmp_path / "dst"))
    restored.load_state_dict(sd_before)

    sd_after = restored.state_dict()
    assert sd_before["state"].keys() == sd_after["state"].keys()
    for key in sd_before["state"]:
        assert int(sd_before["state"][key]["step"]) == int(sd_after["state"][key]["step"])
        assert torch.allclose(sd_before["state"][key]["exp_avg"],
                              sd_after["state"][key]["exp_avg"], atol=1e-6)
        assert torch.allclose(sd_before["state"][key]["exp_avg_sq"],
                              sd_after["state"][key]["exp_avg_sq"], atol=1e-6)
    assert restored.lr == pytest.approx(0.01)

    # One more step on a clone of the module must produce identical results
    # whether we continue the original or the restored optimizer.
    module_cont = copy.deepcopy(module)
    module_rest = copy.deepcopy(module)
    opt.step_module(segment_id=SEG, module=module_cont,
                    gradients=make_gradients(module_cont, 0.1))
    restored.step_module(segment_id=SEG, module=module_rest,
                         gradients=make_gradients(module_rest, 0.1))
    for (_, p_a), (_, p_b) in zip(
        module_cont.named_parameters(), module_rest.named_parameters()
    ):
        assert torch.allclose(p_a, p_b, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Store unit behaviour (in-memory + disk)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_store", [
    lambda tmp: InMemoryOptimizerStateStore(),
    lambda tmp: DiskOptimizerStateStore(tmp / "store"),
])
def test_store_get_put_has_segment_ids_clear(make_store, tmp_path) -> None:
    store = make_store(tmp_path)
    seg_a = SegmentId(0, "mlp", 0)
    seg_b = SegmentId(1, "mlp", 1)

    assert store.get(seg_a) == {}
    assert not store.has(seg_a)
    assert store.segment_ids() == []

    state_a = {"weight": {"step": 1, "exp_avg": torch.ones(2)}}
    store.put(seg_a, state_a)
    store.put(seg_b, {"weight": {"step": 2, "exp_avg": torch.zeros(2)}})

    assert store.has(seg_a)
    assert store.segment_ids() == sorted([seg_a, seg_b])
    loaded = store.get(seg_a)
    assert int(loaded["weight"]["step"]) == 1
    assert torch.allclose(loaded["weight"]["exp_avg"], torch.ones(2))

    store.clear()
    assert not store.has(seg_a)
    assert store.segment_ids() == []
    assert store.get(seg_a) == {}


# ---------------------------------------------------------------------------
# 5. .state property shim with default in-memory store
# ---------------------------------------------------------------------------

def test_state_property_shim_adamw() -> None:
    module = make_module()
    opt = SegmentwiseAdamW(lr=0.01, weight_decay=0.0)  # default in-memory
    opt.step_module(segment_id=SEG, module=module, gradients=make_gradients(module, 0.1))
    opt.step_module(segment_id=SEG, module=module, gradients=make_gradients(module, 0.1))

    key = SegmentParameterKey(1, "attention", 0, "weight")
    assert key in opt.state
    assert opt.state[key]["step"] == 2
    assert "exp_avg" in opt.state[key]
    assert "exp_avg_sq" in opt.state[key]


def test_state_property_shim_sgd_momentum() -> None:
    module = make_module()
    opt = SegmentwiseSGD(lr=0.1, momentum=0.9)  # default in-memory
    opt.step_module(segment_id=SEG, module=module, gradients=make_gradients(module, 0.2))

    key = SegmentParameterKey(1, "attention", 0, "weight")
    assert key in opt.state
    assert "momentum_buffer" in opt.state[key]


def test_state_property_empty_for_stateless_sgd() -> None:
    module = make_module()
    opt = SegmentwiseSGD(lr=0.1, momentum=0.0)  # stateless
    opt.step_module(segment_id=SEG, module=module, gradients=make_gradients(module, 0.2))
    assert opt.state == {}

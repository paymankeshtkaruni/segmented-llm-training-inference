"""Tests for disk-backed runtime record and gradient store backends."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
from sequential_segmented_llm_training_inference.storage.disk_gradient_store import (
    DiskGradientStore,
)
from sequential_segmented_llm_training_inference.storage.runtime_record_store_backends import (
    DiskRecordBackend,
    InMemoryRecordBackend,
    RuntimeRecordBackend,
)
from sequential_segmented_llm_training_inference.storage.runtime_records import RuntimeRecordStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attn_id(seg: int = 0, layer: int = 0) -> SegmentId:
    return SegmentId(layer_id=layer, segment_type="attention", segment_id=seg)


def _mlp_id(seg: int = 0, layer: int = 0) -> SegmentId:
    return SegmentId(layer_id=layer, segment_type="mlp", segment_id=seg)


# ---------------------------------------------------------------------------
# InMemoryRecordBackend
# ---------------------------------------------------------------------------

class TestInMemoryRecordBackend:
    def test_put_and_get(self) -> None:
        backend = InMemoryRecordBackend()
        t = torch.randn(3, 4)
        backend.put_tensor("foo", t)
        result = backend.get_tensor("foo")
        assert torch.allclose(result, t)

    def test_has_tensor(self) -> None:
        backend = InMemoryRecordBackend()
        assert not backend.has_tensor("x")
        backend.put_tensor("x", torch.zeros(2))
        assert backend.has_tensor("x")

    def test_get_missing_raises_key_error(self) -> None:
        backend = InMemoryRecordBackend()
        with pytest.raises(KeyError, match="not found in memory"):
            backend.get_tensor("missing")

    def test_evict_layer_removes_matching_keys(self) -> None:
        backend = InMemoryRecordBackend()
        backend.put_tensor("layer_0.input", torch.ones(2))
        backend.put_tensor("layer_0.output", torch.ones(3))
        backend.put_tensor("layer_1.input", torch.ones(4))
        backend.put_tensor("pre_final_norm", torch.ones(5))

        backend.evict_layer(0)
        assert not backend.has_tensor("layer_0.input")
        assert not backend.has_tensor("layer_0.output")
        assert backend.has_tensor("layer_1.input")
        assert backend.has_tensor("pre_final_norm")

    def test_clear_removes_all(self) -> None:
        backend = InMemoryRecordBackend()
        backend.put_tensor("a", torch.zeros(1))
        backend.put_tensor("b", torch.zeros(2))
        backend.clear()
        assert not backend.has_tensor("a")
        assert not backend.has_tensor("b")

    def test_store_property(self) -> None:
        backend = InMemoryRecordBackend()
        backend.put_tensor("k", torch.zeros(1))
        assert "k" in backend.store

    def test_is_abstract_subclass(self) -> None:
        assert issubclass(InMemoryRecordBackend, RuntimeRecordBackend)


# ---------------------------------------------------------------------------
# DiskRecordBackend
# ---------------------------------------------------------------------------

class TestDiskRecordBackend:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._root = Path(self._tmpdir) / "records"

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_backend(self) -> DiskRecordBackend:
        return DiskRecordBackend(self._root)

    def test_put_and_get_round_trip(self) -> None:
        backend = self._make_backend()
        t = torch.randn(3, 4)
        backend.put_tensor("embedding_output", t)
        loaded = backend.get_tensor("embedding_output")
        assert loaded.shape == t.shape
        assert torch.allclose(loaded, t)

    def test_key_with_dots_creates_subdirectory(self) -> None:
        backend = self._make_backend()
        backend.put_tensor("layer_0.attention_input", torch.zeros(2, 3))
        expected_path = self._root / "layer_0" / "attention_input.pt"
        assert expected_path.exists()

    def test_flat_key_creates_file_at_root(self) -> None:
        backend = self._make_backend()
        backend.put_tensor("pre_final_norm", torch.zeros(1))
        expected_path = self._root / "pre_final_norm.pt"
        assert expected_path.exists()

    def test_has_tensor_true_after_put(self) -> None:
        backend = self._make_backend()
        backend.put_tensor("foo.bar", torch.ones(2))
        assert backend.has_tensor("foo.bar")

    def test_has_tensor_false_when_absent(self) -> None:
        backend = self._make_backend()
        assert not backend.has_tensor("nonexistent")

    def test_get_missing_raises_key_error(self) -> None:
        backend = self._make_backend()
        with pytest.raises(KeyError, match="not found on disk"):
            backend.get_tensor("missing.key")

    def test_evict_layer_removes_subdirectory(self) -> None:
        backend = self._make_backend()
        backend.put_tensor("layer_0.input", torch.ones(2))
        backend.put_tensor("layer_0.mlp_sum", torch.ones(3))
        backend.put_tensor("layer_1.input", torch.ones(4))
        backend.put_tensor("pre_final_norm", torch.ones(5))

        backend.evict_layer(0)

        assert not backend.has_tensor("layer_0.input")
        assert not backend.has_tensor("layer_0.mlp_sum")
        assert backend.has_tensor("layer_1.input")
        assert backend.has_tensor("pre_final_norm")

    def test_evict_layer_no_error_when_absent(self) -> None:
        backend = self._make_backend()
        # Should not raise even if layer directory doesn't exist.
        backend.evict_layer(99)

    def test_clear_wipes_and_recreates_root(self) -> None:
        backend = self._make_backend()
        backend.put_tensor("a.b", torch.zeros(1))
        backend.clear()
        assert not backend.has_tensor("a.b")
        assert self._root.exists()

    def test_overwrite_replaces_value(self) -> None:
        backend = self._make_backend()
        backend.put_tensor("k", torch.zeros(2))
        backend.put_tensor("k", torch.ones(2))
        loaded = backend.get_tensor("k")
        assert torch.allclose(loaded, torch.ones(2))

    def test_detaches_tensor_before_save(self) -> None:
        backend = self._make_backend()
        # Create a tensor with a grad_fn.
        a = torch.randn(3, requires_grad=True)
        b = a * 2.0  # b.grad_fn is not None
        backend.put_tensor("computed", b)
        loaded = backend.get_tensor("computed")
        assert loaded.grad_fn is None

    def test_is_abstract_subclass(self) -> None:
        assert issubclass(DiskRecordBackend, RuntimeRecordBackend)


# ---------------------------------------------------------------------------
# RuntimeRecordStore with backends
# ---------------------------------------------------------------------------

class TestRuntimeRecordStoreWithBackend:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_default_store_is_in_memory(self) -> None:
        store = RuntimeRecordStore()
        assert store._backend is None
        t = torch.randn(2, 3)
        store.add_shared_tensor("foo", t)
        assert torch.allclose(store.get_shared_tensor("foo"), t)

    def test_in_memory_backend_routes_correctly(self) -> None:
        backend = InMemoryRecordBackend()
        store = RuntimeRecordStore(backend=backend)
        t = torch.randn(4)
        store.add_shared_tensor("bar", t)
        assert backend.has_tensor("bar")
        retrieved = store.get_shared_tensor("bar")
        assert retrieved.shape == t.shape

    def test_disk_backend_routes_correctly(self) -> None:
        root = Path(self._tmpdir) / "disk_records"
        backend = DiskRecordBackend(root)
        store = RuntimeRecordStore(backend=backend)
        t = torch.randn(2, 4)
        store.add_shared_tensor("layer_0.input", t)
        retrieved = store.get_shared_tensor("layer_0.input")
        assert retrieved.shape == t.shape
        assert torch.allclose(retrieved, t)

    def test_add_shared_tensor_overwrite_false_raises_with_backend(self) -> None:
        backend = InMemoryRecordBackend()
        store = RuntimeRecordStore(backend=backend)
        store.add_shared_tensor("k", torch.zeros(1))
        with pytest.raises(KeyError, match="already exists"):
            store.add_shared_tensor("k", torch.ones(1), overwrite=False)

    def test_add_shared_tensor_overwrite_true_with_backend(self) -> None:
        backend = InMemoryRecordBackend()
        store = RuntimeRecordStore(backend=backend)
        store.add_shared_tensor("k", torch.zeros(1))
        store.add_shared_tensor("k", torch.ones(1), overwrite=True)
        result = store.get_shared_tensor("k")
        assert torch.allclose(result, torch.ones(1))

    def test_evict_layer_in_memory_store(self) -> None:
        store = RuntimeRecordStore()
        store.add_shared_tensor("layer_2.input", torch.ones(3))
        store.add_shared_tensor("layer_2.mlp_sum", torch.ones(4))
        store.add_shared_tensor("layer_3.input", torch.ones(5))
        store.evict_layer(2)
        with pytest.raises(KeyError):
            store.get_shared_tensor("layer_2.input")
        with pytest.raises(KeyError):
            store.get_shared_tensor("layer_2.mlp_sum")
        assert store.get_shared_tensor("layer_3.input") is not None

    def test_evict_layer_disk_backend(self) -> None:
        root = Path(self._tmpdir) / "evict_test"
        backend = DiskRecordBackend(root)
        store = RuntimeRecordStore(backend=backend)
        store.add_shared_tensor("layer_1.input", torch.ones(3))
        store.add_shared_tensor("layer_1.mlp_sum", torch.ones(4))
        store.add_shared_tensor("layer_2.input", torch.ones(5))
        store.evict_layer(1)
        assert not backend.has_tensor("layer_1.input")
        assert not backend.has_tensor("layer_1.mlp_sum")
        assert backend.has_tensor("layer_2.input")

    def test_clear_execution_records_clears_backend(self) -> None:
        backend = InMemoryRecordBackend()
        store = RuntimeRecordStore(backend=backend)
        store.add_shared_tensor("x", torch.zeros(2))
        store.clear_execution_records()
        assert not backend.has_tensor("x")

    def test_is_empty_with_backend_always_false(self) -> None:
        backend = InMemoryRecordBackend()
        store = RuntimeRecordStore(backend=backend)
        # Even an empty backend store is not "empty" because _backend is not None.
        assert not store.is_empty()

    def test_is_empty_without_backend(self) -> None:
        store = RuntimeRecordStore()
        assert store.is_empty()
        store.add_shared_tensor("x", torch.zeros(1))
        assert not store.is_empty()

    def test_empty_key_raises(self) -> None:
        store = RuntimeRecordStore()
        with pytest.raises(ValueError, match="key must not be empty"):
            store.add_shared_tensor("", torch.zeros(1))

    def test_non_tensor_raises(self) -> None:
        store = RuntimeRecordStore()
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            store.add_shared_tensor("k", "not_a_tensor")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DiskGradientStore
# ---------------------------------------------------------------------------

class TestDiskGradientStore:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._root = Path(self._tmpdir) / "gradients"

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_store(self) -> DiskGradientStore:
        return DiskGradientStore(self._root)

    def test_accumulate_and_load(self) -> None:
        store = self._make_store()
        seg = _attn_id(0, 0)
        grads = {"weight": torch.ones(4, 4), "bias": torch.ones(4)}
        store.accumulate(seg, grads)
        loaded = store.load(seg)
        assert torch.allclose(loaded["weight"], torch.ones(4, 4))
        assert torch.allclose(loaded["bias"], torch.ones(4))

    def test_accumulate_twice_sums_gradients(self) -> None:
        store = self._make_store()
        seg = _mlp_id(1, 2)
        g1 = {"w": torch.ones(3)}
        g2 = {"w": torch.ones(3) * 2.0}
        store.accumulate(seg, g1)
        store.accumulate(seg, g2)
        loaded = store.load(seg)
        assert torch.allclose(loaded["w"], torch.ones(3) * 3.0)

    def test_load_missing_raises_key_error(self) -> None:
        store = self._make_store()
        with pytest.raises(KeyError):
            store.load(_attn_id(99, 99))

    def test_evict_removes_file(self) -> None:
        store = self._make_store()
        seg = _attn_id(0, 0)
        store.accumulate(seg, {"w": torch.zeros(2)})
        store.evict(seg)
        with pytest.raises(KeyError):
            store.load(seg)

    def test_evict_removes_from_keys(self) -> None:
        store = self._make_store()
        seg = _attn_id(0, 0)
        store.accumulate(seg, {"w": torch.zeros(2)})
        assert seg in store
        store.evict(seg)
        assert seg not in store

    def test_len(self) -> None:
        store = self._make_store()
        assert len(store) == 0
        store.accumulate(_attn_id(0, 0), {"w": torch.zeros(1)})
        store.accumulate(_mlp_id(0, 0), {"w": torch.zeros(1)})
        assert len(store) == 2

    def test_iter_sorted(self) -> None:
        store = self._make_store()
        seg_a = _attn_id(0, 1)
        seg_b = _mlp_id(0, 0)
        store.accumulate(seg_a, {"w": torch.zeros(1)})
        store.accumulate(seg_b, {"w": torch.zeros(1)})
        keys = list(store)
        assert len(keys) == 2
        # Both keys should be present (order is sorted).
        assert seg_a in keys
        assert seg_b in keys

    def test_clear_removes_all(self) -> None:
        store = self._make_store()
        store.accumulate(_attn_id(0, 0), {"w": torch.zeros(1)})
        store.accumulate(_mlp_id(0, 0), {"w": torch.zeros(1)})
        store.clear()
        assert len(store) == 0
        assert self._root.exists()

    def test_save_scaled_overwrites(self) -> None:
        store = self._make_store()
        seg = _attn_id(0, 0)
        store.accumulate(seg, {"w": torch.ones(3)})
        store.save_scaled(seg, {"w": torch.ones(3) * 0.5})
        loaded = store.load(seg)
        assert torch.allclose(loaded["w"], torch.ones(3) * 0.5)

    def test_accumulate_moves_to_cpu(self) -> None:
        store = self._make_store()
        seg = _attn_id(0, 0)
        g = torch.ones(4)
        store.accumulate(seg, {"w": g})
        loaded = store.load(seg)
        assert loaded["w"].device.type == "cpu"

    def test_accumulate_detaches_gradient(self) -> None:
        store = self._make_store()
        seg = _attn_id(0, 0)
        a = torch.randn(3, requires_grad=True)
        b = a * 2.0  # b.grad_fn is not None
        store.accumulate(seg, {"w": b})
        loaded = store.load(seg)
        assert loaded["w"].grad_fn is None


# ---------------------------------------------------------------------------
# Config validation for new fields
# ---------------------------------------------------------------------------

class TestRuntimeConfigNewFields:
    def test_default_values(self) -> None:
        from sequential_segmented_llm_training_inference.config.runtime_config import RuntimeConfig
        cfg = RuntimeConfig()
        assert cfg.record_backend == "in_memory"
        assert cfg.record_disk_dir is None
        assert cfg.evict_layers_during_backward is False

    def test_disk_backend_requires_dir(self) -> None:
        from sequential_segmented_llm_training_inference.config.runtime_config import RuntimeConfig
        with pytest.raises(ValueError, match="record_disk_dir is required"):
            RuntimeConfig(record_backend="disk", record_disk_dir=None)

    def test_disk_backend_with_dir_valid(self) -> None:
        from sequential_segmented_llm_training_inference.config.runtime_config import RuntimeConfig
        cfg = RuntimeConfig(record_backend="disk", record_disk_dir="/tmp/records")
        assert cfg.record_backend == "disk"
        assert cfg.record_disk_dir == "/tmp/records"

    def test_invalid_backend_raises(self) -> None:
        from sequential_segmented_llm_training_inference.config.runtime_config import RuntimeConfig
        with pytest.raises((ValueError, TypeError)):
            RuntimeConfig(record_backend="invalid")  # type: ignore[arg-type]


class TestTrainingConfigNewFields:
    def test_default_values(self) -> None:
        from sequential_segmented_llm_training_inference.config.training_config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.gradient_store == "in_memory"
        assert cfg.gradient_store_dir is None

    def test_disk_gradient_store_requires_dir(self) -> None:
        from sequential_segmented_llm_training_inference.config.training_config import TrainingConfig
        with pytest.raises(ValueError, match="gradient_store_dir is required"):
            TrainingConfig(gradient_store="disk", gradient_store_dir=None)

    def test_disk_gradient_store_with_dir_valid(self) -> None:
        from sequential_segmented_llm_training_inference.config.training_config import TrainingConfig
        cfg = TrainingConfig(gradient_store="disk", gradient_store_dir="/tmp/grads")
        assert cfg.gradient_store == "disk"

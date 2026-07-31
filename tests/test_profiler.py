"""Tests for Phase 23 runtime profiler."""

from __future__ import annotations

import time

import pytest

from sequential_segmented_llm_training_inference.training.profiler import (
    SegmentedTrainingProfiler,
    TimerStats,
    get_cpu_peak_memory_bytes,
    get_gpu_peak_memory_bytes,
)


def test_timer_stats_accumulates_time() -> None:
    stats = TimerStats()
    stats.add(0.5)
    stats.add(1.5)

    assert stats.total_seconds == pytest.approx(2.0)
    assert stats.count == 2
    assert stats.last_seconds == pytest.approx(1.5)
    assert stats.average_seconds == pytest.approx(1.0)


def test_timer_stats_rejects_negative_duration() -> None:
    stats = TimerStats()

    with pytest.raises(ValueError, match="non-negative"):
        stats.add(-0.1)


def test_profiler_initializes_default_metrics() -> None:
    profiler = SegmentedTrainingProfiler()

    assert "step" in profiler.timers
    assert "segment_loads" in profiler.counters
    assert "training_loss" in profiler.scalars


def test_time_section_records_elapsed_time() -> None:
    profiler = SegmentedTrainingProfiler()

    with profiler.time_section("forward"):
        time.sleep(0.001)

    assert profiler.timers["forward"].count == 1
    assert profiler.timers["forward"].total_seconds > 0


def test_add_time_custom_timer() -> None:
    profiler = SegmentedTrainingProfiler()

    profiler.add_time("custom_stage", 0.25)

    assert profiler.timers["custom_stage"].total_seconds == pytest.approx(0.25)
    assert profiler.timers["custom_stage"].count == 1


def test_increment_counter() -> None:
    profiler = SegmentedTrainingProfiler()

    profiler.increment("segment_loads")
    profiler.increment("segment_loads", 4)

    assert profiler.counters["segment_loads"] == 5


def test_increment_rejects_negative_amount() -> None:
    profiler = SegmentedTrainingProfiler()

    with pytest.raises(ValueError, match="non-negative"):
        profiler.increment("segment_loads", -1)


def test_record_losses_and_latest_mean() -> None:
    profiler = SegmentedTrainingProfiler()

    profiler.record_training_loss(2.0)
    profiler.record_training_loss(1.0)
    profiler.record_validation_loss(0.5)
    profiler.record_test_loss(0.25)

    assert profiler.latest_scalar("training_loss") == pytest.approx(1.0)
    assert profiler.mean_scalar("training_loss") == pytest.approx(1.5)
    assert profiler.latest_scalar("validation_loss") == pytest.approx(0.5)
    assert profiler.latest_scalar("test_loss") == pytest.approx(0.25)


def test_record_segment_events() -> None:
    profiler = SegmentedTrainingProfiler()

    profiler.record_segment_load(0.1)
    profiler.record_segment_unload(0.2)
    profiler.record_segment_save()
    profiler.record_recompute_mismatch()

    assert profiler.counters["segment_loads"] == 1
    assert profiler.counters["segment_unloads"] == 1
    assert profiler.counters["segment_saves"] == 1
    assert profiler.counters["recompute_mismatches"] == 1
    assert profiler.timers["segment_load"].total_seconds == pytest.approx(0.1)
    assert profiler.timers["segment_unload"].total_seconds == pytest.approx(0.2)


def test_record_multiple_recompute_mismatches() -> None:
    profiler = SegmentedTrainingProfiler()

    profiler.record_recompute_mismatch(3)

    assert profiler.counters["recompute_mismatches"] == 3


def test_cpu_peak_memory_is_positive_int() -> None:
    value = get_cpu_peak_memory_bytes()

    assert isinstance(value, int)
    assert value > 0


def test_gpu_peak_memory_returns_optional_int() -> None:
    value = get_gpu_peak_memory_bytes()

    assert value is None or isinstance(value, int)


def test_record_memory_snapshot() -> None:
    profiler = SegmentedTrainingProfiler()

    profiler.record_memory_snapshot()

    assert profiler.latest_scalar("cpu_peak_memory_bytes") is not None


def test_metadata_storage() -> None:
    profiler = SegmentedTrainingProfiler()

    profiler.set_metadata("device", "cpu")

    assert profiler.metadata["device"] == "cpu"


def test_metadata_rejects_empty_key() -> None:
    profiler = SegmentedTrainingProfiler()

    with pytest.raises(ValueError, match="Metadata key"):
        profiler.set_metadata("", "bad")


def test_summary_contains_timer_counter_scalar_metadata() -> None:
    profiler = SegmentedTrainingProfiler()
    profiler.add_time("forward", 0.5)
    profiler.increment("segment_loads")
    profiler.record_training_loss(1.25)
    profiler.set_metadata("run_id", "abc")

    summary = profiler.summary()

    assert summary["timers"]["forward"]["total_seconds"] == pytest.approx(0.5)
    assert summary["counters"]["segment_loads"] == 1
    assert summary["scalars"]["training_loss"]["latest"] == pytest.approx(1.25)
    assert summary["metadata"]["run_id"] == "abc"


def test_to_dict_is_plain_dictionary() -> None:
    profiler = SegmentedTrainingProfiler()
    profiler.add_time("backward", 0.7)
    profiler.increment("training_steps")
    profiler.record_training_loss(3.0)

    data = profiler.to_dict()

    assert isinstance(data, dict)
    assert data["timers"]["backward"]["total_seconds"] == pytest.approx(0.7)
    assert data["counters"]["training_steps"] == 1
    assert data["scalars"]["training_loss"] == [3.0]
    assert "summary" in data


def test_from_dict_roundtrip() -> None:
    profiler = SegmentedTrainingProfiler()
    profiler.add_time("optimizer_update", 0.33)
    profiler.increment("segment_saves", 2)
    profiler.record_validation_loss(0.9)
    profiler.set_metadata("storage_backend", "disk_streaming")

    restored = SegmentedTrainingProfiler.from_dict(profiler.to_dict())

    assert restored.timers["optimizer_update"].total_seconds == pytest.approx(0.33)
    assert restored.counters["segment_saves"] == 2
    assert restored.scalars["validation_loss"] == [0.9]
    assert restored.metadata["storage_backend"] == "disk_streaming"


def test_record_scalar_rejects_non_numeric() -> None:
    profiler = SegmentedTrainingProfiler()

    with pytest.raises(TypeError, match="numeric"):
        profiler.record_scalar("training_loss", "bad")  # type: ignore[arg-type]

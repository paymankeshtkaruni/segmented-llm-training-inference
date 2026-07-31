"""Final segmented testing after training.

Phase 18 scope:
- Run final test with segmented forward only.
- Never run backward.
- Never update optimizer or model parameters during the test pass.
- Optionally call a checkpoint loader before testing, usually for the best
  segmented checkpoint selected during validation.
- Return averaged final test loss and optional metrics.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn


def _release_memory() -> None:
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


Batch = Mapping[str, Any]
MetricFn = Callable[[Tensor, Tensor], float]
CheckpointLoader = Callable[[str | Path], Any]


@dataclass(slots=True)
class TestResult:
    """Aggregated final test output."""

    loss: Optional[float]
    metrics: dict[str, float] = field(default_factory=dict)
    num_batches: int = 0
    num_examples: int = 0
    checkpoint: str | None = None
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "metrics": dict(self.metrics),
            "num_batches": self.num_batches,
            "num_examples": self.num_examples,
            "checkpoint": self.checkpoint,
            "checkpoint_metadata": dict(self.checkpoint_metadata),
        }


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _get_output_value(output: Any, key: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(key)
    return getattr(output, key, None)


def _infer_num_examples(batch: Batch) -> int:
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, Tensor) and input_ids.ndim >= 1:
        return int(input_ids.shape[0])
    for value in batch.values():
        if isinstance(value, Tensor) and value.ndim >= 1:
            return int(value.shape[0])
    return 1


def _metadata_from_loaded_checkpoint(loaded: Any) -> dict[str, Any]:
    """Extract lightweight metadata from common checkpoint loader outputs."""

    if loaded is None:
        return {}
    if isinstance(loaded, Mapping):
        return dict(loaded.get("metadata", loaded))

    manifest = getattr(loaded, "manifest", None)
    if manifest is not None:
        metadata: dict[str, Any] = {}
        for attr in ("checkpoint_name", "checkpoint_type", "epoch", "global_step"):
            if hasattr(manifest, attr):
                metadata[attr] = getattr(manifest, attr)
        if hasattr(manifest, "validation_metadata"):
            metadata["validation_metadata"] = dict(manifest.validation_metadata)
        return metadata

    checkpoint = getattr(loaded, "checkpoint", None)
    if checkpoint is not None:
        metadata = {}
        for attr in ("epoch", "global_step"):
            if hasattr(checkpoint, attr):
                metadata[attr] = getattr(checkpoint, attr)
        if hasattr(checkpoint, "metadata"):
            metadata["metadata"] = dict(checkpoint.metadata)
        return metadata

    return {}


class SegmentedTester:
    """Run the final test using segmented forward only.

    The tester deliberately performs no backward pass and no optimizer step.
    A checkpoint loader can be supplied to restore the best segmented checkpoint
    before the final test. The loader is responsible for applying checkpoint
    state to the forward engine and its segment storage.
    """

    def __init__(
        self,
        forward_engine: Any,
        *,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        metric_fns: Optional[Mapping[str, MetricFn]] = None,
        device: str | torch.device = "cpu",
        checkpoint_loader: CheckpointLoader | None = None,
    ) -> None:
        self.forward_engine = forward_engine
        self.loss_fn = loss_fn
        self.metric_fns = dict(metric_fns or {})
        self.device = torch.device(device)
        self.checkpoint_loader = checkpoint_loader

    def test(
        self,
        dataloader: Iterable[Batch],
        *,
        checkpoint: str | Path | None = None,
    ) -> TestResult:
        """Run final segmented test.

        Args:
            dataloader: Iterable of test batches.
            checkpoint: Optional best segmented checkpoint identifier/path. When
                provided, ``checkpoint_loader`` must also be provided.
        """

        checkpoint_metadata: dict[str, Any] = {}
        if checkpoint is not None:
            if self.checkpoint_loader is None:
                raise ValueError(
                    "checkpoint was provided, but no checkpoint_loader was configured."
                )
            loaded = self.checkpoint_loader(checkpoint)
            checkpoint_metadata = _metadata_from_loaded_checkpoint(loaded)

        previous_training_state: Optional[bool] = None
        if isinstance(self.forward_engine, nn.Module):
            previous_training_state = self.forward_engine.training
            self.forward_engine.eval()

        total_loss = 0.0
        loss_weight = 0
        metric_totals = {name: 0.0 for name in self.metric_fns}
        metric_weight = 0
        num_batches = 0
        num_examples = 0

        try:
            with torch.no_grad():
                for batch in dataloader:
                    moved_batch = _move_to_device(dict(batch), self.device)
                    batch_examples = _infer_num_examples(moved_batch)

                    output = self._run_forward(moved_batch)

                    loss = _get_output_value(output, "loss")
                    logits = _get_output_value(output, "logits")
                    del output  # free hidden states and other large forward tensors

                    labels = moved_batch.get("labels")
                    del moved_batch  # free input tensors; labels ref keeps tensor alive

                    if loss is None and self.loss_fn is not None:
                        if logits is None:
                            raise ValueError(
                                "forward output does not contain logits, so loss_fn "
                                "cannot compute test loss."
                            )
                        if labels is None:
                            raise ValueError(
                                "batch does not contain labels, so loss_fn cannot "
                                "compute test loss."
                            )
                        loss = self.loss_fn(logits, labels)

                    if isinstance(loss, Tensor):
                        if loss.ndim != 0:
                            raise ValueError(
                                "test loss must be scalar, got shape "
                                f"{tuple(loss.shape)}."
                            )
                        total_loss += float(loss.detach().cpu()) * batch_examples
                        loss_weight += batch_examples
                    elif loss is not None:
                        total_loss += float(loss) * batch_examples
                        loss_weight += batch_examples

                    if self.metric_fns:
                        if logits is None:
                            raise ValueError(
                                "forward output does not contain logits, so metrics "
                                "cannot be computed."
                            )
                        if labels is None:
                            raise ValueError(
                                "batch does not contain labels, so metrics cannot "
                                "be computed."
                            )
                        for name, metric_fn in self.metric_fns.items():
                            metric_totals[name] += float(metric_fn(logits, labels)) * batch_examples
                        metric_weight += batch_examples

                    num_batches += 1
                    num_examples += batch_examples
                    del loss, logits, labels
                    _release_memory()
        finally:
            if previous_training_state is not None:
                self.forward_engine.train(previous_training_state)
            _release_memory()

        avg_loss = total_loss / loss_weight if loss_weight else None
        avg_metrics = (
            {name: value / metric_weight for name, value in metric_totals.items()}
            if metric_weight
            else {}
        )

        return TestResult(
            loss=avg_loss,
            metrics=avg_metrics,
            num_batches=num_batches,
            num_examples=num_examples,
            checkpoint=None if checkpoint is None else str(checkpoint),
            checkpoint_metadata=checkpoint_metadata,
        )

    def _run_forward(self, batch: Batch) -> Any:
        """Call the forward engine with only supported forward fields."""

        forward_batch = {
            "input_ids": batch.get("input_ids"),
            "attention_mask": batch.get("attention_mask"),
            "position_ids": batch.get("position_ids"),
        }
        forward_batch = {k: v for k, v in forward_batch.items() if v is not None}
        if "input_ids" not in forward_batch:
            raise KeyError("batch must contain input_ids for segmented final testing.")

        if hasattr(self.forward_engine, "forward"):
            try:
                return self.forward_engine.forward(**forward_batch, store_records=False)
            except TypeError:
                # Engine does not accept store_records (e.g. test doubles).
                try:
                    return self.forward_engine.forward(**forward_batch)
                except TypeError as exc:
                    if "labels" in batch:
                        return self.forward_engine.forward(**batch)
                    raise exc
        if callable(self.forward_engine):
            try:
                return self.forward_engine(**forward_batch, store_records=False)
            except TypeError:
                try:
                    return self.forward_engine(**forward_batch)
                except TypeError as exc:
                    if "labels" in batch:
                        return self.forward_engine(**batch)
                    raise exc
        raise TypeError("forward_engine must be callable or expose a forward method.")

"""Logical segment identifiers for sequential segmented LLM training.

Segments are repeatedly loaded, unloaded, reconstructed, checkpointed, and
optimized. Optimizer state and checkpoint metadata therefore need stable logical
identifiers, not temporary Python object identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal


SegmentType = Literal["attention", "mlp", "embedding", "output_head", "attention_output_proj"]
VALID_SEGMENT_TYPES: tuple[str, ...] = ("attention", "mlp", "embedding", "output_head", "attention_output_proj")

# Segment types that are global (not per-layer) and use layer_id=-1.
GLOBAL_SEGMENT_TYPES: frozenset[str] = frozenset(("embedding", "output_head"))


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def _validate_segment_type(segment_type: str) -> None:
    if segment_type not in VALID_SEGMENT_TYPES:
        allowed = ", ".join(VALID_SEGMENT_TYPES)
        raise ValueError(
            f"segment_type must be one of {{{allowed}}}, got {segment_type!r}."
        )


@dataclass(frozen=True, slots=True, order=True)
class SegmentId:
    """Stable logical identity for one attention, MLP, embedding, output_head,
    or attention_output_proj segment.

    Attributes:
        layer_id: Transformer layer index.  Use -1 for global segments
            (``embedding`` and ``output_head``).  Must be >= 0 for all other types.
        segment_type: One of ``"attention"``, ``"mlp"``, ``"embedding"``,
            ``"output_head"``, or ``"attention_output_proj"``.
        segment_id: Segment index within that layer and segment type.
    """

    layer_id: int
    segment_type: SegmentType
    segment_id: int

    KEY_PREFIX: ClassVar[str] = "layer"

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, int):
            raise TypeError(f"layer_id must be an int, got {type(self.layer_id).__name__}.")
        _validate_segment_type(self.segment_type)
        _validate_non_negative_int("segment_id", self.segment_id)
        if self.segment_type in GLOBAL_SEGMENT_TYPES:
            if self.layer_id != -1:
                raise ValueError(
                    f"layer_id must be -1 for segment_type={self.segment_type!r}, "
                    f"got {self.layer_id}."
                )
        else:
            if self.layer_id < 0:
                raise ValueError(
                    f"layer_id must be non-negative for segment_type={self.segment_type!r}, "
                    f"got {self.layer_id}."
                )

    def to_key(self) -> str:
        """Return a deterministic compact key."""

        return f"layer_{self.layer_id}.{self.segment_type}.{self.segment_id}"

    def to_path_name(self) -> str:
        """Return a filesystem-safe deterministic name."""

        return f"layer_{self.layer_id}__{self.segment_type}__segment_{self.segment_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe dictionary."""

        return {
            "layer_id": self.layer_id,
            "segment_type": self.segment_type,
            "segment_id": self.segment_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentId":
        """Build a segment identifier from a YAML/JSON dictionary."""

        required = {"layer_id", "segment_type", "segment_id"}
        missing = required - set(data)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing SegmentId field(s): {missing_text}")

        return cls(
            layer_id=int(data["layer_id"]),
            segment_type=data["segment_type"],
            segment_id=int(data["segment_id"]),
        )

    @classmethod
    def from_key(cls, key: str) -> "SegmentId":
        """Parse the deterministic key produced by :meth:`to_key`."""

        parts = key.split(".")
        if len(parts) != 3:
            raise ValueError(
                "SegmentId key must have format "
                "'layer_{layer_id}.{segment_type}.{segment_id}'."
            )

        layer_part, segment_type, segment_id_part = parts
        if not layer_part.startswith("layer_"):
            raise ValueError("SegmentId key layer part must start with 'layer_'.")

        try:
            layer_id = int(layer_part.removeprefix("layer_"))
            segment_id = int(segment_id_part)
        except ValueError as exc:
            raise ValueError(f"Invalid SegmentId key: {key!r}") from exc

        return cls(layer_id=layer_id, segment_type=segment_type, segment_id=segment_id)


@dataclass(frozen=True, slots=True, order=True)
class SegmentParameterKey:
    """Stable logical identity for one parameter inside one segment.

    This key is required for persistent segment optimizer state, especially for
    AdamW, because loaded segment modules can be temporary Python objects.
    """

    layer_id: int
    segment_type: SegmentType
    segment_id: int
    parameter_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, int):
            raise TypeError(f"layer_id must be an int, got {type(self.layer_id).__name__}.")
        _validate_segment_type(self.segment_type)
        _validate_non_negative_int("segment_id", self.segment_id)
        # Allow layer_id=-1 for global segment types
        if self.segment_type in GLOBAL_SEGMENT_TYPES:
            if self.layer_id != -1:
                raise ValueError(
                    f"layer_id must be -1 for segment_type={self.segment_type!r}, "
                    f"got {self.layer_id}."
                )
        else:
            if self.layer_id < 0:
                raise ValueError(
                    f"layer_id must be non-negative for segment_type={self.segment_type!r}, "
                    f"got {self.layer_id}."
                )
        if not isinstance(self.parameter_name, str):
            raise TypeError(
                "parameter_name must be a string, "
                f"got {type(self.parameter_name).__name__}."
            )
        if not self.parameter_name:
            raise ValueError("parameter_name must not be empty.")

    @property
    def segment_id_object(self) -> SegmentId:
        """Return the parent SegmentId."""

        return SegmentId(
            layer_id=self.layer_id,
            segment_type=self.segment_type,
            segment_id=self.segment_id,
        )

    def to_key(self) -> str:
        """Return a deterministic optimizer/checkpoint state key."""

        return (
            f"layer_{self.layer_id}."
            f"{self.segment_type}."
            f"{self.segment_id}."
            f"{self.parameter_name}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe dictionary."""

        return {
            "layer_id": self.layer_id,
            "segment_type": self.segment_type,
            "segment_id": self.segment_id,
            "parameter_name": self.parameter_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentParameterKey":
        """Build a parameter key from a YAML/JSON dictionary."""

        required = {"layer_id", "segment_type", "segment_id", "parameter_name"}
        missing = required - set(data)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing SegmentParameterKey field(s): {missing_text}")

        return cls(
            layer_id=int(data["layer_id"]),
            segment_type=data["segment_type"],
            segment_id=int(data["segment_id"]),
            parameter_name=str(data["parameter_name"]),
        )

    @classmethod
    def from_segment_id(
        cls,
        segment_id: SegmentId,
        parameter_name: str,
    ) -> "SegmentParameterKey":
        """Create a parameter key from a SegmentId and parameter name."""

        return cls(
            layer_id=segment_id.layer_id,
            segment_type=segment_id.segment_type,
            segment_id=segment_id.segment_id,
            parameter_name=parameter_name,
        )

    @classmethod
    def from_key(cls, key: str) -> "SegmentParameterKey":
        """Parse the deterministic key produced by :meth:`to_key`."""

        parts = key.split(".", maxsplit=3)
        if len(parts) != 4:
            raise ValueError(
                "SegmentParameterKey must have format "
                "'layer_{layer_id}.{segment_type}.{segment_id}.{parameter_name}'."
            )

        layer_part, segment_type, segment_id_part, parameter_name = parts
        if not layer_part.startswith("layer_"):
            raise ValueError(
                "SegmentParameterKey layer part must start with 'layer_'."
            )

        try:
            layer_id = int(layer_part.removeprefix("layer_"))
            segment_id = int(segment_id_part)
        except ValueError as exc:
            raise ValueError(f"Invalid SegmentParameterKey: {key!r}") from exc

        return cls(
            layer_id=layer_id,
            segment_type=segment_type,
            segment_id=segment_id,
            parameter_name=parameter_name,
        )

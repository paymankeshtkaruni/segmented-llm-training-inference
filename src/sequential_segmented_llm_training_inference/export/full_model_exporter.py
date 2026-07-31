"""Export a trained segmented checkpoint into a normal full-model artifact.

Phase 19 scope:
- Load a trained segmented checkpoint representation.
- Assemble attention head-group segment parameters into full attention weights.
- Assemble MLP hidden-dimension chunk parameters into full MLP weights.
- Copy non-segment parameters unchanged.
- Save a full-model artifact for later standard inference/deployment.

This exporter is intentionally independent from the training/validation/testing path.
Training, validation, and final test remain segmented. Full-model export happens after
learning and final segmented testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping
import time

import torch
from torch import Tensor

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is a project dependency.
    yaml = None  # type: ignore[assignment]


SegmentTuple = tuple[int, str, int]


@dataclass(frozen=True, slots=True)
class FullModelExportConfig:
    """Configuration for exporting a normal full-model artifact."""

    output_dir: str | Path
    full_model_filename: str = "full_model.pt"
    metadata_filename: str = "full_model_export_metadata.yaml"
    strict: bool = True
    attention_prefix_template: str = "layers.{layer_id}.attention"
    mlp_prefix_template: str = "layers.{layer_id}.mlp"

    def __post_init__(self) -> None:
        if not self.full_model_filename:
            raise ValueError("full_model_filename must not be empty.")
        if not self.metadata_filename:
            raise ValueError("metadata_filename must not be empty.")


@dataclass(slots=True)
class FullModelExportResult:
    """Paths and metadata produced by the exporter."""

    output_dir: Path
    full_model_path: Path
    metadata_path: Path
    num_attention_segments: int
    num_mlp_segments: int
    num_non_segment_parameters: int
    num_full_state_parameters: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _ensure_tensor(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _parse_segment_key(key: Any) -> SegmentTuple:
    """Parse stable segment identity from supported key formats.

    Supported string format:
        layer_{layer_id}.{segment_type}.{segment_id}

    Supported mapping format:
        {"layer_id": int, "segment_type": "attention"|"mlp", "segment_id": int}
    """

    if isinstance(key, Mapping):
        try:
            layer_id = int(key["layer_id"])
            segment_type = str(key["segment_type"])
            segment_id = int(key["segment_id"])
        except KeyError as exc:
            raise ValueError(f"Missing segment key field: {exc.args[0]}") from exc
    elif isinstance(key, str):
        parts = key.split(".")
        if len(parts) != 3:
            raise ValueError(
                "Segment key must have format 'layer_{layer_id}.{segment_type}.{segment_id}', "
                f"got {key!r}."
            )
        layer_part, segment_type, segment_id_text = parts
        if not layer_part.startswith("layer_"):
            raise ValueError(f"Segment key layer part must start with 'layer_', got {key!r}.")
        try:
            layer_id = int(layer_part.removeprefix("layer_"))
            segment_id = int(segment_id_text)
        except ValueError as exc:
            raise ValueError(f"Invalid segment key: {key!r}") from exc
    else:
        raise TypeError(f"Unsupported segment key type: {type(key).__name__}.")

    if layer_id < 0:
        raise ValueError(f"layer_id must be non-negative, got {layer_id}.")
    if segment_id < 0:
        raise ValueError(f"segment_id must be non-negative, got {segment_id}.")
    if segment_type not in {"attention", "mlp"}:
        raise ValueError(f"segment_type must be 'attention' or 'mlp', got {segment_type!r}.")

    return layer_id, segment_type, segment_id


def _normalize_segments(segments: Any) -> dict[SegmentTuple, dict[str, Tensor]]:
    """Normalize supported segment checkpoint formats to one mapping."""

    normalized: dict[SegmentTuple, dict[str, Tensor]] = {}

    if isinstance(segments, Mapping):
        iterable = segments.items()
        for key, state in iterable:
            segment_tuple = _parse_segment_key(key)
            if not isinstance(state, Mapping):
                raise TypeError("Each segment state must be a mapping of parameter names to tensors.")
            normalized[segment_tuple] = {
                str(name): _ensure_tensor(value, name=str(name))
                for name, value in state.items()
            }
        return normalized

    if isinstance(segments, list):
        for item in segments:
            if not isinstance(item, Mapping):
                raise TypeError("Segment list entries must be mappings.")
            if "segment_id" not in item or "state_dict" not in item:
                raise ValueError("Segment list entries must contain 'segment_id' and 'state_dict'.")
            segment_tuple = _parse_segment_key(item["segment_id"])
            state = item["state_dict"]
            if not isinstance(state, Mapping):
                raise TypeError("Segment state_dict must be a mapping.")
            normalized[segment_tuple] = {
                str(name): _ensure_tensor(value, name=str(name))
                for name, value in state.items()
            }
        return normalized

    raise TypeError("segments must be a mapping or a list of segment records.")


def _sorted_segment_items(
    segments: Mapping[SegmentTuple, Mapping[str, Tensor]],
    *,
    layer_id: int,
    segment_type: str,
) -> list[tuple[SegmentTuple, Mapping[str, Tensor]]]:
    return sorted(
        ((key, state) for key, state in segments.items() if key[0] == layer_id and key[1] == segment_type),
        key=lambda item: item[0][2],
    )


def _first_present(state: Mapping[str, Tensor], *names: str) -> Tensor | None:
    """Return the first tensor found among ``names`` (None if none present)."""
    for name in names:
        if name in state:
            return state[name]
    return None


def _cat_required(
    states: list[Mapping[str, Tensor]],
    param_name: str,
    *,
    dim: int,
    strict: bool,
) -> Tensor | None:
    tensors: list[Tensor] = []
    for idx, state in enumerate(states):
        if param_name not in state:
            if strict:
                raise KeyError(f"Missing required segment parameter {param_name!r} in segment index {idx}.")
            return None
        tensors.append(state[param_name])
    if not tensors:
        return None
    return torch.cat(tensors, dim=dim)


class FullModelExporter:
    """Exporter for reconstructing a normal full-model state from segments."""

    def __init__(self, config: FullModelExportConfig) -> None:
        self.config = config

    def export_from_checkpoint_dict(self, checkpoint: Mapping[str, Any]) -> FullModelExportResult:
        """Export a full-model artifact from an in-memory segmented checkpoint dictionary."""

        started = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_config = dict(checkpoint.get("model_config", checkpoint.get("model", {})))
        segmentation_config = dict(checkpoint.get("segmentation_config", checkpoint.get("segmentation", {})))
        non_segment_state = checkpoint.get("non_segment_state", checkpoint.get("non_segment_parameters", {}))
        segments_raw = checkpoint.get("segments")

        if segments_raw is None:
            raise KeyError("checkpoint must contain a 'segments' entry.")
        if not isinstance(non_segment_state, Mapping):
            raise TypeError("non_segment_state must be a mapping.")

        segments = _normalize_segments(segments_raw)

        n_layers = int(model_config.get("n_layers", self._infer_n_layers(segments)))
        attention_segments = int(segmentation_config.get("attention_segments", self._infer_count(segments, "attention")))
        mlp_chunks = int(segmentation_config.get("mlp_chunks", self._infer_count(segments, "mlp")))

        if n_layers <= 0:
            raise ValueError("n_layers must be positive or inferable from segments.")
        if attention_segments <= 1:
            raise ValueError("attention_segments must be > 1 for segmented export.")
        if mlp_chunks <= 1:
            raise ValueError("mlp_chunks must be > 1 for segmented export.")

        full_state: dict[str, Tensor] = {
            str(name): _ensure_tensor(value, name=str(name)).detach().cpu().clone()
            for name, value in non_segment_state.items()
        }

        embedding_segments = int(segmentation_config.get("embedding_segments", 1))
        output_head_segments = int(segmentation_config.get("output_head_segments", 1))

        # In segment-store mode the large GLOBAL components (embedding table,
        # output head, and per-layer attention output projection) live in the
        # segment store as their own segments.  They are passed through under
        # ``global_segments`` keyed by stable segment identity.  Consolidate
        # them — together with the always-resident sliced final norm carried in
        # ``non_segment_state`` — into the dense naming a standard full model
        # expects.  This mutates ``full_state`` in place: it removes any
        # slice-named global keys and inserts the consolidated dense keys.
        global_segments_raw = checkpoint.get("global_segments")
        self._consolidate_global_components(
            full_state,
            global_segments_raw,
            n_layers=n_layers,
            embedding_segments=embedding_segments,
            output_head_segments=output_head_segments,
        )

        for layer_id in range(n_layers):
            attention_states = [
                state for _, state in _sorted_segment_items(segments, layer_id=layer_id, segment_type="attention")
            ]
            mlp_states = [
                state for _, state in _sorted_segment_items(segments, layer_id=layer_id, segment_type="mlp")
            ]

            if len(attention_states) != attention_segments:
                raise ValueError(
                    f"Layer {layer_id} expected {attention_segments} attention segments, "
                    f"found {len(attention_states)}."
                )
            if len(mlp_states) != mlp_chunks:
                raise ValueError(
                    f"Layer {layer_id} expected {mlp_chunks} MLP segments, found {len(mlp_states)}."
                )

            full_state.update(self._assemble_attention_layer(layer_id, attention_states))
            full_state.update(self._assemble_mlp_layer(layer_id, mlp_states))

        metadata = {
            "export_type": "full_model_from_segments",
            "created_at_unix": started,
            "duration_seconds": time.time() - started,
            "model_config": model_config,
            "segmentation_config": segmentation_config,
            "n_layers": n_layers,
            "attention_segments": attention_segments,
            "mlp_chunks": mlp_chunks,
            "num_attention_segments": sum(1 for key in segments if key[1] == "attention"),
            "num_mlp_segments": sum(1 for key in segments if key[1] == "mlp"),
            "num_non_segment_parameters": len(non_segment_state),
            "num_full_state_parameters": len(full_state),
            "strict": self.config.strict,
        }

        artifact = {
            "model_config": model_config,
            "segmentation_config": segmentation_config,
            "state_dict": full_state,
            "export_metadata": metadata,
        }

        full_model_path = output_dir / self.config.full_model_filename
        metadata_path = output_dir / self.config.metadata_filename

        torch.save(artifact, full_model_path)
        self._write_metadata(metadata_path, metadata)

        return FullModelExportResult(
            output_dir=output_dir,
            full_model_path=full_model_path,
            metadata_path=metadata_path,
            num_attention_segments=metadata["num_attention_segments"],
            num_mlp_segments=metadata["num_mlp_segments"],
            num_non_segment_parameters=metadata["num_non_segment_parameters"],
            num_full_state_parameters=metadata["num_full_state_parameters"],
            metadata=metadata,
        )

    def export_from_checkpoint_file(self, checkpoint_path: str | Path) -> FullModelExportResult:
        """Load a segmented checkpoint file and export a full-model artifact."""

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Loaded checkpoint must be a mapping.")
        return self.export_from_checkpoint_dict(checkpoint)

    def _assemble_attention_layer(
        self,
        layer_id: int,
        states: list[Mapping[str, Tensor]],
    ) -> dict[str, Tensor]:
        prefix = self.config.attention_prefix_template.format(layer_id=layer_id)
        assembled: dict[str, Tensor] = {}

        for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
            tensor = _cat_required(states, name, dim=0, strict=self.config.strict)
            if tensor is not None:
                assembled[f"{prefix}.{name}"] = tensor.detach().cpu().clone()

        for name in ("q_proj.bias", "k_proj.bias", "v_proj.bias"):
            tensor = _cat_required(states, name, dim=0, strict=False)
            if tensor is not None:
                assembled[f"{prefix}.{name}"] = tensor.detach().cpu().clone()

        return assembled

    def _assemble_mlp_layer(
        self,
        layer_id: int,
        states: list[Mapping[str, Tensor]],
    ) -> dict[str, Tensor]:
        prefix = self.config.mlp_prefix_template.format(layer_id=layer_id)
        assembled: dict[str, Tensor] = {}

        fc1_weight = _cat_required(states, "fc1.weight", dim=0, strict=self.config.strict)
        fc1_bias = _cat_required(states, "fc1.bias", dim=0, strict=False)
        fc2_weight = _cat_required(states, "fc2.weight", dim=1, strict=self.config.strict)

        if fc1_weight is not None:
            assembled[f"{prefix}.fc1.weight"] = fc1_weight.detach().cpu().clone()
        if fc1_bias is not None:
            assembled[f"{prefix}.fc1.bias"] = fc1_bias.detach().cpu().clone()
        if fc2_weight is not None:
            assembled[f"{prefix}.fc2.weight"] = fc2_weight.detach().cpu().clone()

        return assembled

    # ------------------------------------------------------------------
    # Global (non-per-layer) component consolidation
    # ------------------------------------------------------------------

    def _consolidate_global_components(
        self,
        full_state: dict[str, Tensor],
        global_segments_raw: Any,
        *,
        n_layers: int,
        embedding_segments: int,
        output_head_segments: int,
    ) -> None:
        """Fold the segment-store global components into dense full-model keys.

        ``global_segments_raw`` is a mapping keyed by stable string identity:

            ``"emb.{i}"``         — embedding slice i (or the whole table at i=0)
            ``"head.{i}"``        — output-head vocab slice i (or whole head at i=0)
            ``"attn_proj.{L}"``   — per-layer attention output projection

        Each value is the segment's ``state_dict()``.  When
        ``global_segments_raw`` is empty/None, the function is a no-op (legacy
        always-resident exports already carry dense keys in ``non_segment_state``
        and need no remapping).

        The verified consolidation axes (see ``execution/forward_engine.py``):

        * Embedding slices each produce ``d_model / N_emb`` features and the
          forward concatenates along the feature (last) dim, so per-slice token
          weight is ``[vocab, d_model/N_emb]`` → ``torch.cat(dim=1)`` →
          ``[vocab, d_model]``; likewise position ``[max_seq_len, d_model]``.
        * Output-head slices each project ``d_model → vocab / N_head`` and the
          forward concatenates along the vocab (last) dim, so per-slice weight is
          ``[vocab/N_head, d_model]`` → ``torch.cat(dim=0)`` → ``[vocab, d_model]``.
          The always-resident ``output_slice_final_norm`` (LayerNorm over
          d_model) becomes ``output_head.final_norm``.
        * Attention output projection is a straight rename to
          ``attention_output_projections.projections.{L}.{weight,bias}``.
        """
        if not global_segments_raw:
            # Nothing to consolidate (legacy export path).  Still normalise the
            # sliced final-norm key if it somehow leaked in.
            self._remap_output_slice_final_norm(full_state)
            return

        if not isinstance(global_segments_raw, Mapping):
            raise TypeError("global_segments must be a mapping of identity → state_dict.")

        # Index the raw global segments by identity.
        by_kind: dict[str, dict[int, Mapping[str, Tensor]]] = {
            "emb": {}, "head": {}, "attn_proj": {}
        }
        for raw_key, state in global_segments_raw.items():
            if not isinstance(state, Mapping):
                raise TypeError(f"global_segments[{raw_key!r}] must be a mapping.")
            kind, _, idx_text = str(raw_key).rpartition(".")
            if kind not in by_kind:
                raise ValueError(f"Unrecognised global segment key: {raw_key!r}.")
            by_kind[kind][int(idx_text)] = state

        # ---- Embedding -------------------------------------------------
        emb_states = [by_kind["emb"][i] for i in sorted(by_kind["emb"])]
        if emb_states:
            token_w, pos_w = self._consolidate_embedding(emb_states, embedding_segments)
            full_state["embeddings.token_embedding.weight"] = token_w
            full_state["embeddings.position_embedding.weight"] = pos_w

        # ---- Output head ----------------------------------------------
        head_states = [by_kind["head"][i] for i in sorted(by_kind["head"])]
        if head_states:
            lm_head_w, final_norm = self._consolidate_output_head(
                head_states, output_head_segments, full_state
            )
            full_state["output_head.lm_head.weight"] = lm_head_w
            if final_norm is not None:
                full_state["output_head.final_norm.weight"] = final_norm[0]
                full_state["output_head.final_norm.bias"] = final_norm[1]

        # ---- Attention output projections -----------------------------
        for layer_id, state in by_kind["attn_proj"].items():
            w = state.get("proj.weight")
            b = state.get("proj.bias")
            if w is None:
                raise KeyError(
                    f"attention_output_proj segment for layer {layer_id} missing 'proj.weight'."
                )
            full_state[f"attention_output_projections.projections.{layer_id}.weight"] = (
                _ensure_tensor(w, name="proj.weight").detach().cpu().clone()
            )
            if b is not None:
                full_state[f"attention_output_projections.projections.{layer_id}.bias"] = (
                    _ensure_tensor(b, name="proj.bias").detach().cpu().clone()
                )

        # Strip any leftover sliced final-norm / slice-named keys.
        self._remap_output_slice_final_norm(full_state)
        for stale_prefix in (
            "embeddings.token_embedding_slice",
            "embeddings.position_embedding_slice",
            "embeddings.embeddings.",
            "output_head.linear_slice",
            "output_head.output_head.",
        ):
            for key in [k for k in full_state if k.startswith(stale_prefix)]:
                del full_state[key]
        for key in [
            k for k in full_state
            if k.startswith("layers.") and ".attention_output_proj." in k
        ]:
            del full_state[key]

    @staticmethod
    def _consolidate_embedding(
        emb_states: list[Mapping[str, Tensor]],
        embedding_segments: int,
    ) -> tuple[Tensor, Tensor]:
        """Return consolidated (token_embedding.weight, position_embedding.weight)."""
        if len(emb_states) == 1 and embedding_segments == 1:
            state = emb_states[0]
            # Whole-table module (EmbeddingSegmentModule): keys nested under
            # 'embeddings.'.  Accept both nested and bare forms.
            token_w = _first_present(
                state, "embeddings.token_embedding.weight", "token_embedding.weight"
            )
            pos_w = _first_present(
                state, "embeddings.position_embedding.weight", "position_embedding.weight"
            )
            if token_w is None or pos_w is None:
                raise KeyError("embedding segment missing token/position embedding weight.")
            return (
                _ensure_tensor(token_w, name="token_embedding").detach().cpu().clone(),
                _ensure_tensor(pos_w, name="position_embedding").detach().cpu().clone(),
            )
        # Sliced (EmbeddingSliceSegmentModule): each slice [vocab|max_seq, d/N].
        token_parts = [
            _ensure_tensor(s["token_embedding_slice.weight"], name="token_embedding_slice")
            for s in emb_states
        ]
        pos_parts = [
            _ensure_tensor(s["position_embedding_slice.weight"], name="position_embedding_slice")
            for s in emb_states
        ]
        return (
            torch.cat(token_parts, dim=1).detach().cpu().clone(),
            torch.cat(pos_parts, dim=1).detach().cpu().clone(),
        )

    @staticmethod
    def _consolidate_output_head(
        head_states: list[Mapping[str, Tensor]],
        output_head_segments: int,
        full_state: Mapping[str, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Return (lm_head.weight, optional (final_norm.weight, final_norm.bias))."""
        if len(head_states) == 1 and output_head_segments == 1:
            state = head_states[0]
            # Whole head (OutputHeadSegmentModule): nested under 'output_head.'.
            lm_w = _first_present(state, "output_head.lm_head.weight", "lm_head.weight")
            fn_w = _first_present(
                state, "output_head.final_norm.weight", "final_norm.weight"
            )
            fn_b = _first_present(
                state, "output_head.final_norm.bias", "final_norm.bias"
            )
            if lm_w is None or fn_w is None or fn_b is None:
                raise KeyError("output_head segment missing lm_head/final_norm weights.")
            return (
                _ensure_tensor(lm_w, name="lm_head").detach().cpu().clone(),
                (
                    _ensure_tensor(fn_w, name="final_norm.weight").detach().cpu().clone(),
                    _ensure_tensor(fn_b, name="final_norm.bias").detach().cpu().clone(),
                ),
            )
        # Sliced (OutputHeadSliceSegmentModule): each slice [vocab/N, d_model],
        # concatenated along vocab (dim 0).  The final norm is the always-resident
        # 'output_slice_final_norm' carried in non_segment_state.
        lm_parts = [
            _ensure_tensor(s["linear_slice.weight"], name="linear_slice")
            for s in head_states
        ]
        lm_head_w = torch.cat(lm_parts, dim=0).detach().cpu().clone()
        fn = None
        fn_w = full_state.get("output_slice_final_norm.weight")
        fn_b = full_state.get("output_slice_final_norm.bias")
        if fn_w is not None and fn_b is not None:
            fn = (
                _ensure_tensor(fn_w, name="final_norm.weight").detach().cpu().clone(),
                _ensure_tensor(fn_b, name="final_norm.bias").detach().cpu().clone(),
            )
        return lm_head_w, fn

    @staticmethod
    def _remap_output_slice_final_norm(full_state: dict[str, Tensor]) -> None:
        """Drop the sliced-final-norm keys after they have been consolidated."""
        for suffix in ("weight", "bias"):
            full_state.pop(f"output_slice_final_norm.{suffix}", None)

    @staticmethod
    def _infer_n_layers(segments: Mapping[SegmentTuple, Mapping[str, Tensor]]) -> int:
        if not segments:
            return 0
        return max(key[0] for key in segments) + 1

    @staticmethod
    def _infer_count(segments: Mapping[SegmentTuple, Mapping[str, Tensor]], segment_type: str) -> int:
        counts: dict[int, int] = {}
        for layer_id, current_type, _ in segments:
            if current_type == segment_type:
                counts[layer_id] = counts.get(layer_id, 0) + 1
        if not counts:
            return 0
        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            raise ValueError(f"Inconsistent number of {segment_type} segments across layers: {counts}")
        return unique_counts.pop()

    @staticmethod
    def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
        if yaml is not None:
            path.write_text(yaml.safe_dump(dict(metadata), sort_keys=True), encoding="utf-8")
        else:  # pragma: no cover
            path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def export_full_model_from_checkpoint(
    checkpoint: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    strict: bool = True,
) -> FullModelExportResult:
    """Convenience function for full-model export."""

    exporter = FullModelExporter(
        FullModelExportConfig(output_dir=output_dir, strict=strict)
    )
    if isinstance(checkpoint, (str, Path)):
        return exporter.export_from_checkpoint_file(checkpoint)
    return exporter.export_from_checkpoint_dict(checkpoint)

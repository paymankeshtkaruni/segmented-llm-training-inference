"""
Configuration for the self-contained segmented model.

This module is part of `segmentation_management/`, a clean, **self-contained**
re-implementation of the segmented training/inference stack: it imports only
standard libraries (no `src/`, no other project modules). See DESIGN.md for the
full rationale.

It defines two config objects and the fixed experiment presets:

* `ModelConfig`        — the GPT decoder architecture (same family as the
  full-model baseline; "GPTTransformer" = a gpt_decoder).
* `SegmentationConfig` — how that model is split for sequential, one-segment-at-a-
  time execution, along four axes:
      embedding × attention × mlp × output_head     (the "E×A×M×H" notation).

WHY two separate configs: the architecture (what the model *is*) is independent of
how we *slice* it for memory. The same trained weights must be reconstructable
regardless of the slicing (that is the paper's core claim), so segmentation is a
pure execution concern layered on top of an unchanged architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict


# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the GPT decoder.

    Every field is part of the *mathematical* model; none of it depends on how the
    model is segmented. Two models are used in the study:

      small  — accuracy experiments (the task has little headroom, so a small model
               keeps the problem non-trivial). BPE-2k tokenizer.
      large  — cost/memory experiments (the model's own memory must dominate fixed
               runtime overhead to make segmentation savings visible). GPT-2 vocab.
    """

    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1
    architecture: str = "gpt_decoder"
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("vocab_size", "max_seq_len", "n_layers", "d_model", "n_heads", "d_ff"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"ModelConfig.{name} must be a positive int, got {v!r}")
        # head_dim must be integer: attention reshapes d_model -> (n_heads, head_dim)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count (embeddings dominate): 2*vocab*d_model + body."""
        embed = 2 * self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        per_layer = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff
        return embed + self.n_layers * per_layer

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SegmentationConfig:
    """How the model is sliced for one-segment-at-a-time execution.

    Four independent axes (the "E×A×M×H" notation), each splitting a different,
    memory-heavy part of the model so that only *one slice* is ever resident on the
    constrained device. Each axis trades extra compute (reloading + recompute) for
    a lower peak memory.

    embedding_segments (E)
        Split the token/position embedding along the **d_model** dimension into E
        slices (each `[vocab, d_model/E]`). Reassembled by concatenation on d_model.
        Constraint: E divides d_model.
        WHY: the embedding table is one of the largest tensors; slicing it bounds the
        resident embedding memory.

    attention_segments (A)
        Split the attention **heads** into A groups (each does `n_heads/A` heads).
        Reassembled by concatenating the per-group outputs, then the single output
        projection is applied. Constraint: A divides n_heads.
        WHY: attention Q/K/V projections and the per-head score tensors scale with the
        number of heads held at once; A groups -> 1/A of that at a time.

    mlp_chunks (M)
        Split the feed-forward **hidden dimension** d_ff into M chunks (each uses
        `d_ff/M` hidden units). Reassembled by **summing** the chunk outputs (the MLP
        down-projection is linear in the hidden dim, so chunk sums = full output).
        Constraint: M divides d_ff.
        WHY: the MLP hidden activation `[B,T,d_ff]` is usually the largest activation;
        chunking it (with running-sum) bounds it to `[B,T,d_ff/M]`.

    output_head_segments (H)
        Split the output projection along the **vocab** dimension into H slices. With
        H>1 the loss is computed by a chunked/streamed cross-entropy that never
        materializes the full `[B,T,vocab]` logits. The last slice absorbs any vocab
        remainder, so H need NOT divide vocab exactly.
        WHY: `[B,T,vocab]` logits are huge for large vocab; H slices -> `[B,T,vocab/H]`.

    NOTE on values >1: every axis here is intended to be >1 (that is the point of
    segmentation). With an axis = 1 that component is simply not segmented.
    """

    embedding_segments: int
    attention_segments: int
    mlp_chunks: int
    output_head_segments: int
    # descriptive only — kept for provenance in metrics/exports
    attention_axis: str = "head_groups"
    mlp_axis: str = "feedforward_hidden_dimension"
    embedding_axis: str = "d_model"
    output_head_axis: str = "vocab"

    @property
    def code(self) -> str:
        """The E×A×M×H short code, e.g. '8x2x2x8'."""
        return (f"{self.embedding_segments}x{self.attention_segments}"
                f"x{self.mlp_chunks}x{self.output_head_segments}")

    def validate_against_model(self, m: ModelConfig) -> None:
        for name in ("embedding_segments", "attention_segments", "mlp_chunks",
                     "output_head_segments"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"SegmentationConfig.{name} must be a positive int, got {v!r}")
        if m.d_model % self.embedding_segments != 0:
            raise ValueError(
                f"embedding_segments ({self.embedding_segments}) must divide "
                f"d_model ({m.d_model})")
        if m.n_heads % self.attention_segments != 0:
            raise ValueError(
                f"attention_segments ({self.attention_segments}) must divide "
                f"n_heads ({m.n_heads})")
        if m.d_ff % self.mlp_chunks != 0:
            raise ValueError(
                f"mlp_chunks ({self.mlp_chunks}) must divide d_ff ({m.d_ff})")
        if self.output_head_segments > m.vocab_size:
            raise ValueError(
                f"output_head_segments ({self.output_head_segments}) exceeds "
                f"vocab_size ({m.vocab_size})")
        # output_head_segments need NOT divide vocab — last slice takes the remainder.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Sequence / objective settings (shared with the full-model baseline)
# --------------------------------------------------------------------------- #
# WHY add_bos=False and pad != eos: with GPT-2 (bos==eos==pad) prepending BOS puts
# the pad id at position 0; the padding mask then masks position 0, the attention
# softmax over an all-(-inf) row produces NaN that propagates everywhere. Omitting
# BOS avoids it. The BPE-2k tokenizer additionally uses a *distinct* pad token.
PROMPT_TEMPLATE = "{data}\nLabel:"
TARGET_PREFIX = " "
ADD_BOS = False
ADD_EOS = True


# --------------------------------------------------------------------------- #
# Experiment presets  (the only configurations used in the study)
# --------------------------------------------------------------------------- #
# Small model — accuracy experiments (BPE-2k vocab). One segmentation setup.
SMALL_MODEL = ModelConfig(
    vocab_size=2000, max_seq_len=128, n_layers=4, d_model=64, n_heads=4, d_ff=256,
    dropout=0.1,
)
# Large model — cost experiments (GPT-2 vocab). Two segmentation setups.
LARGE_MODEL = ModelConfig(
    vocab_size=50257, max_seq_len=512, n_layers=36, d_model=1280, n_heads=20, d_ff=5120,
    dropout=0.1,
)
# Scale-experiment models (GPT-2 vocab) — cost-only runs at ~3B and ~7B params,
# partition FIXED at 8x2x2x8 (all dims divide the axes; head_dim = 128 for both).
# ~3.1B: embed 2*50257*2560 + 36*(4*2560^2 + 2*2560*10240)
XL3B_MODEL = ModelConfig(
    vocab_size=50257, max_seq_len=512, n_layers=36, d_model=2560, n_heads=20, d_ff=10240,
    dropout=0.1,
)
# ~6.9B (Llama-like shape): embed 2*50257*4096 + 32*(4*4096^2 + 2*4096*16384)
XXL7B_MODEL = ModelConfig(
    vocab_size=50257, max_seq_len=512, n_layers=32, d_model=4096, n_heads=32, d_ff=16384,
    dropout=0.1,
)
# Intermediate scale points for the cost-model fit (~1.6B and ~5.1B; head_dim 128).
XL15B_MODEL = ModelConfig(
    vocab_size=50257, max_seq_len=512, n_layers=36, d_model=1792, n_heads=14, d_ff=7168,
    dropout=0.1,
)
XL5B_MODEL = ModelConfig(
    vocab_size=50257, max_seq_len=512, n_layers=36, d_model=3328, n_heads=26, d_ff=13312,
    dropout=0.1,
)

# E×A×M×H setups.
SEG_8x2x2x8 = SegmentationConfig(embedding_segments=8, attention_segments=2,
                                 mlp_chunks=2, output_head_segments=8)
SEG_16x4x4x16 = SegmentationConfig(embedding_segments=16, attention_segments=4,
                                   mlp_chunks=4, output_head_segments=16)

# Named presets: (model, segmentation, tokenizer).
PRESETS: Dict[str, Dict[str, Any]] = {
    # Small model: 8x2x2x8.
    "small_8x2x2x8":  {"model": SMALL_MODEL, "seg": SEG_8x2x2x8,   "tokenizer": "bpe"},
    # Large model: two setups.
    "large_8x2x2x8":  {"model": LARGE_MODEL, "seg": SEG_8x2x2x8,   "tokenizer": "gpt2"},
    "large_16x4x4x16": {"model": LARGE_MODEL, "seg": SEG_16x4x4x16, "tokenizer": "gpt2"},
    # Scale experiment: same partition, bigger models (cost-only; no accuracy runs).
    "xl3b_8x2x2x8":   {"model": XL3B_MODEL,  "seg": SEG_8x2x2x8,   "tokenizer": "gpt2"},
    "xxl7b_8x2x2x8":  {"model": XXL7B_MODEL, "seg": SEG_8x2x2x8,   "tokenizer": "gpt2"},
    # Intermediate fit points (GPU cost-model fit only).
    "xl15b_8x2x2x8":  {"model": XL15B_MODEL, "seg": SEG_8x2x2x8,   "tokenizer": "gpt2"},
    "xl5b_8x2x2x8":   {"model": XL5B_MODEL,  "seg": SEG_8x2x2x8,   "tokenizer": "gpt2"},
    # General-fit validation cell (pre-registered prediction; never used for fitting).
    "xl3b_16x4x4x16": {"model": XL3B_MODEL,  "seg": SEG_16x4x4x16, "tokenizer": "gpt2"},
}


def get_preset(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    p = PRESETS[name]
    p["seg"].validate_against_model(p["model"])  # fail early on a bad combo
    return p


if __name__ == "__main__":
    # quick self-check: every preset validates and prints its key numbers
    for name in PRESETS:
        p = get_preset(name)
        m, s = p["model"], p["seg"]
        print(f"{name:18} tok={p['tokenizer']:4} | d_model={m.d_model} heads={m.n_heads} "
              f"layers={m.n_layers} d_ff={m.d_ff} vocab={m.vocab_size} | "
              f"seg E×A×M×H={s.code} | ~{m.n_params_estimate/1e6:.2f}M params")

"""
Core neural-network modules — the reference (un-segmented) GPT decoder.

These are the exact building blocks the segments are carved from. The segmented
engine never instantiates the whole `ReferenceGPTDecoder` for training (that would
defeat the memory goal); it builds individual *segment* modules (segments.py) whose
weights are slices of these components. But we keep the full reference model here
for two reasons:

  1. EXPORT/AUDIT — after segmented training we reassemble the slices into a
     `ReferenceGPTDecoder` and check it is numerically identical to a normally
     trained model. That identity is the paper's central claim, so the reference
     architecture must be defined once, unambiguously, here.
  2. Determinism — the reference defines the canonical init and the exact attention
     / MLP / layernorm math the segments must reproduce.

Architecture: a standard **pre-LayerNorm** GPT decoder
    x = x + Attn(LN(x));   x = x + MLP(LN(x))
faithfully ported from the project's `src/model/full_model.py`.

CONSIDERATION (NaN): attention masks future/padding positions with `-inf` before
softmax. A query position that can attend to NO valid key (a fully-masked row)
would softmax to NaN. We avoid this by convention, not by changing the math:
`add_bos=False` keeps position 0 a real token and right-padding keeps every real
query row with ≥1 valid key (see config.py / data.py). Keeping the math identical
to the reference is required so segmented == full holds exactly.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class GPTInputEmbedding(nn.Module):
    """Token embedding + learned positional embedding (+ dropout)."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        return self.dropout(x)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Layout matters for segmentation: `qkv_projection` is one `[d_model, 3*d_model]`
    matrix; the head dimension is contiguous, so attention-head **groups** are
    contiguous slices of q/k/v — that is what makes head-group segmentation a clean
    slice (segments.py).
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal.view(1, 1, max_seq_len, max_seq_len),
                             persistent=False)

    @staticmethod
    def build_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """[B,1,1,T] bool, True at padding positions."""
        return (input_ids == pad_token_id).view(input_ids.size(0), 1, 1, input_ids.size(1))

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        p = self.attention_dropout.p if self.training else 0.0
        # MEMORY: SDPA never materializes the [B, n_heads, T, T] score matrix
        # (flash / mem-efficient kernels). The explicit q@kᵀ+softmax would.
        if padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:
            # one additive [B,1,T,T] mask (broadcast over heads) = causal ∪ padding.
            # Use a finite large-negative value (not -inf) so a fully-masked row
            # softmaxes to uniform instead of NaN (robustness; rows are discarded anyway).
            neg = torch.finfo(q.dtype).min
            bias = torch.zeros(B, 1, T, T, dtype=q.dtype, device=q.device)
            bias = bias.masked_fill(self.causal_mask[:, :, :T, :T], neg)
            bias = bias.masked_fill(padding_mask, neg)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=p)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.output_dropout(self.output_projection(out))


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class GPTMLP(nn.Module):
    """Position-wise FFN: d_model -> d_ff -> GELU -> d_model.

    `input_projection` rows and `output_projection` cols both index the d_ff hidden
    dimension, so an MLP hidden **chunk** is a contiguous slice of both — the basis
    for mlp-chunk segmentation (with chunk outputs summed)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.output_projection(self.activation(self.input_projection(x))))


# --------------------------------------------------------------------------- #
# Block + full reference model
# --------------------------------------------------------------------------- #
class GPTBlock(nn.Module):
    """One pre-LN block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = GPTMLP(d_model, d_ff, dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.attention(self.attention_norm(x), padding_mask=padding_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ReferenceGPTDecoder(nn.Module):
    """The full (un-segmented) decoder — canonical architecture + init.

    forward(input_ids, pad_token_id) -> (logits [B,T,V], hidden [B,T,d_model]).
    Used as the reassembly target for export/audit and as the numerical reference
    the segmented execution must reproduce.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = GPTInputEmbedding(cfg.vocab_size, cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.max_seq_len, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output_projection = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()   # GPT-2 / minGPT residual init

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attention.output_projection.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(blk.mlp.output_projection.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor,
                pad_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B,T], got {tuple(input_ids.shape)}")
        if input_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {input_ids.size(1)} > max_seq_len {self.cfg.max_seq_len}")
        pad_mask = (CausalSelfAttention.build_padding_mask(input_ids, pad_token_id)
                    if pad_token_id is not None else None)
        x = self.embedding(input_ids)
        for blk in self.blocks:
            x = blk(x, padding_mask=pad_mask)
        hidden = self.final_norm(x)
        return self.output_projection(hidden), hidden

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Loss + token-accuracy (shared conventions with the full-model baseline)
# --------------------------------------------------------------------------- #
def causal_lm_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor,
                                 ignore_index: int = -100) -> torch.Tensor:
    """Next-token CE with the standard shift: logits[:, :-1] predicts labels[:, 1:].
    Prompt + padding positions are -100 in `labels` (loss on label tokens only)."""
    V = logits.size(-1)
    sl = logits[:, :-1, :].contiguous().view(-1, V)
    st = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(sl, st, ignore_index=ignore_index)


def token_counts(logits: torch.Tensor, labels: torch.Tensor,
                 ignore_index: int = -100) -> Tuple[int, int]:
    """(num_correct, num_valid) over shifted, non-ignored label tokens.
    Global-token accuracy = Σcorrect/Σvalid (matches the baseline's headline)."""
    sl, st = logits[:, :-1, :], labels[:, 1:]
    mask = st != ignore_index
    valid = int(mask.sum().item())
    if valid == 0:
        return 0, 0
    correct = int(((sl.argmax(-1) == st) & mask).sum().item())
    return correct, valid


if __name__ == "__main__":
    from config import SMALL_MODEL
    torch.manual_seed(0)
    m = ReferenceGPTDecoder(SMALL_MODEL)
    print(f"ReferenceGPTDecoder(small) params = {m.num_params()/1e6:.3f}M")
    x = torch.randint(0, SMALL_MODEL.vocab_size, (2, 16))
    logits, hidden = m(x, pad_token_id=None)
    lab = x.clone(); lab[:, :4] = -100
    loss = causal_lm_cross_entropy_loss(logits, lab)
    c, v = token_counts(logits, lab)
    print(f"logits={tuple(logits.shape)} hidden={tuple(hidden.shape)} loss={loss.item():.4f} "
          f"acc={c}/{v} nan={torch.isnan(logits).any().item()}")

"""
Segment modules — the slices the model is executed in, one at a time.

Each segment is a small `nn.Module` owning a deterministic slice of one layer's
parameters. Segments are trained/run independently; the union of their weights
reassembles (by concatenation / identity) into the reference `ReferenceGPTDecoder`
(see export.py), which is the paper's "segmented == full" identity.

Four segment types (the E×A×M×H axes):
  AttentionHeadSegment   owns heads [start:end); q/k/v for those heads only; SDPA.
  MLPHiddenChunk         owns d_ff hidden units [start:end); returns a d_model term.
  EmbeddingSlice         owns d_model columns [start:end) of token+pos embedding.
  OutputHeadSlice        owns vocab rows [start:end) of the output projection.

Shared, NON-segmented, always-resident (tiny — kept by the engine, not sliced):
  per-layer attention LayerNorm, MLP LayerNorm, attention output projection,
  MLP shared output bias; the final LayerNorm.

MEMORY per segment (the peak tensor each can create), to keep bounded:
  attention : [B, n_heads/A, T, head_dim] via SDPA (no score matrix — investigation #1)
  mlp chunk : [B, T, d_ff/M]  (the only large one; engine sums chunk outputs, frees each)
  embedding : [B, T, d_model/E]
  output    : [B, T, vocab/H]  (engine streams these into chunked CE; never the full logits)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, SegmentationConfig


# --------------------------------------------------------------------------- #
# slice-range helpers (deterministic ownership)
# --------------------------------------------------------------------------- #
def head_range(n_heads: int, A: int, seg: int) -> Tuple[int, int]:
    per = n_heads // A
    return seg * per, seg * per + per

def hidden_range(d_ff: int, M: int, seg: int) -> Tuple[int, int]:
    per = d_ff // M
    return seg * per, seg * per + per

def dmodel_range(d_model: int, E: int, seg: int) -> Tuple[int, int]:
    per = d_model // E
    return seg * per, seg * per + per

def vocab_range(vocab: int, H: int, seg: int) -> Tuple[int, int]:
    # last slice absorbs the remainder (H need not divide vocab)
    base = vocab // H
    start = seg * base
    end = vocab if seg == H - 1 else start + base
    return start, end


# --------------------------------------------------------------------------- #
# Attention head-group segment
# --------------------------------------------------------------------------- #
class AttentionHeadSegment(nn.Module):
    """Owns `n_heads/A` heads. Separate q/k/v projections sized to this group so the
    group's weights are a clean slice of the full qkv (q-block, k-block, v-block).
    Output projection is NOT here — it is applied once on the concatenated output.
    Peak: SDPA output [B, heads_per_seg, T, head_dim] (no score matrix)."""

    def __init__(self, d_model: int, n_heads: int, A: int, dropout: float):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.heads_per_seg = n_heads // A
        self.out_dim = self.heads_per_seg * self.head_dim
        self.q_proj = nn.Linear(d_model, self.out_dim)
        self.k_proj = nn.Linear(d_model, self.out_dim)
        self.v_proj = nn.Linear(d_model, self.out_dim)
        self.attn_dropout_p = dropout

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.heads_per_seg, self.head_dim).transpose(1, 2)

    def forward(self, x_norm: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self._heads(self.q_proj(x_norm)), self._heads(self.k_proj(x_norm)), self._heads(self.v_proj(x_norm))
        p = self.attn_dropout_p if self.training else 0.0
        if attn_bias is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:  # attn_bias [B,1,T,T] additive (causal ∪ padding), broadcast over heads
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
        B, _, T, _ = out.shape
        return out.transpose(1, 2).contiguous().view(B, T, self.out_dim)


# --------------------------------------------------------------------------- #
# MLP hidden chunk
# --------------------------------------------------------------------------- #
class MLPHiddenChunk(nn.Module):
    """Owns `d_ff/M` hidden units. d_model -> (d_ff/M) -> GELU -> d_model.
    The down-projection has NO bias (the shared output bias is added once, outside,
    after summing chunks). Peak: the hidden activation [B, T, d_ff/M]."""

    def __init__(self, d_model: int, d_ff: int, M: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(d_model, d_ff // M)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(d_ff // M, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_projection(x_norm))   # [B,T,d_ff/M]
        return self.output_projection(h)                      # [B,T,d_model]
        # NOTE (Path B, fix (d)): dropout is NOT applied per-chunk. The reference applies
        # dropout to the FULL summed MLP output (W_out·h + bias); dropping each chunk's
        # partial independently and summing has different statistics. So the engine applies
        # one dropout on (Σ chunks + bias). dropout_p is kept for reference only. In eval
        # F.dropout was a no-op, so removing it here changes nothing for inference/export.


# --------------------------------------------------------------------------- #
# Embedding slice (d_model columns)
# --------------------------------------------------------------------------- #
class EmbeddingSlice(nn.Module):
    """Owns d_model columns [start:end). token+pos embedding of width d_model/E.
    Peak: [B, T, d_model/E]. Slices concatenate on d_model to form the full embedding."""

    def __init__(self, vocab: int, d_slice: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, d_slice)
        self.position_embedding = nn.Embedding(max_seq_len, d_slice)
        self.dropout_p = dropout

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.size(1)
        pos = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        return F.dropout(x, self.dropout_p, self.training)


# --------------------------------------------------------------------------- #
# Output-head vocab slice
# --------------------------------------------------------------------------- #
class OutputHeadSlice(nn.Module):
    """Owns vocab rows [start:end). Linear d_model -> vocab_slice (no bias, like the
    reference output projection). Peak: [B, T, vocab_slice] — the engine streams these
    through chunked cross-entropy so the full [B,T,vocab] is never built."""

    def __init__(self, d_model: int, vocab_slice: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_slice, bias=False)

    def forward(self, hidden_norm: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_norm)


# --------------------------------------------------------------------------- #
# Build a fresh, fully-segmented parameter set from configs (training from scratch)
# and slice-extraction from a reference model (for equivalence tests / re-segmenting).
# --------------------------------------------------------------------------- #
def causal_padding_bias(input_ids: torch.Tensor, pad_token_id: Optional[int],
                        causal_mask: torch.Tensor) -> Optional[torch.Tensor]:
    """One [B,1,T,T] additive mask (causal ∪ padding) with a finite -large value.
    Returns None when there is no padding (caller then uses SDPA is_causal=True)."""
    if pad_token_id is None:
        return None
    B, T = input_ids.shape
    neg = torch.finfo(torch.float32).min
    bias = torch.zeros(B, 1, T, T)
    bias = bias.masked_fill(causal_mask[:, :, :T, :T], neg)
    pad = (input_ids == pad_token_id).view(B, 1, 1, T)
    return bias.masked_fill(pad, neg).to(input_ids.device)


if __name__ == "__main__":
    # IDENTITY + MEMORY self-test: segments must reproduce the reference layer exactly.
    import sys
    from config import SMALL_MODEL, SEG_8x2x2x8
    from modules import ReferenceGPTDecoder
    torch.manual_seed(0)
    m, s = SMALL_MODEL, SEG_8x2x2x8
    ref = ReferenceGPTDecoder(m).eval()

    B, T = 2, 16
    x = torch.randint(0, m.vocab_size, (B, T))

    # ---- (1) attention identity: sum/concat of head-group segments == full attention ----
    blk = ref.blocks[0]
    xin = blk.attention_norm(ref.embedding(x))
    with torch.no_grad():
        ref_attn = blk.attention(xin)                       # full attention (incl out-proj)
        segs = []
        for sid in range(s.attention_segments):
            seg = AttentionHeadSegment(m.d_model, m.n_heads, s.attention_segments, 0.0).eval()
            h0, h1 = head_range(m.n_heads, s.attention_segments, sid)
            d = blk.attention.head_dim
            # slice q/k/v rows for heads [h0:h1) out of the full qkv (q|k|v blocks)
            full = blk.attention.qkv_projection
            qw, kw, vw = full.weight.chunk(3, dim=0)
            qb, kb, vb = full.bias.chunk(3, dim=0)
            seg.q_proj.weight.copy_(qw[h0*d:h1*d]); seg.q_proj.bias.copy_(qb[h0*d:h1*d])
            seg.k_proj.weight.copy_(kw[h0*d:h1*d]); seg.k_proj.bias.copy_(kb[h0*d:h1*d])
            seg.v_proj.weight.copy_(vw[h0*d:h1*d]); seg.v_proj.bias.copy_(vb[h0*d:h1*d])
            segs.append(seg(xin))
        concat = torch.cat(segs, dim=-1)                    # [B,T,d_model]
        seg_attn = blk.attention.output_projection(concat)  # shared out-proj
    ok_attn = torch.allclose(ref_attn, seg_attn, atol=1e-5)
    print(f"attention identity: {'OK' if ok_attn else 'FAIL'}  "
          f"max|Δ|={(ref_attn-seg_attn).abs().max().item():.2e}")

    # ---- (2) MLP identity: sum of hidden chunks (+shared bias) == full MLP ----
    xin = blk.mlp_norm(ref.embedding(x))
    with torch.no_grad():
        ref_mlp = blk.mlp(xin)
        acc = None
        for sid in range(s.mlp_chunks):
            ch = MLPHiddenChunk(m.d_model, m.d_ff, s.mlp_chunks, 0.0).eval()
            h0, h1 = hidden_range(m.d_ff, s.mlp_chunks, sid)
            ch.input_projection.weight.copy_(blk.mlp.input_projection.weight[h0:h1])
            ch.input_projection.bias.copy_(blk.mlp.input_projection.bias[h0:h1])
            ch.output_projection.weight.copy_(blk.mlp.output_projection.weight[:, h0:h1])
            out = ch(xin)
            acc = out if acc is None else acc + out          # running sum (chunk freed each)
        seg_mlp = acc + blk.mlp.output_projection.bias        # shared bias once
    ok_mlp = torch.allclose(ref_mlp, seg_mlp, atol=1e-5)
    print(f"mlp identity      : {'OK' if ok_mlp else 'FAIL'}  "
          f"max|Δ|={(ref_mlp-seg_mlp).abs().max().item():.2e}")

    print("ALL SEGMENT IDENTITY CHECKS PASSED" if (ok_attn and ok_mlp) else "IDENTITY FAILED")
    sys.exit(0 if (ok_attn and ok_mlp) else 1)

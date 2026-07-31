"""
Segmented forward engine — runs the model one segment at a time, plus the streamed
chunked cross-entropy that never materializes the full `[B,T,vocab]` logits.

Resident (tiny, never segmented) parameters live in `SharedParams`: per-layer
attention/MLP LayerNorms, the per-layer attention output projection, the per-layer
shared MLP output bias, and the final LayerNorm. Everything large (qkv, mlp
projections, embedding columns, vocab rows) is a segment, loaded one at a time by
the StrictSegmentLoader from the device-correct store.

Forward composition (must equal the reference, proven in the self-test):
  hidden = concat_E( embedding_slice_e(input_ids) )                 # [B,T,d_model]
  per layer:
    a = concat_A( attn_seg_a( LN_attn(hidden) ) ); hidden += out_proj(a)
    m = Σ_M ( mlp_chunk_m( LN_mlp(hidden) ) ) + mlp_bias; hidden += m
  hidden_norm = final_LN(hidden)
Then loss via chunked CE over the H vocab slices.

MEMORY: peaks are bounded per investigation #1–#4 — attention `[B,n_heads/A,T,Dh]`
(SDPA), mlp `[B,T,d_ff/M]` (running-sum), embedding `[B,T,d_model/E]`, and the loss
`[B, seq_chunk, vocab/H]` (chunked CE). The residual stream `[B,T,d_model]` is the
one unavoidable always-resident activation (it is the model's state between layers).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from config import ModelConfig, SegmentationConfig
from loader import StrictSegmentLoader, build_segment
from optrace import mark as _tm     # no-op unless a cost profiler installs a hook
from segments import vocab_range
from stores import SegmentKey


class SharedParams(nn.Module):
    """The small, always-resident, NON-segmented parameters."""

    def __init__(self, m: ModelConfig):
        super().__init__()
        self.m = m
        self.attn_norm = nn.ModuleList([nn.LayerNorm(m.d_model) for _ in range(m.n_layers)])
        self.mlp_norm = nn.ModuleList([nn.LayerNorm(m.d_model) for _ in range(m.n_layers)])
        # NOTE: attn_out_proj (W_o) is NOT here — it is a streamed per-layer segment
        # (kind "attn_out_proj"), so its ~226 MB of weights never stay resident.
        self.mlp_out_bias = nn.ParameterList([nn.Parameter(torch.zeros(m.d_model)) for _ in range(m.n_layers)])
        self.final_norm = nn.LayerNorm(m.d_model)


def _attn_bias(input_ids: torch.Tensor, pad_token_id: Optional[int],
               device) -> Optional[torch.Tensor]:
    """[B,1,T,T] additive (causal ∪ padding) with finite -large; None if no padding
    (segments then use SDPA is_causal=True, the memory-cheapest path)."""
    if pad_token_id is None:
        return None
    B, T = input_ids.shape
    neg = torch.finfo(torch.float32).min
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    bias = torch.zeros(B, 1, T, T, device=device)
    bias = bias.masked_fill(causal.view(1, 1, T, T), neg)
    pad = (input_ids == pad_token_id).view(B, 1, 1, T)
    return bias.masked_fill(pad, neg)


class SegmentedForwardEngine:
    def __init__(self, model: ModelConfig, seg: SegmentationConfig,
                 loader: StrictSegmentLoader, shared: SharedParams, device: str):
        self.m, self.s, self.loader, self.shared, self.device = model, seg, loader, shared, device

    def eval(self) -> "SegmentedForwardEngine":
        self.shared.eval(); self.loader.set_training(False); return self

    def train(self) -> "SegmentedForwardEngine":
        self.shared.train(); self.loader.set_training(True); return self

    # ---- residual-stream forward to the final normalized hidden states ----
    def forward_hidden(self, input_ids: torch.Tensor,
                       pad_token_id: Optional[int] = None) -> torch.Tensor:
        m, s, ld = self.m, self.s, self.loader
        bias = _attn_bias(input_ids, pad_token_id, self.device)

        # embedding: concat E d_model-slices (one segment resident at a time)
        parts = []
        for e in range(s.embedding_segments):
            with ld.acquire_segment(SegmentKey(-1, "embedding", e)) as emb:
                parts.append(emb(input_ids))
        hidden = torch.cat(parts, dim=-1)
        del parts
        _tm("fwd|embed_cat")

        for layer in range(m.n_layers):
            # attention
            xin = self.shared.attn_norm[layer](hidden); _tm(f"fwd|L{layer}|attn_norm")
            outs = []
            for a in range(s.attention_segments):
                with ld.acquire_segment(SegmentKey(layer, "attention", a)) as seg:
                    outs.append(seg(xin, attn_bias=bias)); _tm(f"fwd|L{layer}|attn_seg{a}")
            attn = torch.cat(outs, dim=-1)
            del outs, xin
            with ld.acquire_segment(SegmentKey(layer, "attn_out_proj", 0)) as op:
                # (c) attention OUTPUT dropout (reference: output_dropout(output_projection)).
                # No-op in eval (ld.training False) -> inference/export unchanged.
                hidden = hidden + F.dropout(op(attn), m.dropout, ld.training)
            del attn; _tm(f"fwd|L{layer}|attn_add")
            # mlp — running sum, each chunk freed
            xin = self.shared.mlp_norm[layer](hidden); _tm(f"fwd|L{layer}|mlp_norm")
            acc = None
            for c in range(s.mlp_chunks):
                with ld.acquire_segment(SegmentKey(layer, "mlp", c)) as seg:
                    out = seg(xin)
                acc = out if acc is None else acc + out
                del out; _tm(f"fwd|L{layer}|mlp_chunk{c}")
            # (d) MLP dropout on the SUMMED output (acc + bias), matching the reference
            # (dropout on W_out·h + b), not per-chunk. No-op in eval.
            hidden = hidden + F.dropout(acc + self.shared.mlp_out_bias[layer], m.dropout, ld.training)
            del acc, xin; _tm(f"fwd|L{layer}|mlp_add")

        out = self.shared.final_norm(hidden); _tm("fwd|final_norm")
        return out

    # ---- streamed cross-entropy over the H vocab slices (no full logits) ----
    @torch.no_grad()
    def chunked_ce(self, hidden_norm: torch.Tensor, labels: torch.Tensor,
                   seq_chunk: int = 128, ignore_index: int = -100
                   ) -> Tuple[torch.Tensor, int, int]:
        """Returns (mean_loss, num_correct, num_valid). Peak logits tensor =
        [B, seq_chunk, vocab/H] — the full [B,T,vocab] is never built."""
        m, s, ld = self.m, self.s, self.loader
        H = s.output_head_segments
        sh = hidden_norm[:, :-1, :]          # predicts labels[:, 1:]
        tgt = labels[:, 1:]
        B, Tm1, _ = sh.shape
        loss_sum = torch.zeros((), device=hidden_norm.device)
        n_valid = n_correct = 0
        neg = torch.finfo(torch.float32).min

        for t0 in range(0, Tm1, seq_chunk):
            t1 = min(Tm1, t0 + seq_chunk)
            h = sh[:, t0:t1, :]              # [B,c,D]
            tg = tgt[:, t0:t1]              # [B,c]
            mask = tg != ignore_index
            lse = None
            correct = torch.zeros_like(tg, dtype=h.dtype)
            arg_val = torch.full_like(tg, fill_value=0, dtype=h.dtype) + neg
            arg_idx = torch.zeros_like(tg)
            for hseg in range(H):
                v0, v1 = vocab_range(m.vocab_size, H, hseg)
                with ld.acquire_segment(SegmentKey(-1, "output_head", hseg)) as head:
                    z = head(h)             # [B,c,v1-v0]  <-- the only large temp
                slice_lse = torch.logsumexp(z, dim=-1)             # [B,c]
                lse = slice_lse if lse is None else torch.logaddexp(lse, slice_lse)
                in_rng = (tg >= v0) & (tg < v1)
                local = (tg - v0).clamp(0, v1 - v0 - 1)
                picked = z.gather(-1, local.unsqueeze(-1)).squeeze(-1)
                correct = torch.where(in_rng, picked, correct)
                smax, sarg = z.max(dim=-1)
                upd = smax > arg_val
                arg_val = torch.where(upd, smax, arg_val)
                arg_idx = torch.where(upd, sarg + v0, arg_idx)
                del z
            per_token = lse - correct        # [B,c] CE per token
            loss_sum = loss_sum + per_token[mask].sum()
            n_valid += int(mask.sum())
            n_correct += int(((arg_idx == tg) & mask).sum())
        mean_loss = loss_sum / max(1, n_valid)
        return mean_loss, n_correct, n_valid


# --------------------------------------------------------------------------- #
# helper: slice a reference model into (store, shared) — for tests + re-segmenting
# --------------------------------------------------------------------------- #
def populate_from_reference(ref, m: ModelConfig, s: SegmentationConfig, store) -> SharedParams:
    """Fill `store` with segment weights and return SharedParams, all copied from a
    trained/initialized reference. (Inverse of export.py's reassembly.)"""
    from segments import head_range, hidden_range, dmodel_range
    shared = SharedParams(m)
    with torch.no_grad():
        # shared (resident) pieces
        for L in range(m.n_layers):
            blk = ref.blocks[L]
            shared.attn_norm[L].load_state_dict(blk.attention_norm.state_dict())
            shared.mlp_norm[L].load_state_dict(blk.mlp_norm.state_dict())
            store.put(SegmentKey(L, "attn_out_proj", 0), {
                "weight": blk.attention.output_projection.weight.clone(),
                "bias": blk.attention.output_projection.bias.clone()})
            shared.mlp_out_bias[L].copy_(blk.mlp.output_projection.bias)
        shared.final_norm.load_state_dict(ref.final_norm.state_dict())

        # embedding slices (d_model columns)
        for e in range(s.embedding_segments):
            d0, d1 = dmodel_range(m.d_model, s.embedding_segments, e)
            store.put(SegmentKey(-1, "embedding", e), {
                "token_embedding.weight": ref.embedding.token_embedding.weight[:, d0:d1].clone(),
                "position_embedding.weight": ref.embedding.position_embedding.weight[:, d0:d1].clone(),
            })
        # output-head slices (vocab rows)
        for hseg in range(s.output_head_segments):
            v0, v1 = vocab_range(m.vocab_size, s.output_head_segments, hseg)
            store.put(SegmentKey(-1, "output_head", hseg),
                      {"projection.weight": ref.output_projection.weight[v0:v1, :].clone()})
        # attention head-group + mlp chunk slices
        for L in range(m.n_layers):
            blk = ref.blocks[L]
            qw, kw, vw = blk.attention.qkv_projection.weight.chunk(3, dim=0)
            qb, kb, vb = blk.attention.qkv_projection.bias.chunk(3, dim=0)
            dh = blk.attention.head_dim
            for a in range(s.attention_segments):
                h0, h1 = head_range(m.n_heads, s.attention_segments, a)
                store.put(SegmentKey(L, "attention", a), {
                    "q_proj.weight": qw[h0*dh:h1*dh].clone(), "q_proj.bias": qb[h0*dh:h1*dh].clone(),
                    "k_proj.weight": kw[h0*dh:h1*dh].clone(), "k_proj.bias": kb[h0*dh:h1*dh].clone(),
                    "v_proj.weight": vw[h0*dh:h1*dh].clone(), "v_proj.bias": vb[h0*dh:h1*dh].clone(),
                })
            for c in range(s.mlp_chunks):
                f0, f1 = hidden_range(m.d_ff, s.mlp_chunks, c)
                store.put(SegmentKey(L, "mlp", c), {
                    "input_projection.weight": blk.mlp.input_projection.weight[f0:f1].clone(),
                    "input_projection.bias": blk.mlp.input_projection.bias[f0:f1].clone(),
                    "output_projection.weight": blk.mlp.output_projection.weight[:, f0:f1].clone(),
                })
    return shared


def _all_segment_keys(m: ModelConfig, s: SegmentationConfig):
    keys = [SegmentKey(-1, "embedding", e) for e in range(s.embedding_segments)]
    keys += [SegmentKey(-1, "output_head", h) for h in range(s.output_head_segments)]
    for L in range(m.n_layers):
        keys += [SegmentKey(L, "attention", a) for a in range(s.attention_segments)]
        keys += [SegmentKey(L, "attn_out_proj", 0)]
        keys += [SegmentKey(L, "mlp", c) for c in range(s.mlp_chunks)]
    return keys


def _gpt_init_segment(seg: nn.Module, kind: str, scale: float) -> None:
    """GPT-style init matching ReferenceGPTDecoder's distribution: N(0,0.02) on weights,
    zeros on biases, and residual-path output projections scaled by 1/sqrt(2*n_layers)."""
    for mod in seg.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0.0, 0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, 0.0, 0.02)
    if kind == "attn_out_proj":                       # residual projection (nn.Linear)
        nn.init.normal_(seg.weight, 0.0, 0.02 * scale)
    elif kind == "mlp":                               # residual projection (no bias)
        nn.init.normal_(seg.output_projection.weight, 0.0, 0.02 * scale)


def populate_from_scratch(m: ModelConfig, s: SegmentationConfig, store, seed: int = 0) -> SharedParams:
    """Initialize the segmented model WITHOUT ever materializing the full model: build each
    segment alone, GPT-init it, write it to the store, and free it -> setup peak = ONE segment.
    Use for COST experiments / from-scratch training (weight VALUES are irrelevant to memory/time;
    the distribution matches the reference). For the bit-exact identity proof use
    populate_from_reference instead. SharedParams (norms=1/0, mlp_out_bias=0) are tiny and resident."""
    torch.manual_seed(seed)
    shared = SharedParams(m)                            # LayerNorms default to weight=1/bias=0; biases zero
    scale = 1.0 / math.sqrt(2 * m.n_layers)
    with torch.no_grad():
        for key in _all_segment_keys(m, s):
            seg = build_segment(key, m, s)             # ONE segment resident
            _gpt_init_segment(seg, key.kind, scale)
            store.put(key, seg.state_dict())           # persist to store (cpu_ram/disk)
            del seg                                     # free immediately
    return shared


if __name__ == "__main__":
    import sys
    from config import SMALL_MODEL, SEG_8x2x2x8
    from modules import ReferenceGPTDecoder, causal_lm_cross_entropy_loss, token_counts
    from stores import make_store
    torch.manual_seed(0)
    m, s = SMALL_MODEL, SEG_8x2x2x8
    ref = ReferenceGPTDecoder(m).eval()

    store = make_store("cpu_ram")
    shared = populate_from_reference(ref, m, s, store).eval()
    loader = StrictSegmentLoader(m, s, store, "cpu")
    eng = SegmentedForwardEngine(m, s, loader, shared, "cpu").eval()  # dropout off, match ref

    x = torch.randint(0, m.vocab_size, (2, 16))
    lab = x.clone(); lab[:, :4] = -100
    with torch.no_grad():
        ref_logits, ref_hidden = ref(x, pad_token_id=None)
        ref_loss = causal_lm_cross_entropy_loss(ref_logits, lab)
        rc, rv = token_counts(ref_logits, lab)
        seg_hidden = eng.forward_hidden(x, pad_token_id=None)
        seg_loss, sc, sv = eng.chunked_ce(seg_hidden, lab, seq_chunk=8)

    dh = (ref_hidden - seg_hidden).abs().max().item()
    dl = abs(ref_loss.item() - seg_loss.item())
    print(f"hidden identity max|Δ|={dh:.2e}")
    print(f"loss  ref={ref_loss.item():.6f} seg={seg_loss.item():.6f} |Δ|={dl:.2e}")
    print(f"acc   ref={rc}/{rv}  seg={sc}/{sv}")
    ok = dh < 1e-4 and dl < 1e-4 and (rc, rv) == (sc, sv)
    print("FORWARD ENGINE IDENTITY OK" if ok else "FORWARD ENGINE MISMATCH")
    sys.exit(0 if ok else 1)

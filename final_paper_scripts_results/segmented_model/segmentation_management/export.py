"""
Export & audit — reassemble the segment store + shared params into one full
`ReferenceGPTDecoder`, and verify the segmented pipeline is numerically faithful.

Reassembly (inverse of forward_engine.populate_from_reference):
  output_projection = concat_H(head slices)            (dim 0, vocab)
  embedding         = concat_E(embedding slices)        (dim 1, d_model)
  per layer:  qkv   = concat[concat_A(q), concat_A(k), concat_A(v)]  (dim 0)
              mlp.in  = concat_M(in chunks) (dim 0); mlp.out = concat_M(out chunks) (dim 1)
              norms / attn out-proj / mlp bias = shared
This single reassembled model is the artifact you deploy / compare, and the basis of
the paper's claim that segmented training yields the same model as full training.

The self-test runs the END-TO-END proof: from one seeded init, train the FULL model
with normal backward+AdamW for N steps, and train the SEGMENTED model for the same N
steps on the same batches; reassemble and compare weights.
"""

from __future__ import annotations

from typing import Dict

import torch

from config import ModelConfig, SegmentationConfig
from modules import ReferenceGPTDecoder
from segments import head_range, hidden_range, dmodel_range, vocab_range
from stores import SegmentKey, SegmentStore


def reassemble_state_dict(store: SegmentStore, shared_state: Dict[str, torch.Tensor],
                          m: ModelConfig, s: SegmentationConfig) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}

    # embedding: concat E d_model-slices
    tok = [store.get(SegmentKey(-1, "embedding", e))["token_embedding.weight"]
           for e in range(s.embedding_segments)]
    pos = [store.get(SegmentKey(-1, "embedding", e))["position_embedding.weight"]
           for e in range(s.embedding_segments)]
    sd["embedding.token_embedding.weight"] = torch.cat(tok, dim=1)
    sd["embedding.position_embedding.weight"] = torch.cat(pos, dim=1)

    # output head: concat H vocab-slices
    sd["output_projection.weight"] = torch.cat(
        [store.get(SegmentKey(-1, "output_head", h))["projection.weight"]
         for h in range(s.output_head_segments)], dim=0)

    for L in range(m.n_layers):
        # attention qkv: concat heads within q,k,v then stack q|k|v
        q = torch.cat([store.get(SegmentKey(L, "attention", a))["q_proj.weight"] for a in range(s.attention_segments)], dim=0)
        k = torch.cat([store.get(SegmentKey(L, "attention", a))["k_proj.weight"] for a in range(s.attention_segments)], dim=0)
        v = torch.cat([store.get(SegmentKey(L, "attention", a))["v_proj.weight"] for a in range(s.attention_segments)], dim=0)
        sd[f"blocks.{L}.attention.qkv_projection.weight"] = torch.cat([q, k, v], dim=0)
        qb = torch.cat([store.get(SegmentKey(L, "attention", a))["q_proj.bias"] for a in range(s.attention_segments)], dim=0)
        kb = torch.cat([store.get(SegmentKey(L, "attention", a))["k_proj.bias"] for a in range(s.attention_segments)], dim=0)
        vb = torch.cat([store.get(SegmentKey(L, "attention", a))["v_proj.bias"] for a in range(s.attention_segments)], dim=0)
        sd[f"blocks.{L}.attention.qkv_projection.bias"] = torch.cat([qb, kb, vb], dim=0)
        _op = store.get(SegmentKey(L, "attn_out_proj", 0))          # W_o now a streamed segment
        sd[f"blocks.{L}.attention.output_projection.weight"] = _op["weight"]
        sd[f"blocks.{L}.attention.output_projection.bias"] = _op["bias"]
        # mlp chunks
        sd[f"blocks.{L}.mlp.input_projection.weight"] = torch.cat(
            [store.get(SegmentKey(L, "mlp", c))["input_projection.weight"] for c in range(s.mlp_chunks)], dim=0)
        sd[f"blocks.{L}.mlp.input_projection.bias"] = torch.cat(
            [store.get(SegmentKey(L, "mlp", c))["input_projection.bias"] for c in range(s.mlp_chunks)], dim=0)
        sd[f"blocks.{L}.mlp.output_projection.weight"] = torch.cat(
            [store.get(SegmentKey(L, "mlp", c))["output_projection.weight"] for c in range(s.mlp_chunks)], dim=1)
        sd[f"blocks.{L}.mlp.output_projection.bias"] = shared_state[f"mlp_out_bias.{L}"]
        # norms
        sd[f"blocks.{L}.attention_norm.weight"] = shared_state[f"attn_norm.{L}.weight"]
        sd[f"blocks.{L}.attention_norm.bias"] = shared_state[f"attn_norm.{L}.bias"]
        sd[f"blocks.{L}.mlp_norm.weight"] = shared_state[f"mlp_norm.{L}.weight"]
        sd[f"blocks.{L}.mlp_norm.bias"] = shared_state[f"mlp_norm.{L}.bias"]

    sd["final_norm.weight"] = shared_state["final_norm.weight"]
    sd["final_norm.bias"] = shared_state["final_norm.bias"]
    return sd


def reassemble_model(store, shared_state, m: ModelConfig, s: SegmentationConfig) -> ReferenceGPTDecoder:
    model = ReferenceGPTDecoder(m)
    missing, unexpected = model.load_state_dict(reassemble_state_dict(store, shared_state, m, s), strict=False)
    # causal_mask is a non-persistent buffer (allowed missing); nothing else may be.
    bad = [k for k in missing if "causal_mask" not in k]
    if bad or unexpected:
        raise RuntimeError(f"reassembly mismatch: missing={bad} unexpected={unexpected}")
    return model


if __name__ == "__main__":
    import sys
    from config import SMALL_MODEL, SEG_8x2x2x8
    from modules import causal_lm_cross_entropy_loss
    from forward_engine import SegmentedForwardEngine, populate_from_reference
    from backward_engine import SegmentedBackwardEngine
    from optimizer import SegmentwiseAdamW, all_segment_keys
    from loader import StrictSegmentLoader
    from stores import make_store
    m, s = SMALL_MODEL, SEG_8x2x2x8

    # ---- (1) round-trip: slice a reference then reassemble -> identical ----
    torch.manual_seed(1)
    ref0 = ReferenceGPTDecoder(m)
    store0 = make_store("cpu_ram")
    shared0 = populate_from_reference(ref0, m, s, store0)
    rt = reassemble_model(store0, shared0.state_dict(), m, s)
    rd = max((ref0.state_dict()[k] - rt.state_dict()[k]).abs().max().item()
             for k in ref0.state_dict() if "causal_mask" not in k)
    print(f"(1) round-trip slice->reassemble max|Δ|={rd:.2e}")

    # ---- (2) END-TO-END: full vs segmented training for N steps from same init ----
    torch.manual_seed(7)
    full = ReferenceGPTDecoder(m).eval()                      # eval = dropout off (determinism)
    # snapshot init, build segmented from the SAME init
    store = make_store("cpu_ram"); shared = populate_from_reference(full, m, s, store)
    loader = StrictSegmentLoader(m, s, store, "cpu")
    fwd = SegmentedForwardEngine(m, s, loader, shared, "cpu").eval()
    gstore = make_store("cpu_ram"); ostore = make_store("cpu_ram")
    bwd = SegmentedBackwardEngine(fwd, gstore)
    sopt = SegmentwiseAdamW(m, s, loader, shared, gstore, ostore, "cpu", lr=3e-4)

    decay = [p for p in full.parameters() if p.dim() >= 2]
    nod = [p for p in full.parameters() if p.dim() < 2]
    fopt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                              {"params": nod, "weight_decay": 0.0}], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    torch.manual_seed(123)
    N = 4
    for step in range(N):
        x = torch.randint(0, m.vocab_size, (3, 20)); lab = x.clone(); lab[:, :4] = -100
        # full step
        fopt.zero_grad(); lg, _ = full(x, pad_token_id=None)
        causal_lm_cross_entropy_loss(lg, lab).backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0); fopt.step()
        # segmented step (same batch)
        for k in all_segment_keys(m, s):
            gstore.evict(k)
        sg = bwd.backward(x, lab, pad_token_id=None)
        sopt.step(sg, clip_norm=1.0)

    reasm = reassemble_model(store, shared.state_dict(), m, s)
    dd = max((full.state_dict()[k] - reasm.state_dict()[k]).abs().max().item()
             for k in full.state_dict() if "causal_mask" not in k)
    print(f"(2) after {N} train steps, full vs segmented-reassembled max|Δ|={dd:.2e}")
    ok = rd < 1e-6 and dd < 1e-4
    print("EXPORT/AUDIT OK — segmented training == full training (end-to-end)" if ok else "AUDIT FAILED")
    sys.exit(0 if ok else 1)

"""
Streamed segment-wise AdamW — the memory-minimal `after_full_backward` update.

ONLY AdamW (matching the full-model baseline: betas=(0.9,0.95), eps=1e-8,
weight_decay=0.1, lr=3e-4; GPT-style grouping = decouple weight-decay on 2D+ weights,
none on biases/LayerNorm). No SGD variants.

The update is applied ONE SEGMENT AT A TIME, so the optimizer never holds more than
one segment's (params + grad + Adam state) on the constrained device:
    for each segment key:
        load params (segment store) + grad (grad store) + (m,v) (opt-state store)
        AdamW update in place
        write params + (m,v) back to their stores;  free
Shared (tiny, resident) params get a normal in-memory AdamW step.

Gradient clipping is TWO-PASS STREAMING: pass 1 accumulates the global sum-of-squares
one segment at a time (then frees), pass 2 reloads and scales. The full gradient is
never materialized on the constrained device — the exact issue flagged in the review.

The per-param math reproduces `torch.optim.AdamW` exactly (verified in the self-test).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch

from config import ModelConfig, SegmentationConfig
from forward_engine import SharedParams
from loader import StrictSegmentLoader, _flag
from optrace import mark as _tm     # no-op unless a cost profiler installs a hook
from stores import SegmentKey, SegmentStore


def all_segment_keys(m: ModelConfig, s: SegmentationConfig) -> List[SegmentKey]:
    keys = [SegmentKey(-1, "embedding", e) for e in range(s.embedding_segments)]
    keys += [SegmentKey(-1, "output_head", h) for h in range(s.output_head_segments)]
    for L in range(m.n_layers):
        keys += [SegmentKey(L, "attention", a) for a in range(s.attention_segments)]
        keys += [SegmentKey(L, "attn_out_proj", 0)]      # W_o, one streamed segment per layer
        keys += [SegmentKey(L, "mlp", c) for c in range(s.mlp_chunks)]
    return keys


def _adamw_(p: torch.Tensor, g: torch.Tensor, st: Dict[str, torch.Tensor],
            t: int, lr: float, b1: float, b2: float, eps: float, wd: float) -> None:
    """In-place AdamW on one param, matching torch.optim.AdamW. Decoupled weight
    decay only on 2D+ params (GPT-style). `st` holds exp_avg / exp_avg_sq."""
    if wd != 0.0 and p.dim() >= 2:
        p.mul_(1.0 - lr * wd)
    if "exp_avg" not in st:
        st["exp_avg"] = torch.zeros_like(p)
        st["exp_avg_sq"] = torch.zeros_like(p)
    m, v = st["exp_avg"], st["exp_avg_sq"]
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
    p.addcdiv_(m, denom, value=-(lr / bc1))


class SegmentwiseAdamW:
    def __init__(self, model: ModelConfig, seg: SegmentationConfig,
                 loader: StrictSegmentLoader, shared: SharedParams,
                 grad_store: SegmentStore, opt_state_store: SegmentStore, device: str,
                 lr: float = 3e-4, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1):
        self.m, self.s, self.ld, self.sh = model, seg, loader, shared
        self.grad_store, self.opt_store, self.device = grad_store, opt_state_store, device
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.t = 0
        self.shared_state: Dict[str, Dict[str, torch.Tensor]] = {}
        # offload_adam ON (default): park shared Adam m,v on host between steps; OFF: keep
        # them resident on the compute device. (Segment opt-state always streams via opt_store.)
        self._offload_adam = _flag(loader.tech, "offload_adam")

    # ---- two-pass streaming global-norm clip ----
    def _clip_scale(self, shared_grads: Dict[str, torch.Tensor], max_norm: float) -> float:
        # accumulate as a plain Python float so segment grads (CPU store) and shared
        # grads (on the compute device) can be summed without a device mismatch.
        total_sq = 0.0
        for key in all_segment_keys(self.m, self.s):           # pass 1: stream grads
            gsd = self.grad_store.get(key)
            if gsd:
                for g in gsd.values():
                    total_sq += float((g.double() ** 2).sum())
            del gsd
        for g in shared_grads.values():
            total_sq += float((g.double() ** 2).sum())
        norm = total_sq ** 0.5
        return 1.0 if norm <= max_norm else max_norm / (norm + 1e-6)

    def step(self, shared_grads: Dict[str, torch.Tensor],
             clip_norm: Optional[float] = None) -> None:
        self.t += 1
        scale = self._clip_scale(shared_grads, clip_norm) if clip_norm else 1.0

        # segments — one at a time (pass 2 of clip folded in via `scale`)
        for key in all_segment_keys(self.m, self.s):
            gsd = self.grad_store.get(key)
            if not gsd:
                continue
            st = self.opt_store.get(key) or {}
            with self.ld.acquire_segment(key, save_on_exit=True) as seg:
                for pname, p in seg.named_parameters():
                    if pname not in gsd:
                        continue
                    g = gsd[pname].to(self.device)
                    if scale != 1.0:
                        g = g * scale
                    # opt state lives in the store (CPU on GPU runs) — move to device
                    pst = {"exp_avg": st.get(f"{pname}.exp_avg"),
                           "exp_avg_sq": st.get(f"{pname}.exp_avg_sq")}
                    pst = {k: v.to(self.device) for k, v in pst.items() if v is not None}
                    _adamw_(p.data, g, pst, self.t, self.lr, self.b1, self.b2, self.eps, self.wd)
                    st[f"{pname}.exp_avg"] = pst["exp_avg"].to("cpu")
                    st[f"{pname}.exp_avg_sq"] = pst["exp_avg_sq"].to("cpu")
            self.opt_store.put(key, st)
            del gsd, st

        # shared params — the params themselves stay resident (used every layer), but
        # their Adam state (m,v) is PARKED on the host (self.shared_state holds CPU
        # tensors) and streamed to the device one param at a time for the update, then
        # moved back. So the ~469 MB of shared optimizer state never sits in VRAM —
        # measured to cut the resident floor 695->226 MB (attn_out_proj dominates it).
        sd = dict(self.sh.named_parameters())
        for pname, p in sd.items():
            if pname not in shared_grads:
                continue
            g = shared_grads[pname].to(self.device)
            if scale != 1.0:
                g = g * scale
            prev_st = self.shared_state.get(pname, {})
            # offload_adam ON: m,v were parked on host -> move to device for the update, then
            # park back. OFF: they already live on device -> keep them there (no host copy).
            pst = {k: (v.to(self.device) if self._offload_adam else v) for k, v in prev_st.items()}
            _adamw_(p.data, g, pst, self.t, self.lr, self.b1, self.b2, self.eps, self.wd)
            self.shared_state[pname] = {k: (v.to("cpu") if self._offload_adam else v)
                                        for k, v in pst.items()}
            del g, pst
            _tm(f"opt|shared|{pname}")


if __name__ == "__main__":
    # VERIFY: one segmented AdamW step == one torch.optim.AdamW step (same grouping).
    import sys
    from config import SMALL_MODEL, SEG_8x2x2x8
    from modules import ReferenceGPTDecoder, causal_lm_cross_entropy_loss
    from forward_engine import SegmentedForwardEngine, populate_from_reference
    from backward_engine import SegmentedBackwardEngine
    from segments import head_range, hidden_range, dmodel_range, vocab_range
    from stores import make_store
    torch.manual_seed(0)
    m, s = SMALL_MODEL, SEG_8x2x2x8
    ref = ReferenceGPTDecoder(m).eval()
    x = torch.randint(0, m.vocab_size, (2, 24)); lab = x.clone(); lab[:, :5] = -100

    # snapshot initial params, build segmented from the SAME initial params
    store = make_store("cpu_ram"); shared = populate_from_reference(ref, m, s, store)
    fwd = SegmentedForwardEngine(m, s, StrictSegmentLoader(m, s, store, "cpu"), shared, "cpu").eval()
    gstore = make_store("cpu_ram")
    shared_g = SegmentedBackwardEngine(fwd, gstore).backward(x, lab, pad_token_id=None)

    # reference: same grads via full backward, torch AdamW with 2-group decay
    ref.zero_grad(); logits, _ = ref(x, pad_token_id=None)
    causal_lm_cross_entropy_loss(logits, lab).backward()
    decay = [p for p in ref.parameters() if p.dim() >= 2]
    nodecay = [p for p in ref.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    opt.step()
    ref_after = {n: p.detach().clone() for n, p in ref.named_parameters()}

    # segmented step
    sopt = SegmentwiseAdamW(m, s, fwd.loader, shared, gstore, make_store("cpu_ram"), "cpu")
    sopt.step(shared_g, clip_norm=1.0)
    # (also clip the reference identically for fair compare)
    # -> redo reference with clip to match:
    torch.manual_seed(0); ref2 = ReferenceGPTDecoder(m).eval()
    store2 = make_store("cpu_ram"); shared2 = populate_from_reference(ref2, m, s, store2)
    ref2.zero_grad(); lg2, _ = ref2(x, pad_token_id=None)
    causal_lm_cross_entropy_loss(lg2, lab).backward()
    torch.nn.utils.clip_grad_norm_(ref2.parameters(), 1.0)
    d2 = [p for p in ref2.parameters() if p.dim() >= 2]; nd2 = [p for p in ref2.parameters() if p.dim() < 2]
    o2 = torch.optim.AdamW([{"params": d2, "weight_decay": 0.1}, {"params": nd2, "weight_decay": 0.0}],
                           lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    o2.step()
    refc = {n: p.detach().clone() for n, p in ref2.named_parameters()}

    # compare updated params: shared + a few segment slices
    maxd = 0.0
    for L in range(m.n_layers):
        maxd = max(maxd, (refc[f"blocks.{L}.attention_norm.weight"] - dict(shared.named_parameters())[f"attn_norm.{L}.weight"]).abs().max().item())
        maxd = max(maxd, (refc[f"blocks.{L}.attention.output_projection.weight"] - store.get(SegmentKey(L, "attn_out_proj", 0))["weight"]).abs().max().item())
    maxd = max(maxd, (refc["final_norm.weight"] - dict(shared.named_parameters())["final_norm.weight"]).abs().max().item())
    mlp0 = fwd.loader.store.get(SegmentKey(0, "mlp", 0)); f0, f1 = hidden_range(m.d_ff, s.mlp_chunks, 0)
    d_mlp = (refc["blocks.0.mlp.input_projection.weight"][f0:f1] - mlp0["input_projection.weight"]).abs().max().item()
    emb0 = fwd.loader.store.get(SegmentKey(-1, "embedding", 0)); d0, d1 = dmodel_range(m.d_model, s.embedding_segments, 0)
    d_emb = (refc["embedding.token_embedding.weight"][:, d0:d1] - emb0["token_embedding.weight"]).abs().max().item()
    print(f"shared updated max|Δ|={maxd:.2e}  mlp slice|Δ|={d_mlp:.2e}  emb slice|Δ|={d_emb:.2e}")
    ok = maxd < 1e-6 and d_mlp < 1e-6 and d_emb < 1e-6
    print("ADAMW STEP IDENTITY OK (segmented == torch.optim.AdamW)" if ok else "ADAMW MISMATCH")
    sys.exit(0 if ok else 1)

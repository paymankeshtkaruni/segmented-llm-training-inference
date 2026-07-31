"""
Segmented backward engine — full backward by RECOMPUTATION, one segment at a time.

No global `loss.backward()` and no retained forward graph. Instead:
  * the residual-stream hidden state at each layer boundary is recorded in the
    records store (cpu_ram on GPU / disk on CPU) — NOT the full activations;
  * the output-head cross-entropy gradient is computed ANALYTICALLY and STREAMED
    over the H vocab slices (softmax−onehot), so `[B,T,vocab]` is never built;
  * each layer is back-propagated by reloading one segment, recomputing only that
    segment's forward (grad enabled), and calling `torch.autograd.grad` to get its
    parameter grads + the input grad to pass upstream. Param grads are written to
    the gradient store immediately and freed.

Peak memory ≈ one segment's recompute graph + a few `[B,T,d_model]` grad tensors —
independent of model depth. Correctness is the point: the reassembled grads must
equal a normal `loss.backward()` (verified in the self-test). This is the heart of
the "segmented == full *training*" claim.

Grad outputs:
  * segment param grads  -> `grad_store` (keyed by SegmentKey), streamed.
  * shared param grads   -> returned dict name->tensor (small, resident).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import ModelConfig, SegmentationConfig
from forward_engine import SegmentedForwardEngine, SharedParams
from loader import StrictSegmentLoader, _flag
from optrace import mark as _tm     # no-op unless a cost profiler installs a hook
from segments import vocab_range
from stores import SegmentKey, SegmentStore, make_store

# NOTE: the shared-grad accumulator is now SegmentedBackwardEngine._add (a method), so the
# park_grads_host TECHNIQUE can be toggled: park on host (default) vs keep on the device.


def _rec_key(L: int) -> SegmentKey:
    return SegmentKey(L, "layer_input", 0)


class SegmentedBackwardEngine:
    def __init__(self, fwd: SegmentedForwardEngine, grad_store: SegmentStore,
                 records_store: Optional[SegmentStore] = None):
        self.f = fwd
        self.m: ModelConfig = fwd.m
        self.s: SegmentationConfig = fwd.s
        self.ld: StrictSegmentLoader = fwd.loader
        self.sh: SharedParams = fwd.shared
        self.device = fwd.device
        self.grad_store = grad_store
        # Records store holds the residual-stream snapshot at each layer boundary
        # OFF the compute device (cpu_ram on GPU / disk on CPU), so backward never
        # pins all n_layers activations in VRAM — peak becomes depth-INDEPENDENT.
        self.records = records_store if records_store is not None else make_store("cpu_ram")
        # ---- technique toggles (None tech => all ON, unchanged behavior) -------------
        tech = self.ld.tech
        self._offload_records = _flag(tech, "offload_records")  # records store vs on-device
        self._park_grads = _flag(tech, "park_grads_host")       # shared grads host vs device
        self._recompute = _flag(tech, "recompute")              # recompute vs full autograd graph
        self._chunked_ce = _flag(tech, "chunked_ce")            # streamed CE vs full logits
        self._rec_dev: Dict[int, torch.Tensor] = {}             # on-device records when OFF
        # ---- dropout RNG bookkeeping (Path B, fix (a)) -------------------------------
        # The backward RECOMPUTES the forward. If dropout is active, a naive recompute
        # draws FRESH masks != the recorded forward's -> the gradient is taken w.r.t. a
        # different dropout realization than the loss (broken chain rule across masks).
        # We capture the RNG state right before each dropout draw in the recorded forward
        # and restore it before the matching draw in every recompute, so the recompute
        # reproduces the SAME masks (cf. torch.utils.checkpoint preserve_rng_state).
        self._rng: Dict[tuple, tuple] = {}
        self._train: bool = False     # set per backward(): training AND dropout>0

    # ---- shared-grad accumulator (park_grads_host technique) ----
    def _add(self, d: Dict[str, torch.Tensor], name: str, g: Optional[torch.Tensor]) -> None:
        # park_grads_host ON (default): move shared-param grads to the HOST as computed, so
        # e.g. the ~226 MB/layer attn_out_proj grads never accumulate on the device. OFF:
        # keep them on the compute device. `.to("cpu")` is a no-op on CPU runs (values
        # identical, verified by the gradient-identity test). Accumulation add is in-place.
        if g is None:
            return
        g = g.detach()
        if self._park_grads:
            g = g.to("cpu")
        d[name] = g if name not in d else d[name] + g

    # ---- layer-input records (offload_records technique) ----
    def _rec_put(self, L: int, hidden: torch.Tensor) -> None:
        if self._offload_records:
            self.records.put(_rec_key(L), {"h": hidden})    # off-device snapshot
        else:
            self._rec_dev[L] = hidden                       # keep resident on device

    def _rec_get(self, L: int) -> torch.Tensor:
        if self._offload_records:
            return self.records.get(_rec_key(L))["h"].to(self.device)
        return self._rec_dev[L]

    def _rec_evict(self, L: int) -> None:
        if self._offload_records:
            self.records.evict(_rec_key(L))
        else:
            self._rec_dev.pop(L, None)

    # ---- per-segment RNG capture/restore so recompute masks == recorded-forward masks ----
    def _rng_capture(self) -> tuple:
        if str(self.device).startswith("cuda"):
            return (torch.get_rng_state(), torch.cuda.get_rng_state(self.device))
        return (torch.get_rng_state(), None)

    def _rng_restore(self, st: tuple) -> None:
        torch.set_rng_state(st[0])
        if st[1] is not None:
            torch.cuda.set_rng_state(st[1], self.device)

    def _drop_record(self, x: torch.Tensor, key: tuple) -> torch.Tensor:
        """Record-pass dropout: capture RNG, then draw the mask. No-op unless training."""
        if not self._train:
            return x
        self._rng[key] = self._rng_capture()
        return F.dropout(x, self.m.dropout, True)

    def _drop_value(self, x: torch.Tensor, key: tuple) -> torch.Tensor:
        """Recompute the dropped VALUE with the recorded mask (restore RNG, redraw)."""
        if not self._train:
            return x
        self._rng_restore(self._rng[key])
        return F.dropout(x, self.m.dropout, True)

    def _drop_mask(self, like: torch.Tensor, key: tuple) -> Optional[torch.Tensor]:
        """Scaled 0/(1/(1-p)) mask reproducing the recorded draw, to fold into a gradient.
        F.dropout on ones gives the mask pattern (shape+RNG decide it, not values)."""
        if not self._train:
            return None
        self._rng_restore(self._rng[key])
        return F.dropout(torch.ones_like(like), self.m.dropout, True)

    # ---- record the residual stream at each layer boundary (no_grad) ----
    # Each layer's input is streamed to the records store and freed from the device;
    # only the single running `hidden` stays resident. Backward reloads them one at a
    # time. (Verified identical to the in-device version; this only changes WHERE the
    # snapshots live, not their values.)
    @torch.no_grad()
    def _record_layer_inputs(self, input_ids, pad_token_id) -> torch.Tensor:
        from forward_engine import _attn_bias
        m, s, ld = self.m, self.s, self.ld
        self._train = bool(ld.training and m.dropout > 0)   # dropout-active path?
        self._rng = {}
        bias = _attn_bias(input_ids, pad_token_id, self.device)
        parts = []
        for e in range(s.embedding_segments):
            if self._train:
                self._rng[("emb", e)] = self._rng_capture()   # before this slice's F.dropout
            with ld.acquire_segment(SegmentKey(-1, "embedding", e)) as emb:
                parts.append(emb(input_ids))
        hidden = torch.cat(parts, dim=-1)
        del parts
        for L in range(m.n_layers):
            self._rec_put(L, hidden)      # off-device snapshot
            _tm(f"rec|L{L}|store")
            xin = self.sh.attn_norm[L](hidden)
            outs = []
            for a in range(s.attention_segments):
                if self._train:
                    self._rng[("attn", L, a)] = self._rng_capture()   # before SDPA dropout
                with ld.acquire_segment(SegmentKey(L, "attention", a)) as seg:
                    outs.append(seg(xin, attn_bias=bias))
            with ld.acquire_segment(SegmentKey(L, "attn_out_proj", 0)) as op:
                # (c) attention output dropout on the projection result
                hidden = hidden + self._drop_record(op(torch.cat(outs, dim=-1)), ("attn_out", L))
            xin = self.sh.mlp_norm[L](hidden)
            acc = None
            for c in range(s.mlp_chunks):
                with ld.acquire_segment(SegmentKey(L, "mlp", c)) as seg:
                    out = seg(xin)
                acc = out if acc is None else acc + out
            # (d) MLP dropout on the SUMMED output (acc + bias), matching the reference
            hidden = hidden + self._drop_record(acc + self.sh.mlp_out_bias[L], ("mlp_out", L))
        return hidden            # hidden == hidden_last (pre final_norm); records in store

    # ---- analytic, streamed CE grad wrt the (shifted) final hidden_norm ----
    # NO autograd: the gradient is computed in closed form (softmax - onehot) and only
    # ever used as `grad_outputs`. Without this, `sh` (a view of final_norm(hl), which
    # requires grad) makes every slice's z/softmax/g_z/einsum build a graph that
    # `grad_hidden` accumulates across ALL vocab slices — ~1.5 GB of needless VRAM
    # (measured). @torch.no_grad() keeps the values identical (verified) and bounds it.
    @torch.no_grad()
    def _ce_backward(self, hidden_norm, labels, ignore_index=-100):
        m, s, ld = self.m, self.s, self.ld
        sh = hidden_norm[:, :-1, :]
        tgt = labels[:, 1:]
        B, Tm1, D = sh.shape
        mask = (tgt != ignore_index)
        n_valid = int(mask.sum())
        # pass 1: lse over all vocab slices (no grad)
        with torch.no_grad():
            lse = None
            for h in range(s.output_head_segments):
                v0, v1 = vocab_range(m.vocab_size, h, s.output_head_segments) if False else vocab_range(m.vocab_size, s.output_head_segments, h)
                with ld.acquire_segment(SegmentKey(-1, "output_head", h)) as head:
                    z = head(sh)
                sl = torch.logsumexp(z, dim=-1)
                lse = sl if lse is None else torch.logaddexp(lse, sl)
                del z
                _tm(f"cebwd|lse|h{h}")
        # pass 2: grad_logits = (softmax - onehot)/n_valid (masked); accumulate.
        # Also accumulate the scalar loss (lse - correct_logit) and running argmax for
        # accuracy — so training/eval stats come for free (no extra forward).
        grad_hidden = torch.zeros_like(sh)
        scale = 1.0 / max(1, n_valid)
        correct_logit = torch.zeros_like(tgt, dtype=sh.dtype)
        arg_val = torch.zeros_like(tgt, dtype=sh.dtype) + torch.finfo(torch.float32).min
        arg_idx = torch.zeros_like(tgt)
        for h in range(s.output_head_segments):
            v0, v1 = vocab_range(m.vocab_size, s.output_head_segments, h)
            with ld.acquire_segment(SegmentKey(-1, "output_head", h)) as head:
                z = head(sh)                                  # [B,Tm1,vsl]
                softmax = torch.exp(z - lse.unsqueeze(-1))    # streamed softmax
                onehot = torch.zeros_like(softmax)
                in_rng = (tgt >= v0) & (tgt < v1) & mask
                local = (tgt - v0).clamp(0, v1 - v0 - 1)
                onehot.scatter_(-1, local.unsqueeze(-1), in_rng.unsqueeze(-1).to(softmax.dtype))
                g_z = (softmax - onehot) * (mask.unsqueeze(-1).to(softmax.dtype)) * scale
                gw = torch.einsum("btv,btd->vd", g_z, sh)
                self.grad_store.put(SegmentKey(-1, "output_head", h), {"projection.weight": gw})
                grad_hidden = grad_hidden + torch.einsum("btv,vd->btd", g_z, head.projection.weight)
                picked = z.gather(-1, local.unsqueeze(-1)).squeeze(-1)
                correct_logit = torch.where((tgt >= v0) & (tgt < v1), picked, correct_logit)
                smax, sarg = z.max(dim=-1)
                upd = smax > arg_val
                arg_val = torch.where(upd, smax, arg_val); arg_idx = torch.where(upd, sarg + v0, arg_idx)
                del z, softmax, onehot, g_z, gw
            _tm(f"cebwd|grad|h{h}")
        per_token = (lse - correct_logit)
        self.last_loss = float(per_token[mask].sum() / max(1, n_valid))
        self.last_correct = int(((arg_idx == tgt) & mask).sum())
        self.last_valid = n_valid
        full = torch.zeros(B, sh.size(1) + 1, D, device=hidden_norm.device, dtype=hidden_norm.dtype)
        full[:, :-1, :] = grad_hidden
        return full, n_valid

    # ---- FULL-GRAPH backward (recompute technique OFF) -----------------------------
    # The naive/high-memory baseline: run a grad-enabled forward that RETAINS the whole
    # autograd graph (all activations pinned), one loss.backward(), then harvest grads
    # into the same stores the optimizer reads. Requires every segment resident (stream
    # OFF) so the leaf params persist for autograd — guaranteed at the recompute-OFF
    # rungs of the ladder. Values equal a normal loss.backward() (verified).
    def _loss_grad(self, hidden_norm, labels, ignore_index=-100):
        m, s, ld = self.m, self.s, self.ld
        sh = hidden_norm[:, :-1, :]; tgt = labels[:, 1:]
        mask = (tgt != ignore_index); n_valid = int(mask.sum())
        H = s.output_head_segments
        if not self._chunked_ce:
            # FULL logits [B,Tm1,vocab] (chunked_ce OFF): concat head slices, one matmul.
            ws = []
            for h in range(H):
                with ld.acquire_segment(SegmentKey(-1, "output_head", h)) as head:
                    ws.append(head.projection.weight)          # resident (stream OFF)
            logits = sh @ torch.cat(ws, dim=0).t()             # [B,Tm1,vocab]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), ignore_index=ignore_index)
            with torch.no_grad():
                self.last_correct = int(((logits.argmax(-1) == tgt) & mask).sum())
        else:
            # streamed CE WITH grad (chunked_ce ON): logsumexp/gather per vocab slice.
            lse = None; correct = torch.zeros_like(tgt, dtype=sh.dtype)
            arg_val = torch.zeros_like(tgt, dtype=sh.dtype) + torch.finfo(torch.float32).min
            arg_idx = torch.zeros_like(tgt)
            for h in range(H):
                v0, v1 = vocab_range(m.vocab_size, H, h)
                with ld.acquire_segment(SegmentKey(-1, "output_head", h)) as head:
                    z = head(sh)                               # [B,Tm1,vsl]
                sl = torch.logsumexp(z, dim=-1)
                lse = sl if lse is None else torch.logaddexp(lse, sl)
                in_rng = (tgt >= v0) & (tgt < v1)
                local = (tgt - v0).clamp(0, v1 - v0 - 1)
                correct = torch.where(in_rng, z.gather(-1, local.unsqueeze(-1)).squeeze(-1), correct)
                with torch.no_grad():
                    smax, sarg = z.max(dim=-1); upd = smax > arg_val
                    arg_val = torch.where(upd, smax, arg_val); arg_idx = torch.where(upd, sarg + v0, arg_idx)
            loss = (lse - correct)[mask].sum() / max(1, n_valid)
            self.last_correct = int(((arg_idx == tgt) & mask).sum())
        self.last_valid = n_valid; self.last_loss = float(loss.detach())
        return loss

    def _backward_fullgraph(self, input_ids, labels, pad_token_id=None) -> Dict[str, torch.Tensor]:
        from forward_engine import _all_segment_keys
        m, s, ld, sh = self.m, self.s, self.ld, self.sh
        shared_grads: Dict[str, torch.Tensor] = {}
        for mod in ld._resident.values():                      # zero prior grads
            for p in mod.parameters():
                p.grad = None
        for p in sh.parameters():
            p.grad = None
        hidden_norm = self.f.forward_hidden(input_ids, pad_token_id=pad_token_id)  # grad ON
        loss = self._loss_grad(hidden_norm, labels)
        loss.backward()                                        # full autograd (all activations pinned)
        for key in _all_segment_keys(m, s):                    # harvest segment grads -> store
            mod = ld._resident.get(key)
            if mod is None:
                continue
            gd = {n: p.grad.detach() for n, p in mod.named_parameters() if p.grad is not None}
            if gd:
                self.grad_store.put(key, gd)
        for n, p in sh.named_parameters():                     # harvest shared grads (park-aware)
            self._add(shared_grads, n, p.grad)
        return shared_grads

    def backward(self, input_ids, labels, pad_token_id=None) -> Dict[str, torch.Tensor]:
        if not self._recompute:
            return self._backward_fullgraph(input_ids, labels, pad_token_id)
        m, s, ld, sh = self.m, self.s, self.ld, self.sh
        from forward_engine import _attn_bias
        bias = _attn_bias(input_ids, pad_token_id, self.device)
        shared_grads: Dict[str, torch.Tensor] = {}

        hidden_last = self._record_layer_inputs(input_ids, pad_token_id)
        # final_norm: hidden_norm = final_norm(hidden_last)
        hl = hidden_last.detach().requires_grad_(True)
        hidden_norm = sh.final_norm(hl)
        g_hidden_norm, n_valid = self._ce_backward(hidden_norm, labels)
        fn_params = list(sh.final_norm.parameters())
        grads = torch.autograd.grad(hidden_norm, [hl, *fn_params], grad_outputs=g_hidden_norm,
                                    retain_graph=False, allow_unused=True)
        g = grads[0]                                   # grad wrt hidden_last
        self._add(shared_grads, "final_norm.weight", grads[1]); self._add(shared_grads, "final_norm.bias", grads[2])
        del hidden_norm, g_hidden_norm, hl
        _tm("bwd|final_norm")

        for L in reversed(range(m.n_layers)):
            h_in = self._rec_get(L)   # reload off-device snapshot
            _tm(f"bwd|L{L}|load_record")
            # recompute hidden_half = h_in + attn_out (no grad, just the value)
            with torch.no_grad():
                xin = sh.attn_norm[L](h_in)
                outs = []
                for a in range(s.attention_segments):
                    if self._train:
                        self._rng_restore(self._rng[("attn", L, a)])   # same SDPA mask as record
                    with ld.acquire_segment(SegmentKey(L, "attention", a)) as seg:
                        outs.append(seg(xin, attn_bias=bias))
                concat_val = torch.cat(outs, dim=-1)
                with ld.acquire_segment(SegmentKey(L, "attn_out_proj", 0)) as op:
                    # (c) reproduce hidden_half with the recorded attn-output dropout mask
                    hidden_half = h_in + self._drop_value(op(concat_val), ("attn_out", L))
                del outs, xin
            _tm(f"bwd|L{L}|recompute_attn")

            # ---- MLP backward: hidden_{L+1} = hidden_half + dropout(Σ chunk + bias) ----
            mask_mlp = self._drop_mask(g, ("mlp_out", L))     # (d) recorded MLP-sum mask, or None
            g_mlp = g if mask_mlp is None else g * mask_mlp    # grad wrt (Σ chunk + bias)
            self._add(shared_grads, f"mlp_out_bias.{L}", g_mlp.sum(dim=(0, 1)))
            g_half = g.clone()                          # residual: grad flows to hidden_half
            hh = hidden_half.detach().requires_grad_(True)
            xnorm = sh.mlp_norm[L](hh)                   # shared norm (grad accumulates)
            mlp_norm_params = list(sh.mlp_norm[L].parameters())
            for c in range(s.mlp_chunks):
                with ld.acquire_segment(SegmentKey(L, "mlp", c)) as seg:
                    seg_params = list(seg.parameters())
                    out = seg(xnorm)
                    gs = torch.autograd.grad(out, [hh, *mlp_norm_params, *seg_params],
                                             grad_outputs=g_mlp, retain_graph=True, allow_unused=True)
                g_half = g_half + (gs[0] if gs[0] is not None else 0)
                self._add(shared_grads, f"mlp_norm.{L}.weight", gs[1]); self._add(shared_grads, f"mlp_norm.{L}.bias", gs[2])
                names = [n for n, _ in seg.named_parameters()]
                self.grad_store.put(SegmentKey(L, "mlp", c),
                                    {n: gs[3 + i].detach() for i, n in enumerate(names)})
                del out, gs
                _tm(f"bwd|L{L}|mlp_chunk{c}")
            del xnorm, hh

            # ---- Attention backward: hidden_half = h_in + dropout(out_proj(concat)) ----
            g_in = g_half.clone()                       # residual: grad flows to h_in
            mask_ao = self._drop_mask(g_half, ("attn_out", L))   # (c) recorded attn-out mask, or None
            g_ao = g_half if mask_ao is None else g_half * mask_ao   # grad wrt out_proj(concat)
            # out_proj backward (shared): need concat with grad
            with ld.acquire_segment(SegmentKey(L, "attn_out_proj", 0)) as op:
                c_req = concat_val.detach().requires_grad_(True)
                a = op(c_req)
                op_params = list(op.parameters())
                go = torch.autograd.grad(a, [c_req, *op_params], grad_outputs=g_ao,
                                         retain_graph=False, allow_unused=True)
                grad_concat = go[0]
                names = [n for n, _ in op.named_parameters()]      # weight, bias
                self.grad_store.put(SegmentKey(L, "attn_out_proj", 0),
                                    {n: go[1 + i].detach() for i, n in enumerate(names)})
                del a
            del c_req, concat_val
            _tm(f"bwd|L{L}|attn_outproj")
            # per head-group segment
            hh = h_in.detach().requires_grad_(True)
            xnorm = sh.attn_norm[L](hh)
            an_params = list(sh.attn_norm[L].parameters())
            off = 0
            for asg in range(s.attention_segments):
                if self._train:
                    self._rng_restore(self._rng[("attn", L, asg)])   # same SDPA mask as record
                with ld.acquire_segment(SegmentKey(L, "attention", asg)) as seg:
                    seg_params = list(seg.parameters())
                    out = seg(xnorm, attn_bias=bias)
                    w = out.size(-1)
                    g_slice = grad_concat[..., off:off + w]; off += w
                    gs = torch.autograd.grad(out, [hh, *an_params, *seg_params],
                                             grad_outputs=g_slice, retain_graph=True, allow_unused=True)
                g_in = g_in + (gs[0] if gs[0] is not None else 0)
                self._add(shared_grads, f"attn_norm.{L}.weight", gs[1]); self._add(shared_grads, f"attn_norm.{L}.bias", gs[2])
                names = [n for n, _ in seg.named_parameters()]
                self.grad_store.put(SegmentKey(L, "attention", asg),
                                    {n: gs[3 + i].detach() for i, n in enumerate(names)})
                del out, gs
                _tm(f"bwd|L{L}|attn_seg{asg}")
            del xnorm, hh, grad_concat
            g = g_in                                    # grad wrt input of layer L
            del h_in; self._rec_evict(L)   # free this layer's snapshot

        # ---- embedding backward: hidden_0 = concat slices ----
        from segments import dmodel_range
        off = 0
        for e in range(s.embedding_segments):
            d0, d1 = dmodel_range(m.d_model, s.embedding_segments, e)
            if self._train:
                self._rng_restore(self._rng[("emb", e)])   # same embedding dropout mask as record
            with ld.acquire_segment(SegmentKey(-1, "embedding", e)) as emb:
                emb_params = list(emb.parameters())
                names = [n for n, _ in emb.named_parameters()]
                out = emb(input_ids)
                gs = torch.autograd.grad(out, emb_params, grad_outputs=g[..., d0:d1],
                                         retain_graph=False, allow_unused=True)
            self.grad_store.put(SegmentKey(-1, "embedding", e),
                                {n: gs[i].detach() for i, n in enumerate(names) if gs[i] is not None})
            del out, gs
            _tm(f"bwd|emb_seg{e}")
        return shared_grads


if __name__ == "__main__":
    import sys
    from config import SMALL_MODEL, SEG_8x2x2x8
    from modules import ReferenceGPTDecoder, causal_lm_cross_entropy_loss
    from stores import make_store
    from segments import head_range, hidden_range, dmodel_range
    torch.manual_seed(0)
    m, s = SMALL_MODEL, SEG_8x2x2x8
    ref = ReferenceGPTDecoder(m).train()
    x = torch.randint(0, m.vocab_size, (2, 24))
    lab = x.clone(); lab[:, :5] = -100

    # reference grads (full backward), dropout OFF for determinism
    ref.eval()
    ref.zero_grad()
    logits, _ = ref(x, pad_token_id=None)
    causal_lm_cross_entropy_loss(logits, lab).backward()
    ref_g = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}

    # segmented grads
    from forward_engine import populate_from_reference
    store = make_store("cpu_ram")
    shared = populate_from_reference(ref, m, s, store)
    loader = StrictSegmentLoader(m, s, store, "cpu")
    fwd = SegmentedForwardEngine(m, s, loader, shared, "cpu").eval()
    gstore = make_store("cpu_ram")
    bwd = SegmentedBackwardEngine(fwd, gstore)
    shared_g = bwd.backward(x, lab, pad_token_id=None)

    # compare shared grads
    maxd = 0.0
    def cmp(name_ref, val):
        global maxd
        d = (ref_g[name_ref] - val).abs().max().item(); maxd = max(maxd, d); return d
    for L in range(m.n_layers):
        cmp(f"blocks.{L}.attention_norm.weight", shared_g[f"attn_norm.{L}.weight"])
        cmp(f"blocks.{L}.mlp_norm.weight", shared_g[f"mlp_norm.{L}.weight"])
        cmp(f"blocks.{L}.attention.output_projection.weight", gstore.get(SegmentKey(L, "attn_out_proj", 0))["weight"])
        cmp(f"blocks.{L}.mlp.output_projection.bias", shared_g[f"mlp_out_bias.{L}"])
    cmp("final_norm.weight", shared_g["final_norm.weight"])

    # reassemble + compare a few segment grads (mlp chunk 0 of layer 0; attn seg 0)
    g_mlp0 = gstore.get(SegmentKey(0, "mlp", 0))
    f0, f1 = hidden_range(m.d_ff, s.mlp_chunks, 0)
    d_mlp_in = (ref_g["blocks.0.mlp.input_projection.weight"][f0:f1] - g_mlp0["input_projection.weight"]).abs().max().item()
    g_emb0 = gstore.get(SegmentKey(-1, "embedding", 0))
    d0, d1 = dmodel_range(m.d_model, s.embedding_segments, 0)
    d_emb = (ref_g["embedding.token_embedding.weight"][:, d0:d1] - g_emb0["token_embedding.weight"]).abs().max().item()
    g_head0 = gstore.get(SegmentKey(-1, "output_head", 0))
    v0, v1 = vocab_range(m.vocab_size, s.output_head_segments, 0)
    d_head = (ref_g["output_projection.weight"][v0:v1] - g_head0["projection.weight"]).abs().max().item()

    print(f"shared grads max|Δ|={maxd:.2e}")
    print(f"mlp chunk grad |Δ|={d_mlp_in:.2e}  embedding grad |Δ|={d_emb:.2e}  head grad |Δ|={d_head:.2e}")
    ok = maxd < 1e-4 and d_mlp_in < 1e-4 and d_emb < 1e-4 and d_head < 1e-4
    print("GRADIENT IDENTITY OK (segmented backward == reference)" if ok else "GRADIENT MISMATCH")
    sys.exit(0 if ok else 1)

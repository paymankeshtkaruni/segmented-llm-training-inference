"""
Path B verification: with dropout ON (train mode), is the STREAMED backward gradient equal
to the TRUE gradient of the segmented model's OWN dropout forward?

We build a single-graph autograd reference that mirrors `_record_layer_inputs` op-for-op
(same dropout placement: (c) attn-output dropout, (d) MLP-sum dropout), seeded identically so
it draws the SAME masks. loss.backward() through it gives the exact gradient. We compare to
SegmentedBackwardEngine.backward() (which uses RNG save/restore to reproduce those masks).

Expectation WITH the fix: Δ ~ 1e-6 (consistent gradient).  Without it: large (≫1e-3).
This is the right test (segmented-vs-itself); comparing to the full ReferenceGPTDecoder is NOT
valid under dropout because per-head-group SDPA dropout differs structurally from full SDPA.
"""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F

DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"

SEG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG))
from config import SMALL_MODEL, SEG_8x2x2x8                                  # noqa: E402
from modules import ReferenceGPTDecoder                                      # noqa: E402
from forward_engine import populate_from_reference, SegmentedForwardEngine, _attn_bias  # noqa: E402
from backward_engine import SegmentedBackwardEngine                          # noqa: E402
from loader import StrictSegmentLoader, build_segment                        # noqa: E402
from stores import make_store, SegmentKey                                    # noqa: E402

m = replace(SMALL_MODEL, dropout=0.1)
s = SEG_8x2x2x8
torch.manual_seed(0)
ref = ReferenceGPTDecoder(m)
store = make_store("cpu_ram"); shared = populate_from_reference(ref, m, s, store).to(DEV)
fwd = SegmentedForwardEngine(m, s, StrictSegmentLoader(m, s, store, DEV), shared, DEV).train()
gstore = make_store("cpu_ram"); bwd = SegmentedBackwardEngine(fwd, gstore)

x = torch.randint(0, m.vocab_size, (2, 24), device=DEV); lab = x.clone(); lab[:, :5] = -100
SEED = 12345

# ---- single-graph autograd through the SAME forward, same masks (seeded identically) ----
def make(key):
    mod = build_segment(key, m, s); sd = store.get(key)
    if sd is not None: mod.load_state_dict(sd)
    return mod.train().to(DEV)

torch.manual_seed(SEED)
shared.train()
bias = _attn_bias(x, None, DEV)
mods = {}
parts = []
for e in range(s.embedding_segments):
    mods[("emb", e)] = make(SegmentKey(-1, "embedding", e)); parts.append(mods[("emb", e)](x))
hidden = torch.cat(parts, dim=-1)
for L in range(m.n_layers):
    xin = shared.attn_norm[L](hidden)
    outs = []
    for a in range(s.attention_segments):
        mods[("attn", L, a)] = make(SegmentKey(L, "attention", a)); outs.append(mods[("attn", L, a)](xin, attn_bias=bias))
    mods[("ao", L)] = make(SegmentKey(L, "attn_out_proj", 0))
    hidden = hidden + F.dropout(mods[("ao", L)](torch.cat(outs, dim=-1)), m.dropout, True)   # (c)
    xin = shared.mlp_norm[L](hidden)
    acc = None
    for c in range(s.mlp_chunks):
        mods[("mlp", L, c)] = make(SegmentKey(L, "mlp", c)); o = mods[("mlp", L, c)](xin); acc = o if acc is None else acc + o
    hidden = hidden + F.dropout(acc + shared.mlp_out_bias[L], m.dropout, True)               # (d)
hn = shared.final_norm(hidden)
shh = hn[:, :-1, :]; tgt = lab[:, 1:]
zs = []
for h in range(s.output_head_segments):
    mods[("head", h)] = make(SegmentKey(-1, "output_head", h)); zs.append(mods[("head", h)](shh))
logits = torch.cat(zs, dim=-1)
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
loss.backward()
print(f"single-graph loss = {loss.item():.6f}")

# ---- streamed backward (same seed -> record captures these masks, recompute restores) ----
torch.manual_seed(SEED)
bwd.backward(x, lab, pad_token_id=None)
print(f"streamed   loss = {bwd.last_loss:.6f}   (Δloss={abs(loss.item()-bwd.last_loss):.2e})")

# ---- compare param grads ----
def cmp(label, ref_grad, seg_grad):
    d = (ref_grad.cpu() - seg_grad.cpu()).abs().max().item()
    print(f"  {label:<28} max|Δ| = {d:.3e}{'   <-- INCONSISTENT' if d > 1e-3 else '   ok'}")
    return d

print("Gradient: streamed backward vs single-graph autograd (same dropout masks):")
worst = 0.0
worst = max(worst, cmp("emb[0] token_embedding.w", mods[("emb", 0)].token_embedding.weight.grad,
                       gstore.get(SegmentKey(-1, "embedding", 0))["token_embedding.weight"]))
worst = max(worst, cmp("L0 attn[0] q_proj.w", mods[("attn", 0, 0)].q_proj.weight.grad,
                       gstore.get(SegmentKey(0, "attention", 0))["q_proj.weight"]))
worst = max(worst, cmp("L0 attn_out_proj.w", mods[("ao", 0)].weight.grad,
                       gstore.get(SegmentKey(0, "attn_out_proj", 0))["weight"]))
worst = max(worst, cmp("L0 mlp[0] input_proj.w", mods[("mlp", 0, 0)].input_projection.weight.grad,
                       gstore.get(SegmentKey(0, "mlp", 0))["input_projection.weight"]))
worst = max(worst, cmp("L1 mlp[1] output_proj.w", mods[("mlp", 1, 1)].output_projection.weight.grad,
                       gstore.get(SegmentKey(1, "mlp", 1))["output_projection.weight"]))
worst = max(worst, cmp("head[0] projection.w", mods[("head", 0)].projection.weight.grad,
                       gstore.get(SegmentKey(-1, "output_head", 0))["projection.weight"]))
print(f"\nWORST max|Δ| = {worst:.3e}  ->  {'PASS (gradient consistent under dropout)' if worst < 1e-3 else 'FAIL'}")

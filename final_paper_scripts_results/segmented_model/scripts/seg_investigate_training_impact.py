"""
DEEP INVESTIGATION: which memory-reduction technique actually changes TRAINING?

Memory techniques split into two kinds:
  (1) pure memory-LAYOUT (where state lives): offload to store, empty_cache/malloc,
      opt-state offload, shared-grads-to-host, layer-input offload, attn_out_proj
      segmentation, streamed AdamW, two-pass streaming clip. These are gradient-exact
      by construction (verified by the eval-mode identity tests, Δ~1e-7) -> NO training impact.
  (2) the RECOMPUTE backward: recomputes the forward. If the forward is STOCHASTIC
      (dropout), the recompute draws DIFFERENT masks than the recorded forward -> the
      gradient is taken w.r.t. a different dropout realization than the loss -> training impact.

This script isolates (2): it measures segmented-vs-reference gradient agreement across the
2x2 grid {dropout 0 / 0.1} x {eval / train}. Expectation:
  - dropout=0 (either mode) and dropout=0.1+eval  -> Δ ~ 1e-7 (recompute is exact)
  - dropout=0.1 + TRAIN                            -> Δ LARGE (recompute mask != forward mask)
That single cell being the only one that breaks identity pinpoints recompute+dropout as the
sole training-impacting technique; all the others are layout-only.
"""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path
import torch

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from config import SMALL_MODEL, SEG_8x2x2x8                              # noqa: E402
from modules import ReferenceGPTDecoder, causal_lm_cross_entropy_loss   # noqa: E402
from forward_engine import populate_from_reference, SegmentedForwardEngine  # noqa: E402
from backward_engine import SegmentedBackwardEngine                     # noqa: E402
from loader import StrictSegmentLoader                                  # noqa: E402
from stores import make_store, SegmentKey                              # noqa: E402


def grad_delta(dropout, mode):
    m = replace(SMALL_MODEL, dropout=dropout)
    s = SEG_8x2x2x8
    torch.manual_seed(0)
    ref = ReferenceGPTDecoder(m)
    x = torch.randint(0, m.vocab_size, (2, 24)); lab = x.clone(); lab[:, :5] = -100
    store = make_store("cpu_ram"); shared = populate_from_reference(ref, m, s, store)
    fwd = SegmentedForwardEngine(m, s, StrictSegmentLoader(m, s, store, "cpu"), shared, "cpu")
    gstore = make_store("cpu_ram"); bwd = SegmentedBackwardEngine(fwd, gstore)

    train = (mode == "train")
    # reference grads (seed before forward so dropout is reproducible within this call)
    ref.train(train)
    torch.manual_seed(123)
    ref.zero_grad()
    logits, _ = ref(x, pad_token_id=None)
    causal_lm_cross_entropy_loss(logits, lab).backward()
    ref_attn0 = ref.blocks[0].attention.qkv_projection.weight.grad.clone()
    ref_mlp0 = ref.blocks[0].mlp.input_projection.weight.grad.clone()

    # segmented grads (same seed before its backward)
    fwd.train() if train else fwd.eval()
    torch.manual_seed(123)
    bwd.backward(x, lab, pad_token_id=None)
    # mlp chunk 0 of layer 0 (compare the input_projection slice)
    from segments import hidden_range
    f0, f1 = hidden_range(m.d_ff, s.mlp_chunks, 0)
    seg_mlp0 = gstore.get(SegmentKey(0, "mlp", 0))["input_projection.weight"]
    d_mlp = (ref_mlp0[f0:f1] - seg_mlp0).abs().max().item()
    return d_mlp


print("=== Gradient identity: segmented backward vs reference loss.backward() ===")
print(f"{'dropout':>8} {'mode':>6}   mlp-chunk grad max|Δ|")
for dropout in (0.0, 0.1):
    for mode in ("eval", "train"):
        d = grad_delta(dropout, mode)
        flag = "  <-- BREAKS identity (training impact)" if d > 1e-3 else "  (exact)"
        print(f"{dropout:>8} {mode:>6}   {d:.3e}{flag}")
print()
print("Reading: only (dropout=0.1, train) should break -> recompute+dropout is the only")
print("technique that changes training. All layout techniques stay exact (Δ~1e-7).")

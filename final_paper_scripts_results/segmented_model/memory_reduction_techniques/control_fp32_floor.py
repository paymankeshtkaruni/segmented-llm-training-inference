#!/usr/bin/env python
"""CONTROL: the reference full model compared against ITSELF, no segmentation anywhere.

Two mathematically identical ways to compute one training step, both pure PyTorch:
  A: one batch of 2 sequences
  B: two batches of 1 sequence with gradient accumulation (grads averaged)

Identical math, different execution order. Checks (same metrics as X1):
  - gradient max|D| over ALL parameters
  - one AdamW step from identical weights/optimizer state: worst element over all
    parameters, with clip (after-full protocol) and without clip (immediate protocol)
  - count of elements above thresholds and the |g| at those elements

Verdict: if this shows the same ~1e-8 grads / ~1e-5..1e-4 step pattern as the
segmented-vs-reference comparison, those numbers are fp32's own floor. If this is
clean while segmented is not, the segmented engine has a real bug.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "segmentation_management"))

from config import get_preset                                            # noqa: E402
from modules import ReferenceGPTDecoder, causal_lm_cross_entropy_loss    # noqa: E402

LR, WD, BETAS, EPS, CLIP = 3e-4, 0.1, (0.9, 0.95), 1e-8, 1.0
SEED_DATA = 1000


def _batch(m, batch, seq):
    g = torch.Generator().manual_seed(SEED_DATA)
    x = torch.randint(0, m.vocab_size, (batch, seq), generator=g)
    lab = x.clone()
    lab[:, :5] = -100
    return x, lab


def _make_opt(model):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                              {"params": nodecay, "weight_decay": 0.0}],
                             lr=LR, betas=BETAS, eps=EPS)


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def run(device, batch, seq, use_clip):
    preset = get_preset("large_8x2x2x8")
    m = replace(preset["model"], dropout=0.0)
    torch.manual_seed(0)
    A = ReferenceGPTDecoder(m).to(device)
    B = copy.deepcopy(A)                       # identical weights, bit for bit
    A.train(); B.train()
    x, lab = _batch(m, batch, seq)
    x, lab = x.to(device), lab.to(device)

    # --- way A: one batch of `batch` --------------------------------------------
    A.zero_grad(set_to_none=False)
    logits, _ = A(x, pad_token_id=None)
    causal_lm_cross_entropy_loss(logits, lab).backward()
    del logits

    # --- way B: `batch` micro-batches of 1, gradients averaged ------------------
    B.zero_grad(set_to_none=False)
    for i in range(batch):
        logits, _ = B(x[i:i + 1], pad_token_id=None)
        (causal_lm_cross_entropy_loss(logits, lab[i:i + 1]) / batch).backward()
        del logits

    gA, gB = grads_of(A), grads_of(B)
    gd = {k: (gA[k] - gB[k]).abs().max().item() for k in gA}
    worst_g = max(gd, key=gd.get)

    # --- one AdamW step each, same protocol -------------------------------------
    optA, optB = _make_opt(A), _make_opt(B)
    if use_clip:
        torch.nn.utils.clip_grad_norm_(A.parameters(), CLIP)
        torch.nn.utils.clip_grad_norm_(B.parameters(), CLIP)
    optA.step(); optB.step()

    worst, worst_name, n5, n6, gmax5 = 0.0, "", 0, 0, 0.0
    pB = dict(B.named_parameters())
    for k, p in A.named_parameters():
        d = (p.detach() - pB[k].detach()).abs()
        mx = d.max().item()
        if mx > worst:
            worst, worst_name = mx, k
        m5 = d > 1e-5
        n5 += int(m5.sum().item()); n6 += int((d > 1e-6).sum().item())
        if m5.any():
            gmax5 = max(gmax5, gA[k].abs()[m5].max().item())
    return {"device": device, "protocol": "clip" if use_clip else "no_clip",
            "grad_max_delta": gd[worst_g], "grad_worst_param": worst_g,
            "step_worst_delta": worst, "step_worst_param": worst_name,
            "n_elements_gt_1e-5": n5, "n_elements_gt_1e-6": n6,
            "max_|g|_at_gt_1e-5": gmax5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "verify_modes")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for use_clip in (True, False):
        r = run(a.device, a.batch, a.seq, use_clip)
        results.append(r)
        print(f"[{r['protocol']:8}] grads max|D|={r['grad_max_delta']:.3e} ({r['grad_worst_param']})")
        print(f"[{r['protocol']:8}] step  max|D|={r['step_worst_delta']:.3e} ({r['step_worst_param']})  "
              f">1e-5: {r['n_elements_gt_1e-5']}  >1e-6: {r['n_elements_gt_1e-6']}  "
              f"max|g| at >1e-5: {r['max_|g|_at_gt_1e-5']:.3e}", flush=True)
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    tag = "gpu" if a.device.startswith("cuda") else "cpu"
    out = a.out_dir / f"control_fp32_floor_{tag}.json"
    json.dump({"run": "control_fp32_floor",
               "note": "full model vs itself: batch-of-2 vs 2x gradient accumulation; "
                       "pure PyTorch, zero segmentation code involved",
               "results": results}, open(out, "w"), indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

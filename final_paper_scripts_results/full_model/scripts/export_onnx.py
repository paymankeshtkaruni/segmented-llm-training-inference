#!/usr/bin/env python
"""ONE-TIME export of the LARGE model to ONNX (uses torch — a build step only).

The produced `model.onnx` is consumed by the torch-free C inference-cost scripts.
This export's own resource use is NOT part of C's measured cost.

Light to run (even on the login node): the export trace uses a tiny dummy input
with DYNAMIC axes, so only the weights (~3.3 GB) are serialized, not a seq-512
forward. The cost batches no padding -> we export with pad_token_id=None (full
causal attention), giving a simpler graph (input_ids -> logits).

Output: full_model/onnx/large_model.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import _common as C


class OnnxWrapper(nn.Module):
    """input_ids -> logits  (no padding mask; full causal attention)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _ = self.model(input_ids, pad_token_id=None)
        return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.FULL_MODEL_DIR / "onnx" / "large_model.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = C.LARGE_CONFIG
    tok = C.build_tokenizer(cfg)            # GPT-2
    vocab = tok.vocab_size
    print(f"[export] LARGE model: d_model={cfg['d_model']} n_layers={cfg['n_layers']} "
          f"vocab={vocab}  -> {args.out}")
    model = C.build_model(vocab, cfg).eval()
    print(f"[export] params={C.count_params(model)/1e6:.1f}M  (~{C.count_params(model)*4/1e9:.1f} GB fp32)")

    wrapper = OnnxWrapper(model).eval()
    dummy = torch.randint(0, vocab, (1, 16), dtype=torch.long)  # tiny trace input
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), str(args.out),
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "logits": {0: "batch", 1: "seq"}},
            opset_version=args.opset, do_constant_folding=True,
        )
    size_gb = args.out.stat().st_size / 1e9
    print(f"[export] done -> {args.out}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()

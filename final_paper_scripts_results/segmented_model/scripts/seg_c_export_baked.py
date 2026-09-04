#!/usr/bin/env python
"""Export the two RESIDENT ONNX artifacts for the inference-cost mode table.

--what full   (O1): the whole model as ONE ONNX session, weights baked. At 0.84B
               this exceeds onnx's 2GB single-file limit, so initializers are
               written as external data alongside the graph.
--what baked  (O2): one ONNX file PER SEGMENT with its weights baked (no
               weights-as-inputs lifting, no dedup), plus the same glue.npz the
               weights-as-inputs runtime uses (norms, W_o, biases). manifest.json
               maps segment name -> onnx file.

Weights are the seeded random init (cost measurements only).
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path

import numpy as np
import torch
import onnx

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from config import get_preset                                   # noqa: E402
from modules import ReferenceGPTDecoder                         # noqa: E402
from forward_engine import populate_from_reference              # noqa: E402
from loader import build_segment                                # noqa: E402
from stores import make_store, SegmentKey                       # noqa: E402
from optimizer import all_segment_keys                          # noqa: E402

KINDS_ACT = {"embedding": "input_ids", "attention": "x_norm",
             "mlp": "x_norm", "output_head": "hidden_norm"}


class _LogitsOnly(torch.nn.Module):
    def __init__(self, ref):
        super().__init__()
        self.ref = ref

    def forward(self, input_ids):
        logits, _ = self.ref(input_ids, pad_token_id=None)
        return logits


def export_full(m, ref, out_dir: Path, opset: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "full_model.onnx"
    wrap = _LogitsOnly(ref).eval()
    dummy = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(wrap, (dummy,), str(path),
                          input_names=["input_ids"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "b", 1: "t"},
                                        "logits": {0: "b", 1: "t"}},
                          opset_version=opset, do_constant_folding=True)
    # >2GB: rewrite with initializers as external data next to the graph
    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="full_model.onnx.data", size_threshold=1024)
    json.dump({"model": "full_model.onnx", "vocab_size": m.vocab_size,
               "max_seq_len": m.max_seq_len}, open(out_dir / "config.json", "w"))
    sz = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e9
    print(f"[export full] -> {out_dir}  ({sz:.2f} GB)")


def export_baked(m, s, ref, out_dir: Path, opset: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    store = make_store("cpu_ram")
    shared = populate_from_reference(ref, m, s, store)

    manifest = {"segments": {}}
    for key in all_segment_keys(m, s):
        if key.kind == "attn_out_proj":
            continue                                   # applied via glue (numpy)
        seg = build_segment(key, m, s)
        seg.load_state_dict(store.get(key), strict=True)
        seg.eval()
        name = f"L{key.layer_id}__{key.kind}__s{key.seg}"
        path = seg_dir / f"{name}.onnx"
        act = KINDS_ACT[key.kind]
        dummy = (torch.zeros(1, 4, dtype=torch.long) if key.kind == "embedding"
                 else torch.zeros(1, 4, m.d_model))
        with torch.no_grad():
            torch.onnx.export(seg, (dummy,), str(path),
                              input_names=[act], output_names=["out"],
                              dynamic_axes={act: {0: "b", 1: "t"},
                                            "out": {0: "b", 1: "t"}},
                              opset_version=opset, do_constant_folding=True)
        manifest["segments"][name] = {"kind": key.kind, "onnx": str(path),
                                      "activation_input": act,
                                      "layer": key.layer_id, "seg": key.seg}
        del seg
        gc.collect()

    glue = {}
    sp = dict(shared.named_parameters())
    for L in range(m.n_layers):
        glue[f"attn_ln.{L}.weight"] = sp[f"attn_norm.{L}.weight"].detach().numpy()
        glue[f"attn_ln.{L}.bias"] = sp[f"attn_norm.{L}.bias"].detach().numpy()
        glue[f"mlp_ln.{L}.weight"] = sp[f"mlp_norm.{L}.weight"].detach().numpy()
        glue[f"mlp_ln.{L}.bias"] = sp[f"mlp_norm.{L}.bias"].detach().numpy()
        glue[f"mlp_out_bias.{L}"] = sp[f"mlp_out_bias.{L}"].detach().numpy()
        wo = store.get(SegmentKey(L, "attn_out_proj", 0))
        glue[f"Wo.{L}.weight"] = wo["weight"].numpy()
        glue[f"Wo.{L}.bias"] = wo["bias"].numpy()
    glue["final_norm.weight"] = sp["final_norm.weight"].detach().numpy()
    glue["final_norm.bias"] = sp["final_norm.bias"].detach().numpy()
    np.savez(out_dir / "glue.npz", **glue)
    json.dump({"d_model": m.d_model, "n_layers": m.n_layers,
               "vocab_size": m.vocab_size, "max_seq_len": m.max_seq_len,
               "embedding_segments": s.embedding_segments,
               "attention_segments": s.attention_segments,
               "mlp_chunks": s.mlp_chunks,
               "output_head_segments": s.output_head_segments},
              open(out_dir / "config.json", "w"))
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)
    n = len(manifest["segments"])
    print(f"[export baked] {n} segment sessions -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--what", required=True, choices=["full", "baked"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args()
    p = get_preset(a.preset)
    m, s = p["model"], p["seg"]
    torch.manual_seed(0)
    ref = ReferenceGPTDecoder(m).eval()
    if a.what == "full":
        export_full(m, ref, a.out_dir, a.opset)
    else:
        export_baked(m, s, ref, a.out_dir, a.opset)


if __name__ == "__main__":
    main()

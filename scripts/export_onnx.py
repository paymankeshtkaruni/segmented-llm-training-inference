#!/usr/bin/env python3
"""Export a trained segmented checkpoint to torch-free ONNX artifacts.

Torch is required here (one-time offline step). After this runs, torch is no
longer needed for inference — use scripts/infer_onnx.py.

The attention_output_proj segments are NOT exported as ONNX. Instead their
weights (proj_w.{L} / proj_b.{L}) are written into glue.npz so the inference
script can do the chunked W_o numpy matmul without ever concatenating all head
outputs into a single tensor.

Usage:
    PYTHONPATH=src python scripts/export_onnx.py \\
        --checkpoint-dir experiments/checkpoints/t02_cpu_infer \\
        --checkpoint-name best \\
        --out-dir onnx_export
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from torch import nn

from scripts.infer_segmented import load_model_from_checkpoint


# ── ONNX export wrappers ─────────────────────────────────────────────────────

class _AttnWrapper(nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, attention_input):
        return self.seg(attention_input, None, causal=True)


def _export_segment(module: nn.Module, seg_id, model_config, out_path: Path) -> None:
    seg_type = seg_id.segment_type
    D = model_config.d_model
    B, S = 2, 8

    module = module.eval()
    with torch.no_grad():
        if seg_type == "embedding":
            input_ids = torch.randint(0, model_config.vocab_size, (B, S), dtype=torch.long)
            pos_ids   = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S).contiguous()
            torch.onnx.export(
                module, (input_ids, pos_ids), str(out_path),
                input_names=["input_ids", "position_ids"],
                output_names=["emb_slice"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                              "position_ids": {0: "batch", 1: "seq"},
                              "emb_slice": {0: "batch", 1: "seq"}},
                opset_version=17, do_constant_folding=True,
            )

        elif seg_type == "attention":
            wrapper = _AttnWrapper(module).eval()
            x = torch.randn(B, S, D)
            torch.onnx.export(
                wrapper, (x,), str(out_path),
                input_names=["attention_input"], output_names=["attention_head_out"],
                dynamic_axes={"attention_input": {0: "batch", 1: "seq"},
                              "attention_head_out": {0: "batch", 1: "seq"}},
                opset_version=17, do_constant_folding=True,
            )

        elif seg_type == "mlp":
            x = torch.randn(B, S, D)
            torch.onnx.export(
                module, (x,), str(out_path),
                input_names=["mlp_input"], output_names=["mlp_out"],
                dynamic_axes={"mlp_input": {0: "batch", 1: "seq"},
                              "mlp_out": {0: "batch", 1: "seq"}},
                opset_version=17, do_constant_folding=True,
            )

        elif seg_type == "output_head":
            x = torch.randn(B, S, D)
            torch.onnx.export(
                module, (x,), str(out_path),
                input_names=["hidden_norm"], output_names=["logits_slice"],
                dynamic_axes={"hidden_norm": {0: "batch", 1: "seq"},
                              "logits_slice": {0: "batch", 1: "seq"}},
                opset_version=17, do_constant_folding=True,
            )

        else:
            raise ValueError(f"Unexpected segment_type for ONNX export: {seg_type!r}")


# ── weights-as-inputs surgery (same as export_segments_winputs.py) ───────────

def _lift_initializers(model: onnx.ModelProto):
    g = model.graph
    weights = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    del g.initializer[:]
    for name, arr in weights.items():
        dtype = helper.np_dtype_to_tensor_dtype(arr.dtype)
        g.input.append(helper.make_tensor_value_info(name, dtype, list(arr.shape)))
    return model, weights


def _graph_sig(model: onnx.ModelProto) -> str:
    g = model.graph
    h = hashlib.sha256()
    for node in g.node:
        h.update(node.op_type.encode())
        for x in list(node.input) + list(node.output):
            h.update(x.encode())
        h.update(node.SerializeToString())
    for vi in list(g.input) + list(g.output):
        h.update(vi.SerializeToString())
    return h.hexdigest()[:16]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir",  required=True)
    ap.add_argument("--checkpoint-name", default="best")
    ap.add_argument("--out-dir",         default="onnx_export")
    ap.add_argument("--device",          default="cpu")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir  = Path(args.out_dir)
    winputs_dir  = out_dir / "segments_winputs"
    weights_dir  = out_dir / "segment_weights"
    for d in [winputs_dir, weights_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("export_onnx.py — segmented checkpoint → torch-free ONNX artifacts")
    print(f"  checkpoint : {ckpt_dir} / {args.checkpoint_name}")
    print(f"  out_dir    : {out_dir}")

    device = torch.device(args.device)
    forward_engine, loader, store, model_config, seg_config = load_model_from_checkpoint(
        ckpt_dir, args.checkpoint_name, device, storage="disk"
    )
    forward_engine.eval()

    D         = model_config.d_model
    n_layers  = model_config.n_layers
    n_attn    = seg_config.attention_segments
    n_mlp     = seg_config.mlp_chunks
    n_emb     = seg_config.embedding_segments
    n_head    = seg_config.output_head_segments

    # ── 1. Export segments (skip attention_output_proj) ───────────────────────
    seg_ids = sorted(store.list_segments())
    factory = loader.module_factory

    manifest = {"segments": {}, "signatures": {}}
    sig_to_weightless: dict[str, str] = {}

    print(f"\n[1] Exporting {len(seg_ids)} segments (skipping attn_output_proj → goes to glue) ...")

    with torch.no_grad():
        for seg_id in seg_ids:
            name = seg_id.to_path_name()

            if seg_id.segment_type == "attention_output_proj":
                print(f"    {name:48s} → glue (chunked W_o, not ONNX)")
                continue

            module = factory(seg_id)
            state  = store.load_segment(seg_id)
            module.load_state_dict(state, strict=True)
            module.eval()

            # Baked ONNX (temporary — we lift weights out next)
            baked_path = out_dir / f"_tmp_{name}.onnx"
            _export_segment(module, seg_id, model_config, baked_path)
            del module, state
            gc.collect()

            # Lift initializers → weightless graph + separate weight files
            onnx_model = onnx.load(str(baked_path))
            onnx_model, weights = _lift_initializers(onnx_model)
            baked_path.unlink()

            weight_names = list(weights.keys())
            all_inputs   = [i.name for i in onnx_model.graph.input]
            act_inputs   = [n for n in all_inputs if n not in set(weight_names)]

            sig = _graph_sig(onnx_model)

            weightless_path = str(winputs_dir / f"{name}.onnx")
            onnx.save(onnx_model, weightless_path)

            # weights npz + per-weight npy (mmappable)
            npz_path = str(weights_dir / f"{name}.npz")
            np.savez(npz_path, **weights)
            seg_npy_dir = weights_dir / name
            seg_npy_dir.mkdir(exist_ok=True)
            npy_files = {}
            for wi, (wname, warr) in enumerate(weights.items()):
                fn = str(seg_npy_dir / f"w{wi}.npy")
                np.save(fn, np.ascontiguousarray(warr))
                npy_files[wname] = fn

            if sig not in sig_to_weightless:
                sig_to_weightless[sig] = weightless_path
                manifest["signatures"][sig] = {
                    "representative_weightless_onnx": weightless_path,
                    "activation_inputs": act_inputs,
                    "weight_input_names": weight_names,
                    "members": [],
                }
            manifest["signatures"][sig]["members"].append(name)

            manifest["segments"][name] = {
                "weightless_onnx":   weightless_path,
                "weights_npz":       npz_path,
                "weights_npy":       npy_files,
                "signature":         sig,
                "activation_inputs": act_inputs,
                "weight_input_names": weight_names,
                "weights_bytes":     int(sum(a.nbytes for a in weights.values())),
            }

            mb = sum(a.nbytes for a in weights.values()) / 1024**2
            print(f"    {name:48s}  sig={sig}  {mb:.2f} MB")

    # ── 2. Build glue.npz ─────────────────────────────────────────────────────
    print("\n[2] Saving glue.npz (layer norms, mlp biases, final norm, W_o) ...")
    glue: dict[str, np.ndarray] = {}

    ln_sd = forward_engine.layer_norms.state_dict()
    for L in range(n_layers):
        glue[f"attn_ln.{L}.weight"] = ln_sd[f"attention_layer_norms.{L}.weight"].cpu().numpy()
        glue[f"attn_ln.{L}.bias"]   = ln_sd[f"attention_layer_norms.{L}.bias"].cpu().numpy()
        glue[f"mlp_ln.{L}.weight"]  = ln_sd[f"mlp_layer_norms.{L}.weight"].cpu().numpy()
        glue[f"mlp_ln.{L}.bias"]    = ln_sd[f"mlp_layer_norms.{L}.bias"].cpu().numpy()

    mlp_bias_sd = forward_engine.mlp_shared_output_biases.state_dict()
    for L in range(n_layers):
        glue[f"mlp_shared_bias.{L}"] = mlp_bias_sd[f"biases.{L}"].cpu().numpy()

    fn_sd = forward_engine.output_slice_final_norm.state_dict()
    glue["final_norm.weight"] = fn_sd["weight"].cpu().numpy()
    glue["final_norm.bias"]   = fn_sd["bias"].cpu().numpy()

    # W_o per layer from attention_output_proj segments (chunked matmul in infer_onnx)
    from sequential_segmented_llm_training_inference.segments.segment_ids import SegmentId
    for L in range(n_layers):
        proj_id = SegmentId(layer_id=L, segment_type="attention_output_proj", segment_id=0)
        proj_module = factory(proj_id)
        proj_state  = store.load_segment(proj_id)
        proj_module.load_state_dict(proj_state, strict=True)
        glue[f"proj_w.{L}"] = proj_module.proj.weight.detach().cpu().numpy()   # [D, D]
        if proj_module.proj.bias is not None:
            glue[f"proj_b.{L}"] = proj_module.proj.bias.detach().cpu().numpy() # [D]
        else:
            glue[f"proj_b.{L}"] = np.zeros(D, dtype=np.float32)
        del proj_module, proj_state

    glue_path = out_dir / "glue.npz"
    np.savez(str(glue_path), **glue)
    glue_mb = glue_path.stat().st_size / 1024**2
    print(f"    glue.npz  {glue_mb:.2f} MB  ({len(glue)} arrays)")

    # ── 3. config.json ────────────────────────────────────────────────────────
    config = {
        "n_layers":           n_layers,
        "d_model":            D,
        "n_heads":            model_config.n_heads,
        "vocab_size":         model_config.vocab_size,
        "max_seq_len":        model_config.max_seq_len,
        "n_emb_segs":         n_emb,
        "n_head_segs":        n_head,
        "attention_segments": n_attn,
        "mlp_chunks":         n_mlp,
        "layer_norm_eps":     1e-5,
        "pad_token_id":       model_config.pad_token_id,
        "eos_token_id":       model_config.eos_token_id,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ── 4. manifest.json ──────────────────────────────────────────────────────
    manifest["n_segments"]          = len(manifest["segments"])
    manifest["n_unique_signatures"] = len(sig_to_weightless)
    manifest["config"]              = config
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[3] config.json: {config}")
    print(f"\n[4] manifest.json: {len(manifest['segments'])} segments, "
          f"{len(sig_to_weightless)} unique signatures")
    for sig, info in manifest["signatures"].items():
        print(f"    sig {sig}: {len(info['members']):2d} members  act={info['activation_inputs']}")

    print(f"\nDone. All artifacts in {out_dir}/")


if __name__ == "__main__":
    main()

"""
C Step 1 — export the SEGMENTED model to torch-free ONNX artifacts (T03/T06 technique).

Per segment TYPE, export its forward to ONNX, then weights-as-inputs surgery
(`_lift_initializers`: initializers -> graph inputs) -> a WEIGHTLESS .onnx. Dedup by
graph signature (identical-shaped segments share one graph -> ~4 unique). Per-segment
weights saved as .npz; shared pieces (layer norms, mlp out-bias, final norm, W_o per
layer) -> glue.npz (applied torch-free in numpy at inference; attn_out_proj is a chunked
numpy matmul, NOT ONNX). Writes manifest.json + config.json.

Built-in VERIFY: each signature's weightless ONNX (weights fed) == its torch segment
forward (identity) — so the runtime is exact. Torch is used HERE (build step only); the
inference-cost runs (seg_c_onnx_cost) import no torch.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
import onnx
from onnx import helper, numpy_helper

SEG_PKG = Path(__file__).resolve().parent.parent / "segmentation_management"
sys.path.insert(0, str(SEG_PKG))
from config import get_preset                                   # noqa: E402
from modules import ReferenceGPTDecoder                         # noqa: E402
from forward_engine import populate_from_reference              # noqa: E402
from loader import build_segment                                # noqa: E402
from stores import make_store, SegmentKey                       # noqa: E402
from optimizer import all_segment_keys                          # noqa: E402

KINDS = ["embedding", "attention", "mlp", "output_head"]        # attn_out_proj -> glue
ACT = {"embedding": "input_ids", "attention": "x_norm", "mlp": "x_norm", "output_head": "hidden_norm"}


def _lift_initializers(model):
    g = model.graph
    weights = {init.name: numpy_helper.to_array(init).copy() for init in g.initializer}
    del g.initializer[:]
    for name, arr in weights.items():
        # SYMBOLIC dims so one weightless graph serves segments of UNEVEN size
        # (e.g. output_head vocab slices 6282 vs 6283 when vocab % H != 0).
        dims = [f"{name}_d{i}" for i in range(arr.ndim)]
        g.input.append(helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(arr.dtype), dims))
    # make graph outputs fully dynamic too (let ORT infer from the fed weights)
    for o in g.output:
        for i, dim in enumerate(o.type.tensor_type.shape.dim):
            dim.ClearField("dim_value"); dim.dim_param = f"{o.name}_d{i}"
    return model, weights


def _sig(model):
    h = hashlib.sha256()
    for node in model.graph.node:
        h.update(node.op_type.encode())
    for vi in list(model.graph.input) + list(model.graph.output):
        h.update(vi.name.encode())
    return h.hexdigest()[:16]


def _rep_key(kind):
    return {"embedding": SegmentKey(-1, "embedding", 0),
            "attention": SegmentKey(0, "attention", 0),
            "mlp": SegmentKey(0, "mlp", 0),
            "output_head": SegmentKey(-1, "output_head", 0)}[kind]


def _dummy(kind, m):
    return (torch.zeros(1, 4, dtype=torch.long) if kind == "embedding"
            else torch.zeros(1, 4, m.d_model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="large_8x2x2x8")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args()
    sig_dir = a.out_dir / "signatures"; w_dir = a.out_dir / "weights"
    for d in (sig_dir, w_dir):
        d.mkdir(parents=True, exist_ok=True)

    p = get_preset(a.preset); m, s = p["model"], p["seg"]
    torch.manual_seed(0)
    ref = ReferenceGPTDecoder(m).eval()
    store = make_store("cpu_ram")
    shared = populate_from_reference(ref, m, s, store)

    manifest = {"signatures": {}, "segments": {}}
    print(f"[c-export-seg] {a.preset}  {m.n_params_estimate/1e6:.0f}M  -> {a.out_dir}")

    # ---- 1+2. export EACH segment, lift weights (onnx-named, transposed as the graph
    #            bakes them), save per-segment npz; save the weightless graph once per
    #            signature (identical-shape segments share it). ----
    def export_seg(key):
        kind = key.kind
        seg = build_segment(key, m, s); seg.load_state_dict(store.get(key), strict=True); seg.eval()
        tmp = a.out_dir / f"_tmp_{kind}_{key.layer_id}_{key.seg}.onnx"
        with torch.no_grad():
            torch.onnx.export(
                seg, (_dummy(kind, m),), str(tmp),
                input_names=[ACT[kind]], output_names=["out"],
                dynamic_axes={ACT[kind]: {0: "b", 1: "t"}, "out": {0: "b", 1: "t"}},
                opset_version=a.opset, do_constant_folding=True)
        model = onnx.load(str(tmp)); model, weights = _lift_initializers(model); tmp.unlink()
        del seg; gc.collect()
        return model, weights

    n = 0
    for key in all_segment_keys(m, s):
        if key.kind not in KINDS:
            continue
        model, weights = export_seg(key)
        sig = _sig(model)
        name = f"L{key.layer_id}__{key.kind}__s{key.seg}"
        np.savez(str(w_dir / f"{name}.npz"), **weights)            # onnx-named (lifted) weights
        if sig not in manifest["signatures"]:                       # save weightless graph once/sig
            wl = sig_dir / f"{key.kind}.onnx"; onnx.save(model, str(wl))
            act_inputs = [i.name for i in model.graph.input if i.name not in weights]
            manifest["signatures"][sig] = {
                "kind": key.kind, "weightless_onnx": str(wl), "signature": sig,
                "activation_inputs": act_inputs, "weight_input_names": list(weights.keys())}
            print(f"  sig {key.kind:12} {sig}  act={act_inputs}  weights={list(weights.keys())}")
        manifest["segments"][name] = {"kind": key.kind, "signature": sig,
                                      "weights_npz": str(w_dir / f"{name}.npz"),
                                      "layer": key.layer_id, "seg": key.seg}
        del model; n += 1
    print(f"  exported {n} segments, {len(manifest['signatures'])} unique signatures")

    # ---- 3. glue.npz (norms, mlp_out_bias, final_norm, W_o per layer) ----
    glue = {}
    sp = dict(shared.named_parameters())
    for L in range(m.n_layers):
        glue[f"attn_ln.{L}.weight"] = sp[f"attn_norm.{L}.weight"].detach().numpy()
        glue[f"attn_ln.{L}.bias"]   = sp[f"attn_norm.{L}.bias"].detach().numpy()
        glue[f"mlp_ln.{L}.weight"]  = sp[f"mlp_norm.{L}.weight"].detach().numpy()
        glue[f"mlp_ln.{L}.bias"]    = sp[f"mlp_norm.{L}.bias"].detach().numpy()
        glue[f"mlp_out_bias.{L}"]   = sp[f"mlp_out_bias.{L}"].detach().numpy()
        op = store.get(SegmentKey(L, "attn_out_proj", 0))
        glue[f"Wo.{L}.weight"] = op["weight"].numpy()
        glue[f"Wo.{L}.bias"]   = op["bias"].numpy()
    glue["final_norm.weight"] = sp["final_norm.weight"].detach().numpy()
    glue["final_norm.bias"]   = sp["final_norm.bias"].detach().numpy()
    np.savez(str(a.out_dir / "glue.npz"), **glue)

    config = {"n_layers": m.n_layers, "d_model": m.d_model, "n_heads": m.n_heads,
              "vocab_size": m.vocab_size, "max_seq_len": m.max_seq_len,
              "embedding_segments": s.embedding_segments, "attention_segments": s.attention_segments,
              "mlp_chunks": s.mlp_chunks, "output_head_segments": s.output_head_segments,
              "preset": a.preset}
    json.dump(config, open(a.out_dir / "config.json", "w"), indent=2)
    json.dump(manifest, open(a.out_dir / "manifest.json", "w"), indent=2)

    # ---- 4. VERIFY: weightless ONNX (fed the LIFTED npz weights) == torch segment.
    #          Test the rep of each kind + a NON-rep segment (confirms onnx weight names
    #          are consistent across segments of a signature). ----
    import onnxruntime as ort
    kind2onnx = {info["kind"]: info["weightless_onnx"] for info in manifest["signatures"].values()}
    test_keys = [_rep_key(k) for k in KINDS]
    if m.n_layers > 1:
        test_keys.append(SegmentKey(1, "attention", 0))             # non-representative layer
        test_keys.append(SegmentKey(1, "mlp", 0))
    print("[verify] weightless ONNX (lifted weights) == torch segment:")
    g = torch.Generator().manual_seed(1)
    maxd = 0.0
    for key in test_keys:
        kind = key.kind
        name = f"L{key.layer_id}__{key.kind}__s{key.seg}"
        seg = build_segment(key, m, s); seg.load_state_dict(store.get(key), strict=True); seg.eval()
        if kind == "embedding":
            x = torch.randint(0, m.vocab_size, (2, 16), generator=g)
            xnp = x.numpy()
        else:
            x = torch.randn(2, 16, m.d_model, generator=g)
            xnp = x.numpy().astype(np.float32)
        with torch.no_grad():
            t_out = seg(x).numpy()
        weights = dict(np.load(str(w_dir / f"{name}.npz")))         # lifted (onnx-named) weights
        feeds = {ACT[kind]: xnp, **weights}
        sess = ort.InferenceSession(kind2onnx[kind], providers=["CPUExecutionProvider"])
        o_out = sess.run(["out"], feeds)[0]
        d = float(np.abs(t_out - o_out).max()); maxd = max(maxd, d)
        print(f"  {name:22} max|Δ|={d:.2e}  shape={o_out.shape}")
    ok = maxd < 1e-4
    print(f"[verify] {'OK — segmented ONNX == torch' if ok else 'MISMATCH'}  max|Δ|={maxd:.2e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

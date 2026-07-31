#!/usr/bin/env python
"""
Aggregate the scale experiment (0.84B / 3.09B / 6.86B, partition fixed 8x2x2x8, all
techniques ON) into ONE small JSON: per-scale full-vs-segmented peak memory + time
for train/infer on GPU/CPU (including the RECORDED full-model OOMs), plus the fitted
performance model:

  time  :  T_step  ~=  c_overhead + c_scale * P     (c_overhead = N*alpha', scale-free)
  memory:  M_peak  ~=  ctx + 2*s_max + c_act * d_model   (total P absent by design)

0.84B references come from the published ladder/full-model results (same protocol:
batch 4, seq 512, warmup + measured steps); 3B/7B from results/scale_*/.
-> results/scale_summary.json   (small; committed — the raw per-run *_metrics.json
   timelines are large and git-ignored)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ROOT = HERE.parents[1]                                    # final_paper_scripts_results
sys.path.insert(0, str(HERE / "segmentation_management"))
from config import get_preset                             # noqa: E402


def L(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def s_max_mb(m, s) -> float:
    """Largest single segment (fp32 weight bytes, MB): emb slice / attn head-group /
    W_o / MLP chunk / output-head vocab slice."""
    emb = m.vocab_size * (m.d_model // s.embedding_segments) * 4
    attn = 3 * m.d_model * (m.d_model // s.attention_segments) * 4
    wo = m.d_model * m.d_model * 4
    mlp = 2 * m.d_model * (m.d_ff // s.mlp_chunks) * 4
    head = -(-m.vocab_size // s.output_head_segments) * m.d_model * 4
    return max(emb, attn, wo, mlp, head) / 1e6


def n_seg(m, s) -> int:
    """Distinct segments = E + L*(A + 1 + M) + H  (the +1 is W_o)."""
    return (s.embedding_segments + m.n_layers * (s.attention_segments + 1 + s.mlp_chunks)
            + s.output_head_segments)


def _scale_entry(tag, preset):
    """Read one scale point's 4 runs from results/scale_{gpu,cpu}_{tag}/."""
    out = {}
    for dev in ("gpu", "cpu"):
        d = {}
        for run in ("full_train", "full_infer", "seg_train", "seg_infer"):
            met = L(RES / f"scale_{dev}_{tag}" / f"{run}_metrics.json")
            if met is None:
                raise SystemExit(
                    f"missing {RES/f'scale_{dev}_{tag}'/f'{run}_metrics.json'} — run the four "
                    f"slurm/mrt_{{gpu,cpu}}_scale_{{3b,7b}}.sbatch jobs first (README section 4d). "
                    f"To only rebuild the figures, use scale_figures.py, which reads the committed "
                    f"results/scale_summary.json.")
            if met.get("oom"):
                d[run] = {"oom": True, "oom_stage": met["oom_stage"],
                          "device_total_mb": met.get("device_total_mb"),
                          "vram_alloc_at_oom_mb": met.get("vram_alloc_at_oom_mb")}
                continue
            o = met.get("overall", {})
            mem = o.get("vram_hw_total_peak_mb") if dev == "gpu" else \
                (o.get("rss_peak_sampled_mb") or o.get("rss_hw_peak_mb"))
            t = met.get("avg_step_time_s") or met.get("per_token_s")
            d[run] = {"oom": False, "mem_mb": mem,
                      ("per_token_s" if run == "seg_infer" else "time_s"): round(t, 3)}
            if run == "seg_train" and "baseline" in met:
                d[run]["cuda_context_mb"] = met["baseline"].get("cuda_context_mb")
        out[dev] = d
    p = get_preset(preset)
    m, s = p["model"], p["seg"]
    out["preset"] = preset
    out["params_billion"] = round(m.n_params_estimate / 1e9, 3)
    out["d_model"] = m.d_model
    out["n_layers"] = m.n_layers
    out["s_max_mb"] = round(s_max_mb(m, s), 1)
    out["n_segments"] = n_seg(m, s)
    return out


def _cpu_084(run, tkey):
    """One 0.84B CPU cell, from the dedicated CPU scale runs (results/scale_cpu_large/).

    These replace the older results/comparison/cost_compare.json + cpu_{train,inference}
    ladder cells: scale_cpu_large was measured under the SAME pinned-16-thread / identical
    allocator environment as the 3.09B and 6.86B CPU points, so the CPU column of the
    scale table is now internally consistent (one environment, one protocol)."""
    m = L(RES / "scale_cpu_large" / f"{run}_metrics.json")
    if m is None:
        raise SystemExit(f"missing {RES/'scale_cpu_large'/f'{run}_metrics.json'} — run the "
                         f"0.84B CPU scale jobs first (slurm/mrt_cpu_scale_large.sbatch).")
    o = m["overall"]
    t = m.get("avg_step_time_s") if tkey == "time_s" else m.get("per_token_s")
    return {"oom": False,
            "mem_mb": o.get("rss_peak_sampled_mb") or o.get("rss_hw_peak_mb"),
            tkey: round(t, 3)}


def _ref_084():
    """0.84B reference point from the published results (same measurement protocol)."""
    cc = L(RES / "comparison" / "cost_compare.json")          # GPU full-model cells only
    lt_g = L(RES / "gpu_train" / "ladder_train.json")["ladder"][-1]
    # GPU segmented inference: the A100-40GB re-run, i.e. the same node class as every
    # other GPU cell here (results/gpu_inference/ was collected on a mixed node pool).
    li_g = L(RES / "gpu_inference_a100_40" / "ladder_infer.json")["ladder"][-1]
    p = get_preset("large_8x2x2x8")
    m, s = p["model"], p["seg"]
    return {
        "preset": "large_8x2x2x8",
        "params_billion": round(m.n_params_estimate / 1e9, 3),
        "d_model": m.d_model, "n_layers": m.n_layers,
        "s_max_mb": round(s_max_mb(m, s), 1), "n_segments": n_seg(m, s),
        "gpu": {
            "full_train": {"oom": False, "mem_mb": cc["GPU train"]["full_mem"],
                           "time_s": round(cc["GPU train"]["full_time"], 3)},
            "full_infer": {"oom": False, "mem_mb": cc["GPU infer"]["full_mem"],
                           "time_s": round(cc["GPU infer"]["full_time"], 3)},
            "seg_train": {"oom": False, "mem_mb": lt_g["vram_peak_mb"],
                          "time_s": lt_g["step_time_s"]},
            "seg_infer": {"oom": False, "mem_mb": li_g["vram_peak_mb"],
                          "per_token_s": li_g["per_token_s"]},
        },
        "cpu": {
            "full_train": _cpu_084("full_train", "time_s"),
            "full_infer": _cpu_084("full_infer", "time_s"),
            "seg_train": _cpu_084("seg_train", "time_s"),
            "seg_infer": _cpu_084("seg_infer", "per_token_s"),
        },
    }


def _fit_line(xs, ys):
    """Least-squares y = a + b*x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def _estimate_full_train(scales):
    """Estimate what full-model training WOULD need where it OOMed (GPU 3B/7B).
    states = 16 B/param exactly (fp32 params + grads + Adam m,v — batch-independent);
    activations = the 0.84B MEASURED activation share, scaled by L*d_model (same
    batch/seq). Anchored twice: the recorded OOM (>= device) and the CPU-measured
    full-train RSS (same computation, no allocator slack)."""
    ref = scales[0]
    ref_states = 16.0 * ref["params_billion"] * 1000            # MB
    ref_ctx = 520.0
    ref_act = ref["gpu"]["full_train"]["mem_mb"] - ref_states - ref_ctx
    ref_ld = ref["n_layers"] * ref["d_model"]
    for e in scales:
        states = 16.0 * e["params_billion"] * 1000
        act = ref_act * (e["n_layers"] * e["d_model"]) / ref_ld
        est = round(states + act + ref_ctx, -2)
        ft = e["gpu"]["full_train"]
        ft["estimated_required_mb"] = est
        ft["estimate_method"] = ("16 B/param optimizer states (exact) + activations "
                                 "scaled from measured 0.84B by L*d_model + context")
        if ft.get("oom"):
            ft["exceeds_device_by"] = round(est / (ft.get("device_total_mb") or 40442.4), 1)


def _fit_point(tag, preset):
    """Optional intermediate GPU-only fit point (seg_train + full_train). Returns None
    if the runs are absent (they come from slurm/mrt_gpu_scale_fitpoints.sbatch)."""
    st = L(RES / f"scale_gpu_{tag}" / "seg_train_metrics.json")
    if st is None:
        return None
    ft = L(RES / f"scale_gpu_{tag}" / "full_train_metrics.json")
    p = get_preset(preset)
    m, s = p["model"], p["seg"]
    o = st["overall"]
    e = {"preset": preset, "fit_point_only": True,
         "params_billion": round(m.n_params_estimate / 1e9, 3),
         "d_model": m.d_model, "n_layers": m.n_layers,
         "s_max_mb": round(s_max_mb(m, s), 1), "n_segments": n_seg(m, s),
         "node": "A100-80GB",   # the fit-point job landed on the 80 GB A100 variant
         "gpu": {"seg_train": {"oom": False, "mem_mb": o["vram_hw_total_peak_mb"],
                               "time_s": round(st["avg_step_time_s"], 3),
                               "cuda_context_mb": st.get("baseline", {}).get("cuda_context_mb")}}}
    if ft is not None:
        if ft.get("oom"):
            e["gpu"]["full_train"] = {"oom": True, "oom_stage": ft["oom_stage"],
                                      "device_total_mb": ft.get("device_total_mb"),
                                      "vram_alloc_at_oom_mb": ft.get("vram_alloc_at_oom_mb")}
        else:
            e["gpu"]["full_train"] = {"oom": False,
                                      "mem_mb": ft["overall"]["vram_hw_total_peak_mb"],
                                      "time_s": round(ft["avg_step_time_s"], 3),
                                      "device_total_mb": ft.get("device_total_mb")}
    return e


def _attach_h100(scales):
    """Attach the H100 (94 GB) requirement-validation runs, when present:
    3B full_train fits there (estimate -> measurement); 7B aborts (lower bound)."""
    for e, tag in zip(scales, ["", "xl3b", "xxl7b"]):
        if not tag:
            continue
        met = L(RES / f"scale_h100_{tag}" / "full_train_metrics.json")
        if met is None:
            continue
        if met.get("oom"):
            e["gpu"]["full_train_h100"] = {"oom": True,
                                           "device_total_mb": met.get("device_total_mb"),
                                           "vram_alloc_at_oom_mb": met.get("vram_alloc_at_oom_mb")}
        else:
            e["gpu"]["full_train_h100"] = {"oom": False,
                                           "mem_mb": met["overall"]["vram_hw_total_peak_mb"],
                                           "time_s": round(met["avg_step_time_s"], 3),
                                           "device_total_mb": met.get("device_total_mb")}


def main():
    scales = [_ref_084(), _scale_entry("xl3b", "xl3b_8x2x2x8"),
              _scale_entry("xxl7b", "xxl7b_8x2x2x8")]
    for e in scales:
        e["node"] = "A100-40GB"
    _estimate_full_train(scales)
    _attach_h100(scales)
    fitx = [e for e in (_fit_point("xl15b", "xl15b_8x2x2x8"),
                        _fit_point("xl5b", "xl5b_8x2x2x8")) if e]

    # ---- performance-model fit (GPU segmented training) ----
    # TIME is fitted per node class (the intercept N*alpha' should replicate across
    # hardware variants; the slope tracks the node's memory system).
    P = [e["params_billion"] for e in scales]
    T = [e["gpu"]["seg_train"]["time_s"] for e in scales]
    a_t, b_t = _fit_line(P, T)                    # A100-40GB series: T ~= a_t + b_t*P
    fit80 = None
    if len(fitx) == 2:
        (p1, t1), (p2, t2) = [(e["params_billion"], e["gpu"]["seg_train"]["time_s"])
                              for e in fitx]
        b80 = (t2 - t1) / (p2 - p1)
        fit80 = {"time_overhead_s": round(t1 - b80 * p1, 1),
                 "time_slope_s_per_bparam": round(b80, 1),
                 "note": "two points, so the line is exact by construction; its content "
                         "is the intercept's agreement with the A100-40GB fit"}
    # MEMORY is one fit across ALL points and node classes (Eq: M = ctx + 2*s_max +
    # c_act*d_model — no dependence on node class or total P).
    allpts = sorted(scales + fitx, key=lambda e: e["params_billion"])
    ctx = next((e["gpu"]["seg_train"].get("cuda_context_mb") for e in allpts[::-1]
                if e["gpu"]["seg_train"].get("cuda_context_mb")), 520.0) or 520.0
    c_act = sum((e["gpu"]["seg_train"]["mem_mb"] - ctx - 2 * e["s_max_mb"]) / e["d_model"]
                for e in allpts) / len(allpts)

    def _tpred(e):
        if e.get("node") == "A100-80GB" and fit80:
            return fit80["time_overhead_s"] + fit80["time_slope_s_per_bparam"] * e["params_billion"]
        return a_t + b_t * e["params_billion"]

    fit = {
        "note": "GPU segmented training, all techniques ON, partition 8x2x2x8, "
                "batch 4, seq 512. Time: T = overhead + slope*P per node class "
                "(overhead = N*alpha', independent of model size AND node class). "
                "Memory: M = ctx + 2*s_max + c_act*d_model, one fit across all points "
                "(total parameter count P absent).",
        "time_overhead_s": round(a_t, 1), "time_slope_s_per_bparam": round(b_t, 1),
        "time_fit_a100_80gb": fit80,
        "mem_ctx_mb": round(ctx, 1), "mem_c_act_mb_per_dmodel": round(c_act, 4),
        "points": [{
            "params_billion": e["params_billion"],
            "node": e.get("node"),
            "time_measured_s": e["gpu"]["seg_train"]["time_s"],
            "time_predicted_s": round(_tpred(e), 1),
            "mem_measured_mb": e["gpu"]["seg_train"]["mem_mb"],
            "mem_predicted_mb": round(ctx + 2 * e["s_max_mb"] + c_act * e["d_model"], 1),
        } for e in allpts],
    }

    out = {"run": "scale_summary",
           "protocol": "cost-only: batch 4, seq 512, warmup + measured steps; infer = "
                       "256-token prompt, 8 greedy tokens; techniques all ON; 8x2x2x8",
           "gpu_node": "main scales: NVIDIA A100 40 GB (device_total 40442 MB); "
                       "fit points: A100 80 GB variant (81154 MB); validation: "
                       "H100 94 GB (95330 MB); all grete:shared / grete-h100:shared",
           "scales": scales, "fit_points": fitx, "performance_model_fit": fit}
    (RES / "scale_summary.json").write_text(json.dumps(out, indent=2))

    print(f"{'scale':8} {'dev':4} {'full train':>22} {'seg train':>18} {'full infer':>14} {'seg infer':>16}")
    for e in scales:
        for dev in ("gpu", "cpu"):
            r = e[dev]
            def fmt(d, tk="time_s"):
                if d.get("oom"):
                    return f"OOM({d['oom_stage']})"
                return f"{d['mem_mb']:.0f}MB/{d.get(tk, d.get('per_token_s')):.6g}s"
            print(f"{e['params_billion']:>6}B {dev:4} {fmt(r['full_train']):>22} "
                  f"{fmt(r['seg_train']):>18} {fmt(r['full_infer']):>14} "
                  f"{fmt(r['seg_infer'], 'per_token_s'):>16}")
    print("\nfit:", {k: v for k, v in fit.items() if k not in ("points", "note")})
    for pt in fit["points"]:
        print(f"  P={pt['params_billion']}B  T {pt['time_measured_s']} vs {pt['time_predicted_s']} "
              f"| M {pt['mem_measured_mb']} vs {pt['mem_predicted_mb']}")
    print(f"-> {RES/'scale_summary.json'}")


if __name__ == "__main__":
    main()

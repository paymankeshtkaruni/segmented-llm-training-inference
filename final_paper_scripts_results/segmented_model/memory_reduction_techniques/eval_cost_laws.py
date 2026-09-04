#!/usr/bin/env python3
"""Evaluate the pre-registered cost laws against the final measured medians.

The coefficients under test are the ones committed in prereg_prediction.json
BEFORE the held-out validation job ran (commit 6c8e8a7):

    T = a + b*P + c*N            [s]   P = params (1e9), N = segment count
    M = c0 + c1*s_vp + c2*d      [MB]  s_vp = vocab-tied slice pair (MB), d = d_model

Features:
    N    = L*(A + M + 1) + E + H   (per-layer: A attention segments, M MLP
           chunks, 1 shared attention output projection; plus E embedding and
           H output-head slices)
    s_vp = V*d*4/E + V*d*4/H bytes -> MB: the embedding slice plus the
           output-head slice, the two vocabulary-tied blocks that set the
           streamed residency floor in these configurations.

Measured cells (streamed, recomputed, deferred, dropout on; medians of 3):
    0.84B partitions  -> results/exp3_grid (8x2x2x8) and results/exp1_compare
    1.5B-6.9B         -> results/exp5_scale (gpu40 *_T3)
Also reports a fresh least-squares refit on the same points for reference,
and the held-out pre-registered cell (3.09B @ 16x4x4x16).

Writes results/cost_model_eval.json.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
V = 50257

# (d_model, d_ff, n_layers, (E, A, M, H))
CONFIGS = {
    "0.84B@8x2x2x8":   (1280, 5120, 36, (8, 2, 2, 8)),
    "0.84B@2x1x1x2":   (1280, 5120, 36, (2, 1, 1, 2)),
    "0.84B@16x4x4x16": (1280, 5120, 36, (16, 4, 4, 16)),
    "1.5B@8x2x2x8":    (1792, 7168, 36, (8, 2, 2, 8)),
    "3.1B@8x2x2x8":    (2560, 10240, 36, (8, 2, 2, 8)),
    "5.1B@8x2x2x8":    (3328, 13312, 36, (8, 2, 2, 8)),
    "6.9B@8x2x2x8":    (4096, 16384, 32, (8, 2, 2, 8)),
}
HELD_OUT = ("3.09B@16x4x4x16", (2560, 10240, 36, (16, 4, 4, 16)))


def params_b(d, ff, L):
    return (2 * V * d + L * (4 * d * d + 2 * d * ff)) / 1e9


def features(d, ff, L, seg):
    E, A, M, H = seg
    n = L * (A + M + 1) + E + H
    s_vp = V * d * 4 / 1e6 / E + V * d * 4 / 1e6 / H
    return params_b(d, ff, L), n, s_vp


def med(xs):
    return statistics.median(xs)


def measured():
    """Median (M_mb, T_s) per config from the committed result summaries."""
    out = {}
    grid = json.load(open(R / "exp3_grid" / "grid_summary.json"))
    g = grid["modes"]["T3"]["gpu"]
    out["0.84B@8x2x2x8"] = (g["vram_mb"], g["step_s"])
    for name, pat in [("0.84B@2x1x1x2", "seg_2x1x1x2_T3"),
                      ("0.84B@16x4x4x16", "seg_16x4x4x16_T3")]:
        reps = []
        for rep in sorted((R / "exp1_compare").glob(pat + "_rep*")):
            d = json.load(open(rep / (pat + "_met.json")))
            steps = []
            def find(dd):
                if isinstance(dd, dict):
                    for k, v in dd.items():
                        if k == "avg_step_time_s":
                            steps.append(v)
                        else:
                            find(v)
            find(d)
            reps.append((d["overall"]["vram_hw_total_peak_mb"], steps[0]))
        out[name] = (med([r[0] for r in reps]), med([r[1] for r in reps]))
    scale = json.load(open(R / "exp5_scale" / "scale_summary.json"))["cells"]
    for name, cell in [("1.5B@8x2x2x8", "gpu40_xl15b_T3"),
                       ("3.1B@8x2x2x8", "gpu40_xl3b_T3"),
                       ("5.1B@8x2x2x8", "gpu40_xl5b_T3"),
                       ("6.9B@8x2x2x8", "gpu40_xxl7b_T3")]:
        reps = scale[cell]
        out[name] = (med([r["vram_mb"] for r in reps]), med([r["step_s"] for r in reps]))
    return out


def lstsq3(X, y):
    A = [[sum(r[a] * r[b] for r in X) for b in range(3)] for a in range(3)]
    b = [sum(r[a] * yy for r, yy in zip(X, y)) for a in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            f = A[j][i] / A[i][i]
            for k in range(3):
                A[j][k] -= f * A[i][k]
            b[j] -= f * b[i]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (b[i] - sum(A[i][k] * x[k] for k in range(i + 1, 3))) / A[i][i]
    return x


def main():
    prereg = json.load(open(R / "prereg_prediction.json"))
    tl, ml = prereg["time_law"], prereg["memory_law"]
    meas = measured()

    rows = []
    for name, spec in CONFIGS.items():
        P, n, s = features(*spec)
        d = spec[0]
        m_meas, t_meas = meas[name]
        t_pred = tl["a"] + tl["b"] * P + tl["c"] * n
        m_pred = ml["c0"] + ml["c1"] * s + ml["c2"] * d
        rows.append({
            "config": name, "P_b": round(P, 3), "N": n, "s_vp_mb": round(s, 1),
            "d_model": d,
            "M_meas": m_meas, "M_pred": round(m_pred, 1),
            "M_err_pct": round((m_pred - m_meas) / m_meas * 100, 1),
            "T_meas": t_meas, "T_pred": round(t_pred, 1),
            "T_err_pct": round((t_pred - t_meas) / t_meas * 100, 1),
        })

    # reference refit on the same seven points (not the pre-registered law)
    X_m = [[1.0, r["s_vp_mb"], r["d_model"]] for r in rows]
    X_t = [[1.0, r["P_b"], r["N"]] for r in rows]
    refit_m = lstsq3(X_m, [r["M_meas"] for r in rows])
    refit_t = lstsq3(X_t, [r["T_meas"] for r in rows])

    # held-out pre-registered cell
    val = json.load(open(R / "prereg_validation_result.json"))
    name, spec = HELD_OUT
    P, n, s = features(*spec)
    held = {
        "config": name, "P_b": round(P, 3), "N": n, "s_vp_mb": round(s, 1),
        "predicted": prereg | {"fit_configs": None},
        "measured_reps": val["reps"],
        "measured_median": val["measured_mean"],
        "err_pct": val["error"],
    }

    out = {
        "run": "cost_model_eval",
        "laws_under_test": {"time": tl, "memory": ml,
                            "source": "prereg_prediction.json (committed before validation job)"},
        "in_sample": rows,
        "worst_in_sample": {
            "M_pct": max(abs(r["M_err_pct"]) for r in rows),
            "T_pct": max(abs(r["T_err_pct"]) for r in rows),
            "T_pct_ge_3B": max(abs(r["T_err_pct"]) for r in rows if r["P_b"] >= 3),
        },
        "reference_refit": {"memory_c0_c1_c2": [round(v, 3) for v in refit_m],
                            "time_a_b_c": [round(v, 4) for v in refit_t]},
        "held_out": held,
    }
    dst = R / "cost_model_eval.json"
    json.dump(out, open(dst, "w"), indent=2)
    for r in rows:
        print(f"{r['config']:18s} M {r['M_meas']:7.0f}/{r['M_pred']:7.0f} ({r['M_err_pct']:+5.1f}%)"
              f"  T {r['T_meas']:6.1f}/{r['T_pred']:6.1f} ({r['T_err_pct']:+5.1f}%)")
    print("worst:", out["worst_in_sample"])
    print("held-out:", held["measured_median"], held["err_pct"])
    print("->", dst)


if __name__ == "__main__":
    main()

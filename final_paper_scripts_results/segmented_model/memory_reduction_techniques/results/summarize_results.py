#!/usr/bin/env python3
"""Compact medians of all evaluation summaries for paper v4 sec:eval."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent  # run from results/


def j(p):
    return json.load(open(R / p))


d = j("exp3_grid/grid_summary.json")
print("== 12-mode grid (medians)")
for m, v in d["modes"].items():
    g, c = v.get("gpu", {}), v.get("cpu", {})
    print(m, "GPU vram=%s rss=%s step=%s opt=%s" % (g.get("vram_mb"), g.get("rss_mb"), g.get("step_s"), g.get("opt_phase_s")),
          "| CPU rss=%s step=%s opt=%s" % (c.get("rss_mb"), c.get("step_s"), c.get("opt_phase_s")))

d = j("exp4_infer/infer_summary.json")
print("\n== infer torch")
for m, v in d["torch"].items():
    g, c = v["gpu"], v["cpu"]
    print(m, "GPU vram=%s rss=%s tok=%.3f" % (g["vram_mb"], g["rss_mb"], g["per_token_s"]),
          "| CPU rss=%s tok=%.3f" % (c["rss_mb"], c["per_token_s"]))
print("== infer onnx")
for m, v in d["onnx"].items():
    g, c = v["gpu"], v["cpu"]
    print(m, "GPU vram=%s rss=%s tok=%.3f" % (g["vram_mb"], g["rss_mb"], g["per_token_s"]),
          "| CPU rss=%s tok=%.3f" % (c["rss_mb"], c["per_token_s"]))

for f in ["exp5_scale/scale_summary.json", "exp6_fast/exp6_summary.json"]:
    print("\n==", f)
    d = j(f)
    print(json.dumps(d, indent=1)[:4000])

print("\n== exp1_compare / exp2_anchor directory metrics")
for sub in ["exp1_compare", "exp2_anchor"]:
    for p in sorted((R / sub).rglob("*met*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        o = d.get("overall", {})
        keep = {k: d[k] for k in ("run", "preset", "device", "per_token_s", "generate_total_s") if k in d}
        steps = []
        def find(dd, key, out):
            if isinstance(dd, dict):
                for k, v in dd.items():
                    if k == key:
                        out.append(v)
                    else:
                        find(v, key, out)
        find(d, "avg_step_time_s", steps)
        print(p.relative_to(R), keep, "vram=%s rss=%s" % (o.get("vram_hw_total_peak_mb"), o.get("rss_hw_peak_mb")),
              "step=%s" % [round(s, 2) for s in steps])

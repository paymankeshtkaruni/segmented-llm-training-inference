#!/usr/bin/env python
"""A2 — profiling-overhead figure. Reads the seg_a2_profiling_overhead.json produced by
scripts/seg_profiling_overhead.py and writes figures/plots/seg_a2_profiling_overhead.png
(full cost step = method + profiling, stacked; per-mechanism overhead bars)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEG = Path(__file__).resolve().parents[2]
FIG = SEG / "figures" / "plots"; FIG.mkdir(parents=True, exist_ok=True)
# overhead metrics live in temp/ (written by seg_profiling_overhead.py); override with $1 if given
import sys
MET = Path(sys.argv[1]) if len(sys.argv) > 1 else SEG / "temp" / "seg_a2_profiling_overhead.json"
if not MET.exists():
    print(f"MISSING {MET} — run scripts/seg_profiling_overhead.py first"); raise SystemExit(1)

d = json.load(open(MET))
method = d["method_only_step_s"]; full = d["full_profiled_step_s"]; mechs = d["mechanisms"]
names = [m["name"].split(" (")[0] for m in mechs]
ovh = [m["overhead_s"] * 1000 for m in mechs]; pct = [m["pct_of_method"] for m in mechs]
cols = ["#55A868", "#DD8452", "#C44E52"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
a1.bar(0, method * 1000, color="#4C72B0", label=f"method (none)  {method*1000:.0f} ms"); b = method * 1000
for n, v, c in zip(names, ovh, cols):
    a1.bar(0, v, bottom=b, color=c, label=f"{n}  {v:.0f} ms"); b += v
a1.text(0, full * 1000 * 1.01, f"full {full*1000:.0f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
a1.set_xticks([]); a1.set_xlim(-0.8, 0.8); a1.set_ylabel("step time (ms)")
a1.set_title(f"Full cost step = method + profiling\n(profiling = {d['total_profiling_pct_of_method']}% of method)")
a1.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1)
y = np.arange(len(names))[::-1]
a2.barh(y, ovh, color=cols)
for yi, v, p in zip(y, ovh, pct):
    a2.text(v + max(ovh) * 0.02, yi, f"{v:.0f} ms  ({p:.2f}%)", va="center", fontsize=10)
a2.set_yticks(y); a2.set_yticklabels(names); a2.set_xlabel("overhead per step (ms)")
a2.set_xlim(0, max(ovh) * 1.35)
a2.set_title(f"Profiling overhead by mechanism\n(total {d['total_profiling_overhead_s']*1000:.0f} ms = {d['total_profiling_pct_of_method']}%)")
a2.grid(True, axis="x", alpha=0.25)
fig.suptitle(f"Real profiling overhead  [{d['preset']}  {d['device']}  bs={d['batch']}  {d['steps_per_config']} steps/config]", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG / "seg_a2_profiling_overhead.png", dpi=110, bbox_inches="tight")
print(f"  wrote {FIG/'seg_a2_profiling_overhead.png'}")

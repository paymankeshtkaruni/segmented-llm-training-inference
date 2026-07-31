# Memory-reduction techniques — incremental ablation

Shows how **peak memory goes DOWN and step time goes UP** as the segmented model's
memory-reduction techniques are switched on **one at a time, cumulatively**, on the
**FIXED large cost model** (`large_8x2x2x8`). The segmentation (E×A×M×H) is **not** varied —
only the *techniques* are toggled. Four categories: **GPU-train, CPU-train, GPU-inference,
CPU-inference**.

Self-contained: local copies of `segmentation_management/` + `seg_cost_lib.py`; the shared
project modules are **never** touched. Only the copies here carry the toggle flags.

## The ladder (rung 0 = all techniques OFF, then +1 technique per rung)

`techniques.py` is the single source of truth (`Tech` flags + `TRAIN_LADDER` / `INFER_LADDER`
+ `cumulative()`).

**Train:** r0 baseline → +sdpa → +mlp_running_sum → +chunked_ce → +recompute (full-graph→
recompute backward) → +stream_segments (lazy load) → +offload_records → +park_grads_host →
+offload_adam → +segment_wo → +free_device.

**Inference:** r0 baseline → +sdpa → +mlp_running_sum → +chunked_ce → +stream_segments →
+segment_wo → +free_device → +no_kv_cache (real K,V cache → recompute). (No backward/optimizer
techniques.)

Every toggle is verified value/token-identical OFF-vs-ON; all original identity self-tests
still pass (default `tech=None` = all ON = unchanged behavior).

## Run (large model, on a compute node)

```bash
sbatch memory_reduction_techniques/slurm/mrt_gpu_train.sbatch
sbatch memory_reduction_techniques/slurm/mrt_cpu_train.sbatch
sbatch memory_reduction_techniques/slurm/mrt_gpu_inference.sbatch
sbatch memory_reduction_techniques/slurm/mrt_cpu_inference.sbatch
```

Each writes `results/<category>/ladder_*.json` (+ per-rung `*_metrics.json`) and prints a
staircase table. Then:

```bash
python memory_reduction_techniques/plot_ladders.py     # -> results/figures/ladder_*.png
```

## Quick local check (small model, no GPU, seconds)

```bash
python cpu_train_cost.py     --preset small_8x2x2x8 --device cpu --n-steps 1  --out-dir results/gpu_train
python cpu_inference_cost.py --preset small_8x2x2x8 --device cpu --gen-tokens 4 --out-dir results/gpu_inference
```

## Scale experiment (3B / 7B, paper's scale section)

Cost-only full-vs-segmented comparison at ~3.1B (`xl3b_8x2x2x8`) and ~6.9B
(`xxl7b_8x2x2x8`) with the partition fixed at 8x2x2x8 and all techniques ON. On the 40 GB
A100 the full model no longer fits (the OOM is caught and **recorded** as the data point);
on CPU the full model still runs, giving a measured full-vs-segmented ratio at scale.

```bash
sbatch memory_reduction_techniques/slurm/mrt_gpu_scale_3b.sbatch
sbatch memory_reduction_techniques/slurm/mrt_gpu_scale_7b.sbatch
sbatch memory_reduction_techniques/slurm/mrt_cpu_scale_3b.sbatch
sbatch memory_reduction_techniques/slurm/mrt_cpu_scale_7b.sbatch
python scale_summary.py    # aggregate -> results/scale_summary.json (+ fitted cost model)
python scale_figures.py    # -> results/figures/paper/comp_scale_{memory,model}.png
```

`scale_summary.py` also fits the paper's cost model on the GPU training series:
`T_step ~ 94 s + 73 s x P[B]` (the intercept is the size-independent per-event overhead
N*alpha') and `M_peak ~ ctx + 2*s_max + 0.31 MB x d_model` (total P absent — the peak
tracks one segment, not the model).

## Files
- `techniques.py` — Tech flags + ladders + `cumulative()`
- `ladder_runner.py` — `run_train_ladder` / `run_infer_ladder`
- `seg_cost_lib.py` (copy) — MemFlow profiler + `run_cost` (training)
- `infer_cost_lib.py` — `run_infer_cost` (inference prefill+decode)
- `gpu_train_cost.py` / `cpu_train_cost.py` / `gpu_inference_cost.py` / `cpu_inference_cost.py` — drivers
- `scale_cost.py` — scale-experiment driver (one run per invocation: `--run seg_train|seg_infer|full_train|full_infer`)
- `scale_summary.py` / `scale_figures.py` — scale aggregation, cost-model fit, figures
- `plot_ladders.py` — staircase figures
- `segmentation_management/` (copy) — engines with the toggle flags wired in (+ 3B/7B presets)
- `slurm/` — the sbatch scripts; `results/` — outputs (git-ignored except aggregated JSONs)

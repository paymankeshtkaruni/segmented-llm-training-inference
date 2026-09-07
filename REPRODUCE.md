# Reproducing the paper

Every figure and table in the paper is rebuilt from committed result JSONs — no GPU and no
training is required to regenerate them. To regenerate the *underlying results*, re-run the
SLURM jobs listed in [README.md](README.md) §5; each writes the JSONs referenced below.

Paths are relative to
`final_paper_scripts_results/segmented_model/memory_reduction_techniques/` unless noted.
`SEG = final_paper_scripts_results/segmented_model/outputs`,
`FULL = final_paper_scripts_results/full_model/outputs`,
`RES = .../memory_reduction_techniques/results`.

## Regenerate everything (seconds, CPU-only)

```bash
export PROJECT_ROOT=$PWD
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
python scale_summary.py      # RES/scale_summary.json   (feeds the scale table + 2 figures)
python general_fit.py        # RES/general_fit.json     (general size+partition law, pre-registered cell)
python comparison_figures.py # comp_*  -> RES/figures/paper/
python paper_figures.py      # fig*    -> RES/figures/paper/
python scale_figures.py      # comp_scale_* -> RES/figures/paper/
```

Verified on the measurement environment: re-running these scripts reproduces
`RES/scale_summary.json` and `RES/general_fit.json` **byte-identically**, and 11 of the 12
paper PNGs **pixel-identically** (`fig4_pareto.png` differs only in renderer-level
antialiasing; every plotted value and label is identical).

## Figure → generating script → source data

| Paper figure | Script (function) | Source JSON(s) |
|---|---|---|
| `fig1_schematic` — 4-axis segmentation schematic | `paper_figures.py` `fig_schematic` | (drawn; no data) |
| `fig2_waterfall` — peak-memory waterfall (naive → all-on) + step time | `paper_figures.py` `fig_waterfall` | `RES/gpu_train/ladder_train.json` (also reads `{gpu,cpu}_{train,inference}/ladder_*.json`) |
| `fig4_pareto` — memory–time decision frontier + ZeRO points | `paper_figures.py` `fig_pareto` | `RES/gpu_train/ladder_train.json`, `RES/comparison/cost_compare.json` (`"GPU train"` cell), `RES/deepspeed_baseline_16t{,_rep1,_rep2}/zero{2,3}_offload_metrics.json` |
| `fig5_loo` — leave-one-out marginal ablation | `paper_figures.py` `fig_loo` | `RES/gpu_train_loo/loo_train.json` |
| `fig7_phase` — forward / backward / optimizer / validation split | `paper_figures.py` `fig_phase` | `RES/gpu_train/r10_free_device_metrics.json`, `RES/scale_cpu_large/seg_train_metrics.json` |
| `comp_learning` — full vs segmented loss/accuracy coincide | `comparison_figures.py` `fig_learning` | `FULL/train_gpu/metrics.json`, `SEG/seg_train_gpu_pathb/metrics.json` |
| `comp_cost_dumbbell` — peak-memory collapse (4 settings) | `comparison_figures.py` `fig_cost_dumbbell` | `RES/scale_summary.json` (identical cells to the scale table) |
| `comp_onnx` — full→seg and torch→ONNX inference memory | `comparison_figures.py` `fig_onnx` | `FULL/{infer,onnx_infer}_cost_profiler_{gpu,cpu}/*.json`, `SEG/seg_b2_gpu`, `SEG/seg_c_{gpu,cpu}`, `RES/seg_b2_cpu_clean/seg_b2_clean_metrics.json` |
| `comp_granularity` — E×A×M×H memory–time trajectory | `comparison_figures.py` `fig_granularity` | `RES/gpu_granularity/granularity_train.json` |
| `comp_memflow` — bounded sawtooth memory-over-time (2×2) | `comparison_figures.py` `fig_memflow` | `SEG/seg_cost_{gpu,cpu}/seg_cost_metrics.json`, `SEG/seg_b2_{gpu,cpu}/seg_b2_metrics.json`, `RES/scale_cpu_large/seg_train_metrics.json` |
| `comp_scale_memory` — peak training memory across scale | `scale_figures.py` `fig_scale_memory` | `RES/scale_summary.json` |
| `comp_scale_model` — fitted cost model (time + memory) | `scale_figures.py` `fig_scale_model` | `RES/scale_summary.json` |

Two further figures (`fig3_master`, `fig6_granularity`) are still produced by
`paper_figures.py` but are not used in the current paper; they are kept because the script
generates them.

## Table → source

| Paper table | Source |
|---|---|
| `tab:identity` (numerical identity) | `tests/` (`test_attention_segmentation`, `test_mlp_segmentation`, `test_losses`, `test_backward_engine`, `test_optimizer_state`, `test_true_recomputation_trainer`, `test_full_model_export_parity`) |
| `tab:techniques` (the eleven techniques) | `memory_reduction_techniques/techniques.py` |
| `tab:scale` (peak memory / step time across scale) | `RES/scale_summary.json` (printed by `scale_summary.py`) |
| `tab:fit` (cost model vs measurement) | `RES/scale_summary.json` → `performance_model_fit` |
| `tab:recipe` (deployment recipe) | `RES/gpu_train/ladder_train.json` (printed by `plot_paper_extras.py`) |

## Result JSON → generating job

| Result directory | SLURM job / driver |
|---|---|
| `FULL/train_gpu`, `FULL/infer_*`, `FULL/onnx_infer_*`, `FULL/cost_profiler_*` | `full_model/slurm/{train_gpu,infer_cost,onnx_infer_cost,cost}.sbatch` |
| `SEG/seg_train_gpu_pathb`, `SEG/seg_b1_pathb` | `segmented_model/slurm/seg_train_gpu_pathb.sbatch` |
| `SEG/seg_cost_{gpu,cpu}`, `SEG/seg_b2_*`, `SEG/seg_c_*` | `segmented_model/slurm/seg_{cost,b2,c}_{gpu,cpu}*.sbatch` |
| `RES/{gpu,cpu}_train`, `RES/{gpu,cpu}_inference` | `mrt_{gpu,cpu}_{train,inference}.sbatch` |
| `RES/{gpu,cpu}_train_loo`, `RES/{gpu,cpu}_inference_loo` | `mrt_{gpu,cpu}_loo.sbatch` |
| `RES/gpu_granularity` | `mrt_gpu_granularity.sbatch` |
| `RES/seg_b2_cpu_clean` | `mrt_cpu_infer_clean.sbatch` |
| `RES/scale_gpu_{xl3b,xxl7b}` | `mrt_gpu_scale_{3b,7b}.sbatch` |
| `RES/scale_gpu_{xl15b,xl5b}` | `mrt_gpu_scale_fitpoints.sbatch` |
| `RES/scale_gpu_xl3b_16x4x4x16` | `mrt_gpu_validate_cell.sbatch` (pre-registered cell) |
| `RES/scale_h100_{xl3b,xxl7b}` | `mrt_h100_scale_validate.sbatch` |
| `RES/scale_cpu_{xl3b,xxl7b}` | `mrt_cpu_scale_{3b,7b}.sbatch` |
| `RES/scale_cpu_large` | `mrt_cpu_full084_rerun.sbatch` + `rerun_A_cpu_bundle.sbatch` |
| `RES/scale_cpu_large_oldenv` | `rerun_D_oldenv_control.sbatch` |
| `RES/deepspeed_baseline` | `mrt_deepspeed_baseline.sbatch` |
| `RES/deepspeed_baseline_16t{,_rep1,_rep2}` | `rerun_B_deepspeed_16t{,_v2}.sbatch`, `rerun_ds_16t_rep{1,2}.sbatch` |
| `RES/gpu_inference_a100_40` | `rerun_C_gpu_infer_a40{,_v2}.sbatch` |
| `RES/full_decode_{gpu,cpu}` | `mrt_full_decode.sbatch` |
| `RES/comparison/cost_compare.json` | hand-assembled aggregate of the 0.84B full-model cells from `FULL/cost_profiler_{gpu,cpu}` and `FULL/{infer,onnx_infer}_cost_*`. Only its **GPU** entries are still consumed (by `paper_figures.py::fig_pareto` and `scale_summary.py::_ref_084`); its CPU entries are superseded by `RES/scale_cpu_large/` and are not read by any figure. |

## PROVENANCE — headline number → script → JSON → node class and thread environment

Node classes are reported as *classes*, never machine names. Where the producing script
records `device_total_mb`, the class is verifiable from the committed JSON itself
(40442 MB = A100-40 GB, 81154 MB = A100-80 GB, 95330 MB = H100-94 GB). "8 CPU / OMP unset"
is the environment of the GPU job family (the GPU jobs are not CPU-thread bound);
"16 pinned" means `--cpus-per-task=16` with `OMP_NUM_THREADS=16` and
`MALLOC_{MMAP,TRIM}_THRESHOLD_=131072`.

| Headline result | Producing script | Result JSON | Node class | Threads / allocator |
|---|---|---|---|---|
| GPU training, full model: **25.4 GB / 0.739 s** | `full_model/scripts/full_model_cost_profiler_gpu.py` | `FULL/cost_profiler_gpu/cost_profiler_metrics.json` → `RES/comparison/cost_compare.json` `"GPU train"` | A100 (`grete:shared`; capacity not recorded by this early script) | 8 CPU / OMP unset |
| GPU training, segmented: **956 MB / 164.5 s** | `gpu_train_cost.py` (rung `r10_free_device`) | `RES/gpu_train/{ladder_train,r10_free_device_metrics}.json` | A100 (`grete:shared`; class not pinned) | 8 CPU / OMP unset |
| GPU training, no-reclamation operating point: **1.2 GB / 28.4 s** | `gpu_train_cost.py` (rung `r9_segment_wo`) | `RES/gpu_train/ladder_train.json` | A100 (`grete:shared`) | 8 CPU / OMP unset |
| Leave-one-out reference (all-on): **124.8 s** | `loo_cost.py` | `RES/gpu_train_loo/{loo_train,all_on_metrics}.json` | A100 (`grete:shared`) | 8 CPU / OMP unset |
| `free_device` marginal cost: **146 MB for +97 s** | `loo_cost.py` | `RES/gpu_train_loo/drop_free_device_metrics.json` | A100 (`grete:shared`) | 8 CPU / OMP unset |
| CPU training, full model: **26.9 GB / 33.0 s** | `scale_cost.py --run full_train --device cpu` | `RES/scale_cpu_large/full_train_metrics.json` | CPU node (`grete:shared`) | **16 pinned + fixed malloc** |
| CPU training, segmented: **997 MB / 247.2 s** | `scale_cost.py --run seg_train --device cpu` | `RES/scale_cpu_large/seg_train_metrics.json` | CPU node (`grete:shared`) | **16 pinned + fixed malloc** |
| Old-environment control (same workload): **≈114 s / ≈33 GB** | `scale_cost.py --run full_train --device cpu` | `RES/scale_cpu_large_oldenv/full_train_metrics.json` | CPU node (`grete:shared`) | 8 CPU / OMP unset / no malloc tuning |
| GPU inference, segmented: **634 MB / 22.39 s per token** | `gpu_inference_cost.py` | `RES/gpu_inference_a100_40/{ladder_infer,r7_no_kv_cache_metrics}.json` | **A100-40** (`--constraint=40gb_vram` + 80 GB-class exclusion) | 8 CPU / OMP unset |
| GPU inference, full model: **5.27 GB / 0.042 s per token** | `full_decode_cost.py --device cuda` | `RES/full_decode_gpu/full_decode_metrics.json`; peak from `FULL/infer_cost_profiler_gpu` | **A100-40** (`--constraint=40gb_vram`) | 16 pinned |
| CPU inference, full model: **4.91 GB / 0.88 s per token** | `full_decode_cost.py --device cpu` | `RES/full_decode_cpu/full_decode_metrics.json` | CPU node | 16 pinned |
| Scale, 3.09B GPU full-model **OOM** (est. ≈73 GB) | `scale_cost.py --preset xl3b_8x2x2x8 --run full_train` | `RES/scale_gpu_xl3b/full_train_metrics.json` (`"oom": true`) | **A100-40** (`device_total_mb` 40442) | 8 CPU / OMP unset |
| …its H100 check: **67.6 GB measured** | `scale_cost.py --run full_train` | `RES/scale_h100_xl3b/full_train_metrics.json` | **H100-94** (`device_total_mb` 95330) | 8 CPU / OMP unset |
| Scale, 6.86B GPU full-model **OOM** (> 95 GB on H100) | `scale_cost.py --preset xxl7b_8x2x2x8 --run full_train` | `RES/scale_gpu_xxl7b/`, `RES/scale_h100_xxl7b/full_train_metrics.json` | **A100-40** / **H100-94** | 8 CPU / OMP unset |
| Scale, segmented: **1.53 GB / 304 s** (3.09B), **2.34 GB / 601 s** (6.86B) | `scale_cost.py --run seg_train` | `RES/scale_gpu_{xl3b,xxl7b}/seg_train_metrics.json` | **A100-40** | 8 CPU / OMP unset |
| Cost-model fit points: **1.57B / 5.12B** | `scale_cost.py` | `RES/scale_gpu_{xl15b,xl5b}/full_train_metrics.json` | **A100-80** (`device_total_mb` 81154) | 8 CPU / OMP unset |
| CPU scale: **59.4 GB / 68 s** (3.09B), **111.7 GB / 217 s** (6.86B) full | `scale_cost.py --device cpu` | `RES/scale_cpu_{xl3b,xxl7b}/full_train_metrics.json` | CPU node | **16 pinned + fixed malloc** |
| Pre-registered cell, 3.09B @ 16×4×4×16: predicted **466 s / 1162 MB**, measured **351 s / 1190 MB** (−2.4% memory) | `general_fit.py` (predicts) / `scale_cost.py --preset xl3b_16x4x4x16` (measures) | `RES/general_fit.json`, `RES/scale_gpu_xl3b_16x4x4x16/seg_train_metrics.json` | **A100-40** (`--constraint=40gb_vram`) | 8 CPU / OMP unset |
| ZeRO-2 offload: **21.5 GB**, 2.09 / 2.12 / 2.87 s per node | `deepspeed_cost.py --mode zero2_offload` | `RES/deepspeed_baseline_16t{_rep1,,_rep2}/zero2_offload_metrics.json` | **A100-40** × 3 distinct nodes (`device_total_mb` 40442) | **16 pinned** |
| ZeRO-3 offload: **23.6 GB**, 2.91 / 2.93 / 3.67 s per node | `deepspeed_cost.py --mode zero3_offload` | `RES/deepspeed_baseline_16t{_rep1,,_rep2}/zero3_offload_metrics.json` | **A100-40** × 3 distinct nodes | **16 pinned** |
| ZeRO baseline before thread pinning (superseded, kept for the control) | `deepspeed_cost.py` | `RES/deepspeed_baseline/zero{2,3}_offload_metrics.json` | A100-40 | 8 CPU / OMP unset |
| Granularity sweep: **1830 MB @ 2×1×1×2 → 824 MB @ 20×5×5×20** | `granularity_cost.py` | `RES/gpu_granularity/{granularity_train,g_*_metrics}.json` | A100 (`grete:shared`) | 8 CPU / OMP unset |
| Torch-free ONNX CPU inference: **876 MB → 555 MB** | `segmented_model/scripts/seg_c_onnx_cost.py` | `SEG/seg_c_cpu/seg_c_metrics.json`, `RES/seg_b2_cpu_clean/seg_b2_clean_metrics.json` | CPU node | 16 pinned + fixed malloc |
| Quality equivalence: **0.98972 val accuracy, both**; test 0.987 vs 0.987 | `full_model_train_gpu.py` / `seg_train_gpu.py` + `seg_infer_gpu.py` | `FULL/train_gpu/metrics.json`, `SEG/seg_train_gpu_pathb/metrics.json`, `SEG/seg_b1_pathb/` | A100 (`grete:shared`) | 8 CPU / OMP unset |

No machine names and no SLURM job IDs appear anywhere in this repository. The partition
names (`grete:shared`, `grete-h100:shared`), `module load` lines, `OMP_NUM_THREADS`,
`MALLOC_*` settings and `--gres` / `--constraint` directives are retained deliberately:
they are the measurement provenance, not identity.

## Cross-check: figures regenerate identically

Re-running the figure scripts overwrites `RES/figures/paper/*.png` — the figures used in
the paper. Any difference between the regenerated and the committed PNGs is a difference in
the committed result JSONs, not in the plotting code.

---

# Reproducing the v4 paper (paper_jsa: "Sequential Training and Inference on Resource-Constrained Devices")

The v4 campaign lives under
\`final_paper_scripts_results/segmented_model/memory_reduction_techniques/\`
(scripts + \`slurm/\` jobs + \`results/\` summaries) and the manuscript under
\`final_paper_scripts_results/segmented_model/paper_jsa/\`.

## Regenerate tables and figures from committed results (CPU-only, seconds)

\`\`\`bash
export PROJECT_ROOT=$PWD
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
python build_exp_summaries.py   # rebuilds grid/infer/scale/exp6/exp7 summary JSONs from per-rep dirs
python eval_cost_laws.py        # results/cost_model_eval.json  (Table: cost-model accuracy + held-out cell)
python sustained_report.py      # reads the gzipped 6.9B trace -> per-step peaks (exp8_sustained)
cd ../paper_jsa
python paper_v4_figures.py      # figures/*.pdf (all paper figures, incl. the Fig. 1 concept schematic — drawn, no data)
\`\`\`

Mapping (paper -> source):
- Exactness tables: results/verify_modes/x1_train_exact_{gpu,cpu}.json, infer_exact_*.json,
  control_fp32_floor_*.json, step_delta_analysis_*.json (jobs: slurm/x1_verify_*.sbatch)
- Matched-pair learning + prediction agreement: results/quality_fresh_pair/*
  (jobs: slurm/quality_triple.sbatch, full_repaired.sbatch, onnx accuracy jobs)
- Comparison + anchors: results/exp1_compare, exp2_anchor, deepspeed_baseline_16t*
- 12-mode grid: results/exp3_grid/grid_summary.json (slurm/exp3_grid_*.sbatch)
- Inference modes + ONNX + full-decode anchors: results/exp4_infer/infer_summary.json
- Scale: results/exp5_scale/scale_summary.json; fastest modes: results/exp6_fast/exp6_summary.json
- Batch/sequence sensitivity: results/exp7_sensitivity/sensitivity_summary.json (slurm/exp7_sensitivity.sbatch)
- Sustained 6.9B stability: results/exp8_sustained/sustained_summary.json (slurm/exp8_sustained.sbatch)
- Cost model + pre-registration: results/prereg_prediction.json (committed BEFORE the
  validation job), results/prereg_validation_result.json, results/cost_model_eval.json

Notes: per-rep metrics JSONs larger than 2 MB carry their raw sampling timelines replaced by a
stripped marker (the summary builders read only scalar fields); regenerate full traces by
re-running the corresponding sbatch job. Every measurement runs via Slurm using the
site-independent PROJECT_ROOT/PYTHON convention documented at the top of each sbatch file.

## Long-horizon validation (exp8), validation-cost bridge (exp9), and measurement self-checks

These back the paper's sustained-training/serving tables and figures and its
Sec. "Long-Horizon Validation and Measurement Self-Checks".

Scripts (in \`final_paper_scripts_results/segmented_model/memory_reduction_techniques/\`):
- \`exp8_long_run.py\` / \`exp8_long_serve.py\` — training-realistic long runs
  (no per-step validation, no allocator trimming, no tracer; per-step/-request
  wall time + peak memory; rolling writes; \`--max-hours\` graceful stop).
  Jobs: \`slurm/exp8_g1...g9.sbatch\`, \`slurm/exp8_ia...id.sbatch\`.
- \`sustained_report.py\` — aggregates all runs into
  \`results/exp8_sustained/sustained_summary.json\`.
- \`exp8_plots.py\` — the per-experiment evolution figures
  (\`results/exp8_sustained/plots/fig_exp8_{training,serving}.pdf\`) and the
  validation-cost figure (\`fig_exp9_validation_cost.pdf\`).
- \`exp9_bridge_report.py\` — validation-cost bridge from
  \`results/exp9_validation_bridge/\` (5-step four-phase re-measurements;
  job \`slurm/exp9_validation_bridge.sbatch\`) ->
  \`results/exp9_validation_bridge/bridge_summary.json\`.
- \`table_enrichment_report.py\` — full-model anchors, per-mode ratios vs
  the full model (Table VI), and long-run deltas vs the short cost protocol
  (Table IX), from committed results only (job
  \`slurm/table_enrichment.sbatch\`) -> \`results/table_enrichment.json\`.
- \`slurm/measurement_checks.sbatch\` + \`slurm/sampler_off_rerun.sbatch\` —
  tracer-overhead bridge (profiled vs unprofiled serving, same node) and the
  RSS-sampler paired control; results in \`results/measurement_checks/\`.

Regenerate the derived artifacts from committed data (CPU-only, seconds):
\`\`\`bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
python sustained_report.py && python exp9_bridge_report.py && python exp8_plots.py && python table_enrichment_report.py
\`\`\`
Per-rep JSONs carry raw per-step rows; large sampling timelines are replaced
by a stripped marker (regenerate via the sbatch jobs). Step counts are scaled
inversely to per-step cost so each configuration covers a comparable
multi-hour horizon in one allocation.

# Reproducing the paper

This file covers the manuscript *Sequential Training and Inference on
Resource-Constrained Devices*. Everything it prints — every table, every figure,
every number in the prose — comes from the campaign under
`final_paper_scripts_results/segmented_model/memory_reduction_techniques/`
(scripts, `slurm/` jobs, committed `results/` JSONs); the manuscript sources are
under `final_paper_scripts_results/segmented_model/paper_jsa/`.

Below, `MRT = final_paper_scripts_results/segmented_model/memory_reduction_techniques`
and `results/` is relative to it. Earlier campaigns from this project's previous
papers are still shipped for provenance and are listed, with reasons, under
[Superseded material](#superseded-material-not-used-by-this-paper) at the end —
no number in this manuscript comes from them.

## Regenerate every table and figure from committed results (CPU-only, seconds)

No GPU and no training is needed: the summary builders and the plotting scripts
read only committed JSONs.

Run them with the project interpreter — `poetry run python …`, or
`$(poetry env info -e)`; a bare `python` on PATH is whatever the shell finds.
Total runtime ≈10 s on a CPU-only laptop.

```bash
export PROJECT_ROOT=$PWD
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
python build_exp_summaries.py      # results/exp{1,2,3,4,5,6,7}*/ per-rep dirs -> the summary JSONs
python eval_cost_laws.py           # results/cost_model_eval.json          (cost-model tables)
python sustained_report.py         # results/exp8_sustained/sustained_summary.json
python exp9_bridge_report.py       # results/exp9_validation_bridge/bridge_summary.json
python exp8_plots.py               # results/exp8_sustained/plots/*.pdf    (3 paper figures)
python table_enrichment_report.py  # results/table_enrichment.json         (anchors, ratios, bands)
cd ../paper_jsa
python paper_v4_figures.py         # figures/*.pdf                         (6 paper figures)
```

That is the same chain `slurm/verify_artifact.sbatch` runs against a fresh clone.

The paper's nine figures and the script that draws each:

| figure file (`paper_jsa/figures/`) | script | paper |
|---|---|---|
| `fig_schematic.pdf` | `paper_v4_figures.py` `fig_schematic` (drawn; no data) | Fig. 1 |
| `learning_curves.pdf` | `paper_v4_figures.py` `fig_learning_curves` | Fig. 3 |
| `frontier_train.pdf` | `paper_v4_figures.py` `fig_frontier` | Fig. 4 |
| `traces_gpu.pdf` | `paper_v4_figures.py` `fig_traces` | Fig. 5 |
| `inference_frontier.pdf` | `paper_v4_figures.py` `fig_inference` | Fig. 6 |
| `scale_train.pdf` | `paper_v4_figures.py` `fig_scale` | Fig. 8 |
| `fig_exp9_validation_cost.pdf` | `exp8_plots.py` `fig_validation_cost` | Fig. 2 |
| `fig_exp8_serving.pdf` | `exp8_plots.py` `plot("serve", …)` | Fig. 7 |
| `fig_exp8_training.pdf` | `exp8_plots.py` `plot("long", …)` | Fig. 9 |

`exp8_plots.py` writes its three PDFs to `results/exp8_sustained/plots/`; the
copies in `paper_jsa/figures/` are those files under the same names.
`paper_v4_figures.py` writes directly into `paper_jsa/figures/`.

**The chain above does not perform that copy.** After re-running `exp8_plots.py`,
the three regenerated PDFs sit only under `results/exp8_sustained/plots/`, and
`main.tex` still compiles against the older committed copies. To refresh them:

```bash
cp final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/exp8_sustained/plots/fig_exp8_{serving,training}.pdf \
   final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/exp8_sustained/plots/fig_exp9_validation_cost.pdf \
   final_paper_scripts_results/segmented_model/paper_jsa/figures/
```

## Paper object → source data

- **Exactness tables** (training identity, inference identity, the fp32 control rows):
  `results/verify_modes/x1_train_exact_{gpu,cpu}.json`, `infer_exact_{gpu,cpu}.json`,
  `control_fp32_floor_{gpu,cpu}.json`, `step_delta_analysis_{gpu,cpu}.json`,
  `scale_forward_*.json`
  (jobs: `slurm/x1_verify_gpu.sbatch`, `slurm/x1_verify_cpu.sbatch`,
  `slurm/a2_infer_exact_gpu.sbatch`, `slurm/a2_infer_exact_cpu.sbatch`,
  `slurm/x1_step_delta.sbatch`, `slurm/control_floor.sbatch`,
  `slurm/verify_scale_gpu.sbatch`).
- **Matched-pair learning (Sec. VI-C)**: `results/quality_fresh_pair/T{1,2}/metrics.json`,
  `full_model_10ep.log`, `full_model_repaired.json`
  (jobs: `slurm/quality_triple.sbatch`, `slurm/full_repaired.sbatch`).
- **Prediction agreement across the eight serving pipelines (Sec. VI-D)**:
  `results/quality_fresh_pair/inference_accuracy_fresh.json` (the 0.9548 exact-match
  and the torch-vs-export agreement), `onnx_accuracy_{cpu,cuda}_{stream,preload,full}.json`,
  `ia1_agreement_report.json`, `ia1_agreement_full_{cpu,cuda}.json`,
  `preds_torch.json`, `preds_onnx_*.json`
  (jobs: `slurm/eval_quality_infer.sbatch`, `slurm/ia1_agreement.sbatch`,
  `slurm/ia1_fix.sbatch`, `slurm/onnx_fresh_accuracy2.sbatch`,
  `slurm/onnx_full_accuracy.sbatch`). Five of the eight pipelines are compared
  example by example; three by aggregate exact-match.
- **Comparison and anchors**: `results/exp1_compare/`, `results/exp2_anchor/`,
  `results/deepspeed_baseline_16t{,_rep1,_rep2}/`. The naive-segmented anchor
  (partition only, every technique off: tech code `00000000000`, dropout 0.1) is
  `results/exp2_anchor/{gpu,cpu}_rep*/naive_anchor_met.json`
  (jobs: `slurm/exp2_anchor_gpu.sbatch`, `slurm/exp2_anchor_cpu.sbatch`); its medians,
  host RSS, optimizer-phase time and ratios against the full model are the
  `naive_anchors` block of `results/table_enrichment.json`.
- **12-mode grid**: `results/exp3_grid/grid_summary.json`
  (jobs: `slurm/exp3_grid_gpu.sbatch`, `slurm/exp3_grid_cpu_A.sbatch`,
  `slurm/exp3_grid_cpu_B.sbatch`).
- **Inference modes, ONNX and full-decode anchors**: `results/exp4_infer/infer_summary.json`
  (jobs: `slurm/exp4_infer_torch_gpu.sbatch`, `slurm/exp4_infer_torch_cpu.sbatch`,
  `slurm/exp4_infer_onnx_gpu.sbatch`, `slurm/exp4_infer_onnx_cpu.sbatch`,
  `slurm/exp4_onnx_resident_gpu.sbatch`, `slurm/exp4_onnx_resident_cpu.sbatch`,
  `slurm/exp4_full_anchor.sbatch`).
- **Scale**: `results/exp5_scale/scale_summary.json`
  (jobs: `slurm/exp5_xl15b_gpu.sbatch`, `slurm/exp5_xl3b_gpu.sbatch`,
  `slurm/exp5_xl5b_gpu.sbatch`, `slurm/exp5_xl5b_resident_gpu.sbatch`,
  `slurm/exp5_xxl7b_gpu.sbatch`, `slurm/exp5_cpu_xl3b.sbatch`,
  `slurm/exp5_cpu_xxl7b.sbatch`, `slurm/exp5_h100.sbatch`,
  `slurm/exp5_80gb_gpu.sbatch`, plus the added repetitions in
  `slurm/audit_gpu_reps.sbatch` and `slurm/audit_cpu_reps.sbatch`).
  Fastest modes and the granularity cells: `results/exp6_fast/exp6_summary.json`
  (jobs: `slurm/exp6_fast_scale_gpu.sbatch`, `slurm/exp6_fast_scale_cpu.sbatch`,
  `slurm/exp6_gran_fast_gpu.sbatch`, `slurm/exp6b_remainder.sbatch`,
  `slurm/gran_std_decode_gpu.sbatch`).
- **Granularity table** (three partitions at 0.84B): training rows from
  `results/exp1_compare/seg_{2x1x1x2,16x4x4x16}_{T2,T3}_rep*/` and `results/exp3_grid`
  (standard partition); decode rows from the `gran_*` cells of `results/exp6_fast`.
  All decode cells use prompt 256 / 8 generated tokens: `gran_std_decode_gpu.sbatch`
  adds the standard partition at that protocol so the three partitions are
  comparable, instead of borrowing the 32-token cells of the inference table.
- **Inference-at-scale table**: resident + KV cells from `results/exp6_fast` (I1),
  streamed no-cache cells from `results/exp5_scale` (I4), both at prompt 256 /
  8 generated tokens.
- **Scale-training figure** (`paper_jsa/figures/scale_train.pdf`):
  `paper_v4_figures.py` `fig_scale`, reading `exp3_grid` (0.84B), `exp5_scale`
  (full model, 3.1B/5.1B/6.9B resident, all streamed) and `exp6_fast` (the 1.5B
  resident cell, which is the retained-graph in-backward mode). Memory is plotted
  in the paper's GB (1,000 MB), the unit every table uses.
- **Batch/sequence sensitivity**: `results/exp7_sensitivity/sensitivity_summary.json`
  (job: `slurm/exp7_sensitivity.sbatch`).
- **Long-horizon training and serving**: `results/exp8_sustained/sustained_summary.json`
  (jobs listed in the long-horizon section below).
- **Validation-cost bridge**: `results/exp9_validation_bridge/bridge_summary.json`
  (jobs: `slurm/exp9_validation_bridge.sbatch`, `slurm/exp9_report.sbatch`).
- **Transport accounting (Sec. IX)**: `results/measurement_checks/store_bandwidth_large_8x2x2x8_cuda_cpu_ram.json`
  and `…_cpu_disk.json` (job: `slurm/store_bandwidth.sbatch`).
- **Instrumentation self-checks**: `results/measurement_checks/{i1,i4}_{profiled,unprofiled}/`
  and `sampler_{on,off}/` (jobs: `slurm/measurement_checks.sbatch`,
  `slurm/sampler_off_rerun.sbatch`).
- **Cost model and pre-registration**: `results/prereg_prediction.json` (committed
  BEFORE the validation job), `results/prereg_validation_result.json`,
  `results/cost_model_eval.json` (jobs: `slurm/prereg_validate.sbatch`,
  `slurm/mrt_gpu_validate_cell.sbatch`). Provenance of the pre-registration (this
  public repository received both files in one squashed commit, so the order is
  recorded here): in the development history the prediction file was committed at
  2026-09-01 14:32:33 CEST (commit 6c8e8a7, "PRE-REGISTERED prediction for held-out
  cell (3.09B, 16x4x4x16, T3): 110s / 1330MB"); the validation job
  (`slurm/mrt_gpu_validate_cell.sbatch`) was submitted at 14:32:37, started 14:33:20
  and ended 15:10:04; the result file was committed at 15:27:20 (commit ff015f2).
  The job ID is stored in `results/prereg_validation_result.json`.

Notes: per-rep metrics JSONs larger than 2 MB carry their raw sampling timelines
replaced by a stripped marker (the summary builders read only scalar fields);
regenerate full traces by re-running the corresponding sbatch job. Every
measurement runs via Slurm using the site-independent `PROJECT_ROOT`/`PYTHON`
convention documented at the top of each sbatch file. Nothing runs on a login node.

## Exactness / identity jobs

Every job below rebuilds a reference full model from a fixed seed and compares the
segmented stack against it; the verdict lands in `results/verify_modes/`.

| job (sbatch) | what it proves | output |
|---|---|---|
| `slurm/x1_verify_gpu.sbatch`, `slurm/x1_verify_cpu.sbatch` | training identity at 0.84B: eval forward, gradient slices, and **all** parameters after 1 and 4 AdamW steps, over the 12 dropout × recompute × streaming × update-style modes | `results/verify_modes/x1_train_exact_{gpu,cpu}.json` |
| `slurm/a2_infer_exact_gpu.sbatch`, `slurm/a2_infer_exact_cpu.sbatch` (`verify_infer_modes.py`) | inference identity at 0.84B: prefill hidden state + greedy tokens over the four inference modes (+ the ONNX streaming/preload runs) | `results/verify_modes/infer_exact_{gpu,cpu}.json` |
| `slurm/x1_step_delta.sbatch` | the fp32 summation-order floor that the residual deltas sit on | `results/verify_modes/step_delta_analysis_{gpu,cpu}.json` |
| `slurm/control_floor.sbatch` (`control_fp32_floor.py`) | the device-side fp32 controls the identity table is read against (same computation, summation order perturbed) | `results/verify_modes/control_fp32_floor_{gpu,cpu}.json` |
| `slurm/verify_scale_gpu.sbatch` (`verify_scale_forward.py`) | **forward-state identity at 3.1B and 6.9B**: training-forward hidden states for the three training forward modes and prefill + greedy tokens for the four inference modes, at 0.84B, 3.1B and 6.9B | `results/verify_modes/scale_forward_{large_8x2x2x8,xl3b_8x2x2x8,xxl7b_8x2x2x8}_cuda.json` |

**The scale job, in detail.** At 3.1B and 6.9B the reference and a resident segmented
engine cannot both sit on a 40 GB A100 (6.9B fp32 weights are 27.4 GB per copy), so
`verify_scale_forward.py` sequences the check: it builds the reference on CPU, fills the
segment store from it, computes **every** reference output on the GPU (training-forward
hidden states + greedy tokens), copies those to the host, deletes the reference and
empties the allocator, and only then builds each segmented mode from the same store. The
JSON records `ref_peak_reserved_mb` and a per-mode `peak_reserved_mb` so the two phases
are visibly disjoint. Gradient/optimizer identity is *not* re-checked at scale — that
needs the reference's autograd graph resident alongside the comparison, which is exactly
what does not fit; it stays covered at 0.84B by `x1_verify_*`.

```bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
sbatch slurm/verify_scale_gpu.sbatch      # smoke, then 0.84B → 3.1B → 6.9B (8 h limit)
```

`verify_scale_gpu.sbatch` honours `PROJECT_ROOT` and `PYTHON` if exported, and otherwise
falls back to the cluster paths baked into it:

```bash
PROJECT_ROOT=/path/to/repo PYTHON=/path/to/python \
  sbatch final_paper_scripts_results/segmented_model/memory_reduction_techniques/slurm/verify_scale_gpu.sbatch
```

## The accuracy tier: the quality checkpoint and its matched pair (Sec. VI-C)

Every cost campaign trains on synthetic tokens. The accuracy tier of **Sec. VI-C**
(*Matched-Pair Learning*) trains the real model that the prediction-agreement jobs of
**Sec. VI-D** all evaluate: one full-model reference and two segmented modes, same
data, seed, schedule and epoch count, in one job and one environment.

| job (sbatch) | what it runs | output | paper object |
|---|---|---|---|
| `slurm/quality_triple.sbatch` | the matched triple: full-model reference (`scripts/seg_full_train_b1.py`), segmented T1 (resident) and segmented T2 (resident + recompute), 10 epochs, batch 64, seed 42, warmup 200, cosine, dropout 0.1 | `results/quality_fresh_pair/T{1,2}/metrics.json` (+ `checkpoints/best.pt`), `results/quality_fresh_pair/{full_model_10ep.log,segmented_T1_10ep.log,segmented_T2_10ep.log}` | the matched-pair table and the learning-curve figure: losses (0.0191 / 0.0190) and token accuracies (0.9950 / 0.9953) |
| `slurm/eval_quality_infer.sbatch` (`eval_quality_inference.py`) | full-test greedy decoding of the segmented T1 checkpoint in the PyTorch runtime, 11,654 rows, equal-length batching | `results/quality_fresh_pair/inference_accuracy_fresh.json` | the **0.9548** exact-match and the 11,654/11,654 torch-vs-export agreement quoted in **Sec. VI-D**, in the serving-frontier caption and in contribution 6 |
| `slurm/full_repaired.sbatch` (`eval_full_model_repaired.py`) | retrains the full model under the identical protocol, saves the checkpoint, and evaluates it under the equal-length batching fix of Sec. V-D | `results/quality_fresh_pair/{full_model_10ep.pt,full_model_10ep_rerun.log,full_model_repaired.json}` | the **0.9556** independent full-model exact-match and the full-model row of the matched-pair table |

## Full-test prediction agreement across the eight serving pipelines (Sec. VI-D)

The identity jobs above compare tensors; these compare *predictions* — the greedy
exact-match of every serving pipeline over the full 11,654-row test set, and the
per-example agreement of its token stream with the torch segmented engine. Together
they back the **eight-pipeline** agreement claim of **Sec. VI-D** and the
serving-frontier caption: eight pipelines across both engines, six of them
PyTorch-free, return the same exact-match; five are compared example by example and
the remaining three by aggregate exact-match.

| job (sbatch) | what it runs | output |
|---|---|---|
| `slurm/eval_quality_infer.sbatch` | the two PyTorch pipelines (segmented engine and the exported dense model), with per-example prediction dumps | `results/quality_fresh_pair/inference_accuracy_fresh.json`, `preds_torch.json` |
| `slurm/onnx_fresh_accuracy2.sbatch` | segmented ONNX (`--mode seg`), streamed + preloaded weights, CPU and CUDA | `results/quality_fresh_pair/onnx_accuracy_{cpu,cuda}_{stream,preload}.json` |
| `slurm/ia1_agreement.sbatch` | torch dump + segmented-ONNX CPU-preload dump, then the near-tie margin analysis | `results/quality_fresh_pair/{preds_torch,preds_onnx_cpu_preload}.json`, `ia1_agreement_report.json` |
| `slurm/ia1_fix.sbatch` | the same two steps re-run under the equal-length batching fix of Sec. V-D, reusing the committed ONNX prediction dump; this is the run the committed `inference_accuracy_fresh.json` and `ia1_agreement_report.json` come from | `results/quality_fresh_pair/inference_accuracy_fresh.json`, `ia1_agreement_report.json` |
| `slurm/onnx_full_accuracy.sbatch` | **full-session ONNX** (`--mode full`): one baked `InferenceSession` over the whole model, CUDA then CPU, then the same agreement check per provider | `results/quality_fresh_pair/onnx_accuracy_{cuda,cpu}_full.json`, `preds_onnx_{cuda,cpu}_full.json`, `ia1_agreement_full_{cpu,cuda}.json` |

The full-session job first exports the *trained* quality checkpoint as one monolithic
ONNX graph — `scripts/seg_c_export_baked.py --what full --checkpoint …` reassembles the
segment store into a dense `ReferenceGPTDecoder` before exporting — into
`final_paper_scripts_results/segmented_model/onnx/c_full_quality_fresh/` (the `onnx/`
tree is gitignored and regenerated by the job, exactly like `c_seg_quality_fresh`). The
export step is skipped when `full_model.onnx` is already there. Without `--checkpoint`
the exporter is unchanged: seeded random init, as the 0.84B cost exports require.

```bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
sbatch slurm/onnx_full_accuracy.sbatch     # export (if needed) + 2 evals + 2 agreements
```

Like `verify_scale_gpu.sbatch`, it honours `PROJECT_ROOT` and `PYTHON` if exported, and
derives the ORT CUDA library path from the interpreter (the eval process imports no
torch, so cuDNN/cuBLAS must come from the venv nvidia wheels).

`eval_onnx_accuracy.py --limit N` evaluates only the first N test rows and suffixes its
output names with `_limit<N>`; it is for smoke runs and never overwrites a real result.

## Long-horizon validation (exp8), validation-cost bridge (exp9)

These back the sustained-training and sustained-serving tables and figures and
Sec. V-F, "Long-Horizon Validation and Measurement Self-Checks".

Scripts (in `MRT/`):
- `exp8_long_run.py` / `exp8_long_serve.py` — training-realistic long runs
  (no per-step validation, no allocator trimming, no tracer; per-step/-request
  wall time + peak memory; rolling writes; `--max-hours` graceful stop).
  Training jobs: `slurm/exp8_g1_084b_resident_imm.sbatch`,
  `slurm/exp8_g2_084b_resident_recomp.sbatch`, `slurm/exp8_g3_084b_streamed_def.sbatch`,
  `slurm/exp8_g4_084b_streamed_imm.sbatch`, `slurm/exp8_g5_69b_resident_recomp.sbatch`,
  `slurm/exp8_g6_69b_streamed_imm.sbatch`, `slurm/exp8_g7_084b_cpu_streamed_def.sbatch`,
  `slurm/exp8_g8_084b_cpu_resident_imm.sbatch`, `slurm/exp8_g9_69b_cpu_streamed_imm.sbatch`.
  Serving jobs: `slurm/exp8_ia_084b_cache.sbatch`, `slurm/exp8_ib_084b_streamed.sbatch`,
  `slurm/exp8_ic_084b_onnx_disk.sbatch`, `slurm/exp8_id_69b_cache.sbatch`.
- `sustained_report.py` — aggregates all runs into
  `results/exp8_sustained/sustained_summary.json`.
- `exp8_plots.py` — the per-experiment evolution figures
  (`results/exp8_sustained/plots/fig_exp8_{training,serving}.pdf`) and the
  validation-cost figure (`fig_exp9_validation_cost.pdf`).
- `exp9_bridge_report.py` — validation-cost bridge from
  `results/exp9_validation_bridge/` (5-step four-phase re-measurements;
  jobs `slurm/exp9_validation_bridge.sbatch`, `slurm/exp9_report.sbatch`) ->
  `results/exp9_validation_bridge/bridge_summary.json`.
- `table_enrichment_report.py` — full-model anchors, naive-segmented anchors
  (every technique off), per-mode ratios against the full model, long-run deltas
  against the short cost protocol and within-run spread (the band columns), from
  committed results only (job `slurm/table_enrichment.sbatch`) ->
  `results/table_enrichment.json`.
- `slurm/exp8_aggregate.sbatch` runs `sustained_report.py` and `exp8_plots.py`
  together, which is how the long-horizon tables and figures are refreshed after a
  long run finishes.

Step counts are scaled inversely to per-step cost so each configuration covers a
comparable multi-hour horizon in one allocation. Per-rep JSONs carry raw per-step
rows; large sampling timelines are replaced by a stripped marker (regenerate via
the sbatch jobs).

## Measurement self-checks (`results/measurement_checks/`)

These jobs do not measure the method; they measure the *measurement* — that a
reported number is not an artefact of the instrumentation, and not an assumed
constant standing in for a measured one.

| job (sbatch) | what it checks | output |
|---|---|---|
| `slurm/measurement_checks.sbatch` | **profiler bridge**: the profiled cost protocol vs the unprofiled long-serve runner, back to back on one node (inference modes I1 and I4); **RSS-sampler interference**: 25 CPU training steps with the memory sampler and 25 without | `results/measurement_checks/{i1,i4}_{profiled,unprofiled}/`, `results/measurement_checks/sampler_{on,off}/` |
| `slurm/sampler_off_rerun.sbatch` | the paired sampler-off control re-run on the same node class | `results/measurement_checks/sampler_off/` |
| `slurm/store_bandwidth.sbatch` (`store_bandwidth.py`) | **the achieved bandwidth of the segment-store path** at 0.84B, timed through the real `StrictSegmentLoader` and store objects under the streamed technique code `11111111100` (the code T3/T6/T9/T12 run): FETCH = `load_segment` (store clone or `torch.load` + `load_state_dict` + `module.to(device)`), PARK = `release_segment(save=True)`, the host-copy vs host-to-device split of the fetch, and a raw 256 MiB pageable/pinned H2D copy as hardware context | `results/measurement_checks/store_bandwidth_large_8x2x2x8_cuda_cpu_ram.json`, `…_cpu_disk.json` |

**Why the bandwidth job exists.** Sec. IX attributes the three training phases of
the streamed step at 0.84B (23.1 s of the 28.3 s four-phase step; the remaining
5.1 s is the protocol's validation pass, excluded from both sides of the account)
to transport at about 15% (~3.4 s: 2.2 s for the 34 GB crossing the link, each
direction at its own measured rate — 20.9 GB inbound at 16.9 GB/s, 13.4 GB outbound
at the 14.4 GB/s park rate — plus 1.2 s for the inbound share the store also clones
on the host at 17.1 GB/s) and to synchronous orchestration for the remaining 85%. That split was
originally computed against an *assumed* ~20 GB/s PCIe figure — a link rating,
whereas the engine's store copies go through **pageable** host memory, one
parameter tensor at a time. `store_bandwidth.py` replaces the assumption with a
measurement of the path as the engine actually walks it, so the attribution rests
on a number. One job covers both paper cell bindings: GPU with the `cpu_ram`
store, then CPU with the `disk` store under the CPU cells' `OMP_NUM_THREADS=16`
and glibc `MALLOC_*` environment. Each JSON records per-pass and median-pass
bytes/seconds/GB/s (decimal GB) and MiB/s for both directions, the segment count
and largest segment, peak reserved VRAM, device name and host, `raw_pageable_h2d_gbs`
and `raw_pinned_h2d_gbs`, and — for the disk store — the filesystem type of the
store root. The disk store is written under the output directory and removed when
the job ends (like `measure_one.py`'s `_work` scratch); the two JSONs are tracked.

```bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
sbatch slurm/store_bandwidth.sbatch     # both bindings, 3 passes each, 1 h limit
```

Like `verify_scale_gpu.sbatch`, it honours `PROJECT_ROOT` and `PYTHON` if exported.
The script also runs standalone, e.g.
`python store_bandwidth.py --device cuda --store cpu_ram --preset small_8x2x2x8 --passes 2`
for a seconds-long smoke.

## The paper's measurement campaigns (every table and figure)

Each campaign is one or more sbatch files under `MRT/slurm/`; each writes
per-repetition `*_metrics.json` under `results/<campaign>/<cell>_rep<n>/`, and
`build_exp_summaries.py` folds those into the committed summary JSONs the paper's
tables are read from.

| campaign (sbatch) | cells | summary | paper |
|---|---|---|---|
| `exp1_gpu_compare.sbatch` | the coarse and fine partitions at 0.84B, modes T2/T3 | `results/exp1_compare/seg_*_rep*/` | granularity table (training rows) |
| `exp2_anchor_gpu.sbatch`, `exp2_anchor_cpu.sbatch` | naive segmented anchor | `results/exp2_anchor/` | mode-grid anchor rows |
| `rerun_B_deepspeed_16t.sbatch` (`rerun_B_deepspeed_16t_v2.sbatch` = same job, node-excluding retry), `rerun_ds_16t_rep1.sbatch`, `rerun_ds_16t_rep2.sbatch` (`deepspeed_cost.py`) | DeepSpeed ZeRO-2 and ZeRO-3 Offload on the same 0.84B model and protocol (fp32, batch 4, seq 512, 3 steps, `OMP_NUM_THREADS=16`, A100-40GB) | `results/deepspeed_baseline_16t{,_rep1,_rep2}/zero{2,3}_offload_metrics.json` | the two ZeRO-Offload rows of the mode-comparison table, and the "twenty times / 19.9x below the measured ZeRO-Offload floor" headline |
| `exp3_grid_gpu.sbatch`, `exp3_grid_cpu_A.sbatch`, `exp3_grid_cpu_B.sbatch` | all 12 training modes, both devices | `results/exp3_grid/grid_summary.json` | mode grid, training frontier figure |
| `exp4_infer_torch_gpu.sbatch`, `exp4_infer_torch_cpu.sbatch`, `exp4_infer_onnx_gpu.sbatch`, `exp4_infer_onnx_cpu.sbatch`, `exp4_full_anchor.sbatch`, `exp4_onnx_resident_gpu.sbatch`, `exp4_onnx_resident_cpu.sbatch` | 4 PyTorch + 3 ONNX serving modes at 0.84B, prompt 256 / 32 generated tokens | `results/exp4_infer/infer_summary.json` | inference table, serving frontier |
| `exp5_xl15b_gpu.sbatch`, `exp5_xl3b_gpu.sbatch`, `exp5_xl5b_gpu.sbatch`, `exp5_xl5b_resident_gpu.sbatch`, `exp5_xxl7b_gpu.sbatch`, `exp5_cpu_xl3b.sbatch`, `exp5_cpu_xxl7b.sbatch`, `exp5_h100.sbatch`, `exp5_80gb_gpu.sbatch` | 1.5B–6.9B: full-model references (incl. the OOM cells and the 5.1B resident cell), streamed T3/T9, decode I4 | `results/exp5_scale/scale_summary.json` | scale table and figure, inference-at-scale table |
| `audit_gpu_reps.sbatch`, `audit_cpu_reps.sbatch` | the added rep2/rep3 for cells that had a single repetition (the 6.9B resident T2 headline; the 0.84B CPU full-train anchor; the CPU scale and CPU fast-mode cells) | `results/exp2_anchor/`, `results/exp5_scale/`, `results/exp6_fast/` | the medians the scale and anchor tables print |
| `exp6_fast_scale_gpu.sbatch`, `exp6_fast_scale_cpu.sbatch`, `exp6_gran_fast_gpu.sbatch`, `exp6b_remainder.sbatch`, `gran_std_decode_gpu.sbatch` | fastest resident modes (T7/T8) and resident decode (I1) at scale; the coarse, standard and fine partitions at 0.84B for T7/I1/I4, all decode cells at prompt 256 / 8 generated tokens | `results/exp6_fast/exp6_summary.json` | granularity table (decode rows), inference-at-scale table, 1.5B resident bar of the scale figure |
| `exp7_sensitivity.sbatch` | batch × sequence sweep of the streamed deferred mode | `results/exp7_sensitivity/sensitivity_summary.json` | sensitivity table |
| the nine `exp8_g*.sbatch` and four `exp8_i*.sbatch` jobs, then `exp8_aggregate.sbatch` | 128–2,048-step training runs and 500–2,000-request serving sessions | `results/exp8_sustained/sustained_summary.json` | long-horizon tables and figures |
| `exp9_validation_bridge.sbatch`, `exp9_report.sbatch` | five-step four-phase re-measurements on the long-run nodes | `results/exp9_validation_bridge/` | validation-cost figure, protocol bridge |
| `prereg_validate.sbatch`, `mrt_gpu_validate_cell.sbatch` | the held-out 3.1B at 16×4×4×16 cell (pre-registered) | `results/prereg_validation_result.json` | cost-model validation row |

The ZeRO-Offload cells the paper prints are the medians over the three
`deepspeed_baseline_16t{,_rep1,_rep2}` directories.

`gran_std_decode_gpu.sbatch` exists so that the granularity table compares decode
cells under one protocol: the coarse and fine partitions were measured with the
scale campaign's 8-token generation, so the standard partition is measured the
same way rather than reused from the 32-token cells of the inference table.

## Node classes and provenance policy

Node classes are reported as *classes*, never machine names. Where the producing
script records `device_total_mb`, the class is verifiable from the committed JSON
itself (40442 MB = A100-40 GB, 81154 MB = A100-80 GB, 95330 MB = H100-94 GB).
Machine names and Slurm job IDs are kept out of the repository, with two recorded
exceptions: the job ID in `results/prereg_validation_result.json`, which is part of
the pre-registration record, and the `"hostname": "ggpu128"` field that
`store_bandwidth.py` writes into
`results/measurement_checks/store_bandwidth_large_8x2x2x8_{cuda_cpu_ram,cpu_disk}.json`
(the node identity of the bandwidth measurement, kept so the two bindings are
visibly from one node). The partition names (`grete:shared`,
`grete-h100:shared`), `module load` lines, `OMP_NUM_THREADS`, `MALLOC_*` settings
and `--gres` / `--constraint` directives are retained deliberately: they are the
measurement provenance, not identity.

## Regenerating the summaries, the figures and the paper

```bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
sbatch slurm/rebuild_summaries.sbatch    # build_exp_summaries.py + paper_v4_figures.py
sbatch slurm/figures.sbatch              # figures only (paper_jsa/paper_v4_figures.py)
sbatch slurm/exp8_aggregate.sbatch       # sustained_report.py + exp8_plots.py
sbatch slurm/verify_artifact.sbatch      # fresh clone -> tests -> regenerate -> diff
```

`verify_artifact.sbatch` needs two variables beyond the `PROJECT_ROOT`/`PYTHON`
convention: **`PROJECT_PARENT`** (it clones `${PROJECT_PARENT}/segmented-llm-training-inference`)
and **`TMPDIR`** (the clone and the smoke checkpoint land under it). Set both, or
edit lines 12 and 16.

`verify_artifact.sbatch` is the gate: it clones the publication repository, runs the
test suite, regenerates every summary (`build_exp_summaries.py`, `eval_cost_laws.py`,
`sustained_report.py`, `exp9_bridge_report.py`, `exp8_plots.py`,
`table_enrichment_report.py`) and every figure, and prints `VERDICT: PASS` only when
all of them exit 0 and no committed JSON changed. Figure PDFs are excluded from the
diff (they are not byte-reproducible); every number is not.

## Superseded material, not used by this paper

These files are kept because they are the record of how the project got here, and
because their result directories are still referenced by the earlier papers. **No
number in the present manuscript is read from any of them.**

Earlier-campaign Slurm jobs (the v1–v3 "ladder", leave-one-out, granularity and
scale campaigns, all of them measured before the mode lattice, the pinned-thread
CPU protocol and the three-repetition rule of Sec. V were adopted):

| job (sbatch) | why it is not used |
|---|---|
| `mrt_gpu_train.sbatch`, `mrt_gpu_train_reordered.sbatch`, `mrt_cpu_train.sbatch` | the technique *ladder* (one technique added per rung), replaced by the 12-mode lattice of `exp3_grid_*` |
| `mrt_gpu_inference.sbatch`, `mrt_cpu_inference.sbatch`, `mrt_cpu_infer_clean.sbatch` | ladder-era serving cells, replaced by `exp4_*` |
| `mrt_gpu_loo.sbatch`, `mrt_gpu_loo_repeats.sbatch`, `mrt_cpu_loo.sbatch` | leave-one-out ablation of the ladder; the present paper ablates by mode, not by technique |
| `mrt_gpu_granularity.sbatch` | granularity sweep at the old protocol, replaced by `exp1_gpu_compare.sbatch` + `exp6_gran_fast_gpu.sbatch` + `gran_std_decode_gpu.sbatch` |
| `mrt_gpu_scale_3b.sbatch`, `mrt_gpu_scale_7b.sbatch`, `mrt_gpu_scale_fitpoints.sbatch`, `mrt_cpu_scale_3b.sbatch`, `mrt_cpu_scale_7b.sbatch`, `mrt_h100_scale_validate.sbatch` | scale campaign at the old protocol, replaced by `exp5_*` |
| `mrt_cpu_full084_rerun.sbatch`, `rerun_A_cpu_bundle.sbatch`, `rerun_D_oldenv_control.sbatch` | the CPU-protocol repair sequence and its old-environment control; superseded by `exp2_anchor_cpu.sbatch` and `audit_cpu_reps.sbatch` |
| `rerun_C_gpu_infer_a40.sbatch`, `rerun_C_gpu_infer_a40_v2.sbatch` | ladder-era serving re-run pinned to A100-40, replaced by `exp4_*` |
| `mrt_full_decode.sbatch` | full-model decode anchor at the old protocol, replaced by `exp4_full_anchor.sbatch` |
| `mrt_deepspeed_baseline.sbatch` | the first ZeRO-Offload baseline, run with 8 CPUs and `OMP_NUM_THREADS` unset; superseded by the thread-pinned `rerun_B_deepspeed_16t*` / `rerun_ds_16t_rep*` runs, and kept only as the un-pinned control |
| `exp8_sustained.sbatch` | the 30-step sizing pilot that chose the exp8 step counts (`results/exp8_sustained/pilot_30step/`); the paper's long-horizon rows come from the nine `exp8_g*` and four `exp8_i*` jobs |
| `quality_t1.sbatch` | the earlier five-epoch T1-only quality run (`results/quality_T1_gpu/`), kept as a record |
| `a2_infer_accuracy.sbatch`, `a2_onnx_only.sbatch` | first-generation accuracy evaluations against the v3 checkpoint, before the matched-triple protocol; replaced by `eval_quality_infer.sbatch`, `onnx_fresh_accuracy2.sbatch` and `onnx_full_accuracy.sbatch` |
| `onnx_fresh_accuracy.sbatch` | the first segmented-ONNX export + accuracy pass on the fresh checkpoint; superseded by `onnx_fresh_accuracy2.sbatch`, which evaluates both providers in both weight placements with `eval_onnx_accuracy.py` |

Earlier-campaign scripts, kept for the same reason: `ladder_runner.py`,
`reassemble_ladder.py`, `loo_runner.py`, `loo_cost.py`, `gpu_train_cost.py`,
`cpu_train_cost.py`, `gpu_inference_cost.py`, `cpu_inference_cost.py`,
`granularity_cost.py`, `full_decode_cost.py`, `onnx_resident_cost.py`,
`scale_summary.py`, `general_fit.py`, `paper_figures.py`, `comparison_figures.py`,
`scale_figures.py`, `plot_ladders.py`, `plot_loo.py`, `plot_decision.py`,
`plot_paper_figs.py`, `plot_paper_extras.py`, and the whole of
`final_paper_scripts_results/full_model/` and
`final_paper_scripts_results/segmented_model/outputs/`. They regenerate the earlier
papers' figures and tables from the earlier papers' result trees; the present
manuscript's regeneration chain is the one at the top of this file.

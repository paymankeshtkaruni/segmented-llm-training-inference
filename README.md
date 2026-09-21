# Memory is a Budget, not a Barrier

**Quality-preserving segmented training and inference of LLMs under minimal resources.**

This repository contains the complete, self-contained code, data, and results to
reproduce every experiment, table, and figure in the paper. A GPT decoder is partitioned
along four axes — embedding width, attention head-groups, feed-forward hidden units, and
output vocabulary — and executed **one segment at a time**, streaming exactly one segment
onto the constrained device while every other segment, gradient, and optimizer moment is
parked in a backing store. With stochastic layers disabled the segmented run reproduces
full-model execution to machine precision (gradients ≈1e-7, one AdamW step ≈1e-8,
reassembly bitwise); with dropout on it trains to the same final accuracy to every
recorded digit, while peak training memory drops by more than an order of magnitude on
both GPU and CPU.

The learning is invariant; only the memory schedule changes.

---

## Results at a glance

At 0.84 B parameters in fp32, the streamed deferred mode (T3) trains in **1,080 MB of
VRAM against the full model's 25,406 MB — 23.5×** — and on a CPU node in **1,288 MB
against 26,934 MB — 20.9×**. The twelve measured training modes span that whole range,
from 11.5 GB at 5.4× the full-model step time down to 1.08 GB at 39×. Serving the same
model streamed through ONNX Runtime needs **401 MB of RAM**, with neither a GPU nor a
training framework. Sources: `grid_ratios_vs_full` and `full_anchors` in
`memory_reduction_techniques/results/table_enrichment.json`,
`results/exp3_grid/grid_summary.json`, `results/exp4_infer/infer_summary.json`.

> **Provenance of the three figures below.** They are the figures of the *earlier*
> (v1–v3) campaign, drawn by `comparison_figures.py` / `scale_figures.py` from
> `results/scale_summary.json`, and they are kept here because they make the shape of
> the result legible at a glance. The manuscript this artifact accompanies uses the nine
> figures under `final_paper_scripts_results/segmented_model/paper_jsa/figures/`;
> **[REPRODUCE.md](REPRODUCE.md) is the authoritative regeneration chain** for every
> table, figure and number in that manuscript. See §4.

![Peak-memory collapse, full vs segmented](final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/figures/paper/comp_cost_dumbbell.png)

…at identical quality — the segmented and full-model validation curves coincide:

![Validation curves coincide](final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/figures/paper/comp_learning.png)

…and the gap widens with scale, past the point where full-model training fits the device
at all:

![Peak training memory across scale](final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/figures/paper/comp_scale_memory.png)

All **nine** manuscript figures regenerate from the committed result JSONs in about ten
seconds (§4 / REPRODUCE.md). The 14 PNGs under
`memory_reduction_techniques/results/figures/paper/`, including the three above, belong
to the earlier campaign.

---

## Repository layout

```
src/sequential_segmented_llm_training_inference/   the library (full model + reference engine, data, export)
scripts/                                          library drivers (segmented train / infer, ONNX export / infer)
tests/                                            434 tests (pytest; 433 run, 1 skipped)
configs/                                          model / segmentation / runtime configs
bpe_tokenizer/ , gpt2_tokenizer/                  tokenizers (BPE-2k small model, GPT-2 large model)
log_lines/                                        dataset (raw logs, processed, generative splits)
environment_versions.txt                          exact package versions of the measurement environment
final_paper_scripts_results/
├── full_model/                                   full-model baseline: scripts, SLURM, result JSONs, figures
└── segmented_model/
    ├── paper_jsa/                                THE MANUSCRIPT: main.tex, references.bib
    │   ├── figures/                              its nine figures (PDF)
    │   └── paper_v4_figures.py                   the script that draws six of them
    ├── segmentation_management/                  segmented execution engine (flat-import port)
    ├── scripts/                                  segmented train / inference / cost / ONNX drivers
    ├── memory_reduction_techniques/              the paper's campaign: drivers, SLURM, results, figures
    │   ├── results/                              every committed result JSON
    │   ├── results/figures/paper/                the EARLIER papers' figures (not the manuscript's)
    │   └── slurm/                                the SLURM job files for every run
    ├── outputs/                                  committed result metrics (JSON), earlier campaign
    ├── figures/ , jsons/ , tables/ , markdown_files/ , smoke_tests/   earlier-campaign material
    └── slurm/                                    SLURM batch scripts
```

> **Not committed (regenerable):** trained checkpoints, offloaded segment stores
> (`_work/`, `*.pt`), exported dense/ONNX model weights (`onnx/`, `*.onnx`, `*.weights`,
> `*.npz`), and SLURM stdout/stderr logs. These are recreated by re-running the training /
> export scripts and are listed in `.gitignore`. Everything needed to **regenerate them** —
> and all result *metrics* and figures — is committed.
>
> One regenerable path is *not* in `.gitignore`: `scripts/train_segmented.py` writes an
> `exports/` directory (YAML protocol + export metadata) into the repository root, so the
> smoke run of §5d leaves it behind as untracked files. Delete it (`rm -rf exports`)
> before comparing `git status`.
>
> **Profiler timelines.** The cost runs record a per-sample memory trace
> (`vram_timeline` / `cpu_ram_timeline` / `moves`, up to half a million samples per run).
> Those arrays are kept for `SEG/seg_cost_{gpu,cpu}` and `SEG/seg_b2_{gpu,cpu}`.
> Elsewhere they are removed, and the file carries either a `timelines_omitted` note
> (earlier campaign) or, in the 13 newer per-rep JSONs, the field value
> `"[stripped for repository size: N samples; regenerate via the corresponding sbatch
> job]"`. Every aggregate the paper quotes — `overall`, `per_phase`, `per_operation`,
> `baseline`, `avg_step_time_s` — is committed in full, and re-running the producing
> script regenerates the complete trace.
>
> **Known consequence.** `results/scale_cpu_large/seg_train_metrics.json` — the CPU
> training panel of the earlier campaign's `comp_memflow` figure — is one of the stripped
> files, so the superseded `comparison_figures.py` aborts in `fig_memflow()` with
> `IndexError: string index out of range` on a fresh clone. Its four other `comp_*`
> figures are written before it and are unaffected, and no figure or number in the
> present manuscript comes from that script. Re-running
> `slurm/mrt_cpu_full084_rerun.sbatch` (itself a superseded job) restores the trace.

---

## 1. Environment

Python 3.10, PyTorch 2.7.0 (CUDA 12.8, cuDNN 9.7.1) on Rocky Linux 8.10 (CUDA toolkit
12.8.2, GCC 13.2.0). Dependencies are pinned in `pyproject.toml` / `poetry.lock`:

```bash
poetry env use python3.10     # see "Which interpreters work" below
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install
```

**Which interpreters work.** `pyproject.toml` requires `python >=3.10,<3.13`, i.e. 3.10,
3.11 or 3.12. The measurements were made on **3.10**; a fresh `poetry install` from the
committed `poetry.lock` was additionally verified end to end on **3.11** on a CPU-only
laptop (≈8 min, no re-lock, no resolution error, `pytest` 433 passed / 1 skipped). If
`poetry env use python3.10` prints `Could not find the python executable python3.10`
(note that Poetry exits 0 on that message, so it fails silently inside a script), either
install 3.10 — `pyenv install 3.10.14`, or the deadsnakes PPA on Debian/Ubuntu, or
`conda create -n seg python=3.10` — or substitute an in-range interpreter:
`poetry env use python3.11`.

**Headless machines.** Poetry 1.8 wants a keyring, and without an unlocked one
`poetry install` aborts with `SecretServiceNotAvailableException … Cannot install
certifi`. Prefixing `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` (as above)
avoids it. The install also pulls the CUDA wheels of `torch 2.7.0+cu128` (≈3 GB) even
with no GPU present; `torch.cuda.is_available()` is then simply `False`, and the CPU
paths, the test suite (§6) and the whole of §4 still run.

`environment_versions.txt` is a `pip freeze` of the exact environment used for the
measurements, and pins the versions the paper quotes (`torch==2.7.0+cu128`,
`transformers==5.12.0`, `tokenizers==0.22.2`, `numpy==2.2.6`, `psutil==7.2.2`,
`onnx==1.22.0`, `nvidia-cudnn-cu12==9.7.1.26`). The torch-free inference path was measured
with ONNX Runtime 1.23.2 (`onnxruntime-gpu`, which provides **both** the CPU and CUDA
execution providers — do not also install plain `onnxruntime`, they conflict).
Note: the committed lock resolves a newer `onnxruntime-gpu` 1.x than the measured
1.23.2; install `onnxruntime-gpu==1.23.2` to match the measured configuration exactly.

`pandas` is in the dependency set but not in `environment_versions.txt`: it is imported
only by the dataset-split regeneration of §2, which the measurement environment never
ran. It is capped below 3.0 on purpose — see §2.

### Optional dependency: DeepSpeed (ZeRO-Offload baseline only)

The ZeRO-Offload comparison (Sec. VII of the paper) needs **`deepspeed==0.19.2`**, which
is *not* part of `pyproject.toml` and is deliberately kept out of the lock file:
DeepSpeed pulls a large CUDA/JIT toolchain and is only used by one script. Install it
into a **separate virtualenv** built on the same `torch==2.7.0+cu128` wheel:

```bash
python3.10 -m venv .venv-deepspeed
.venv-deepspeed/bin/pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv-deepspeed/bin/pip install deepspeed==0.19.2
```

and point one of the **thread-pinned** jobs at it — those are the runs the paper's ZeRO
cells come from (see REPRODUCE.md):

```bash
PYTHON=$PWD/.venv-deepspeed/bin/python \
  sbatch final_paper_scripts_results/segmented_model/memory_reduction_techniques/slurm/rerun_B_deepspeed_16t.sbatch
```

`rerun_ds_16t_rep1.sbatch` and `rerun_ds_16t_rep2.sbatch` are the other two repetitions.
`mrt_deepspeed_baseline.sbatch` is the un-pinned 8-CPU *control*, listed as superseded in
REPRODUCE.md; it defaults to `${PROJECT_PARENT}/dsenv/bin/python` (the interpreter the
original run used) and honours `PYTHON` when you export it. Nothing else in the
repository imports DeepSpeed.

### Running the SLURM job files

Every `*.sbatch` under `final_paper_scripts_results/**/slurm/` is the exact invocation used
for the corresponding result, with the site-specific paths replaced by environment
variables:

```bash
export PROJECT_ROOT=$PWD                 # the repository root
export PYTHON=$(poetry env info -e)      # optional; defaults to `python` on PATH
cd "$PROJECT_ROOT" && sbatch final_paper_scripts_results/.../slurm/<job>.sbatch
```

Submit from the repository root: the `--output` / `--error` paths are relative to the
submit directory, and the three directories they name are kept in the repository by a
committed `.gitkeep` (their contents stay ignored), so a fresh clone can submit without
creating anything first. The `#SBATCH --partition` / `--gres` / `--constraint` /
`--cpus-per-task` directives and the `module load`, `OMP_NUM_THREADS` and `MALLOC_*`
lines are **left intact as measurement provenance** (§3) — adapt them to your site.

Two further variables are used by a handful of jobs, and both must be set before those
jobs run:

- **`PROJECT_PARENT`** — the directory the repository sits in. `verify_artifact.sbatch`
  clones `${PROJECT_PARENT}/segmented-llm-training-inference` into a scratch directory,
  and `mrt_deepspeed_baseline.sbatch` defaults its interpreter to
  `${PROJECT_PARENT}/dsenv/bin/python`. No other job needs it.
- **`TMPDIR`** — scratch space. `verify_artifact.sbatch` puts its clone and smoke
  checkpoint under it; `a2_infer_exact_{gpu,cpu}.sbatch`, `onnx_fresh_accuracy.sbatch`
  and `a2_onnx_only.sbatch` use it for their ONNX export scratch.

One thing to change when adapting a job: every job runs under `set -euo pipefail`, so on
a site without environment modules the `module load gcc/13.2.0 cuda/12.8.2` line is a
hard abort rather than a warning. Delete or replace that line.

---

## 2. Data

The dataset is a system-log labeling task rendered as next-token generation (the log line
is the prompt, the model generates its `Issue: …, level: …` label). The splits are
committed under `log_lines/generative_splits/{train,validation,test}.csv` — 32,569 /
11,521 / 11,655 lines including the header row — so every experiment runs out of the box.
`log_lines/*.csv` are the raw per-node exports, and `log_lines/processed_data/` their
cleaned form, produced by the sibling `data_prep.py`.

To regenerate the splits from the cleaned logs (the split script reads
`log_lines/processed_data/`, not the raw per-node CSVs):

```bash
cd src/sequential_segmented_llm_training_inference/data
poetry run python create_generative_splits.py
```

It rewrites `log_lines/generative_splits/` **in place** (≈11 s), so use
`git status` / `git checkout -- log_lines/` afterwards if you only meant to check.

> **What reproduces.** With the locked `pandas` (2.3.3), all three split files come back
> **byte-for-byte identical** to the committed ones — md5 `c776446f…` / `c8d33c46…` /
> `101ec995…` for `train` / `validation` / `test`. This is why `pyproject.toml` caps
> `pandas` below 3.0: on pandas 3.x the same seed returns the *same 32,569 training
> rows in a different order* (the row set is identical; `diff <(sort a) <(sort b)` is
> empty). The difference arises upstream of the seeded sampling, in how the
> level-balancing step groups and concatenates rows; the sampling itself is
> identical on both majors. The committed split files are canonical either way — every experiment reads
> them, not a regenerated copy.

Tokenizers are committed; the BPE-2k tokenizer can be retrained with
`bpe_tokenizer/train_bpe.py`.

> **Pseudonymization note.** The logs were collected from a production compute cluster.
> Before release, the machine names were replaced by a deterministic pseudonym mapping
> (`node01 … node09`), both in the per-node file names and wherever a machine name occurs
> inside a file (the `source_file` column of the splits, and the `instance="…"` field of
> the raw log text). The substitution is a pure string replacement: row counts, column
> structure and label distributions are unchanged, and the mapping is not published.
> All comparisons in the paper — full vs. segmented equivalence in particular — put the two
> implementations on **identical data**, so they are unaffected by the renaming. If a model
> is retrained on the released (pseudonymized) text, exact token statistics may differ
> marginally from the internal runs, because the substituted strings tokenize slightly
> differently.

---

## 3. Measurement environment (mirrors Sec. V of the paper)

Every number in the paper is a peak recorded by the repository's own profiler, which
samples device VRAM and host RSS on one synchronized clock and annotates every segment
load/release. GPU peak VRAM is a no-miss allocator high-water mark reset per phase; host
RSS is the per-configuration sampled peak (deliberately **not** `ru_maxrss`, which is
monotonic across configurations run in one process). Training is measured at batch 4,
sequence 512, over three steps after a warm-up. Inference uses a 256-token prompt at
batch 1: **32 generated tokens** for the inference table and the serving frontier, and
**8 generated tokens** for the granularity and inference-at-scale tables, where the
decode cells are compared against the scale campaign.

Three environment facts materially affect the numbers, and every job file encodes them:

- **CPU thread pinning, and the SLURM-prolog trap.** CPU jobs request
  `--cpus-per-task=16` **and** export `OMP_NUM_THREADS=16`. The cluster's SLURM prolog
  otherwise leaves `OMP_NUM_THREADS=1`, which serializes PyTorch's CPU GEMMs and inflates
  every CPU time by roughly an order of magnitude. This is the practical trap that forced
  the CPU series to be re-measured (§5c below).
- **glibc allocator settings.** CPU jobs additionally fix
  `MALLOC_MMAP_THRESHOLD_=MALLOC_TRIM_THRESHOLD_=131072`, because peak host RSS is
  allocator-policy sensitive. Every compared CPU cell uses one identical environment.
  The committed control for the pair is `rerun_D_oldenv_control.sbatch`, which reruns the
  identical script and workload under the *old* environment (8 CPUs, `OMP_NUM_THREADS`
  unset, no MALLOC tuning): `results/scale_cpu_large_oldenv/full_train_metrics.json`
  measures **112.0 s** per step at **29,615 MB** peak RSS against **33.0 s** and
  **26,928 MB** for the same cell under the pinned environment
  (`results/scale_cpu_large/full_train_metrics.json`) — 3.4× on time, 10% on memory.
- **GPU node classes.** Results are labeled by node *class* — A100-40 GB, A100-80 GB,
  H100-94 GB — and every compared pair is measured within one class. Each run records the
  device capacity in its metrics JSON (`device_total_mb`: 40442 for A100-40, 81154 for
  A100-80, 95330 for H100), so the class of any committed result is verifiable after the
  fact. **On this cluster `--constraint=40gb_vram` means "≥ 40 GB"**, so the 80 GB nodes
  satisfy it too; pinning the true A100-40 class additionally requires excluding the 80 GB
  node names. The job files that need it carry a comment at that point explaining where a
  site-specific `#SBATCH --exclude=…` goes (the original node names are not published).

---

## 4. Regenerate the paper's tables and figures (seconds, CPU-only)

Every summary JSON and all nine manuscript figures rebuild from the committed result
JSONs — no GPU, no training, about ten seconds in total. This is the chain
`slurm/verify_artifact.sbatch` gates on, and **[REPRODUCE.md](REPRODUCE.md) documents it
object by object**:

```bash
export PROJECT_ROOT=$PWD
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
poetry run python build_exp_summaries.py      # results/exp{1..7}*/ per-rep dirs -> the summary JSONs
poetry run python eval_cost_laws.py           # results/cost_model_eval.json
poetry run python sustained_report.py         # results/exp8_sustained/sustained_summary.json
poetry run python exp9_bridge_report.py       # results/exp9_validation_bridge/bridge_summary.json
poetry run python exp8_plots.py               # results/exp8_sustained/plots/*.pdf   (3 paper figures)
poetry run python table_enrichment_report.py  # results/table_enrichment.json
cd ../paper_jsa
poetry run python paper_v4_figures.py         # figures/*.pdf                        (6 paper figures)
```

Use the project interpreter (`poetry run python`, or `$(poetry env info -e)`); a bare
`python` on `PATH` is whatever the shell happens to find. Verified on a CPU-only laptop:
all seven scripts exit 0, **no tracked `.json` changes**, and only the figure PDFs differ
byte-wise (PDFs are not byte-reproducible). REPRODUCE.md carries the exact
figure → script → source-JSON mapping and the provenance of every headline number.

<details><summary>Earlier campaign (v1–v3) figures — superseded, not the manuscript's</summary>

These five scripts redraw the 14 PNGs under `results/figures/paper/`. REPRODUCE.md lists
them under *Superseded material*; **no number or figure in the present manuscript comes
from them**, and they are kept only as the record of how the project got here.
`scale_summary.py` and `general_fit.py` regenerate their JSONs byte-identically;
`comparison_figures.py` aborts partway for the reason given in the profiler-timeline note
above.

```bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
poetry run python scale_summary.py       # -> results/scale_summary.json
poetry run python general_fit.py         # -> results/general_fit.json
poetry run python comparison_figures.py  # comp_learning, comp_cost_dumbbell, comp_onnx, comp_granularity, [comp_memflow FAILS]
poetry run python paper_figures.py       # fig1_schematic, fig2_waterfall, fig4_pareto, fig5_loo, fig7_phase (+ extras)
poetry run python scale_figures.py       # comp_scale_memory, comp_scale_model
```

</details>

---

## 5. Re-run the measurements

Every measurement in the paper is one or more `*.sbatch` jobs under
`memory_reduction_techniques/slurm/`, and **[REPRODUCE.md](REPRODUCE.md) maps every paper
object to the job that produced it** — the exactness jobs, the matched-pair accuracy
tier, the eight serving pipelines, the twelve-mode grid, the scale and granularity
campaigns, the long-horizon runs, the measurement self-checks and the pre-registered
cell. Run all commands from the repository root, offline:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PROJECT_ROOT=$PWD
export PYTHON=$(poetry env info -e)
```

Two model sizes (see `memory_reduction_techniques/segmentation_management/config.py`
presets): `small_8x2x2x8` (BPE-2k) for the **accuracy-equivalence** experiments (CPU or a
modest GPU); `large_8x2x2x8` (≈0.84B, GPT-2 vocab) for the **memory/time cost**
experiments (A100-class GPU or a large-RAM CPU node). The scale study adds
`xl15b_*`, `xl3b_*`, `xl5b_*`, `xxl7b_*`.

### 5a. The paper's campaign

```bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
sbatch slurm/x1_verify_gpu.sbatch          # training identity, 12 modes
sbatch slurm/a2_infer_exact_gpu.sbatch     # inference identity, 4 modes
sbatch slurm/quality_triple.sbatch         # matched-pair accuracy tier (full + T1 + T2)
sbatch slurm/exp3_grid_gpu.sbatch          # the 12-mode training grid
sbatch slurm/exp4_infer_torch_gpu.sbatch   # the serving modes
sbatch slurm/exp5_xl3b_gpu.sbatch          # the scale series
sbatch slurm/exp8_g1_084b_resident_imm.sbatch   # a long-horizon training run
sbatch slurm/table_enrichment.sbatch       # anchors, per-mode ratios, long-run bands
```

REPRODUCE.md lists the full set (every `exp*`, `rerun_*`, `audit_*`, `x1_*`, `a2_*`,
`onnx_*`, `exp8_g*`, `exp8_i*`, `exp9_*`, `prereg_*` job) together with the cells each
one writes and the paper object it feeds.

### 5b. The experiments added after the first release

| Experiment | Driver | SLURM | Results |
|---|---|---|---|
| **Scale study, GPU** — 0.84B / 1.57B / 3.09B / 5.12B / 6.86B, partition fixed at 8×2×2×8 | `scale_cost.py --preset {xl3b,xl5b,xl15b,xxl7b}_8x2x2x8 --device cuda --run {full,seg}_{train,infer}` | `exp5_xl15b_gpu.sbatch`, `exp5_xl3b_gpu.sbatch`, `exp5_xl5b_gpu.sbatch`, `exp5_xl5b_resident_gpu.sbatch`, `exp5_xxl7b_gpu.sbatch`, `exp5_80gb_gpu.sbatch` (superseded predecessors: `mrt_gpu_scale_{3b,7b}.sbatch`, `mrt_gpu_scale_fitpoints.sbatch` → `results/scale_gpu_*`) | `results/exp5_scale/gpu{40,80}_*`, folded into `results/exp5_scale/scale_summary.json` |
| **Scale study, CPU** | `scale_cost.py … --device cpu` | `exp5_cpu_xl3b.sbatch`, `exp5_cpu_xxl7b.sbatch` (superseded predecessors: `mrt_cpu_scale_{3b,7b}.sbatch` → `results/scale_cpu_{xl3b,xxl7b}`) | `results/exp5_scale/cpu_*` |
| **OOM protocol** — at 3.1B/6.9B on the 40 GB card the full-model step aborts out-of-memory during its first step; the abort is *caught and recorded as the measurement* (`"OOM(warmup)"`), never silently retried at a smaller batch | same as above | same | `results/exp5_scale/gpu40_{xl3b,xxl7b}_full_train_rep1/full_train_metrics.json` |
| **H100 validation of the OOM estimates** — the estimated requirement behind each abort is then tested on a 94 GB H100: 3.1B fits and measures 67.6 GB, 6.9B aborts with 94.4 GB already reserved on a 95.3 GB device | `scale_cost.py --run full_train` | `exp5_h100.sbatch` (superseded predecessor: `mrt_h100_scale_validate.sbatch` → `results/scale_h100_{xl3b,xxl7b}`) | `results/exp5_scale/h100_{xl3b,xxl7b}_full_train_rep1` |
| **DeepSpeed ZeRO-Offload baseline** — same model, protocol and precision; ZeRO-2 and ZeRO-3 with CPU offload, no activation checkpointing, `torch` AdamW client optimizer | `deepspeed_cost.py --mode {zero2,zero3}_offload` | `mrt_deepspeed_baseline.sbatch` (superseded: the original un-pinned 8-CPU control) | `results/deepspeed_baseline` |
| **…three-node repetition protocol** — the baseline is re-run under the paper's 16-pinned-thread environment on **three distinct A100-40 nodes**, because the ZeRO optimizer runs on CPU threads. Peak was bit-identical across the three; step time was not, so each ZeRO point is plotted as the **median with a bar spanning the node-to-node spread**. These are the cells the paper prints | `deepspeed_cost.py` | `rerun_B_deepspeed_16t.sbatch` (+`_v2`), `rerun_ds_16t_rep1.sbatch`, `rerun_ds_16t_rep2.sbatch` | `results/deepspeed_baseline_16t{,_rep1,_rep2}` |
| **Pre-registered cell** — the general memory/time law is fitted on the granularity sweep + the scale series, then tested out of sample on a **new size and a new partition at once** (3.09B @ 16×4×4×16). The prediction (**110.3 s, 1330 MB**) was committed *before* the job ran (`results/prereg_prediction.json`); the measured mean over three reps is **102.7 s / 1268 MB**, i.e. time **+7.1%** and memory **+4.9%** (`results/prereg_validation_result.json`), the additive time law an upper bound | `general_fit.py` (predicts), `scale_cost.py --preset xl3b_16x4x4x16` (measures) | `prereg_validate.sbatch`, `mrt_gpu_validate_cell.sbatch` | `results/prereg_prediction.json`, `results/prereg_validation_result.json`, `results/scale_gpu_xl3b_16x4x4x16` |
| **Naive-segmented anchor** — the partition with every technique off (tech code `00000000000`, dropout 0.1), measured on GPU and CPU; the reference point that isolates what the six always-on bounds buy (Table V naive row, Table VI anchor rows) | `measure_one.py --tech-code 00000000000` | `exp2_anchor_gpu.sbatch`, `exp2_anchor_cpu.sbatch`; summarized by `table_enrichment.sbatch` | `results/exp2_anchor/{gpu,cpu}_rep*/naive_anchor_met.json`, `results/table_enrichment.json` (`naive_anchors`) |
| **0.84B CPU re-measurement** — the 0.84B CPU cells are re-measured under the *same* 16-pinned-thread + fixed-malloc environment as the 3.09B/6.86B CPU cells, so the CPU column of the scale table is internally consistent | `scale_cost.py --preset large_8x2x2x8 --device cpu` | `audit_cpu_reps.sbatch` → `results/exp2_anchor/cpu_full_rep*` (superseded predecessors: `mrt_cpu_full084_rerun.sbatch`, `rerun_A_cpu_bundle.sbatch` → `results/scale_cpu_large`) | `results/exp2_anchor/cpu_full_rep*`, `results/scale_cpu_large` |
| **Old-environment control** — the identical script and workload under the *old* environment (8 CPUs, `OMP_NUM_THREADS` unset, no MALLOC tuning), to prove the correction is environment-caused and not script-caused | `scale_cost.py` | `rerun_D_oldenv_control.sbatch` | `results/scale_cpu_large_oldenv` |
| **Long-horizon validation (exp8/exp9)** — every headline training and serving mode run for 128–2,048 consecutive steps / 500–2,000 requests with per-step cost recording, plus the validation-cost bridge and the measurement self-checks | `exp8_long_run.py`, `exp8_long_serve.py`, `store_bandwidth.py` | the nine `exp8_g*.sbatch`, four `exp8_i*.sbatch`, `exp9_validation_bridge.sbatch`, `measurement_checks.sbatch`, `store_bandwidth.sbatch` | `results/exp8_sustained/`, `results/exp9_validation_bridge/`, `results/measurement_checks/` |

### 5c. Quick GPU-free sanity check (under a minute)

The smoke run `slurm/verify_artifact.sbatch` finishes with — a tiny model, end to end
through the segmented trainer, on CPU in about 50 s:

```bash
PYTHONPATH=src poetry run python scripts/train_segmented.py \
  --epochs 1 --batch-size 2 --n-layers 2 --d-model 128 --n-heads 4 --d-ff 256 \
  --attention-segments 2 --mlp-chunks 2 --checkpoint-dir /tmp/smoke \
  --device cpu --max-train-steps 3 --max-val-steps 3
rm -rf exports          # see the note under "Repository layout"
```

<details><summary>Earlier campaign (v1–v3) cost drivers — superseded</summary>

The technique *ladder*, the leave-one-out ablation and the first granularity and scale
sweeps were measured before the mode lattice, the pinned-thread CPU protocol and the
three-repetition rule of Sec. V were adopted. REPRODUCE.md lists them under *Superseded
material*; no number in the present manuscript comes from them.

```bash
cd final_paper_scripts_results/segmented_model/memory_reduction_techniques
poetry run python gpu_train_cost.py     --preset large_8x2x2x8 --device cuda
poetry run python cpu_train_cost.py     --preset large_8x2x2x8 --device cpu
poetry run python gpu_inference_cost.py --preset large_8x2x2x8 --device cuda
poetry run python cpu_inference_cost.py --preset large_8x2x2x8 --device cpu
poetry run python loo_cost.py           --preset large_8x2x2x8 --device cuda
poetry run python granularity_cost.py   --device cuda
poetry run python full_decode_cost.py   --preset large_8x2x2x8 --device cuda
```

SLURM: `slurm/mrt_{gpu,cpu}_{train,inference,loo}.sbatch`, `mrt_gpu_granularity.sbatch`,
`mrt_full_decode.sbatch`. The full-model baseline drivers under
`final_paper_scripts_results/full_model/` and the result trees under
`final_paper_scripts_results/segmented_model/outputs/` belong to the same earlier
campaigns.

</details>

---

## 6. Tests

```bash
poetry run pytest -q      # 433 passed, 1 skipped
```

434 tests are collected. One of them needs CUDA: on a machine without a GPU it is
skipped (**433 passed, 1 skipped**, verified on a CPU-only laptop with Python 3.11,
≈100 s), and on a CUDA machine all **434 pass** (≈38 s, `verify_artifact.sbatch`).
Nothing else in the suite requires a GPU.

The suite covers the segmentation identities (attention concat, MLP running sum, chunked
cross-entropy), the recomputation backward, the segment stores and loader, RNG capture and
restore, segment-wise AdamW, checkpointing, the YAML protocol export/import, and the
full-model export parity that underpins the equivalence claims.

---

## Citation / license

MIT (see `LICENSE`). The paper is not part of this repository; this is the artifact it
points to.

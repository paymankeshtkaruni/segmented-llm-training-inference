#!/bin/bash
#SBATCH --job-name=audit_cpu_reps
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=12:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/audit_cpu_reps_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
run(){ echo "=== $1 ==="; shift; "$@" || echo "[recorded failure]"; }
# audit: 0.84B CPU full-train anchor was a single Jul-31 run; fresh x3 on the pinned protocol
for rep in 1 2 3; do
  run "0.84B cpu full_train rep$rep" "$PY" scale_cost.py --preset large_8x2x2x8 --device cpu --run full_train --out-dir "results/exp2_anchor/cpu_full_rep$rep"
done
# audit: single-rep CPU scale cells -> add rep2/rep3
for rep in 2 3; do
  run "xl3b cpu full_infer rep$rep" "$PY" scale_cost.py --preset xl3b_8x2x2x8 --device cpu --run full_infer --out-dir "results/exp5_scale/cpu_xl3b_full_infer_rep$rep"
  run "xl3b cpu I4 rep$rep" "$PY" measure_one.py --kind infer --preset xl3b_8x2x2x8 --device cpu --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp5_scale/cpu_xl3b_I4_rep$rep"
  run "xl15b cpu full_infer rep$rep" "$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cpu --run full_infer --out-dir "results/exp6_fast/cpu_xl15b_full_infer_rep$rep"
  run "xl15b cpu I4 rep$rep" "$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cpu --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp6_fast/cpu_xl15b_I4_rep$rep"
  run "xxl7b cpu full_train rep$rep" "$PY" scale_cost.py --preset xxl7b_8x2x2x8 --device cpu --run full_train --out-dir "results/exp5_scale/cpu_xxl7b_full_train_rep$rep"
  run "xxl7b cpu full_infer rep$rep" "$PY" scale_cost.py --preset xxl7b_8x2x2x8 --device cpu --run full_infer --out-dir "results/exp5_scale/cpu_xxl7b_full_infer_rep$rep"
  run "xxl7b cpu I4 rep$rep" "$PY" measure_one.py --kind infer --preset xxl7b_8x2x2x8 --device cpu --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp5_scale/cpu_xxl7b_I4_rep$rep"
done
echo "=== rebuild summaries ==="
"$PY" build_exp_summaries.py
echo "Done: $(date)"

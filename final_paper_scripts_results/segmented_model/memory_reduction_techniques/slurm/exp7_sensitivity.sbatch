#!/bin/bash
#SBATCH --job-name=exp7_sensitivity
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=08:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp7_sensitivity_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
# Sensitivity of the streamed floor to batch and sequence length at 0.84B:
# streamed deferred mode (dropout on), batch {1,4,8} x seq {128,256,512}, x3.
for rep in 1 2 3; do
  for B in 1 4 8; do
    for S in 128 256 512; do
      echo "=== b${B}_s${S} rep$rep ==="
      "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda \
          --tech-code 11111111100 --update-style after_full \
          --batch $B --seq-len $S --n-steps 3 --prefix T3 \
          --out-dir "results/exp7_sensitivity/b${B}_s${S}_rep$rep"
    done
  done
done
echo "=== rebuild summaries ==="
"$PY" build_exp_summaries.py
echo "Done: $(date)"

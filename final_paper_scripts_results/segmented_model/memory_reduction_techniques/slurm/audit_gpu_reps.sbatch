#!/bin/bash
#SBATCH --job-name=audit_gpu_reps
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/audit_gpu_reps_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
# audit: the 6.9B resident headline (T2) had a single rep; add rep2/rep3
for rep in 2 3; do
  echo "=== xxl7b T2 rep$rep ==="
  "$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11110111000 --update-style after_full --n-steps 2 --prefix T2 --out-dir "results/exp5_scale/gpu40_xxl7b_T2_rep$rep"
done
echo "Done: $(date)"

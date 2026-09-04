#!/bin/bash
#SBATCH --job-name=prereg_validate
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/prereg_validate_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/prereg_validate_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
for rep in 1 2 3; do
  echo "=== rep $rep ==="
  "$PY" measure_one.py --kind train --preset xl3b_16x4x4x16 --device cuda \
      --tech-code 11111111100 --update-style after_full --n-steps 3 \
      --prefix T3 --out-dir "results/prereg_validate/rep$rep"
done
echo "Done: $(date)"

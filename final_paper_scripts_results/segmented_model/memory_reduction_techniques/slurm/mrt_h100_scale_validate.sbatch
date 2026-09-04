#!/bin/bash
#SBATCH --job-name=mrt_h100_validate
#SBATCH --partition=grete-h100:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_h100_validate_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_h100_validate_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# Validate the full-model requirement ESTIMATES on a 94 GB H100:
#   3B  (est. ~73 GB)  -> expected to FIT and be measured (estimate -> measurement);
#   7B  (est. ~143 GB) -> expected to OOM even here (recorded).
"$PYTHON" "$MRT/scale_cost.py" --preset xl3b_8x2x2x8 --device cuda --run full_train \
    --batch 4 --n-steps 2 --out-dir "$MRT/results/scale_h100_xl3b"
"$PYTHON" "$MRT/scale_cost.py" --preset xxl7b_8x2x2x8 --device cuda --run full_train \
    --batch 4 --n-steps 2 --out-dir "$MRT/results/scale_h100_xxl7b"
echo "Done: $(date)"

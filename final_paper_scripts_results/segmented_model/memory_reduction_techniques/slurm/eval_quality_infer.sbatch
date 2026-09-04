#!/bin/bash
#SBATCH --job-name=eval_qual_infer
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=03:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/eval_qual_infer_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/eval_qual_infer_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
"$PY" eval_quality_inference.py --device cuda --batch 64
echo "Done: $(date)"

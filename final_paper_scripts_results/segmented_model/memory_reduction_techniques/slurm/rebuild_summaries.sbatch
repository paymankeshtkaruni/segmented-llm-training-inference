#!/bin/bash
#SBATCH --job-name=rebuild_summaries
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:20:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rebuild_summaries_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
PY="${PYTHON:-python}"
"$PY" build_exp_summaries.py
cd ../paper_v4_tpds
"$PY" paper_v4_figures.py
echo "Done: $(date)"

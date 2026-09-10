#!/bin/bash
#SBATCH --job-name=v4_figs
#SBATCH --partition=grete:shared
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/figures_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
PY="${PYTHON:-python}"
cd "$SLURM_SUBMIT_DIR/../paper_jsa"
$PY paper_v4_figures.py

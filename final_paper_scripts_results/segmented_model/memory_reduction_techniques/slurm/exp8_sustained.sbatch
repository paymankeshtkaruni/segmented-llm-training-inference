#!/bin/bash
#SBATCH --job-name=exp8_sustained
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=05:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp8_sustained_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
# Sustained-training stability: 6.9B resident (recompute, deferred), 30 steps.
echo "=== 6.9B resident, 30 steps ==="
"$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda \
    --tech-code 11110111000 --update-style after_full \
    --n-steps 30 --prefix T2sus \
    --out-dir "results/exp8_sustained/rep1"
echo "=== per-step peak report ==="
"$PY" sustained_report.py results/exp8_sustained/rep1/T2sus_met.json
echo "Done: $(date)"

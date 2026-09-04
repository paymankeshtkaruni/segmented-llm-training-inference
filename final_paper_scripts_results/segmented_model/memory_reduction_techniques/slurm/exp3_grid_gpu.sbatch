#!/bin/bash
#SBATCH --job-name=exp3_grid_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=06:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp3_grid_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp3_grid_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
# The 12-mode grid: dropout x recompute x streaming x update style.
# columns: name  tech_code  update_style  dropout(- = preset default 0.1)
MODES="
T1  11100111000 after_full -
T2  11110111000 after_full -
T3  11111111100 after_full -
T4  11100111000 after_full 0.0
T5  11110111000 after_full 0.0
T6  11111111100 after_full 0.0
T7  11100111000 immediate  -
T8  11110111000 immediate  -
T9  11111111100 immediate  -
T10 11100111000 immediate  0.0
T11 11110111000 immediate  0.0
T12 11111111100 immediate  0.0
"
for rep in 1 2 3; do
  echo "$MODES" | while read -r name code style drop; do
    [ -z "$name" ] && continue
    DARG=""; [ "$drop" != "-" ] && DARG="--dropout $drop"
    echo "=== rep $rep: $name ($code $style drop=$drop) ==="
    "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda \
        --tech-code "$code" --update-style "$style" $DARG \
        --prefix "$name" --out-dir "results/exp3_grid/gpu_${name}_rep$rep"
  done
done
echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=exp9_validation_bridge
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=08:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp9_validation_bridge_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
# exp9: validation-cost bridge. The EXACT cost protocol (measure_one/run_cost:
# forward+backward+optimizer+VALIDATION per step, per-step cache trim, fresh
# process per cell) at 5 steps, for each exp8 training configuration.
# avg_step(exp9) - avg_step(exp8, first 5 steps) = cost of one validation pass;
# cross-checked against run_cost's own per-phase validation timing.
run(){ echo "=== $1 ==="; shift; "$@" || echo "[recorded failure]"; }
run g1 "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate  --n-steps 5 --prefix g1 --out-dir results/exp9_validation_bridge/g1
run g2 "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda --tech-code 11110111000 --update-style after_full --n-steps 5 --prefix g2 --out-dir results/exp9_validation_bridge/g2
run g3 "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda --tech-code 11111111100 --update-style after_full --n-steps 5 --prefix g3 --out-dir results/exp9_validation_bridge/g3
run g4 "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cuda --tech-code 11111111100 --update-style immediate  --n-steps 5 --prefix g4 --out-dir results/exp9_validation_bridge/g4
run g5 "$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11110111000 --update-style after_full --n-steps 5 --prefix g5 --out-dir results/exp9_validation_bridge/g5
run g6 "$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11111111100 --update-style immediate  --n-steps 5 --prefix g6 --out-dir results/exp9_validation_bridge/g6
run g7 "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cpu  --tech-code 11111111100 --update-style after_full --n-steps 5 --prefix g7 --out-dir results/exp9_validation_bridge/g7
run g8 "$PY" measure_one.py --kind train --preset large_8x2x2x8 --device cpu  --tech-code 11100111000 --update-style immediate  --n-steps 5 --prefix g8 --out-dir results/exp9_validation_bridge/g8
run g9 "$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cpu  --tech-code 11111111100 --update-style immediate  --n-steps 5 --prefix g9 --out-dir results/exp9_validation_bridge/g9
echo "Done: $(date)"

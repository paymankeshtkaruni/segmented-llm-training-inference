#!/bin/bash
#SBATCH --job-name=exp5_xl15b_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=10:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp5_xl15b_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp5_xl15b_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
echo "=== xl15b_8x2x2x8 full_train rep1 ==="
"$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_train --out-dir "results/exp5_scale/gpu40_xl15b_full_train_rep1" || echo "[recorded failure/OOM]"
echo "=== xl15b_8x2x2x8 full_train rep2 ==="
"$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_train --out-dir "results/exp5_scale/gpu40_xl15b_full_train_rep2" || echo "[recorded failure/OOM]"
echo "=== xl15b_8x2x2x8 full_train rep3 ==="
"$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_train --out-dir "results/exp5_scale/gpu40_xl15b_full_train_rep3" || echo "[recorded failure/OOM]"
echo "=== xl15b_8x2x2x8 T3 rep1 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11111111100 --update-style after_full --n-steps 3 --prefix T3 --out-dir "results/exp5_scale/gpu40_xl15b_T3_rep1"
echo "=== xl15b_8x2x2x8 T3 rep2 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11111111100 --update-style after_full --n-steps 3 --prefix T3 --out-dir "results/exp5_scale/gpu40_xl15b_T3_rep2"
echo "=== xl15b_8x2x2x8 T3 rep3 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11111111100 --update-style after_full --n-steps 3 --prefix T3 --out-dir "results/exp5_scale/gpu40_xl15b_T3_rep3"
echo "=== xl15b_8x2x2x8 T9 rep1 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11111111100 --update-style immediate --n-steps 3 --prefix T9 --out-dir "results/exp5_scale/gpu40_xl15b_T9_rep1"
echo "=== xl15b_8x2x2x8 T9 rep2 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11111111100 --update-style immediate --n-steps 3 --prefix T9 --out-dir "results/exp5_scale/gpu40_xl15b_T9_rep2"
echo "=== xl15b_8x2x2x8 T9 rep3 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11111111100 --update-style immediate --n-steps 3 --prefix T9 --out-dir "results/exp5_scale/gpu40_xl15b_T9_rep3"
echo "Done: $(date)"

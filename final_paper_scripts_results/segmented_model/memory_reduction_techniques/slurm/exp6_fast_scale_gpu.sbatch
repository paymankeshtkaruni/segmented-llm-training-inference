#!/bin/bash
#SBATCH --job-name=exp6_fast_scale_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=20:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp6_fast_scale_gpu_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp6_fast_scale_gpu_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
echo "=== gpu40_xl15b T7 rep1 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 3 --prefix T7 --out-dir "results/exp6_fast/gpu40_xl15b_T7_rep1"
echo "=== gpu40_xl15b T7 rep2 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 3 --prefix T7 --out-dir "results/exp6_fast/gpu40_xl15b_T7_rep2"
echo "=== gpu40_xl15b T7 rep3 ==="
"$PY" measure_one.py --kind train --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 3 --prefix T7 --out-dir "results/exp6_fast/gpu40_xl15b_T7_rep3"
echo "=== gpu40_xl3b T7 rep1 ==="
"$PY" measure_one.py --kind train --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 3 --prefix T7 --out-dir "results/exp6_fast/gpu40_xl3b_T7_rep1"
echo "=== gpu40_xl3b T7 rep2 ==="
"$PY" measure_one.py --kind train --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 3 --prefix T7 --out-dir "results/exp6_fast/gpu40_xl3b_T7_rep2"
echo "=== gpu40_xl3b T7 rep3 ==="
"$PY" measure_one.py --kind train --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 3 --prefix T7 --out-dir "results/exp6_fast/gpu40_xl3b_T7_rep3"
echo "=== gpu40_xxl7b T7 rep1 ==="
"$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 2 --prefix T7 --out-dir "results/exp6_fast/gpu40_xxl7b_T7_rep1"
echo "=== gpu40_xxl7b T7 rep2 ==="
"$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 2 --prefix T7 --out-dir "results/exp6_fast/gpu40_xxl7b_T7_rep2"
echo "=== gpu40_xxl7b T7 rep3 ==="
"$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --update-style immediate --n-steps 2 --prefix T7 --out-dir "results/exp6_fast/gpu40_xxl7b_T7_rep3"
echo "=== gpu40_xl15b I1 rep1 ==="
"$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl15b_I1_rep1"
echo "=== gpu40_xl15b I1 rep2 ==="
"$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl15b_I1_rep2"
echo "=== gpu40_xl15b I1 rep3 ==="
"$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl15b_I1_rep3"
echo "=== gpu40_xl3b I1 rep1 ==="
"$PY" measure_one.py --kind infer --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl3b_I1_rep1"
echo "=== gpu40_xl3b I1 rep2 ==="
"$PY" measure_one.py --kind infer --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl3b_I1_rep2"
echo "=== gpu40_xl3b I1 rep3 ==="
"$PY" measure_one.py --kind infer --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl3b_I1_rep3"
echo "=== gpu40_xxl7b I1 rep1 ==="
"$PY" measure_one.py --kind infer --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xxl7b_I1_rep1"
echo "=== gpu40_xxl7b I1 rep2 ==="
"$PY" measure_one.py --kind infer --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xxl7b_I1_rep2"
echo "=== gpu40_xxl7b I1 rep3 ==="
"$PY" measure_one.py --kind infer --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xxl7b_I1_rep3"
echo "=== gpu40_xl15b full_infer rep1 ==="
"$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_infer --out-dir "results/exp6_fast/gpu40_xl15b_full_infer_rep1" || echo "[recorded failure]"
echo "=== gpu40_xl15b full_infer rep2 ==="
"$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_infer --out-dir "results/exp6_fast/gpu40_xl15b_full_infer_rep2" || echo "[recorded failure]"
echo "=== gpu40_xl15b full_infer rep3 ==="
"$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_infer --out-dir "results/exp6_fast/gpu40_xl15b_full_infer_rep3" || echo "[recorded failure]"
echo "=== gpu40_xl15b I4 rep1 ==="
"$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp6_fast/gpu40_xl15b_I4_rep1"
echo "=== gpu40_xl15b I4 rep2 ==="
"$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp6_fast/gpu40_xl15b_I4_rep2"
echo "=== gpu40_xl15b I4 rep3 ==="
"$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp6_fast/gpu40_xl15b_I4_rep3"
echo "Done: $(date)"

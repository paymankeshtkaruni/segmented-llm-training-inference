#!/bin/bash
#SBATCH --job-name=exp6b_remainder
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=16:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp6b_remainder_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp6b_remainder_%j.err
set -uo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
run() { echo "=== $1 ==="; shift; "$@" || echo "[recorded failure: $?]"; }
run "gpu40_xl15b I1 rep1" "$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl15b_I1_rep1"
run "gpu40_xl15b I1 rep2" "$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl15b_I1_rep2"
run "gpu40_xl15b I1 rep3" "$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl15b_I1_rep3"
run "gpu40_xl3b I1 rep1" "$PY" measure_one.py --kind infer --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl3b_I1_rep1"
run "gpu40_xl3b I1 rep2" "$PY" measure_one.py --kind infer --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl3b_I1_rep2"
run "gpu40_xl3b I1 rep3" "$PY" measure_one.py --kind infer --preset xl3b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xl3b_I1_rep3"
run "gpu40_xxl7b I1 rep1" "$PY" measure_one.py --kind infer --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xxl7b_I1_rep1"
run "gpu40_xxl7b I1 rep2" "$PY" measure_one.py --kind infer --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xxl7b_I1_rep2"
run "gpu40_xxl7b I1 rep3" "$PY" measure_one.py --kind infer --preset xxl7b_8x2x2x8 --device cuda --tech-code 11100111000 --prompt-len 256 --gen-tokens 8 --prefix I1 --out-dir "results/exp6_fast/gpu40_xxl7b_I1_rep3"
run "gpu40_xxl7b T8 rep1" "$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11110111000 --update-style immediate --n-steps 2 --prefix T8 --out-dir "results/exp6_fast/gpu40_xxl7b_T8_rep1"
run "gpu40_xxl7b T8 rep2" "$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11110111000 --update-style immediate --n-steps 2 --prefix T8 --out-dir "results/exp6_fast/gpu40_xxl7b_T8_rep2"
run "gpu40_xxl7b T8 rep3" "$PY" measure_one.py --kind train --preset xxl7b_8x2x2x8 --device cuda --tech-code 11110111000 --update-style immediate --n-steps 2 --prefix T8 --out-dir "results/exp6_fast/gpu40_xxl7b_T8_rep3"
run "gpu40_xl15b full_infer rep1" "$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_infer --out-dir "results/exp6_fast/gpu40_xl15b_full_infer_rep1"
run "gpu40_xl15b I4 rep1" "$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp6_fast/gpu40_xl15b_I4_rep1"
run "gpu40_xl15b full_infer rep2" "$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_infer --out-dir "results/exp6_fast/gpu40_xl15b_full_infer_rep2"
run "gpu40_xl15b I4 rep2" "$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp6_fast/gpu40_xl15b_I4_rep2"
run "gpu40_xl15b full_infer rep3" "$PY" scale_cost.py --preset xl15b_8x2x2x8 --device cuda --run full_infer --out-dir "results/exp6_fast/gpu40_xl15b_full_infer_rep3"
run "gpu40_xl15b I4 rep3" "$PY" measure_one.py --kind infer --preset xl15b_8x2x2x8 --device cuda --tech-code 11101111101 --prompt-len 256 --gen-tokens 8 --prefix I4 --out-dir "results/exp6_fast/gpu40_xl15b_I4_rep3"
echo "Done: $(date)"

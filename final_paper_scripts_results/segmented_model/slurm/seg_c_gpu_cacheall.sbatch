#!/bin/bash
#SBATCH --job-name=seg_c_gpu_ca
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:45:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/seg_c_gpu_ca_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/seg_c_gpu_ca_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PYTHON="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
"$PYTHON" final_paper_scripts_results/segmented_model/scripts/seg_c_onnx_cost.py --onnx-dir final_paper_scripts_results/segmented_model/onnx/c_seg_large --provider cuda     --batch 4 --seq 512 --gen-tokens 2 --prompt-len 64 --preload-weights --cache-all     --out-dir final_paper_scripts_results/segmented_model/outputs/seg_c_gpu_cacheall --prefix seg_c
echo "Done: $(date)"

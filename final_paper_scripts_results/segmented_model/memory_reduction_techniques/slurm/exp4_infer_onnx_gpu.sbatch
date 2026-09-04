#!/bin/bash
#SBATCH --job-name=exp4_onnx_gpu
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=05:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp4_onnx_gpu\_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/exp4_onnx_gpu\_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
PY="${PYTHON:-python}"
echo "Node: $(hostname)  Start: $(date)"
EXP="final_paper_scripts_results/segmented_model/onnx/c_seg_large"
for rep in 1 2 3; do
  for mode in O4_stream O3_preload; do
    FLAG=""; [ "$mode" = "O3_preload" ] && FLAG="--preload-weights"
    echo "=== rep $rep: $mode ==="
    "$PY" final_paper_scripts_results/segmented_model/scripts/seg_c_onnx_cost.py --onnx-dir "$EXP" --provider cuda \
        --batch 1 --prompt-len 256 --gen-tokens 32 $FLAG \
        --prefix "$mode" --out-dir "final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/exp4_infer/onnx_gpu_${mode}_rep$rep"
  done
done
echo "Done: $(date)"

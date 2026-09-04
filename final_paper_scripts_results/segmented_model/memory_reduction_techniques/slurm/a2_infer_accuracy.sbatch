#!/bin/bash
#SBATCH --job-name=a2_infer_acc
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a2_infer_acc_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/a2_infer_acc_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
OUT=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/a2_infer_accuracy
mkdir -p "$OUT"
echo "Node: $(hostname)  Start: $(date)"
# A2: full test-set evaluation (11,654 samples) through the segmented streaming
# inference path, greedy decoding, paper checkpoint (GPU quality run, 'best').
PYTHONPATH=src "$PY" scripts/infer_segmented.py \
    --checkpoint experiments/outputs_paper/checkpoints/segmented_gpu \
    --checkpoint-name best \
    --evaluate-test --num-test-examples 11654 --eval-batch-size 32 \
    --device cuda \
    --metrics-output "$OUT/metrics.json"
echo "Done: $(date)"

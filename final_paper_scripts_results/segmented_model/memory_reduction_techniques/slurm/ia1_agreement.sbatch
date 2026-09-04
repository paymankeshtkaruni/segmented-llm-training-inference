#!/bin/bash
#SBATCH --job-name=ia1_agreement
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/ia1_agreement_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/ia1_agreement_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
BASE="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
MRT="$BASE/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
cd "$MRT"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PY="${PYTHON:-python}"
EXP="$BASE/final_paper_scripts_results/segmented_model/onnx/c_seg_quality_fresh"
OUT="$MRT/results/quality_fresh_pair"
echo "Node: $(hostname)  Start: $(date)"
echo "=== torch eval with prediction dump ==="
"$PY" eval_quality_inference.py --device cuda --batch 64
echo "=== ONNX cpu preload with prediction dump ==="
"$PY" eval_onnx_accuracy.py --onnx-dir "$EXP" --provider cpu --preload-weights
echo "=== agreement + near-tie margins ==="
"$PY" ia1_agreement.py \
    --torch-preds "$OUT/preds_torch.json" \
    --onnx-preds "$OUT/preds_onnx_cpu_preload.json" \
    --checkpoint "$OUT/T1/checkpoints/best.pt" --device cuda
echo "Done: $(date)"

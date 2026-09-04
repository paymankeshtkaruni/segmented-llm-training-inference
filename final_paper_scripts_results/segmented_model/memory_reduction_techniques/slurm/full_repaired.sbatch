#!/bin/bash
#SBATCH --job-name=full_repaired
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/full_repaired_%j.out
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
BASE="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
MRT="$BASE/final_paper_scripts_results/segmented_model/memory_reduction_techniques"
cd "$MRT"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
OUT="$MRT/results/quality_fresh_pair"

echo "=== 1/2 retrain full model (identical protocol), save checkpoint ==="
"$PY" "$BASE/final_paper_scripts_results/segmented_model/scripts/seg_full_train_b1.py" \
    --preset small_8x2x2x8 --device cuda --epochs 10 --batch-size 64 --seed 42 --warmup 200 \
    --save-ckpt "$OUT/full_model_10ep.pt" \
    2>&1 | tee "$OUT/full_model_10ep_rerun.log"

echo "=== 2/2 repaired equal-length eval ==="
"$PY" eval_full_model_repaired.py --checkpoint "$OUT/full_model_10ep.pt" --device cuda
echo "Done: $(date)"

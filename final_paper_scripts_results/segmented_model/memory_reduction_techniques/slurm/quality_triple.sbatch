#!/bin/bash
#SBATCH --job-name=quality_triple
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/quality_triple_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/quality_triple_%j.err
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
mkdir -p "$OUT"
echo "Node: $(hostname)  Start: $(date)"
# Matched-pair quality experiment, all in the CURRENT environment, identical
# protocol: 10 epochs (the archived run's true count), batch 64, seed 42,
# warmup 200, cosine, dropout 0.1.

echo "=== 1/3 FULL model (reference), 10 epochs + full-test greedy eval ==="
"$PY" "$BASE/final_paper_scripts_results/segmented_model/scripts/seg_full_train_b1.py" \
    --preset small_8x2x2x8 --device cuda --epochs 10 --batch-size 64 --seed 42 --warmup 200 \
    2>&1 | tee "$OUT/full_model_10ep.log"

echo "=== 2/3 segmented T1 (resident, no recompute), 10 epochs ==="
"$PY" segmentation_management/trainer.py \
    --preset small_8x2x2x8 --device cuda \
    --tech-code 11100111000 --full-run \
    --epochs 10 --batch-size 64 --warmup 200 \
    --out-dir "$OUT/T1" 2>&1 | tee "$OUT/segmented_T1_10ep.log"

echo "=== 3/3 segmented T2 (resident + recompute), 10 epochs ==="
"$PY" segmentation_management/trainer.py \
    --preset small_8x2x2x8 --device cuda \
    --tech-code 11110111000 --full-run \
    --epochs 10 --batch-size 64 --warmup 200 \
    --out-dir "$OUT/T2" 2>&1 | tee "$OUT/segmented_T2_10ep.log"
echo "Done: $(date)"

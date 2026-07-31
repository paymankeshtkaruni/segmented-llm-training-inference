#!/bin/bash
#SBATCH --job-name=seg_train_pathb
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=final_paper_scripts_results/segmented_model/outputs/logs/seg_train_pathb_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/outputs/logs/seg_train_pathb_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
SEG=final_paper_scripts_results/segmented_model
OUT=$SEG/outputs/seg_train_gpu_pathb
echo "=== Path B segmented A1 retrain (dropout 0.1) ===  Node: $(hostname)  Start: $(date)"
"$PY" $SEG/scripts/seg_train_gpu.py --device cuda --epochs 5 --batch-size 64 --seed 42 \
    --store-kind cpu_ram --log-every 50 --out-dir "$OUT"
echo "=== B1 full-test on the retrained checkpoint ===  $(date)"
"$PY" $SEG/scripts/seg_infer_gpu.py --checkpoint "$OUT/checkpoints/best.pt" --device cuda \
    --num-test -1 --batch-size 64 --max-new-tokens 40 --out-dir "$SEG/outputs/seg_b1_pathb"
echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=mrt_ds_baseline
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_ds_baseline_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/mrt_ds_baseline_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export PYTHONUNBUFFERED=1
# separate venv: torch 2.7.0+cu128 (same as the paper's runs) + deepspeed; the
# project's frozen poetry env is untouched. torch.optim.AdamW client optimizer ->
# no JIT-compiled DeepSpeed ops needed.
# Keep the separate DeepSpeed virtualenv as the default, but let the caller
# override it with PYTHON=<interpreter>, as the header note above promises.
PYTHON="${PYTHON:-${PROJECT_PARENT}/dsenv/bin/python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
# DeepSpeed ZeRO-Offload baselines on the SAME 0.84B model + protocol (fp32, bs4, seq512)
for MODE in zero2_offload zero3_offload; do
  echo "===== $MODE  $(date) ====="
  "$PYTHON" "$MRT/deepspeed_cost.py" --mode "$MODE" --batch 4 --n-steps 3
done
echo "Done: $(date)"

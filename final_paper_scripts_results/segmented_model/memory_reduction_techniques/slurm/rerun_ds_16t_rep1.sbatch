#!/bin/bash
#SBATCH --job-name=rerun_ds_rep1
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# (site note: pin to one homogeneous node pool at your site;
#  the original runs excluded non-matching nodes here)
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_ds_rep1_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_ds_rep1_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
# Audit SEV-1 #3: ZeRO-Offload runs its optimizer on CPU threads; the original
# baseline (jobs 14841235/14841281) ran 8 CPUs with OMP unset (oversubscribed).
# This rerun pins 16 threads, matching the paper's segmented CPU protocol, and
# pins the A100-40 node class the eager baseline used.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)  OMP_NUM_THREADS=$OMP_NUM_THREADS"
for MODE in zero2_offload zero3_offload; do
  echo "===== $MODE  $(date) ====="
  "$PYTHON" "$MRT/deepspeed_cost.py" --mode "$MODE" --batch 4 --n-steps 3 \
      --out-dir "$MRT/results/deepspeed_baseline_16t_rep1"
done
echo "Done: $(date)"

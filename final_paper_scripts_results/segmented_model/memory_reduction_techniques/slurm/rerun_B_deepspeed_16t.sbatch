#!/bin/bash
#SBATCH --job-name=rerun_ds_16t
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_ds_16t_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_ds_16t_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
# Thread-pinning consistency: ZeRO-Offload runs its optimizer on CPU threads; the original
# baseline ran 8 CPUs with OMP unset (oversubscribed).
# This rerun pins 16 threads, matching the paper's segmented CPU protocol, and
# pins the A100-40 node class the eager baseline used.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
# DeepSpeed runs in a separate virtualenv (deepspeed==0.19.2 on the same
# torch 2.7.0+cu128 build); point PYTHON at that interpreter for this job.
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)  OMP_NUM_THREADS=$OMP_NUM_THREADS"
for MODE in zero2_offload zero3_offload; do
  echo "===== $MODE  $(date) ====="
  "$PYTHON" "$MRT/deepspeed_cost.py" --mode "$MODE" --batch 4 --n-steps 3 \
      --out-dir "$MRT/results/deepspeed_baseline_16t"
done
echo "Done: $(date)"

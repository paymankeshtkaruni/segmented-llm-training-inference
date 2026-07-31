#!/bin/bash
#SBATCH --job-name=rerun_ds_rep2
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
# NOTE: on this cluster --constraint=40gb_vram means ">= 40 GB VRAM", so the
# 80 GB A100 nodes satisfy it too. Selecting the true A100-40 class therefore also
# needs an explicit exclusion of the 80 GB-class nodes; re-add a site-specific
#   #SBATCH --exclude=<comma-separated names of the 80 GB-class nodes>
# directive here. Independently of node selection, every run records the device
# capacity in its metrics JSON (40442 MB = the 40 GB class), so the node class of
# each committed result is verifiable after the fact.
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_ds_rep2_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_ds_rep2_%j.err
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
      --out-dir "$MRT/results/deepspeed_baseline_16t_rep2"
done
echo "Done: $(date)"

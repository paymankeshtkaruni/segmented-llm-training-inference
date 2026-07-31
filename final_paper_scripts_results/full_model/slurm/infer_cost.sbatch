#!/bin/bash
#SBATCH --job-name=fm_infer_cost
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:00:00
#SBATCH --output=final_paper_scripts_results/full_model/outputs/logs/infer_cost_%j.out
#SBATCH --error=final_paper_scripts_results/full_model/outputs/logs/infer_cost_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
S=final_paper_scripts_results/full_model/scripts
echo "=== B2 inference cost (LARGE) ===  Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "--- GPU profiler ---"; "$PYTHON" $S/full_model_infer_cost_profiler_gpu.py
echo "--- GPU light ---";    "$PYTHON" $S/full_model_infer_cost_gpu.py
echo "--- CPU profiler ---"; "$PYTHON" $S/full_model_infer_cost_profiler_cpu.py
echo "--- CPU light ---";    "$PYTHON" $S/full_model_infer_cost_cpu.py
echo "Done: $(date)"

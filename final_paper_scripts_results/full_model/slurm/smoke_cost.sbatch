#!/bin/bash
#SBATCH --job-name=fm_smoke_cost
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:20:00
#SBATCH --output=final_paper_scripts_results/full_model/outputs/logs/smoke_cost_%j.out
#SBATCH --error=final_paper_scripts_results/full_model/outputs/logs/smoke_cost_%j.err
# Site-independent: export PROJECT_ROOT=<path to this repository> and, if the
# interpreter is not on PATH, PYTHON=<interpreter> before submitting. Submit from
# the repository root so the relative --output/--error log paths resolve.

# A2 cost-training SMOKE: runs all 4 variants (profiler+light, GPU+CPU) so the
# GPU VRAM sampler (nvidia-smi + torch categories) and host-RAM sampler are
# actually exercised on a node. Verifies the path before real cost runs.

set -euo pipefail

REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
cd "$REPO"
mkdir -p final_paper_scripts_results/full_model/outputs/logs

module load gcc/13.2.0 cuda/12.8.2
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="${PYTHON:-python}"
S=final_paper_scripts_results/full_model/scripts
O=final_paper_scripts_results/full_model/outputs

echo "=== fm smoke cost ===  Job: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"

echo "--- GPU profiler ---"
"$PYTHON" $S/full_model_cost_profiler_gpu.py --n-steps 3 --out-dir $O/smoke_cost_profiler_gpu
echo "--- GPU light ---"
"$PYTHON" $S/full_model_training_cost_gpu.py --n-steps 3 --out-dir $O/smoke_training_cost_gpu
echo "--- CPU profiler ---"
"$PYTHON" $S/full_model_cost_profiler_cpu.py --n-steps 3 --out-dir $O/smoke_cost_profiler_cpu
echo "--- CPU light ---"
"$PYTHON" $S/full_model_training_cost_cpu.py --n-steps 3 --out-dir $O/smoke_training_cost_cpu

echo "Done: $(date)"

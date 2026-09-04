#!/bin/bash
#SBATCH --job-name=rerun_gpu_infer_a40
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=40gb_vram
#SBATCH --time=01:30:00
#SBATCH --output=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_gpu_infer_a40_%j.out
#SBATCH --error=final_paper_scripts_results/segmented_model/memory_reduction_techniques/results/logs/rerun_gpu_infer_a40_%j.err
# Site-independent: export PROJECT_ROOT=<repo root> and, if needed,
# PYTHON=<interpreter> before submitting from the repository root.
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo root}"
module load gcc/13.2.0 cuda/12.8.2
# Audit SEV-2 #5: the original GPU segmented-inference ladder (job 14640777)
# landed on an A100-80 (ggpu144) while all its comparison partners are A100-40.
# Same 8-CPU/unset-OMP env as the whole GPU job family; only the node class is
# pinned via --constraint=40gb_vram.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MRT=final_paper_scripts_results/segmented_model/memory_reduction_techniques
echo "Node: $(hostname)  Start: $(date)"
"$PYTHON" "$MRT/gpu_inference_cost.py" \
    --preset large_8x2x2x8 --device cuda --prompt-len 256 --gen-tokens 8 \
    --out-dir "$MRT/results/gpu_inference_a100_40"
echo "Done: $(date)"

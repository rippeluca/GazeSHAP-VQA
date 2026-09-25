#!/bin/bash
# Stage 1 (SHAP-gaze alignment) on a single A100 40GB.
#
# Fill in the site-specific values below before the first run:
#   --partition, --account, the module names in slurm/setup_env.sh
#
#SBATCH --job-name=gazeshap-stage1
#SBATCH --partition=CHANGEME
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

source slurm/setup_env.sh

# W&B stays offline on the compute node; sync afterwards from the login node
# with `wandb sync wandb/offline-run-*`.
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

srun .venv/bin/python scripts/train_stage1.py \
  --config configs/stage1_shap_gaze.yaml \
  "$@"

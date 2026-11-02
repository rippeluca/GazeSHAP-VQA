#!/bin/bash
# Stage 1, second attempt: attribute the sum of the teacher-forced answer logits.
# Same memory settings as the run that fit on this cluster (batch 1, 2 baselines,
# 64-128 visual tokens).
#
# Submit from the repo root on a login node. PY must be an interpreter whose
# torch build can see the GPU. WANDB_ENTITY is optional.
#   PY=/path/to/python sbatch --export=ALL slurm/stage1_answer.sbatch
#
#SBATCH --job-name=gazeshap-s1ans
#SBATCH --partition=all_gpu.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage1_answer_%j.out
#SBATCH --error=logs/stage1_answer_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

if [ -z "${PY:-}" ]; then
  echo "Set PY to a Python whose torch build can see the GPU." >&2
  exit 1
fi

echo "host=$(hostname) python=$PY"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$PY" scripts/train_stage1.py --config configs/stage1_shap_gaze.yaml \
  run_id=stage1_answer \
  training.per_device_train_batch_size=1 \
  shap.n_baselines=2 \
  model.min_visual_tokens=64 \
  model.max_visual_tokens=128 \
  logging.wandb_project=GazeSHAP-VQA \
  logging.wandb_mode=online

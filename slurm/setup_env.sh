#!/bin/bash
# Environment bootstrap for the compute nodes. Sourced by the sbatch scripts and
# safe to source manually on a login node.
#
# Site-specific: adjust the module names to whatever the cluster provides.

set -euo pipefail

# module load cuda/12.4
# module load gcc/12

if [ ! -d .venv ]; then
  echo "Creating .venv (Python 3.11, CUDA 12.x wheels)"
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -e ".[dev,train,eval]"
fi

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p "$HF_HOME"

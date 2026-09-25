# CPU image. Builds on Apple Silicon and on amd64, and is what CI and local
# `docker compose run test` use. Training that needs a GPU uses Dockerfile.cuda.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/app/.cache/huggingface \
    WANDB_MODE=offline \
    MPLCONFIGDIR=/tmp/mpl \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies before copying the rest of the tree, so editing a script
# does not re-download torch.
COPY pyproject.toml README.md ./
COPY src ./src

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG EXTRAS=dev
RUN uv pip install --system --extra-index-url "${TORCH_INDEX_URL}" ".[${EXTRAS}]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY slurm ./slurm
COPY AGENTS.md ./AGENTS.md
COPY data/README.md ./data/README.md
COPY data/processed/splits ./data/processed/splits

CMD ["pytest", "tests/", "-q", "--device", "cpu"]

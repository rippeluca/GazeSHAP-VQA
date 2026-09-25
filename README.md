# GazeSHAP-VQA

Gaze-supervised SHAP training for Visual Question Answering with explicit
reasoning chains.

A VQA model is trained to be *right for the right reasons*: its SHAP attribution
over the image is aligned with the human fixation heatmap recorded while people
answered the **same question** (AiR-D eye-tracking data), before a reasoning
chain and an answer are generated on top.

The full specification lives in [idea.md](idea.md); the condensed rules for
day-to-day work are in [AGENTS.md](AGENTS.md).

## Status

Milestone 1 (Stage 1) is implemented and verified end to end on
`Qwen/Qwen3-VL-2B-Instruct` with the real AiR-D data: data pipeline,
`VLMWrapper` with the eager-attention guard, differentiable expected gradients,
alignment metrics, and the SHAP-gaze alignment trainer. GRPO, the faithfulness
suite and the dashboard are not built yet.

Reference points on the test split (272 questions): a uniform map scores
AUC-Judd 0.500, the centre-prior control 0.724, and human inter-subject
agreement 0.895. The interesting band is narrow and every result must be read
against 0.724, not 0.5.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

On a CUDA node, add the training and evaluation extras:

```bash
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -e ".[dev,train,eval]"
```

## Pipeline

```bash
python scripts/download_data.py --data-root data      # AiR-D + GQA + VG images
python scripts/preprocess_gaze.py --validate          # Path B vs Path A, CC ~0.96
python scripts/preprocess_gaze.py --data-root data    # fixation maps -> 16x16
python scripts/make_splits.py --data-root data        # image-level 70/10/20
python scripts/check_double_backward.py               # once per machine
python scripts/train_stage1.py --config configs/stage1_shap_gaze.yaml
```

On the cluster: `sbatch slurm/stage1.sbatch`. Fill in the partition and module
names at the top of `slurm/stage1.sbatch` and `slurm/setup_env.sh` first.

The download fetches roughly 500 MB: AiR-D (154 MB), the GQA balanced validation
questions (114 MB, extracted from a 1.5 GB zip over HTTP range requests) and 987
Visual Genome images (143 MB, instead of GQA's 21.8 GB image archive).

## Tests

```bash
pytest tests/ -v --device cpu
```

The suite runs on CPU with synthetic fixtures and needs no downloaded data. The
test that matters most is the gradient-flow assertion in
`tests/test_shap_module.py`: it fails if the attribution stops being
differentiable with respect to the LoRA parameters.

## Data and licensing

Images come from Visual Genome (CC BY 4.0) via GQA. See
[data/README.md](data/README.md) for provenance and attribution requirements.

- AiR-D: Chen, Jiang, Yang & Zhao, *AiR: Attention with Reasoning Capability*, ECCV 2020
- GQA: Hudson & Manning, CVPR 2019
- Visual Genome: Krishna et al., IJCV 2017

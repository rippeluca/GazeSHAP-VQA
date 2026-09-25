# AGENTS.md — GazeSHAP-VQA

Condensed working rules. The full specification is [idea.md](idea.md); when the
two disagree, `idea.md` wins and this file must be updated.

## What the project does

Train Qwen3-VL-2B so that its SHAP attribution over the image matches the human
gaze heatmap recorded while answering the *same* question (AiR-D), then let GRPO
produce a `<think>`/`<answer>` chain on top. Stage 1 (gaze alignment) is the part
currently implemented.

## Hard rules

1. **Never import `captum` or `shap` in the training path.** `captum` appears in
   `tests/` only; `shap` is evaluation-only. Captum calls `backward()` without
   `create_graph=True`, so no gradient reaches the LoRA weights and the loss
   silently decreases while nothing is learned.
2. **Stage 1 requires `attn_implementation="eager"` and forbids gradient
   checkpointing.** Fused SDPA/FlashAttention kernels have no double-backward
   rule. `VLMWrapper` asserts this at construction.
3. **`pixel_values` is not an image tensor.** The Qwen processor emits
   `(grid_h*grid_w, C*temporal_patch*patch*patch)` with merge-window interleaved
   patch order. Use `src/models/patch_grid.py` to map between the flat layout and
   a spatial `(grid_h, grid_w)` grid; never `view(gh, gw)` directly.
4. **Qwen3-VL geometry:** `patch_size=16`, `spatial_merge_size=2`, so one visual
   token is 32x32 px. Pixel budgets are expressed as `n_tokens * 32 * 32`.
5. **Attribution target** is the logit of the answer's first token at the
   position right after the teacher-forced `<answer>` tag.
6. **Discard the first second of gaze** (`TEMPORAL_BINS = ("1s", "2s")`); the
   0-1s bin is dominated by the forced central starting fixation.
7. **Split by image, never by question.** 299 AiR-D images carry more than one
   question.
8. **LoRA only.** 1,422 questions is too small for full fine-tuning; report at
   least 3 seeds for any number that enters a conclusion.

## Verified on the real backbone

`scripts/check_double_backward.py` was run against `Qwen/Qwen3-VL-2B-Instruct`
with eager attention: the double backward completes and gradients land on the
LoRA weights. Expect **half** of the trainable LoRA tensors to report a zero
gradient on the first step — `peft` initialises `lora_B` to zeros, so
`dL/d(lora_A)` is exactly zero until `lora_B` moves. Only an all-zero result is a
failure.

## Conventions

- `from __future__ import annotations` in every module; type hints everywhere.
- Google-style docstrings with Args/Returns/Raises on public functions.
- No module-level singletons; pass the `Config` object explicitly.
- Never hardcode `"cuda"`; use `next(model.parameters()).device`.
- `ruff check --fix` and `ruff format` before every commit.

## Commands

```bash
uv pip install -e ".[dev]"          # local, CPU
pytest tests/ -v --device cpu       # full suite, no GPU needed
ruff check . && ruff format --check .
python scripts/download_data.py --data-root data
python scripts/preprocess_gaze.py --data-root data
python scripts/train_stage1.py --config configs/stage1_shap_gaze.yaml
sbatch slurm/stage1.sbatch          # cluster
```

## Reference numbers, measured

On our test split (272 questions, 16x16 grids, bins 1s+2s):

- uniform map: AUC-Judd 0.500
- `C1` centre prior (sigma 15): AUC-Judd 0.724, CC 0.136, NSS 0.352
- human ceiling (AiR-D inter-subject): AUC-Judd 0.895

The band that matters is 0.724 to 0.895. Read every gain against 0.724.

Known calibration issue: with P = 16, `loss_align` lands around 1e-4 while
`lambda_kl * loss_kl` is around 0.15, so Stage 1 is effectively KL-driven under
the default weights. Decide this deliberately before the first full run.

## Current state

Implemented: config system, data download, gaze preprocessing, splits, dataset,
`VLMWrapper`, `DifferentiableSHAPModule`, alignment metrics, Stage 1 losses and
trainer. Verified end to end on `Qwen/Qwen3-VL-2B-Instruct` with real AiR-D data.

Not implemented (v1 backlog, in this order): GRPO rewards and `train_stage2.py`,
faithfulness metrics, `results/` artefacts, dashboard.

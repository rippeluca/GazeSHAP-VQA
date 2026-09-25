#!/usr/bin/env python
"""Stage 1: align the model's SHAP attribution with the human fixation map.

Example:
    python scripts/train_stage1.py --config configs/stage1_shap_gaze.yaml
    python scripts/train_stage1.py --config configs/stage1_shap_gaze.yaml training.lr=1e-4
    sbatch slurm/stage1.sbatch
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import collate_gaze_vqa  # noqa: E402
from src.data.dataset import GazeVQADataset  # noqa: E402
from src.data.splits import read_splits  # noqa: E402
from src.models.shap_module import DifferentiableSHAPModule  # noqa: E402
from src.models.vlm_wrapper import VLMWrapper  # noqa: E402
from src.training.stage1_trainer import Stage1Trainer  # noqa: E402
from src.utils.config import Config, load_config  # noqa: E402
from src.utils.logging_utils import RunLogger, git_commit_hash, setup_logging  # noqa: E402

log = logging.getLogger("train_stage1")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        Parsed arguments, with dotted config overrides in ``overrides``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/stage1_shap_gaze.yaml"))
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. training.lr=1e-4")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    """Pick the device to run on.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.

    Returns:
        The resolved device.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, numpy and torch.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(config: Config, processor: Any) -> tuple[DataLoader, DataLoader | None]:
    """Build the train and validation loaders from the committed splits.

    Args:
        config: Run configuration.
        processor: Qwen3-VL processor.

    Returns:
        The training loader and, when the validation split exists, its loader.
    """
    splits = read_splits(config.data.processed_dir / "splits")
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
    collate = partial(collate_gaze_vqa, pad_token_id=pad_token_id)

    def make(qids: list[str], limit: int | None, shuffle: bool) -> DataLoader:
        dataset = GazeVQADataset(
            config.data.root,
            processor,
            qids=qids[:limit] if limit else qids,
            patch_grid_size=config.data.patch_grid_size,
            temporal_bins=config.data.temporal_bins,
            use_correct_only=config.data.use_correct_only,
            per_subject_augment=config.data.per_subject_augment,
        )
        return DataLoader(
            dataset,
            batch_size=config.training.per_device_train_batch_size,
            shuffle=shuffle,
            num_workers=config.data.num_workers,
            collate_fn=collate,
        )

    train = make(splits["train"], config.training.max_train_samples, shuffle=True)
    val = make(splits["val"], config.training.max_val_samples, shuffle=False)
    return train, val


def main() -> int:
    """Run Stage 1 training.

    Returns:
        Process exit code.
    """
    args = parse_args()
    setup_logging(logging.INFO)

    overrides = list(args.overrides)
    if args.data_root is not None:
        overrides.append(f"data.root={args.data_root}")
    config = load_config(args.config, overrides)
    config.assert_double_backward_safe()

    seed_everything(config.seed)
    device = resolve_device(config.device)
    commit = git_commit_hash()
    log.info("run_id=%s device=%s commit=%s", config.run_id, device, commit)

    vlm = VLMWrapper.from_config(config.model, device=device)
    shap = DifferentiableSHAPModule(
        vlm,
        patch_grid_size=config.data.patch_grid_size,
        n_baselines=config.shap.n_baselines,
        baseline=config.shap.baseline,
        attr_agg=config.shap.attr_agg,
    )
    train_loader, val_loader = build_loaders(config, vlm.processor)
    log.info("train batches=%d val batches=%d", len(train_loader), len(val_loader or []))

    run_logger = RunLogger(
        run_id=config.run_id,
        output_dir=Path(config.logging.checkpoint_dir) / config.run_id,
        wandb_project=config.logging.wandb_project,
        wandb_mode=config.logging.wandb_mode,
        config=config.model_dump(mode="json") | {"git_commit": commit},
    )
    try:
        trainer = Stage1Trainer(config, vlm, shap, train_loader, val_loader, run_logger)
        state = trainer.train()
    finally:
        run_logger.close()

    log.info("done after %d steps, best val KL %.4f", state.step, state.best_val_kl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end CPU smoke test for Stage 1.

Runs the real trainer, the real attribution and the real loss against a tiny
model and a synthetic dataset, so CI catches a broken training loop without a
GPU or a hub download.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from src.data.collate import collate_gaze_vqa
from src.data.dataset import GazeVQADataset
from src.models.shap_module import DifferentiableSHAPModule
from src.models.vlm_wrapper import VLMWrapper
from src.training.stage1_trainer import Stage1Trainer
from src.utils.config import load_config
from tests.fixtures import FakeProcessor, make_fake_dataset_root, tiny_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIGS / "stage1_smoke.yaml")
    cfg.data.root = tmp_path
    cfg.logging.checkpoint_dir = tmp_path / "checkpoints"
    cfg.logging.wandb_mode = "disabled"
    return cfg


@pytest.fixture
def wrapper():
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return VLMWrapper(get_peft_model(tiny_model(), lora), FakeProcessor())


@pytest.fixture
def loaders(config, tmp_path):
    make_fake_dataset_root(tmp_path, n_images=2, questions_per_image=1)
    dataset = GazeVQADataset(tmp_path, FakeProcessor(), patch_grid_size=config.data.patch_grid_size)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=partial(collate_gaze_vqa, pad_token_id=0)
    )
    return loader, loader


@pytest.fixture
def trainer(config, wrapper, loaders):
    shap = DifferentiableSHAPModule(
        wrapper, patch_grid_size=config.data.patch_grid_size, n_baselines=config.shap.n_baselines
    )
    return Stage1Trainer(config, wrapper, shap, loaders[0], loaders[1])


def test_smoke_run_completes(trainer):
    state = trainer.train()
    assert state.step >= 1
    assert state.epoch == 0


def test_lora_weights_actually_move(trainer, wrapper):
    """The point of the whole exercise: the adapter must change."""
    before = {
        name: param.detach().clone()
        for name, param in wrapper.model.named_parameters()
        if "lora_B" in name and param.requires_grad
    }
    assert before

    trainer.train()

    moved = [
        name
        for name, param in wrapper.model.named_parameters()
        if name in before and not torch.allclose(before[name], param.detach())
    ]
    assert moved, "no LoRA weight changed during training"


def test_validation_reports_finite_metrics(trainer):
    result = trainer.evaluate()
    assert result.n_samples == 2
    assert 0.0 <= result.auc_judd <= 1.0
    assert result.kl >= 0.0
    assert -1.0 <= result.cc <= 1.0
    assert set(result.as_dict()) == {
        "val/kl_gaze_shap",
        "val/auc_judd",
        "val/cc",
        "val/loss",
    }


def test_checkpoint_is_written(trainer, config):
    trainer.train()
    best = config.logging.checkpoint_dir / config.run_id / "stage1_best"
    last = config.logging.checkpoint_dir / config.run_id / "stage1_last"
    assert (best / "state.pt").exists()
    assert (last / "state.pt").exists()

    state = torch.load(best / "state.pt", weights_only=False)
    assert set(state) == {"step", "epoch", "best_val_kl", "optimizer", "scheduler", "trainable"}
    assert state["trainable"], "checkpoint holds no trainable parameters"


def test_best_checkpoint_tracks_val_kl(trainer):
    state = trainer.train()
    assert state.best_val_kl < float("inf")
    assert state.history, "no validation history recorded"


def test_trainer_refuses_unsafe_config(config, wrapper, loaders):
    config.model.attn_implementation = "sdpa"
    shap = DifferentiableSHAPModule(wrapper, patch_grid_size=config.data.patch_grid_size)
    with pytest.raises(ValueError, match="eager"):
        Stage1Trainer(config, wrapper, shap, loaders[0], loaders[1])


def test_trainer_refuses_a_model_without_trainable_parameters(config, loaders):
    frozen = tiny_model()
    for param in frozen.parameters():
        param.requires_grad_(False)
    wrapper = VLMWrapper(frozen, FakeProcessor())
    shap = DifferentiableSHAPModule(wrapper, patch_grid_size=config.data.patch_grid_size)
    with pytest.raises(ValueError, match="no trainable parameters"):
        Stage1Trainer(config, wrapper, shap, loaders[0], loaders[1])


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)`")
def test_scheduler_warms_up_then_decays(config, wrapper, loaders):
    config.training.num_epochs = 20  # 2 batches per epoch -> 40 scheduled steps
    config.training.warmup_steps = 5
    shap = DifferentiableSHAPModule(wrapper, patch_grid_size=config.data.patch_grid_size)
    trainer = Stage1Trainer(config, wrapper, shap, loaders[0], loaders[1])

    rates = []
    for _ in range(40):
        rates.append(trainer.scheduler.get_last_lr()[0])
        trainer.scheduler.step()

    assert rates[0] == pytest.approx(0.0)
    assert rates[5] == pytest.approx(config.training.lr)  # peak at the end of warmup
    assert rates[2] < rates[5]  # warming up
    assert rates[-1] < rates[5]  # decaying afterwards
    decay = rates[5:]
    assert all(b <= a + 1e-12 for a, b in zip(decay, decay[1:], strict=False))

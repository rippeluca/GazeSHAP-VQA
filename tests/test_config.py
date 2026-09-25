from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_base_config_loads():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.seed == 42
    assert cfg.model.attn_implementation == "eager"
    assert cfg.data.patch_grid_size == 16


def test_stage1_inherits_from_base():
    cfg = load_config(CONFIGS / "stage1_shap_gaze.yaml")
    # Own value.
    assert cfg.run_id == "stage1"
    assert cfg.training.lr == pytest.approx(2e-4)
    # Inherited value.
    assert cfg.model.name == "Qwen/Qwen3-VL-2B-Instruct"
    assert cfg.data.temporal_bins == ["1s", "2s"]


def test_transitive_inheritance():
    cfg = load_config(CONFIGS / "stage1_smoke.yaml")
    assert cfg.device == "cpu"
    assert cfg.shap.n_baselines == 2
    assert cfg.data.patch_grid_size == 4
    # Two levels up, untouched by either child.
    assert cfg.model.lora.r == 16
    # One level up, untouched by the child.
    assert cfg.training.lambda_kl == pytest.approx(0.1)


def test_dotted_overrides():
    cfg = load_config(
        CONFIGS / "stage1_shap_gaze.yaml", ["training.lr=1e-5", "data.patch_grid_size=32"]
    )
    assert cfg.training.lr == pytest.approx(1e-5)
    assert cfg.data.patch_grid_size == 32


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        load_config(CONFIGS / "base.yaml", ["training.nonexistent=1"])


def test_double_backward_guard():
    cfg = Config()
    cfg.assert_double_backward_safe()

    cfg_sdpa = Config.model_validate({"model": {"attn_implementation": "sdpa"}})
    with pytest.raises(ValueError, match="eager"):
        cfg_sdpa.assert_double_backward_safe()

    cfg_ckpt = Config.model_validate({"model": {"gradient_checkpointing": True}})
    with pytest.raises(ValueError, match="checkpointing"):
        cfg_ckpt.assert_double_backward_safe()


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"split_fractions": [0.8, 0.1, 0.2]}})


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config(CONFIGS / "does_not_exist.yaml")

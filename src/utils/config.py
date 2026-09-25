"""Typed configuration loading for GazeSHAP-VQA.

Configs are YAML files under ``configs/``. A config may inherit from another via
the ``_base_`` key; inheritance is resolved recursively and the child overrides
the parent. CLI overrides use dotted keys (``training.lr=1e-4``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_KEY = "_base_"


class LoRAConfig(BaseModel):
    """LoRA adapter hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class ModelConfig(BaseModel):
    """Backbone and adapter configuration."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    gradient_checkpointing: bool = False
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    min_visual_tokens: int = 256
    max_visual_tokens: int = 512
    freeze_vision: bool = True
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    @field_validator("max_visual_tokens")
    @classmethod
    def _max_above_min(cls, v: int, info: Any) -> int:
        min_tokens = info.data.get("min_visual_tokens")
        if min_tokens is not None and v < min_tokens:
            raise ValueError(f"max_visual_tokens ({v}) < min_visual_tokens ({min_tokens})")
        return v


class DataConfig(BaseModel):
    """Dataset locations and gaze preprocessing parameters."""

    model_config = ConfigDict(extra="forbid")

    root: Path = Path("data")
    patch_grid_size: int = 16
    map_size: int = 256
    gaze_sigma_px: float = 9.0
    temporal_bins: list[str] = Field(default_factory=lambda: ["1s", "2s"])
    use_correct_only: bool = False
    per_subject_augment: bool = False
    answer_format: Literal["open", "mc"] = "open"
    split_fractions: tuple[float, float, float] = (0.7, 0.1, 0.2)
    num_workers: int = 0

    @field_validator("split_fractions")
    @classmethod
    def _sums_to_one(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"split_fractions must sum to 1.0, got {sum(v)}")
        return v

    @property
    def raw_dir(self) -> Path:
        """Directory holding the downloaded, unmodified source data."""
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory holding derived artefacts (fixation maps, splits)."""
        return self.root / "processed"


class SHAPConfig(BaseModel):
    """Expected-gradients attribution parameters."""

    model_config = ConfigDict(extra="forbid")

    n_baselines: int = 5
    baseline: Literal["black", "gaussian"] = "black"
    attr_agg: Literal["sum", "abs"] = "sum"


class TrainingConfig(BaseModel):
    """Stage 1 optimisation settings."""

    model_config = ConfigDict(extra="forbid")

    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    num_epochs: int = 5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lambda_kl: float = 0.1
    alpha_wmse: float = 1.1
    max_grad_norm: float = 1.0
    attribution_fp32: bool = False
    eval_every_epoch: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = None


class LoggingConfig(BaseModel):
    """Experiment tracking and checkpointing."""

    model_config = ConfigDict(extra="forbid")

    wandb_project: str = "gazeshap-vqa"
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    log_every: int = 10
    checkpoint_dir: Path = Path("checkpoints")


class Config(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    run_id: str = "dev"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    shap: SHAPConfig = Field(default_factory=SHAPConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def assert_double_backward_safe(self) -> None:
        """Fail if the configuration cannot support second-order gradients.

        Raises:
            ValueError: If attention is not eager or gradient checkpointing is on.
        """
        if self.model.attn_implementation != "eager":
            raise ValueError(
                "Stage 1 needs attn_implementation='eager'; fused SDPA/FlashAttention "
                f"kernels have no double-backward rule (got {self.model.attn_implementation!r})."
            )
        if self.model.gradient_checkpointing:
            raise ValueError(
                "Stage 1 cannot use gradient checkpointing: recomputation breaks the "
                "second-order graph."
            )


def _resolve_bases(path: Path, seen: set[Path] | None = None) -> DictConfig:
    """Load a YAML config and recursively merge its ``_base_`` ancestors.

    Args:
        path: Path to the YAML file.
        seen: Paths already visited, used to detect inheritance cycles.

    Returns:
        The merged OmegaConf node, with ``_base_`` stripped.

    Raises:
        FileNotFoundError: If the config or one of its bases does not exist.
        ValueError: If the inheritance chain contains a cycle.
    """
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Cyclic `{BASE_KEY}` inheritance involving {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    seen = seen | {path}

    node = OmegaConf.load(path)
    base_ref = node.pop(BASE_KEY, None)
    if base_ref is None:
        return node

    base_paths = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    merged = OmegaConf.create({})
    for ref in base_paths:
        merged = OmegaConf.merge(merged, _resolve_bases(path.parent / ref, seen))
    return OmegaConf.merge(merged, node)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load, merge and validate a configuration.

    Args:
        path: Path to the stage config YAML.
        overrides: Dotted-key overrides, e.g. ``["training.lr=1e-4"]``.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the merged config violates the schema.
    """
    node = _resolve_bases(Path(path))
    if overrides:
        node = OmegaConf.merge(node, OmegaConf.from_dotlist(list(overrides)))
    raw = OmegaConf.to_container(node, resolve=True)
    return Config.model_validate(raw)

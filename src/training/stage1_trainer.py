"""Stage 1: supervised alignment of the SHAP map with the human fixation map.

A plain torch loop rather than ``transformers.Trainer``: the training step needs
a second-order backward through a hand-rolled attribution, and the Trainer's
assumptions about loss computation, autocast and checkpointing all get in the
way of that.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.evaluation.gaze_alignment import auc_judd, cc, kl_divergence
from src.models.shap_module import DifferentiableSHAPModule
from src.models.vlm_wrapper import VLMWrapper
from src.training.losses import stage1_loss
from src.utils.config import Config
from src.utils.logging_utils import RunLogger

logger = logging.getLogger(__name__)

MODEL_INPUT_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "target_token_id",
)


@dataclass
class EpochResult:
    """Metrics from one validation pass."""

    kl: float = math.inf
    auc_judd: float = 0.0
    cc: float = 0.0
    loss: float = math.inf
    n_samples: int = 0

    def as_dict(self, prefix: str = "val") -> dict[str, float]:
        """Flatten to a loggable dictionary.

        Args:
            prefix: Metric name prefix.

        Returns:
            Mapping from prefixed metric name to value.
        """
        return {
            f"{prefix}/kl_gaze_shap": self.kl,
            f"{prefix}/auc_judd": self.auc_judd,
            f"{prefix}/cc": self.cc,
            f"{prefix}/loss": self.loss,
        }


@dataclass
class TrainingState:
    """Mutable progress of a training run."""

    step: int = 0
    epoch: int = 0
    best_val_kl: float = math.inf
    history: list[dict[str, float]] = field(default_factory=list)


class Stage1Trainer:
    """Trains the LoRA adapters so the SHAP map matches human gaze."""

    def __init__(
        self,
        config: Config,
        vlm: VLMWrapper,
        shap_module: DifferentiableSHAPModule,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        """Set up the trainer.

        Args:
            config: Full run configuration.
            vlm: The wrapped backbone with LoRA attached.
            shap_module: The differentiable attribution module.
            train_loader: Training data loader.
            val_loader: Validation loader, or ``None`` to skip validation.
            run_logger: Metric sink; a no-op when ``None``.

        Raises:
            ValueError: If the configuration cannot support double backward.
        """
        config.assert_double_backward_safe()
        self.config = config
        self.vlm = vlm
        self.shap = shap_module
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_logger = run_logger
        self.state = TrainingState()

        params = vlm.trainable_parameters()
        if not params:
            raise ValueError("no trainable parameters; did attach_lora run?")
        self.optimizer = torch.optim.AdamW(
            params, lr=config.training.lr, weight_decay=config.training.weight_decay
        )
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warmup followed by cosine decay.

        Returns:
            The scheduler.
        """
        training = self.config.training
        steps_per_epoch = max(
            1, len(self.train_loader) // max(training.gradient_accumulation_steps, 1)
        )
        total = max(1, steps_per_epoch * training.num_epochs)
        warmup = min(training.warmup_steps, max(total - 1, 0))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return step / warmup
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # --- one step -----------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        """Move the tensor entries of a batch to the model device.

        Args:
            batch: Collated batch.

        Returns:
            A dictionary of the model inputs plus the gaze map, on device.
        """
        device = self.vlm.device
        moved = {k: batch[k].to(device) for k in MODEL_INPUT_KEYS if k in batch}
        moved["gaze_map"] = batch["gaze_map"].to(device)
        return moved

    def _shap_map(self, batch: dict, create_graph: bool) -> torch.Tensor:
        """Compute the SHAP map for a batch.

        Attribution is numerically delicate, so ``training.attribution_fp32``
        forces this block out of autocast (idea.md 10.8).

        Args:
            batch: On-device batch.
            create_graph: Keep the second-order graph.

        Returns:
            ``(B, P, P)`` SHAP distribution.
        """
        inputs = {k: v for k, v in batch.items() if k != "gaze_map"}
        if self.config.training.attribution_fp32:
            with torch.autocast(device_type=self.vlm.device.type, enabled=False):
                return self.shap(**inputs, create_graph=create_graph)
        return self.shap(**inputs, create_graph=create_graph)

    def train_step(self, batch: dict) -> dict[str, float]:
        """Run one forward/backward pass without stepping the optimiser.

        Args:
            batch: Collated batch, already on device.

        Returns:
            Loss components for logging.
        """
        shap_map = self._shap_map(batch, create_graph=True)
        loss, parts = stage1_loss(
            shap_map,
            batch["gaze_map"],
            alpha=self.config.training.alpha_wmse,
            lambda_kl=self.config.training.lambda_kl,
        )
        (loss / self.config.training.gradient_accumulation_steps).backward()
        return parts

    # --- loops --------------------------------------------------------------

    def train(self) -> TrainingState:
        """Run the full training schedule.

        Returns:
            The final training state.
        """
        training = self.config.training
        accumulation = max(training.gradient_accumulation_steps, 1)
        self.vlm.model.train()

        for epoch in range(training.num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            for index, raw_batch in enumerate(self.train_loader):
                batch = self._to_device(raw_batch)
                parts = self.train_step(batch)

                if (index + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.vlm.trainable_parameters(), training.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.step += 1

                    if self.state.step % self.config.logging.log_every == 0:
                        self._log(
                            {f"train/{k}": v for k, v in parts.items()}
                            | {"train/lr": self.scheduler.get_last_lr()[0]}
                        )

            logger.info("epoch %d finished in %.1fs", epoch, time.time() - epoch_start)
            if training.eval_every_epoch and self.val_loader is not None:
                result = self.evaluate()
                self._log(result.as_dict())
                self.state.history.append({"epoch": epoch, **result.as_dict()})
                if result.kl < self.state.best_val_kl:
                    self.state.best_val_kl = result.kl
                    self.save_checkpoint("stage1_best")
                    logger.info("new best val KL %.4f", result.kl)

        self.save_checkpoint("stage1_last")
        return self.state

    def evaluate(self) -> EpochResult:
        """Score the validation split.

        Gradients are still required inside the attribution, so this cannot run
        under ``torch.no_grad``; only ``create_graph`` is switched off.

        Returns:
            Averaged validation metrics.
        """
        if self.val_loader is None:
            return EpochResult()

        self.vlm.model.eval()
        kls: list[float] = []
        aucs: list[float] = []
        ccs: list[float] = []
        losses: list[float] = []

        for raw_batch in self.val_loader:
            batch = self._to_device(raw_batch)
            shap_map = self._shap_map(batch, create_graph=False).detach()
            _, parts = stage1_loss(
                shap_map,
                batch["gaze_map"],
                alpha=self.config.training.alpha_wmse,
                lambda_kl=self.config.training.lambda_kl,
            )
            losses.append(parts["loss_total"])
            for predicted, gaze in zip(shap_map, batch["gaze_map"], strict=True):
                kls.append(kl_divergence(predicted, gaze))
                aucs.append(auc_judd(predicted, gaze))
                ccs.append(cc(predicted, gaze))

        self.vlm.model.train()
        if not kls:
            return EpochResult()
        return EpochResult(
            kl=float(sum(kls) / len(kls)),
            auc_judd=float(sum(aucs) / len(aucs)),
            cc=float(sum(ccs) / len(ccs)),
            loss=float(sum(losses) / len(losses)),
            n_samples=len(kls),
        )

    # --- persistence --------------------------------------------------------

    def _log(self, metrics: dict[str, float]) -> None:
        """Send metrics to the run logger, if there is one.

        Args:
            metrics: Metrics to record.
        """
        if self.run_logger is not None:
            self.run_logger.log(metrics, step=self.state.step)
        else:
            logger.info("step=%d %s", self.state.step, metrics)

    def checkpoint_dir(self) -> Path:
        """Directory this run writes checkpoints to."""
        return Path(self.config.logging.checkpoint_dir) / self.config.run_id

    def save_checkpoint(self, name: str) -> Path:
        """Write the LoRA adapter and the run state.

        The adapter is saved separately from the state dict so it can be loaded
        onto a fresh backbone in Stage 2.

        Args:
            name: Checkpoint name, e.g. ``"stage1_best"``.

        Returns:
            The directory written.
        """
        target = self.checkpoint_dir() / name
        target.mkdir(parents=True, exist_ok=True)
        self.vlm.save_adapter(target / "adapter")
        torch.save(
            {
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_val_kl": self.state.best_val_kl,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "trainable": {
                    name_: param.detach().cpu()
                    for name_, param in self.vlm.model.named_parameters()
                    if param.requires_grad
                },
            },
            target / "state.pt",
        )
        return target

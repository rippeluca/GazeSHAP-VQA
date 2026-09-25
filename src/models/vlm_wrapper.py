"""Qwen3-VL backbone with LoRA and the guards Stage 1 depends on.

Two configuration mistakes are fatal to Stage 1 and both fail quietly, so they
are refused at construction:

* a fused attention kernel (SDPA, FlashAttention) has no double-backward rule,
  which surfaces as an opaque ``RuntimeError`` deep inside autograd;
* gradient checkpointing recomputes the forward pass and breaks the second-order
  graph, which surfaces as silently missing gradients.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.utils.config import ModelConfig

logger = logging.getLogger(__name__)

# Qwen3-VL: patch_size 16 with spatial_merge_size 2, so one visual token is
# 32x32 pixels. The Qwen2-VL family used 28x28; using that here silently
# changes the token budget.
VISUAL_TOKEN_EDGE_PX = 32

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class DoubleBackwardUnsafeError(RuntimeError):
    """Raised when the configuration cannot support second-order gradients."""


class VLMWrapper(nn.Module):
    """Wraps a Qwen3-VL model and exposes the pieces attribution needs."""

    def __init__(
        self,
        model: nn.Module,
        processor: Any | None = None,
        require_double_backward: bool = True,
    ) -> None:
        """Wrap an already constructed model.

        Use :meth:`from_config` for the normal path; this constructor exists so
        tests can inject a tiny model.

        Args:
            model: A ``Qwen3VLForConditionalGeneration`` or compatible model.
            processor: The matching processor, if available.
            require_double_backward: Enforce the Stage 1 guards.

        Raises:
            DoubleBackwardUnsafeError: If the guards are on and the model is
                configured with a fused attention kernel or checkpointing.
        """
        super().__init__()
        self.model = model
        self.processor = processor
        self.require_double_backward = require_double_backward
        if require_double_backward:
            self.assert_double_backward_safe()

    # --- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        device: torch.device | str = "cpu",
        require_double_backward: bool = True,
    ) -> VLMWrapper:
        """Load the backbone described by a :class:`ModelConfig` and attach LoRA.

        Args:
            config: Model configuration.
            device: Device to place the model on.
            require_double_backward: Enforce the Stage 1 guards.

        Returns:
            The wrapped model.

        Raises:
            DoubleBackwardUnsafeError: If the configuration is unsafe for Stage 1.
        """
        if require_double_backward and config.attn_implementation != "eager":
            raise DoubleBackwardUnsafeError(
                f"attn_implementation={config.attn_implementation!r} has no double-backward "
                "rule; Stage 1 requires 'eager'."
            )
        if require_double_backward and config.gradient_checkpointing:
            raise DoubleBackwardUnsafeError(
                "gradient checkpointing breaks the second-order graph; disable it for Stage 1."
            )

        unit = VISUAL_TOKEN_EDGE_PX * VISUAL_TOKEN_EDGE_PX
        processor = AutoProcessor.from_pretrained(
            config.name,
            min_pixels=config.min_visual_tokens * unit,
            max_pixels=config.max_visual_tokens * unit,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.name,
            dtype=DTYPES[config.dtype],
            attn_implementation=config.attn_implementation,
        )
        model.to(device)

        wrapper = cls(model, processor, require_double_backward=require_double_backward)
        wrapper.attach_lora(config)
        if config.freeze_vision:
            wrapper.freeze_vision_tower()
        return wrapper

    def attach_lora(self, config: ModelConfig) -> None:
        """Attach LoRA adapters to the attention projections.

        Args:
            config: Model configuration carrying the LoRA hyperparameters.
        """
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "LoRA attached: %d trainable of %d parameters (%.3f%%)",
            trainable,
            total,
            100 * trainable / max(total, 1),
        )

    # --- guards -------------------------------------------------------------

    def assert_double_backward_safe(self) -> None:
        """Verify the model can produce second-order gradients.

        Raises:
            DoubleBackwardUnsafeError: If a fused attention kernel is active or
                gradient checkpointing is enabled anywhere in the model.
        """
        config = getattr(self.model, "config", None)
        implementations = set()
        for candidate in (
            config,
            getattr(config, "vision_config", None),
            getattr(config, "text_config", None),
        ):
            impl = getattr(candidate, "_attn_implementation", None)
            if impl is not None:
                implementations.add(impl)
        bad = implementations - {"eager"}
        if bad:
            raise DoubleBackwardUnsafeError(
                f"attention implementations {sorted(bad)} have no double-backward rule; "
                "Stage 1 requires 'eager' everywhere."
            )

        for module in self.model.modules():
            if getattr(module, "gradient_checkpointing", False):
                raise DoubleBackwardUnsafeError(
                    f"{type(module).__name__} has gradient checkpointing enabled, which breaks "
                    "the second-order graph."
                )

    def freeze_vision_tower(self) -> None:
        """Freeze every vision-tower parameter, including its LoRA layers."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if ".visual." in name or name.startswith("visual."):
                param.requires_grad_(False)
                frozen += param.numel()
        logger.info("Froze %d vision-tower parameters", frozen)

    def unfreeze_vision_lora(self) -> None:
        """Re-enable the vision tower's LoRA parameters, as Stage 2 requires."""
        unfrozen = 0
        for name, param in self.model.named_parameters():
            if ("visual." in name) and ("lora_" in name):
                param.requires_grad_(True)
                unfrozen += param.numel()
        logger.info("Unfroze %d vision-tower LoRA parameters", unfrozen)

    # --- forward ------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device the parameters live on."""
        return next(self.model.parameters()).device

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that currently require gradients."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass and return the logits.

        Args:
            pixel_values: ``(n_patches, patch_dim)`` flat Qwen patch tensor.
            image_grid_thw: ``(B, 3)`` grid metadata.
            input_ids: ``(B, L)`` token IDs, left-padded.
            attention_mask: ``(B, L)`` attention mask.
            mm_token_type_ids: ``(B, L)`` marking image-token positions. Derived
                from the image token ID when omitted.

        Returns:
            ``(B, L, vocab)`` logits.
        """
        if mm_token_type_ids is None:
            mm_token_type_ids = self.infer_mm_token_type_ids(input_ids)
        outputs = self.model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )
        return outputs.logits

    def infer_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mark image-token positions from the configured image token ID.

        Args:
            input_ids: ``(B, L)`` token IDs.

        Returns:
            ``(B, L)`` long tensor with 1 at image positions.
        """
        config = getattr(self.model, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            return torch.zeros_like(input_ids)
        return (input_ids == image_token_id).long()

    @staticmethod
    def target_logit(logits: torch.Tensor, target_token_id: torch.Tensor) -> torch.Tensor:
        """Select the logit attribution is computed against.

        The prompt ends with the opening ``<answer>`` tag and is left-padded, so
        the final position predicts the answer's first token for every sample in
        the batch.

        Args:
            logits: ``(B, L, vocab)`` logits.
            target_token_id: ``(B,)`` or scalar token IDs.

        Returns:
            ``(B,)`` scalar logits.
        """
        last = logits[:, -1, :]
        ids = torch.as_tensor(target_token_id, device=last.device).reshape(-1)
        if ids.numel() == 1 and last.shape[0] > 1:
            ids = ids.expand(last.shape[0])
        return last.gather(1, ids.unsqueeze(1).long()).squeeze(1)

    def save_adapter(self, path: Path | str) -> None:
        """Write the LoRA adapter to disk.

        Args:
            path: Destination directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(path))

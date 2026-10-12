"""Differentiable expected gradients over Qwen's visual patches.

``L_align`` is a function of ``d(logit)/d(pixel)``, so its gradient with respect
to the model parameters is a second derivative. ``captum.attr.GradientShap``
calls ``backward()`` internally *without* ``create_graph=True``, which makes the
returned attribution a leaf: no gradient reaches the LoRA weights and the loss
decreases in the logs while nothing is learned. Expected gradients are therefore
implemented here directly, with ``create_graph`` under the caller's control.

Attribution is taken with respect to ``pixel_values``. Since that tensor is
already patch-shaped, summing over its feature dimension gives one scalar per
16x16 patch; :mod:`src.models.patch_grid` then undoes the merge-window
interleaving before the result is pooled to the PxP grid.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from src.models.patch_grid import pool_to_square_grid, unflatten_patch_grid
from src.models.vlm_wrapper import VLMWrapper

logger = logging.getLogger(__name__)

BaselineKind = Literal["black", "gaussian"]
AggKind = Literal["sum", "abs"]


class DifferentiableSHAPModule(nn.Module):
    """Expected gradients producing a PxP spatial attribution distribution."""

    def __init__(
        self,
        vlm: VLMWrapper,
        patch_grid_size: int = 16,
        n_baselines: int = 5,
        baseline: BaselineKind = "black",
        attr_agg: AggKind = "sum",
        merge_size: int = 2,
    ) -> None:
        """Create the attribution module.

        Args:
            vlm: The wrapped backbone.
            patch_grid_size: Edge length P of the output grid.
            n_baselines: Number of baseline/interpolation samples.
            baseline: ``"black"`` (zeros, absence of visual information) or
                ``"gaussian"`` noise.
            attr_agg: ``"sum"`` keeps the sign, ``"abs"`` does not. Unsigned
                attributions look better and mean less.
            merge_size: Spatial merge window of the vision tower.
        """
        super().__init__()
        self.vlm = vlm
        self.patch_grid_size = patch_grid_size
        self.n_baselines = n_baselines
        self.baseline = baseline
        self.attr_agg = attr_agg
        self.merge_size = merge_size

    def _make_baseline(self, pixel_values: torch.Tensor, index: int) -> torch.Tensor:
        """Build one baseline for the expected-gradients sum.

        Args:
            pixel_values: ``(n_patches, patch_dim)`` reference tensor.
            index: Index of this baseline, used to vary the noise seed.

        Returns:
            A tensor shaped like ``pixel_values``.
        """
        if self.baseline == "black":
            return torch.zeros_like(pixel_values)
        generator = torch.Generator(device="cpu").manual_seed(index)
        noise = torch.randn(pixel_values.shape, generator=generator, dtype=torch.float32)
        return noise.to(device=pixel_values.device, dtype=pixel_values.dtype)

    def _interpolation_weights(
        self, image_grid_thw: torch.Tensor, n_patches: int, device: torch.device, index: int
    ) -> torch.Tensor:
        """Draw one interpolation coefficient per image, broadcast over its patches.

        Expected gradients interpolate per *sample*, not per patch; the patches
        of one image are concatenated in ``pixel_values``, so the scalar drawn
        for an image is repeated across its rows.

        Args:
            image_grid_thw: ``(B, 3)`` grid metadata.
            n_patches: Total number of patch rows.
            device: Device for the result.
            index: Baseline index, used to vary the seed deterministically.

        Returns:
            ``(n_patches, 1)`` coefficients in ``[0, 1)``.
        """
        thw = image_grid_thw.reshape(-1, 3)
        counts = [int(t * gh * gw) for t, gh, gw in thw.tolist()]
        generator = torch.Generator(device="cpu").manual_seed(index)
        per_image = torch.rand(len(counts), generator=generator)
        weights = torch.repeat_interleave(per_image, torch.tensor(counts))
        if weights.shape[0] != n_patches:
            weights = weights[:n_patches]
        return weights.to(device).unsqueeze(1)

    def attributions(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_ids: torch.Tensor,
        answer_lengths: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Compute raw per-patch attributions.

        Args:
            pixel_values: ``(n_patches, patch_dim)`` flat Qwen patch tensor.
            image_grid_thw: ``(B, 3)`` grid metadata.
            input_ids: ``(B, L)`` token IDs.
            attention_mask: ``(B, L)`` attention mask.
            target_token_ids: ``(B, T)`` teacher-forced answer token IDs.
            answer_lengths: ``(B,)`` number of real answer tokens per sample.
            mm_token_type_ids: ``(B, L)`` image-position markers.
            create_graph: Keep the graph so the result stays differentiable with
                respect to the model parameters. Required in Stage 1, wasteful
                in Stage 2.

        Returns:
            ``(n_patches,)`` attribution per patch.
        """
        pixel_values = pixel_values.detach()
        total = torch.zeros_like(pixel_values)

        for index in range(self.n_baselines):
            baseline = self._make_baseline(pixel_values, index)
            alpha = self._interpolation_weights(
                image_grid_thw, pixel_values.shape[0], pixel_values.device, index
            )
            interpolated = (baseline + alpha * (pixel_values - baseline)).requires_grad_(True)

            logits = self.vlm(
                pixel_values=interpolated,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            target = self.vlm.target_logit(logits, target_token_ids, answer_lengths)
            (grad,) = torch.autograd.grad(target.sum(), interpolated, create_graph=create_graph)
            total = total + grad * (pixel_values - baseline)

        per_patch = (total / self.n_baselines).sum(dim=-1)
        return per_patch.abs() if self.attr_agg == "abs" else per_patch

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_ids: torch.Tensor,
        answer_lengths: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Compute the spatial SHAP map.

        Args:
            pixel_values: ``(n_patches, patch_dim)`` flat Qwen patch tensor.
            image_grid_thw: ``(B, 3)`` grid metadata.
            input_ids: ``(B, L)`` token IDs.
            attention_mask: ``(B, L)`` attention mask.
            target_token_ids: ``(B, T)`` teacher-forced answer token IDs.
            answer_lengths: ``(B,)`` number of real answer tokens per sample.
            mm_token_type_ids: ``(B, L)`` image-position markers.
            create_graph: Keep the second-order graph; see :meth:`attributions`.

        Returns:
            ``(B, P, P)`` attribution normalised to a probability distribution
            by a softmax over the P*P cells.
        """
        per_patch = self.attributions(
            pixel_values,
            image_grid_thw,
            input_ids,
            attention_mask,
            target_token_ids,
            answer_lengths,
            mm_token_type_ids,
            create_graph=create_graph,
        )
        grids = unflatten_patch_grid(per_patch, image_grid_thw, self.merge_size)
        pooled = torch.stack([pool_to_square_grid(g, self.patch_grid_size) for g in grids], dim=0)
        flat = pooled.flatten(1).float()
        return F.softmax(flat, dim=-1).view(-1, self.patch_grid_size, self.patch_grid_size)

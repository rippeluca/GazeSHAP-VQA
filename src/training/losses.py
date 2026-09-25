"""Stage 1 alignment losses between the SHAP map and the human fixation map.

Both maps are probability distributions over the PxP patch grid. The weighted
MSE sharpens the penalty in the regions people actually looked at, which matters
because a fixation map is mostly near-zero: a plain MSE is minimised well enough
by predicting the background everywhere.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812

EPS = 1e-8


def loss_align(shap_map: torch.Tensor, gaze_map: torch.Tensor, alpha: float = 1.1) -> torch.Tensor:
    """Weighted mean squared error between the SHAP map and the gaze map.

    The weight ``1 / (alpha - gaze)`` grows as the gaze density approaches
    ``alpha``, so errors on fixated cells cost more than errors on background.
    ``alpha`` must exceed 1.0 to keep the weight positive for any normalised
    gaze value.

    Args:
        shap_map: ``(B, P, P)`` softmax-normalised attribution.
        gaze_map: ``(B, P, P)`` normalised fixation distribution.
        alpha: Weight-sharpening constant, strictly greater than 1.0.

    Returns:
        Scalar loss.

    Raises:
        ValueError: If ``alpha <= 1.0`` or the shapes disagree.
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1.0 to keep the weights positive, got {alpha}")
    if shap_map.shape != gaze_map.shape:
        raise ValueError(f"shape mismatch: {tuple(shap_map.shape)} vs {tuple(gaze_map.shape)}")

    gaze = gaze_map.to(shap_map.dtype)
    weights = 1.0 / (alpha - gaze.clamp(max=alpha - 1e-6))
    return (weights * (shap_map - gaze).pow(2)).mean()


def loss_kl(shap_map: torch.Tensor, gaze_map: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """``KL(gaze || shap)``, the divergence Stage 1 reports and minimises.

    Args:
        shap_map: ``(B, P, P)`` softmax-normalised attribution.
        gaze_map: ``(B, P, P)`` normalised fixation distribution.
        eps: Smoothing constant keeping the logarithm finite.

    Returns:
        Scalar divergence, averaged over the batch.
    """
    shap = (shap_map + eps).flatten(1)
    gaze = (gaze_map.to(shap_map.dtype) + eps).flatten(1)
    shap = shap / shap.sum(dim=-1, keepdim=True)
    gaze = gaze / gaze.sum(dim=-1, keepdim=True)
    return F.kl_div(shap.log(), gaze, reduction="batchmean")


def stage1_loss(
    shap_map: torch.Tensor,
    gaze_map: torch.Tensor,
    alpha: float = 1.1,
    lambda_kl: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine the weighted MSE and the auxiliary KL term.

    Args:
        shap_map: ``(B, P, P)`` softmax-normalised attribution.
        gaze_map: ``(B, P, P)`` normalised fixation distribution.
        alpha: Weight-sharpening constant for the weighted MSE.
        lambda_kl: Weight of the KL term.

    Returns:
        The total loss and a dictionary of its components for logging.
    """
    align = loss_align(shap_map, gaze_map, alpha)
    kl = loss_kl(shap_map, gaze_map)
    total = align + lambda_kl * kl
    return total, {
        "loss_align": float(align.detach()),
        "loss_kl": float(kl.detach()),
        "loss_total": float(total.detach()),
    }

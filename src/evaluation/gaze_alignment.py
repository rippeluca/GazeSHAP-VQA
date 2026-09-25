"""Saliency-metric implementations comparing a SHAP map to a human gaze map.

Definitions follow Bylinskii et al., TPAMI 2019. AiR-D supplies its own
reference point: splitting the participants in half gives an inter-subject
AUC-Judd of **0.895**, which is the ceiling any model is measured against.

The gaze maps here are continuous densities rather than discrete fixation
points, so the location-based metrics (AUC-Judd, NSS) binarise the gaze map at a
quantile first. The quantile is a parameter and must be reported alongside the
number.
"""

from __future__ import annotations

import numpy as np
import torch

HUMAN_AUC_JUDD_CEILING = 0.895
DEFAULT_FIXATION_QUANTILE = 0.9
EPS = 1e-12


def _as_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor or array to a flat float64 numpy array.

    Args:
        array: Input map.

    Returns:
        Flattened float64 array.
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64).ravel()


def binarise_gaze(
    gaze_map: torch.Tensor | np.ndarray, quantile: float = DEFAULT_FIXATION_QUANTILE
) -> np.ndarray:
    """Turn a continuous gaze density into a fixation mask.

    Args:
        gaze_map: Continuous gaze density.
        quantile: Cells above this quantile of the map count as fixated.

    Returns:
        Flat boolean array. At least one cell is always marked, so degenerate
        maps do not produce empty ground truth.
    """
    flat = _as_numpy(gaze_map)
    threshold = float(np.quantile(flat, quantile))
    mask = flat > threshold
    if not mask.any():
        mask = flat >= flat.max()
    return mask


def auc_judd(
    saliency: torch.Tensor | np.ndarray,
    gaze_map: torch.Tensor | np.ndarray,
    quantile: float = DEFAULT_FIXATION_QUANTILE,
) -> float:
    """AUC-Judd: how well the saliency map ranks fixated cells above the rest.

    Computed as the Mann-Whitney U statistic over average ranks rather than by
    sweeping thresholds. The two agree wherever the saliency values are
    distinct, but the rank form degrades correctly under ties: a constant map
    scores exactly 0.5 instead of collapsing to 0, which matters because a
    degenerate SHAP map is a failure mode we expect to see (idea.md 14).

    Args:
        saliency: Predicted saliency, e.g. the SHAP map.
        gaze_map: Human gaze density.
        quantile: Quantile used to binarise the gaze map.

    Returns:
        Area under the curve in ``[0, 1]``; 0.5 is chance.
    """
    from scipy.stats import rankdata

    scores = _as_numpy(saliency)
    mask = binarise_gaze(gaze_map, quantile)
    n_fix = int(mask.sum())
    n_other = scores.size - n_fix
    if n_fix == 0 or n_other == 0:
        return 0.5

    ranks = rankdata(scores)  # ascending, ties get the average rank
    rank_sum = float(ranks[mask].sum())
    return (rank_sum - n_fix * (n_fix + 1) / 2.0) / (n_fix * n_other)


def cc(saliency: torch.Tensor | np.ndarray, gaze_map: torch.Tensor | np.ndarray) -> float:
    """Pearson correlation between two maps.

    Args:
        saliency: Predicted saliency.
        gaze_map: Human gaze density.

    Returns:
        Correlation in ``[-1, 1]``; 0.0 if either map is constant.
    """
    x = _as_numpy(saliency)
    y = _as_numpy(gaze_map)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return 0.0 if denom < EPS else float((x * y).sum() / denom)


def nss(
    saliency: torch.Tensor | np.ndarray,
    gaze_map: torch.Tensor | np.ndarray,
    quantile: float = DEFAULT_FIXATION_QUANTILE,
) -> float:
    """Normalised scanpath saliency: standardised saliency at fixated cells.

    Args:
        saliency: Predicted saliency.
        gaze_map: Human gaze density.
        quantile: Quantile used to binarise the gaze map.

    Returns:
        Mean standardised saliency over fixated cells; 0.0 for a constant map.
    """
    scores = _as_numpy(saliency)
    std = scores.std()
    if std < EPS:
        return 0.0
    standardised = (scores - scores.mean()) / std
    return float(standardised[binarise_gaze(gaze_map, quantile)].mean())


def kl_divergence(
    saliency: torch.Tensor | np.ndarray, gaze_map: torch.Tensor | np.ndarray, eps: float = 1e-8
) -> float:
    """``KL(gaze || saliency)``, the direction Stage 1 optimises.

    This penalises saliency mass missing where people looked, rather than extra
    mass where they did not. It is the same quantity as
    :func:`src.training.losses.loss_kl`, reported per sample.

    Args:
        saliency: Predicted saliency.
        gaze_map: Human gaze density.
        eps: Smoothing constant.

    Returns:
        Non-negative divergence.
    """
    p = _as_numpy(gaze_map) + eps
    q = _as_numpy(saliency) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def all_metrics(
    saliency: torch.Tensor | np.ndarray,
    gaze_map: torch.Tensor | np.ndarray,
    quantile: float = DEFAULT_FIXATION_QUANTILE,
) -> dict[str, float]:
    """Compute every alignment metric for one map pair.

    Args:
        saliency: Predicted saliency.
        gaze_map: Human gaze density.
        quantile: Quantile used to binarise the gaze map.

    Returns:
        Mapping from metric name to value.
    """
    return {
        "auc_judd": auc_judd(saliency, gaze_map, quantile),
        "cc": cc(saliency, gaze_map),
        "nss": nss(saliency, gaze_map, quantile),
        "kl": kl_divergence(saliency, gaze_map),
    }


def batch_metrics(
    saliency: torch.Tensor,
    gaze_map: torch.Tensor,
    quantile: float = DEFAULT_FIXATION_QUANTILE,
) -> dict[str, float]:
    """Average the alignment metrics over a batch of map pairs.

    Args:
        saliency: ``(B, P, P)`` predicted saliency.
        gaze_map: ``(B, P, P)`` human gaze density.
        quantile: Quantile used to binarise the gaze maps.

    Returns:
        Mapping from metric name to the batch mean.
    """
    per_sample = [all_metrics(s, g, quantile) for s, g in zip(saliency, gaze_map, strict=True)]
    return {key: float(np.mean([m[key] for m in per_sample])) for key in per_sample[0]}


def centre_prior(size: int, sigma: float = 15.0, map_size: int = 256) -> np.ndarray:
    """The C1 control: an isotropic Gaussian at the image centre.

    Human fixations are strongly centre-biased, so a model that learns "look at
    the middle" scores well on every saliency metric without having learned
    anything about the question. ``sigma`` is expressed in the 256x256 map space
    the AiR authors used and rescaled to the requested grid.

    Args:
        size: Edge length of the square output grid.
        sigma: Standard deviation in 256x256 map pixels.
        map_size: Reference map size the sigma refers to.

    Returns:
        ``(size, size)`` array summing to 1.0.
    """
    scaled_sigma = sigma * size / map_size
    coords = np.arange(size) - (size - 1) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    prior = np.exp(-(xx**2 + yy**2) / (2.0 * scaled_sigma**2))
    return prior / prior.sum()

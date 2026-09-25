"""Turn AiR-D gaze data into the PxP fixation grids Stage 1 trains against.

Two paths produce the same output format:

* **Path A** reads the saliency maps AiR-D ships and downsamples them.
* **Path B** rebuilds the maps from the raw scanpaths, which also makes
  per-subject maps available as Stage 1 augmentation.

Temporal binning is not optional. AiR-D aligns the first fixation with the image
centre, so the 0-1 s bin is an experimental artefact rather than
question-driven attention; the default keeps only the 1-2 s and 2-3 s bins.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from src.data.air_d import AiRDataset, Scanpath

logger = logging.getLogger(__name__)

TEMPORAL_BINS: tuple[str, ...] = ("1s", "2s")
ALL_BINS: tuple[str, ...] = ("0s", "1s", "2s")
GAZE_SIGMA_PX = 9.0
PATCH_GRID_SIZE = 16
MAP_SIZE = 256


def normalise(array: np.ndarray) -> np.ndarray:
    """Scale a non-negative array to sum to one.

    Args:
        array: Array of non-negative values.

    Returns:
        A float32 array summing to 1.0, or a uniform distribution if the input
        is all zeros.
    """
    arr = np.asarray(array, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float32)
    return arr / total


def fixation_map_from_scanpaths(
    scanpaths: Iterable[Scanpath],
    image_size: tuple[int, int],
    map_size: int = MAP_SIZE,
    sigma: float = GAZE_SIGMA_PX,
    duration_weighted: bool = True,
) -> np.ndarray:
    """Rebuild an aggregated fixation map from raw scanpaths (Path B).

    Fixation coordinates are stored in original image pixels, so they are scaled
    into the square ``map_size`` grid first. Each fixation contributes a
    duration-weighted point mass; the Gaussian is applied once to the summed
    point masses, which is equivalent to summing per-fixation Gaussians.

    Per idea.md 10.5, per-subject maps are raw duration sums and only the final
    aggregate is normalised.

    Args:
        scanpaths: Scanpaths to aggregate, typically all subjects of one question.
        image_size: ``(width, height)`` of the original image.
        map_size: Edge length of the square output map.
        sigma: Gaussian standard deviation in map pixels (~1 degree of visual angle).
        duration_weighted: Weight each fixation by its duration.

    Returns:
        ``(map_size, map_size)`` float32 array summing to 1.0.
    """
    width, height = image_size
    points = np.zeros((map_size, map_size), dtype=np.float64)

    for scanpath in scanpaths:
        if len(scanpath.xy) == 0:
            continue
        xs = scanpath.xy[:, 0] / max(width, 1) * map_size
        ys = scanpath.xy[:, 1] / max(height, 1) * map_size
        weights = scanpath.durations_ms if duration_weighted else np.ones(len(xs))
        cols = np.clip(xs.astype(np.int64), 0, map_size - 1)
        rows = np.clip(ys.astype(np.int64), 0, map_size - 1)
        np.add.at(points, (rows, cols), weights)

    if points.sum() <= 0.0:
        return normalise(points)
    smoothed = gaussian_filter(points, sigma=sigma, mode="constant", cval=0.0)
    return normalise(smoothed)


def downsample_to_grid(fixation_map: np.ndarray, grid_size: int = PATCH_GRID_SIZE) -> np.ndarray:
    """Downsample a fixation map to the patch grid and renormalise.

    ``cv2.INTER_AREA`` averages over each destination cell, which preserves the
    relative mass of a region; the explicit renormalisation afterwards restores
    the probability simplex exactly.

    Args:
        fixation_map: ``(H, W)`` non-negative map.
        grid_size: Edge length of the square output grid.

    Returns:
        ``(grid_size, grid_size)`` float32 array summing to 1.0.
    """
    small = cv2.resize(
        np.asarray(fixation_map, dtype=np.float32),
        (grid_size, grid_size),
        interpolation=cv2.INTER_AREA,
    )
    return normalise(np.clip(small, 0.0, None))


def load_map_path_a(
    dataset: AiRDataset,
    qid: str,
    bins: Sequence[str] = TEMPORAL_BINS,
    variant: str = "fixmaps",
) -> np.ndarray:
    """Load an aggregated fixation map from the shipped saliency maps (Path A).

    Each temporal bin ships as an independently normalised PNG, so the relative
    mass between bins cannot be recovered; summing the selected bins weights
    them equally. Passing ``bins=()`` uses the whole-trial map instead, which
    keeps the original weighting but includes the centre-biased first second.

    Args:
        dataset: Open AiR-D dataset.
        qid: Question ID.
        bins: Temporal bins to combine, e.g. ``("1s", "2s")``.
        variant: ``fixmaps``, ``fixmaps_corr`` or ``fixmaps_incorr``.

    Returns:
        ``(map_size, map_size)`` float32 array summing to 1.0.

    Raises:
        FileNotFoundError: If none of the requested maps exist.
    """
    if not bins:
        return normalise(dataset.load_map(dataset.aggregated_map_path(qid, variant)))

    total: np.ndarray | None = None
    for bin_name in bins:
        path = dataset.temporal_map_path(qid, bin_name, variant)
        if not path.exists():
            continue
        contribution = normalise(dataset.load_map(path))
        total = contribution if total is None else total + contribution
    if total is None:
        raise FileNotFoundError(
            f"no temporal maps for qid={qid} variant={variant} bins={tuple(bins)}"
        )
    return normalise(total)


def load_map_path_b(
    dataset: AiRDataset,
    qid: str,
    bins: Sequence[str] = TEMPORAL_BINS,
    sigma: float = GAZE_SIGMA_PX,
    map_size: int = MAP_SIZE,
    correct_only: bool = False,
) -> np.ndarray:
    """Rebuild an aggregated fixation map from the raw scanpaths (Path B).

    Args:
        dataset: Open AiR-D dataset.
        qid: Question ID.
        bins: Temporal bins to keep.
        sigma: Gaussian standard deviation in map pixels.
        map_size: Edge length of the square output map.
        correct_only: Keep only participants who answered correctly.

    Returns:
        ``(map_size, map_size)`` float32 array summing to 1.0.
    """
    scanpaths = dataset.load_scanpaths(qid)
    if correct_only:
        scanpaths = [s for s in scanpaths if dataset.subject_was_correct(qid, s.subject) is True]
    selected = [s.in_bins(tuple(bins)) for s in scanpaths]
    return fixation_map_from_scanpaths(selected, dataset.image_size(qid), map_size, sigma)


def per_subject_maps(
    dataset: AiRDataset,
    qid: str,
    bins: Sequence[str] = TEMPORAL_BINS,
    sigma: float = GAZE_SIGMA_PX,
    map_size: int = MAP_SIZE,
) -> dict[str, np.ndarray]:
    """Build one fixation map per participant, for Stage 1 augmentation.

    Args:
        dataset: Open AiR-D dataset.
        qid: Question ID.
        bins: Temporal bins to keep.
        sigma: Gaussian standard deviation in map pixels.
        map_size: Edge length of the square output map.

    Returns:
        Mapping from participant code to a normalised map. Participants whose
        trial has no fixation in the selected bins are omitted.
    """
    size = dataset.image_size(qid)
    maps: dict[str, np.ndarray] = {}
    for scanpath in dataset.load_scanpaths(qid):
        binned = scanpath.in_bins(tuple(bins))
        if len(binned.xy) == 0:
            continue
        maps[scanpath.subject] = fixation_map_from_scanpaths([binned], size, map_size, sigma)
    return maps


def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two maps.

    Args:
        a: First map.
        b: Second map.

    Returns:
        Correlation in ``[-1, 1]``; 0.0 if either map is constant.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return 0.0 if denom == 0.0 else float((x * y).sum() / denom)

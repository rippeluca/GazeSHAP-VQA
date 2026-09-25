from __future__ import annotations

import numpy as np
import pytest

from src.data.air_d import Scanpath
from src.data.gaze_preprocessing import (
    ALL_BINS,
    TEMPORAL_BINS,
    correlation_coefficient,
    downsample_to_grid,
    fixation_map_from_scanpaths,
    normalise,
)


def _scanpath(xy, t, qid="q", subject="AL"):
    return Scanpath(qid, subject, np.asarray(xy, float), np.asarray(t, np.int64))


# --- normalisation ----------------------------------------------------------


def test_normalise_sums_to_one():
    out = normalise(np.array([[1.0, 3.0], [0.0, 4.0]]))
    assert out.sum() == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(0.5)


def test_normalise_all_zero_returns_uniform():
    out = normalise(np.zeros((4, 4)))
    assert out.sum() == pytest.approx(1.0)
    assert np.allclose(out, 1 / 16)


# --- Gaussian placement -----------------------------------------------------


def test_single_fixation_peaks_at_its_location():
    # Fixation at 3/4 across, 1/4 down of a 100x200 image.
    sp = _scanpath([[75.0, 50.0]], [[1000, 1200]])
    m = fixation_map_from_scanpaths([sp], image_size=(100, 200), map_size=64, sigma=2.0)
    row, col = np.unravel_index(int(np.argmax(m)), m.shape)
    assert col == pytest.approx(48, abs=1)  # 0.75 * 64
    assert row == pytest.approx(16, abs=1)  # 0.25 * 64
    assert m.sum() == pytest.approx(1.0, abs=1e-5)


def test_map_is_symmetric_for_a_centred_fixation():
    # Odd map size so a single fixation can land on the exact centre pixel;
    # on an even grid the floor() of the scaled coordinate is half a pixel off.
    sp = _scanpath([[50.0, 50.0]], [[0, 100]])
    m = fixation_map_from_scanpaths([sp], image_size=(100, 100), map_size=65, sigma=3.0)
    assert np.unravel_index(int(np.argmax(m)), m.shape) == (32, 32)
    assert np.allclose(m, np.fliplr(m), atol=1e-6)
    assert np.allclose(m, np.flipud(m), atol=1e-6)


def test_duration_weighting_favours_the_longer_fixation():
    short = [[20.0, 50.0]]
    long_ = [[80.0, 50.0]]
    sp = _scanpath(short + long_, [[0, 50], [50, 1050]])

    weighted = fixation_map_from_scanpaths([sp], (100, 100), map_size=64, sigma=2.0)
    uniform = fixation_map_from_scanpaths(
        [sp], (100, 100), map_size=64, sigma=2.0, duration_weighted=False
    )
    left, right = slice(None), slice(None)
    assert weighted[left, 40:].sum() > weighted[left, :24].sum()
    assert uniform[right, 40:].sum() == pytest.approx(uniform[right, :24].sum(), rel=0.05)


def test_empty_scanpaths_give_a_uniform_map():
    empty = _scanpath(np.zeros((0, 2)), np.zeros((0, 2)))
    m = fixation_map_from_scanpaths([empty], (100, 100), map_size=16, sigma=2.0)
    assert np.allclose(m, 1 / 256)


def test_subject_aggregation_is_the_mean_of_the_parts():
    a = _scanpath([[20.0, 20.0]], [[0, 500]], subject="A")
    b = _scanpath([[80.0, 80.0]], [[0, 500]], subject="B")
    both = fixation_map_from_scanpaths([a, b], (100, 100), 64, 3.0)
    mean_of_parts = normalise(
        fixation_map_from_scanpaths([a], (100, 100), 64, 3.0)
        + fixation_map_from_scanpaths([b], (100, 100), 64, 3.0)
    )
    assert np.allclose(both, mean_of_parts, atol=1e-6)


def test_coordinates_outside_the_image_are_clipped_not_wrapped():
    sp = _scanpath([[199.0, 199.0]], [[0, 500]])
    m = fixation_map_from_scanpaths([sp], (100, 100), map_size=32, sigma=1.0)
    row, col = np.unravel_index(int(np.argmax(m)), m.shape)
    assert (row, col) == (31, 31)


# --- temporal binning -------------------------------------------------------


def test_bins_select_by_fixation_start():
    sp = _scanpath(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]],
        [[100, 900], [1100, 1900], [2100, 2900], [950, 1050]],
    )
    kept = sp.in_bins(TEMPORAL_BINS)
    # The 0-1s fixation is dropped; the one starting at 950 ms is too.
    assert kept.xy.tolist() == [[20.0, 20.0], [30.0, 30.0]]

    assert len(sp.in_bins(ALL_BINS).xy) == 4
    assert len(sp.in_bins(()).xy) == 4
    assert len(sp.in_bins(("0s",)).xy) == 2


def test_dropping_the_first_second_changes_the_map():
    centre = [[50.0, 50.0]]
    corner = [[90.0, 10.0]]
    sp = _scanpath(centre + corner, [[0, 900], [1100, 2000]])

    full = fixation_map_from_scanpaths([sp.in_bins(ALL_BINS)], (100, 100), 32, 2.0)
    late = fixation_map_from_scanpaths([sp.in_bins(TEMPORAL_BINS)], (100, 100), 32, 2.0)
    assert np.unravel_index(int(np.argmax(late)), late.shape)[1] > 20
    assert correlation_coefficient(full, late) < 0.95


# --- downsampling -----------------------------------------------------------


def test_downsample_preserves_mass_and_shape():
    rng = np.random.default_rng(0)
    m = normalise(rng.random((256, 256)))
    grid = downsample_to_grid(m, 16)
    assert grid.shape == (16, 16)
    assert grid.sum() == pytest.approx(1.0, abs=1e-5)


def test_downsample_preserves_where_the_mass_is():
    m = np.zeros((256, 256), dtype=np.float32)
    m[0:16, 240:256] = 1.0  # top-right corner block, exactly one 16x16 cell
    grid = downsample_to_grid(normalise(m), 16)
    assert grid[0, 15] == pytest.approx(1.0, abs=1e-5)
    assert grid.sum() == pytest.approx(1.0, abs=1e-5)


def test_downsample_of_uniform_is_uniform():
    grid = downsample_to_grid(np.ones((256, 256)), 16)
    assert np.allclose(grid, 1 / 256, atol=1e-6)


# --- correlation ------------------------------------------------------------


def test_correlation_bounds():
    rng = np.random.default_rng(1)
    a = rng.random((16, 16))
    assert correlation_coefficient(a, a) == pytest.approx(1.0)
    assert correlation_coefficient(a, -a) == pytest.approx(-1.0)
    assert correlation_coefficient(a, np.ones((16, 16))) == 0.0

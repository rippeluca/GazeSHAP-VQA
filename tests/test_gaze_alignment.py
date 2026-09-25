from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.gaze_alignment import (
    all_metrics,
    auc_judd,
    batch_metrics,
    binarise_gaze,
    cc,
    centre_prior,
    kl_divergence,
    nss,
)

P = 16


def _peaked(row=3, col=4, size=P):
    m = np.zeros((size, size))
    m[row, col] = 1.0
    return m


def _blob(row=3, col=4, size=P, sigma=1.5):
    yy, xx = np.mgrid[0:size, 0:size]
    m = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma**2))
    return m / m.sum()


# --- binarisation -----------------------------------------------------------


def test_binarise_marks_the_top_cells():
    mask = binarise_gaze(_blob(), quantile=0.9)
    assert mask.sum() == pytest.approx(P * P * 0.1, abs=3)


def test_binarise_never_returns_an_empty_mask():
    assert binarise_gaze(np.zeros((P, P))).any()


# --- AUC-Judd ---------------------------------------------------------------


def test_auc_judd_is_one_for_a_perfect_prediction():
    gaze = _blob()
    assert auc_judd(gaze, gaze) == pytest.approx(1.0, abs=1e-6)


def test_auc_judd_is_near_chance_for_a_uniform_prediction():
    uniform = np.full((P, P), 1.0 / (P * P))
    assert auc_judd(uniform, _blob()) == pytest.approx(0.5, abs=0.1)


def test_auc_judd_is_low_for_an_anticorrelated_prediction():
    gaze = _blob(row=2, col=2)
    assert auc_judd(-gaze, gaze) < 0.2


def test_auc_judd_beats_chance_for_a_partially_right_prediction():
    gaze = _blob(row=8, col=8)
    noisy = gaze + np.random.default_rng(0).normal(0, gaze.max() * 0.3, gaze.shape)
    assert auc_judd(noisy, gaze) > 0.7


def test_auc_judd_accepts_torch_tensors():
    gaze = torch.from_numpy(_blob())
    assert auc_judd(gaze, gaze) == pytest.approx(1.0, abs=1e-6)


def test_auc_judd_is_invariant_to_monotonic_rescaling():
    gaze = _blob()
    prediction = _blob(row=5, col=5)
    assert auc_judd(prediction, gaze) == pytest.approx(auc_judd(prediction * 100 + 3, gaze))


# --- CC ---------------------------------------------------------------------


def test_cc_bounds():
    gaze = _blob()
    assert cc(gaze, gaze) == pytest.approx(1.0)
    assert cc(-gaze, gaze) == pytest.approx(-1.0)
    assert cc(np.ones((P, P)), gaze) == 0.0


# --- NSS --------------------------------------------------------------------


def test_nss_is_positive_when_saliency_peaks_on_fixations():
    gaze = _blob(row=6, col=6)
    assert nss(gaze, gaze) > 1.0


def test_nss_is_zero_for_a_constant_map():
    assert nss(np.ones((P, P)), _blob()) == 0.0


def test_nss_is_negative_when_saliency_avoids_fixations():
    gaze = _blob(row=2, col=2)
    away = _blob(row=13, col=13)
    assert nss(away, gaze) < 0.0


# --- KL ---------------------------------------------------------------------


def test_kl_is_zero_for_identical_maps():
    gaze = _blob()
    assert kl_divergence(gaze, gaze) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_non_negative_and_larger_for_worse_maps():
    gaze = _blob(row=2, col=2)
    close = _blob(row=3, col=3)
    far = _blob(row=13, col=13)
    assert 0.0 <= kl_divergence(close, gaze) < kl_divergence(far, gaze)


def test_kl_matches_the_training_loss():
    from src.training.losses import loss_kl

    torch.manual_seed(0)
    shap = torch.softmax(torch.randn(1, P * P), dim=-1).view(1, P, P)
    gaze = torch.from_numpy(_blob()).float().unsqueeze(0)
    assert kl_divergence(shap[0], gaze[0]) == pytest.approx(float(loss_kl(shap, gaze)), abs=1e-4)


# --- aggregation ------------------------------------------------------------


def test_all_metrics_keys():
    assert set(all_metrics(_blob(), _blob())) == {"auc_judd", "cc", "nss", "kl"}


def test_batch_metrics_averages():
    a = torch.from_numpy(np.stack([_blob(2, 2), _blob(9, 9)]))
    assert batch_metrics(a, a)["auc_judd"] == pytest.approx(1.0, abs=1e-6)


# --- centre prior (C1 control) ----------------------------------------------


def test_centre_prior_is_normalised_and_symmetric():
    prior = centre_prior(P)
    assert prior.sum() == pytest.approx(1.0)
    assert np.allclose(prior, np.fliplr(prior))
    assert np.allclose(prior, np.flipud(prior))
    assert np.allclose(prior, prior.T)


def test_centre_prior_peaks_in_the_middle():
    prior = centre_prior(P)
    row, col = np.unravel_index(int(prior.argmax()), prior.shape)
    assert abs(row - (P - 1) / 2) <= 0.5
    assert abs(col - (P - 1) / 2) <= 0.5


def test_centre_prior_sigma_scales_with_grid_size():
    """Sigma is quoted in 256x256 map pixels, so the shape must not depend on P."""
    small = centre_prior(8, sigma=15.0)
    large = centre_prior(16, sigma=15.0)
    # The mass within the central quarter should be comparable at both resolutions.
    assert small[2:6, 2:6].sum() == pytest.approx(large[4:12, 4:12].sum(), abs=0.05)

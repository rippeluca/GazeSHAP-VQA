from __future__ import annotations

import pytest
import torch

from src.training.losses import loss_align, loss_kl, stage1_loss

P = 8


def _uniform(batch=1, size=P):
    return torch.full((batch, size, size), 1.0 / (size * size))


def _peaked(batch=1, size=P, row=2, col=5):
    m = torch.zeros(batch, size, size)
    m[:, row, col] = 1.0
    return m


# --- loss_align -------------------------------------------------------------


def test_align_is_non_negative():
    torch.manual_seed(0)
    a = torch.softmax(torch.randn(3, P * P), dim=-1).view(3, P, P)
    b = torch.softmax(torch.randn(3, P * P), dim=-1).view(3, P, P)
    assert float(loss_align(a, b)) >= 0.0


def test_align_is_zero_for_identical_maps():
    m = _uniform()
    assert float(loss_align(m, m)) == pytest.approx(0.0, abs=1e-12)
    assert float(loss_align(_peaked(), _peaked())) == pytest.approx(0.0, abs=1e-12)


def test_align_is_large_when_uniform_meets_peaked():
    mismatch = float(loss_align(_uniform(), _peaked()))
    match = float(loss_align(_peaked(), _peaked()))
    assert mismatch > match
    assert mismatch > 1e-3


def test_align_weights_fixated_cells_more_heavily():
    """Missing a fixated cell must cost more than the same error on background."""
    gaze = _peaked(row=0, col=0)

    miss_the_peak = gaze.clone()
    miss_the_peak[0, 0, 0] = 0.5
    miss_the_peak[0, 7, 7] = 0.5

    wrong_background = gaze.clone()
    wrong_background[0, 4, 4] = 0.5
    wrong_background[0, 5, 5] = -0.5 + 0.5  # keep the peak intact
    wrong_background[0, 0, 0] = 1.0

    assert float(loss_align(miss_the_peak, gaze)) > float(loss_align(wrong_background, gaze))


def test_align_alpha_must_exceed_one():
    with pytest.raises(ValueError, match="alpha"):
        loss_align(_uniform(), _uniform(), alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        loss_align(_uniform(), _uniform(), alpha=0.5)


def test_align_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        loss_align(_uniform(size=8), _uniform(size=4))


def test_align_weights_stay_finite_at_the_extreme():
    """A gaze cell holding all the mass must not blow the weight up."""
    gaze = _peaked()
    out = loss_align(_uniform(), gaze, alpha=1.0 + 1e-5)
    assert torch.isfinite(out)


def test_align_is_differentiable():
    shap = torch.softmax(torch.randn(1, P * P, requires_grad=True), dim=-1).view(1, P, P)
    shap.retain_grad()
    loss_align(shap, _uniform()).backward()
    assert shap.grad is not None


# --- loss_kl ----------------------------------------------------------------


def test_kl_is_zero_for_identical_maps():
    m = _uniform()
    assert float(loss_kl(m, m)) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_positive_and_asymmetric():
    a, b = _uniform(), torch.softmax(_peaked().flatten(1) * 8, dim=-1).view(1, P, P)
    forward = float(loss_kl(a, b))
    backward = float(loss_kl(b, a))
    assert forward > 0 and backward > 0
    assert forward != pytest.approx(backward, rel=1e-3)


def test_kl_penalises_missing_gaze_mass():
    """KL(gaze || shap) explodes where the gaze has mass and the SHAP map does not."""
    gaze = _peaked(row=0, col=0)
    covers_the_peak = torch.full((1, P, P), 0.001)
    covers_the_peak[0, 0, 0] = 1.0 - 0.001 * (P * P - 1)
    misses_the_peak = torch.full((1, P, P), 0.001)
    misses_the_peak[0, 7, 7] = 1.0 - 0.001 * (P * P - 1)

    assert float(loss_kl(misses_the_peak, gaze)) > float(loss_kl(covers_the_peak, gaze))


def test_kl_handles_zeros_without_nan():
    gaze = _peaked()
    shap = torch.zeros(1, P, P)
    shap[0, 0, 0] = 1.0
    out = loss_kl(shap, gaze)
    assert torch.isfinite(out)


def test_kl_batch_mean_is_independent_of_batch_size():
    single = loss_kl(_uniform(1), _peaked(1))
    doubled = loss_kl(_uniform(2), _peaked(2))
    assert float(single) == pytest.approx(float(doubled), rel=1e-5)


# --- combined ---------------------------------------------------------------


def test_stage1_loss_components():
    shap, gaze = _uniform(), _peaked()
    total, parts = stage1_loss(shap, gaze, alpha=1.1, lambda_kl=0.1)
    assert set(parts) == {"loss_align", "loss_kl", "loss_total"}
    assert parts["loss_total"] == pytest.approx(
        parts["loss_align"] + 0.1 * parts["loss_kl"], rel=1e-5
    )
    assert float(total) == pytest.approx(parts["loss_total"], rel=1e-5)


def test_stage1_loss_lambda_zero_drops_the_kl_term():
    shap, gaze = _uniform(), _peaked()
    total, parts = stage1_loss(shap, gaze, lambda_kl=0.0)
    assert float(total) == pytest.approx(parts["loss_align"], rel=1e-6)

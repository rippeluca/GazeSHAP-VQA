"""Checks against the real download. Skipped when ``data/raw`` is not populated.

These encode what was established by inspecting the archives (see
``data/README.md``); if an upstream re-release changes the structure, these are
the tests that will say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.air_d import AiRDataset
from src.data.gaze_preprocessing import (
    ALL_BINS,
    TEMPORAL_BINS,
    correlation_coefficient,
    downsample_to_grid,
    load_map_path_a,
    load_map_path_b,
    normalise,
    per_subject_maps,
)

pytestmark = pytest.mark.slow

DATA_ROOT = "data"


@pytest.fixture(scope="module")
def dataset():
    try:
        return AiRDataset(DATA_ROOT)
    except FileNotFoundError:
        pytest.skip("data/raw not populated; run scripts/download_data.py")


@pytest.fixture(scope="module")
def sample_qids(dataset):
    rng = np.random.default_rng(42)
    return list(rng.choice(dataset.qids, 40, replace=False))


def test_canonical_question_count(dataset):
    assert len(dataset.qids) == 1422


def test_twenty_participants_ten_per_question(dataset):
    assert len(dataset.subjects) == 20
    counts = {len(dataset.scanpath_subjects(q)) for q in dataset.qids[:50]}
    assert counts == {10}


def test_accuracy_is_in_tenths(dataset):
    accuracies = list(dataset.responses["accuracy"].values())
    assert len(accuracies) == 1422
    assert all(abs(a * 10 - round(a * 10)) < 1e-9 for a in accuracies)
    assert sum(1 for a in accuracies if a == 1.0) == 479


def test_maps_are_256_grayscale(dataset, sample_qids):
    for qid in sample_qids[:5]:
        m = dataset.load_map(dataset.aggregated_map_path(qid))
        assert m.shape == (256, 256)
        assert m.min() >= 0.0 and m.max() <= 1.0


def test_fixations_lie_inside_the_image(dataset, sample_qids):
    for qid in sample_qids:
        width, height = dataset.image_size(qid)
        for scanpath in dataset.load_scanpaths(qid):
            if len(scanpath.xy) == 0:
                continue
            assert scanpath.xy[:, 0].max() <= width
            assert scanpath.xy[:, 1].max() <= height
            assert scanpath.xy.min() >= 0


def test_fixation_times_are_inside_the_trial(dataset, sample_qids):
    for qid in sample_qids:
        for scanpath in dataset.load_scanpaths(qid):
            if len(scanpath.t) == 0:
                continue
            assert scanpath.t.min() >= 0
            assert scanpath.t.max() <= 3000
            assert (scanpath.durations_ms >= 0).all()


def test_path_b_reproduces_path_a(dataset, sample_qids):
    """Rebuilding from scanpaths must reproduce the shipped maps.

    Measured at the 16x16 resolution Stage 1 actually trains on. With the
    documented sigma of 9 px this lands around CC 0.95; the shipped maps behave
    like an effective sigma near 13, so the threshold is deliberately loose.
    """
    scores = []
    for qid in sample_qids:
        a = downsample_to_grid(normalise(dataset.load_map(dataset.aggregated_map_path(qid))), 16)
        b = downsample_to_grid(load_map_path_b(dataset, qid, bins=ALL_BINS), 16)
        scores.append(correlation_coefficient(a, b))
    assert float(np.mean(scores)) > 0.9
    assert float(np.percentile(scores, 10)) > 0.8


def test_path_b_reproduces_path_a_on_the_training_bins(dataset, sample_qids):
    scores = [
        correlation_coefficient(
            downsample_to_grid(load_map_path_a(dataset, qid, TEMPORAL_BINS), 16),
            downsample_to_grid(load_map_path_b(dataset, qid, TEMPORAL_BINS), 16),
        )
        for qid in sample_qids
    ]
    assert float(np.mean(scores)) > 0.9


def test_first_second_is_centre_biased(dataset, sample_qids):
    """The 0-1s bin sits on the forced central fixation; 1-3s does not.

    This is the empirical justification for dropping the first second
    (idea.md 4.3) and doubles as a check that the binning works at all.
    """
    centre = np.zeros((256, 256), dtype=np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    centre = np.exp(-((xx - 127.5) ** 2 + (yy - 127.5) ** 2) / (2 * 15.0**2))
    centre = normalise(centre)

    early = [
        correlation_coefficient(centre, load_map_path_a(dataset, q, ("0s",))) for q in sample_qids
    ]
    late = [
        correlation_coefficient(centre, load_map_path_a(dataset, q, TEMPORAL_BINS))
        for q in sample_qids
    ]
    assert float(np.mean(early)) > float(np.mean(late)) + 0.2


def test_per_subject_maps_average_to_the_aggregate(dataset, sample_qids):
    for qid in sample_qids[:10]:
        maps = per_subject_maps(dataset, qid, TEMPORAL_BINS)
        if len(maps) < 2:
            continue
        aggregate = load_map_path_b(dataset, qid, TEMPORAL_BINS)
        mean_of_subjects = normalise(np.mean(list(maps.values()), axis=0))
        # Not identical: the aggregate weights subjects by total fixation
        # duration, the mean weights them equally.
        assert correlation_coefficient(aggregate, mean_of_subjects) > 0.85


def test_correct_only_maps_exist_only_for_mixed_questions(dataset):
    mixed = [q for q in dataset.qids if 0.0 < dataset.accuracy(q) < 1.0]
    assert len(mixed) == 934
    assert all(dataset.aggregated_map_path(q, "fixmaps_corr").exists() for q in mixed[:20])
    unanimous = [q for q in dataset.qids if dataset.accuracy(q) == 1.0][:20]
    assert not any(dataset.aggregated_map_path(q, "fixmaps_corr").exists() for q in unanimous)

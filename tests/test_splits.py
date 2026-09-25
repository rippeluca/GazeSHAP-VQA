from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.splits import (
    check_no_image_leakage,
    group_by_image,
    make_splits,
    read_splits,
    write_splits,
)

REPO = Path(__file__).resolve().parents[1]


def _questions(n_images=100, questions_per_image=2):
    """Synthetic GQA-like records: every image carries several questions."""
    out = {}
    for i in range(n_images):
        for j in range(questions_per_image):
            out[f"q{i:03d}_{j}"] = {"imageId": f"img{i:03d}", "answer": "yes"}
    return out


def test_group_by_image():
    questions = _questions(3, 2)
    grouped = group_by_image(list(questions), questions)
    assert set(grouped) == {"img000", "img001", "img002"}
    assert grouped["img000"] == ["q000_0", "q000_1"]


def test_split_covers_every_question_exactly_once():
    questions = _questions()
    splits = make_splits(list(questions), questions)
    allocated = [q for qs in splits.values() for q in qs]
    assert sorted(allocated) == sorted(questions)
    assert len(allocated) == len(set(allocated))


def test_no_image_appears_in_two_splits():
    questions = _questions()
    splits = make_splits(list(questions), questions)
    assert check_no_image_leakage(splits, questions) == []


def test_questions_of_one_image_stay_together():
    questions = _questions(50, 3)
    splits = make_splits(list(questions), questions)
    where = {q: name for name, qs in splits.items() for q in qs}
    for image_id, qids in group_by_image(list(questions), questions).items():
        assert len({where[q] for q in qids}) == 1, image_id


def test_fractions_are_respected_at_the_image_level():
    questions = _questions(200, 1)
    splits = make_splits(list(questions), questions, fractions=(0.7, 0.1, 0.2))
    assert len(splits["train"]) == 140
    assert len(splits["val"]) == 20
    assert len(splits["test"]) == 40


def test_split_is_deterministic_and_seed_dependent():
    questions = _questions()
    a = make_splits(list(questions), questions, seed=42)
    b = make_splits(list(questions), questions, seed=42)
    c = make_splits(list(questions), questions, seed=7)
    assert a == b
    assert a != c


def test_bad_fractions_rejected():
    questions = _questions(10, 1)
    with pytest.raises(ValueError, match="summing to 1.0"):
        make_splits(list(questions), questions, fractions=(0.5, 0.2, 0.2))


def test_leakage_detector_catches_a_bad_split():
    questions = _questions(4, 2)
    bad = {"train": ["q000_0"], "val": ["q000_1"], "test": ["q001_0"]}
    assert check_no_image_leakage(bad, questions) == ["img000"]


def test_round_trip_through_disk(tmp_path):
    questions = _questions(20, 2)
    splits = make_splits(list(questions), questions)
    write_splits(splits, tmp_path)
    assert read_splits(tmp_path) == splits


def test_missing_split_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_splits"):
        read_splits(tmp_path)


@pytest.mark.slow
def test_committed_splits_are_clean():
    """The committed split files must stay leak-free and cover all 1,422 questions."""
    split_dir = REPO / "data" / "processed" / "splits"
    questions_path = REPO / "data" / "raw" / "gqa" / "val_balanced_questions.json"
    if not split_dir.exists() or not questions_path.exists():
        pytest.skip("splits or GQA questions not present")

    splits = read_splits(split_dir)
    questions = json.loads(questions_path.read_text())
    assert sum(len(v) for v in splits.values()) == 1422
    assert check_no_image_leakage(splits, questions) == []

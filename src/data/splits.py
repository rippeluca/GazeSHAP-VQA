"""Image-level train/validation/test partitioning.

Splitting by question would leak: 299 of the 987 AiR-D images carry more than
one question, so a question-level split puts the same scene in train and test.
Everything here partitions by ``image_id`` and assigns a question to the split
its image landed in.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.7, 0.1, 0.2)
DEFAULT_SEED = 42


def group_by_image(qids: Sequence[str], questions: Mapping[str, dict]) -> dict[str, list[str]]:
    """Group question IDs by the image they refer to.

    Args:
        qids: Question IDs.
        questions: GQA question records keyed by question ID.

    Returns:
        Mapping from image ID to its question IDs, sorted.

    Raises:
        KeyError: If a question ID is absent from ``questions``.
    """
    by_image: dict[str, list[str]] = defaultdict(list)
    for qid in qids:
        by_image[str(questions[qid]["imageId"])].append(qid)
    return {image_id: sorted(qs) for image_id, qs in sorted(by_image.items())}


def make_splits(
    qids: Sequence[str],
    questions: Mapping[str, dict],
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Partition questions into train/val/test by image.

    Images are shuffled with a seeded generator and cut at the fraction
    boundaries, so the question counts only approximate the requested fractions:
    images carry between one and several questions each.

    Args:
        qids: Question IDs to partition.
        questions: GQA question records keyed by question ID.
        fractions: Train/val/test fractions, summing to 1.0.
        seed: Seed for the image shuffle.

    Returns:
        Mapping from split name to sorted question IDs.

    Raises:
        ValueError: If the fractions are malformed.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be three values summing to 1.0, got {tuple(fractions)}")

    by_image = group_by_image(qids, questions)
    image_ids = np.array(sorted(by_image), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    bounds = {
        "train": image_ids[:n_train],
        "val": image_ids[n_train : n_train + n_val],
        "test": image_ids[n_train + n_val :],
    }
    splits = {
        name: sorted(q for image_id in group for q in by_image[image_id])
        for name, group in bounds.items()
    }
    for name in SPLIT_NAMES:
        logger.info("%-5s %4d images, %4d questions", name, len(bounds[name]), len(splits[name]))
    return splits


def write_splits(splits: Mapping[str, Sequence[str]], out_dir: Path) -> None:
    """Write one JSON file of question IDs per split.

    Args:
        splits: Mapping from split name to question IDs.
        out_dir: Directory to write ``{train,val,test}.json`` into.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, qids in splits.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(list(qids), indent=0) + "\n", encoding="utf-8"
        )


def read_splits(split_dir: Path) -> dict[str, list[str]]:
    """Read the committed split files.

    Args:
        split_dir: Directory containing ``{train,val,test}.json``.

    Returns:
        Mapping from split name to question IDs.

    Raises:
        FileNotFoundError: If a split file is missing.
    """
    splits = {}
    for name in SPLIT_NAMES:
        path = split_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/make_splits.py")
        splits[name] = json.loads(path.read_text(encoding="utf-8"))
    return splits


def check_no_image_leakage(
    splits: Mapping[str, Sequence[str]], questions: Mapping[str, dict]
) -> list[str]:
    """Find images appearing in more than one split.

    Args:
        splits: Mapping from split name to question IDs.
        questions: GQA question records keyed by question ID.

    Returns:
        Sorted image IDs that leak across splits; empty means the split is clean.
    """
    images_per_split = {
        name: {str(questions[qid]["imageId"]) for qid in qids} for name, qids in splits.items()
    }
    leaked: set[str] = set()
    names = list(images_per_split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            leaked |= images_per_split[a] & images_per_split[b]
    return sorted(leaked)

#!/usr/bin/env python
"""Build the static Data-tab payload for the GitHub Pages site.

Reads the local AiR-D download and writes ``site/data.json`` plus a handful of
downscaled example images. The raw dataset is not committed; this gallery is.

Example:
    python scripts/build_data_gallery.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.air_d import AiRDataset  # noqa: E402
from src.data.gaze_preprocessing import TEMPORAL_BINS, load_map_path_a  # noqa: E402
from src.data.splits import group_by_image, read_splits  # noqa: E402

LONG_EDGE = 480
SITE = Path("site")
EXAMPLES = SITE / "examples"


def _resize(image: Image.Image, long_edge: int = LONG_EDGE) -> Image.Image:
    width, height = image.size
    scale = long_edge / max(width, height)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _overlay(image: Image.Image, gaze: np.ndarray) -> Image.Image:
    """Draw a fixation density on top of an image.

    Args:
        image: RGB image.
        gaze: Non-negative fixation map, any resolution.

    Returns:
        The composited RGB image.
    """
    import cv2

    heat = cv2.resize(gaze.astype(np.float32), image.size, interpolation=cv2.INTER_CUBIC)
    peak = float(heat.max())
    heat = heat / peak if peak > 0 else heat
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    base = np.asarray(image)
    alpha = np.clip(heat * 1.6, 0, 0.72)[..., None]
    out = (base * (1 - alpha) + colour * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _pick_examples(dataset: AiRDataset, by_image: dict[str, list[str]]) -> list[dict]:
    """Choose a fixed, diverse set of questions for the gallery.

    Args:
        dataset: Open AiR-D dataset.
        by_image: Question IDs grouped by image.

    Returns:
        Records describing each example, without image paths yet.
    """
    questions = dataset.questions

    def structural(qid: str) -> str:
        return str(questions[qid]["types"]["structural"])

    def answer(qid: str) -> str:
        return str(questions[qid]["answer"]).lower()

    def acc(qid: str) -> float:
        return float(dataset.accuracy(qid) or 0)

    specs: list[tuple[str, object]] = [
        ("Everyone got it right", lambda q: acc(q) == 1.0 and structural(q) == "verify"),
        (
            "A query, not yes or no",
            lambda q: structural(q) == "query" and answer(q) not in {"yes", "no"} and acc(q) >= 0.8,
        ),
        ("Left or right", lambda q: answer(q) in {"left", "right"} and acc(q) >= 0.7),
        ("Most people missed it", lambda q: acc(q) <= 0.2),
        ("Same scene, another question", None),
        ("A comparison", lambda q: structural(q) == "compare" and acc(q) >= 0.6),
    ]
    chosen: list[str] = []
    notes: list[str] = []
    for label, predicate in specs:
        if predicate is None:
            continue
        match = next((q for q in dataset.qids if q not in chosen and predicate(q)), None)
        if match is not None:
            chosen.append(match)
            notes.append(label)

    # A second question about an image that already has one in the gallery.
    for qid in list(chosen):
        image_id = str(questions[qid]["imageId"])
        siblings = [s for s in by_image[image_id] if s != qid and s not in chosen]
        if siblings:
            chosen.append(siblings[0])
            notes.append("Same scene, another question")
            break

    records = []
    for qid, note in zip(chosen, notes, strict=True):
        record = questions[qid]
        records.append(
            {
                "qid": qid,
                "note": note,
                "question": record["question"],
                "answer": record["answer"],
                "full_answer": record.get("fullAnswer", ""),
                "image_id": str(record["imageId"]),
                "structural": record["types"]["structural"],
                "semantic": record["types"]["semantic"],
                "accuracy": dataset.accuracy(qid),
                "n_questions_on_image": len(by_image[str(record["imageId"])]),
            }
        )
    return records


def build(data_root: Path = Path("data")) -> None:
    """Write the gallery files.

    Args:
        data_root: The ``data/`` directory.
    """
    dataset = AiRDataset(data_root)
    splits = read_splits(data_root / "processed" / "splits")
    by_image = group_by_image(dataset.qids, dataset.questions)
    split_of = {qid: name for name, qids in splits.items() for qid in qids}

    accuracies = [float(dataset.accuracy(q) or 0) for q in dataset.qids]
    structural = Counter(dataset.questions[q]["types"]["structural"] for q in dataset.qids)
    semantic = Counter(dataset.questions[q]["types"]["semantic"] for q in dataset.qids)
    answers = Counter(str(dataset.questions[q]["answer"]).lower() for q in dataset.qids)
    sizes = {}
    for qid in dataset.qids:
        image_id = str(dataset.questions[qid]["imageId"])
        if image_id not in sizes:
            with Image.open(dataset.image_path(qid)) as image:
                sizes[image_id] = image.size
    widths = [s[0] for s in sizes.values()]
    heights = [s[1] for s in sizes.values()]

    payload = {
        "counts": {
            "questions": len(dataset.qids),
            "images": len(by_image),
            "images_with_several_questions": sum(1 for v in by_image.values() if len(v) > 1),
            "max_questions_per_image": max(len(v) for v in by_image.values()),
            "participants_pool": len(dataset.subjects),
            "participants_per_question": 10,
            "answer_vocabulary": len(answers),
        },
        "splits": {
            name: {
                "questions": len(qids),
                "images": len({str(dataset.questions[q]["imageId"]) for q in qids}),
            }
            for name, qids in splits.items()
        },
        "human_accuracy": {
            "mean": round(float(np.mean(accuracies)), 4),
            "std": round(float(np.std(accuracies)), 4),
            "histogram": [
                {
                    "accuracy": round(i / 10, 1),
                    "n": sum(1 for a in accuracies if abs(a - i / 10) < 1e-9),
                }
                for i in range(11)
            ],
            "all_correct": sum(1 for a in accuracies if a == 1.0),
            "all_wrong": sum(1 for a in accuracies if a == 0.0),
            "mixed": sum(1 for a in accuracies if 0 < a < 1),
        },
        "question_types": {
            "structural": structural.most_common(),
            "semantic": semantic.most_common(),
        },
        "top_answers": answers.most_common(8),
        "images": {
            "width_min": min(widths),
            "width_median": int(np.median(widths)),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_median": int(np.median(heights)),
            "height_max": max(heights),
        },
        "gaze": {
            "trial_seconds": 3,
            "bins_used": list(TEMPORAL_BINS),
            "bin_discarded": "0s",
            "map_native": 256,
            "training_grid": 16,
            "sigma_px": 9,
        },
        "examples": [],
    }

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for record in _pick_examples(dataset, by_image):
        qid = record["qid"]
        with Image.open(dataset.image_path(qid)) as image:
            rgb = _resize(image.convert("RGB"))
        gaze = load_map_path_a(dataset, qid, TEMPORAL_BINS)
        photo = EXAMPLES / f"{qid}.jpg"
        heat = EXAMPLES / f"{qid}_gaze.jpg"
        rgb.save(photo, quality=85)
        _overlay(rgb, gaze).save(heat, quality=85)
        record["split"] = split_of.get(qid, "")
        record["image"] = f"examples/{photo.name}"
        record["gaze"] = f"examples/{heat.name}"
        payload["examples"].append(record)

    (SITE / "data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SITE / 'data.json'} and {len(payload['examples'])} examples")


if __name__ == "__main__":
    build()

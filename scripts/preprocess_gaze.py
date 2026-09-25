#!/usr/bin/env python
"""Convert AiR-D gaze data into the PxP fixation grids Stage 1 trains against.

Writes ``data/processed/fixation_maps/{qid}.npy`` (and, with
``--per-subject-augment``, ``{qid}__{subject}.npy``).

Example:
    python scripts/preprocess_gaze.py --data-root data
    python scripts/preprocess_gaze.py --from-scanpaths --bins 0s,1s,2s
    python scripts/preprocess_gaze.py --validate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.air_d import AiRDataset  # noqa: E402
from src.data.gaze_preprocessing import (  # noqa: E402
    GAZE_SIGMA_PX,
    PATCH_GRID_SIZE,
    TEMPORAL_BINS,
    correlation_coefficient,
    downsample_to_grid,
    load_map_path_a,
    load_map_path_b,
    per_subject_maps,
)
from src.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger("preprocess_gaze")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--from-scanpaths",
        action="store_true",
        help="rebuild maps from raw fixations (Path B) instead of the shipped maps",
    )
    p.add_argument(
        "--bins",
        type=str,
        default=",".join(TEMPORAL_BINS),
        help="comma-separated temporal bins; the 0-1s bin is centre-biased by construction",
    )
    p.add_argument("--grid-size", type=int, default=PATCH_GRID_SIZE)
    p.add_argument("--sigma", type=float, default=GAZE_SIGMA_PX)
    p.add_argument(
        "--use-correct-only",
        action="store_true",
        help="use only the attention of participants who answered correctly",
    )
    p.add_argument(
        "--per-subject-augment",
        action="store_true",
        help="additionally write one map per participant",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="compare Path B against Path A on a sample and exit",
    )
    p.add_argument("--validate-samples", type=int, default=100)
    p.add_argument("--min-cc", type=float, default=0.9, help="failure threshold for --validate")
    return p.parse_args()


def validate(dataset: AiRDataset, args: argparse.Namespace, bins: tuple[str, ...]) -> int:
    """Check that Path B reproduces the shipped maps.

    Args:
        dataset: Open AiR-D dataset.
        args: Parsed command line.
        bins: Temporal bins to compare on.

    Returns:
        Process exit code.
    """
    rng = np.random.default_rng(42)
    qids = list(
        rng.choice(dataset.qids, min(args.validate_samples, len(dataset.qids)), replace=False)
    )
    scores = []
    for qid in tqdm(qids, desc="validate", unit="q"):
        a = downsample_to_grid(load_map_path_a(dataset, qid, bins), args.grid_size)
        b = downsample_to_grid(
            load_map_path_b(dataset, qid, bins, sigma=args.sigma), args.grid_size
        )
        scores.append(correlation_coefficient(a, b))
    mean, p10 = float(np.mean(scores)), float(np.percentile(scores, 10))
    log.info("Path B vs Path A on %d questions: CC mean=%.3f p10=%.3f", len(scores), mean, p10)
    if mean < args.min_cc:
        log.error("mean CC %.3f below threshold %.3f", mean, args.min_cc)
        return 1
    return 0


def main() -> int:
    """Run the preprocessing.

    Returns:
        Process exit code.
    """
    args = parse_args()
    setup_logging(logging.INFO)
    bins = tuple(b.strip() for b in args.bins.split(",") if b.strip())
    dataset = AiRDataset(args.data_root)

    if args.validate:
        return validate(dataset, args, bins)

    out_dir = args.data_root / "processed" / "fixation_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = "fixmaps_corr" if args.use_correct_only else "fixmaps"
    log.info(
        "Writing %s maps to %s (path %s, bins=%s, variant=%s)",
        args.grid_size,
        out_dir,
        "B" if args.from_scanpaths else "A",
        bins,
        variant,
    )

    written = skipped = augmented = 0
    for qid in tqdm(dataset.qids, desc="maps", unit="q"):
        try:
            if args.from_scanpaths:
                full = load_map_path_b(
                    dataset, qid, bins, sigma=args.sigma, correct_only=args.use_correct_only
                )
            else:
                full = load_map_path_a(dataset, qid, bins, variant=variant)
        except FileNotFoundError:
            # fixmaps_corr/incorr only exist for the 934 questions with mixed answers.
            skipped += 1
            continue
        np.save(out_dir / f"{qid}.npy", downsample_to_grid(full, args.grid_size))
        written += 1

        if args.per_subject_augment:
            for subject, subject_map in per_subject_maps(
                dataset, qid, bins, sigma=args.sigma
            ).items():
                np.save(
                    out_dir / f"{qid}__{subject}.npy",
                    downsample_to_grid(subject_map, args.grid_size),
                )
                augmented += 1

    log.info("Wrote %d maps (%d skipped, %d per-subject)", written, skipped, augmented)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

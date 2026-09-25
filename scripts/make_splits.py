#!/usr/bin/env python
"""Partition the AiR-D questions into train/val/test by image.

The resulting files are committed so results are reproducible without
redistributing any data.

Example:
    python scripts/make_splits.py --data-root data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.air_d import AiRDataset  # noqa: E402
from src.data.splits import (  # noqa: E402
    DEFAULT_FRACTIONS,
    DEFAULT_SEED,
    check_no_image_leakage,
    make_splits,
    write_splits,
)
from src.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger("make_splits")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--fractions", type=float, nargs=3, default=list(DEFAULT_FRACTIONS))
    return p.parse_args()


def main() -> int:
    """Build and write the splits.

    Returns:
        Process exit code.
    """
    args = parse_args()
    setup_logging(logging.INFO)

    dataset = AiRDataset(args.data_root)
    splits = make_splits(dataset.qids, dataset.questions, tuple(args.fractions), args.seed)

    leaked = check_no_image_leakage(splits, dataset.questions)
    if leaked:
        log.error("%d images appear in more than one split: %s", len(leaked), leaked[:5])
        return 1

    out_dir = args.data_root / "processed" / "splits"
    write_splits(splits, out_dir)
    log.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

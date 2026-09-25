#!/usr/bin/env python
"""Fetch AiR-D, the GQA balanced validation questions and the Visual Genome images.

The script fails loudly rather than leaving a partially populated ``data/raw``.
The optional AiR-D hold-out metadata is allowed to be missing.

Example:
    python scripts/download_data.py --data-root data
    python scripts/download_data.py --verify-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import DownloadError, download_all, verify  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--skip-images", action="store_true", help="do not fetch Visual Genome JPGs")
    p.add_argument("--no-holdout", action="store_true", help="skip the optional hold-out metadata")
    p.add_argument("--workers", type=int, default=16, help="parallel image downloads")
    p.add_argument("--verify-only", action="store_true", help="check an existing download")
    return p.parse_args()


def main() -> int:
    """Run the download.

    Returns:
        Process exit code.
    """
    args = parse_args()
    setup_logging(logging.INFO)
    log = logging.getLogger("download")

    if args.verify_only:
        problems = verify(args.data_root, require_images=not args.skip_images)
        for problem in problems:
            log.error("%s", problem)
        if problems:
            return 1
        log.info("Download looks complete.")
        return 0

    try:
        report = download_all(
            args.data_root,
            skip_images=args.skip_images,
            include_holdout=not args.no_holdout,
            max_workers=args.workers,
        )
    except DownloadError as exc:
        log.error("%s", exc)
        return 1

    for warning in report.warnings:
        log.warning("%s", warning)
    for entry in report.entries:
        size = entry.bytes / 1e6
        detail = f"{entry.count} files" if entry.count is not None else "file"
        log.info("%-40s %8.1f MB  %s", entry.path, size, detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

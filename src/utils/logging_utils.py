"""Logging helpers: stdlib logging setup plus an optional W&B sink.

W&B is imported lazily and every call degrades to a no-op when it is absent or
disabled, so CPU smoke tests and CI never need an account.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Logging level for the root logger.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")


def git_commit_hash(repo_root: Path | None = None) -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo.

    Args:
        repo_root: Repository root; defaults to the current working directory.

    Returns:
        The full commit SHA, suffixed with ``"-dirty"`` if the tree is modified.
    """
    cwd = str(repo_root) if repo_root else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


class RunLogger:
    """Metric sink writing to stdout, a JSONL file and optionally W&B."""

    def __init__(
        self,
        run_id: str,
        output_dir: Path,
        wandb_project: str | None = None,
        wandb_mode: str = "offline",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Create a run logger.

        Args:
            run_id: Identifier for this run; used for the output subdirectory.
            output_dir: Directory the JSONL metric log is written to.
            wandb_project: W&B project name, or ``None`` to skip W&B entirely.
            wandb_mode: One of ``"online"``, ``"offline"``, ``"disabled"``.
            config: Configuration dictionary recorded with the run.
        """
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._log = logging.getLogger("gazeshap.run")
        self._wandb = None

        if wandb_project and wandb_mode != "disabled":
            try:
                import wandb

                self._wandb = wandb.init(
                    project=wandb_project,
                    name=run_id,
                    mode=wandb_mode,
                    config=config or {},
                    dir=str(self.output_dir),
                )
            except Exception as exc:  # noqa: BLE001 - W&B must never break training
                self._log.warning("W&B disabled (%s)", exc)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Record a metric dictionary.

        Args:
            metrics: Scalar metrics keyed by name.
            step: Global step the metrics belong to.
        """
        record = {"step": step, **metrics}
        self._jsonl.write(json.dumps(record, default=str) + "\n")
        self._jsonl.flush()
        pretty = " ".join(
            f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
        )
        self._log.info("step=%s %s", step, pretty)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        """Flush and close all sinks."""
        self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()

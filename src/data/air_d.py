"""Readers for the AiR-D eye-tracking release and the GQA question file.

The on-disk structure is documented in ``data/README.md``. Two facts drive the
API here and both differ from what the upstream README suggests:

* scanpaths live in ``fix/{qid}/{subject}.mat``, one directory per question;
* ``consolidated_answers.json`` is keyed by *field* (``accuracy``, ``question``,
  ``answer`` and one key per participant), not by question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.io import loadmat

TRIAL_DURATION_MS = 3000
BIN_DURATION_MS = 1000
AGGREGATED_VARIANTS = ("fixmaps", "fixmaps_corr", "fixmaps_incorr")


@dataclass(frozen=True)
class Scanpath:
    """One participant's fixation sequence for one question.

    Attributes:
        qid: GQA question ID.
        subject: Anonymised participant code, e.g. ``"AL"``.
        xy: ``(n_fixations, 2)`` fixation locations in original image pixels.
        t: ``(n_fixations, 2)`` fixation start and end times in milliseconds.
    """

    qid: str
    subject: str
    xy: np.ndarray
    t: np.ndarray

    @property
    def durations_ms(self) -> np.ndarray:
        """Fixation durations in milliseconds."""
        return (self.t[:, 1] - self.t[:, 0]).astype(np.float64)

    def in_bins(self, bins: tuple[str, ...]) -> Scanpath:
        """Restrict the scanpath to the given one-second temporal bins.

        A fixation belongs to bin ``k`` if its *start* falls inside
        ``[k*1000, (k+1)*1000)``. Bins are named by their starting second, so
        ``"1s"`` selects the 1-2 s window, matching the AiR-D ``temporal_maps``
        naming.

        Args:
            bins: Bin names such as ``("1s", "2s")``.

        Returns:
            A new scanpath containing only the selected fixations.
        """
        if not bins:
            return self
        seconds = {int(b.removesuffix("s")) for b in bins}
        starts = self.t[:, 0].astype(np.int64)
        keep = np.isin(starts // BIN_DURATION_MS, sorted(seconds))
        return Scanpath(self.qid, self.subject, self.xy[keep], self.t[keep])


class AiRDataset:
    """Access to the AiR-D maps, scanpaths and responses, joined to GQA."""

    def __init__(self, data_root: Path | str) -> None:
        """Open a downloaded AiR-D dataset.

        Args:
            data_root: The ``data/`` directory.

        Raises:
            FileNotFoundError: If the expected directories are missing.
        """
        self.root = Path(data_root)
        self.raw = self.root / "raw"
        self.air_d = self.raw / "air_d"
        self.aggregated_maps = self.air_d / "aggregated_maps"
        self.temporal_maps = self.air_d / "temporal_maps"
        self.scanpath_dir = self.air_d / "scanpaths" / "fix"
        self.answers_path = self.air_d / "scanpaths" / "consolidated_answers.json"
        self.images_dir = self.raw / "images"
        self.questions_path = self.raw / "gqa" / "val_balanced_questions.json"

        for required in (self.aggregated_maps, self.questions_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{required} not found. Run scripts/download_data.py first."
                )

    # --- Metadata -----------------------------------------------------------

    @cached_property
    def responses(self) -> dict[str, dict]:
        """The parsed ``consolidated_answers.json``, keyed by field."""
        if not self.answers_path.exists():
            return {}
        return json.loads(self.answers_path.read_text(encoding="utf-8"))

    @cached_property
    def questions(self) -> dict[str, dict]:
        """The GQA balanced validation questions, keyed by question ID."""
        return json.loads(self.questions_path.read_text(encoding="utf-8"))

    @cached_property
    def subjects(self) -> list[str]:
        """The participant codes present in the response file."""
        return sorted(k for k in self.responses if k not in {"accuracy", "question", "answer"})

    @cached_property
    def qids(self) -> list[str]:
        """Question IDs usable for training.

        ``fixmaps`` ships more maps than there are complete trials, so this is
        the intersection of the aggregated maps, the temporal bins, the recorded
        responses and the GQA question file.

        Returns:
            Sorted question IDs.
        """
        qids = self._map_qids(self.aggregated_maps / "fixmaps")
        for bin_name in ("1s", "2s"):
            binned = self._map_qids(self.temporal_maps / f"fixmaps_{bin_name}")
            if binned:
                qids &= binned
        if self.responses:
            qids &= set(self.responses.get("question", {}))
        qids &= set(self.questions)
        return sorted(qids)

    @staticmethod
    def _map_qids(directory: Path) -> set[str]:
        """Collect question IDs from map filenames.

        Args:
            directory: Directory of per-question maps.

        Returns:
            Set of question IDs, empty if the directory is absent.
        """
        if not directory.is_dir():
            return set()
        return {p.stem for p in directory.iterdir() if p.suffix.lower() in {".png", ".npy"}}

    def accuracy(self, qid: str) -> float | None:
        """Human accuracy for a question, in steps of 0.1 over ten participants.

        Args:
            qid: Question ID.

        Returns:
            The accuracy, or ``None`` if unrecorded.
        """
        return self.responses.get("accuracy", {}).get(qid)

    def question(self, qid: str) -> dict:
        """The GQA record for a question.

        Args:
            qid: Question ID.

        Returns:
            The GQA question dictionary.

        Raises:
            KeyError: If the question is not in the GQA file.
        """
        return self.questions[qid]

    def image_path(self, qid: str) -> Path:
        """Path to the image a question refers to.

        Args:
            qid: Question ID.

        Returns:
            Path to the JPG.
        """
        return self.images_dir / f"{self.questions[qid]['imageId']}.jpg"

    def image_size(self, qid: str) -> tuple[int, int]:
        """Pixel size of the image a question refers to.

        Args:
            qid: Question ID.

        Returns:
            ``(width, height)``.

        Raises:
            FileNotFoundError: If the image has not been downloaded.
        """
        from PIL import Image

        path = self.image_path(qid)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/download_data.py")
        with Image.open(path) as im:
            return im.size

    # --- Maps ---------------------------------------------------------------

    def aggregated_map_path(self, qid: str, variant: str = "fixmaps") -> Path:
        """Path to an aggregated saliency map.

        Args:
            qid: Question ID.
            variant: One of ``fixmaps``, ``fixmaps_corr``, ``fixmaps_incorr``.

        Returns:
            Path to the PNG.

        Raises:
            ValueError: If ``variant`` is not a known map variant.
        """
        if variant not in AGGREGATED_VARIANTS:
            raise ValueError(
                f"unknown map variant {variant!r}, expected one of {AGGREGATED_VARIANTS}"
            )
        return self.aggregated_maps / variant / f"{qid}.png"

    def temporal_map_path(self, qid: str, bin_name: str, variant: str = "fixmaps") -> Path:
        """Path to a temporally binned saliency map.

        Args:
            qid: Question ID.
            bin_name: Bin name such as ``"1s"``.
            variant: Base variant, see :meth:`aggregated_map_path`.

        Returns:
            Path to the PNG.
        """
        return self.temporal_maps / f"{variant}_{bin_name}" / f"{qid}.png"

    def load_map(self, path: Path) -> np.ndarray:
        """Load a saliency map as float in ``[0, 1]``.

        Args:
            path: Path to a PNG or NPY map.

        Returns:
            ``(H, W)`` float32 array.

        Raises:
            FileNotFoundError: If the map does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
        else:
            from PIL import Image

            with Image.open(path) as im:
                arr = np.asarray(im.convert("L"), dtype=np.float32)
        return arr / 255.0 if arr.max() > 1.0 else arr

    # --- Scanpaths ----------------------------------------------------------

    def scanpath_subjects(self, qid: str) -> list[str]:
        """Participants with a recorded scanpath for a question.

        Args:
            qid: Question ID.

        Returns:
            Sorted participant codes; empty if the question has no scanpaths.
        """
        directory = self.scanpath_dir / qid
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.mat"))

    def load_scanpath(self, qid: str, subject: str) -> Scanpath:
        """Load one participant's fixation sequence.

        Args:
            qid: Question ID.
            subject: Participant code.

        Returns:
            The scanpath. Empty trials yield zero-length arrays.

        Raises:
            FileNotFoundError: If the ``.mat`` file does not exist.
        """
        path = self.scanpath_dir / qid / f"{subject}.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        mat = loadmat(path)
        # 313 of 14,220 trials recorded nothing usable: 308 ship as (0, 0) and
        # five have mismatched xy/t shapes. All are treated as empty trials.
        xy = np.asarray(mat["xy"], dtype=np.float64)
        t = np.asarray(mat["t"], dtype=np.int64)
        if xy.ndim != 2 or xy.shape[1] != 2 or t.shape[:1] != xy.shape[:1] or t.shape[-1] != 2:
            return Scanpath(qid, subject, np.zeros((0, 2)), np.zeros((0, 2), dtype=np.int64))
        return Scanpath(qid=qid, subject=subject, xy=xy, t=t)

    def load_scanpaths(self, qid: str) -> list[Scanpath]:
        """Load every participant's scanpath for a question.

        Args:
            qid: Question ID.

        Returns:
            One scanpath per participant with recorded data.
        """
        return [self.load_scanpath(qid, s) for s in self.scanpath_subjects(qid)]

    def subject_was_correct(self, qid: str, subject: str) -> bool | None:
        """Whether a participant answered a question correctly.

        Args:
            qid: Question ID.
            subject: Participant code.

        Returns:
            ``True``/``False``, or ``None`` if the participant did not attempt
            the question (recorded as ``nan`` upstream).
        """
        given = self.responses.get(subject, {}).get(qid)
        truth = self.responses.get("answer", {}).get(qid)
        if given is None or truth is None:
            return None
        if isinstance(given, float) and np.isnan(given):
            return None
        return str(given).strip().lower() == str(truth).strip().lower()

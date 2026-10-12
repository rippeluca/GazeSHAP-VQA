"""The torch Dataset joining images, questions and human fixation maps."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.air_d import AiRDataset
from src.data.gaze_preprocessing import (
    PATCH_GRID_SIZE,
    TEMPORAL_BINS,
    downsample_to_grid,
    load_map_path_a,
)
from src.models.reasoning_head import stage1_prompt

if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = logging.getLogger(__name__)


class GazeVQADataset(Dataset):
    """AiR-D questions with their images and human fixation grids.

    Each item carries everything Stage 1 needs: the Qwen patch tensor, the
    tokenised prompt ending in ``<answer>`` plus the teacher-forced answer, the
    answer token IDs the attribution sums over,
    and the human gaze grid to align against.
    """

    def __init__(
        self,
        data_root: Path | str,
        processor: ProcessorMixin,
        qids: Sequence[str] | None = None,
        patch_grid_size: int = PATCH_GRID_SIZE,
        temporal_bins: Sequence[str] = TEMPORAL_BINS,
        use_correct_only: bool = False,
        per_subject_augment: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        """Build the dataset.

        Args:
            data_root: The ``data/`` directory.
            processor: A Qwen3-VL processor (tokeniser plus image processor).
            qids: Question IDs to serve; defaults to every usable question.
            patch_grid_size: Edge length P of the gaze grid.
            temporal_bins: Gaze bins to use, e.g. ``("1s", "2s")``.
            use_correct_only: Use the correct-answer-only maps where available.
            per_subject_augment: Expand each question into one item per
                participant, using that participant's own fixation map.
            cache_dir: Directory of precomputed ``{qid}.npy`` grids; defaults to
                ``data/processed/fixation_maps``. Missing entries are computed
                on the fly.

        Raises:
            ValueError: If no question IDs remain after filtering.
        """
        self.air_d = AiRDataset(data_root)
        self.processor = processor
        self.patch_grid_size = patch_grid_size
        self.temporal_bins = tuple(temporal_bins)
        self.variant = "fixmaps_corr" if use_correct_only else "fixmaps"
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path(data_root) / "processed" / "fixation_maps"
        )

        base_qids = list(qids) if qids is not None else self.air_d.qids
        unknown = [q for q in base_qids if q not in set(self.air_d.qids)]
        if unknown:
            logger.warning("dropping %d question ids without usable gaze data", len(unknown))
        base_qids = [q for q in base_qids if q not in set(unknown)]
        if not base_qids:
            raise ValueError("no usable question ids")

        # (qid, subject) where subject is None for the aggregated map.
        self.items: list[tuple[str, str | None]] = []
        for qid in base_qids:
            self.items.append((qid, None))
            if per_subject_augment:
                self.items.extend((qid, s) for s in self.air_d.scanpath_subjects(qid))

    def __len__(self) -> int:
        """Number of items."""
        return len(self.items)

    def _gaze_grid(self, qid: str, subject: str | None) -> np.ndarray:
        """Load or compute the PxP fixation grid for one item.

        Args:
            qid: Question ID.
            subject: Participant code, or ``None`` for the aggregated map.

        Returns:
            ``(P, P)`` float32 grid summing to 1.0.
        """
        if subject is None:
            cached = self.cache_dir / f"{qid}.npy"
            if cached.exists():
                grid = np.load(cached).astype(np.float32)
                if grid.shape == (self.patch_grid_size, self.patch_grid_size):
                    return grid
            full = load_map_path_a(self.air_d, qid, self.temporal_bins, self.variant)
            return downsample_to_grid(full, self.patch_grid_size)

        cached = self.cache_dir / f"{qid}__{subject}.npy"
        if cached.exists():
            grid = np.load(cached).astype(np.float32)
            if grid.shape == (self.patch_grid_size, self.patch_grid_size):
                return grid
        from src.data.gaze_preprocessing import fixation_map_from_scanpaths

        scanpath = self.air_d.load_scanpath(qid, subject).in_bins(self.temporal_bins)
        full = fixation_map_from_scanpaths([scanpath], self.air_d.image_size(qid))
        return downsample_to_grid(full, self.patch_grid_size)

    def _answer_token_ids(self, answer: str) -> torch.Tensor:
        """Token IDs of the ground-truth answer, which attribution sums over.

        Args:
            answer: Ground-truth answer string.

        Returns:
            One-dimensional token IDs, without special tokens.

        Raises:
            ValueError: If the answer tokenises to nothing.
        """
        ids = self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"answer {answer!r} produced no tokens")
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index: int) -> dict:
        """Return one training item.

        Args:
            index: Item index.

        Returns:
            The dictionary contract from idea.md 4.6, with ``pixel_values`` in
            Qwen's flat patch layout and the accompanying ``image_grid_thw``.
        """
        qid, subject = self.items[index]
        record = self.air_d.question(qid)
        answer = str(record["answer"])
        answer_ids = self._answer_token_ids(answer)

        with Image.open(self.air_d.image_path(qid)) as im:
            image = im.convert("RGB")
            encoded = self.processor(
                text=[stage1_prompt(self.processor, str(record["question"]))],
                images=[image],
                return_tensors="pt",
            )

        # transformers >= 5 requires mm_token_type_ids for M-RoPE; older
        # processors do not emit it, so it is derived from the image token id.
        if "mm_token_type_ids" in encoded:
            mm_token_type_ids = encoded["mm_token_type_ids"][0]
        else:
            image_token_id = getattr(self.processor, "image_token_id", None)
            mm_token_type_ids = (
                (encoded["input_ids"][0] == image_token_id).long()
                if image_token_id is not None
                else torch.zeros_like(encoded["input_ids"][0])
            )

        input_ids = torch.cat([encoded["input_ids"][0], answer_ids])
        attention_mask = torch.cat([encoded["attention_mask"][0], torch.ones_like(answer_ids)])
        mm_token_type_ids = torch.cat([mm_token_type_ids, torch.zeros_like(answer_ids)])

        return {
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"][0],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "gaze_map": torch.from_numpy(self._gaze_grid(qid, subject)),
            "target_token_ids": answer_ids,
            "answer_str": answer,
            "question": str(record["question"]),
            "qid": qid,
            "image_id": str(record["imageId"]),
            "subject": subject or "",
        }

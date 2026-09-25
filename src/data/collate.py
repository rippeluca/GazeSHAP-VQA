"""Batching for variable-resolution Qwen inputs.

Two things vary per sample and neither can be stacked naively:

* ``pixel_values`` has one row per image patch, and the patch count depends on
  the image's aspect ratio. Qwen expects the patches of all images in a batch
  *concatenated* along dim 0, with ``image_grid_thw`` of shape ``(B, 3)``
  describing how to split them again.
* ``input_ids`` differ in length. Padding is applied on the **left** so that the
  final position is the same for every sample, which is what the Stage 1
  attribution target relies on.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

TEXT_KEYS = ("answer_str", "question", "qid", "image_id", "subject")


def collate_gaze_vqa(batch: Sequence[dict], pad_token_id: int = 0) -> dict:
    """Collate items from :class:`~src.data.dataset.GazeVQADataset`.

    Args:
        batch: Items to combine.
        pad_token_id: Token used for left padding.

    Returns:
        A batch dictionary with concatenated patches, left-padded token
        sequences and stacked gaze grids.

    Raises:
        ValueError: If the batch is empty.
    """
    if not batch:
        raise ValueError("empty batch")

    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    mm_token_type_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[i, max_len - length :] = item["input_ids"]
        attention_mask[i, max_len - length :] = item["attention_mask"]
        mm_token_type_ids[i, max_len - length :] = item["mm_token_type_ids"]

    collated = {
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.stack([item["image_grid_thw"] for item in batch], dim=0),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "gaze_map": torch.stack([item["gaze_map"] for item in batch], dim=0),
        "target_token_id": torch.stack([item["target_token_id"] for item in batch], dim=0),
    }
    for key in TEXT_KEYS:
        collated[key] = [item[key] for item in batch]
    return collated

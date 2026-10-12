from __future__ import annotations

import pytest
import torch

from src.data.collate import collate_gaze_vqa
from src.data.dataset import GazeVQADataset
from tests.fixtures import MERGE_SIZE, PATCH_DIM, FakeProcessor, make_fake_dataset_root


@pytest.fixture
def dataset(tmp_path):
    make_fake_dataset_root(tmp_path, n_images=4, questions_per_image=2)
    return GazeVQADataset(tmp_path, FakeProcessor(), patch_grid_size=8)


def test_len(dataset):
    assert len(dataset) == 8


def test_item_contract(dataset):
    item = dataset[0]
    _, grid_h, grid_w = item["image_grid_thw"].tolist()
    expected = {
        "pixel_values",
        "image_grid_thw",
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "gaze_map",
        "target_token_ids",
        "answer_str",
        "question",
        "qid",
        "image_id",
        "subject",
    }
    assert set(item) == expected

    assert item["pixel_values"].shape == (grid_h * grid_w, PATCH_DIM)
    assert item["input_ids"].dtype == torch.long
    assert item["input_ids"].shape == item["attention_mask"].shape
    assert item["input_ids"].shape == item["mm_token_type_ids"].shape
    assert item["gaze_map"].shape == (8, 8)
    assert item["gaze_map"].dtype == torch.float32
    assert item["target_token_ids"].dtype == torch.long
    assert item["input_ids"][-item["target_token_ids"].shape[0] :].tolist() == item[
        "target_token_ids"
    ].tolist()
    assert isinstance(item["qid"], str)


def test_gaze_map_is_a_distribution(dataset):
    for i in range(len(dataset)):
        gaze = dataset[i]["gaze_map"]
        assert torch.isclose(gaze.sum(), torch.tensor(1.0), atol=1e-5)
        assert (gaze >= 0).all()


def test_mm_token_type_ids_mark_the_image_block(dataset):
    item = dataset[0]
    _, grid_h, grid_w = item["image_grid_thw"].tolist()
    n_image_tokens = (grid_h // MERGE_SIZE) * (grid_w // MERGE_SIZE)
    assert int(item["mm_token_type_ids"].sum()) == n_image_tokens


def test_unknown_qids_are_dropped(tmp_path):
    make_fake_dataset_root(tmp_path, n_images=2, questions_per_image=1)
    ds = GazeVQADataset(tmp_path, FakeProcessor(), qids=["0000", "does_not_exist"])
    assert len(ds) == 1


def test_empty_qid_list_raises(tmp_path):
    make_fake_dataset_root(tmp_path, n_images=2, questions_per_image=1)
    with pytest.raises(ValueError, match="no usable question ids"):
        GazeVQADataset(tmp_path, FakeProcessor(), qids=["nope"])


def test_cached_grid_is_used(tmp_path):
    import numpy as np

    make_fake_dataset_root(tmp_path, n_images=1, questions_per_image=1)
    cache = tmp_path / "processed" / "fixation_maps"
    cache.mkdir(parents=True)
    marker = np.zeros((8, 8), dtype=np.float32)
    marker[3, 5] = 1.0
    np.save(cache / "0000.npy", marker)

    ds = GazeVQADataset(tmp_path, FakeProcessor(), patch_grid_size=8)
    assert ds[0]["gaze_map"][3, 5] == pytest.approx(1.0)


def test_cached_grid_of_wrong_size_is_recomputed(tmp_path):
    import numpy as np

    make_fake_dataset_root(tmp_path, n_images=1, questions_per_image=1)
    cache = tmp_path / "processed" / "fixation_maps"
    cache.mkdir(parents=True)
    np.save(cache / "0000.npy", np.zeros((16, 16), dtype=np.float32))

    ds = GazeVQADataset(tmp_path, FakeProcessor(), patch_grid_size=8)
    assert ds[0]["gaze_map"].shape == (8, 8)


# --- collate ----------------------------------------------------------------


def test_collate_concatenates_patches_and_stacks_grids(dataset):
    a, b = dataset[0], dataset[1]
    batch = collate_gaze_vqa([a, b])
    assert batch["pixel_values"].shape == (
        a["pixel_values"].shape[0] + b["pixel_values"].shape[0],
        PATCH_DIM,
    )
    assert batch["image_grid_thw"].shape == (2, 3)
    assert batch["gaze_map"].shape == (2, 8, 8)
    assert batch["target_token_ids"].shape[0] == 2
    assert batch["answer_lengths"].tolist() == [
        a["target_token_ids"].shape[0],
        b["target_token_ids"].shape[0],
    ]
    assert batch["qid"] == [dataset[0]["qid"], dataset[1]["qid"]]


def test_collate_left_pads(dataset):
    a, b = dataset[0], dataset[1]
    b = dict(b)
    b["input_ids"] = a["input_ids"][:-3]
    b["attention_mask"] = a["attention_mask"][:-3]
    b["mm_token_type_ids"] = a["mm_token_type_ids"][:-3]

    batch = collate_gaze_vqa([a, b], pad_token_id=0)
    length = batch["input_ids"].shape[1]
    pad = length - b["input_ids"].shape[0]
    assert length == a["input_ids"].shape[0]
    # The shorter sequence is padded on the left, so its real tokens end flush right.
    assert batch["attention_mask"][1, :pad].tolist() == [0] * pad
    assert batch["attention_mask"][1, pad:].tolist() == [1] * (length - pad)
    assert batch["input_ids"][1, -1] == b["input_ids"][-1]
    assert batch["input_ids"][0, -1] == a["input_ids"][-1]


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        collate_gaze_vqa([])

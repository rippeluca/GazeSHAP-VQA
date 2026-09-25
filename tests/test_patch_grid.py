"""The un-permutation is validated against the real processor, not a restatement
of the same index arithmetic: a brightness ramp is pushed through
``Qwen2VLImageProcessor`` and the recovered grid must stay monotonic along the
ramp axis.
"""

from __future__ import annotations

import pytest
import torch

from src.models.patch_grid import flatten_patch_grid, pool_to_square_grid, unflatten_patch_grid
from tests.fixtures import MERGE_SIZE, image_processor, ramp_image


def _per_patch_brightness(encoded) -> tuple[torch.Tensor, torch.Tensor]:
    return encoded["pixel_values"].float().mean(dim=-1), encoded["image_grid_thw"]


def test_horizontal_ramp_is_monotonic_after_unflattening():
    encoded = image_processor()(images=ramp_image(128, 64, "x"), return_tensors="pt")
    flat, thw = _per_patch_brightness(encoded)
    grid = unflatten_patch_grid(flat, thw)[0]

    for row in grid:
        assert torch.all(row[1:] >= row[:-1]), row
    # Rows are identical because the ramp does not vary vertically.
    assert torch.allclose(grid[0], grid[-1], atol=1e-4)


def test_naive_reshape_would_be_wrong():
    """Guards the reason this module exists."""
    encoded = image_processor()(images=ramp_image(128, 64, "x"), return_tensors="pt")
    flat, thw = _per_patch_brightness(encoded)
    _, gh, gw = thw[0].tolist()
    naive = flat.view(gh, gw)
    assert not torch.all(naive[0][1:] >= naive[0][:-1])


def test_vertical_ramp_is_monotonic_after_unflattening():
    encoded = image_processor()(images=ramp_image(128, 128, "y"), return_tensors="pt")
    flat, thw = _per_patch_brightness(encoded)
    grid = unflatten_patch_grid(flat, thw)[0]

    for col in grid.T:
        assert torch.all(col[1:] >= col[:-1]), col
    assert torch.allclose(grid[:, 0], grid[:, -1], atol=1e-4)


def test_grid_shape_matches_metadata():
    encoded = image_processor()(images=ramp_image(192, 96, "x"), return_tensors="pt")
    flat, thw = _per_patch_brightness(encoded)
    _, gh, gw = thw[0].tolist()
    assert unflatten_patch_grid(flat, thw)[0].shape == (gh, gw)


def test_batch_of_differently_shaped_images():
    proc = image_processor()
    a = proc(images=ramp_image(128, 64, "x"), return_tensors="pt")
    b = proc(images=ramp_image(96, 96, "y"), return_tensors="pt")
    flat = torch.cat([a["pixel_values"], b["pixel_values"]]).float().mean(dim=-1)
    thw = torch.cat([a["image_grid_thw"], b["image_grid_thw"]])

    grids = unflatten_patch_grid(flat, thw)
    assert len(grids) == 2
    assert grids[0].shape == tuple(a["image_grid_thw"][0][1:].tolist())
    assert grids[1].shape == tuple(b["image_grid_thw"][0][1:].tolist())
    assert torch.all(grids[0][0][1:] >= grids[0][0][:-1])
    assert torch.all(grids[1][:, 0][1:] >= grids[1][:, 0][:-1])


def test_round_trip_flatten_unflatten():
    gh, gw = 6, 8
    grid = torch.arange(gh * gw, dtype=torch.float32).reshape(gh, gw)
    flat = flatten_patch_grid(grid)
    thw = torch.tensor([[1, gh, gw]])
    assert torch.equal(unflatten_patch_grid(flat, thw)[0], grid)


def test_round_trip_preserves_trailing_dims():
    gh, gw, k = 4, 4, 3
    grid = torch.randn(gh, gw, k)
    flat = flatten_patch_grid(grid)
    assert flat.shape == (gh * gw, k)
    assert torch.equal(unflatten_patch_grid(flat, torch.tensor([[1, gh, gw]]))[0], grid)


def test_patch_count_mismatch_raises():
    with pytest.raises(ValueError, match="patches"):
        unflatten_patch_grid(torch.zeros(10), torch.tensor([[1, 4, 4]]))


def test_odd_grid_raises():
    with pytest.raises(ValueError, match="merge_size"):
        unflatten_patch_grid(torch.zeros(15), torch.tensor([[1, 3, 5]]))


def test_merge_block_ordering_is_row_major_inside_the_block():
    """Explicit statement of the layout, independent of the processor."""
    gh = gw = MERGE_SIZE * 2
    grid = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ]
    )
    flat = flatten_patch_grid(grid)
    # First merge block covers rows 0-1, cols 0-1 of the grid.
    assert flat[:4].tolist() == [0.0, 1.0, 4.0, 5.0]
    # Second block covers rows 0-1, cols 2-3.
    assert flat[4:8].tolist() == [2.0, 3.0, 6.0, 7.0]
    assert unflatten_patch_grid(flat, torch.tensor([[1, gh, gw]]))[0].tolist() == grid.tolist()


# --- pooling ----------------------------------------------------------------


def test_pool_downsamples_and_preserves_mass_ordering():
    grid = torch.zeros(16, 16)
    grid[0, 15] = 1.0
    pooled = pool_to_square_grid(grid, 4)
    assert pooled.shape == (4, 4)
    assert int(pooled.argmax()) == 3


def test_pool_upsamples_when_grid_is_smaller():
    grid = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    pooled = pool_to_square_grid(grid, 8)
    assert pooled.shape == (8, 8)
    assert pooled[0, 0] < pooled[-1, -1]


def test_pool_is_identity_for_matching_size():
    grid = torch.randn(8, 8)
    assert torch.allclose(pool_to_square_grid(grid, 8), grid, atol=1e-6)

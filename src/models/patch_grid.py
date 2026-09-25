"""Mapping between Qwen's flat patch sequence and a spatial patch grid.

The image processor does not hand the model an image tensor. It emits

    pixel_values: (sum_i grid_h_i * grid_w_i, C * temporal_patch * p * p)

and the patch order is **merge-window interleaved**. From
``image_processing_qwen2_vl.py``:

    patches = images.reshape(B, C, gh // m, m, p, gw // m, m, p)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)

so the flat index runs over ``(gh // m, gw // m, m, m)`` — merge block row, merge
block column, then the row and column *inside* the block. Reading the sequence
as ``view(gh, gw)`` scrambles the map; the functions here invert the permutation
instead.
"""

from __future__ import annotations

import torch

MERGE_SIZE = 2


def unflatten_patch_grid(
    flat: torch.Tensor, image_grid_thw: torch.Tensor, merge_size: int = MERGE_SIZE
) -> list[torch.Tensor]:
    """Turn a flat per-patch vector into one spatial grid per image.

    Args:
        flat: ``(n_patches,)`` or ``(n_patches, k)`` values, one row per patch,
            in the order the processor emitted them for the whole batch.
        image_grid_thw: ``(B, 3)`` or ``(3,)`` grid metadata ``(t, gh, gw)``.
        merge_size: Spatial merge window of the vision tower.

    Returns:
        One ``(gh, gw)`` tensor per image (trailing dims of ``flat`` preserved).

    Raises:
        ValueError: If the patch count does not match the grid metadata, or a
            grid side is not divisible by ``merge_size``.
    """
    thw = image_grid_thw.reshape(-1, 3)
    counts = [int(t * gh * gw) for t, gh, gw in thw.tolist()]
    if flat.shape[0] != sum(counts):
        raise ValueError(
            f"flat has {flat.shape[0]} patches but image_grid_thw implies {sum(counts)}"
        )

    grids: list[torch.Tensor] = []
    offset = 0
    tail = flat.shape[1:]
    for (t, gh, gw), count in zip(thw.tolist(), counts, strict=True):
        if gh % merge_size or gw % merge_size:
            raise ValueError(f"grid ({gh}, {gw}) not divisible by merge_size {merge_size}")
        chunk = flat[offset : offset + count]
        offset += count
        # (t, gh/m, gw/m, m, m, *tail) -> (t, gh/m, m, gw/m, m, *tail)
        block = chunk.reshape(t, gh // merge_size, gw // merge_size, merge_size, merge_size, *tail)
        block = block.permute(0, 1, 3, 2, 4, *range(5, 5 + len(tail)))
        grid = block.reshape(t, gh, gw, *tail)
        # A still image has t == 1; average any temporal frames.
        grids.append(grid.mean(dim=0) if t > 1 else grid[0])
    return grids


def flatten_patch_grid(grid: torch.Tensor, merge_size: int = MERGE_SIZE) -> torch.Tensor:
    """Inverse of :func:`unflatten_patch_grid` for a single image.

    Args:
        grid: ``(gh, gw)`` spatial grid.
        merge_size: Spatial merge window of the vision tower.

    Returns:
        ``(gh * gw,)`` values in the processor's patch order.

    Raises:
        ValueError: If a grid side is not divisible by ``merge_size``.
    """
    gh, gw = grid.shape[:2]
    if gh % merge_size or gw % merge_size:
        raise ValueError(f"grid ({gh}, {gw}) not divisible by merge_size {merge_size}")
    tail = grid.shape[2:]
    block = grid.reshape(gh // merge_size, merge_size, gw // merge_size, merge_size, *tail)
    block = block.permute(0, 2, 1, 3, *range(4, 4 + len(tail)))
    return block.reshape(gh * gw, *tail)


def pool_to_square_grid(grid: torch.Tensor, size: int) -> torch.Tensor:
    """Average-pool a ``(gh, gw)`` grid to ``(size, size)``.

    Bilinear interpolation is used when the grid is smaller than the target,
    because adaptive average pooling cannot upsample.

    Args:
        grid: ``(gh, gw)`` grid.
        size: Edge length of the square output.

    Returns:
        ``(size, size)`` tensor.
    """
    x = grid.unsqueeze(0).unsqueeze(0)
    if grid.shape[0] >= size and grid.shape[1] >= size:
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, size)
    else:
        pooled = torch.nn.functional.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False
        )
    return pooled.squeeze(0).squeeze(0)

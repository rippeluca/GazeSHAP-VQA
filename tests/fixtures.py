"""Synthetic fixtures: a tiny real Qwen3-VL model and a stand-in processor.

The model is a genuine ``Qwen3VLForConditionalGeneration`` built from a small
config, so tests exercise the real vision tower, DeepStack routing and attention
path without touching the Hugging Face hub.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
PATCH_DIM = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE

IMAGE_TOKEN_ID = 90
VIDEO_TOKEN_ID = 91
VISION_START_TOKEN_ID = 92
VISION_END_TOKEN_ID = 93
VOCAB_SIZE = 99


def tiny_config(attn_implementation: str = "eager") -> Qwen3VLConfig:
    """Build a small Qwen3-VL config with the real geometry.

    Args:
        attn_implementation: Attention kernel to request.

    Returns:
        The configuration.
    """
    vision = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=32,
        patch_size=PATCH_SIZE,
        spatial_merge_size=MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1],
    )
    text = Qwen3VLTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    config = Qwen3VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
    )
    config._attn_implementation = attn_implementation
    return config


def tiny_model(attn_implementation: str = "eager", seed: int = 0):
    """Instantiate a tiny Qwen3-VL model with random weights.

    Args:
        attn_implementation: Attention kernel to request.
        seed: Seed for weight initialisation.

    Returns:
        The model in eval mode.
    """
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(tiny_config(attn_implementation))
    return model.eval()


def image_processor(min_tokens: int = 4, max_tokens: int = 16) -> Qwen2VLImageProcessor:
    """Build the real Qwen image processor without hitting the hub.

    Args:
        min_tokens: Lower bound on visual tokens, in 32x32-pixel units.
        max_tokens: Upper bound on visual tokens, in 32x32-pixel units.

    Returns:
        The configured image processor.
    """
    unit = PATCH_SIZE * MERGE_SIZE
    return Qwen2VLImageProcessor(
        patch_size=PATCH_SIZE,
        merge_size=MERGE_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        min_pixels=min_tokens * unit * unit,
        max_pixels=max_tokens * unit * unit,
    )


def ramp_image(width: int = 128, height: int = 64, axis: str = "x") -> Image.Image:
    """Create an image whose brightness increases along one axis.

    Useful for checking that a recovered patch grid preserves spatial order.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        axis: ``"x"`` for a horizontal ramp, ``"y"`` for a vertical one.

    Returns:
        An RGB image.
    """
    if axis == "x":
        arr = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    else:
        arr = np.tile(np.linspace(0, 255, height, dtype=np.uint8)[:, None], (1, width))
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def dummy_batch(
    grid_h: int = 4,
    grid_w: int = 4,
    batch_size: int = 1,
    n_text_tokens: int = 3,
    seed: int = 0,
) -> dict:
    """Build a model-ready batch without a processor.

    Args:
        grid_h: Patch rows per image.
        grid_w: Patch columns per image.
        batch_size: Number of images, one per sequence.
        n_text_tokens: Text tokens appended after the image block.
        seed: Seed for the random patch values.

    Returns:
        Keyword arguments accepted by the tiny model, plus ``gaze_map``.
    """
    generator = torch.Generator().manual_seed(seed)
    n_patches = grid_h * grid_w
    n_image_tokens = (grid_h // MERGE_SIZE) * (grid_w // MERGE_SIZE)

    pixel_values = torch.randn(batch_size * n_patches, PATCH_DIM, generator=generator)
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]] * batch_size, dtype=torch.long)

    prefix = [1, VISION_START_TOKEN_ID]
    suffix = [VISION_END_TOKEN_ID, *range(2, 2 + n_text_tokens)]
    row = prefix + [IMAGE_TOKEN_ID] * n_image_tokens + suffix
    input_ids = torch.tensor([row] * batch_size, dtype=torch.long)

    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[:, len(prefix) : len(prefix) + n_image_tokens] = 1

    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "mm_token_type_ids": mm_token_type_ids,
        "target_token_ids": torch.full((batch_size, 1), 7, dtype=torch.long),
        "answer_lengths": torch.ones(batch_size, dtype=torch.long),
    }


class FakeTokenizer:
    """Minimal tokeniser: one token per character code, deterministic."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        """Tokenise a string.

        Args:
            text: Input text.
            add_special_tokens: Ignored; present for API compatibility.

        Returns:
            A dict with ``input_ids``.
        """
        ids = [(ord(c) % (VOCAB_SIZE - 10)) + 5 for c in text[:8]] or [5]
        return {"input_ids": ids}


class FakeProcessor:
    """Stand-in processor: the real image processor plus a toy tokeniser.

    The image path must be real, because a synthetic all-zero ``pixel_values``
    makes expected gradients vanish identically (the attribution is
    ``grad * (x - baseline)``, and the black baseline is zero), which would make
    the training smoke test pass while learning nothing.
    """

    image_token_id = IMAGE_TOKEN_ID

    def __init__(self, min_tokens: int = 4, max_tokens: int = 16) -> None:
        """Create the processor.

        Args:
            min_tokens: Lower bound on visual tokens, in 32x32-pixel units.
            max_tokens: Upper bound on visual tokens, in 32x32-pixel units.
        """
        self.image_processor = image_processor(min_tokens, max_tokens)
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        """Render messages to a flat string.

        Args:
            messages: Chat messages.
            tokenize: Ignored.
            add_generation_prompt: Ignored.

        Returns:
            The rendered prompt.
        """
        parts = []
        for message in messages:
            for chunk in message["content"]:
                parts.append("<image>" if chunk["type"] == "image" else chunk["text"])
        return " ".join(parts)

    def __call__(self, text, images, return_tensors="pt") -> dict:
        """Encode text and images.

        Args:
            text: List of prompt strings.
            images: List of PIL images.
            return_tensors: Ignored; always returns torch tensors.

        Returns:
            Encoded tensors in the Qwen layout.
        """
        encoded = self.image_processor(images=images, return_tensors="pt")
        grid_h, grid_w = encoded["image_grid_thw"][0][1:].tolist()
        n_image_tokens = (grid_h // MERGE_SIZE) * (grid_w // MERGE_SIZE)

        ids = self.tokenizer(text[0])["input_ids"]
        row = [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * n_image_tokens
        row += [VISION_END_TOKEN_ID] + ids
        input_ids = torch.tensor([row], dtype=torch.long)
        mm = torch.zeros_like(input_ids)
        mm[:, 1 : 1 + n_image_tokens] = 1
        return {
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": mm,
        }


def make_fake_dataset_root(root: Path, n_images: int = 4, questions_per_image: int = 2) -> Path:
    """Write a miniature AiR-D/GQA tree that :class:`GazeVQADataset` can open.

    Args:
        root: Directory to populate; ``root/raw/...`` is created.
        n_images: Number of images.
        questions_per_image: Questions per image.

    Returns:
        The data root.
    """
    raw = root / "raw"
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "gqa").mkdir(parents=True, exist_ok=True)
    agg = raw / "air_d" / "aggregated_maps" / "fixmaps"
    agg.mkdir(parents=True, exist_ok=True)

    questions = {}
    rng = np.random.default_rng(0)
    for i in range(n_images):
        image_id = f"img{i:03d}"
        ramp_image(64, 64).save(raw / "images" / f"{image_id}.jpg")
        for j in range(questions_per_image):
            qid = f"{i:03d}{j}"
            questions[qid] = {
                "imageId": image_id,
                "question": f"Is there a thing number {j}?",
                "answer": "yes" if j % 2 == 0 else "no",
                "fullAnswer": "Yes, there is.",
                "types": {"structural": "verify", "semantic": "rel", "detailed": "existRel"},
            }
            for bin_name in ("0s", "1s", "2s"):
                d = raw / "air_d" / "temporal_maps" / f"fixmaps_{bin_name}"
                d.mkdir(parents=True, exist_ok=True)
                arr = (rng.random((256, 256)) * 255).astype(np.uint8)
                Image.fromarray(arr, mode="L").save(d / f"{qid}.png")
            arr = (rng.random((256, 256)) * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(agg / f"{qid}.png")

    (raw / "gqa" / "val_balanced_questions.json").write_text(json.dumps(questions))
    return root

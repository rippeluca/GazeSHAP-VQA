#!/usr/bin/env python
"""Verify that a real Qwen3-VL checkpoint supports the Stage 1 double backward.

The unit tests prove the mechanism on a tiny model built from config. This
script proves it on the actual backbone and reports peak memory, which is what
decides the Stage 1 batch size. Run it once per machine before training.

Example:
    python scripts/check_double_backward.py --visual-tokens 64
    python scripts/check_double_backward.py --device cuda --n-baselines 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.shap_module import DifferentiableSHAPModule  # noqa: E402
from src.models.vlm_wrapper import VLMWrapper  # noqa: E402
from src.utils.config import ModelConfig  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger("check_double_backward")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--visual-tokens", type=int, default=16, help="token budget, in 32x32 units")
    p.add_argument("--n-baselines", type=int, default=2)
    p.add_argument("--patch-grid-size", type=int, default=16)
    return p.parse_args()


def main() -> int:
    """Run one attribution step and one backward pass.

    Returns:
        Process exit code.
    """
    args = parse_args()
    setup_logging(logging.INFO)

    config = ModelConfig(
        name=args.model,
        dtype=args.dtype,
        min_visual_tokens=max(4, args.visual_tokens // 2),
        max_visual_tokens=args.visual_tokens,
    )
    log.info("Loading %s on %s (%s)", config.name, args.device, config.dtype)
    vlm = VLMWrapper.from_config(config, device=args.device)

    from PIL import Image

    image = Image.new("RGB", (256, 256), color=(120, 120, 120))
    from src.models.reasoning_head import stage1_prompt

    encoded = vlm.processor(
        text=[stage1_prompt(vlm.processor, "What colour is the object on the left?")],
        images=[image],
        return_tensors="pt",
    ).to(args.device)
    answer_ids = torch.tensor(
        vlm.processor.tokenizer("red", add_special_tokens=False)["input_ids"],
        dtype=torch.long,
        device=args.device,
    ).view(1, -1)
    ones = torch.ones(
        1, answer_ids.shape[1], dtype=encoded["attention_mask"].dtype, device=args.device
    )
    encoded["input_ids"] = torch.cat([encoded["input_ids"], answer_ids], dim=1)
    encoded["attention_mask"] = torch.cat([encoded["attention_mask"], ones], dim=1)
    if encoded.get("mm_token_type_ids") is not None:
        zeros = torch.zeros_like(ones)
        encoded["mm_token_type_ids"] = torch.cat([encoded["mm_token_type_ids"], zeros], dim=1)

    log.info(
        "pixel_values=%s image_grid_thw=%s input_ids=%s",
        tuple(encoded["pixel_values"].shape),
        encoded["image_grid_thw"].tolist(),
        tuple(encoded["input_ids"].shape),
    )

    shap = DifferentiableSHAPModule(
        vlm, patch_grid_size=args.patch_grid_size, n_baselines=args.n_baselines
    )
    start = time.time()
    shap_map = shap(
        pixel_values=encoded["pixel_values"],
        image_grid_thw=encoded["image_grid_thw"],
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        mm_token_type_ids=encoded.get("mm_token_type_ids"),
        target_token_ids=answer_ids,
        answer_lengths=torch.tensor([answer_ids.shape[1]], device=args.device),
        create_graph=True,
    )
    forward_s = time.time() - start
    log.info(
        "SHAP map %s, sums to %.6f, in %.1fs",
        tuple(shap_map.shape),
        float(shap_map.sum()),
        forward_s,
    )

    gaze = torch.full_like(shap_map, 1.0 / shap_map[0].numel())
    loss = ((shap_map - gaze) ** 2).mean()
    start = time.time()
    loss.backward()
    backward_s = time.time() - start

    lora = {n: p for n, p in vlm.model.named_parameters() if "lora_" in n and p.requires_grad}
    with_grad = [n for n, p in lora.items() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    log.info("backward in %.1fs", backward_s)
    log.info("LoRA parameters with a non-zero gradient: %d / %d", len(with_grad), len(lora))

    # peft initialises lora_B to zeros, so dL/d(lora_A) is exactly zero on the
    # very first step. Only lora_B carrying gradient is the expected picture here.
    b_total = sum(1 for n in lora if "lora_B" in n)
    b_with_grad = sum(1 for n in with_grad if "lora_B" in n)
    log.info(
        "  of which lora_B: %d / %d (lora_A is zero at init by construction)", b_with_grad, b_total
    )

    if args.device.startswith("cuda"):
        log.info("peak CUDA memory: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)

    if not with_grad:
        log.error("NO gradient reached the LoRA weights. Stage 1 would train nothing.")
        return 1
    log.info("Double backward works on %s.", config.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

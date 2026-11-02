"""Save SHAP maps for the three questions shown in the README.

Attribution matches Stage 1: the sum of the teacher-forced answer logits, two
black baselines, and 64-128 visual tokens. Pass the Stage 1 adapter so the
maps are the trained run rather than the untouched backbone.

Example:
    python scripts/render_example_shap.py --device cuda \\
        --adapter checkpoints/stage1_answer/stage1_last/adapter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import collate_gaze_vqa  # noqa: E402
from src.data.dataset import GazeVQADataset  # noqa: E402
from src.evaluation.gaze_alignment import all_metrics  # noqa: E402
from src.models.shap_module import DifferentiableSHAPModule  # noqa: E402
from src.models.vlm_wrapper import VLMWrapper  # noqa: E402
from src.utils.config import ModelConfig  # noqa: E402

QIDS = ("00137872", "00137901", "00156120")
OUT = Path("site/examples")


def _load_adapter(vlm: VLMWrapper, path: Path) -> None:
    """Copy a saved Stage 1 adapter onto the freshly attached LoRA."""
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    incompatible = set_peft_model_state_dict(vlm.model, load_peft_weights(str(path)))
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    lora_missing = [key for key in missing if "lora_" in key]
    print(f"loaded adapter {path} lora_missing={len(lora_missing)}", flush=True)


def main() -> None:
    """Attribute the three README questions and write one ``.npy`` each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    config = ModelConfig(min_visual_tokens=64, max_visual_tokens=128, dtype="bfloat16")
    print(f"device={args.device} dtype={config.dtype}", flush=True)
    vlm = VLMWrapper.from_config(config, device=args.device)
    if args.adapter is not None:
        _load_adapter(vlm, args.adapter)
    vlm.model.eval()
    for parameter in vlm.model.parameters():
        parameter.requires_grad_(False)
    device = args.device

    dataset = GazeVQADataset(args.data_root, vlm.processor, qids=list(QIDS))
    shap = DifferentiableSHAPModule(vlm, n_baselines=2, baseline="black", attr_agg="sum")
    args.out.mkdir(parents=True, exist_ok=True)

    for index in range(len(dataset)):
        item = dataset[index]
        batch = collate_gaze_vqa([item], pad_token_id=vlm.processor.tokenizer.pad_token_id or 0)
        dtype = next(vlm.model.parameters()).dtype
        pixels = batch["pixel_values"].to(device=device, dtype=dtype)
        grid = batch["image_grid_thw"].to(device)
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        mm = batch["mm_token_type_ids"].to(device)
        targets = batch["target_token_ids"].to(device)
        lengths = batch["answer_lengths"].to(device)

        shap_map = shap(
            pixel_values=pixels,
            image_grid_thw=grid,
            input_ids=input_ids,
            attention_mask=mask,
            mm_token_type_ids=mm,
            target_token_ids=targets,
            answer_lengths=lengths,
            create_graph=False,
        )
        array = shap_map[0].float().cpu().numpy()
        qid = item["qid"]
        np.save(args.out / f"{qid}_shap.npy", array)
        scores = all_metrics(array, item["gaze_map"].numpy())
        print(
            f"{qid} answer={item['answer_str']!r} "
            f"auc={scores['auc_judd']:.3f} cc={scores['cc']:.3f} kl={scores['kl']:.3f}",
            flush=True,
        )
        del shap_map
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

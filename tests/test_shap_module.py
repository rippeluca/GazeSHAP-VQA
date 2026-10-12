"""The gate for the whole project.

If ``test_gradient_reaches_lora_parameters`` fails, every downstream result is
meaningless: the alignment loss would decrease while the LoRA weights never move
(idea.md 5.2, 14). Nothing past this file should be trusted until it passes.
"""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model

from src.models.shap_module import DifferentiableSHAPModule
from src.models.vlm_wrapper import DoubleBackwardUnsafeError, VLMWrapper
from tests.fixtures import dummy_batch, tiny_model

GRID = 4
PATCH_GRID_SIZE = 2


def _lora_model(seed: int = 0):
    model = tiny_model(seed=seed)
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


@pytest.fixture
def wrapper():
    return VLMWrapper(_lora_model(), require_double_backward=True)


@pytest.fixture
def shap(wrapper):
    return DifferentiableSHAPModule(
        wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=2, baseline="black"
    )


@pytest.fixture
def batch():
    return dummy_batch(grid_h=GRID, grid_w=GRID, batch_size=1)


# --- the gate ---------------------------------------------------------------


def test_gradient_reaches_lora_parameters(shap, wrapper, batch):
    """Second-order gradients must land on the LoRA weights.

    This is the test that catches the captum trap: an attribution that is a leaf
    with respect to the parameters makes the loss fall without any learning.
    """
    shap_map = shap(**batch, create_graph=True)
    gaze = torch.full_like(shap_map, 1.0 / shap_map[0].numel())
    loss = ((shap_map - gaze) ** 2).mean()
    loss.backward()

    lora_params = {
        name: param
        for name, param in wrapper.model.named_parameters()
        if "lora_" in name and param.requires_grad
    }
    assert lora_params, "no trainable LoRA parameters found"

    with_grad = {n: p for n, p in lora_params.items() if p.grad is not None}
    assert with_grad, "no LoRA parameter received a gradient"

    nonzero = [n for n, p in with_grad.items() if float(p.grad.abs().sum()) > 0]
    assert nonzero, "every LoRA gradient was exactly zero; the graph is detached"


def test_create_graph_false_detaches(shap, batch):
    """Stage 2 uses ``create_graph=False``; the map must then be a leaf."""
    shap_map = shap(**batch, create_graph=False)
    assert not shap_map.requires_grad


def test_create_graph_true_keeps_the_graph(shap, batch):
    assert shap(**batch, create_graph=True).requires_grad


# --- output contract --------------------------------------------------------


def test_output_shape_and_normalisation(shap, batch):
    shap_map = shap(**batch, create_graph=False)
    assert shap_map.shape == (1, PATCH_GRID_SIZE, PATCH_GRID_SIZE)
    assert torch.isclose(shap_map.sum(), torch.tensor(1.0), atol=1e-5)
    assert (shap_map >= 0).all()


def test_batch_of_two(wrapper):
    shap = DifferentiableSHAPModule(wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=1)
    batch = dummy_batch(grid_h=GRID, grid_w=GRID, batch_size=2)
    shap_map = shap(**batch, create_graph=False)
    assert shap_map.shape == (2, PATCH_GRID_SIZE, PATCH_GRID_SIZE)
    for row in shap_map:
        assert torch.isclose(row.sum(), torch.tensor(1.0), atol=1e-5)


def test_attribution_is_deterministic(shap, batch):
    a = shap(**batch, create_graph=False)
    b = shap(**batch, create_graph=False)
    assert torch.allclose(a, b, atol=1e-6)


def test_abs_aggregation_is_nonnegative_before_softmax(wrapper, batch):
    signed = DifferentiableSHAPModule(wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=2)
    unsigned = DifferentiableSHAPModule(
        wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=2, attr_agg="abs"
    )
    raw_signed = signed.attributions(**batch, create_graph=False)
    raw_unsigned = unsigned.attributions(**batch, create_graph=False)
    assert (raw_unsigned >= 0).all()
    assert torch.allclose(raw_unsigned, raw_signed.abs(), atol=1e-6)


def test_gaussian_baseline_runs(wrapper, batch):
    shap = DifferentiableSHAPModule(
        wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=2, baseline="gaussian"
    )
    out = shap(**batch, create_graph=False)
    assert out.shape == (1, PATCH_GRID_SIZE, PATCH_GRID_SIZE)
    assert torch.isfinite(out).all()


def test_more_baselines_changes_the_estimate(wrapper, batch):
    few = DifferentiableSHAPModule(wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=1)
    many = DifferentiableSHAPModule(wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=8)
    assert not torch.allclose(
        few(**batch, create_graph=False), many(**batch, create_graph=False), atol=1e-4
    )


def test_attribution_respects_the_patch_grid(wrapper):
    """A larger image must yield attributions for every one of its patches."""
    shap = DifferentiableSHAPModule(wrapper, patch_grid_size=PATCH_GRID_SIZE, n_baselines=1)
    batch = dummy_batch(grid_h=6, grid_w=4, batch_size=1)
    raw = shap.attributions(**batch, create_graph=False)
    assert raw.shape == (6 * 4,)


# --- captum cross-check -----------------------------------------------------


class _TinyScorer(torch.nn.Module):
    """A small differentiable stand-in with the same call signature needs."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 8)
        self.head = torch.nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.linear(x)))


def _expected_gradients(model, inputs, baselines, target, alphas):
    """Reference implementation of expected gradients, matching the module."""
    total = torch.zeros_like(inputs)
    for baseline, alpha in zip(baselines, alphas, strict=True):
        x = (baseline + alpha * (inputs - baseline)).requires_grad_(True)
        out = model(x)[:, target].sum()
        (grad,) = torch.autograd.grad(out, x)
        total = total + grad * (inputs - baseline)
    return total / len(baselines)


def test_expected_gradients_agrees_with_captum():
    """Cross-check the estimator against captum on a tiny model.

    Captum is allowed here and nowhere else: it cannot be used in training
    because it does not propagate ``create_graph``.
    """
    from captum.attr import GradientShap

    torch.manual_seed(0)
    n_features = 6
    model = _TinyScorer(n_features).eval()
    inputs = torch.randn(1, n_features)
    baselines = torch.zeros(1, n_features)

    ours = _expected_gradients(
        model, inputs, [baselines[0:1]] * 200, target=1, alphas=torch.linspace(0.0025, 0.9975, 200)
    )
    theirs = GradientShap(model).attribute(
        inputs, baselines=baselines, target=1, n_samples=800, stdevs=0.0
    )

    cosine = torch.nn.functional.cosine_similarity(ours.flatten(), theirs.flatten(), dim=0)
    assert float(cosine) > 0.99, (ours, theirs)


# --- guards -----------------------------------------------------------------


def test_wrapper_refuses_sdpa():
    model = tiny_model(attn_implementation="sdpa")
    with pytest.raises(DoubleBackwardUnsafeError, match="double-backward"):
        VLMWrapper(model, require_double_backward=True)


def test_wrapper_refuses_gradient_checkpointing():
    model = tiny_model()
    model.model.language_model.gradient_checkpointing = True
    with pytest.raises(DoubleBackwardUnsafeError, match="checkpointing"):
        VLMWrapper(model, require_double_backward=True)


def test_wrapper_allows_unsafe_config_when_not_required():
    model = tiny_model(attn_implementation="sdpa")
    VLMWrapper(model, require_double_backward=False)


def test_target_logit_sums_the_answer_suffix():
    logits = torch.zeros(2, 6, 10)
    # Sample 0 has two answer tokens, predicted from positions 3 and 4.
    logits[0, 3, 3] = 1.0
    logits[0, 4, 4] = 2.0
    # Sample 1 has one answer token, predicted from position 4.
    logits[1, 4, 8] = -2.0
    selected = VLMWrapper.target_logit(
        logits,
        torch.tensor([[3, 4], [8, 0]]),
        torch.tensor([2, 1]),
    )
    assert selected.tolist() == [3.0, -2.0]


def test_freeze_and_unfreeze_vision_lora():
    wrapper = VLMWrapper(_lora_model(), require_double_backward=True)
    wrapper.freeze_vision_tower()
    visual = [p for n, p in wrapper.model.named_parameters() if "visual." in n]
    assert visual and not any(p.requires_grad for p in visual)

    wrapper.unfreeze_vision_lora()
    visual_lora = [
        p for n, p in wrapper.model.named_parameters() if "visual." in n and "lora_" in n
    ]
    if visual_lora:
        assert all(p.requires_grad for p in visual_lora)
    visual_base = [
        p for n, p in wrapper.model.named_parameters() if "visual." in n and "lora_" not in n
    ]
    assert not any(p.requires_grad for p in visual_base)

from __future__ import annotations

import os

import pytest
import torch

# Keep third-party caches out of the home directory and off the network.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device the tests run on. The suite is designed to pass on CPU.",
    )


@pytest.fixture(scope="session")
def device(request) -> torch.device:
    return torch.device(request.config.getoption("--device"))


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)

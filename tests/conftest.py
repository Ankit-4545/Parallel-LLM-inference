import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pllm.config import DEFAULT_THREADS, REFERENCE_FILE  # noqa: E402
from pllm.model_loader import resolve_model_dir, set_determinism  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _determinism():
    set_determinism(threads=DEFAULT_THREADS)
    yield


@pytest.fixture(scope="session")
def model_dir():
    return resolve_model_dir()


@pytest.fixture(scope="session")
def reference():
    return json.loads(Path(REFERENCE_FILE).read_text())


@pytest.fixture(scope="session")
def hf_model(model_dir):
    from baseline import load_model

    model, _, _ = load_model(model_dir)
    return model


@pytest.fixture(scope="session")
def stages(model_dir):
    from pllm.partition import load_stages

    return load_stages(model_dir, num_stages=2, split="6")


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.inference_mode():
        yield

"""Locate pretrained GPT-2 files and load only the weights a stage needs.

Workers never instantiate the full Hugging Face model: they read the selected
tensors straight out of `model.safetensors`, so a stage's memory holds only its
own layers (plus the embedding table / LM head where needed).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from pllm.config import DEFAULT_SEED, MODEL_NAME, MODEL_REVISION

MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "generation_config.json",
]


@dataclass
class GPT2Config:
    n_layer: int
    n_head: int
    n_embd: int
    n_positions: int
    vocab_size: int
    layer_norm_epsilon: float
    activation_function: str

    @classmethod
    def from_json(cls, path: Path) -> "GPT2Config":
        d = json.loads(Path(path).read_text())
        if d.get("model_type", "gpt2") != "gpt2":
            raise ValueError(f"only GPT-2 checkpoints are supported, got {d.get('model_type')}")
        return cls(
            n_layer=d["n_layer"],
            n_head=d["n_head"],
            n_embd=d["n_embd"],
            n_positions=d["n_positions"],
            vocab_size=d["vocab_size"],
            layer_norm_epsilon=d.get("layer_norm_epsilon", 1e-5),
            activation_function=d.get("activation_function", "gelu_new"),
        )

    def params_per_layer(self) -> int:
        d = self.n_embd
        # ln_1, c_attn, c_proj, ln_2, c_fc, mlp.c_proj
        return 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + 2 * d + (d * 4 * d + 4 * d) + (4 * d * d + d)


def resolve_model_dir(model_name: str = MODEL_NAME, revision: str = MODEL_REVISION,
                      local_only: bool | None = None) -> Path:
    """Return the local snapshot directory, downloading it from the Hugging Face Hub if needed."""
    if os.path.isdir(model_name):
        return Path(model_name)
    from huggingface_hub import snapshot_download

    if local_only is None:
        local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    try:
        path = snapshot_download(model_name, revision=revision, allow_patterns=MODEL_FILES,
                                 local_files_only=True)
    except Exception:
        if local_only:
            raise
        path = snapshot_download(model_name, revision=revision, allow_patterns=MODEL_FILES)
    return Path(path)


def snapshot_commit(model_dir: Path) -> str:
    """The Hub commit hash of a snapshot directory (its folder name), for provenance."""
    return Path(model_dir).name


def load_tensors(model_dir: Path, keys_filter) -> dict[str, torch.Tensor]:
    """Load only the tensors whose (normalised) name satisfies `keys_filter`."""
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            name = key[len("transformer."):] if key.startswith("transformer.") else key
            if keys_filter(name):
                out[name] = f.get_tensor(key).to(torch.float32)
    return out


def set_determinism(seed: int = DEFAULT_SEED, threads: int | None = None) -> None:
    """Seed every RNG and force deterministic kernels."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.use_deterministic_algorithms(True)
    if threads is not None:
        torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

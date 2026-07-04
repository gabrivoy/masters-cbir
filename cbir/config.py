"""Central configuration for the CBIR system.

Everything tunable lives here so the CLI, indexer, and Streamlit app share a
single source of truth. Values can be overridden at call sites, but these are
the defaults the whole system agrees on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Milvus
# Host/port are overridable by env vars so the same code reaches Milvus at
# 127.0.0.1 locally and at the `milvus` service name inside docker-compose.
MILVUS_HOST = os.getenv("CBIR_MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("CBIR_MILVUS_PORT", "19530")
MILVUS_ALIAS = "default"

# Embedding models
# Maps a short, stable name to the open_clip loading spec. The slug is what we
# bake into collection names and cache filenames so different models never
# collide.
MODEL_SPECS: dict[str, dict[str, str]] = {
    "openclip-vit-b-32": {
        "open_clip_model_name": "ViT-B-32",
        "pretrained": "openai",
        "slug": "openclip_vit_b_32_openai",
    },
    "openclip-vit-b-16": {
        "open_clip_model_name": "ViT-B-16",
        "pretrained": "openai",
        "slug": "openclip_vit_b_16_openai",
    },
}
DEFAULT_MODEL = "openclip-vit-b-32"

# Crop / extraction
# The persisted unit is the raw bbox; padding is a runtime knob. 0.0 keeps the
# crop tight to the annotation (see docs/CONSIDERATIONS_01.md).
DEFAULT_PADDING_RATIO = 0.0
DEFAULT_BATCH_SIZE = 32

# Default device. "auto" prefers CUDA, then Apple Silicon GPU (MPS/Metal), then
# multi-threaded CPU. Any explicit device falls back to CPU if unavailable.
DEFAULT_DEVICE = "auto"

# Projection
# PCA is deterministic, so a seed is only cosmetic, but we fix one anyway so
# the SVD solver behaves identically across runs.
PCA_SEED = 42

# Paths


def repo_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parents[1]


def cache_dir() -> Path:
    """Directory where the indexer writes the local embedding cache."""
    return repo_root() / "artifacts" / "cbir_index"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved embedding-model specification."""

    name: str
    open_clip_model_name: str
    pretrained: str
    slug: str


def resolve_model(name: str) -> ModelSpec:
    """Look up a model spec by short name, with a helpful error if unknown."""
    try:
        spec = MODEL_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model {name!r}. Available: {available}") from exc
    return ModelSpec(
        name=name,
        open_clip_model_name=spec["open_clip_model_name"],
        pretrained=spec["pretrained"],
        slug=spec["slug"],
    )

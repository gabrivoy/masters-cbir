"""Embedding extraction with OpenCLIP.

Loads a pretrained OpenCLIP model, derives a runtime crop from each manifest
record's bounding box, encodes it, and L2-normalizes the result so cosine
similarity in Milvus behaves as inner product.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

from cbir.config import DEFAULT_PADDING_RATIO, ModelSpec, repo_root, resolve_model
from cbir.core.manifest import ManifestRecord, clipped_crop_box


def resolve_image_path(image_path: str) -> Path:
    """Resolve a manifest image path.

    Absolute paths are used as-is; relative paths (used by the committable
    sample) are resolved against the repository root so the system works
    regardless of the current working directory.
    """
    path = Path(image_path)
    return path if path.is_absolute() else repo_root() / path


def resolve_device(requested: str) -> str:
    """Resolve the compute device, degrading gracefully.

    - ``auto`` picks the best available: CUDA, then Apple Silicon GPU (MPS /
      Metal), then multi-threaded CPU.
    - ``cuda`` / ``mps`` fall back to CPU when unavailable rather than raising,
      so the same command runs on any machine.
    """
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if requested == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"
    if requested == "cuda" and not has_cuda:
        return "mps" if has_mps else "cpu"
    if requested == "mps" and not has_mps:
        return "cpu"
    return requested


class Embedder:
    """A loaded embedding model plus its preprocessing transform.

    Instances are cheap to *use* but expensive to *build* (model download +
    load), so callers should create one and reuse it across a batch or a whole
    Streamlit session.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.spec: ModelSpec = resolve_model(model_name)
        self.device = resolve_device(device)
        # On the CPU fallback, use every core: torch defaults to a single
        # intra-op thread in some environments, which makes indexing needlessly
        # slow. Harmless on GPU devices.
        if self.device == "cpu":
            cores = os.cpu_count() or 1
            torch.set_num_threads(cores)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"QuickGELU mismatch between final model config .*",
                category=UserWarning,
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.spec.open_clip_model_name,
                pretrained=self.spec.pretrained,
                device=self.device,
            )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self.embedding_dim = int(self._model.visual.output_dim)

    # --- crops ------------------------------------------------------------
    def crop_from_record(
        self,
        record: ManifestRecord,
        *,
        padding_ratio: float = DEFAULT_PADDING_RATIO,
    ) -> Image.Image:
        """Open the source frame and crop the record's bounding box.

        ``image_path`` may be absolute (the full pool) or relative to the repo
        root (the committable sample); both are resolved here.
        """
        with Image.open(resolve_image_path(record.image_path)) as handle:
            image = handle.convert("RGB")
        box = clipped_crop_box(record.bbox_xywh, image.width, image.height, padding_ratio)
        return image.crop(box)

    # --- embeddings -------------------------------------------------------
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images into an (N, dim) float32 array."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        tensors = [self._preprocess(image) for image in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().cpu().numpy().astype(np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single image into a (dim,) float32 vector."""
        return self.embed_images([image])[0]

    def embed_path(self, path: str | Path) -> np.ndarray:
        """Load a whole image from disk and embed it (used for query uploads)."""
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        return self.embed_image(image)

    def embed_records(
        self,
        records: list[ManifestRecord],
        *,
        batch_size: int,
        padding_ratio: float = DEFAULT_PADDING_RATIO,
    ) -> list[dict[str, Any]]:
        """Embed a list of records in batches.

        Returns one dict per record with its ``item_id``, the original
        ``record``, and its ``embedding``. Batching bounds peak memory while
        keeping the GPU/CPU busy.
        """
        outputs: list[dict[str, Any]] = []
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            crops = [self.crop_from_record(r, padding_ratio=padding_ratio) for r in chunk]
            embeddings = self.embed_images(crops)
            for record, embedding in zip(chunk, embeddings, strict=True):
                outputs.append(
                    {
                        "item_id": record.item_id,
                        "record": record,
                        "embedding": embedding,
                    }
                )
        return outputs

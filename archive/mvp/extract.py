from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import open_clip
import torch
from PIL import Image

from data_prep import DEFAULT_PADDING_RATIO, clipped_crop_box, slugify_label

MODEL_SPECS: dict[str, dict[str, str]] = {
    "openclip-vit-b-32": {
        "open_clip_model_name": "ViT-B-32",
        "pretrained": "openai",
        "model_slug": "openclip_vit_b_32_openai",
    },
    "openclip-vit-b-16": {
        "open_clip_model_name": "ViT-B-16",
        "pretrained": "openai",
        "model_slug": "openclip_vit_b_16_openai",
    },
}


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def model_spec(model_name: str) -> dict[str, str]:
    try:
        return MODEL_SPECS[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}") from exc


def load_model(model_name: str = "openclip-vit-b-32", device: str = "cpu") -> dict[str, Any]:
    spec = model_spec(model_name)
    resolved_device = resolve_device(device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"QuickGELU mismatch between final model config .*",
            category=UserWarning,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec["open_clip_model_name"],
            pretrained=spec["pretrained"],
            device=resolved_device,
        )
    model.eval()
    embedding_dim = int(getattr(model.visual, "output_dim"))
    return {
        "model_name": model_name,
        "model_slug": spec["model_slug"],
        "open_clip_model_name": spec["open_clip_model_name"],
        "pretrained": spec["pretrained"],
        "device": resolved_device,
        "model": model,
        "preprocess": preprocess,
        "embedding_dim": embedding_dim,
    }


def crop_box_from_record(
    record: dict[str, Any],
    image_width: int,
    image_height: int,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> tuple[int, int, int, int]:
    return clipped_crop_box(
        (
            float(record["bbox_x"]),
            float(record["bbox_y"]),
            float(record["bbox_w"]),
            float(record["bbox_h"]),
        ),
        image_width,
        image_height,
        padding_ratio,
    )


def crop_from_record(
    record: dict[str, Any],
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> Image.Image:
    image_path = Path(record["image_path"])
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    crop_box = crop_box_from_record(record, image.width, image.height, padding_ratio)
    return image.crop(crop_box)


def preprocess_images(
    preprocess: Any,
    images: list[Image.Image],
    *,
    device: str,
) -> torch.Tensor:
    tensors = [preprocess(image) for image in images]
    batch = torch.stack(tensors)
    return batch.to(device)


def normalize_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_batch(
    model_bundle: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> list[dict[str, Any]]:
    if not records:
        return []

    model = model_bundle["model"]
    preprocess = model_bundle["preprocess"]
    resolved_device = resolve_device(device)
    outputs: list[dict[str, Any]] = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_images = [crop_from_record(record, padding_ratio=padding_ratio) for record in batch_records]
        batch_tensor = preprocess_images(preprocess, batch_images, device=resolved_device)

        with torch.inference_mode():
            features = model.encode_image(batch_tensor)
            features = normalize_embeddings(features)
        embeddings = features.detach().cpu().numpy().astype(np.float32)

        for batch_record, embedding in zip(batch_records, embeddings, strict=True):
            outputs.append(
                {
                    "item_id": batch_record["item_id"],
                    "record": batch_record,
                    "embedding": embedding,
                }
            )

    return outputs


def collection_name_for(target_class: str, model_name: str) -> str:
    return f"cbir_bbox_{slugify_label(target_class)}_{model_spec(model_name)['model_slug']}_v1"

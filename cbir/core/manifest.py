"""Manifest records: the bbox-level data contract this system consumes.

A manifest is a JSONL file, one record per annotated bounding box, produced by
the data-preparation stage (the frozen ``mvp/data_prep.py`` writes v2
manifests under ``data/cbir/<class>/v1/``). This module only *reads* manifests;
it never derives them. Each record carries the raw bbox plus metadata, and the
crop is computed at runtime from the source frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Fields every record must have for the system to work. Extra fields are kept
# but never required.
REQUIRED_FIELDS = (
    "item_id",
    "target_class",
    "split",
    "image_path",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
)


@dataclass(frozen=True)
class ManifestRecord:
    """One bounding-box item, wrapping the raw manifest dict.

    The dict is kept intact under ``raw`` so downstream code can read any
    metadata field (camera_id, size_bucket, timestamp, ...) without this class
    having to enumerate all of them.
    """

    raw: dict[str, Any]

    @property
    def item_id(self) -> str:
        return str(self.raw["item_id"])

    @property
    def target_class(self) -> str:
        return str(self.raw["target_class"])

    @property
    def split(self) -> str:
        return str(self.raw["split"])

    @property
    def image_path(self) -> str:
        return str(self.raw["image_path"])

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        return (
            float(self.raw["bbox_x"]),
            float(self.raw["bbox_y"]),
            float(self.raw["bbox_w"]),
            float(self.raw["bbox_h"]),
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def _validate(record: dict[str, Any], source: Path) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"Manifest record in {source} is missing required fields {missing}: "
            f"{record.get('item_id', '<no item_id>')}"
        )


def load_manifest(path: Path) -> list[ManifestRecord]:
    """Load a single manifest JSONL file into validated records."""
    records: list[ManifestRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            _validate(raw, path)
            records.append(ManifestRecord(raw))
    return records


def load_manifests(paths: list[Path]) -> list[ManifestRecord]:
    """Load and concatenate several manifests, rejecting duplicate item_ids.

    Duplicate item_ids across manifests would silently overwrite each other in
    Milvus (item_id is the primary key), so we fail loudly instead.
    """
    records: list[ManifestRecord] = []
    seen: set[str] = set()
    for path in paths:
        for record in load_manifest(path):
            if record.item_id in seen:
                raise ValueError(f"Duplicate item_id across manifests: {record.item_id}")
            seen.add(record.item_id)
            records.append(record)
    return records


def filter_records(
    records: list[ManifestRecord],
    *,
    split: str = "all",
    benchmark_only: bool = False,
    target_classes: set[str] | None = None,
) -> list[ManifestRecord]:
    """Filter records by split, benchmark flag, and/or a class allowlist."""
    result = records
    if split != "all":
        result = [r for r in result if r.split == split]
    if benchmark_only:
        result = [r for r in result if bool(r.get("is_benchmark_candidate", False))]
    if target_classes is not None:
        result = [r for r in result if r.target_class in target_classes]
    return result


def sample_head_per_class(
    records: list[ManifestRecord],
    *,
    per_class: int | None,
) -> list[ManifestRecord]:
    """Deterministically cap the number of records per class.

    Records are already stored in a stable order by the manifest builder, so
    taking the head is reproducible without any randomness. This keeps the
    indexer fast for a demo-sized gallery while staying deterministic.
    """
    if per_class is None:
        return list(records)
    if per_class <= 0:
        raise ValueError("per_class must be a positive integer.")
    counts: dict[str, int] = {}
    selected: list[ManifestRecord] = []
    for record in records:
        current = counts.get(record.target_class, 0)
        if current >= per_class:
            continue
        counts[record.target_class] = current + 1
        selected.append(record)
    return selected


def clipped_crop_box(
    bbox_xywh: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Compute an integer crop box, padded by a fraction of the bbox size and
    clipped to the image boundary."""
    x, y, width, height = bbox_xywh
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    x1 = max(0, math.floor(x - pad_x))
    y1 = max(0, math.floor(y - pad_y))
    x2 = min(image_width, math.ceil(x + width + pad_x))
    y2 = min(image_height, math.ceil(y + height + pad_y))
    return x1, y1, x2, y2

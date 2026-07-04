"""Tests for manifest loading, filtering, sampling, and crop-box math."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbir.core.manifest import (
    ManifestRecord,
    clipped_crop_box,
    filter_records,
    load_manifests,
    sample_head_per_class,
)


def _record(item_id: str, cls: str, split: str = "train", benchmark: bool = True) -> dict:
    return {
        "item_id": item_id,
        "target_class": cls,
        "split": split,
        "image_path": "x.jpg",
        "bbox_x": 0.0,
        "bbox_y": 0.0,
        "bbox_w": 10.0,
        "bbox_h": 10.0,
        "is_benchmark_candidate": benchmark,
    }


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_load_rejects_duplicate_item_ids(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_manifest(a, [_record("dup", "Traineira")])
    _write_manifest(b, [_record("dup", "Lancha")])
    with pytest.raises(ValueError, match="Duplicate item_id"):
        load_manifests([a, b])


def test_load_requires_mandatory_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"item_id": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing required fields"):
        load_manifests([path])


def test_filter_by_split_and_benchmark() -> None:
    records = [
        ManifestRecord(_record("a", "Traineira", split="train", benchmark=True)),
        ManifestRecord(_record("b", "Traineira", split="test", benchmark=True)),
        ManifestRecord(_record("c", "Traineira", split="train", benchmark=False)),
    ]
    train = filter_records(records, split="train")
    assert {r.item_id for r in train} == {"a", "c"}
    bench = filter_records(records, split="train", benchmark_only=True)
    assert {r.item_id for r in bench} == {"a"}


def test_sample_head_is_deterministic_and_per_class() -> None:
    records = [ManifestRecord(_record(f"t{i}", "Traineira")) for i in range(5)]
    records += [ManifestRecord(_record(f"l{i}", "Lancha")) for i in range(3)]
    picked = sample_head_per_class(records, per_class=2)
    by_class: dict[str, int] = {}
    for r in picked:
        by_class[r.target_class] = by_class.get(r.target_class, 0) + 1
    assert by_class == {"Traineira": 2, "Lancha": 2}
    # Deterministic: same input, same output order.
    again = sample_head_per_class(records, per_class=2)
    assert [r.item_id for r in picked] == [r.item_id for r in again]


def test_sample_head_none_returns_all() -> None:
    records = [ManifestRecord(_record(f"t{i}", "Traineira")) for i in range(4)]
    assert len(sample_head_per_class(records, per_class=None)) == 4


def test_clipped_crop_box_clips_to_image_bounds() -> None:
    # A bbox near the edge, padded, must not exceed the image.
    box = clipped_crop_box(
        (90.0, 90.0, 20.0, 20.0), image_width=100, image_height=100, padding_ratio=0.5
    )
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100


def test_clipped_crop_box_zero_padding_is_tight() -> None:
    box = clipped_crop_box(
        (10.0, 20.0, 30.0, 40.0), image_width=500, image_height=500, padding_ratio=0.0
    )
    assert box == (10, 20, 40, 60)

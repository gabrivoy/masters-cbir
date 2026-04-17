from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
DEFAULT_PADDING_RATIO = 0.15
DEFAULT_BENCHMARK_AREA_THRESHOLD = 1024.0
SIZE_BUCKETS = (
    ("micro 0-5", 0.0, 25.0),
    ("tiniest 5-10", 25.0, 100.0),
    ("tiny 10-15", 100.0, 225.0),
    ("very small2 15-20", 225.0, 400.0),
    ("very small1 20-25", 400.0, 625.0),
    ("smaller 25-32", 625.0, 1024.0),
    ("medium1 32-48", 1024.0, 2304.0),
    ("medium2 48-64", 2304.0, 4096.0),
    ("medium3 64-80", 4096.0, 6400.0),
    ("medium4 80-96", 6400.0, 9216.0),
    ("large 96-192", 9216.0, 36864.0),
    ("larger 192-384", 36864.0, 147456.0),
    ("largest 384-768", 147456.0, 589824.0),
    ("max 768-inf", 589824.0, float("inf")),
)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_root(repo_root: Path | None = None) -> Path:
    root = repo_root or resolve_repo_root()
    return (
        root
        / "data"
        / "Etapa6_Dataset_PTZ"
        / "runs_2025_05_22_10-46-01"
        / "run_2025_05-22_10-46-14_347849_PTZ"
    )


def image_root(repo_root: Path | None = None) -> Path:
    root = repo_root or resolve_repo_root()
    return root / "data" / "images_from_dataset"


def coco_annotation_path(split: str, repo_root: Path | None = None) -> Path:
    return (
        dataset_root(repo_root)
        / "result"
        / "Dataset_coco_instances"
        / "annotations"
        / f"instances_{split}.json"
    )


def datumaro_annotation_path(split: str, repo_root: Path | None = None) -> Path:
    return (
        dataset_root(repo_root)
        / "result"
        / "Dataset_datumaro"
        / "annotations"
        / f"{split}.json"
    )


def default_manifest_path(
    target_class: str,
    repo_root: Path | None = None,
) -> Path:
    root = repo_root or resolve_repo_root()
    return root / "mvp" / "manifests" / f"{slugify_label(target_class)}_manifest.jsonl"


def slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower())
    return slug.strip("_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_timestamp(value: Any) -> int | str | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return str(value)


def camera_from_filename(filename: str) -> str:
    parts = filename.split("_")
    return "_".join(parts[:2])


def size_bucket(area: float) -> str:
    for name, lower, upper in SIZE_BUCKETS:
        if lower <= area < upper:
            return name
    raise ValueError(f"Could not map bbox area {area} to a size bucket.")


def clipped_crop_box(
    bbox_xywh: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox_xywh
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    x1 = max(0, math.floor(x - pad_x))
    y1 = max(0, math.floor(y - pad_y))
    x2 = min(image_width, math.ceil(x + width + pad_x))
    y2 = min(image_height, math.ceil(y + height + pad_y))
    return x1, y1, x2, y2


def object_categories(coco_data: dict[str, Any]) -> dict[int, str]:
    return {
        category["id"]: category["name"]
        for category in coco_data["categories"]
        if not category["name"].startswith("[TAG]")
    }


def final_object_classes(repo_root: Path | None = None) -> list[str]:
    categories = object_categories(load_json(coco_annotation_path("train", repo_root)))
    return sorted(categories.values())


def load_datumaro_metadata(
    split: str,
    repo_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    datumaro_data = load_json(datumaro_annotation_path(split, repo_root))
    label_names = {
        index: label["name"]
        for index, label in enumerate(datumaro_data["categories"]["label"]["labels"])
    }
    metadata_by_item_id: dict[str, dict[str, Any]] = {}

    for item in datumaro_data["items"]:
        item_metadata: dict[str, Any] = {
            "camera_id": None,
            "timestamp": None,
        }
        for annotation in item["annotations"]:
            if annotation.get("type") != "label":
                continue
            label_name = label_names.get(annotation.get("label_id"))
            attributes = annotation.get("attributes", {})
            if label_name == "[TAG] Camera Configuration":
                metadata = attributes.get("metadata")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = None
                if isinstance(metadata, dict):
                    item_metadata["camera_id"] = metadata.get("id")
            elif label_name == "[TAG] Datetime":
                item_metadata["timestamp"] = normalize_timestamp(attributes.get("timestamp"))

        metadata_by_item_id[item["id"]] = item_metadata
    return metadata_by_item_id


def load_split_context(
    split: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    coco_data = load_json(coco_annotation_path(split, repo_root))
    categories = object_categories(coco_data)
    images_by_id = {image["id"]: image for image in coco_data["images"]}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in coco_data["annotations"]:
        label = categories.get(annotation["category_id"])
        if label is None:
            continue
        enriched_annotation = dict(annotation)
        enriched_annotation["label"] = label
        annotations_by_image_id[annotation["image_id"]].append(enriched_annotation)

    return {
        "categories": categories,
        "images_by_id": images_by_id,
        "annotations_by_image_id": annotations_by_image_id,
    }


def build_class_manifest(
    target_class: str,
    *,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    benchmark_area_threshold: float = DEFAULT_BENCHMARK_AREA_THRESHOLD,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    root = repo_root or resolve_repo_root()
    available_classes = final_object_classes(root)
    if target_class not in available_classes:
        joined = ", ".join(available_classes)
        raise ValueError(f"Unknown target class {target_class!r}. Available classes: {joined}")

    manifests_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split in SPLITS:
        context = load_split_context(split, root)
        datumaro_metadata = load_datumaro_metadata(split, root)
        annotations_by_image_id = context["annotations_by_image_id"]
        images_by_id = context["images_by_id"]

        labels_by_image_id = {
            image_id: sorted({annotation["label"] for annotation in annotations})
            for image_id, annotations in annotations_by_image_id.items()
        }

        for image_id, annotations in annotations_by_image_id.items():
            image = images_by_id[image_id]
            frame_labels = labels_by_image_id[image_id]
            frame_stem = Path(image["file_name"]).stem
            image_metadata = datumaro_metadata.get(frame_stem, {})
            image_path = image_root(root) / image["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            for annotation in annotations:
                if annotation["label"] != target_class:
                    continue

                bbox_x, bbox_y, bbox_w, bbox_h = annotation["bbox"]
                crop_x1, crop_y1, crop_x2, crop_y2 = clipped_crop_box(
                    (bbox_x, bbox_y, bbox_w, bbox_h),
                    image["width"],
                    image["height"],
                    padding_ratio,
                )

                manifests_by_split[split].append(
                    {
                        "target_class": target_class,
                        "split": split,
                        "idx_in_class_split": 0,
                        "annotation_id": annotation["id"],
                        "image_path": str(image_path),
                        "image_filename": image["file_name"],
                        "image_id": image_id,
                        "camera_id": image_metadata.get("camera_id") or camera_from_filename(image["file_name"]),
                        "timestamp": image_metadata.get("timestamp"),
                        "bbox_x": float(bbox_x),
                        "bbox_y": float(bbox_y),
                        "bbox_w": float(bbox_w),
                        "bbox_h": float(bbox_h),
                        "bbox_area": float(annotation["area"]),
                        "size_bucket": size_bucket(float(annotation["area"])),
                        "crop_x1": crop_x1,
                        "crop_y1": crop_y1,
                        "crop_x2": crop_x2,
                        "crop_y2": crop_y2,
                        "crop_w": crop_x2 - crop_x1,
                        "crop_h": crop_y2 - crop_y1,
                        "padding_ratio": padding_ratio,
                        "occluded": bool(annotation.get("attributes", {}).get("occluded", False)),
                        "difficult": bool(annotation.get("attributes", {}).get("difficult", False)),
                        "n_objects_in_frame": len(annotations),
                        "other_labels_in_frame": [label for label in frame_labels if label != target_class],
                        "is_benchmark_candidate": float(annotation["area"]) >= benchmark_area_threshold,
                    }
                )

    manifest: list[dict[str, Any]] = []
    for split in SPLITS:
        split_records = manifests_by_split[split]
        split_records.sort(
            key=lambda record: (
                record["image_filename"],
                record["bbox_y"],
                record["bbox_x"],
                record["bbox_w"],
                record["bbox_h"],
                record["annotation_id"],
            )
        )
        for index, record in enumerate(split_records, start=1):
            record["idx_in_class_split"] = index
            manifest.append(record)

    return manifest


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        summary[split] = {
            "records": len(split_records),
            "benchmark_records": sum(
                1 for record in split_records if record["is_benchmark_candidate"]
            ),
        }
    return summary


def print_summary(target_class: str, records: list[dict[str, Any]], output_path: Path) -> None:
    summary = manifest_summary(records)
    print(f"Manifest written for class: {target_class}")
    print(f"Output path: {output_path}")
    print(f"Total records: {len(records)}")
    for split in SPLITS:
        split_summary = summary[split]
        print(
            f"{split}: {split_summary['records']} total, "
            f"{split_summary['benchmark_records']} benchmark candidates"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a bbox-level CBIR manifest for a target class."
    )
    parser.add_argument(
        "--target-class",
        required=True,
        help="Final class label from the COCO export, for example 'Traineira'.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=DEFAULT_PADDING_RATIO,
        help="Relative crop padding applied around the bbox.",
    )
    parser.add_argument(
        "--benchmark-area-threshold",
        type=float,
        default=DEFAULT_BENCHMARK_AREA_THRESHOLD,
        help="Minimum bbox area used to mark benchmark candidates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the manifest. Defaults to mvp/manifests/<class>_manifest.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or default_manifest_path(args.target_class)
    manifest = build_class_manifest(
        args.target_class,
        padding_ratio=args.padding_ratio,
        benchmark_area_threshold=args.benchmark_area_threshold,
    )
    write_manifest(manifest, output_path)
    print_summary(args.target_class, manifest, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

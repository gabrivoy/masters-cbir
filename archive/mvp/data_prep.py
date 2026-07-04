from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 2
SPLITS = ("train", "val", "test")
DEFAULT_PADDING_RATIO = 0.0
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


def slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower())
    return slug.strip("_")


def class_artifact_dir(target_class: str, repo_root: Path | None = None) -> Path:
    root = repo_root or resolve_repo_root()
    return root / "data" / "cbir" / slugify_label(target_class) / "v1"


def default_manifest_path(target_class: str, repo_root: Path | None = None) -> Path:
    return class_artifact_dir(target_class, repo_root) / "manifest.jsonl"


def default_summary_path(target_class: str, repo_root: Path | None = None) -> Path:
    return class_artifact_dir(target_class, repo_root) / "summary.json"


def default_rejected_path(target_class: str, repo_root: Path | None = None) -> Path:
    return class_artifact_dir(target_class, repo_root) / "rejected.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def validate_bbox(
    bbox_xywh: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[bool, str | None]:
    x, y, width, height = bbox_xywh
    if width <= 0 or height <= 0:
        return False, "non_positive_extent"
    if x < 0 or y < 0:
        return False, "negative_origin"
    if x + width > image_width + 1e-6:
        return False, "bbox_exceeds_image_width"
    if y + height > image_height + 1e-6:
        return False, "bbox_exceeds_image_height"
    return True, None


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


def rejected_record(
    *,
    target_class: str,
    split: str,
    reason: str,
    image: dict[str, Any] | None = None,
    image_path: Path | None = None,
    annotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target_class": target_class,
        "split": split,
        "reason": reason,
    }
    if image is not None:
        payload["image_id"] = image["id"]
        payload["image_filename"] = image["file_name"]
    if image_path is not None:
        payload["image_path"] = str(image_path)
    if annotation is not None:
        payload["annotation_id"] = annotation["id"]
        payload["bbox"] = annotation.get("bbox")
    return payload


def build_class_manifest(
    target_class: str,
    *,
    benchmark_area_threshold: float = DEFAULT_BENCHMARK_AREA_THRESHOLD,
    repo_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = repo_root or resolve_repo_root()
    available_classes = final_object_classes(root)
    if target_class not in available_classes:
        joined = ", ".join(available_classes)
        raise ValueError(f"Unknown target class {target_class!r}. Available classes: {joined}")

    class_slug = slugify_label(target_class)
    manifests_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []

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

            target_annotations = [
                annotation for annotation in annotations if annotation["label"] == target_class
            ]
            if not target_annotations:
                continue

            if not image_path.exists():
                for annotation in target_annotations:
                    rejected.append(
                        rejected_record(
                            target_class=target_class,
                            split=split,
                            reason="image_missing",
                            image=image,
                            image_path=image_path,
                            annotation=annotation,
                        )
                    )
                continue

            for annotation in target_annotations:
                bbox_x, bbox_y, bbox_w, bbox_h = annotation["bbox"]
                bbox = (float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h))
                is_valid, reason = validate_bbox(
                    bbox,
                    image_width=image["width"],
                    image_height=image["height"],
                )
                if not is_valid:
                    rejected.append(
                        rejected_record(
                            target_class=target_class,
                            split=split,
                            reason=reason or "invalid_bbox",
                            image=image,
                            image_path=image_path,
                            annotation=annotation,
                        )
                    )
                    continue

                manifests_by_split[split].append(
                    {
                        "item_id": f"{class_slug}:{split}:{annotation['id']}",
                        "target_class": target_class,
                        "split": split,
                        "idx_in_class_split": 0,
                        "annotation_id": annotation["id"],
                        "image_path": str(image_path),
                        "image_filename": image["file_name"],
                        "image_id": image_id,
                        "camera_id": image_metadata.get("camera_id")
                        or camera_from_filename(image["file_name"]),
                        "timestamp": image_metadata.get("timestamp"),
                        "bbox_x": bbox[0],
                        "bbox_y": bbox[1],
                        "bbox_w": bbox[2],
                        "bbox_h": bbox[3],
                        "bbox_area": float(annotation["area"]),
                        "size_bucket": size_bucket(float(annotation["area"])),
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

    rejected.sort(
        key=lambda record: (
            record["split"],
            record.get("image_filename", ""),
            record.get("annotation_id", -1),
            record["reason"],
        )
    )
    return manifest, rejected


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_records(
    target_class: str,
    records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    counts_by_split: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        counts_by_split[split] = {
            "records": len(split_records),
            "benchmark_records": sum(
                1 for record in split_records if record["is_benchmark_candidate"]
            ),
        }

    counts_by_camera = Counter(record["camera_id"] for record in records)
    counts_by_size_bucket = Counter(record["size_bucket"] for record in records)
    rejected_by_reason = Counter(record["reason"] for record in rejected)

    return {
        "manifest_version": MANIFEST_VERSION,
        "target_class": target_class,
        "total_records": len(records),
        "total_rejected": len(rejected),
        "counts_by_split": counts_by_split,
        "counts_by_camera_id": dict(sorted(counts_by_camera.items())),
        "counts_by_size_bucket": {
            bucket_name: counts_by_size_bucket.get(bucket_name, 0)
            for bucket_name, _, _ in SIZE_BUCKETS
        },
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
    }


def write_manifest_artifacts(
    target_class: str,
    records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    root = repo_root or resolve_repo_root()
    artifact_dir = output_dir or class_artifact_dir(target_class, root)
    manifest_path = artifact_dir / "manifest.jsonl"
    summary_path = artifact_dir / "summary.json"
    rejected_path = artifact_dir / "rejected.jsonl"

    write_jsonl(records, manifest_path)
    write_json(summary_path, summarize_records(target_class, records, rejected))
    write_jsonl(rejected, rejected_path)
    return {
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "rejected_path": rejected_path,
    }


def select_manifest_records(
    records: list[dict[str, Any]],
    *,
    split: str = "all",
    benchmark_only: bool = False,
) -> list[dict[str, Any]]:
    filtered = records
    if split != "all":
        filtered = [record for record in filtered if record["split"] == split]
    if benchmark_only:
        filtered = [record for record in filtered if record["is_benchmark_candidate"]]
    return filtered


def print_summary(
    target_class: str,
    records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
) -> None:
    summary = summarize_records(target_class, records, rejected)
    print(f"Manifest written for class: {target_class}")
    print(f"Artifact dir: {artifact_paths['artifact_dir']}")
    print(f"Manifest path: {artifact_paths['manifest_path']}")
    print(f"Summary path: {artifact_paths['summary_path']}")
    print(f"Rejected path: {artifact_paths['rejected_path']}")
    print(f"Total records: {summary['total_records']}")
    print(f"Total rejected: {summary['total_rejected']}")
    for split in SPLITS:
        split_summary = summary["counts_by_split"][split]
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
        "--benchmark-area-threshold",
        type=float,
        default=DEFAULT_BENCHMARK_AREA_THRESHOLD,
        help="Minimum bbox area used to mark benchmark candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to data/cbir/<class>/v1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, rejected = build_class_manifest(
        args.target_class,
        benchmark_area_threshold=args.benchmark_area_threshold,
    )
    artifact_paths = write_manifest_artifacts(
        args.target_class,
        manifest,
        rejected,
        output_dir=args.output_dir,
    )
    print_summary(args.target_class, manifest, rejected, artifact_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

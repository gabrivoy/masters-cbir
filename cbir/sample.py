"""Build a small, committable sample dataset.

The full image pool is ~33 GB and lives outside the repo. To make the system
runnable and testable by anyone who clones it, this module distils a tiny
sample: for a handful of classes it takes the largest (visually clearest)
benchmark crops, writes them as small PNGs under ``cbir/sample_data/crops/``,
and emits a self-contained manifest whose ``image_path`` entries are *relative*
to the repo root. The result is a few megabytes that can be committed.

Because the sample crops are already tight to the object, the manifest's bbox
covers the whole saved crop (x=0, y=0, w=W, h=H): the runtime crop then equals
the saved image, and no absolute path to the original 33 GB pool is needed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cbir.core.manifest import (
    ManifestRecord,
    clipped_crop_box,
    filter_records,
    load_manifest,
)
from cbir.observability import get_logger, timed_event

_log = get_logger("sample")

# Source manifests produced by the frozen mvp/data_prep.py, one per class.
DEFAULT_SOURCE_MANIFESTS: dict[str, Path] = {
    "Traineira": Path("data/cbir/traineira/v1/manifest.jsonl"),
    "Lancha / Iate": Path("data/cbir/lancha_iate/v1/manifest.jsonl"),
    "Rebocador": Path("data/cbir/rebocador/v1/manifest.jsonl"),
    "Navio de Carga Geral": Path("data/cbir/navio_de_carga_geral/v1/manifest.jsonl"),
}


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def _largest_first(records: list[ManifestRecord]) -> list[ManifestRecord]:
    """Sort by bbox area descending, ties broken by item_id for determinism."""
    return sorted(
        records,
        key=lambda r: (-float(r.get("bbox_area", 0.0)), r.item_id),
    )


def build_sample(
    *,
    repo_root: Path,
    output_dir: Path,
    per_class: int = 40,
    source_manifests: dict[str, Path] | None = None,
    split: str = "train",
    max_side: int = 384,
) -> Path:
    """Create the committable sample and return the manifest path.

    For each class, the ``per_class`` largest benchmark crops from ``split`` are
    cropped from their source frame and saved as PNGs. A single manifest.jsonl
    referencing them (with repo-relative paths) is written to ``output_dir``.

    Crops are downscaled so their longest side is at most ``max_side`` pixels.
    CLIP models see 224 px anyway, so 384 keeps the sample crisp while keeping
    the committed footprint to a few megabytes.
    """
    sources = source_manifests or DEFAULT_SOURCE_MANIFESTS
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    lines: list[str] = []
    with timed_event(_log, "sample.build", classes=len(sources), per_class=per_class) as event:
        written = 0
        for target_class, rel_manifest in sources.items():
            manifest_file = repo_root / rel_manifest
            if not manifest_file.exists():
                _log.warning(
                    "source manifest missing; skipping class",
                    extra={"event": "sample.skip", "cls": target_class, "path": str(manifest_file)},
                )
                continue
            records = load_manifest(manifest_file)
            candidates = filter_records(records, split=split, benchmark_only=True)
            candidates = _largest_first(candidates)

            class_slug = _slug(target_class)
            class_dir = crops_dir / class_slug
            class_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for record in candidates:
                if saved >= per_class:
                    break
                source_image = Path(record.image_path)
                if not source_image.exists():
                    continue
                with Image.open(source_image) as handle:
                    image = handle.convert("RGB")
                box = clipped_crop_box(record.bbox_xywh, image.width, image.height, 0.0)
                crop = image.crop(box)
                if crop.width < 8 or crop.height < 8:
                    continue
                if max(crop.width, crop.height) > max_side:
                    crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

                filename = f"{class_slug}_{saved:03d}.png"
                crop.save(class_dir / filename, optimize=True)
                rel_path = (output_dir / "crops" / class_slug / filename).relative_to(repo_root)

                lines.append(
                    _manifest_line(
                        item_id=f"{class_slug}:{split}:{saved}",
                        target_class=target_class,
                        split=split,
                        camera_id=str(record.get("camera_id") or ""),
                        size_bucket=str(record.get("size_bucket") or ""),
                        rel_path=rel_path.as_posix(),
                        width=crop.width,
                        height=crop.height,
                    )
                )
                saved += 1
                written += 1
            _log.info(
                "sampled class",
                extra={"event": "sample.class", "cls": target_class, "saved": saved},
            )

        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        event["written"] = written
    return manifest_path


def _manifest_line(
    *,
    item_id: str,
    target_class: str,
    split: str,
    camera_id: str,
    size_bucket: str,
    rel_path: str,
    width: int,
    height: int,
) -> str:
    import json

    record = {
        "item_id": item_id,
        "target_class": target_class,
        "split": split,
        "idx_in_class_split": int(item_id.rsplit(":", 1)[1]) + 1,
        "camera_id": camera_id,
        "size_bucket": size_bucket,
        "image_path": rel_path,
        "image_filename": Path(rel_path).name,
        # The saved image *is* the crop, so the bbox covers the whole file.
        "bbox_x": 0.0,
        "bbox_y": 0.0,
        "bbox_w": float(width),
        "bbox_h": float(height),
        "bbox_area": float(width * height),
        "is_benchmark_candidate": True,
    }
    return json.dumps(record, ensure_ascii=False)

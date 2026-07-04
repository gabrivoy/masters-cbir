"""The indexer: manifest(s) -> crops -> embeddings -> Milvus.

This is the write side of the system. It reads bbox manifests, embeds each
crop with OpenCLIP, and inserts the vectors + metadata into a Milvus
collection. It also writes a small JSON summary next to the cache directory so
a run is reproducible and inspectable after the fact.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cbir.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_PADDING_RATIO,
    cache_dir,
    resolve_model,
)
from cbir.core.extractor import Embedder
from cbir.core.manifest import (
    ManifestRecord,
    filter_records,
    load_manifests,
    sample_head_per_class,
)
from cbir.core.milvus_client import MilvusClient
from cbir.models import IndexResult
from cbir.observability import get_logger, timed_event

_log = get_logger("index")


def _counts_by_class(records: list[ManifestRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.target_class] = counts.get(record.target_class, 0) + 1
    return dict(sorted(counts.items()))


def _row_from(item: dict[str, Any]) -> dict[str, Any]:
    record: ManifestRecord = item["record"]
    x, y, w, h = record.bbox_xywh
    return {
        "item_id": record.item_id,
        "target_class": record.target_class,
        "split": record.split,
        "camera_id": str(record.get("camera_id") or ""),
        "size_bucket": str(record.get("size_bucket") or ""),
        "image_path": record.image_path,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "embedding": item["embedding"],
    }


def run_index(
    *,
    manifest_paths: list[Path],
    collection_name: str,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    split: str = "train",
    benchmark_only: bool = False,
    per_class: int | None = None,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    batch_size: int = DEFAULT_BATCH_SIZE,
    insert_batch_size: int = 128,
    recreate: bool = True,
    client: MilvusClient | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> IndexResult:
    """Index selected manifest records into a Milvus collection.

    ``progress`` is an optional callback ``(done, total)`` invoked after each
    embedding batch, so the CLI or app can render a bar.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    started = time.time()

    records = load_manifests(manifest_paths)
    selected = filter_records(records, split=split, benchmark_only=benchmark_only)
    selected = sample_head_per_class(selected, per_class=per_class)
    if not selected:
        raise ValueError(
            "No records selected. Check --split / --benchmark-only / --per-class."
        )

    # Resolve the model early so the collection name / dim are known before the
    # (slow) model load, and fail fast on an unknown model name.
    spec = resolve_model(model_name)
    embedder = Embedder(model_name, device=device)
    embedding_dim = embedder.embedding_dim

    client = client or MilvusClient()
    if recreate:
        client.recreate_collection(
            collection_name, embedding_dim, model_name=model_name, model_slug=spec.slug
        )
    else:
        client.ensure_collection(
            collection_name, embedding_dim, model_name=model_name, model_slug=spec.slug
        )

    inserted = 0
    buffer: list[dict[str, Any]] = []
    total = len(selected)
    with timed_event(
        _log,
        "index.run",
        collection=collection_name,
        model=model_name,
        device=embedder.device,
        records=total,
    ) as event:
        for start in range(0, total, batch_size):
            chunk = selected[start : start + batch_size]
            items = embedder.embed_records(
                chunk, batch_size=batch_size, padding_ratio=padding_ratio
            )
            buffer.extend(_row_from(item) for item in items)
            while len(buffer) >= insert_batch_size:
                inserted += client.insert(collection_name, buffer[:insert_batch_size])
                del buffer[:insert_batch_size]
            if progress is not None:
                progress(min(start + batch_size, total), total)
        if buffer:
            inserted += client.insert(collection_name, buffer)

        collection_count = client.count(collection_name)
        event["inserted"] = inserted
        event["count"] = collection_count

    result = IndexResult(
        collection_name=collection_name,
        model_name=model_name,
        model_slug=spec.slug,
        embedding_dim=embedding_dim,
        selected_records=total,
        inserted_count=inserted,
        collection_count=collection_count,
        classes=sorted({r.target_class for r in selected}),
        counts_by_class=_counts_by_class(selected),
        device_resolved=embedder.device,
        split=split,
        benchmark_only=benchmark_only,
        per_class=per_class,
        padding_ratio=padding_ratio,
        duration_seconds=time.time() - started,
        manifest_paths=[str(p) for p in manifest_paths],
    )

    summary_dir = cache_dir() / collection_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    result.summary_path = str(summary_path)
    return result

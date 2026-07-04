from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data_prep import DEFAULT_PADDING_RATIO, load_manifest
from db import DEFAULT_HOST, DEFAULT_PORT, connect, count, ensure_collection, recreate_collection
from db import insert_batch as milvus_insert_batch
from extract import load_model
from extract import extract_batch as extract_embeddings
from sampling import (
    count_by_class,
    count_by_field,
    count_by_split,
    filter_records,
    sample_records_per_class,
)


def output_dir_for_collection(collection_name: str, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / "artifacts" / "ingest" / collection_name


def prepare_rows(extracted_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in extracted_items:
        record = item["record"]
        rows.append(
            {
                "item_id": record["item_id"],
                "annotation_id": record["annotation_id"],
                "target_class": record["target_class"],
                "split": record["split"],
                "image_path": record["image_path"],
                "camera_id": record["camera_id"],
                "bbox_area": record["bbox_area"],
                "size_bucket": record["size_bucket"],
                "is_benchmark_candidate": record["is_benchmark_candidate"],
                "embedding": item["embedding"],
            }
        )
    return rows


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_records_from_manifests(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    for manifest_path in manifest_paths:
        for record in load_manifest(manifest_path):
            item_id = record["item_id"]
            if item_id in seen_item_ids:
                raise ValueError(f"Duplicate item_id across manifests: {item_id}")
            seen_item_ids.add(item_id)
            records.append(record)
    return records


def enqueue_with_backpressure(
    write_queue: queue.Queue[object],
    item: object,
    writer_errors: list[BaseException],
) -> None:
    while True:
        if writer_errors:
            raise RuntimeError("Writer thread failed") from writer_errors[0]
        try:
            write_queue.put(item, timeout=0.5)
            return
        except queue.Full:
            continue


def ingest_records(
    *,
    manifest_paths: list[Path],
    collection_name: str,
    model_name: str,
    device: str,
    split: str,
    benchmark_only: bool,
    padding_ratio: float,
    batch_size: int,
    insert_batch_size: int,
    recreate: bool,
    host: str,
    port: str,
    sample_per_class: int | None = None,
    sample_strategy: str = "stratified",
    sample_seed: int = 42,
    limit: int | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if insert_batch_size <= 0:
        raise ValueError("--insert-batch-size must be positive.")

    started_at = time.time()
    records = load_records_from_manifests(manifest_paths)
    filtered_records = filter_records(
        records,
        split=split,
        benchmark_only=benchmark_only,
    )
    selected_records = sample_records_per_class(
        filtered_records,
        sample_per_class=sample_per_class,
        strategy=sample_strategy,
        seed=sample_seed,
    )
    if limit is not None:
        selected_records = selected_records[:limit]
    if not selected_records:
        raise ValueError("No records selected for ingestion.")

    connect(host=host, port=port)
    model_bundle = load_model(model_name=model_name, device=device)
    embedding_dim = int(model_bundle["embedding_dim"])

    if recreate:
        recreate_collection(collection_name, embedding_dim)
    else:
        ensure_collection(collection_name, embedding_dim)

    write_queue: queue.Queue[object] = queue.Queue(maxsize=4)
    stop_token = object()
    writer_errors: list[BaseException] = []
    writer_state = {"inserted_count": 0}

    def flush_rows(buffer: list[dict[str, Any]]) -> None:
        while len(buffer) >= insert_batch_size:
            chunk = buffer[:insert_batch_size]
            del buffer[:insert_batch_size]
            writer_state["inserted_count"] += milvus_insert_batch(collection_name, chunk)

    def writer() -> None:
        buffer: list[dict[str, Any]] = []
        try:
            while True:
                item = write_queue.get()
                try:
                    if item is stop_token:
                        if buffer:
                            writer_state["inserted_count"] += milvus_insert_batch(collection_name, buffer)
                        return
                    buffer.extend(item)
                    flush_rows(buffer)
                finally:
                    write_queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            writer_errors.append(exc)
            while True:
                try:
                    pending_item = write_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    write_queue.task_done()
                    if pending_item is stop_token:
                        break

    writer_thread = threading.Thread(target=writer, name="milvus-writer", daemon=True)
    writer_thread.start()

    extraction_batches = range(0, len(selected_records), batch_size)
    for start in tqdm(extraction_batches, desc="Extracting and queuing", unit="batch"):
        batch_records = selected_records[start : start + batch_size]
        extracted_items = extract_embeddings(
            model_bundle,
            batch_records,
            batch_size=batch_size,
            device=model_bundle["device"],
            padding_ratio=padding_ratio,
        )
        enqueue_with_backpressure(write_queue, prepare_rows(extracted_items), writer_errors)

    enqueue_with_backpressure(write_queue, stop_token, writer_errors)
    write_queue.join()
    writer_thread.join()
    if writer_errors:
        raise RuntimeError("Writer thread failed") from writer_errors[0]

    duration_seconds = time.time() - started_at
    collection_count = count(collection_name)
    target_classes = sorted({record["target_class"] for record in selected_records})
    summary = {
        "manifest_paths": [str(path) for path in manifest_paths],
        "collection_name": collection_name,
        "target_classes": target_classes,
        "available_records": len(records),
        "filtered_records": len(filtered_records),
        "selected_records": len(selected_records),
        "available_records_by_class": count_by_class(records),
        "filtered_records_by_class": count_by_class(filtered_records),
        "selected_records_by_class": count_by_class(selected_records),
        "selected_records_by_split": count_by_split(selected_records),
        "selected_records_by_camera_id": count_by_field(selected_records, "camera_id"),
        "selected_records_by_size_bucket": count_by_field(selected_records, "size_bucket"),
        "inserted_count": writer_state["inserted_count"],
        "collection_count": collection_count,
        "split": split,
        "benchmark_only": benchmark_only,
        "padding_ratio": padding_ratio,
        "sample_per_class": sample_per_class,
        "sample_strategy": sample_strategy,
        "sample_seed": sample_seed,
        "limit": limit,
        "model_name": model_bundle["model_name"],
        "model_slug": model_bundle["model_slug"],
        "embedding_dim": embedding_dim,
        "device_requested": device,
        "device_resolved": model_bundle["device"],
        "batch_size": batch_size,
        "insert_batch_size": insert_batch_size,
        "recreate": recreate,
        "host": host,
        "port": port,
        "duration_seconds": duration_seconds,
    }
    summary_path = output_dir_for_collection(collection_name) / "summary.json"
    write_summary(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest bbox-level embeddings into Milvus.")
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Path to manifest.jsonl. Repeat for multiclass collections.",
    )
    parser.add_argument("--collection", required=True, help="Target Milvus collection name.")
    parser.add_argument(
        "--model",
        default="openclip-vit-b-32",
        help="Embedding model. Default: openclip-vit-b-32.",
    )
    parser.add_argument("--device", default="cpu", help="Execution device: cpu or cuda.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="train",
        help="Split filter applied to the manifest.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Restrict ingestion to medium+ benchmark candidates.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=DEFAULT_PADDING_RATIO,
        help="Runtime crop padding. Default is 0.0 for bbox-tight crops.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of records encoded per embedding batch.",
    )
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=128,
        help="Number of rows inserted per Milvus batch.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before inserting.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Optional deterministic cap per class after split/benchmark filtering.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=("stratified", "head"),
        default="stratified",
        help="Sampling strategy when --sample-per-class is set.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used by deterministic sampling.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Milvus host.")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Milvus port.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of selected records, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = ingest_records(
        manifest_paths=args.manifest,
        collection_name=args.collection,
        model_name=args.model,
        device=args.device,
        split=args.split,
        benchmark_only=args.benchmark_only,
        padding_ratio=args.padding_ratio,
        batch_size=args.batch_size,
        insert_batch_size=args.insert_batch_size,
        recreate=args.recreate,
        host=args.host,
        port=args.port,
        sample_per_class=args.sample_per_class,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Precomputed embedding cache (Parquet).

Embedding a gallery needs the model (a download + a forward pass per crop).
For a committable demo we snapshot the vectors once and store them next to the
sample, so anyone can reconstruct the Milvus collection in seconds — no model
download, no GPU — with ``cbir seed``.

The cache is a single Parquet file: metadata columns plus an ``embedding``
column holding the float vector as a list. The embedding model name and
dimension travel in the Parquet file's key-value metadata so the seeded
collection carries the same model-consistency guarantee as a fresh index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cbir.core.milvus_client import _BBOX_FIELDS, _META_FIELDS, MilvusClient
from cbir.observability import get_logger, timed_event

_log = get_logger("cache")

_COLUMNS = ["item_id", *_META_FIELDS, *_BBOX_FIELDS]


def export_collection(
    collection: str,
    output_path: Path,
    *,
    model_name: str,
    client: MilvusClient | None = None,
) -> Path:
    """Dump a Milvus collection's vectors + metadata to a Parquet file."""
    client = client or MilvusClient()
    data = client.fetch_all(collection)
    embeddings: np.ndarray = data["embeddings"]
    rows: list[dict[str, Any]] = data["rows"]
    if embeddings.shape[0] == 0:
        raise ValueError(f"Collection {collection!r} is empty; nothing to export.")

    columns = {name: [row.get(name) for row in rows] for name in _COLUMNS}
    columns["embedding"] = [vec.tolist() for vec in embeddings]
    table = pa.table(columns)
    table = table.replace_schema_metadata(
        {
            b"cbir.model_name": model_name.encode(),
            b"cbir.embedding_dim": str(embeddings.shape[1]).encode(),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    _log.info(
        "exported cache",
        extra={
            "event": "cache.export",
            "collection": collection,
            "rows": len(rows),
            "path": str(output_path),
        },
    )
    return output_path


def seed_collection(
    parquet_path: Path,
    collection: str,
    *,
    client: MilvusClient | None = None,
    recreate: bool = True,
) -> int:
    """Reconstruct a Milvus collection from a Parquet cache (no model needed)."""
    table = pq.read_table(parquet_path)
    metadata = table.schema.metadata or {}
    model_name = metadata.get(b"cbir.model_name", b"").decode()
    embedding_dim = int(metadata.get(b"cbir.embedding_dim", b"0").decode())
    if embedding_dim == 0:
        raise ValueError(f"{parquet_path} has no embedding dimension metadata.")

    from cbir.config import resolve_model

    slug = resolve_model(model_name).slug if model_name else ""
    client = client or MilvusClient()
    with timed_event(_log, "cache.seed", collection=collection, model=model_name) as event:
        if recreate:
            client.recreate_collection(
                collection, embedding_dim, model_name=model_name, model_slug=slug
            )
        else:
            client.ensure_collection(
                collection, embedding_dim, model_name=model_name, model_slug=slug
            )
        records = table.to_pylist()
        rows = [{**record, "embedding": record["embedding"]} for record in records]
        inserted = client.insert(collection, rows)
        event["inserted"] = inserted
    return inserted

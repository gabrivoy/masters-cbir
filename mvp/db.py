from __future__ import annotations

from typing import Any

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

DEFAULT_ALIAS = "default"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "19530"


def connect(host: str = DEFAULT_HOST, port: str = DEFAULT_PORT, alias: str = DEFAULT_ALIAS) -> str:
    connections.connect(alias=alias, host=host, port=port)
    return alias


def collection_schema(embedding_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(
            name="item_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=128,
        ),
        FieldSchema(name="annotation_id", dtype=DataType.INT64),
        FieldSchema(name="target_class", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="split", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="camera_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="bbox_area", dtype=DataType.FLOAT),
        FieldSchema(name="size_bucket", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="is_benchmark_candidate", dtype=DataType.BOOL),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]
    return CollectionSchema(fields=fields, description="CBIR bbox-level MVP collection")


def get_collection(name: str) -> Collection:
    return Collection(name=name)


def recreate_collection(name: str, embedding_dim: int) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)
    collection = Collection(name=name, schema=collection_schema(embedding_dim))
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "FLAT", "params": {}},
    )
    collection.load()
    return collection


def ensure_collection(name: str, embedding_dim: int) -> Collection:
    if utility.has_collection(name):
        collection = Collection(name=name)
        collection.load()
        return collection
    return recreate_collection(name, embedding_dim)


def insert_batch(collection_name: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    collection = Collection(name=collection_name)
    columns = [
        [row["item_id"] for row in rows],
        [int(row["annotation_id"]) for row in rows],
        [row["target_class"] for row in rows],
        [row["split"] for row in rows],
        [row["image_path"] for row in rows],
        [row["camera_id"] for row in rows],
        [float(row["bbox_area"]) for row in rows],
        [row["size_bucket"] for row in rows],
        [bool(row["is_benchmark_candidate"]) for row in rows],
        [np.asarray(row["embedding"], dtype=np.float32).tolist() for row in rows],
    ]
    mutation = collection.insert(columns)
    collection.flush()
    return mutation.insert_count


def count(collection_name: str) -> int:
    collection = Collection(name=collection_name)
    collection.flush()
    return int(collection.num_entities)


def search(
    collection_name: str,
    query_vector: np.ndarray,
    top_k: int,
    *,
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    collection = Collection(name=collection_name)
    collection.load()
    fields = output_fields or [
        "annotation_id",
        "target_class",
        "split",
        "image_path",
        "camera_id",
        "bbox_area",
        "size_bucket",
        "is_benchmark_candidate",
    ]
    results = collection.search(
        data=[np.asarray(query_vector, dtype=np.float32).tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=top_k,
        output_fields=fields,
    )
    formatted: list[dict[str, Any]] = []
    for hit in results[0]:
        entity = hit.entity
        payload = {
            "item_id": str(hit.id),
            "score": float(hit.distance),
        }
        for field in fields:
            payload[field] = entity.get(field)
        formatted.append(payload)
    return formatted

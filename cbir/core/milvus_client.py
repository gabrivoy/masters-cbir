"""Milvus access layer.

A thin object wrapper around pymilvus that the indexer and the visualizer both
use. Collections store one vector per bounding-box item plus the metadata the
visualizer needs to colour and label points. Search uses cosine similarity over
the normalized embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from cbir.config import MILVUS_ALIAS, MILVUS_HOST, MILVUS_PORT

# Metadata fields carried alongside each vector. Chosen to be exactly what the
# visualizer needs: class (colour), split/camera/size_bucket (facets), and the
# image + bbox so a hit can be cropped and shown.
_META_FIELDS = (
    "target_class",
    "split",
    "camera_id",
    "size_bucket",
    "image_path",
)
_BBOX_FIELDS = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")


class MilvusClient:
    """Connection-scoped helper for one Milvus instance."""

    def __init__(
        self,
        host: str = MILVUS_HOST,
        port: str = MILVUS_PORT,
        alias: str = MILVUS_ALIAS,
    ) -> None:
        self.alias = alias
        connections.connect(alias=alias, host=host, port=port)

    # schema / lifecycle
    @staticmethod
    def _schema(
        embedding_dim: int,
        description: str = "CBIR bbox-level collection",
    ) -> CollectionSchema:
        fields = [
            FieldSchema(
                "item_id", DataType.VARCHAR, is_primary=True, auto_id=False, max_length=160
            ),
            FieldSchema("target_class", DataType.VARCHAR, max_length=128),
            FieldSchema("split", DataType.VARCHAR, max_length=16),
            FieldSchema("camera_id", DataType.VARCHAR, max_length=64),
            FieldSchema("size_bucket", DataType.VARCHAR, max_length=64),
            FieldSchema("image_path", DataType.VARCHAR, max_length=512),
            FieldSchema("bbox_x", DataType.FLOAT),
            FieldSchema("bbox_y", DataType.FLOAT),
            FieldSchema("bbox_w", DataType.FLOAT),
            FieldSchema("bbox_h", DataType.FLOAT),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        return CollectionSchema(fields, description=description)

    def recreate_collection(
        self,
        name: str,
        embedding_dim: int,
        *,
        model_name: str = "",
        model_slug: str = "",
    ) -> Collection:
        """Drop the collection if present and create a fresh one (FLAT/COSINE).

        The embedding model that produced the vectors is recorded as a
        collection property so the visualizer can later refuse to embed a query
        with a different model (see ``model_of``). This is the guarantee that
        index-time and query-time embeddings are always comparable.
        """
        if utility.has_collection(name, using=self.alias):
            utility.drop_collection(name, using=self.alias)
        description = f"CBIR bbox-level collection | model={model_name or 'unknown'}"
        schema = self._schema(embedding_dim, description=description)
        collection = Collection(name=name, schema=schema, using=self.alias)
        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "COSINE", "index_type": "FLAT", "params": {}},
        )
        if model_name:
            collection.set_properties(
                {"cbir.model_name": model_name, "cbir.model_slug": model_slug}
            )
        collection.load()
        return collection

    def ensure_collection(
        self,
        name: str,
        embedding_dim: int,
        *,
        model_name: str = "",
        model_slug: str = "",
    ) -> Collection:
        """Load an existing collection or create it if missing."""
        if utility.has_collection(name, using=self.alias):
            collection = Collection(name=name, using=self.alias)
            collection.load()
            return collection
        return self.recreate_collection(
            name, embedding_dim, model_name=model_name, model_slug=model_slug
        )

    def model_of(self, name: str) -> str:
        """Return the embedding-model name recorded on a collection.

        Reads back the ``cbir.model_name`` property set at creation time. Falls
        back to parsing the collection description if the property is absent
        (e.g. an older collection). Returns an empty string if unknown.
        """
        collection = Collection(name=name, using=self.alias)
        properties = getattr(collection, "properties", {}) or {}
        recorded = properties.get("cbir.model_name")
        if recorded:
            return str(recorded)
        description = collection.description or ""
        marker = "model="
        if marker in description:
            candidate = description.split(marker, 1)[1].strip()
            if candidate and candidate != "unknown":
                return candidate
        return ""

    def has_collection(self, name: str) -> bool:
        return bool(utility.has_collection(name, using=self.alias))

    def list_collections(self) -> list[str]:
        return list(utility.list_collections(using=self.alias))

    def count(self, name: str) -> int:
        collection = Collection(name=name, using=self.alias)
        collection.flush()
        return int(collection.num_entities)

    # writes
    def insert(self, name: str, rows: list[dict[str, Any]]) -> int:
        """Insert rows (each a dict with the schema fields + embedding)."""
        if not rows:
            return 0
        collection = Collection(name=name, using=self.alias)
        columns = [
            [str(row["item_id"]) for row in rows],
            [str(row["target_class"]) for row in rows],
            [str(row["split"]) for row in rows],
            [str(row["camera_id"]) for row in rows],
            [str(row["size_bucket"]) for row in rows],
            [str(row["image_path"]) for row in rows],
            [float(row["bbox_x"]) for row in rows],
            [float(row["bbox_y"]) for row in rows],
            [float(row["bbox_w"]) for row in rows],
            [float(row["bbox_h"]) for row in rows],
            [np.asarray(row["embedding"], dtype=np.float32).tolist() for row in rows],
        ]
        mutation = collection.insert(columns)
        collection.flush()
        return int(mutation.insert_count)

    # reads
    def fetch_all(self, name: str) -> dict[str, Any]:
        """Load every item's embedding + metadata for offline projection.

        Returns a dict with an ``embeddings`` (N, dim) float32 array and a
        parallel ``rows`` list of metadata dicts (same order). Milvus caps a
        single query at 16384 rows, so we page with an item_id keyset cursor.
        """
        collection = Collection(name=name, using=self.alias)
        collection.load()
        output_fields = ["item_id", *_META_FIELDS, *_BBOX_FIELDS, "embedding"]

        rows: list[dict[str, Any]] = []
        embeddings: list[np.ndarray] = []
        page = 16384
        cursor = ""
        while True:
            hits = collection.query(
                expr=f'item_id > "{cursor}"',
                output_fields=output_fields,
                limit=page,
                consistency_level="Strong",
            )
            if not hits:
                break
            hits.sort(key=lambda hit: hit["item_id"])
            for hit in hits:
                embeddings.append(np.asarray(hit.pop("embedding"), dtype=np.float32))
                rows.append(hit)
            cursor = hits[-1]["item_id"]
            if len(hits) < page:
                break

        matrix = (
            np.vstack(embeddings).astype(np.float32)
            if embeddings
            else np.zeros((0, 0), dtype=np.float32)
        )
        return {"embeddings": matrix, "rows": rows}

    def search(
        self,
        name: str,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Cosine top-k search; returns hits with score + metadata."""
        collection = Collection(name=name, using=self.alias)
        collection.load()
        output_fields = [*_META_FIELDS, *_BBOX_FIELDS]
        results = collection.search(
            data=[np.asarray(query_vector, dtype=np.float32).tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=output_fields,
        )
        formatted: list[dict[str, Any]] = []
        for hit in results[0]:
            payload: dict[str, Any] = {"item_id": str(hit.id), "score": float(hit.distance)}
            for field in output_fields:
                payload[field] = hit.entity.get(field)
            formatted.append(payload)
        return formatted

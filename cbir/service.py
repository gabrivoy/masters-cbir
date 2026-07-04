"""Application service layer.

This is the orchestration the API and (indirectly) the CLI share. It wires the
backend pieces together — Milvus, the embedder, PCA projection, KNN — and
caches the expensive artefacts (loaded models, fitted projections) so repeated
requests are cheap.

It enforces the system's central guarantee: a query is always embedded with the
*same* model that built the collection it is searched against.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

from cbir.core.extractor import Embedder
from cbir.core.milvus_client import MilvusClient
from cbir.knn import predict_class
from cbir.models import (
    BBox,
    CollectionInfo,
    ProjectionPoint,
    ProjectResponse,
    QueryResponse,
    SearchHit,
)
from cbir.observability import get_logger, timed_event
from cbir.viz.projection import ProjectionModel, fit_projection

_log = get_logger("service")


class ModelMismatchError(RuntimeError):
    """Raised when a query would be embedded with a different model than the
    one that built the target collection."""


class CBIRService:
    """Stateful facade over the backend, safe to share across requests.

    Caches embedders (per model name) and projections (per collection +
    n_components) so the first request pays the cost and the rest are fast.
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = device
        self.client = MilvusClient()
        self._embedders: dict[str, Embedder] = {}
        # projection cache: (collection, n_components) -> (model, rows)
        self._projections: dict[tuple[str, int], tuple[ProjectionModel, list[dict]]] = {}

    # --- models -----------------------------------------------------------
    def embedder(self, model_name: str) -> Embedder:
        """Return a cached embedder for ``model_name``, loading it on first use."""
        if model_name not in self._embedders:
            with timed_event(_log, "service.load_model", model=model_name, device=self.device):
                self._embedders[model_name] = Embedder(model_name, device=self.device)
        return self._embedders[model_name]

    def model_for_collection(self, collection: str) -> str:
        """The embedding model recorded on a collection (empty if unknown)."""
        return self.client.model_of(collection)

    # --- collections ------------------------------------------------------
    def list_collections(self) -> list[CollectionInfo]:
        infos: list[CollectionInfo] = []
        for name in self.client.list_collections():
            infos.append(
                CollectionInfo(
                    name=name,
                    model_name=self.client.model_of(name),
                    count=self.client.count(name),
                )
            )
        return infos

    # --- projection -------------------------------------------------------
    def _load_projection(
        self, collection: str, n_components: int
    ) -> tuple[ProjectionModel, list[dict]]:
        key = (collection, n_components)
        if key not in self._projections:
            data = self.client.fetch_all(collection)
            embeddings: np.ndarray = data["embeddings"]
            rows: list[dict] = data["rows"]
            if embeddings.shape[0] == 0:
                raise ValueError(f"Collection {collection!r} is empty; nothing to project.")
            with timed_event(
                _log, "service.fit_projection", collection=collection, n=embeddings.shape[0]
            ):
                model = fit_projection(embeddings, n_components)
            self._projections[key] = (model, rows)
        return self._projections[key]

    def project(self, collection: str, n_components: int = 3) -> ProjectResponse:
        """Project the whole gallery to ``n_components`` PCA dimensions."""
        model, rows = self._load_projection(collection, n_components)
        points = [
            ProjectionPoint(
                item_id=row["item_id"],
                target_class=row.get("target_class", ""),
                split=row.get("split", ""),
                camera_id=row.get("camera_id", ""),
                size_bucket=row.get("size_bucket", ""),
                image_path=row.get("image_path", ""),
                bbox=_bbox_from_row(row),
                coords=[float(v) for v in coords],
            )
            for row, coords in zip(rows, model.coords, strict=True)
        ]
        return ProjectResponse(
            collection=collection,
            model_name=self.model_for_collection(collection),
            n_components=n_components,
            explained_variance_ratio=[float(v) for v in model.explained_variance_ratio],
            cumulative_variance=model.cumulative_variance,
            points=points,
        )

    # --- query ------------------------------------------------------------
    def query(
        self,
        collection: str,
        image: Image.Image,
        *,
        top_k: int = 10,
        n_components: int = 3,
        weighted: bool = True,
        model_name: str | None = None,
    ) -> QueryResponse:
        """Embed an uploaded image, search, project it, and predict its class.

        The model is taken from the collection unless explicitly overridden. An
        override that disagrees with the collection's model is rejected: mixing
        embedding spaces would make the distances meaningless.
        """
        collection_model = self.model_for_collection(collection)
        chosen = model_name or collection_model
        if not chosen:
            raise ValueError(
                f"Collection {collection!r} has no recorded model and none was provided."
            )
        if collection_model and model_name and model_name != collection_model:
            raise ModelMismatchError(
                f"Collection {collection!r} was built with {collection_model!r}, "
                f"but the query requested {model_name!r}. Embeddings would be incomparable."
            )

        embedder = self.embedder(chosen)
        with timed_event(
            _log, "service.query", collection=collection, model=chosen, top_k=top_k
        ) as event:
            query_vec = embedder.embed_image(image.convert("RGB"))
            raw_hits = self.client.search(collection, query_vec, top_k=top_k)
            hits = [_search_hit(raw, rank) for rank, raw in enumerate(raw_hits, start=1)]
            prediction = predict_class(hits, weighted=weighted)

            projection, _ = self._load_projection(collection, n_components)
            query_coords = projection.transform(query_vec)[0]
            event["predicted"] = prediction.predicted_class or "none"
            event["confidence"] = prediction.confidence

        return QueryResponse(
            collection=collection,
            model_name=chosen,
            query_coords=[float(v) for v in query_coords],
            hits=hits,
            prediction=prediction,
        )


def _bbox_from_row(row: dict) -> BBox | None:
    keys = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    if not all(k in row for k in keys):
        return None
    return BBox(x=row["bbox_x"], y=row["bbox_y"], w=row["bbox_w"], h=row["bbox_h"])


def _search_hit(raw: dict, rank: int) -> SearchHit:
    return SearchHit(
        item_id=raw["item_id"],
        target_class=raw.get("target_class", ""),
        split=raw.get("split", ""),
        camera_id=raw.get("camera_id", ""),
        size_bucket=raw.get("size_bucket", ""),
        image_path=raw.get("image_path", ""),
        bbox=_bbox_from_row(raw),
        score=float(raw["score"]),
        rank=rank,
    )


@lru_cache(maxsize=1)
def get_service(device: str = "auto") -> CBIRService:
    """Process-wide singleton service (used by the API)."""
    return CBIRService(device=device)

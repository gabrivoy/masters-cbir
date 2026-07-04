"""Shared Pydantic v2 models.

These are the data contracts that travel across the whole system: the backend
produces them, the FastAPI layer serializes them, and the Streamlit frontend
consumes them. Keeping them in one place guarantees the three layers agree on
shape and field names.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """A bounding box in COCO xywh pixel coordinates."""

    x: float
    y: float
    w: float
    h: float


class GalleryItem(BaseModel):
    """One indexed item as seen by the frontend (metadata, no raw vector)."""

    item_id: str
    target_class: str
    split: str = ""
    camera_id: str = ""
    size_bucket: str = ""
    image_path: str = ""
    bbox: BBox | None = None


class ProjectionPoint(GalleryItem):
    """A gallery item with its PCA coordinates for plotting."""

    coords: list[float] = Field(description="PCA coordinates, length = n_components")


class SearchHit(GalleryItem):
    """A retrieval neighbour with its cosine similarity to the query."""

    score: float = Field(description="Cosine similarity in [-1, 1]")
    rank: int = Field(ge=1)


class ClassVote(BaseModel):
    """Aggregated KNN evidence for one candidate class."""

    target_class: str
    count: int = Field(ge=0)
    weight: float = Field(description="Summed similarity of neighbours in this class")


class Prediction(BaseModel):
    """The KNN class prediction for a query image."""

    predicted_class: str | None
    confidence: float = Field(ge=0.0, le=1.0, description="Winning vote share")
    votes: list[ClassVote] = Field(default_factory=list, description="Sorted, strongest first")
    k: int = Field(ge=0)
    weighted: bool


class CollectionInfo(BaseModel):
    """Summary of an indexed collection, including the model that built it."""

    name: str
    model_name: str = Field(default="", description="Embedding model recorded on the collection")
    count: int = Field(ge=0)


class IndexResult(BaseModel):
    """Outcome of an indexing run, serialized to summary.json."""

    model_config = ConfigDict(protected_namespaces=())  # allow model_name/model_slug fields

    collection_name: str
    model_name: str
    model_slug: str = ""
    embedding_dim: int
    selected_records: int
    inserted_count: int
    collection_count: int
    classes: list[str] = Field(default_factory=list)
    counts_by_class: dict[str, int] = Field(default_factory=dict)
    device_resolved: str = ""
    split: str = ""
    benchmark_only: bool = False
    per_class: int | None = None
    padding_ratio: float = 0.0
    duration_seconds: float = 0.0
    manifest_paths: list[str] = Field(default_factory=list)
    summary_path: str = ""


class ProjectResponse(BaseModel):
    """Everything the frontend needs to draw the gallery scatter."""

    model_config = ConfigDict(protected_namespaces=())

    collection: str
    model_name: str
    n_components: int
    explained_variance_ratio: list[float]
    cumulative_variance: float
    points: list[ProjectionPoint]


class QueryResponse(BaseModel):
    """Result of embedding + searching + predicting for one uploaded query."""

    model_config = ConfigDict(protected_namespaces=())

    collection: str
    model_name: str
    query_coords: list[float] = Field(description="Query projected into the gallery PCA space")
    hits: list[SearchHit]
    prediction: Prediction

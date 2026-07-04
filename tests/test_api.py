"""API tests using a fake service, so no Milvus or torch is required.

These verify the HTTP contract and — most importantly — that the
model-consistency guarantee surfaces as a 409 Conflict.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cbir.models import (
    CollectionInfo,
    Prediction,
    ProjectionPoint,
    ProjectResponse,
    QueryResponse,
    SearchHit,
)
from cbir.service import ModelMismatchError


class FakeService:
    """Stand-in for CBIRService with canned, deterministic responses."""

    def list_collections(self) -> list[CollectionInfo]:
        return [CollectionInfo(name="demo", model_name="openclip-vit-b-32", count=2)]

    def project(self, name: str, n_components: int = 3) -> ProjectResponse:
        if name != "demo":
            raise ValueError("Unknown collection.")
        pts = [
            ProjectionPoint(item_id="a", target_class="Traineira", coords=[0.1, 0.2, 0.3]),
            ProjectionPoint(item_id="b", target_class="Lancha", coords=[-0.1, 0.0, 0.1]),
        ]
        return ProjectResponse(
            collection=name,
            model_name="openclip-vit-b-32",
            n_components=n_components,
            explained_variance_ratio=[0.5, 0.3, 0.1],
            cumulative_variance=0.9,
            points=pts,
        )

    def query(self, name, image, *, top_k=10, n_components=3, weighted=True, model_name=None):
        if model_name and model_name != "openclip-vit-b-32":
            raise ModelMismatchError("model mismatch")
        hits = [SearchHit(item_id="a", target_class="Traineira", score=0.9, rank=1)]
        return QueryResponse(
            collection=name,
            model_name="openclip-vit-b-32",
            query_coords=[0.1, 0.2, 0.3],
            hits=hits,
            prediction=Prediction(
                predicted_class="Traineira", confidence=1.0, votes=[], k=1, weighted=weighted
            ),
        )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    import cbir.api.app as app_module

    monkeypatch.setattr(app_module, "get_service", lambda device="auto": FakeService())
    return TestClient(app_module.create_app())


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 90, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_collections_report_their_model(client: TestClient) -> None:
    data = client.get("/collections").json()
    assert data[0]["name"] == "demo"
    assert data[0]["model_name"] == "openclip-vit-b-32"


def test_project_returns_points_with_coords(client: TestClient) -> None:
    data = client.get("/collections/demo/project", params={"n_components": 3}).json()
    assert len(data["points"]) == 2
    assert len(data["points"][0]["coords"]) == 3
    assert data["cumulative_variance"] == pytest.approx(0.9)


def test_project_unknown_collection_is_404(client: TestClient) -> None:
    assert client.get("/collections/nope/project").status_code == 404


def test_query_happy_path_predicts_class(client: TestClient) -> None:
    response = client.post(
        "/collections/demo/query",
        files={"file": ("q.png", _png_bytes(), "image/png")},
        data={"top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["predicted_class"] == "Traineira"
    assert body["hits"][0]["rank"] == 1


def test_query_rejects_invalid_image(client: TestClient) -> None:
    response = client.post(
        "/collections/demo/query",
        files={"file": ("q.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400

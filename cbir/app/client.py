"""HTTP client the frontend uses to reach the CBIR API.

A tiny wrapper over ``requests`` that returns parsed Pydantic models, so the
Streamlit app never touches raw JSON. Keeping it separate keeps the frontend
declarative and makes the transport layer independently testable.
"""

from __future__ import annotations

from urllib.parse import quote

import requests

from cbir.models import CollectionInfo, ProjectResponse, QueryResponse


class APIError(RuntimeError):
    """A non-2xx response from the API, carrying the server's detail message."""


class CBIRClient:
    """Thin synchronous client bound to one API base URL."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _detail(self, response: requests.Response) -> str:
        try:
            return str(response.json().get("detail", response.text))
        except Exception:  # noqa: BLE001 - best-effort error extraction
            return response.text

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/models", timeout=self.timeout)
        response.raise_for_status()
        return list(response.json()["models"])

    def collections(self) -> list[CollectionInfo]:
        response = requests.get(f"{self.base_url}/collections", timeout=self.timeout)
        response.raise_for_status()
        return [CollectionInfo.model_validate(item) for item in response.json()]

    def project(self, collection: str, n_components: int = 3) -> ProjectResponse:
        response = requests.get(
            f"{self.base_url}/collections/{quote(collection)}/project",
            params={"n_components": n_components},
            timeout=self.timeout,
        )
        if not response.ok:
            raise APIError(self._detail(response))
        return ProjectResponse.model_validate(response.json())

    def query(
        self,
        collection: str,
        image_bytes: bytes,
        filename: str,
        *,
        top_k: int = 10,
        n_components: int = 3,
        weighted: bool = True,
    ) -> QueryResponse:
        response = requests.post(
            f"{self.base_url}/collections/{quote(collection)}/query",
            files={"file": (filename, image_bytes)},
            data={"top_k": top_k, "n_components": n_components, "weighted": str(weighted).lower()},
            timeout=self.timeout,
        )
        if not response.ok:
            raise APIError(self._detail(response))
        return QueryResponse.model_validate(response.json())

    def crop_url(self, image_path: str) -> str:
        """URL that serves a gallery/hit image by its manifest path."""
        return f"{self.base_url}/crop?image_path={quote(image_path)}"

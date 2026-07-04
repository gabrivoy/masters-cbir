"""FastAPI application: the middle tier between the backend and the frontend.

Endpoints are thin — they parse the request, delegate to :class:`CBIRService`,
and return Pydantic models. Every request is logged as one wide event carrying
method, path, status, and latency, so the API's behaviour is observable from a
single line per call.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from cbir.config import MODEL_SPECS
from cbir.core.extractor import resolve_image_path
from cbir.models import CollectionInfo, ProjectResponse, QueryResponse
from cbir.observability import configure_logging, get_logger, log_event, log_startup
from cbir.service import ModelMismatchError, get_service

_log = get_logger("api")


def create_app(device: str = "auto") -> FastAPI:
    """Build the FastAPI app bound to a service on the given device."""
    configure_logging()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log_startup("api", device=device, models=len(MODEL_SPECS))
        yield

    app = FastAPI(
        title="CBIR API",
        version="0.1.0",
        summary="Index and explore vessel bbox embeddings in a vector space.",
        lifespan=lifespan,
    )
    service = get_service(device)

    @app.middleware("http")
    async def wide_event_logging(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        log_event(
            _log,
            "api.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    # --- meta -------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models")
    def models() -> dict[str, list[str]]:
        return {"models": sorted(MODEL_SPECS)}

    @app.get("/collections", response_model=list[CollectionInfo])
    def collections() -> list[CollectionInfo]:
        return service.list_collections()

    # --- projection -------------------------------------------------------
    @app.get("/collections/{name}/project", response_model=ProjectResponse)
    def project(
        name: str,
        n_components: int = Query(default=3, ge=2, le=3),
    ) -> ProjectResponse:
        try:
            return service.project(name, n_components=n_components)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- query ------------------------------------------------------------
    @app.post("/collections/{name}/query", response_model=QueryResponse)
    async def query(
        name: str,
        file: UploadFile = File(...),
        top_k: int = Form(default=10),
        n_components: int = Form(default=3),
        weighted: bool = Form(default=True),
    ) -> QueryResponse:
        raw = await file.read()
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a valid image."
            ) from exc
        try:
            return service.query(
                name, image, top_k=top_k, n_components=n_components, weighted=weighted
            )
        except ModelMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- crop images ------------------------------------------------------
    @app.get("/crop")
    def crop(image_path: str = Query(...)):
        """Serve a gallery/hit image by its manifest path so the FE can show it.

        Only paths that resolve to an existing file are served; anything else
        is a 404. This lets the frontend render thumbnails without shared disk.
        """
        resolved = resolve_image_path(image_path)
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="Image not found.")
        return FileResponse(resolved)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        _log.error("unhandled error", extra={"event": "api.error", "path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "Internal error."})

    return app

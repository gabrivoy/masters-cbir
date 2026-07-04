# AGENTS.md

Guidance for agentic coding tools (Claude Code, Codex, Copilot, OpenCode) working in this repository.

## What this is

A Content-Based Image Retrieval (CBIR) system for vessel bounding-box crops
(TecGraf PUC / Embraer; images from cameras over Guanabara Bay). It indexes
image embeddings into Milvus and lets you explore the vector space: project the
gallery to 2D/3D with PCA, drop in a query image, and see which class a KNN vote
over its nearest neighbours would assign it — the retrieval-based auto-labeling
idea, made visual.

The central guarantee: **a query is always embedded with the same model that
built the collection it is searched against.** The model is recorded on the
Milvus collection at index time and enforced by the service layer.

## Architecture (three tiers, cleanly separated)

- **Backend** (`cbir/core`, `cbir/index`, `cbir/viz`, `cbir/knn.py`,
  `cbir/service.py`) — manifests, OpenCLIP extraction, Milvus access, PCA
  projection, KNN prediction, and the orchestrating service.
- **API** (`cbir/api`) — a thin FastAPI layer over the service.
- **Frontend** (`cbir/app`) — a Streamlit app that talks *only* to the API.

Shared data contracts live in `cbir/models.py` (Pydantic v2). Structured
logging (stdlib `logging`, wide events) lives in `cbir/observability.py`.

## Development

- Python 3.13, managed with `uv`.
- Run commands: `uv run cbir --help` (subcommands: `sample`, `index`, `export`,
  `seed`, `api`, `app`).
- Lint: `uv run ruff check cbir/ tests/`
- Type check: `uv run mypy cbir/`
- Test: `uv run pytest`

All three must stay green. The BE core is pure and unit-tested without Milvus or
torch; the API is tested with a fake service.

## Conventions

- Pydantic v2 for data models (not dataclasses).
- Wide-event logging: one structured log line per operation (`log_event` /
  `timed_event`), not many fragmented lines.
- Device resolution is `auto` by default (CUDA → Apple MPS → multi-thread CPU).
- Keep the backend simple and linear; no premature threading.

## Boundaries

- `archive/mvp/` is a frozen proof-of-concept. **Never import from it.**
- `archive/docs/` and `ROADMAP.md` are historical reference only.
- `data/` (the ~33 GB image pool) and `artifacts/` are never committed. The
  runnable demo lives in `cbir/sample_data/` (crops + manifest + a precomputed
  embeddings Parquet).

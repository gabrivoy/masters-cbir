# AGENTS.md

Guidance for agentic coding tools (Claude Code, Codex, Copilot, OpenCode) working in this repository.

## What this is

A Content-Based Image Retrieval (CBIR) system for image bounding-box crops. It
indexes image embeddings into Milvus and lets you explore the vector space:
project the gallery to 2D/3D with PCA, drop in a query image, and see which
class a KNN vote over its nearest neighbours would assign it (the
retrieval-based auto-labeling idea, made visual).

The system is generic. The committed sample data comes from a vessel-monitoring
case study (TecGraf PUC-Rio / Embraer, Guanabara Bay), but nothing in the code
is domain-specific.

The central guarantee: **a query is always embedded with the same model that
built the collection it is searched against.** The model is recorded on the
Milvus collection at index time and enforced by the service layer.

## Architecture (three tiers, cleanly separated)

- **Backend**: `cbir/core` (manifests, OpenCLIP extraction, Milvus access),
  `cbir/analysis` (PCA projection, KNN prediction), `cbir/index` (indexer and
  Parquet cache), `cbir/service` (orchestration).
- **API**: `cbir/api`, a thin FastAPI layer over the service.
- **Frontend**: `cbir/app`, a Streamlit app that talks *only* to the API.

Shared foundation lives in `cbir/common`: `models.py` (Pydantic v2 data
contracts) and `observability.py` (stdlib `logging` with wide events).
`cbir/config.py` holds configuration; `cbir/cli.py` is the Typer entry point;
`cbir/data/sample.py` builds the committable sample.

## Development

- Python 3.13, managed with `uv`.
- Run commands: `uv run cbir --help` (subcommands: `sample`, `index`, `export`,
  `seed`, `api`, `app`).
- Lint: `uv run ruff check cbir/ tests/`
- Type check: `uv run mypy cbir/`
- Test: `uv run pytest`

All three must stay green (strict ruff and mypy). The backend core is pure and
unit-tested without Milvus or torch; the API is tested with a fake service.

## Conventions

- Pydantic v2 for data models (not dataclasses).
- Wide-event logging: one structured log line per operation (`log_event` /
  `timed_event`), not many fragmented lines.
- Device resolution is `auto` by default (CUDA, then Apple MPS, then
  multi-thread CPU).
- Keep the backend simple and linear; no premature threading.
- Verbose variable names; no banner comments; no em-dashes (use `:` or `()`).

## Boundaries

- `archive/mvp/` is a frozen proof-of-concept. **Never import from it.**
- `archive/docs/` and `ROADMAP.md` are historical reference only.
- `data/` (the ~33 GB image pool) and `artifacts/` are never committed. The
  runnable demo lives in `cbir/sample_data/` (crops + manifest + a precomputed
  embeddings Parquet).

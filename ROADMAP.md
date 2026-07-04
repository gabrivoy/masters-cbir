# CBIR System Roadmap — Deliverable 1 (Programming Project)

**Timeline:** March 25 – May 31, 2026 (~10 weeks)
**Goal:** Fully functional CBIR system with pluggable extractors, vector search, API, CLI, dashboard, evaluation, and auto-labeling heuristics — all containerized.

---

## MVP / Full System Relationship

The project starts with an MVP (Epic 0) that validates the entire pipeline end-to-end in a minimal, exploratory way. The MVP lives in the `mvp/` folder and follows these rules:

- **`mvp/` is self-contained and standalone.** It must work independently, with no imports from the main `cbir/` package.
- **`mvp/` is frozen after Epic 0.** Once the MVP is complete and validated, the folder is not modified. It serves as a reference and proof-of-concept.
- **The main system draws inspiration from `mvp/`, but does not depend on it.** Patterns, learnings, and code snippets from the MVP can inform the architecture in `cbir/`, but there should be no cross-imports or shared state.
- **MVP structure:** A few `.py` utility files for reusable logic (extraction, Milvus helpers, etc.) and a Jupyter notebook that drives the flow. Keep it simple — no abstractions, no classes, just functions and a notebook.
- **MVP services (Milvus, etc.) are set up via Docker Compose** in a way that carries forward to the full system. The `docker-compose.yml` at the repo root is shared — Epic 0 starts it, and later epics extend it with more services.

---

## Epic 0: MVP — End-to-End Proof of Concept

**Dates:** Mar 25 – Apr 4 (Week 1–2)

> All code lives in `mvp/`. Structure: utility `.py` files + a Jupyter notebook for the flow.

### Story 0.1: MVP environment and services setup

- [ ] Create `mvp/` folder structure:
  - `mvp/extract.py` — embedding extraction utilities
  - `mvp/db.py` — Milvus connection and operations
  - `mvp/notebook.ipynb` — main flow notebook
- [ ] Add core dependencies to `pyproject.toml`: `torch`, `torchvision`, `Pillow`, `pymilvus`, `jupyter`, `tqdm`
- [ ] Set up `docker-compose.yml` at repo root with Milvus standalone service (etcd + minio + milvus)
  - Use named volumes for persistence
  - Expose standard ports (19530 for gRPC, 9091 for health)
  - This compose file will be extended in later epics — design it with that in mind
- [ ] Verify Milvus is reachable from local Python with a simple connection test

### Story 0.2: MVP feature extraction

- [ ] In `mvp/extract.py`, implement simple functions (no classes):
  - `load_model(model_name: str, device: str) -> torch.nn.Module` — load a pretrained model (start with ViT-B/16), strip classification head
  - `preprocess_image(image_path: str) -> torch.Tensor` — load, resize, normalize for ImageNet
  - `extract_embedding(model, image_tensor, device) -> np.ndarray` — forward pass, return flat vector
  - `extract_batch(model, image_paths: list[str], device, batch_size) -> list[dict]` — batch extraction with progress bar, returns list of `{"path": ..., "embedding": ...}`
- [ ] Test in the notebook: extract embedding for a single image, verify shape and dtype

### Story 0.3: MVP Milvus integration

- [ ] In `mvp/db.py`, implement simple functions:
  - `connect(host, port)` — connect to Milvus
  - `create_collection(name, embedding_dim)` — create collection with schema: id (auto), image_path (varchar), label (varchar), embedding (float_vector)
  - `insert_embeddings(collection_name, data: list[dict])` — insert batch of `{"path", "label", "embedding"}`
  - `search(collection_name, query_vector, top_k) -> list[dict]` — returns `{"path", "label", "score"}`
- [ ] Test in the notebook: create collection, insert a few vectors, search, verify results

### Story 0.4: MVP end-to-end flow in notebook

- [ ] Prepare a small test dataset: ~100–200 labeled images, 3–5 classes, organized as `data/sample/{class_name}/image.jpg`
  - Document where the sample data comes from (subset of the 200K labeled set, or a public dataset for testing)
- [ ] In the notebook, run the full flow:
  1. Load model (ViT-B/16)
  2. Extract embeddings for all sample images
  3. Create Milvus collection and insert all embeddings
  4. Pick a few query images from each class
  5. Search top-10 for each query
  6. Display results: query image, top-k results with scores and labels
  7. Compute basic metrics: precision@5, precision@10 per class
- [ ] Visualize the embedding space (2D projection with UMAP or t-SNE, colored by class)
- [ ] Document observations and learnings as markdown cells in the notebook
- [ ] Verify the entire flow works end-to-end with `docker compose up` + `uv run jupyter notebook`

---

## Epic 1: Project Foundation

**Dates:** Apr 5 – Apr 11 (Week 3)

> From this point forward, all code lives in `cbir/` and `tests/`. The `mvp/` folder is frozen.

### Story 1.1: Project structure and tooling setup

- [ ] Define the project package structure:
  - `cbir/` — main package
  - `cbir/extractors/` — feature extraction models
  - `cbir/db/` — Milvus integration
  - `cbir/api/` — FastAPI app
  - `cbir/cli/` — Typer CLI
  - `cbir/dashboard/` — Streamlit app
  - `cbir/evaluation/` — metrics and experiment tracking
  - `cbir/heuristics/` — auto-labeling strategies
  - `cbir/config/` — configuration management
  - `tests/` — test suite mirroring `cbir/` structure
- [ ] Set up `pyproject.toml` with dependency groups: core, dev, dashboard
- [ ] Configure `ruff` (linter + formatter) in `pyproject.toml`
- [ ] Configure `mypy` with strict mode in `pyproject.toml`
- [ ] Configure `pytest` in `pyproject.toml`
- [ ] Add a `Makefile` with targets: `lint`, `format`, `typecheck`, `test`, `run`
- [ ] Create a base configuration system (e.g., Pydantic Settings or similar) for managing DB hosts, model names, paths, etc.

### Story 1.2: Data model and core abstractions

- [ ] Define core data models (Pydantic):
  - `Image` — id, path, label (optional), metadata
  - `Embedding` — image_id, vector, model_name, extraction_params
  - `SearchResult` — image, score, rank
  - `Pipeline` — extractor config + preprocessing config
- [ ] Define abstract base class `BaseExtractor` with interface:
  - `extract(image: PIL.Image) -> np.ndarray`
  - `batch_extract(images: list[PIL.Image]) -> np.ndarray`
  - `embedding_dim -> int` (property)
  - `model_name -> str` (property)
- [ ] Define abstract base class `BaseHeuristic` with interface:
  - `label(query_image, search_results) -> LabelResult`
- [ ] Write unit tests for data models

---

## Epic 2: Feature Extraction Pipeline

**Dates:** Apr 12 – Apr 20 (Weeks 4)

### Story 2.1: Implement first extractor — pretrained ViT

- [ ] Implement `ViTExtractor(BaseExtractor)` using `torchvision.models.vit_b_16` (pretrained)
  - Handle image preprocessing (resize, normalize to ImageNet stats)
  - Extract embedding from the CLS token (before classification head)
  - Support both CPU and GPU execution
- [ ] Write tests: verify output shape matches `embedding_dim`, deterministic output for same input

### Story 2.2: Implement CNN extractors

- [ ] Implement `ResNetExtractor(BaseExtractor)` using `torchvision.models.resnet50`
  - Extract from avgpool layer (before FC)
- [ ] Write tests for each extractor

### Story 2.3: Extractor registry and factory

- [ ] Create an extractor registry: `get_extractor(name: str, **kwargs) -> BaseExtractor`
- [ ] Support listing available extractors
- [ ] Allow configuration via the config system (model name, device, batch size)
- [ ] Write tests for registry

### Story 2.4: Batch extraction service

- [ ] Implement a batch extraction function that:
  - Takes a directory/glob of images + extractor name
  - Extracts embeddings with progress bar (`tqdm`)
  - Returns structured results (list of `Embedding` objects)
  - Handles errors gracefully (corrupted images, OOM) — logs and skips
- [ ] Support resumable extraction (skip already-processed images)
- [ ] Write integration test with a small image set

---

## Epic 3: Vector Database (Milvus)

**Dates:** Apr 16 – Apr 27 (Weeks 4–5, overlaps with Epic 2)

### Story 3.1: Milvus connection and schema management

- [ ] Implement `MilvusClient` wrapper class (inspired by `mvp/db.py`, but proper OOP):
  - Connect/disconnect with configurable host/port
  - Create collection for a given extractor (collection name = `{dataset}_{model_name}`)
  - Schema: `id` (primary), `image_path` (varchar), `label` (varchar, optional), `embedding` (float_vector), `metadata` (json)
  - Create appropriate index (IVF_FLAT or HNSW — make configurable)
- [ ] Handle collection lifecycle: create, drop, list, describe
- [ ] Write tests using Milvus Lite (in-process, no server needed for tests)

### Story 3.2: Insert and query operations

- [ ] Implement `insert_embeddings(collection, embeddings: list[Embedding])` with batching
- [ ] Implement `search(collection, query_vector, top_k, metric_type) -> list[SearchResult]`
  - Support L2 and cosine similarity
- [ ] Implement `search_by_image(collection, image_path, extractor) -> list[SearchResult]`
  - Convenience method: extract + search in one call
- [ ] Implement `get_by_label(collection, label) -> list[Image]` for evaluation
- [ ] Write integration tests

### Story 3.3: End-to-end pipeline validation

- [ ] Script/test that: extracts embeddings with ViT → inserts into Milvus → queries and prints top-k results
- [ ] Validate that same-class images rank higher than cross-class (using the same sample dataset from MVP)
- [ ] This is the "smoke test" for the main `cbir/` pipeline

---

## Epic 4: CLI (Typer)

**Dates:** Apr 23 – Apr 30 (Week 5–6)

### Story 4.1: CLI skeleton and core commands

- [ ] Add `typer` dependency, create `cbir/cli/app.py` as main CLI entry point
- [ ] Register CLI as script entry point in `pyproject.toml` (`cbir = "cbir.cli.app:app"`)
- [ ] Implement commands:
  - `cbir extract --model <name> --input <path> --collection <name>` — batch extract and store
  - `cbir search --image <path> --collection <name> --top-k <n>` — search similar images
  - `cbir collections list` — list Milvus collections
  - `cbir collections create --name <name> --model <name>` — create collection
  - `cbir collections delete --name <name>` — drop collection
- [ ] Each command should have `--help` with clear descriptions
- [ ] Write tests using Typer's `CliRunner`

### Story 4.2: CLI output formatting

- [ ] Support `--output-format` flag: `table` (default, using `rich`), `json`
- [ ] For search results: show image path, score, label (if available), rank
- [ ] Add `--verbose` flag for debug logging
- [ ] Add progress bars for long-running operations (extract)

---

## Epic 5: API (FastAPI)

**Dates:** Apr 28 – May 8 (Weeks 6–7)

### Story 5.1: API skeleton and health endpoints

- [ ] Add `fastapi` and `uvicorn` dependencies
- [ ] Create `cbir/api/app.py` with FastAPI app
- [ ] Implement `GET /health` — returns status + Milvus connection state
- [ ] Implement `GET /models` — returns available extractors
- [ ] Implement `GET /collections` — returns Milvus collections
- [ ] Add CORS middleware
- [ ] Write tests using `httpx` + FastAPI `TestClient`

### Story 5.2: Search and extraction endpoints

- [ ] `POST /extract` — accepts image file upload, model name; returns embedding vector
- [ ] `POST /search` — accepts image file upload, collection name, top_k; returns ranked results
- [ ] `POST /batch-extract` — accepts list of image paths (server-side), model name, collection name; triggers extraction job
- [ ] Add request validation and meaningful error responses
- [ ] Write tests for each endpoint

### Story 5.3: Collection management endpoints

- [ ] `POST /collections` — create collection
- [ ] `DELETE /collections/{name}` — drop collection
- [ ] `GET /collections/{name}/stats` — count, index info, model used
- [ ] Write tests

---

## Epic 6: Evaluation and Experiment Tracking

**Dates:** May 5 – May 15 (Weeks 7–8)

### Story 6.1: Evaluation metrics

- [ ] Implement metrics in `cbir/evaluation/metrics.py`:
  - `precision_at_k(results, query_label, k)` — fraction of top-k with same label
  - `recall_at_k(results, query_label, k, total_relevant)` — fraction of relevant items found in top-k
  - `mean_average_precision(results, query_label)` — mAP
  - `reciprocal_rank(results, query_label)` — rank of first relevant result
- [ ] Implement `evaluate_extractor(extractor, collection, test_images) -> dict` that runs all metrics across a test set and aggregates
- [ ] Write thorough unit tests with known inputs/outputs

### Story 6.2: MLflow integration

- [ ] Add `mlflow` dependency
- [ ] Add MLflow tracking server to `docker-compose.yml`
- [ ] Implement experiment tracking wrapper:
  - Log extractor name, parameters, dataset info as MLflow params
  - Log all metrics from Story 6.1 as MLflow metrics
  - Log sample search results as artifacts (image grids or JSON)
- [ ] CLI command: `cbir evaluate --collection <name> --test-set <path> --model <name>`
  - Runs evaluation and logs to MLflow
- [ ] API endpoint: `POST /evaluate` — same as CLI but via API
- [ ] Write tests (mock MLflow tracking)

### Story 6.3: Comparative evaluation

- [ ] Implement `compare_extractors(extractors: list[str], collection_prefix, test_set) -> DataFrame`
  - Runs evaluation for each extractor, returns comparison table
- [ ] CLI command: `cbir compare --models resnet50,vit_b_16 --test-set <path>`
- [ ] Log comparison as MLflow parent run with child runs per model

---

## Epic 7: Auto-Labeling Heuristics

**Dates:** May 11 – May 20 (Weeks 8–9)

### Story 7.1: Threshold-based heuristic

- [ ] Implement `ThresholdHeuristic(BaseHeuristic)`:
  - Given query image and search results, label as class A if top result similarity > threshold
  - Configurable threshold, configurable metric (L2 / cosine)
- [ ] Write tests with mock search results

### Story 7.2: KNN majority vote heuristic

- [ ] Implement `KNNHeuristic(BaseHeuristic)`:
  - Given query image and top-k results, label based on majority class among k neighbors
  - Configurable k, optional distance weighting
- [ ] Write tests

### Story 7.3: Cluster-based heuristic (nice-to-have)

- [ ] Implement `ClusterHeuristic(BaseHeuristic)`:
  - Cluster embeddings (e.g., KMeans or DBSCAN)
  - Label query image based on majority class in its cluster
  - Configurable clustering algorithm and params
- [ ] Write tests

### Story 7.4: Heuristic evaluation and CLI/API integration

- [ ] Implement `evaluate_heuristic(heuristic, test_set) -> dict` — accuracy, precision, recall, F1 of the auto-labels vs ground truth
- [ ] CLI command: `cbir label --heuristic <name> --collection <name> --input <path> --output <path>`
  - Applies heuristic to unlabeled images and outputs proposed labels
- [ ] API endpoint: `POST /label` — same as CLI
- [ ] Log heuristic evaluation to MLflow
- [ ] Write tests

---

## Epic 8: Dashboard (Streamlit)

**Dates:** May 18 – May 25 (Week 9)

### Story 8.1: Basic dashboard with search

- [ ] Add `streamlit` to dashboard dependency group
- [ ] Create `cbir/dashboard/app.py`
- [ ] Pages:
  - **Search**: upload image, select collection/model, show top-k results as image grid with scores
  - **Collections**: list collections, show stats

### Story 8.2: Vector space visualization

- [ ] Implement dimensionality reduction (UMAP or t-SNE) for embedding visualization
- [ ] Page: **Vector Space** — 2D scatter plot of embeddings, colored by label
  - Highlight query image position when searching
  - Interactive: hover to see image thumbnail
- [ ] Allow selecting subset of data to visualize (by label, by count)

### Story 8.3: Evaluation dashboard (nice-to-have)

- [ ] Page: **Experiments** — show MLflow experiment results
  - Compare models side by side
  - Show per-class performance
- [ ] Page: **Auto-Labeling** — apply heuristic interactively, review proposed labels

---

## Epic 9: Containerization and Deployment

**Dates:** May 22 – May 31 (Weeks 9–10)

### Story 9.1: Dockerfile

- [ ] Create multi-stage `Dockerfile`:
  - Base stage: Python 3.13, `uv`, core dependencies
  - Dev stage: adds dev dependencies (ruff, mypy, pytest)
  - API stage: runs FastAPI with uvicorn
  - Dashboard stage: runs Streamlit
- [ ] Optimize image size (no cache, minimal layers)
- [ ] Test building and running each stage

### Story 9.2: Docker Compose — experimentation profile

- [ ] Extend `docker-compose.yml` (started in Epic 0) with:
  - `mlflow` (tracking server with local storage)
  - `api` (FastAPI service)
  - `dashboard` (Streamlit service)
- [ ] Shared volumes for image data and MLflow artifacts
- [ ] `docker compose --profile dev up` starts everything
- [ ] Test full flow: extract → store → search → evaluate

### Story 9.3: Docker Compose — execution profile (nice-to-have)

- [ ] `docker compose --profile prod up` starts only:
  - `milvus`
  - `api`
- [ ] Minimal resource footprint, no dev tools
- [ ] Health checks on all services
- [ ] Document the two profiles in README

### Story 9.4: Final polish and documentation

- [ ] Update README with installation, usage, and architecture diagrams
- [ ] Add example usage for CLI, API, and dashboard
- [ ] Ensure all tests pass in containerized environment
- [ ] Review and clean up all TODOs in code

---

## Timeline Summary

| Week | Dates | Focus |
|------|-------|-------|
| 1–2 | Mar 25 – Apr 4 | **MVP**: end-to-end in `mvp/` (extract → Milvus → search → visualize) |
| 3 | Apr 5–11 | Project structure, tooling, core abstractions |
| 4 | Apr 12–20 | Extractors (ViT, ResNet), registry, start Milvus wrapper |
| 5 | Apr 21–27 | Finish Milvus, CLI |
| 6 | Apr 28 – May 4 | Finish CLI, start API |
| 7 | May 5–11 | Finish API, evaluation metrics, MLflow |
| 8 | May 12–18 | Comparative evaluation, heuristics |
| 9 | May 19–25 | Finish heuristics, dashboard, start Docker |
| 10 | May 26–31 | Finish Docker, final polish, documentation |

---

## Nice-to-Have (trim if behind schedule)

- Story 7.3: Cluster-based heuristic — threshold + KNN cover the core use cases
- Story 8.3: Evaluation dashboard — CLI/API evaluation is sufficient
- Story 9.3: Prod Docker profile — one dev profile is enough for deliverable 1

---

## Notes

- **`mvp/` is read-only after Epic 0.** It's a reference, not a dependency.
- The `docker-compose.yml` is shared between MVP and the full system. Epic 0 creates it with Milvus; later epics add services.
- Each story is designed to be self-contained and testable. Feed the story description (with its tasks) to Claude Code and it should be able to implement it.
- Prioritize the core pipeline (extract → store → search) over UI/CLI polish early on.
- SAHI integration for image preprocessing can be added as an optional preprocessing step in the extractor pipeline once the base system works.

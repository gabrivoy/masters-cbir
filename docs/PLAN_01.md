# Plan 01

## Objective

Deliver a reproducible bbox-level CBIR baseline for `Traineira` using:

- the final annotated bboxes as the canonical data unit;
- `padding_ratio = 0.0` by default;
- an external image embedding model;
- Milvus standalone through Docker Compose;
- notebook-first inspection plus reproducible CLI scripts.

## Execution sequence

| Stage | Goal | Main outputs | Acceptance focus |
| --- | --- | --- | --- |
| 1. Data contract reset | Persist the raw bbox as the stable unit | `manifest.jsonl`, `summary.json`, `rejected.jsonl` under `data/cbir/<class>/v1/` | Stable `item_id`, correct split counts, no persisted padded-crop coordinates |
| 2. BBox audit tooling | Inspect class examples before retrieval | notebook audit cells and `mvp/visualize.py inspect` | One command or cell renders frame + bbox + runtime crop + metadata |
| 3. Milvus local infra | Stand up the vector DB in a reproducible way | `docker-compose.yml` | `docker compose up -d` brings up healthy `etcd`, `minio`, and `milvus` |
| 4. External embedding baseline | Extract normalized vectors from tight bbox crops | `mvp/extract.py` | `OpenCLIP ViT-B/32` produces normalized vectors with the expected dimension |
| 5. Ingestion pipeline | Push embeddings and metadata into Milvus | `mvp/ingest.py` and ingestion summary artifacts | Collection count matches inserted rows for a smoke-test subset |
| 6. Search visualization | Query Milvus and inspect returned neighbors | notebook search cells and `mvp/visualize.py search` | Query crop plus top-k hits render with score and metadata |
| 7. Early metrics | Measure the first retrieval behavior | notebook metric cells | `precision@5` and `precision@10` computed on a controlled subset |
| 8. Post-baseline exploration | Move into structure discovery | notebook cells or follow-up scripts using `UMAP + HDBSCAN` | Class-wise 2D embedding plots and cluster assignments |

## Default operating assumptions

| Topic | Default |
| --- | --- |
| First class | `Traineira` |
| Benchmark filter | `medium+` (`bbox_area >= 1024`) |
| Crop policy | Tight bbox, `padding_ratio = 0.0` |
| First embedder | `OpenCLIP ViT-B/32` with `openai` weights |
| Search metric | Cosine similarity |
| Milvus index | `FLAT` |
| First exploration surface | Notebook |
| First search surface outside notebook | `mvp/visualize.py search` |

## Immediate follow-up after this slice

Once the baseline is stable, the next sequence is:

1. run `UMAP + HDBSCAN` by class;
2. compare `Traineira` against other strong classes;
3. test whether camera-specific clustering behaves differently from global clustering;
4. only then introduce embeddings derived from an in-house detector backbone.

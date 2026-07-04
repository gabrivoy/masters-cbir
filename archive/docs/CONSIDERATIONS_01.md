# Considerations 01

## Context

This note consolidates the feedback received after the first bbox-level CBIR prototype review. The objective is to turn the feedback into implementation decisions, not into a generic wishlist.

The central concern is crop quality. The current dataset already provides object annotations, but the first prototype treated padded crops as if they were the canonical data unit. That is too aggressive for a baseline: if the annotation is already loose, additional padding amplifies background, camera, and scene bias.

## Immediate pipeline corrections

| Topic | Decision |
| --- | --- |
| Canonical object unit | The annotated bbox is the current ground-truth unit. |
| Default crop policy | `padding_ratio = 0.0` for the baseline. |
| First QA objective | Audit bbox quality before evaluating retrieval quality. |
| First analysis granularity | One strong class at a time, starting with `Traineira`. |
| First benchmark slice | `medium+`, defined as `bbox_area >= 1024`. |
| Retrieval baseline | Use external image embeddings before any detector-trained embedding variant. |

### Why this correction is necessary

- A loose bbox plus positive padding can cause the embedding to overfit water, horizon, shoreline, and camera-specific context.
- That bias would make the first retrieval results hard to interpret: same-class proximity could come from the scene instead of the vessel.
- The first benchmark must isolate the representation quality of the object instance as cleanly as possible.

## Scientific baseline for the first CBIR slice

| Item | Decision |
| --- | --- |
| Baseline embedding family | External, off-the-shelf visual embeddings |
| First model | `OpenCLIP ViT-B/32` with `openai` weights |
| Storage/search backend | Milvus standalone |
| Search metric | Cosine similarity over normalized vectors |
| First retrieval unit | One bbox-derived crop per object instance |
| First inspection surface | Notebook-first, with a reproducible CLI rendering script |

### Why the baseline starts here

- The first slice must answer a narrow question: do tight bbox crops from a strong class cluster and retrieve coherently with a generic image embedding model?
- Using external embeddings first keeps the baseline defensible and fast to iterate.
- Loading a detector trained in-house and extracting intermediate features is a strong next experiment, but it is not the correct first baseline.

## Research directions that remain valid after the baseline

The feedback also identified several valuable research tracks. They remain important, but they should start only after the bbox-level ingestion and search baseline is stable.

| Track | Status for slice 1 | Reason |
| --- | --- | --- |
| Per-class visualization | In scope | Needed immediately for bbox QA and data understanding |
| Per-class embedding similarity and grouping | Next step after baseline | Natural continuation once embeddings and Milvus search are stable |
| `UMAP + HDBSCAN` by class | Next step after baseline | Good exploratory clustering setup without forcing a fixed number of groups |
| Compare external embeddings vs. YOLO-derived embeddings | Deferred to slice 2 | Strong experiment, but not part of the first reproducible baseline |
| Group by vessel identity / instance | Deferred exploratory research | No explicit identity ground truth is available |
| Annotation error detection through outliers | Deferred exploratory research | Outliers may reflect noise, rarity, occlusion, or viewpoint, not only labeling mistakes |
| Camera-specific vs. global clustering | Deferred analytical question | Depends on a stable embedding/clustering baseline |
| Redundancy detection in the dataset | Deferred analytical question | Depends on stable retrieval and cluster inspection tooling |
| Using the embedding bank as a detector/template matcher | Out of scope for slice 1 | Interesting but high-risk relative to the primary thesis question |
| Correlating groups with detector IoU/score | Deferred analytical question | Depends on both the embedding baseline and detector evaluation artifacts |
| Cluster-aware data augmentation / fine-tuning | Deferred analytical question | Requires credible evidence from the previous steps |

## Sequence of work implied by the feedback

1. Rebuild the class manifest so the persisted unit is the raw bbox, not a padded crop.
2. Add bbox audit tooling for class, split, camera, and size bucket inspection.
3. Stand up Milvus with a minimal local Docker Compose stack.
4. Ingest bbox-derived crops with `padding_ratio = 0.0` using a generic external embedder.
5. Validate search visually and with simple retrieval metrics.
6. Only then move into class-wise `UMAP + HDBSCAN`.
7. Keep detector-derived embeddings and instance-level research for the next phase.

## Consequence for the repository

The repository should treat the first Epic 0 deliverable as a bbox-level ingestion and retrieval baseline, not as a detector-aware system. That keeps the first slice narrow, reproducible, and interpretable.

# Models

## Purpose

This note documents the external embedding models considered for the first CBIR baseline. The goal is not to select the globally best model in advance, but to choose a first baseline that is easy to study, easy to run, and technically defensible.

## Shortlist

| Model | Status in this repo | Embedding dimension | Patch size | Typical input size | Cost profile | Why it is relevant |
| --- | --- | --- | --- | --- | --- | --- |
| `OpenCLIP ViT-B/32` | Baseline choice | `512` | `32` | `224` | Lowest among the shortlisted ViT baselines | Fastest iteration loop and closest to the classic CLIP baseline story |
| `OpenCLIP ViT-B/16` | Next experiment | `512` | `16` | `224` | Higher than `B/32` | Finer spatial partitioning and a natural follow-up once the ingestion baseline is stable |
| `DINOv2 ViT-S/14` | Strong later baseline | `384` | `14` | `518` in the published Hugging Face config | Moderate | Strong generic visual features and a good non-CLIP comparison point |

## Selection rationale

| Criterion | `OpenCLIP ViT-B/32` | `OpenCLIP ViT-B/16` | `DINOv2 ViT-S/14` |
| --- | --- | --- | --- |
| Easiest first baseline | Best | Good | Good, but less aligned with the current CLIP-like notebook flow |
| Runtime cost | Best | Worse than `B/32` | Moderate |
| Retrieval-friendly off-the-shelf image embedding | Good | Good | Good |
| Ease of explanation in the thesis | Best | Good | Good |
| Best role in the sequence | First | Second | Third |

## Repository decision

The first implemented baseline is `OpenCLIP ViT-B/32` loaded through `open_clip_torch` with `pretrained="openai"`.

This does **not** make it the permanent preferred model. It only makes it the correct first baseline because:

- it is simple to explain;
- it is simple to run;
- it has a compact vector dimension (`512`);
- it keeps the first loop focused on bbox quality and retrieval behavior instead of model complexity.

## Recommended study order

| Order | Model | What to learn from it |
| --- | --- | --- |
| 1 | `OpenCLIP ViT-B/32` | Establish the first bbox-level retrieval baseline quickly |
| 2 | `OpenCLIP ViT-B/16` | Test whether smaller patches help vessel-crop retrieval |
| 3 | `DINOv2 ViT-S/14` | Compare a strong self-supervised visual feature baseline against the CLIP family |
| 4 | Detector-derived embedding from an in-house YOLO model | Evaluate whether a task-specific representation improves grouping and retrieval |

## Primary references

### OpenCLIP ViT-B/32

- OpenCLIP repository: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- OpenCLIP package documentation: [open-clip-torch on PyPI](https://pypi.org/project/open-clip-torch/)
- Original CLIP paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Official CLIP repository: [openai/CLIP](https://github.com/openai/CLIP)
- Official OpenAI config showing `projection_dim = 512`: [openai/clip-vit-base-patch32 config.json](https://huggingface.co/openai/clip-vit-base-patch32/blob/main/config.json)

### OpenCLIP ViT-B/16

- OpenCLIP repository: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- OpenCLIP package documentation: [open-clip-torch on PyPI](https://pypi.org/project/open-clip-torch/)
- Original CLIP paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Official CLIP repository: [openai/CLIP](https://github.com/openai/CLIP)
- Official OpenAI config showing `projection_dim = 512`: [openai/clip-vit-base-patch16 config.json](https://huggingface.co/openai/clip-vit-base-patch16/blob/main/config.json)

### DINOv2 ViT-S/14

- Official repository: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- Original paper: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- Transformers documentation: [DINOv2 model documentation](https://huggingface.co/docs/transformers/model_doc/dinov2)
- Official model config showing `hidden_size = 384` and `patch_size = 14`: [facebook/dinov2-small config.json](https://huggingface.co/facebook/dinov2-small/blob/main/config.json)

## Practical notes for this project

| Topic | Current choice |
| --- | --- |
| Retrieval vector normalization | L2 normalization before cosine search |
| First crop policy | Tight bbox crop with `padding_ratio = 0.0` |
| First dataset slice | `Traineira`, `medium+` |
| First vector DB index | Milvus `FLAT` with cosine similarity |
| First clustering follow-up | `UMAP + HDBSCAN` by class |

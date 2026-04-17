# Dataset Report

## 1. Dataset Assets and Provenance

| Asset | Path | Role | Notes |
| --- | --- | --- | --- |
| Raw image pool | `data/images_from_dataset/` | Source frames used by the current local dataset | Contains the actual JPG files referenced by the final annotations |
| Dataset-generation run | `data/Etapa6_Dataset_PTZ/` | Provenance, filtering, remapping, and statistics | Stores the run outputs, not a second copy of the raw images |
| Final COCO export | `data/Etapa6_Dataset_PTZ/runs_2025_05_22_10-46-01/run_2025_05-22_10-46-14_347849_PTZ/result/Dataset_coco_instances/annotations/` | Ground truth for bbox-level object annotations | Split into `instances_train.json`, `instances_val.json`, and `instances_test.json` |
| Final Datumaro export | `data/Etapa6_Dataset_PTZ/runs_2025_05_22_10-46-01/run_2025_05-22_10-46-14_347849_PTZ/result/Dataset_datumaro/annotations/` | Image-level metadata and richer annotation structure | Useful for `camera_id`, `timestamp`, and tag-derived metadata |
| Base fixed dataset snapshot | `data/Etapa6_Dataset_PTZ/runs_2025_05_22_10-46-01/run_2025_05-22_10-46-14_347849_PTZ/zip_files/fixed_dataset/annotations/` | Previous dataset used as input to the final filtering pass | Larger than the final dataset and includes labels later remapped away |
| Dataset-generation config | `data/Etapa6_Dataset_PTZ/runs_2025_05_22_10-46-01/run_2025_05-22_10-46-14_347849_PTZ/args.csv` | Source of filtering, remapping, and split parameters | Best source of truth for the final dataset recipe |
| Detector config | `data/test_cfg.py` | Confirms how the final COCO dataset is consumed by MMYOLO/YOLOv8 | Useful as provenance, not as a CBIR artifact |

## 2. Dataset Snapshot

| Item | Value |
| --- | ---: |
| Raw images available locally | 20,338 |
| Unique filenames referenced by the final split | 20,338 |
| Raw images not referenced by the final split | 0 |
| Final COCO categories | 31 |
| Final object classes | 25 |
| `[TAG]` categories still present in the COCO taxonomy | 6 |
| Total final bbox annotations | 185,268 |

### Raw image pool

| Property | Value |
| --- | --- |
| Storage path | `data/images_from_dataset/` |
| Approximate size on disk | `33 GB` |
| File format | JPG |
| Filename pattern | `CAMERA_TIMESTAMP.jpg` |
| Observed cameras | `CDG_PTZ1`, `HON_PTZ1`, `SB8_PTZ1` |

## 3. Dataset Construction and Filtering

### Construction summary

| Item | Value |
| --- | --- |
| Base input dataset | `/project/data/data_dataset_generation/runs_2025_02_22_00-34-33/reduced_21k_datumaro/` |
| Export format | `coco_instances` |
| Camera filter | `CDG_PTZ1`, `SB8_PTZ1`, `HON_PTZ1` |
| Daylight rule | `(6, 17)` |
| Difficult filtering | Enabled |
| Image-quality filtering | Enabled |
| Tags removed from final COCO objects | No; `[TAG]` categories remain in the taxonomy with zero annotations |
| Remove-images rule | Images with labels `Embarcação` and `Jet Ski` were removed |
| Label remapping | Enabled before filtering |
| Optimization strategy | `multidimensional_knapsack` |
| Grouping rule | `k_means`, `k=1000` |
| Configured split target | `train=0.7`, `val=0.2`, `test=0.1` |
| Effective final split | `train=76.08%`, `val=15.05%`, `test=8.87%` |

### Dataset size before and after the final filtering pass

| Split | Base fixed dataset | Final dataset | Removed in final pass |
| --- | ---: | ---: | ---: |
| Train | 16,388 | 15,473 | 915 |
| Val | 3,216 | 3,060 | 156 |
| Test | 1,894 | 1,805 | 89 |
| Total | 21,498 | 20,338 | 1,160 |

### Label remapping applied by the generation run

| Original label | Final label |
| --- | --- |
| `Navio Container (Porta-contetores)` | `Navio de Carga Geral` |
| `Petroleiro` | `Navio de Carga Geral` |
| `Farol` | `Boia` |
| `Navio-aeródromo (Nae / Navio Porta-aviões)` | `Embarcação Militar` |
| `Patrulha Naval` | `Embarcação Militar` |
| `Fragata` | `Embarcação Militar` |
| `Embarcação` | `Traineira` |

### Labels present in the base fixed dataset but not in the final one

| Removed or absorbed label |
| --- |
| `Embarcação` |
| `Farol` |
| `Fragata` |
| `Navio Container (Porta-contetores)` |
| `Navio-aeródromo (Nae / Navio Porta-aviões)` |
| `Patrulha Naval` |
| `Petroleiro` |

## 4. Class Taxonomy

### Taxonomy summary

| Category group | Count |
| --- | ---: |
| Final object classes | 25 |
| Final `[TAG]` categories | 6 |
| Total COCO categories | 31 |

### Final object classes

| # | Class |
| ---: | --- |
| 1 | `Avião` |
| 2 | `Barca` |
| 3 | `Boia` |
| 4 | `Bote de Pesca` |
| 5 | `Caiaque` |
| 6 | `Calhau / Forte / Ilha` |
| 7 | `Chata` |
| 8 | `Draga` |
| 9 | `Traineira` |
| 10 | `Embarcação Militar` |
| 11 | `Escuna` |
| 12 | `Guindaste Flutuante` |
| 13 | `Jet Ski` |
| 14 | `Lancha / Iate` |
| 15 | `Navio de Carga Geral` |
| 16 | `Navio Plataforma (FPSO)` |
| 17 | `Navio de Cruzeiro Marítimo` |
| 18 | `Navio de Pesquisa (Oceanográfico)` |
| 19 | `Pilar` |
| 20 | `Plataforma Petrolífera (semi-sub)` |
| 21 | `Porta Veículos` |
| 22 | `Pássaro` |
| 23 | `Rebocador` |
| 24 | `Submarino` |
| 25 | `Veleiro` |

### `[TAG]` categories in the final COCO taxonomy

| `[TAG]` category | COCO annotations in final export |
| --- | ---: |
| `[TAG] Camera Configuration` | 0 |
| `[TAG] Datetime` | 0 |
| `[TAG] METAR` | 0 |
| `[TAG] Metadatum` | 0 |
| `[TAG] Observações` | 0 |
| `[TAG] Qualidade da Imagem` | 0 |

The useful metadata behind these tags remains available in the Datumaro export, where image-level label annotations still carry camera configuration and timestamp values.

## 5. Global Split Summary

### Split totals

| Split | Images | Share of final dataset | BBox annotations |
| --- | ---: | ---: | ---: |
| Train | 15,473 | 76.08% | 138,903 |
| Val | 3,060 | 15.05% | 30,395 |
| Test | 1,805 | 8.87% | 15,970 |
| Total | 20,338 | 100.00% | 185,268 |

### Split by camera

| Camera | Train | Val | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| `CDG_PTZ1` | 906 | 191 | 114 | 1,211 |
| `HON_PTZ1` | 5,966 | 1,179 | 689 | 7,834 |
| `SB8_PTZ1` | 8,601 | 1,690 | 1,002 | 11,293 |

The camera distribution is stable across the three splits, which is consistent with the dataset-optimization step described in `args.csv`.

## 6. Train / Val / Test Support by Category

### Support by image count

The table below reports how many images contain at least one instance of each class. Because the dataset is multi-label at image level, these counts do not sum to the split size.

| Class | Train images | Val images | Test images | Total images |
| --- | ---: | ---: | ---: | ---: |
| `Avião` | 709 | 156 | 78 | 943 |
| `Barca` | 2,457 | 565 | 269 | 3,291 |
| `Boia` | 6,808 | 1,343 | 833 | 8,984 |
| `Bote de Pesca` | 1,733 | 394 | 158 | 2,285 |
| `Caiaque` | 251 | 50 | 24 | 325 |
| `Calhau / Forte / Ilha` | 0 | 0 | 0 | 0 |
| `Chata` | 1,069 | 233 | 132 | 1,434 |
| `Draga` | 1,660 | 295 | 181 | 2,136 |
| `Traineira` | 10,250 | 2,118 | 1,253 | 13,621 |
| `Embarcação Militar` | 3,814 | 873 | 429 | 5,116 |
| `Escuna` | 504 | 105 | 60 | 669 |
| `Guindaste Flutuante` | 1,736 | 371 | 225 | 2,332 |
| `Jet Ski` | 0 | 0 | 0 | 0 |
| `Lancha / Iate` | 5,018 | 1,082 | 493 | 6,593 |
| `Navio de Carga Geral` | 8,478 | 1,777 | 983 | 11,238 |
| `Navio Plataforma (FPSO)` | 255 | 39 | 31 | 325 |
| `Navio de Cruzeiro Marítimo` | 8 | 1 | 0 | 9 |
| `Navio de Pesquisa (Oceanográfico)` | 905 | 197 | 97 | 1,199 |
| `Pilar` | 0 | 0 | 0 | 0 |
| `Plataforma Petrolífera (semi-sub)` | 159 | 22 | 19 | 200 |
| `Porta Veículos` | 133 | 33 | 14 | 180 |
| `Pássaro` | 1,772 | 329 | 184 | 2,285 |
| `Rebocador` | 4,962 | 1,082 | 626 | 6,670 |
| `Submarino` | 93 | 20 | 9 | 122 |
| `Veleiro` | 2,615 | 446 | 308 | 3,369 |

### Support by bbox instance count

| Class | Train instances | Val instances | Test instances | Total instances |
| --- | ---: | ---: | ---: | ---: |
| `Avião` | 3,044 | 488 | 215 | 3,747 |
| `Barca` | 4,500 | 1,041 | 444 | 5,985 |
| `Boia` | 11,411 | 2,164 | 1,352 | 14,927 |
| `Bote de Pesca` | 2,105 | 486 | 171 | 2,762 |
| `Caiaque` | 385 | 79 | 39 | 503 |
| `Calhau / Forte / Ilha` | 0 | 0 | 0 | 0 |
| `Chata` | 1,287 | 291 | 159 | 1,737 |
| `Draga` | 1,794 | 315 | 214 | 2,323 |
| `Traineira` | 28,664 | 6,464 | 3,476 | 38,604 |
| `Embarcação Militar` | 10,033 | 2,487 | 1,202 | 13,722 |
| `Escuna` | 512 | 107 | 60 | 679 |
| `Guindaste Flutuante` | 1,797 | 382 | 231 | 2,410 |
| `Jet Ski` | 0 | 0 | 0 | 0 |
| `Lancha / Iate` | 10,054 | 2,079 | 964 | 13,097 |
| `Navio de Carga Geral` | 42,240 | 9,578 | 5,125 | 56,943 |
| `Navio Plataforma (FPSO)` | 255 | 39 | 31 | 325 |
| `Navio de Cruzeiro Marítimo` | 8 | 1 | 0 | 9 |
| `Navio de Pesquisa (Oceanográfico)` | 1,181 | 245 | 124 | 1,550 |
| `Pilar` | 0 | 0 | 0 | 0 |
| `Plataforma Petrolífera (semi-sub)` | 174 | 26 | 20 | 220 |
| `Porta Veículos` | 133 | 33 | 14 | 180 |
| `Pássaro` | 3,631 | 609 | 351 | 4,591 |
| `Rebocador` | 8,152 | 1,944 | 1,003 | 11,099 |
| `Submarino` | 93 | 20 | 9 | 122 |
| `Veleiro` | 7,450 | 1,517 | 766 | 9,733 |

### Strong candidate classes for the first CBIR iteration

| Class | Total images with class | Total bbox instances | Initial suitability |
| --- | ---: | ---: | --- |
| `Traineira` | 13,621 | 38,604 | Best default candidate for the first pass |
| `Navio de Carga Geral` | 11,238 | 56,943 | Strong support and broad coverage |
| `Boia` | 8,984 | 14,927 | Strong support but visually distinct from vessel-only classes |
| `Rebocador` | 6,670 | 11,099 | Good support and useful as a hard negative against other vessels |
| `Lancha / Iate` | 6,593 | 13,097 | Good support and useful as a hard negative against `Traineira` |
| `Embarcação Militar` | 5,116 | 13,722 | Good support for later expansion |

### Classes that should not be used in the first benchmark

| Class | Total images with class | Reason to postpone |
| --- | ---: | --- |
| `Calhau / Forte / Ilha` | 0 | No support in the final dataset |
| `Jet Ski` | 0 | Removed from the final dataset |
| `Pilar` | 0 | No support in the final dataset |
| `Navio de Cruzeiro Marítimo` | 9 | No test support and extremely low overall support |
| `Submarino` | 122 | Very small support |
| `Porta Veículos` | 180 | Very small support |

## 7. Multi-object Treatment and Crop Status

### Image-level object multiplicity

| Split | Single-label images | Multi-label images | Zero-object images |
| --- | ---: | ---: | ---: |
| Train | 1,538 | 13,935 | 0 |
| Val | 263 | 2,797 | 0 |
| Test | 196 | 1,609 | 0 |

### Multi-label share

| Split | Multi-label share |
| --- | ---: |
| Train | 90.06% |
| Val | 91.41% |
| Test | 89.14% |

### Current treatment of multi-object frames and crops

| Question | Current state |
| --- | --- |
| What is the storage unit today? | Full camera frame |
| What is the annotation unit today? | One bbox per object instance |
| Are there pre-generated crops in `data/`? | No |
| Is there a CBIR-ready crop manifest today? | No |
| Is the final dataset frame-level or crop-level? | Frame-level detection dataset |
| Can the dataset support crop-level CBIR? | Yes, because the final COCO export already provides bboxes |
| Are crops already persisted anywhere in the repository? | No |
| Is there metadata per frame beyond the COCO bbox export? | Yes, in the Datumaro export |
| Is full-frame retrieval a good starting point? | No, because most frames contain more than one object class |

### Existing preprocessing versus missing CBIR preprocessing

| Topic | Exists today | Status |
| --- | --- | --- |
| Camera filtering | Yes | Applied during dataset generation |
| Daylight filtering | Yes | Applied during dataset generation |
| Difficult filtering | Yes | Applied during dataset generation |
| Image-quality filtering | Yes | Applied during dataset generation |
| Label remapping | Yes | Applied during dataset generation |
| Detector-side augmentations | Yes | Defined in `data/test_cfg.py` |
| Persisted bbox crops | No | Still missing for CBIR |
| Bbox-level manifest for CBIR | No | Still missing for CBIR |
| Retrieval-ready benchmark policy | No | Still missing for CBIR |

The dataset therefore supports CBIR preparation, but the preparation layer still has to be derived from the final detection annotations.

## 8. BBox Size Distribution

### Size buckets used by the dataset-generation run

The final dataset already encodes object-size buckets through the `annotation_size_rule` stored in `args.csv`. These buckets are derived from bbox area in square pixels.

| Size bucket | Area range (px^2) | Approximate side length (px) | Included in the initial `medium+` benchmark |
| --- | ---: | --- | --- |
| `micro 0-5` | 0 to 25 | 0 to 5 | No |
| `tiniest 5-10` | 25 to 100 | 5 to 10 | No |
| `tiny 10-15` | 100 to 225 | 10 to 15 | No |
| `very small2 15-20` | 225 to 400 | 15 to 20 | No |
| `very small1 20-25` | 400 to 625 | 20 to 25 | No |
| `smaller 25-32` | 625 to 1,024 | 25 to 32 | No |
| `medium1 32-48` | 1,024 to 2,304 | 32 to 48 | Yes |
| `medium2 48-64` | 2,304 to 4,096 | 48 to 64 | Yes |
| `medium3 64-80` | 4,096 to 6,400 | 64 to 80 | Yes |
| `medium4 80-96` | 6,400 to 9,216 | 80 to 96 | Yes |
| `large 96-192` | 9,216 to 36,864 | 96 to 192 | Yes |
| `larger 192-384` | 36,864 to 147,456 | 192 to 384 | Yes |
| `largest 384-768` | 147,456 to 589,824 | 384 to 768 | Yes |
| `max 768-inf` | 589,824+ | 768+ | Yes |

### BBox size distribution by split

| Size bucket | Train | Val | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| `micro 0-5` | 202 | 39 | 24 | 265 |
| `tiniest 5-10` | 9,035 | 2,238 | 1,096 | 12,369 |
| `tiny 10-15` | 15,829 | 3,397 | 1,841 | 21,067 |
| `very small2 15-20` | 14,913 | 3,306 | 1,666 | 19,885 |
| `very small1 20-25` | 12,028 | 2,579 | 1,278 | 15,885 |
| `smaller 25-32` | 14,008 | 3,086 | 1,729 | 18,823 |
| `medium1 32-48` | 22,464 | 4,904 | 2,731 | 30,099 |
| `medium2 48-64` | 14,245 | 3,011 | 1,456 | 18,712 |
| `medium3 64-80` | 8,989 | 2,036 | 1,054 | 12,079 |
| `medium4 80-96` | 6,331 | 1,479 | 801 | 8,611 |
| `large 96-192` | 14,268 | 3,057 | 1,550 | 18,875 |
| `larger 192-384` | 4,631 | 893 | 516 | 6,040 |
| `largest 384-768` | 1,665 | 338 | 210 | 2,213 |
| `max 768-inf` | 295 | 32 | 18 | 345 |

The size distribution confirms that many objects are very small. A bbox-level CBIR benchmark can therefore be created immediately, but a first stable benchmark should start from `medium1 32-48` and larger.

## 9. Implications for CBIR

### Current readiness

| CBIR question | Current answer |
| --- | --- |
| Are there enough annotations to bootstrap a class-specific benchmark? | Yes |
| Is the current dataset already organized as one vessel per image? | No |
| Is full-frame retrieval a clean first benchmark? | No |
| Can bbox-level retrieval be derived from the current annotations? | Yes |
| Is there enough support for a strong first class? | Yes, especially for `Traineira` |
| Is a minimum-size policy needed? | Yes |

### Recommended initial retrieval unit

| Candidate retrieval unit | Initial recommendation | Rationale |
| --- | --- | --- |
| Full frame | Do not use as the first benchmark | Most frames contain several object classes |
| One bbox crop per object | Use for the first benchmark | Matches the annotation granularity and isolates embedding quality |
| Full frame + target class filter | Keep as a later comparison | Useful after the bbox-level baseline exists |

### Existing artifacts versus missing artifacts for Epic 0

| Artifact | Exists today | Comment |
| --- | --- | --- |
| Raw images | Yes | Stored in `data/images_from_dataset/` |
| Final object bboxes | Yes | Stored in the COCO export |
| Image-level metadata | Yes | Stored in the Datumaro export |
| Crop-level manifest | No | To be derived from the final COCO and Datumaro exports |
| Persisted crop directory | No | Not required for the first iteration |
| Class-inspection notebook | No | To be created for Epic 0 |
| Embedding extraction MVP | Not yet | Next stage after the crop-level inspection baseline |

## 10. Class Evaluation Plan

### Fixed decisions for the first iteration

| Decision | Selected value |
| --- | --- |
| First class | `Traineira` |
| Retrieval unit | One bbox crop per object instance |
| Split policy | Preserve the current `train/val/test` split |
| Manifest scope | Keep all instances of the selected class |
| Initial benchmark filter | `medium+`, defined as `bbox_area >= 1024` |
| Crop persistence | Crops generated on demand only |
| First interactive interface | Notebook |
| Example selector | `class=<label>`, `split=<split>`, `idx=<1-based index>` |

### First deliverables

| Deliverable | Description |
| --- | --- |
| BBox-level class manifest | One record per bbox for a selected class, enriched with crop coordinates and metadata |
| Inspection notebook | Notebook capable of rendering the frame, the selected bbox, the padded crop, and a compact per-item report |
| Default benchmark class | `Traineira`, with the same notebook interface usable for other classes later |

### Manifest schema for the first pass

| Field | Meaning |
| --- | --- |
| `target_class` | Selected final class label |
| `split` | `train`, `val`, or `test` |
| `idx_in_class_split` | Stable 1-based index within the class and split |
| `image_path` | Absolute path to the source frame |
| `image_filename` | Source frame filename |
| `image_id` | Image id from the COCO export |
| `camera_id` | Camera id, enriched from Datumaro when available |
| `timestamp` | Frame timestamp, enriched from Datumaro when available |
| `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` | Original bbox in COCO `xywh` format |
| `bbox_area` | Original bbox area |
| `size_bucket` | Size bucket derived from `bbox_area` |
| `crop_x1`, `crop_y1`, `crop_x2`, `crop_y2` | Padded crop coordinates clipped to image limits |
| `crop_w`, `crop_h` | Final crop dimensions in pixels |
| `padding_ratio` | Crop padding ratio |
| `occluded` | Occlusion flag from the bbox attributes |
| `difficult` | Difficult flag from the bbox attributes |
| `n_objects_in_frame` | Number of object bboxes in the frame |
| `other_labels_in_frame` | Unique object labels present in the frame besides the target class |
| `is_benchmark_candidate` | `True` when `bbox_area >= 1024` |

### Notebook contract for the first pass

| Parameter | Default value | Meaning |
| --- | --- | --- |
| `target_class` | `Traineira` | Class under inspection |
| `split` | `train` | Split under inspection |
| `idx` | `1` | 1-based index within the current selection |
| `padding_ratio` | `0.15` | Relative crop padding around the bbox |
| `benchmark_only` | `True` | Restrict the notebook view to `medium+` candidates |

### Expected notebook output per example

| Output block | Content |
| --- | --- |
| Frame view | Original frame with the selected bbox highlighted |
| Crop view | Padded crop clipped to the image boundary |
| Per-item report | Class, split, filename, path, camera, timestamp, bbox coordinates, crop dimensions in pixels, bbox area, size bucket, flags, number of objects in frame, other labels in frame, benchmark status |

### Execution path for the first CBIR evaluation

| Step | Output | Purpose |
| --- | --- | --- |
| 1. Derive a class manifest from the final COCO and Datumaro exports | BBox-level manifest | Establish the CBIR-ready data unit |
| 2. Inspect the selected class in a notebook | Frame + crop + report | Validate crop quality and class support qualitatively |
| 3. Keep all items in the manifest, but benchmark only `medium+` | Stable first benchmark | Reduce noise from tiny objects while keeping full coverage in the data artifact |
| 4. Use `train` as gallery, `val` for tuning, and `test` for the first held-out retrieval checks | Evaluation protocol | Preserve the current dataset partitioning |
| 5. Move from inspection to embedding extraction only after the class-level crops look correct | Epic 0 handoff | Prevent retrieval errors from being confounded with crop errors |

The first Epic 0 checkpoint should therefore be understood as:

| Epic 0 sub-stage | Objective |
| --- | --- |
| `data prep` | Build the bbox-level class manifest |
| `crop view` | Validate the selected class visually |
| `embedding` | Extract representations only after crop quality is validated |
| `Milvus` | Store and query embeddings once the bbox-level item definition is stable |
| `retrieval` | Measure nearest-neighbor behavior for the selected class and then expand to other strong classes |

# HOWTOUSE

Guia rápido para usar o MVP bbox-level de CBIR.

Este fluxo assume:

- dataset local já presente em `data/images_from_dataset`
- anotações finais COCO/Datumaro já presentes em `data/Etapa6_Dataset_PTZ/...`
- Python e dependências gerenciados por `uv`

## 1. Preparar o ambiente

Na raiz do repositório:

```bash
uv sync
```

Isso instala as dependências do MVP, incluindo:

- `open-clip-torch`
- `pymilvus`
- `jupyter`
- `matplotlib`
- `torch`

## 2. Subir a infraestrutura local

Suba o stack do Milvus:

```bash
docker compose up -d
```

Serviços expostos:

- Milvus gRPC: `localhost:19530`
- Milvus health: `localhost:9091`
- Attu UI: `localhost:8000`

Verificar status:

```bash
docker compose ps
```

Parar tudo:

```bash
docker compose down
```

## 3. Gerar o manifesto da classe

O manifesto é o artefato canônico do MVP. Cada linha representa uma bbox da classe escolhida.

Exemplo com `Traineira`:

```bash
uv run python mvp/data_prep.py --target-class Traineira
```

Saída esperada:

- `data/cbir/traineira/v1/manifest.jsonl`
- `data/cbir/traineira/v1/summary.json`
- `data/cbir/traineira/v1/rejected.jsonl`

Observações:

- o benchmark inicial é `medium+`
- `medium+` significa `bbox_area >= 1024`
- `padding_ratio` não é persistido no manifesto
- a unidade canônica persistida é a bbox crua

## 4. Inspecionar visualmente um item

Para auditar um exemplo específico sem abrir notebook:

```bash
uv run python mvp/visualize.py inspect \
  --manifest data/cbir/traineira/v1/manifest.jsonl \
  --class Traineira \
  --split train \
  --idx 1 \
  --benchmark-only \
  --padding-ratio 0.0
```

Saída padrão:

- `artifacts/visualization/traineira/inspect_train_idx_1.png`

O render inclui:

- frame original com bbox destacada
- crop derivado em runtime
- relatório textual com `item_id`, bbox, câmera, timestamp, flags e dimensões

Gerar HTML em vez de PNG:

```bash
uv run python mvp/visualize.py inspect \
  --manifest data/cbir/traineira/v1/manifest.jsonl \
  --class Traineira \
  --split train \
  --idx 1 \
  --benchmark-only \
  --output artifacts/visualization/traineira/inspect_train_idx_1.html
```

## 5. Abrir o notebook principal

O notebook é a superfície principal do Epic 0.

```bash
uv run jupyter notebook mvp/notebook.ipynb
```

Parâmetros default no notebook:

```python
target_class = "Traineira"
split = "train"
idx = 1
padding_ratio = 0.0
benchmark_only = True
top_k = 10
model_name = "openclip-vit-b-32"
device = "cpu"
```

O notebook cobre:

- geração do manifesto
- auditoria por split, câmera e size bucket
- inspeção de um item
- ingestão opcional no Milvus
- busca opcional no Milvus
- métricas iniciais de retrieval

## 6. Fazer uma ingestão de smoke test

Antes de ingerir muitos itens, rode um teste pequeno.

```bash
uv run python mvp/ingest.py \
  --manifest data/cbir/traineira/v1/manifest.jsonl \
  --collection cbir_bbox_traineira_openclip_vit_b_32_openai_v1_smoke \
  --model openclip-vit-b-32 \
  --device cpu \
  --split train \
  --benchmark-only \
  --padding-ratio 0.0 \
  --batch-size 4 \
  --insert-batch-size 8 \
  --recreate \
  --limit 8
```

Isso:

- recorta as bboxes on-demand
- extrai embeddings com `OpenCLIP ViT-B/32`
- normaliza os vetores
- escreve no Milvus em paralelo com um writer thread

Artefato gerado:

- `artifacts/ingest/cbir_bbox_traineira_openclip_vit_b_32_openai_v1_smoke/summary.json`

## 7. Fazer a ingestão “normal”

Exemplo mais próximo do fluxo real:

```bash
uv run python mvp/ingest.py \
  --manifest data/cbir/traineira/v1/manifest.jsonl \
  --collection cbir_bbox_traineira_openclip_vit_b_32_openai_v1 \
  --model openclip-vit-b-32 \
  --device cpu \
  --split train \
  --benchmark-only \
  --padding-ratio 0.0 \
  --batch-size 32 \
  --insert-batch-size 128 \
  --recreate
```

Notas:

- use `--device cuda` se tiver GPU compatível
- `--recreate` apaga e recria a collection
- se quiser manter a collection existente, remova `--recreate`

## 8. Consultar o Milvus pelo Attu

Abra:

```text
http://localhost:8000
```

O Attu está no mesmo `docker compose` e conversa com o Milvus do stack local.

Use o Attu para:

- listar collections
- inspecionar schema
- verificar contagem de entidades
- testar queries e filtros

## 9. Renderizar uma busca top-k fora do notebook

Depois de ingerir uma collection, gere uma visualização de query + vizinhos:

```bash
uv run python mvp/visualize.py search \
  --manifest data/cbir/traineira/v1/manifest.jsonl \
  --collection cbir_bbox_traineira_openclip_vit_b_32_openai_v1_smoke \
  --class Traineira \
  --split train \
  --idx 1 \
  --benchmark-only \
  --padding-ratio 0.0 \
  --top-k 4
```

Saída padrão:

- `artifacts/visualization/traineira/search_train_idx_1.png`

O render inclui:

- query crop
- top-k retornado pelo Milvus
- score
- `item_id`
- `split`
- `camera_id`
- `size_bucket`

## 10. Fluxo recomendado do zero

### Fluxo curto

```bash
uv sync
docker compose up -d
uv run python mvp/data_prep.py --target-class Traineira
uv run python mvp/visualize.py inspect --manifest data/cbir/traineira/v1/manifest.jsonl --class Traineira --split train --idx 1 --benchmark-only --padding-ratio 0.0
uv run jupyter notebook mvp/notebook.ipynb
```

### Fluxo completo com ingestão

```bash
uv sync
docker compose up -d
uv run python mvp/data_prep.py --target-class Traineira
uv run python mvp/ingest.py --manifest data/cbir/traineira/v1/manifest.jsonl --collection cbir_bbox_traineira_openclip_vit_b_32_openai_v1 --model openclip-vit-b-32 --device cpu --split train --benchmark-only --padding-ratio 0.0 --batch-size 32 --insert-batch-size 128 --recreate
uv run python mvp/visualize.py search --manifest data/cbir/traineira/v1/manifest.jsonl --collection cbir_bbox_traineira_openclip_vit_b_32_openai_v1 --class Traineira --split train --idx 1 --benchmark-only --padding-ratio 0.0 --top-k 10
uv run jupyter notebook mvp/notebook.ipynb
```

## 11. Onde cada coisa fica

| Artefato | Local |
| --- | --- |
| Manifesto da classe | `data/cbir/<class_slug>/v1/manifest.jsonl` |
| Sumário do manifesto | `data/cbir/<class_slug>/v1/summary.json` |
| Rejeições do manifesto | `data/cbir/<class_slug>/v1/rejected.jsonl` |
| Artefatos de ingestão | `artifacts/ingest/<collection_name>/` |
| Artefatos de visualização | `artifacts/visualization/<class_slug>/` |

## 12. Problemas comuns

### `No records selected for ingestion`

O filtro atual não encontrou itens. Verifique:

- `--split`
- `--benchmark-only`
- se o manifesto foi gerado para a classe certa

### `idx must be between 1 and N`

O `idx` é 1-based dentro da seleção atual. Ajuste:

- `--split`
- `--benchmark-only`
- `--idx`

### Milvus não responde

Cheque:

```bash
docker compose ps
```

Os serviços esperados são:

- `cbir-etcd`
- `cbir-minio`
- `cbir-milvus`
- `cbir-attu`

### Quero usar outra classe

Exemplo:

```bash
uv run python mvp/data_prep.py --target-class "Navio de Carga Geral"
```

Depois troque:

- o caminho do manifesto
- o nome da collection
- `--class` nos comandos de visualização

## 13. Próximo passo depois deste MVP

Depois que o baseline estiver estável:

1. rodar `UMAP + HDBSCAN` por classe
2. comparar `Traineira` com outras classes fortes
3. testar separação por câmera
4. só depois avaliar embeddings derivados de modelos treinados internamente

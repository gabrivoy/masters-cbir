"""Command-line interface (Typer).

One entry point, six commands:

* ``cbir sample``: build the committable sample dataset from source manifests
* ``cbir index``: embed a manifest's crops into a Milvus collection
* ``cbir export``: snapshot a collection's embeddings to a Parquet cache
* ``cbir seed``: reconstruct a collection from a Parquet cache
* ``cbir api``: run the FastAPI service
* ``cbir app``: run the Streamlit frontend

Every command shares the same model/device options so the embedding model is
selectable everywhere and consistent end to end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from cbir.common.observability import configure_logging, get_logger
from cbir.config import DEFAULT_DEVICE, DEFAULT_MODEL, MODEL_SPECS, repo_root

app = typer.Typer(
    name="cbir",
    help="Content-Based Image Retrieval: index image crops and explore the vector space.",
    add_completion=False,
    no_args_is_help=True,
)
_log = get_logger("cli")


def _model_names() -> str:
    return ", ".join(sorted(MODEL_SPECS))


@app.command()
def sample(
    per_class: int = typer.Option(40, help="Crops to keep per class."),
    output_dir: Path | None = typer.Option(None, help="Where to write the sample."),
    split: str = typer.Option("train", help="Which split to sample from."),
) -> None:
    """Build the small committable sample dataset (crops + manifest)."""
    configure_logging()
    from cbir.data.sample import build_sample

    root = repo_root()
    out = output_dir or root / "cbir" / "sample_data"
    manifest = build_sample(repo_root=root, output_dir=out, per_class=per_class, split=split)
    typer.echo(f"Sample manifest written to {manifest}")


@app.command()
def index(
    manifest: list[Path] = typer.Option(
        ..., "--manifest", "-m", help="Manifest JSONL (repeatable)."
    ),
    collection: str = typer.Option(..., "--collection", "-c", help="Target Milvus collection."),
    model: str = typer.Option(DEFAULT_MODEL, help=f"Embedding model. One of: {_model_names()}"),
    device: str = typer.Option(DEFAULT_DEVICE, help="auto | cuda | mps | cpu."),
    split: str = typer.Option("train", help="Split filter (train/val/test/all)."),
    benchmark_only: bool = typer.Option(False, help="Keep only medium+ benchmark crops."),
    per_class: int | None = typer.Option(None, help="Cap records per class (deterministic head)."),
    batch_size: int = typer.Option(32, help="Embedding batch size."),
    recreate: bool = typer.Option(True, help="Drop and recreate the collection first."),
) -> None:
    """Embed a manifest's crops and index them into Milvus."""
    configure_logging()
    from cbir.index.indexer import run_index

    result = run_index(
        manifest_paths=list(manifest),
        collection_name=collection,
        model_name=model,
        device=device,
        split=split,
        benchmark_only=benchmark_only,
        per_class=per_class,
        batch_size=batch_size,
        recreate=recreate,
    )
    typer.echo(
        f"Indexed {result.inserted_count} items into '{result.collection_name}' "
        f"({result.model_name}, dim={result.embedding_dim}) in {result.duration_seconds:.1f}s."
    )
    typer.echo(f"Summary: {result.summary_path}")


@app.command()
def export(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to export."),
    model: str = typer.Option(DEFAULT_MODEL, help="Model recorded on the collection."),
    output: Path | None = typer.Option(None, help="Parquet output path."),
) -> None:
    """Snapshot a collection's embeddings to a committable Parquet cache."""
    configure_logging()
    from cbir.index.cache import export_collection

    out = output or repo_root() / "cbir" / "sample_data" / "embeddings.parquet"
    path = export_collection(collection, out, model_name=model)
    typer.echo(f"Exported cache to {path}")


@app.command()
def seed(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection to (re)create."),
    parquet: Path | None = typer.Option(None, help="Parquet cache to seed from."),
) -> None:
    """Reconstruct a Milvus collection from a Parquet cache (no model needed)."""
    configure_logging()
    from cbir.index.cache import seed_collection

    src = parquet or repo_root() / "cbir" / "sample_data" / "embeddings.parquet"
    inserted = seed_collection(src, collection)
    typer.echo(f"Seeded {inserted} items into '{collection}' from {src}.")


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8100, help="Bind port."),
    device: str = typer.Option(DEFAULT_DEVICE, help="auto | cuda | mps | cpu."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev)."),
) -> None:
    """Run the FastAPI service."""
    import uvicorn

    configure_logging()
    if reload:
        # Reload needs an import string, so device falls back to its default.
        uvicorn.run("cbir.api.app:create_app", host=host, port=port, factory=True, reload=True)
    else:
        from cbir.api.app import create_app

        uvicorn.run(create_app(device=device), host=host, port=port)


@app.command(name="app")
def app_ui(
    api_url: str = typer.Option("http://localhost:8100", help="Base URL of the CBIR API."),
    port: int = typer.Option(8501, help="Streamlit port."),
) -> None:
    """Run the Streamlit frontend (talks to the API)."""
    configure_logging()
    script = repo_root() / "cbir" / "app" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        f"--server.port={port}",
        "--",
        "--api-url",
        api_url,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    app()

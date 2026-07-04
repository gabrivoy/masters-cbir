from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from data_prep import DEFAULT_PADDING_RATIO, load_manifest, load_split_context, select_manifest_records
from data_prep import slugify_label
from db import DEFAULT_HOST, DEFAULT_PORT, connect, search as milvus_search
from extract import crop_from_record, extract_batch, load_model


def output_dir_for_class(target_class: str, repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / "artifacts" / "visualization" / slugify_label(target_class)


def default_output_path(
    *,
    target_class: str,
    mode: str,
    split: str,
    idx: int,
    extension: str = ".png",
) -> Path:
    return output_dir_for_class(target_class) / f"{mode}_{split}_idx_{idx}{extension}"


def ensure_output_path(
    *,
    target_class: str,
    mode: str,
    split: str,
    idx: int,
    output: Path | None,
) -> Path:
    path = output or default_output_path(
        target_class=target_class,
        mode=mode,
        split=split,
        idx=idx,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def select_record(
    *,
    manifest_path: Path,
    target_class: str,
    split: str,
    idx: int,
    benchmark_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_manifest(manifest_path)
    class_records = [record for record in records if record["target_class"] == target_class]
    selected_records = select_manifest_records(
        class_records,
        split=split,
        benchmark_only=benchmark_only,
    )
    if not selected_records:
        raise ValueError("No records match the current selection.")
    if idx < 1 or idx > len(selected_records):
        raise IndexError(f"idx must be between 1 and {len(selected_records)}; got {idx}.")
    return selected_records[idx - 1], class_records


def frame_overlay(record: dict[str, Any], *, target_class: str) -> Image.Image:
    context = load_split_context(record["split"])
    with Image.open(record["image_path"]) as handle:
        image = handle.convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    frame_annotations = context["annotations_by_image_id"][record["image_id"]]
    for annotation in frame_annotations:
        x, y, width, height = annotation["bbox"]
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        if annotation["id"] == record["annotation_id"]:
            outline = "#ef4444"
            line_width = 4
        elif annotation["label"] == target_class:
            outline = "#f59e0b"
            line_width = 3
        else:
            outline = "#94a3b8"
            line_width = 2
        draw.rectangle((x1, y1, x2, y2), outline=outline, width=line_width)
    return overlay


def report_lines(record: dict[str, Any], *, padding_ratio: float) -> list[str]:
    crop = crop_from_record(record, padding_ratio=padding_ratio)
    return [
        f"class: {record['target_class']}",
        f"split: {record['split']}",
        f"item_id: {record['item_id']}",
        f"annotation_id: {record['annotation_id']}",
        f"image: {record['image_filename']}",
        f"camera_id: {record['camera_id']}",
        f"timestamp: {record['timestamp']}",
        (
            "bbox_xywh: "
            f"({record['bbox_x']:.1f}, {record['bbox_y']:.1f}, "
            f"{record['bbox_w']:.1f}, {record['bbox_h']:.1f})"
        ),
        f"bbox_area: {record['bbox_area']:.1f}",
        f"size_bucket: {record['size_bucket']}",
        f"crop_size_px: {crop.width} x {crop.height}",
        f"padding_ratio: {padding_ratio:.2f}",
        f"occluded: {record['occluded']}",
        f"difficult: {record['difficult']}",
        f"n_objects_in_frame: {record['n_objects_in_frame']}",
        f"other_labels_in_frame: {', '.join(record['other_labels_in_frame']) or '-'}",
        f"benchmark_candidate: {record['is_benchmark_candidate']}",
    ]


def save_figure_and_optional_html(
    figure: plt.Figure,
    *,
    output_path: Path,
    title: str,
    report_lines_for_html: list[str],
) -> Path:
    if output_path.suffix.lower() == ".html":
        png_path = output_path.with_suffix(".png")
        figure.savefig(png_path, dpi=160, bbox_inches="tight")
        with png_path.open("rb") as handle:
            encoded_image = base64.b64encode(handle.read()).decode("ascii")
        items = "".join(f"<li><code>{line}</code></li>" for line in report_lines_for_html)
        html = (
            "<html><body>"
            f"<h1>{title}</h1>"
            f"<img src=\"data:image/png;base64,{encoded_image}\" alt=\"{title}\" />"
            f"<ul>{items}</ul>"
            "</body></html>"
        )
        output_path.write_text(html, encoding="utf-8")
        return output_path

    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return output_path


def render_inspect(
    *,
    record: dict[str, Any],
    padding_ratio: float,
) -> tuple[plt.Figure, list[str]]:
    overlay = frame_overlay(record, target_class=record["target_class"])
    crop = crop_from_record(record, padding_ratio=padding_ratio)
    report = report_lines(record, padding_ratio=padding_ratio)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(18, 11),
        gridspec_kw={"height_ratios": [4, 1.8]},
    )
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title("Frame with bbox overlay")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(crop)
    axes[0, 1].set_title("Crop derived at runtime")
    axes[0, 1].axis("off")

    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    report_text = "\n".join(report)
    axes[1, 0].text(0.0, 1.0, report_text, va="top", ha="left", family="monospace", fontsize=10)
    figure.suptitle(
        f"{record['target_class']} | {record['split']} | idx={record['idx_in_class_split']}",
        fontsize=16,
    )
    figure.tight_layout()
    return figure, report


def search_hits(
    *,
    record: dict[str, Any],
    collection_name: str,
    manifest_records: list[dict[str, Any]],
    model_name: str,
    device: str,
    padding_ratio: float,
    top_k: int,
    host: str,
    port: str,
) -> list[dict[str, Any]]:
    connect(host=host, port=port)
    model_bundle = load_model(model_name=model_name, device=device)
    extracted = extract_batch(
        model_bundle,
        [record],
        batch_size=1,
        device=model_bundle["device"],
        padding_ratio=padding_ratio,
    )
    query_embedding = extracted[0]["embedding"]
    raw_hits = milvus_search(collection_name, query_embedding, top_k=top_k)
    records_by_item_id = {item["item_id"]: item for item in manifest_records}

    hits: list[dict[str, Any]] = []
    for hit in raw_hits:
        hit_record = records_by_item_id.get(hit["item_id"])
        if hit_record is None:
            continue
        payload = dict(hit)
        payload["record"] = hit_record
        hits.append(payload)
    return hits


def render_search(
    *,
    record: dict[str, Any],
    hits: list[dict[str, Any]],
    padding_ratio: float,
) -> tuple[plt.Figure, list[str]]:
    query_crop = crop_from_record(record, padding_ratio=padding_ratio)
    columns = 3
    rows = max(2, math.ceil((len(hits) + 1) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(18, 5 * rows))
    axes = list(axes.flatten())

    axes[0].imshow(query_crop)
    axes[0].set_title(f"Query\n{record['item_id']}")
    axes[0].axis("off")

    report = report_lines(record, padding_ratio=padding_ratio)
    for axis, hit in zip(axes[1:], hits, strict=False):
        hit_record = hit["record"]
        axis.imshow(crop_from_record(hit_record, padding_ratio=padding_ratio))
        axis.set_title(
            "\n".join(
                [
                    f"score={hit['score']:.4f}",
                    hit["item_id"],
                    f"{hit_record['split']} | {hit_record['camera_id']}",
                    hit_record["size_bucket"],
                ]
            ),
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes[len(hits) + 1 :]:
        axis.axis("off")

    figure.suptitle(
        f"Query + top-{len(hits)} search results | {record['target_class']}",
        fontsize=16,
    )
    figure.tight_layout()
    return figure, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render bbox-level inspection and search outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Render a single inspected item.")
    inspect_parser.add_argument("--manifest", type=Path, required=True)
    inspect_parser.add_argument("--class", dest="target_class", required=True)
    inspect_parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    inspect_parser.add_argument("--idx", type=int, default=1)
    inspect_parser.add_argument("--padding-ratio", type=float, default=DEFAULT_PADDING_RATIO)
    inspect_parser.add_argument("--benchmark-only", action="store_true")
    inspect_parser.add_argument("--output", type=Path, default=None)

    search_parser = subparsers.add_parser("search", help="Render a query and its top-k search results.")
    search_parser.add_argument("--manifest", type=Path, required=True)
    search_parser.add_argument("--collection", required=True)
    search_parser.add_argument("--class", dest="target_class", required=True)
    search_parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    search_parser.add_argument("--idx", type=int, default=1)
    search_parser.add_argument("--padding-ratio", type=float, default=DEFAULT_PADDING_RATIO)
    search_parser.add_argument("--benchmark-only", action="store_true")
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--model", default="openclip-vit-b-32")
    search_parser.add_argument("--device", default="cpu")
    search_parser.add_argument("--host", default=DEFAULT_HOST)
    search_parser.add_argument("--port", default=DEFAULT_PORT)
    search_parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record, manifest_records = select_record(
        manifest_path=args.manifest,
        target_class=args.target_class,
        split=args.split,
        idx=args.idx,
        benchmark_only=args.benchmark_only,
    )
    output_path = ensure_output_path(
        target_class=args.target_class,
        mode=args.command,
        split=args.split,
        idx=args.idx,
        output=args.output,
    )

    if args.command == "inspect":
        figure, report = render_inspect(record=record, padding_ratio=args.padding_ratio)
        saved_path = save_figure_and_optional_html(
            figure,
            output_path=output_path,
            title=f"Inspect {record['item_id']}",
            report_lines_for_html=report,
        )
        plt.close(figure)
        print(saved_path)
        return 0

    hits = search_hits(
        record=record,
        collection_name=args.collection,
        manifest_records=manifest_records,
        model_name=args.model,
        device=args.device,
        padding_ratio=args.padding_ratio,
        top_k=args.top_k,
        host=args.host,
        port=args.port,
    )
    figure, report = render_search(record=record, hits=hits, padding_ratio=args.padding_ratio)
    saved_path = save_figure_and_optional_html(
        figure,
        output_path=output_path,
        title=f"Search {record['item_id']}",
        report_lines_for_html=report,
    )
    plt.close(figure)
    print(saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

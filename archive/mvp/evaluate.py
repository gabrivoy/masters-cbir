from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data_prep import DEFAULT_PADDING_RATIO, load_manifest
from db import DEFAULT_HOST, DEFAULT_PORT, connect, count, search as milvus_search
from extract import extract_batch, load_model
from sampling import count_by_class, filter_records, sample_records_per_class

DEFAULT_THRESHOLDS = (0.8, 0.7, 0.6, 0.5)


def threshold_column(threshold: float) -> str:
    return f"score_ge_{threshold:.2f}".replace(".", "_")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_records_from_manifests(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        for record in load_manifest(manifest_path):
            item_id = record["item_id"]
            if item_id in seen:
                continue
            seen.add(item_id)
            records.append(record)
    return records


def record_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["item_id"]: record for record in records}


def selected_query_records(
    records: list[dict[str, Any]],
    *,
    query_split: str,
    benchmark_only: bool,
    target_classes: set[str] | None,
    sample_per_class: int | None,
    sample_strategy: str,
    sample_seed: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    filtered = filter_records(
        records,
        split=query_split,
        benchmark_only=benchmark_only,
        target_classes=target_classes,
    )
    sampled = sample_records_per_class(
        filtered,
        sample_per_class=sample_per_class,
        strategy=sample_strategy,
        seed=sample_seed,
    )
    if limit is not None:
        sampled = sampled[:limit]
    return sampled


def hit_value(
    hit_record: dict[str, Any] | None,
    hit_payload: dict[str, Any],
    field: str,
    default: Any = None,
) -> Any:
    if hit_record is not None and field in hit_record:
        return hit_record[field]
    return hit_payload.get(field, default)


def build_ranking_row(
    *,
    query_record: dict[str, Any],
    hit_payload: dict[str, Any],
    hit_record: dict[str, Any] | None,
    rank: int,
    thresholds: list[float],
) -> dict[str, Any]:
    score = float(hit_payload["score"])
    hit_item_id = hit_payload["item_id"]
    hit_class = hit_value(hit_record, hit_payload, "target_class")
    row: dict[str, Any] = {
        "query_item_id": query_record["item_id"],
        "query_class": query_record["target_class"],
        "query_split": query_record["split"],
        "query_image_filename": query_record["image_filename"],
        "query_bbox_x": query_record["bbox_x"],
        "query_bbox_y": query_record["bbox_y"],
        "query_bbox_w": query_record["bbox_w"],
        "query_bbox_h": query_record["bbox_h"],
        "query_camera_id": query_record["camera_id"],
        "query_timestamp": query_record["timestamp"],
        "rank": rank,
        "hit_item_id": hit_item_id,
        "hit_class": hit_class,
        "hit_split": hit_value(hit_record, hit_payload, "split"),
        "hit_image_filename": hit_value(hit_record, hit_payload, "image_filename"),
        "hit_bbox_x": hit_value(hit_record, hit_payload, "bbox_x"),
        "hit_bbox_y": hit_value(hit_record, hit_payload, "bbox_y"),
        "hit_bbox_w": hit_value(hit_record, hit_payload, "bbox_w"),
        "hit_bbox_h": hit_value(hit_record, hit_payload, "bbox_h"),
        "hit_camera_id": hit_value(hit_record, hit_payload, "camera_id"),
        "hit_timestamp": hit_value(hit_record, hit_payload, "timestamp"),
        "score": score,
        "is_same_class": query_record["target_class"] == hit_class,
        "is_self_hit": query_record["item_id"] == hit_item_id,
        "same_camera": query_record["camera_id"] == hit_value(hit_record, hit_payload, "camera_id"),
        "same_image_filename": query_record["image_filename"]
        == hit_value(hit_record, hit_payload, "image_filename"),
    }
    for threshold in thresholds:
        row[threshold_column(threshold)] = bool(score >= threshold)
    return row


def build_query_summary(
    *,
    query_record: dict[str, Any],
    hit_rows: list[dict[str, Any]],
    thresholds: list[float],
) -> dict[str, Any]:
    top1 = hit_rows[0] if hit_rows else {}
    row: dict[str, Any] = {
        "query_item_id": query_record["item_id"],
        "query_class": query_record["target_class"],
        "top1_hit_class": top1.get("hit_class"),
        "top1_score": top1.get("score"),
        "top1_is_same_class": bool(top1.get("is_same_class", False)),
    }
    for threshold in thresholds:
        column = threshold_column(threshold)
        suffix = column.removeprefix("score_ge_")
        row[f"any_hit_ge_{suffix}"] = any(bool(hit.get(column, False)) for hit in hit_rows)
        row[f"count_hits_ge_{suffix}"] = sum(1 for hit in hit_rows if bool(hit.get(column, False)))
    return row


def precision_at(ranking_df: pd.DataFrame, k: int) -> float | None:
    if ranking_df.empty:
        return None
    subset = ranking_df[ranking_df["rank"] <= k]
    if subset.empty:
        return None
    per_query = subset.groupby("query_item_id")["is_same_class"].mean()
    return finite_float(per_query.mean())


def mean_reciprocal_rank(ranking_df: pd.DataFrame) -> float | None:
    if ranking_df.empty:
        return None
    reciprocal_ranks: list[float] = []
    for _, group in ranking_df.groupby("query_item_id"):
        same_class = group[group["is_same_class"]].sort_values("rank")
        reciprocal_ranks.append(0.0 if same_class.empty else 1.0 / float(same_class.iloc[0]["rank"]))
    if not reciprocal_ranks:
        return None
    return finite_float(sum(reciprocal_ranks) / len(reciprocal_ranks))


def threshold_metrics(query_summary_df: pd.DataFrame, thresholds: list[float]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    query_count = len(query_summary_df)
    for threshold in thresholds:
        suffix = threshold_column(threshold).removeprefix("score_ge_")
        accepted = query_summary_df[query_summary_df["top1_score"].fillna(float("-inf")) >= threshold]
        precision = None
        if not accepted.empty:
            precision = finite_float(accepted["top1_is_same_class"].mean())
        metrics[str(threshold)] = {
            "accepted_queries": int(len(accepted)),
            "thresholded_precision": precision,
            "thresholded_coverage": finite_float(len(accepted) / query_count) if query_count else None,
            "any_hit_column": f"any_hit_ge_{suffix}",
            "count_hit_column": f"count_hits_ge_{suffix}",
        }
    return metrics


def build_threshold_distribution(
    ranking_df: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    if ranking_df.empty:
        return pd.DataFrame(
            columns=[
                "threshold",
                "rank",
                "query_class",
                "hit_class",
                "total_hits",
                "hits_ge_threshold",
                "hit_rate_ge_threshold",
            ]
        )

    frames: list[pd.DataFrame] = []
    for threshold in thresholds:
        column = threshold_column(threshold)
        grouped = (
            ranking_df.groupby(["rank", "query_class", "hit_class"], dropna=False)[column]
            .agg(total_hits="count", hits_ge_threshold="sum", hit_rate_ge_threshold="mean")
            .reset_index()
        )
        grouped.insert(0, "threshold", threshold)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def top1_confusion(query_summary_df: pd.DataFrame) -> pd.DataFrame:
    if query_summary_df.empty:
        return pd.DataFrame()
    return pd.crosstab(
        query_summary_df["query_class"],
        query_summary_df["top1_hit_class"].fillna("<no-hit>"),
        rownames=["query_class"],
        colnames=["top1_hit_class"],
    )


def write_plots(
    *,
    ranking_df: pd.DataFrame,
    query_summary_df: pd.DataFrame,
    thresholds: list[float],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if not ranking_df.empty:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(ranking_df["score"], bins=30, color="#1f77b4", edgecolor="white")
        axis.set_title("Retrieved Hit Score Distribution")
        axis.set_xlabel("Cosine similarity")
        axis.set_ylabel("Hit count")
        path = output_dir / "score_histogram.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths["score_histogram"] = str(path)

    if not query_summary_df.empty:
        figure, axis = plt.subplots(figsize=(8, 5))
        query_summary_df["top1_score"].dropna().plot.hist(
            ax=axis,
            bins=30,
            color="#ff7f0e",
            edgecolor="white",
        )
        axis.set_title("Top-1 Score Distribution")
        axis.set_xlabel("Top-1 cosine similarity")
        axis.set_ylabel("Query count")
        path = output_dir / "top1_score_distribution.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths["top1_score_distribution"] = str(path)

    if not ranking_df.empty:
        figure, axis = plt.subplots(figsize=(9, 5))
        for threshold in thresholds:
            column = threshold_column(threshold)
            by_rank = ranking_df.groupby("rank")[column].mean().sort_index()
            axis.plot(by_rank.index, by_rank.values, marker="o", label=f">= {threshold:.2f}")
        axis.set_title("Rate of Hits Above Threshold by Rank")
        axis.set_xlabel("Rank")
        axis.set_ylabel("Fraction of hits")
        axis.set_ylim(0, 1.05)
        axis.legend()
        path = output_dir / "threshold_rate_by_rank.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths["threshold_rate_by_rank"] = str(path)

    return paths


def evaluate(
    *,
    collection_name: str,
    query_manifest_paths: list[Path],
    query_split: str,
    benchmark_only: bool,
    model_name: str,
    device: str,
    padding_ratio: float,
    top_k: int,
    thresholds: list[float],
    output_dir: Path,
    host: str,
    port: str,
    target_classes: set[str] | None = None,
    batch_size: int = 32,
    sample_per_class: int | None = None,
    sample_strategy: str = "stratified",
    sample_seed: int = 42,
    limit: int | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    started_at = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_records_from_manifests(query_manifest_paths)
    records_by_item_id = record_lookup(all_records)
    query_records = selected_query_records(
        all_records,
        query_split=query_split,
        benchmark_only=benchmark_only,
        target_classes=target_classes,
        sample_per_class=sample_per_class,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        limit=limit,
    )
    if not query_records:
        raise ValueError("No query records selected for evaluation.")

    connect(host=host, port=port)
    collection_count = count(collection_name)
    model_bundle = load_model(model_name=model_name, device=device)

    ranking_rows: list[dict[str, Any]] = []
    query_summary_rows: list[dict[str, Any]] = []
    removed_self_hits = 0
    missing_hit_manifest_count = 0

    for start in tqdm(range(0, len(query_records), batch_size), desc="Evaluating", unit="batch"):
        batch_records = query_records[start : start + batch_size]
        extracted_items = extract_batch(
            model_bundle,
            batch_records,
            batch_size=batch_size,
            device=model_bundle["device"],
            padding_ratio=padding_ratio,
        )
        for item in extracted_items:
            query_record = item["record"]
            raw_hits = milvus_search(
                collection_name,
                item["embedding"],
                top_k=max(1, min(top_k + 1, collection_count)),
            )
            hit_rows: list[dict[str, Any]] = []
            for hit_payload in raw_hits:
                if hit_payload["item_id"] == query_record["item_id"]:
                    removed_self_hits += 1
                    continue
                hit_record = records_by_item_id.get(hit_payload["item_id"])
                if hit_record is None:
                    missing_hit_manifest_count += 1
                row = build_ranking_row(
                    query_record=query_record,
                    hit_payload=hit_payload,
                    hit_record=hit_record,
                    rank=len(hit_rows) + 1,
                    thresholds=thresholds,
                )
                hit_rows.append(row)
                if len(hit_rows) >= top_k:
                    break

            ranking_rows.extend(hit_rows)
            query_summary_rows.append(
                build_query_summary(
                    query_record=query_record,
                    hit_rows=hit_rows,
                    thresholds=thresholds,
                )
            )

    ranking_df = pd.DataFrame(ranking_rows)
    query_summary_df = pd.DataFrame(query_summary_rows)
    threshold_distribution_df = build_threshold_distribution(ranking_df, thresholds)
    confusion_df = top1_confusion(query_summary_df)

    ranking_path = output_dir / "ranking.csv"
    query_summary_path = output_dir / "query_summary.csv"
    threshold_distribution_path = output_dir / "threshold_distribution.csv"
    confusion_path = output_dir / "top1_confusion.csv"
    ranking_df.to_csv(ranking_path, index=False)
    query_summary_df.to_csv(query_summary_path, index=False)
    threshold_distribution_df.to_csv(threshold_distribution_path, index=False)
    confusion_df.to_csv(confusion_path)

    plot_paths = write_plots(
        ranking_df=ranking_df,
        query_summary_df=query_summary_df,
        thresholds=thresholds,
        output_dir=output_dir,
    )

    query_classes = sorted(query_summary_df["query_class"].dropna().unique().tolist())
    hit_classes = sorted(ranking_df["hit_class"].dropna().unique().tolist()) if not ranking_df.empty else []
    top1_accuracy = None
    if not query_summary_df.empty:
        top1_accuracy = finite_float(query_summary_df["top1_is_same_class"].mean())

    summary = {
        "collection_name": collection_name,
        "collection_count": collection_count,
        "query_manifest_paths": [str(path) for path in query_manifest_paths],
        "query_split": query_split,
        "benchmark_only": benchmark_only,
        "padding_ratio": padding_ratio,
        "model_name": model_bundle["model_name"],
        "model_slug": model_bundle["model_slug"],
        "embedding_dim": int(model_bundle["embedding_dim"]),
        "device_requested": device,
        "device_resolved": model_bundle["device"],
        "top_k": top_k,
        "thresholds": thresholds,
        "sample_per_class": sample_per_class,
        "sample_strategy": sample_strategy,
        "sample_seed": sample_seed,
        "limit": limit,
        "records_loaded": len(all_records),
        "records_loaded_by_class": count_by_class(all_records),
        "query_count": len(query_records),
        "query_count_by_class": count_by_class(query_records),
        "ranking_rows": len(ranking_df),
        "expected_full_ranking_rows": len(query_records) * top_k,
        "classes_in_queries": query_classes,
        "classes_in_hits": hit_classes,
        "is_single_class_calibration": len(set(query_classes) | set(hit_classes)) <= 1,
        "metrics": {
            "top1_accuracy": top1_accuracy,
            "precision_at_5": precision_at(ranking_df, 5),
            "precision_at_10": precision_at(ranking_df, 10),
            "precision_at_30": precision_at(ranking_df, 30),
            "mrr": mean_reciprocal_rank(ranking_df),
            "thresholds": threshold_metrics(query_summary_df, thresholds),
        },
        "leakage_checks": {
            "removed_self_hits": removed_self_hits,
            "remaining_self_hits": int(ranking_df["is_self_hit"].sum()) if not ranking_df.empty else 0,
            "same_image_filename_hits": int(ranking_df["same_image_filename"].sum())
            if not ranking_df.empty
            else 0,
            "hit_splits_observed": sorted(ranking_df["hit_split"].dropna().unique().tolist())
            if not ranking_df.empty
            else [],
            "missing_hit_manifest_count": missing_hit_manifest_count,
        },
        "outputs": {
            "ranking_csv": str(ranking_path),
            "query_summary_csv": str(query_summary_path),
            "threshold_distribution_csv": str(threshold_distribution_path),
            "top1_confusion_csv": str(confusion_path),
            "plots": plot_paths,
        },
        "duration_seconds": time.time() - started_at,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bbox-level CBIR ranking results.")
    parser.add_argument("--collection", required=True, help="Milvus collection to query.")
    parser.add_argument(
        "--query-manifest",
        type=Path,
        action="append",
        required=True,
        help="Path to manifest.jsonl used for queries and hit metadata lookup. Repeat for multiclass.",
    )
    parser.add_argument(
        "--query-split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="Split used as query set.",
    )
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--model", default="openclip-vit-b-32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--padding-ratio", type=float, default=DEFAULT_PADDING_RATIO)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Score thresholds used for boolean columns and summaries.",
    )
    parser.add_argument(
        "--class",
        dest="target_classes",
        action="append",
        default=None,
        help="Optional query class filter. Repeat for multiple classes.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument(
        "--sample-strategy",
        choices=("stratified", "head"),
        default="stratified",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate(
        collection_name=args.collection,
        query_manifest_paths=args.query_manifest,
        query_split=args.query_split,
        benchmark_only=args.benchmark_only,
        model_name=args.model,
        device=args.device,
        padding_ratio=args.padding_ratio,
        top_k=args.top_k,
        thresholds=list(args.thresholds),
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        target_classes=set(args.target_classes) if args.target_classes else None,
        batch_size=args.batch_size,
        sample_per_class=args.sample_per_class,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

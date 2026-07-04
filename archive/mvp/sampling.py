from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any


def deterministic_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("target_class", ""),
        record.get("split", ""),
        str(record.get("camera_id") or ""),
        record.get("size_bucket", ""),
        record.get("image_filename", ""),
        float(record.get("bbox_y", 0.0)),
        float(record.get("bbox_x", 0.0)),
        float(record.get("bbox_h", 0.0)),
        float(record.get("bbox_w", 0.0)),
        int(record.get("annotation_id", 0)),
        record.get("item_id", ""),
    )


def filter_records(
    records: list[dict[str, Any]],
    *,
    split: str = "all",
    benchmark_only: bool = False,
    target_classes: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered = records
    if split != "all":
        filtered = [record for record in filtered if record["split"] == split]
    if benchmark_only:
        filtered = [record for record in filtered if record["is_benchmark_candidate"]]
    if target_classes is not None:
        filtered = [record for record in filtered if record["target_class"] in target_classes]
    return sorted(filtered, key=deterministic_sort_key)


def records_by_class(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["target_class"]].append(record)
    return {label: sorted(items, key=deterministic_sort_key) for label, items in grouped.items()}


def count_by_field(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(field)) for record in records).items()))


def count_by_class(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["target_class"] for record in records).items()))


def count_by_split(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["split"] for record in records).items()))


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _shuffled(
    records: list[dict[str, Any]],
    *,
    seed: int,
    parts: tuple[object, ...],
) -> list[dict[str, Any]]:
    items = sorted(records, key=deterministic_sort_key)
    random.Random(_stable_seed(seed, *parts)).shuffle(items)
    return items


def _allocate_stratified_quotas(
    strata: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    sample_size: int,
) -> dict[tuple[str, str], int]:
    total = sum(len(items) for items in strata.values())
    quotas: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, int, tuple[str, str]]] = []

    for stratum_key, items in sorted(strata.items()):
        raw_quota = len(items) * sample_size / total
        quota = min(len(items), int(raw_quota))
        quotas[stratum_key] = quota
        remainders.append((raw_quota - quota, len(items), stratum_key))

    remaining = sample_size - sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))

    while remaining > 0:
        changed = False
        for _, _, stratum_key in remainders:
            if remaining <= 0:
                break
            if quotas[stratum_key] >= len(strata[stratum_key]):
                continue
            quotas[stratum_key] += 1
            remaining -= 1
            changed = True
        if not changed:
            break

    return quotas


def sample_records_per_class(
    records: list[dict[str, Any]],
    *,
    sample_per_class: int | None,
    strategy: str = "stratified",
    seed: int = 42,
) -> list[dict[str, Any]]:
    if sample_per_class is None:
        return sorted(records, key=deterministic_sort_key)
    if sample_per_class <= 0:
        raise ValueError("--sample-per-class must be a positive integer.")
    if strategy not in {"stratified", "head"}:
        raise ValueError("sample strategy must be either 'stratified' or 'head'.")

    selected: list[dict[str, Any]] = []
    for class_label, class_records in sorted(records_by_class(records).items()):
        if len(class_records) <= sample_per_class:
            selected.extend(class_records)
            continue

        if strategy == "head":
            selected.extend(class_records[:sample_per_class])
            continue

        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in class_records:
            stratum_key = (
                str(record.get("camera_id") or ""),
                str(record.get("size_bucket") or ""),
            )
            strata[stratum_key].append(record)

        quotas = _allocate_stratified_quotas(strata, sample_size=sample_per_class)
        for stratum_key, quota in quotas.items():
            if quota <= 0:
                continue
            shuffled = _shuffled(strata[stratum_key], seed=seed, parts=(class_label, *stratum_key))
            selected.extend(shuffled[:quota])

    return sorted(selected, key=deterministic_sort_key)

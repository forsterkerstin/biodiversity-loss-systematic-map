from __future__ import annotations

import ast
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data_helpers._config import (
    LABEL_CONFIGS_RAW as _LABEL_CONFIGS_RAW,
    MISSING_KEY,
    MULTI_LABEL_KEY,
    get_split_label_path,
    get_split_ratios,
    get_split_seed,
)

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class LabelConfig:
    name: str
    strat_column: str
    missing_key: str = MISSING_KEY
    drop_if_all_missing: tuple[str, ...] | None = None
    source_weights: dict[str, float] | None = None


LABEL_CONFIGS: dict[str, LabelConfig] = {
    name: LabelConfig(
        name=name,
        strat_column=raw["strat_column"],
        missing_key=raw.get("missing_key", MISSING_KEY),
        drop_if_all_missing=tuple(raw["drop_if_all_missing"]) if raw.get("drop_if_all_missing") else None,
    )
    for name, raw in _LABEL_CONFIGS_RAW.items()
}


class SplitError(RuntimeError):
    pass


def _parse_labels(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    if text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [
                    _clean_label(item)
                    for item in parsed
                    if _clean_label(item)
                ]
            if isinstance(parsed, str):
                cleaned = _clean_label(parsed)
                return [cleaned] if cleaned else []

    if ";" in text:
        parts = text.split(";")
    elif "|" in text:
        parts = text.split("|")
    else:
        parts = [text]

    return [_clean_label(part) for part in parts if _clean_label(part)]


def _clean_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("\"").strip("'")


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise SplitError(f"Empty CSV: {path}")
        rows: list[list[str]] = [row for row in reader]
    return header, rows


def _write_csv_rows(path: Path, header: list[str], rows: Iterable[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _col_index(header: list[str], name: str) -> int:
    for idx, col in enumerate(header):
        if col == name:
            return idx
    raise SplitError(f"Column '{name}' not found in header: {header}")


def _cell(row: list[str], idx: int) -> str:
    if idx < 0:
        return ""
    if idx >= len(row):
        return ""
    return row[idx]


def _allocation_counts(n: int, ratios: list[float]) -> list[int]:
    if n == 0:
        return [0 for _ in ratios]
    raw = [n * ratio for ratio in ratios]
    base = [int(math.floor(value)) for value in raw]
    remainder = n - sum(base)
    fractions = [raw[i] - base[i] for i in range(len(raw))]
    order = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    return base


def _split_strata(
    strata: dict[str, list[list[str]]],
    ratios: dict[str, float],
    seed: int,
) -> tuple[dict[str, list[list[str]]], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    split_rows = {split: [] for split in SPLITS}
    strata_counts: dict[str, dict[str, int]] = {}

    ratio_list = [ratios[split] for split in SPLITS]

    for stratum in sorted(strata.keys()):
        rows = list(strata[stratum])
        rng.shuffle(rows)
        counts = _allocation_counts(len(rows), ratio_list)
        offsets = [0, counts[0], counts[0] + counts[1]]
        split_rows["train"].extend(rows[offsets[0] : offsets[0] + counts[0]])
        split_rows["dev"].extend(rows[offsets[1] : offsets[1] + counts[1]])
        split_rows["test"].extend(rows[offsets[2] : offsets[2] + counts[2]])
        strata_counts[stratum] = {
            "total": len(rows),
            "train": counts[0],
            "dev": counts[1],
            "test": counts[2],
        }

    return split_rows, strata_counts


def _split_with_source_weights(
    source_groups: dict[str, list[list[str]]],
    target_weights: dict[str, float],
    ratios: dict[str, float],
    seed: int,
) -> tuple[dict[str, list[list[str]]], dict[str, dict[str, int]]]:
    """Split rows by source, rebalancing to target_weights proportions.

    Finds the largest balanced total N such that no source needs more records
    than it has, then samples each source to its target count and splits
    train/dev/test within each source.

    Records from sources not listed in target_weights are excluded (noted in
    strata_counts under their source name with sampled=0).
    """
    rng = random.Random(seed)

    total_w = sum(target_weights.values())
    norm_weights = {s: w / total_w for s, w in target_weights.items()}

    # Maximum balanced N: limited by the scarcest source relative to its weight.
    max_n: float = float("inf")
    for source, weight in norm_weights.items():
        available = len(source_groups.get(source, []))
        if weight > 0:
            max_n = min(max_n, available / weight)
    if max_n == float("inf"):
        max_n = 0.0

    target_n = int(math.floor(max_n))

    ratio_list = [ratios[split] for split in SPLITS]
    split_rows: dict[str, list[list[str]]] = {split: [] for split in SPLITS}
    strata_counts: dict[str, dict[str, int]] = {}

    for source in sorted(norm_weights.keys()):
        rows = list(source_groups.get(source, []))
        rng.shuffle(rows)
        n = min(int(round(target_n * norm_weights[source])), len(rows))
        sampled = rows[:n]
        counts = _allocation_counts(len(sampled), ratio_list)
        offsets = [0, counts[0], counts[0] + counts[1]]
        split_rows["train"].extend(sampled[offsets[0]: offsets[0] + counts[0]])
        split_rows["dev"].extend(sampled[offsets[1]: offsets[1] + counts[1]])
        split_rows["test"].extend(sampled[offsets[2]: offsets[2] + counts[2]])
        strata_counts[source] = {
            "total": len(rows),
            "sampled": n,
            "train": counts[0],
            "dev": counts[1],
            "test": counts[2],
        }

    # Record excluded sources (not in target_weights) for transparency.
    for source, rows in source_groups.items():
        if source not in norm_weights:
            strata_counts[f"__excluded__{source}"] = {
                "total": len(rows),
                "sampled": 0,
                "train": 0,
                "dev": 0,
                "test": 0,
            }

    return split_rows, strata_counts


def _get_ratios() -> dict[str, float]:
    try:
        return get_split_ratios()
    except ValueError as exc:
        raise SplitError(str(exc)) from exc


def _get_label_path(label: str) -> Path:
    try:
        return get_split_label_path(label)
    except ValueError as exc:
        raise SplitError(str(exc)) from exc


def _stratum_from_value(value: str, missing_key: str) -> tuple[str, list[str]]:
    labels = _parse_labels(value)
    if not labels:
        return missing_key, labels
    if len(labels) == 1:
        return labels[0].lower(), labels
    return MULTI_LABEL_KEY, labels


def split_label_dataset(
    label: str,
    label_path: Path,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, object]:
    config = LABEL_CONFIGS[label]
    csv_files = list(label_path.glob("*.csv"))
    if not csv_files:
        raise SplitError(f"No CSV found in {label_path}")
    if len(csv_files) > 1:
        raise SplitError(f"Multiple CSV files found in {label_path}: {csv_files}")

    input_file = csv_files[0]
    header, rows = _read_csv_rows(input_file)
    key_idx = _col_index(header, "UT (Unique WOS ID)")
    strat_idx = _col_index(header, config.strat_column)

    drop_columns: list[int] = []
    if config.drop_if_all_missing:
        for col in config.drop_if_all_missing:
            drop_columns.append(_col_index(header, col))

    seen_keys: set[str] = set()
    deduped_rows: list[list[str]] = []
    duplicates: list[list[str]] = []
    missing_key_count = 0

    for row in rows:
        key = _cell(row, key_idx).strip()
        if not key:
            missing_key_count += 1
            deduped_rows.append(row)
            continue
        if key in seen_keys:
            duplicates.append(row)
            continue
        seen_keys.add(key)
        deduped_rows.append(row)

    abstract_idx = header.index("Abstract") if "Abstract" in header else -1
    no_abstract_rows: list[list[str]] = []
    rows_after_abstract_drop: list[list[str]] = []
    for row in deduped_rows:
        if abstract_idx >= 0 and not _cell(row, abstract_idx).strip():
            no_abstract_rows.append(row)
        else:
            rows_after_abstract_drop.append(row)
    deduped_rows = rows_after_abstract_drop

    dropped_rows: list[list[str]] = []
    kept_rows: list[list[str]] = []
    if drop_columns:
        for row in deduped_rows:
            parsed_values = [_parse_labels(_cell(row, idx)) for idx in drop_columns]
            if all(not labels for labels in parsed_values):
                dropped_rows.append(row)
            else:
                kept_rows.append(row)
    else:
        kept_rows = deduped_rows

    strata: dict[str, list[list[str]]] = {}
    for row in kept_rows:
        raw_value = _cell(row, strat_idx)
        stratum, _ = _stratum_from_value(raw_value, config.missing_key)
        strata.setdefault(stratum, []).append(row)

    if config.source_weights is not None:
        split_rows, strata_counts = _split_with_source_weights(strata, config.source_weights, ratios, seed)
    else:
        split_rows, strata_counts = _split_strata(strata, ratios, seed)

    train_dir = label_path / "train"
    dev_dir = label_path / "dev"
    test_dir = label_path / "test"
    inprocess_dir = label_path / "in-process"
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    inprocess_dir.mkdir(parents=True, exist_ok=True)

    _write_csv_rows(train_dir / input_file.name, header, split_rows["train"])
    _write_csv_rows(dev_dir / input_file.name, header, split_rows["dev"])
    _write_csv_rows(test_dir / input_file.name, header, split_rows["test"])

    if duplicates:
        _write_csv_rows(inprocess_dir / "duplicates.csv", header, duplicates)

    if no_abstract_rows:
        _write_csv_rows(inprocess_dir / "no_abstract_rows.csv", header, no_abstract_rows)

    if dropped_rows:
        drop_header = header + ["__drop_reason"]
        dropped_with_reason = [row + ["all_label_columns_missing"] for row in dropped_rows]
        _write_csv_rows(inprocess_dir / "dropped_rows.csv", drop_header, dropped_with_reason)

    strata_counts_path = inprocess_dir / "strata_counts.csv"
    has_sampled = any("sampled" in counts for counts in strata_counts.values())
    if has_sampled:
        strata_header = ["stratum", "total", "sampled", "train", "dev", "test"]
        strata_rows = [
            [stratum, counts["total"], counts.get("sampled", ""), counts["train"], counts["dev"], counts["test"]]
            for stratum, counts in sorted(strata_counts.items())
        ]
    else:
        strata_header = ["stratum", "total", "train", "dev", "test"]
        strata_rows = [
            [stratum, counts["total"], counts["train"], counts["dev"], counts["test"]]
            for stratum, counts in sorted(strata_counts.items())
        ]
    _write_csv_rows(strata_counts_path, strata_header, strata_rows)

    summary = {
        "label": label,
        "input_file": str(input_file),
        "input_rows": len(rows),
        "dedup_rows": len(deduped_rows) + len(no_abstract_rows),
        "duplicates": len(duplicates),
        "missing_key_rows": missing_key_count,
        "no_abstract_rows": len(no_abstract_rows),
        "dropped_rows": len(dropped_rows),
        "kept_rows": len(kept_rows),
        "split_ratios": {split: ratios[split] for split in SPLITS},
        "split_counts": {split: len(split_rows[split]) for split in SPLITS},
        "stratification": {
            "column": config.strat_column,
            "multi_label_group": MULTI_LABEL_KEY,
            "missing_group": config.missing_key,
            **({"source_weights": config.source_weights} if config.source_weights else {}),
        },
        "seed": seed,
        "outputs": {
            "train": str(train_dir / input_file.name),
            "dev": str(dev_dir / input_file.name),
            "test": str(test_dir / input_file.name),
            "in_process": str(inprocess_dir),
            "strata_counts": str(strata_counts_path),
        },
    }
    summary_path = inprocess_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def run_from_config(labels: list[str] | None = None, seed: int | None = None) -> list[dict[str, object]]:
    ratios = _get_ratios()
    split_seed = seed if seed is not None else get_split_seed()

    label_list = labels or list(LABEL_CONFIGS.keys())
    summaries: list[dict[str, object]] = []

    for label in label_list:
        if label not in LABEL_CONFIGS:
            raise SplitError(f"Unknown label: {label}")
        label_path = _get_label_path(label)
        summaries.append(split_label_dataset(label, label_path, ratios, split_seed))

    return summaries

import ast
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from data_helpers._config import (
    LABEL_CONFIGS_RAW as _LABEL_CONFIGS_RAW,
    MISSING_KEY,
    MULTI_LABEL_KEY as MULTI_KEY,
    N_SAMPLE,
    REPO_ROOT,
    SCREENING_CFG as _SCREENING_CFG,
    get_split_label_path,
    get_split_seed,
)

ROOT = REPO_ROOT

OUT_DIR = ROOT / _SCREENING_CFG["manual_sample_output_dir"]
STRAT_COL = _LABEL_CONFIGS_RAW["l0"]["strat_column"]


def _load_config_seed() -> int:
    return get_split_seed()


def _get_train_path() -> Path:
    return get_split_label_path("l0") / "train"


def _parse_labels(raw: str) -> list[str]:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(v).strip().strip("\"'") for v in parsed if str(v).strip().strip("\"'")]
    return [text.strip().strip("\"'")]


def _stratum(value: str) -> str:
    labels = _parse_labels(value)
    if not labels:
        return MISSING_KEY
    if len(labels) == 1:
        return labels[0].lower()
    return MULTI_KEY


def _allocation_counts(n: int, ratios: list[float]) -> list[int]:
    if n == 0:
        return [0] * len(ratios)
    raw = [n * r for r in ratios]
    base = [int(math.floor(v)) for v in raw]
    remainder = n - sum(base)
    fracs = [raw[i] - base[i] for i in range(len(raw))]
    order = sorted(range(len(fracs)), key=lambda i: fracs[i], reverse=True)
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    return base


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _col(header: list[str], name: str) -> int:
    return header.index(name)


def _cell(row: list[str], idx: int) -> str:
    return row[idx] if idx < len(row) else ""


def sample_l0(seed: int) -> dict:
    train_path = _get_train_path()
    csv_files = list(train_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {train_path}")
    src = csv_files[0]
    header, rows = _read_csv(src)

    strat_idx = _col(header, STRAT_COL)

    # Group into strata by source
    strata: dict[str, list[list[str]]] = {}
    for row in rows:
        key = _stratum(_cell(row, strat_idx))
        strata.setdefault(key, []).append(row)

    total = sum(len(v) for v in strata.values())
    n = min(N_SAMPLE, total)

    # Equal allocation across strata
    sorted_keys = sorted(strata.keys())
    n_strata = len(sorted_keys)
    ratios = [1 / n_strata for _ in sorted_keys]
    counts = _allocation_counts(n, ratios)
    # Cap each stratum at its available pool size, redistribute shortfall
    counts = [min(c, len(strata[k])) for c, k in zip(counts, sorted_keys)]
    shortfall = n - sum(counts)
    for i in range(len(sorted_keys)):
        if shortfall <= 0:
            break
        headroom = len(strata[sorted_keys[i]]) - counts[i]
        add = min(headroom, shortfall)
        counts[i] += add
        shortfall -= add

    rng = random.Random(seed)
    sampled: list[list[str]] = []
    strata_meta: dict[str, dict] = {}

    for key, count in zip(sorted_keys, counts):
        pool = list(strata[key])
        rng.shuffle(pool)
        picked = pool[:count]
        sampled.extend(picked)
        strata_meta[key] = {"total_in_train": len(pool), "sampled": count}

    return {
        "source_file": str(src),
        "train_rows": total,
        "sampled_rows": len(sampled),
        "header": header,
        "rows": sampled,
        "strata": strata_meta,
    }


def main():
    seed = _load_config_seed()
    print(f"Seed: {seed}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = sample_l0(seed)

    out_csv = OUT_DIR / "l0_manual_sample.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(result["header"])
        writer.writerows(result["rows"])

    print("\n=== L0 ===")
    print(f"  train rows : {result['train_rows']}")
    print(f"  sampled    : {result['sampled_rows']}")
    print("  strata breakdown (by source):")
    for stratum, info in sorted(result["strata"].items()):
        print(f"    {stratum}: {info['sampled']} / {info['total_in_train']}")
    print(f"  saved → {out_csv}")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_sample": N_SAMPLE,
        "label": "l0",
        "source_file": result["source_file"],
        "train_rows": result["train_rows"],
        "sampled_rows": result["sampled_rows"],
        "output_file": str(out_csv),
        "strata": result["strata"],
    }
    meta_path = OUT_DIR / "sample_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nMeta saved → {meta_path}")


if __name__ == "__main__":
    main()

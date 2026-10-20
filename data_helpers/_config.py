"""Loads dataset configuration from checklists/mappings/repo_config.json."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = REPO_ROOT / "checklists" / "mappings" / "repo_config.json"

with _CONFIG_PATH.open(encoding="utf-8") as _f:
    _repo_cfg: dict = json.load(_f)

_cfg = _repo_cfg.get("dataset_config")
if not isinstance(_cfg, dict):
    raise ValueError(
        f"Required config section dataset_config is missing in {_CONFIG_PATH}."
    )

# Special keys — shared across splitter and samplers
MISSING_KEY: str = _cfg["special_keys"]["missing"]
MULTI_LABEL_KEY: str = _cfg["special_keys"]["multi_label"]

# Column names — shared across creator scripts
BASE_COLS: list[str] = _cfg["base_columns"]

# Sampling
N_SAMPLE: int = _cfg["sampling"]["n_sample"]

# Train/dev/test split settings
SPLIT_CFG: dict = _cfg["splits"]

# Raw label configs — used by splitter and samplers to rebuild their own formats
LABEL_CONFIGS_RAW: dict = _cfg["label_configs"]

# Dataset-specific config sections
SCREENING_CFG: dict = _cfg["screening_dataset"]
CODING_CFG: dict = _cfg["coding_dataset"]
GBIF_CFG: dict = _cfg["gbif"]


def _require_section(config: dict, name: str, path: str) -> dict:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(
            f"Required config section dataset_config.{path}.{name} is missing in {_CONFIG_PATH}."
        )
    return value


def _require_int(config: dict, name: str, path: str) -> int:
    value = config.get(name)
    if not isinstance(value, int):
        raise ValueError(
            f"Required integer config key dataset_config.{path}.{name} is missing in {_CONFIG_PATH}."
        )
    return value


def _require_string(config: dict, name: str, path: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Required string config key dataset_config.{path}.{name} is missing in {_CONFIG_PATH}."
        )
    return value.strip()


def _require_string_list(config: dict, name: str, path: str) -> list[str]:
    value = config.get(name)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(
            f"Required non-empty string list dataset_config.{path}.{name} is missing in {_CONFIG_PATH}."
        )
    cleaned = [item.strip() for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(
            f"Config key dataset_config.{path}.{name} contains duplicates in {_CONFIG_PATH}."
        )
    return cleaned


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def get_merged_corpus_config() -> dict[str, object]:
    """Load and validate the merged-corpus configuration on demand."""
    config = _cfg.get("merged_corpus")
    if not isinstance(config, dict):
        raise ValueError(
            f"Required config section dataset_config.merged_corpus is missing in {_CONFIG_PATH}."
        )

    path = "merged_corpus"
    key_column = _require_string(config, "key_column", path)
    chunksize = _require_int(config, "chunksize", path)
    if chunksize <= 0:
        raise ValueError(
            f"Config key dataset_config.{path}.chunksize must be > 0 in {_CONFIG_PATH}."
        )

    screening = _require_section(config, "screening", path)
    screening_path = _resolve_repo_path(
        _require_string(screening, "path", f"{path}.screening")
    )
    eligibility_column = _require_string(
        screening, "eligibility_column", f"{path}.screening"
    )
    screening_columns = _require_string_list(screening, "columns", f"{path}.screening")
    if key_column not in screening_columns:
        raise ValueError(
            f"Config key dataset_config.{path}.screening.columns must include {key_column!r} in {_CONFIG_PATH}."
        )
    if eligibility_column in screening_columns:
        raise ValueError(
            f"Config key dataset_config.{path}.screening.columns must not include the filtering-only column "
            f"{eligibility_column!r} in {_CONFIG_PATH}."
        )

    raw_sources = config.get("coding_sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(
            f"Required non-empty list dataset_config.{path}.coding_sources is missing in {_CONFIG_PATH}."
        )

    coding_sources: list[dict[str, object]] = []
    names: set[str] = set()
    output_columns = set(screening_columns)
    for index, raw_source in enumerate(raw_sources):
        source_path = f"{path}.coding_sources[{index}]"
        if not isinstance(raw_source, dict):
            raise ValueError(
                f"Config section dataset_config.{source_path} must be an object in {_CONFIG_PATH}."
            )
        name = _require_string(raw_source, "name", source_path)
        if name in names:
            raise ValueError(
                f"Duplicate coding source name {name!r} in {_CONFIG_PATH}."
            )
        names.add(name)
        columns = _require_string_list(raw_source, "columns", source_path)
        if key_column in columns:
            raise ValueError(
                f"Config key dataset_config.{source_path}.columns must not repeat key column {key_column!r}."
            )
        overlap = output_columns.intersection(columns)
        if overlap:
            raise ValueError(
                f"Coding source {name!r} repeats output column(s) {sorted(overlap)} in {_CONFIG_PATH}."
            )
        output_columns.update(columns)
        coding_sources.append(
            {
                "name": name,
                "path": _resolve_repo_path(
                    _require_string(raw_source, "path", source_path)
                ),
                "columns": columns,
            }
        )

    return {
        "key_column": key_column,
        "chunksize": chunksize,
        "screening": {
            "path": screening_path,
            "eligibility_column": eligibility_column,
            "columns": screening_columns,
        },
        "coding_sources": coding_sources,
    }


def get_split_ratios() -> dict[str, float]:
    ratios = _require_section(SPLIT_CFG, "ratios", "splits")
    raw = {
        split: _require_int(ratios, split, "splits.ratios")
        for split in ("train", "dev", "test")
    }
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(
            f"Config key dataset_config.splits.ratios must sum to > 0 in {_CONFIG_PATH}."
        )
    return {split: raw[split] / total for split in raw}


def get_split_seed() -> int:
    return _require_int(SPLIT_CFG, "seed", "splits")


def get_split_label_path(label: str) -> Path:
    label_paths = _require_section(SPLIT_CFG, "label_paths", "splits")
    raw = _require_string(label_paths, label, "splits.label_paths")
    return _resolve_repo_path(raw)

"""Build the analysis corpus by joining screening and coding outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype

from data_helpers._config import get_merged_corpus_config


class CorpusMergeError(ValueError):
    """Raised when corpus inputs do not satisfy the configured merge contract."""


def _eligible_mask(series: pd.Series) -> pd.Series:
    if is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def _read_csv(path: Path, usecols: list[str], **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False, **kwargs)
    except ValueError as exc:
        raise CorpusMergeError(
            f"Could not read required columns {usecols} from {path}: {exc}"
        ) from exc


def _validate_key(frame: pd.DataFrame, key_column: str, source: str) -> None:
    missing = int(frame[key_column].isna().sum())
    if missing:
        raise CorpusMergeError(
            f"{source} has {missing:,} row(s) with a missing {key_column} value."
        )

    duplicates = frame.loc[frame[key_column].duplicated(keep=False), key_column]
    if not duplicates.empty:
        examples = duplicates.drop_duplicates().head(10).tolist()
        raise CorpusMergeError(
            f"{source} is not one row per {key_column}; found {duplicates.nunique():,} duplicated value(s). "
            f"Examples: {examples}"
        )


def _read_eligible_screening(
    path: Path,
    columns: list[str],
    eligibility_column: str,
    key_column: str,
    chunksize: int,
) -> pd.DataFrame:
    usecols = [*columns, eligibility_column]
    chunks: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            path, usecols=usecols, chunksize=chunksize, low_memory=False
        )
        for chunk in reader:
            chunks.append(chunk.loc[_eligible_mask(chunk[eligibility_column]), columns])
    except ValueError as exc:
        raise CorpusMergeError(
            f"Could not read required columns {usecols} from {path}: {exc}"
        ) from exc

    eligible = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=columns)
    )
    _validate_key(eligible, key_column, f"Eligible screening data at {path}")
    return eligible


def _join_one_to_one(
    left: pd.DataFrame,
    path: Path,
    value_columns: list[str],
    key_column: str,
    source_name: str,
) -> pd.DataFrame:
    overlap = set(left.columns).intersection(value_columns)
    if overlap:
        raise CorpusMergeError(
            f"Coding source {source_name!r} repeats output column(s): {sorted(overlap)}"
        )

    right = _read_csv(path, [key_column, *value_columns])
    source = f"Coding source {source_name!r} at {path}"
    _validate_key(right, key_column, source)

    left_keys = pd.Index(left[key_column])
    right_keys = pd.Index(right[key_column])
    left_only = left_keys.difference(right_keys, sort=False)
    right_only = right_keys.difference(left_keys, sort=False)
    if len(left_only) or len(right_only):
        both = len(left_keys) - len(left_only)
        raise CorpusMergeError(
            f"{source} does not exactly match the current {key_column} grain. "
            f"Counts: {{'both': {both}, 'left_only': {len(left_only)}, 'right_only': {len(right_only)}}}; "
            f"left-only examples: {left_only[:10].tolist()}; right-only examples: {right_only[:10].tolist()}"
        )

    original_keys = left[key_column].reset_index(drop=True)
    merged = left.merge(
        right, on=key_column, how="left", sort=False, validate="one_to_one"
    )
    if len(merged) != len(left):
        raise CorpusMergeError(
            f"Joining {source} changed the row count from {len(left):,} to {len(merged):,}."
        )
    if not merged[key_column].reset_index(drop=True).equals(original_keys):
        raise CorpusMergeError(f"Joining {source} changed the {key_column} row order.")
    return merged


def build_merged_corpus() -> pd.DataFrame:
    """Return the configured eligible screening corpus with coding columns joined."""
    config = get_merged_corpus_config()
    key_column = str(config["key_column"])
    screening = config["screening"]
    if not isinstance(
        screening, dict
    ):  # Defensive: the config loader normally guarantees this.
        raise CorpusMergeError(
            "Merged-corpus screening configuration must be an object."
        )

    corpus = _read_eligible_screening(
        path=Path(screening["path"]),
        columns=list(screening["columns"]),
        eligibility_column=str(screening["eligibility_column"]),
        key_column=key_column,
        chunksize=int(config["chunksize"]),
    )

    coding_sources = config["coding_sources"]
    if not isinstance(
        coding_sources, list
    ):  # Defensive: the config loader normally guarantees this.
        raise CorpusMergeError(
            "Merged-corpus coding_sources configuration must be a list."
        )
    for source in coding_sources:
        if not isinstance(source, dict):
            raise CorpusMergeError(
                "Each merged-corpus coding source must be an object."
            )
        corpus = _join_one_to_one(
            corpus,
            path=Path(source["path"]),
            value_columns=list(source["columns"]),
            key_column=key_column,
            source_name=str(source["name"]),
        )
    return corpus


def build_merged_corpus_from_eligible_screening(
    eligible_screening: pd.DataFrame,
) -> pd.DataFrame:
    """Join configured coding outputs onto a prepared eligible screening table.

    This is the data-processing counterpart to :func:`build_merged_corpus`.
    It avoids rescanning the multi-gigabyte full-screening CSV when an audited
    one-row-per-UT eligible screening artifact already exists.
    """
    config = get_merged_corpus_config()
    key_column = str(config["key_column"])
    screening = config["screening"]
    if not isinstance(screening, dict):
        raise CorpusMergeError(
            "Merged-corpus screening configuration must be an object."
        )
    screening_columns = list(screening["columns"])
    missing = set(screening_columns).difference(eligible_screening.columns)
    if missing:
        raise CorpusMergeError(
            "Prepared eligible screening data lacks configured columns: "
            f"{sorted(missing)}"
        )
    corpus = eligible_screening[screening_columns].copy()
    _validate_key(corpus, key_column, "Prepared eligible screening data")

    coding_sources = config["coding_sources"]
    if not isinstance(coding_sources, list):
        raise CorpusMergeError(
            "Merged-corpus coding_sources configuration must be a list."
        )
    for source in coding_sources:
        if not isinstance(source, dict):
            raise CorpusMergeError(
                "Each merged-corpus coding source must be an object."
            )
        corpus = _join_one_to_one(
            corpus,
            path=Path(source["path"]),
            value_columns=list(source["columns"]),
            key_column=key_column,
            source_name=str(source["name"]),
        )
    return corpus


__all__ = [
    "CorpusMergeError",
    "build_merged_corpus",
    "build_merged_corpus_from_eligible_screening",
]

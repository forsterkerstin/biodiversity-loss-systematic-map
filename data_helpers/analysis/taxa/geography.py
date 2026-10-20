"""Publication-country assignments for the taxonomic analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TaxaGeographyError(ValueError):
    """Raised when the geographic join cannot be built as specified."""


def load_publication_countries(
    source: str | Path,
    *,
    id_col: str = "UT",
    country_col: str = "pred_countries",
) -> pd.DataFrame:
    """Read one row per publication with its coded study countries.

    Reads the *data-processing* corpus handoff rather than any results notebook's
    output, so this analysis stays a sibling of ``taxa_analysis_prep`` rather than
    a dependant of Result 02.

    Parquet returns list columns as numpy arrays, and ``parse_list_labels`` raises
    on an empty array (``pd.isna`` is ambiguous there), so values are normalised to
    plain lists on the way out.
    """
    source = Path(source)
    if not source.exists():
        raise TaxaGeographyError(f"Country source not found: {source}")
    frame = pd.read_parquet(source, columns=[id_col, country_col])

    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    frame[country_col] = frame[country_col].map(_as_list)
    duplicated = int(frame[id_col].duplicated().sum())
    if duplicated:
        raise TaxaGeographyError(
            f"Country source is not one row per {id_col}: {duplicated:,} duplicates."
        )
    return frame


__all__ = [
    "TaxaGeographyError",
    "load_publication_countries",
]

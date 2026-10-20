"""
Build and cache the GBIF canonical-name lookup dictionary.

Usage:
    python -m data_helpers.sources.build_gbif
    # or directly:
    python data_helpers/sources/build_gbif.py

Reads:  data/gbif/curated/gbif_curated.csv
Writes: checklists/mappings/gbif_lookup_cache.pkl
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

from data_helpers._config import GBIF_CFG as _GBIF_CFG

REPO_ROOT   = Path(__file__).resolve().parents[2]
GBIF_PATH   = REPO_ROOT / _GBIF_CFG["source"]
CACHE_PATH  = REPO_ROOT / _GBIF_CFG["cache"]
CHUNKSIZE   = _GBIF_CFG["chunksize"]
TAX_COLS    = _GBIF_CFG["taxonomic_columns"]
NEEDED_COLS = ["canonicalName", "taxonomicStatus"] + TAX_COLS


def build_lookup(gbif_path: Path, chunksize: int = CHUNKSIZE) -> dict:
    """
    Returns {canonical_name_lower: {kingdom, phylum, class, order, genus}}.
    Accepted-status rows take priority over others.
    """
    lookup_accepted: dict = {}
    lookup_fallback: dict = {}

    reader = pd.read_csv(
        gbif_path,
        usecols=lambda c: c in NEEDED_COLS,
        dtype=str,
        chunksize=chunksize,
    )

    for i, chunk in enumerate(reader):
        if i % 10 == 0:
            print(f"  processed {i * chunksize:,} rows …", end="\r", flush=True)

        chunk = chunk.dropna(subset=["canonicalName"]).copy()
        chunk["_key"] = chunk["canonicalName"].str.strip().str.lower()
        chunk[TAX_COLS] = chunk[TAX_COLS].replace("nan", pd.NA)
        chunk = chunk.dropna(subset=TAX_COLS, how="all")
        if chunk.empty:
            continue

        is_accepted = chunk["taxonomicStatus"].str.lower().str.contains("accepted", na=False)

        for sub, target in [
            (chunk[is_accepted], lookup_accepted),
            (chunk[~is_accepted], lookup_fallback),
        ]:
            if sub.empty:
                continue
            records = (
                sub.drop_duplicates(subset="_key", keep="first")[["_key"] + TAX_COLS]
                .to_dict("records")
            )
            for row in records:
                key = row["_key"]
                if key not in target:
                    target[key] = {
                        c: (None if pd.isna(row[c]) else row[c]) for c in TAX_COLS
                    }

    print(
        f"\nLookup built — accepted: {len(lookup_accepted):,}  "
        f"fallback: {len(lookup_fallback):,}"
    )
    return {**lookup_fallback, **lookup_accepted}


def main() -> None:
    if not GBIF_PATH.exists():
        print(f"ERROR: GBIF file not found at {GBIF_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Building GBIF lookup from:\n  {GBIF_PATH}")
    lookup = build_lookup(GBIF_PATH)
    print(f"Total unique canonical names: {len(lookup):,}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved → {CACHE_PATH}")


if __name__ == "__main__":
    main()

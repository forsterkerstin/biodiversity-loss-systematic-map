from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import DATASETS_DIR, DEFAULT_COLUMNS


def split_semicolon(value) -> List[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def find_column(df: pd.DataFrame, *variants: str) -> str | None:
    """Find a column by trying multiple name variants (case-insensitive)."""
    for variant in variants:
        if variant in df.columns:
            return variant
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == variant.lower():
                return col
    return None


def load_wos_excels(input_dir: Path, columns: Iterable[str] = DEFAULT_COLUMNS, limit: int | None = None) -> pd.DataFrame:
    """Read every .xls, .xlsx, or .csv in a folder and return a combined dataframe."""
    paths = sorted(
        [path for pattern in ("*.xls", "*.xlsx", "*.csv") for path in Path(input_dir).glob(pattern)]
    )
    if not paths:
        raise FileNotFoundError(f"No .xls, .xlsx, or .csv files found in {input_dir}")

    frames = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Select columns flexibly - include both requested columns and any found variants
    selected_cols = []
    for c in columns:
        if c in combined.columns:
            selected_cols.append(c)

    # Also include common variants if they exist
    common_variants = {
        "article_title": "Article Title",
        "abstract": "Abstract",
        "title": "Article Title",
        "authors": "Authors",
        "publisher": "Publisher",
        "publication_year": "Publication Year",
        "year": "Publication Year",
        "wos_categories": "WoS Categories",
        "doi": "DOI",
        "ut": "UT",
        "custom_id": "UT",
    }
    for variant in combined.columns:
        variant_lower = variant.lower()
        if variant_lower in common_variants and variant not in selected_cols:
            selected_cols.append(variant)

    subset = combined[selected_cols].copy() if selected_cols else combined.copy()
    subset.rename(columns={"UT (Unique WOS ID)": "UT", "custom_id": "UT"}, inplace=True)

    # Find UT column flexibly
    ut_col = find_column(subset, "UT", "ut", "UT (Unique WOS ID)", "custom_id")
    if not ut_col:
        raise KeyError("Document ID column not found. Expected one of: UT, ut, UT (Unique WOS ID), custom_id.")

    if ut_col != "UT":
        subset.rename(columns={ut_col: "UT"}, inplace=True)

    subset["UT"] = subset["UT"].astype(str).str.strip()
    subset.drop_duplicates(subset=["UT"], inplace=True)
    if limit is not None:
        subset = subset.head(limit)
    return subset


def dataframe_to_documents(df: pd.DataFrame) -> list[dict]:
    # Find column names with flexible matching
    title_col = find_column(df, "Article Title", "article_title", "Title", "title")
    abstract_col = find_column(df, "Abstract", "abstract")
    authors_col = find_column(df, "Authors", "authors", "Author")
    publisher_col = find_column(df, "Publisher", "publisher", "Source", "source")
    pub_year_col = find_column(df, "Publication Year", "publication_year", "Year", "year")
    wos_cat_col = find_column(df, "WoS Categories", "wos_categories", "Categories")
    doi_col = find_column(df, "DOI", "doi")
    ut_col = find_column(df, "UT", "ut", "UT (Unique WOS ID)", "custom_id")

    documents = []
    for _, row in df.iterrows():
        # Extract values using flexible column names
        ut_val = row.get(ut_col) if ut_col else ""
        title_val = row.get(title_col) if title_col else ""
        abstract_val = row.get(abstract_col) if abstract_col else ""
        authors_val = row.get(authors_col) if authors_col else ""
        publisher_val = row.get(publisher_col) if publisher_col else ""
        pub_year_val = row.get(pub_year_col) if pub_year_col else ""
        wos_cat_val = row.get(wos_cat_col) if wos_cat_col else ""
        doi_val = row.get(doi_col) if doi_col else ""

        documents.append(
            {
                "UT": str(ut_val).strip() if ut_val and not pd.isna(ut_val) else "",
                "title": "" if pd.isna(title_val) else str(title_val).strip(),
                "authors": split_semicolon(authors_val),
                "abstract": "" if pd.isna(abstract_val) else str(abstract_val).strip(),
                "source": "" if pd.isna(publisher_val) else str(publisher_val).strip(),
                "publication_year": "" if pd.isna(pub_year_val) else str(pub_year_val).strip(),
                "wos_categories": split_semicolon(wos_cat_val),
                "doi": "" if pd.isna(doi_val) else str(doi_val).strip(),
            }
        )
    return documents


def save_dataset(documents: list[dict], path: Path | None = None) -> Path:
    """Persist documents to JSON for downstream batch creation."""
    payload = {"documents": {doc["UT"]: doc for doc in documents}}
    path = path or DATASETS_DIR / "dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

"""Search comprehensiveness: recall of the a priori test list (Supplementary Table S3).

The test list (``data/consistency-check-datasets/reviews-and-test-list-reproduced/
Test List.xlsx``) holds the WoS-indexed records cited by the benchmark sources.
Recall is the share of those records present in the retrieved search results,
overall (unique records) and per source.

Records are matched on the Web of Science accession number (UT). The source of
each test-list record is read from a ``source`` column or, for a workbook with
one sheet per source, from the sheet name (a trailing record count such as
" (105)" is dropped). The union sheet and the sheet of retrieved records are not
sources (``EXCLUDED_SHEETS``); sheets without an identifier column (overview,
query) are skipped.

Two retrieved sets are useful:

* the records the comprehensiveness check returned in Web of Science, stored in
  the test-list workbook (sheet "Retreival Results"). This reproduces the
  reported numbers. The check query had no publication-year filter, so it also
  returned test-list records published before 2000;
* the screening handoff (2,414,191 records, the default), i.e. the search as
  run with its 2000-2026 window. It excludes the 12,617 records without an
  abstract. ``--start-year 2000`` restricts the test list to the same window.

    python -m data_helpers.datasets.search_recall --start-year 2000
    python -m data_helpers.datasets.search_recall --retrieved "<Test List.xlsx>" --retrieved-sheet "Retreival Results"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_LIST = (
    ROOT / "data" / "consistency-check-datasets" / "reviews-and-test-list-reproduced" / "Test List.xlsx"
)
DEFAULT_RETRIEVED = (
    ROOT / "notebooks" / "data_processing" / "outputs" / "03_screening_analysis_prep" / "screening_publications.parquet"
)
ID_COLUMN_CANDIDATES = ("UT (Unique WOS ID)", "UT", "ut", "WOS ID", "Accession Number")
SOURCE_COLUMN_CANDIDATES = ("source", "Source", "review", "Review", "source_review")
YEAR_COLUMN_CANDIDATES = ("Publication Year", "publication_year", "PY")
# Union of all sources and the check's retrieved records, as named in the workbook.
EXCLUDED_SHEETS = ("Test List (581)", "Retreival Results")
_UT_PATTERN = re.compile(r"^(?:WOS:)?([A-Z0-9]+)$")
_SHEET_COUNT_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


class SearchRecallError(ValueError):
    """Raised when an input cannot be read or lacks the required columns."""


def normalize_ut(value: object) -> str | None:
    """Return a WoS accession number as ``WOS:<id>``, or None if absent."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper().replace(" ", "")
    if not text or text in {"NAN", "NONE"}:
        return None
    match = _UT_PATTERN.match(text)
    if match is None:
        raise SearchRecallError(f"Not a Web of Science accession number: {value!r}")
    return f"WOS:{match.group(1)}"


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def load_test_list(
    path: Path = DEFAULT_TEST_LIST,
    *,
    id_column: str | None = None,
    exclude_sheets: tuple[str, ...] = EXCLUDED_SHEETS,
) -> pd.DataFrame:
    """Return one row per (source, UT), with the publication year when available."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = []
    for sheet_name, frame in sheets.items():
        if sheet_name in exclude_sheets:
            continue
        ut_column = id_column if id_column in frame.columns else _first_present(
            frame.columns, ID_COLUMN_CANDIDATES
        )
        if ut_column is None:
            print(f"  skipped sheet {sheet_name!r}: no identifier column")
            continue
        source_column = _first_present(frame.columns, SOURCE_COLUMN_CANDIDATES)
        source = (
            frame[source_column].astype(str).str.strip() if source_column
            else _SHEET_COUNT_SUFFIX.sub("", sheet_name)
        )
        year_column = _first_present(frame.columns, YEAR_COLUMN_CANDIDATES)
        year = (
            pd.to_numeric(frame[year_column], errors="coerce") if year_column
            else pd.Series(float("nan"), index=frame.index)
        )
        frames.append(
            pd.DataFrame(
                {"source": source, "UT": frame[ut_column].map(normalize_ut), "publication_year": year}
            )
        )
    if not frames:
        raise SearchRecallError(
            f"No sheet in {path} has an identifier column; tried {ID_COLUMN_CANDIDATES}. "
            "Pass --id-column."
        )
    test_list = pd.concat(frames, ignore_index=True)
    missing = int(test_list["UT"].isna().sum())
    if missing:
        print(f"  {missing} test-list rows without an accession number were dropped")
    return test_list.dropna(subset=["UT"]).drop_duplicates(["source", "UT"], ignore_index=True)


def load_retrieved_uts(
    path: Path = DEFAULT_RETRIEVED,
    *,
    column: str | None = None,
    sheet: str | None = None,
) -> set[str]:
    """Return the normalized accession numbers of the retrieved records."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype=str, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
    else:
        raise SearchRecallError(f"Unsupported retrieved-set format: {path.suffix}")
    ut_column = column or _first_present(frame.columns, ID_COLUMN_CANDIDATES)
    if ut_column is None or ut_column not in frame.columns:
        raise SearchRecallError(f"No identifier column in {path}; pass --retrieved-column.")
    return {ut for ut in frame[ut_column].map(normalize_ut) if ut is not None}


def search_recall(
    test_list: pd.DataFrame,
    retrieved: set[str],
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-source and overall recall of the test list against the retrieved set.

    ``start_year``/``end_year`` restrict the test list to records published in
    the search window; records without a year are then excluded as well.
    """
    if start_year is not None or end_year is not None:
        years = test_list["publication_year"]
        keep = years.notna()
        if start_year is not None:
            keep &= years >= start_year
        if end_year is not None:
            keep &= years <= end_year
        test_list = test_list.loc[keep]
    flagged = test_list.assign(retrieved=test_list["UT"].isin(retrieved))
    per_source = (
        flagged.groupby("source", sort=True)
        .agg(test_list_n=("UT", "size"), retrieved_n=("retrieved", "sum"))
        .reset_index()
    )
    per_source["recall_pct"] = 100 * per_source["retrieved_n"] / per_source["test_list_n"]
    unique = flagged.drop_duplicates("UT")
    overall = {
        "test_list_n": int(len(unique)),
        "retrieved_n": int(unique["retrieved"].sum()),
        "recall_pct": 100 * float(unique["retrieved"].mean()) if len(unique) else float("nan"),
        "records_in_multiple_sources": int(flagged.groupby("UT")["source"].nunique().gt(1).sum()),
    }
    return per_source, overall


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-list", type=Path, default=DEFAULT_TEST_LIST)
    parser.add_argument("--id-column", help="Identifier column in the test list (default: auto-detect).")
    parser.add_argument("--retrieved", type=Path, default=DEFAULT_RETRIEVED)
    parser.add_argument("--retrieved-column", help="Identifier column in the retrieved set.")
    parser.add_argument("--retrieved-sheet", help="Worksheet holding the retrieved set (Excel only).")
    parser.add_argument("--start-year", type=int, help="Keep test-list records published from this year.")
    parser.add_argument("--end-year", type=int, help="Keep test-list records published up to this year.")
    parser.add_argument("--out", type=Path, help="Directory for search_recall_by_source.csv.")
    args = parser.parse_args(argv)

    try:
        test_list = load_test_list(args.test_list, id_column=args.id_column)
        retrieved = load_retrieved_uts(
            args.retrieved, column=args.retrieved_column, sheet=args.retrieved_sheet
        )
    except SearchRecallError as exc:
        raise SystemExit(f"[search_recall] {exc}") from None
    per_source, overall = search_recall(
        test_list, retrieved, start_year=args.start_year, end_year=args.end_year
    )

    print(f"Retrieved set: {args.retrieved} ({len(retrieved):,} records)")
    if args.retrieved.resolve() == DEFAULT_RETRIEVED.resolve():
        print("  note: excludes records without an abstract; see the module docstring.")
    if args.start_year is not None or args.end_year is not None:
        print(f"Test list restricted to publication years {args.start_year or '...'}-{args.end_year or '...'}")
    print(per_source.round({"recall_pct": 2}).to_string(index=False))
    print(
        f"Overall: {overall['retrieved_n']} of {overall['test_list_n']} unique test-list records "
        f"retrieved ({overall['recall_pct']:.2f}%); "
        f"{overall['records_in_multiple_sources']} records are cited by more than one source."
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "search_recall_by_source.csv"
        per_source.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

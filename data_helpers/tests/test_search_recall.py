"""Search comprehensiveness against the a priori test list."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_helpers.datasets import search_recall


def test_normalize_ut_accepts_prefixed_and_bare_accession_numbers() -> None:
    assert search_recall.normalize_ut(" wos:000276003400008 ") == "WOS:000276003400008"
    assert search_recall.normalize_ut("A1997XP84700006") == "WOS:A1997XP84700006"
    assert search_recall.normalize_ut(float("nan")) is None
    assert search_recall.normalize_ut("") is None
    with pytest.raises(search_recall.SearchRecallError):
        search_recall.normalize_ut("10.1111/gcb.14606")


def test_recall_is_reported_per_source_and_over_unique_records() -> None:
    test_list = pd.DataFrame(
        {
            "source": ["Urban", "Urban", "Urban", "Beckmann", "Beckmann"],
            "UT": ["WOS:1", "WOS:2", "WOS:3", "WOS:3", "WOS:4"],
        }
    )
    per_source, overall = search_recall.search_recall(test_list, {"WOS:1", "WOS:3"})
    table = per_source.set_index("source")
    assert table.loc["Urban", ["test_list_n", "retrieved_n"]].tolist() == [3, 2]
    assert table.loc["Beckmann", "recall_pct"] == pytest.approx(50.0)
    # WOS:3 is cited by both sources but counted once overall.
    assert overall["test_list_n"] == 4
    assert overall["retrieved_n"] == 2
    assert overall["recall_pct"] == pytest.approx(50.0)
    assert overall["records_in_multiple_sources"] == 1


def test_sheet_per_source_workbook_is_read_and_summary_sheets_skipped(tmp_path: Path) -> None:
    path = tmp_path / "Test List.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["readme"]}).to_excel(writer, sheet_name="README", index=False)
        pd.DataFrame({"UT (Unique WOS ID)": ["WOS:1", "2", None]}).to_excel(
            writer, sheet_name="Urban", index=False
        )
        pd.DataFrame({"UT (Unique WOS ID)": ["WOS:3"]}).to_excel(
            writer, sheet_name="Beckmann", index=False
        )
    test_list = search_recall.load_test_list(path)
    assert sorted(map(tuple, test_list[["source", "UT"]].to_numpy())) == [
        ("Beckmann", "WOS:3"), ("Urban", "WOS:1"), ("Urban", "WOS:2"),
    ]


def test_source_column_takes_precedence_over_sheet_name(tmp_path: Path) -> None:
    path = tmp_path / "Test List.xlsx"
    pd.DataFrame({"UT": ["WOS:1", "WOS:2"], "source": ["Urban", "Moullec"]}).to_excel(
        path, index=False
    )
    test_list = search_recall.load_test_list(path)
    assert set(test_list["source"]) == {"Urban", "Moullec"}


def test_retrieved_set_is_read_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "retrieved.csv"
    pd.DataFrame({"UT": ["WOS:1", "wos:2", None]}).to_csv(path, index=False)
    assert search_recall.load_retrieved_uts(path) == {"WOS:1", "WOS:2"}


def test_union_and_results_sheets_are_not_sources_and_counts_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "Test List.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, uts in (
            ("Test List (581)", ["WOS:1", "WOS:2"]),
            ("Urban Et al (2)", ["WOS:1", "WOS:2"]),
            ("Retreival Results", ["WOS:1"]),
        ):
            pd.DataFrame({"UT (Unique WOS ID)": uts, "Publication Year": [1999, 2005][: len(uts)]}).to_excel(
                writer, sheet_name=sheet, index=False
            )
    test_list = search_recall.load_test_list(path)
    assert set(test_list["source"]) == {"Urban Et al"}
    retrieved = search_recall.load_retrieved_uts(path, sheet="Retreival Results")
    assert retrieved == {"WOS:1"}


def test_year_window_restricts_the_test_list() -> None:
    test_list = pd.DataFrame(
        {
            "source": ["Urban"] * 4,
            "UT": ["WOS:1", "WOS:2", "WOS:3", "WOS:4"],
            "publication_year": [1995, 2001, 2010, float("nan")],
        }
    )
    _, everything = search_recall.search_recall(test_list, {"WOS:1", "WOS:2"})
    _, in_window = search_recall.search_recall(test_list, {"WOS:1", "WOS:2"}, start_year=2000)
    assert (everything["retrieved_n"], everything["test_list_n"]) == (2, 4)
    assert (in_window["retrieved_n"], in_window["test_list_n"]) == (1, 2)

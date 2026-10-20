"""Tests for the regional evidence trend against a threatened-species benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_helpers.analysis.taxa import threat_gap as tg


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included = pd.DataFrame(
        {
            "UT": ["a", "b", "c", "d"],
            "benchmark_groups": [("Chordates",), ("Chordates",), ("Chordates",), ("Plants",)],
            "publication_year": [2000, 2000, 2001, 2000],
        }
    )
    countries = pd.DataFrame(
        {
            "UT": ["a", "b", "c", "d"],
            # a: two African countries (one Africa occurrence); b: Africa and Asia.
            "pred_countries": [["KEN", "TZA"], ["KEN", "CHN"], ["CHN"], ["KEN"]],
        }
    )
    crosswalk = pd.DataFrame(
        {
            "iso3": ["KEN", "TZA", "CHN"],
            "region": ["Africa", "Africa", "Asia and the Pacific"],
        }
    )
    regional = pd.DataFrame(
        {"region": ["Africa", "Asia and the Pacific"], "threat_share_pct": [20.0, 80.0]}
    )
    return included, countries, regional, crosswalk


def test_trend_counts_each_publication_once_per_region_and_year() -> None:
    trend = tg.regional_evidence_trend(*_inputs(), group="Chordates").set_index(
        ["publication_year", "region"]
    )
    # 2000: a and b in Africa (2), b in Asia (1); the Plants record d is excluded.
    assert trend.loc[(2000, "Africa"), "n_publications"] == 2
    assert trend.loc[(2000, "Asia and the Pacific"), "n_publications"] == 1
    assert trend.loc[(2000, "Africa"), "evidence_share_pct"] == pytest.approx(200 / 3)
    assert trend.loc[(2000, "Africa"), "evidence_to_threat_ratio"] == pytest.approx(10 / 3)
    # 2001 has no African evidence; the year-region grid is filled with zero.
    assert trend.loc[(2001, "Africa"), "n_publications"] == 0
    assert trend.loc[(2001, "Asia and the Pacific"), "evidence_share_pct"] == pytest.approx(100.0)


def test_trend_rejects_an_absent_group() -> None:
    with pytest.raises(tg.TaxaThreatGapError, match="Fungi"):
        tg.regional_evidence_trend(*_inputs(), group="Fungi")


def test_summary_reports_endpoints_slope_and_ratios() -> None:
    trend = pd.DataFrame(
        {
            "region": ["Africa"] * 3,
            "publication_year": [2000, 2005, 2010],
            "evidence_share_pct": [10.0, 15.0, 20.0],
            "threat_share_pct": [20.0] * 3,
        }
    )
    row = tg.summarise_regional_trend(trend).iloc[0]
    assert (row["start_year"], row["end_year"]) == (2000, 2010)
    assert row["change_pp"] == pytest.approx(10.0)
    assert row["slope_pp_per_decade"] == pytest.approx(10.0)
    assert (row["start_ratio"], row["end_ratio"]) == (pytest.approx(0.5), pytest.approx(1.0))
    assert not np.isnan(row["threat_share_pct"])

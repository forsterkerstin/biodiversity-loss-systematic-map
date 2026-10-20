"""Tests for publication-level taxonomic skew calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_helpers.analysis.taxa import skew as ts

GROUP_ORDER = (
    "Chordates",
    "Non-chordate animals",
    "Plants",
    "Fungi",
    "Other",
    "Unresolved",
)
BENCHMARK_GROUPS = GROUP_ORDER[:-1]


def _prepared_articles() -> pd.DataFrame:
    """Four publications spanning both directions and the year boundary."""
    rows = [
        ("A", 2010, "negative", ("Chordates",)),
        ("B", 2010, "positive", ("Non-chordate animals",)),
        ("C", 1998, "negative", ("Plants",)),
        ("D", 2026, "negative", ("Fungi",)),
    ]
    return pd.DataFrame(
        {
            "UT": [r[0] for r in rows],
            "publication_year": [r[1] for r in rows],
            "s2_dir": [r[2] for r in rows],
            "broad_groups": [r[3] for r in rows],
            "broad_groups_all": [r[3] for r in rows],
            "taxa_broad_inclusion_state": [ts.INCLUSION_ORDER[0]] * len(rows),
        }
    )


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame()


def _prepare(articles: pd.DataFrame, **scope):
    return ts.prepare_taxa_skew_from_prepared(
        articles,
        benchmark_groups=BENCHMARK_GROUPS,
        unresolved_group="Unresolved",
        match_status_audit=_empty_audit(),
        group_reason_audit=_empty_audit(),
        special_value_audit=_empty_audit(),
        auxiliary_label_audit=_empty_audit(),
        **scope,
    )


def test_analysis_renormalizes_benchmark_and_reports_uncertainty() -> None:
    evidence = _prepare(
        pd.DataFrame(
            {
                "UT": ["A"],
                "publication_year": [2020],
                "broad_groups": [("Chordates", "Plants")],
                "broad_groups_all": [("Chordates", "Plants")],
                "taxa_broad_inclusion_state": [ts.INCLUSION_ORDER[0]],
            }
        )
    )
    benchmark = ts.DescribedDiversity(
        counts=pd.DataFrame(
            {
                "broad_group": GROUP_ORDER,
                "described_species_count": [1, 6, 2, 1, 1, 4],
                "described_species_share": np.array([1, 6, 2, 1, 1, 4]) / 15,
                "described_species_share_pct": (
                    np.array([1, 6, 2, 1, 1, 4]) / 15 * 100
                ),
            }
        ),
        audit=pd.DataFrame(),
    )
    analysis = ts.analyze_taxonomic_skew(
        evidence,
        benchmark,
        benchmark_groups=BENCHMARK_GROUPS,
        n_bootstrap=100,
        seed=7,
    )

    comparison = analysis.comparison.set_index("broad_group")
    assert comparison["described_species_share_pct"].sum() == pytest.approx(100)
    assert comparison["fractional_attention_share_pct"].sum() == pytest.approx(100)
    assert comparison.at["Chordates", "fractional_attention_share_pct"] == 50
    assert comparison.at["Plants", "fractional_attention_share_pct"] == 50
    assert analysis.bootstrap_intervals["attention_ci_low_pct"].notna().all()


def test_prepared_scope_defaults_to_the_whole_corpus() -> None:
    evidence = _prepare(_prepared_articles())

    assert set(evidence.included["UT"]) == {"A", "B", "C", "D"}


def test_prepared_scope_restricts_direction_and_year_window() -> None:
    evidence = _prepare(
        _prepared_articles(),
        direction="negative",
        start_year=2000,
        end_year=2025,
    )

    # B is positive-direction, C precedes the window, D follows it.
    assert set(evidence.included["UT"]) == {"A"}


def test_prepared_scope_requires_direction_column_when_filtering() -> None:
    articles = _prepared_articles().drop(columns=["s2_dir"])

    with pytest.raises(ts.TaxaSkewError, match="s2_dir"):
        _prepare(articles, direction="negative")


def test_prepared_scope_rejects_an_empty_universe() -> None:
    with pytest.raises(ts.TaxaSkewError, match="analytical scope"):
        _prepare(_prepared_articles(), direction="mixed")


def _trend_articles() -> pd.DataFrame:
    """Two years; 2001 adds a multi-group publication so weights must split."""
    rows = [
        ("A", 2000, ("Chordates",)),
        ("B", 2000, ("Non-chordate animals",)),
        ("C", 2001, ("Chordates",)),
        ("D", 2001, ("Chordates", "Non-chordate animals")),
    ]
    return pd.DataFrame(
        {
            "UT": [r[0] for r in rows],
            "publication_year": [r[1] for r in rows],
            "benchmark_groups": [r[2] for r in rows],
        }
    )


def test_trend_splits_multi_group_publications_and_closes_each_year() -> None:
    trend = ts.taxonomic_attention_trend(
        _trend_articles(), benchmark_groups=("Chordates", "Non-chordate animals")
    )

    # 2001: A vertebrate-only publication plus a half-half one -> 1.5 vs 0.5.
    counts = trend.set_index(["publication_year", "broad_group"])[
        "effective_publication_count"
    ]
    assert counts.loc[(2001, "Chordates")] == pytest.approx(1.5)
    assert counts.loc[(2001, "Non-chordate animals")] == pytest.approx(0.5)

    shares = trend.groupby("publication_year")["fractional_attention_share_pct"].sum()
    assert np.allclose(shares.to_numpy(), 100.0)


def test_trend_summary_reports_slope_and_both_representation_ratios() -> None:
    trend = pd.DataFrame(
        {
            "publication_year": [2000, 2010, 2000, 2010],
            "broad_group": ["Chordates"] * 2 + ["Non-chordate animals"] * 2,
            "fractional_attention_share_pct": [40.0, 30.0, 20.0, 30.0],
            "effective_publication_count": [4.0, 3.0, 2.0, 3.0],
        }
    )

    summary = ts.summarise_attention_trend(
        trend, described_shares={"Chordates": 5.0, "Non-chordate animals": 60.0}
    ).set_index("broad_group")

    # Ten points lost across ten years is minus ten points per decade.
    assert summary.loc["Chordates", "slope_pp_per_decade"] == pytest.approx(-10.0)
    assert summary.loc["Chordates", "change_pp"] == pytest.approx(-10.0)
    assert summary.loc["Chordates", "start_representation_ratio"] == pytest.approx(8.0)
    assert summary.loc["Chordates", "end_representation_ratio"] == pytest.approx(6.0)
    assert summary.loc["Non-chordate animals", "slope_pp_per_decade"] == pytest.approx(10.0)

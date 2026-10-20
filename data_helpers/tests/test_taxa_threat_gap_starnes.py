"""Tests for the Starnes-based regional threat benchmark and the pooled-evidence fix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_helpers.analysis.taxa import threat_gap_starnes as tgs

REGION_ORDER = ("Africa", "Americas")


def _crosswalk() -> pd.DataFrame:
    """Two regions, two subregions each, three countries -- enough to test both the
    subregion-to-region rollup and the country-to-region evidence mapping."""
    return pd.DataFrame(
        {
            "iso3": ["KEN", "TZA", "USA"],
            "country": ["Kenya", "Tanzania", "United States"],
            "region": ["Africa", "Africa", "Americas"],
            "subregion": ["East Africa", "East Africa", "North America"],
        }
    )


def _starnes_csv(tmp_path: Path, *, extra_row: dict | None = None) -> Path:
    """One subregion per region, plus a trailing all-blank row like the real export."""
    rows = [
        {
            "ipbes_region": "Africa", "ipbes_subregion": "East Africa",
            "EX_count": 1, "EW_count": 0, "CR_count": 10, "EN_count": 20, "VU_count": 30,
            "NT_count": 5, "LC_count": 100, "DD_count": 8,
            "threatened_min_count": 60, "threatened_best_count": 63, "threatened_max_count": 68,
        },
        {
            "ipbes_region": "Americas", "ipbes_subregion": "North America",
            "EX_count": 0, "EW_count": 1, "CR_count": 5, "EN_count": 5, "VU_count": 10,
            "NT_count": 2, "LC_count": 50, "DD_count": 4,
            "threatened_min_count": 21, "threatened_best_count": 22, "threatened_max_count": 24,
        },
    ]
    if extra_row is not None:
        rows.append(extra_row)
    path = tmp_path / "starnes.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    # Append a trailing all-blank line, matching the real Starnes export exactly.
    with path.open("a") as handle:
        handle.write("," * (len(frame.columns) - 1) + "\n")
    return path


def test_load_starnes_subregion_counts_computes_cr_en_vu_and_drops_blank_rows(
    tmp_path: Path,
) -> None:
    frame = tgs.load_starnes_subregion_counts(_starnes_csv(tmp_path))

    assert len(frame) == 2
    east_africa = frame.set_index("subregion").loc["East Africa"]
    assert east_africa["threatened_cr_en_vu"] == 10 + 20 + 30
    north_america = frame.set_index("subregion").loc["North America"]
    assert north_america["threatened_cr_en_vu"] == 5 + 5 + 10  # EW not counted


def test_load_starnes_subregion_counts_rejects_duplicate_subregion_rows(
    tmp_path: Path,
) -> None:
    duplicate = {
        "ipbes_region": "Africa", "ipbes_subregion": "East Africa",
        "EX_count": 0, "EW_count": 0, "CR_count": 1, "EN_count": 1, "VU_count": 1,
        "NT_count": 0, "LC_count": 0, "DD_count": 0,
        "threatened_min_count": 3, "threatened_best_count": 3, "threatened_max_count": 3,
    }
    with pytest.raises(tgs.TaxaThreatGapStarnesError, match="duplicate"):
        tgs.load_starnes_subregion_counts(_starnes_csv(tmp_path, extra_row=duplicate))


def test_load_starnes_subregion_counts_reports_missing_source(tmp_path: Path) -> None:
    with pytest.raises(tgs.TaxaThreatGapStarnesError, match="not found"):
        tgs.load_starnes_subregion_counts(tmp_path / "absent.csv")


def test_regional_benchmark_sums_subregions_and_shares_to_100(tmp_path: Path) -> None:
    bench = tgs.starnes_regional_threat_benchmark(
        _starnes_csv(tmp_path), _crosswalk(), region_order=REGION_ORDER
    )

    assert list(bench["region"]) == list(REGION_ORDER)
    africa = bench.set_index("region").loc["Africa"]
    assert africa["threatened_species"] == 60  # 10 + 20 + 30, the one Africa subregion
    assert bench["threat_share_pct"].sum() == pytest.approx(100.0)
    assert africa["threat_share_pct"] == pytest.approx(60 / (60 + 20) * 100.0)


def test_regional_benchmark_rejects_unmatched_subregion(tmp_path: Path) -> None:
    bad_row = {
        "ipbes_region": "Africa", "ipbes_subregion": "Nowhere",
        "EX_count": 0, "EW_count": 0, "CR_count": 1, "EN_count": 1, "VU_count": 1,
        "NT_count": 0, "LC_count": 0, "DD_count": 0,
        "threatened_min_count": 3, "threatened_best_count": 3, "threatened_max_count": 3,
    }
    # Overwrite rather than append, so the fixture doesn't also trip the duplicate check.
    frame = pd.DataFrame(
        [
            {
                "ipbes_region": "Africa", "ipbes_subregion": "East Africa",
                "EX_count": 1, "EW_count": 0, "CR_count": 10, "EN_count": 20, "VU_count": 30,
                "NT_count": 5, "LC_count": 100, "DD_count": 8,
                "threatened_min_count": 60, "threatened_best_count": 63, "threatened_max_count": 68,
            },
            bad_row,
        ]
    )
    path = tmp_path / "starnes_bad.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(tgs.TaxaThreatGapStarnesError, match="no match"):
        tgs.starnes_regional_threat_benchmark(path, _crosswalk(), region_order=REGION_ORDER)


def test_regional_benchmark_excludes_regions_outside_region_order(tmp_path: Path) -> None:
    """Antarctica-style rows in the source must not force their way into a 4-region table."""
    bench = tgs.starnes_regional_threat_benchmark(
        _starnes_csv(tmp_path), _crosswalk(), region_order=("Africa",)
    )
    assert list(bench["region"]) == ["Africa"]
    assert bench["threat_share_pct"].iloc[0] == pytest.approx(100.0)


def _included() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UT": ["A1", "A2", "A3"],
            "benchmark_groups": [("Vertebrates",), ("Vertebrates",), ("Plants",)],
        }
    )


def _countries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UT": ["A1", "A2", "A3"],
            # A1 studies two countries in the *same* region (Africa) -- must dedup to one.
            "pred_countries": [["KEN", "TZA"], ["USA"], ["USA"]],
        }
    )


def test_pooled_evidence_dedups_multi_country_same_region_article_to_one() -> None:
    """The bug being fixed: a paper naming Kenya and Tanzania must count once for Africa."""
    result = tgs.pooled_regional_evidence(
        _included(), _countries(), _crosswalk(), group="Vertebrates", region_order=REGION_ORDER
    )
    africa = result.set_index("region").loc["Africa"]
    assert africa["n_publications"] == 1
    assert result.attrs["n_region_occurrences"] == 2  # A1 -> Africa, A2 -> Americas


def test_pooled_evidence_counts_multi_region_article_once_per_region() -> None:
    included = pd.DataFrame(
        {"UT": ["B1"], "benchmark_groups": [("Chordates",)]}
    )
    countries = pd.DataFrame({"UT": ["B1"], "pred_countries": [["KEN", "USA"]]})

    result = tgs.pooled_regional_evidence(
        included, countries, _crosswalk(), group="Chordates", region_order=REGION_ORDER
    )
    assert set(result.set_index("region")["n_publications"]) == {1}
    assert result.attrs["n_multi_region_articles"] == 1
    assert result.attrs["n_articles_included"] == 1


def test_pooled_evidence_shares_sum_to_100() -> None:
    result = tgs.pooled_regional_evidence(
        _included(), _countries(), _crosswalk(), group="Vertebrates", region_order=REGION_ORDER
    )
    assert result["evidence_share_pct"].sum() == pytest.approx(100.0)


def test_pooled_evidence_rejects_absent_group() -> None:
    with pytest.raises(tgs.TaxaThreatGapStarnesError, match="No publications"):
        tgs.pooled_regional_evidence(
            _included(), _countries(), _crosswalk(), group="Fungi", region_order=REGION_ORDER
        )


def test_combine_benchmark_merges_and_scores_ratio(tmp_path: Path) -> None:
    threat = tgs.starnes_regional_threat_benchmark(
        _starnes_csv(tmp_path), _crosswalk(), region_order=REGION_ORDER
    )
    evidence = tgs.pooled_regional_evidence(
        _included(), _countries(), _crosswalk(), group="Vertebrates", region_order=REGION_ORDER
    )
    combined = tgs.combine_starnes_benchmark(evidence, threat)

    assert set(combined["region"]) == set(REGION_ORDER)
    africa = combined.set_index("region").loc["Africa"]
    assert africa["evidence_to_threat_ratio"] == pytest.approx(
        africa["evidence_share_pct"] / africa["threat_share_pct"]
    )


def test_combine_benchmark_rejects_mismatched_regions(tmp_path: Path) -> None:
    threat = tgs.starnes_regional_threat_benchmark(
        _starnes_csv(tmp_path), _crosswalk(), region_order=REGION_ORDER
    )
    evidence = tgs.pooled_regional_evidence(
        _included(), _countries(), _crosswalk(), group="Vertebrates", region_order=("Africa",)
    )
    with pytest.raises(tgs.TaxaThreatGapStarnesError, match="same set of regions"):
        tgs.combine_starnes_benchmark(evidence, threat)

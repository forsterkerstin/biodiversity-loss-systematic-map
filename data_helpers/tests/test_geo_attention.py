"""Tests for compact country and region research-attention summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from data_helpers.analysis.geo.attention import (
    GeographyAttentionError,
    prepare_geography_attention,
)


def _crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "country": "Aland",
                "region": "Africa",
                "subregion": "West",
            },
            {
                "iso3": "BBB",
                "country": "Bland",
                "region": "Americas",
                "subregion": "North",
            },
            {
                "iso3": "DDD",
                "country": "Dland",
                "region": "Asia and the Pacific",
                "subregion": "East",
            },
            {
                "iso3": "",
                "country": "Uncoded place",
                "region": "Americas",
                "subregion": "North",
            },
        ]
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UT": ["A", "B", "C", "D"],
            "pred_countries": [
                ["AAA", "BBB", "aaa"],
                '["AAA"]',
                ["Not Applicable", "DDD"],
                ["Unclear", "BBB"],
            ],
        }
    )


def test_fractionalizes_country_attention_and_summarizes_regions() -> None:
    result = prepare_geography_attention(_evidence(), crosswalk=_crosswalk())

    assert result.publication_count == 4
    assert result.assignment_count == 5
    countries = result.countries.set_index("country")
    assert countries.index.tolist() == ["Aland", "Bland", "Dland"]
    assert countries.loc["Aland", "whole_publications"] == 2
    assert countries.loc["Aland", "fractional_publication_equivalent"] == pytest.approx(
        1.5
    )
    assert countries.loc["Bland", "fractional_publication_equivalent"] == pytest.approx(
        1.5
    )
    assert countries.loc["Dland", "fractional_publication_equivalent"] == pytest.approx(
        1.0
    )
    assert countries["attention_share_pct"].sum() == pytest.approx(100.0)

    regions = result.regions.set_index("region")
    assert regions.loc["Africa", "whole_publications"] == 2
    assert regions.loc["Americas", "whole_publications"] == 2
    assert regions.loc["Asia and the Pacific", "whole_publications"] == 1
    assert regions.loc["Africa", "attention_share_pct"] == pytest.approx(37.5)
    assert regions.loc["Americas", "attention_share_pct"] == pytest.approx(37.5)
    assert regions.loc["Asia and the Pacific", "attention_share_pct"] == pytest.approx(
        25.0
    )
    assert regions["attention_share_pct"].sum() == pytest.approx(100.0)


def test_inherited_base_must_be_unique_and_country_complete() -> None:
    duplicated = pd.concat([_evidence(), _evidence().iloc[[0]]], ignore_index=True)
    with pytest.raises(GeographyAttentionError, match="one non-missing row per UT"):
        prepare_geography_attention(duplicated, crosswalk=_crosswalk())

    no_country = _evidence()
    no_country.at[0, "pred_countries"] = ["Not Applicable"]
    with pytest.raises(GeographyAttentionError, match="not country-complete"):
        prepare_geography_attention(no_country, crosswalk=_crosswalk())


def test_unresolved_country_token_fails_visibly() -> None:
    evidence = _evidence()
    evidence.at[0, "pred_countries"] = ["AAA", "XXX"]
    with pytest.raises(GeographyAttentionError, match="unresolved country tokens"):
        prepare_geography_attention(evidence, crosswalk=_crosswalk())


def test_crosswalk_validation_rejects_duplicate_or_invalid_iso3() -> None:
    duplicate = pd.concat([_crosswalk(), _crosswalk().iloc[[0]]], ignore_index=True)
    with pytest.raises(GeographyAttentionError, match="duplicates"):
        prepare_geography_attention(_evidence(), crosswalk=duplicate)

    invalid = _crosswalk()
    invalid.loc[0, "iso3"] = "AA"
    with pytest.raises(GeographyAttentionError, match="three ASCII letters"):
        prepare_geography_attention(_evidence(), crosswalk=invalid)


def test_repository_crosswalk_returns_only_observed_countries() -> None:
    evidence = pd.DataFrame({"UT": ["A"], "pred_countries": [["USA"]]})
    result = prepare_geography_attention(evidence)

    assert result.countries["iso3"].tolist() == ["USA"]
    assert result.countries["attention_share_pct"].tolist() == [100.0]
    assert result.regions[["region", "country_count"]].to_dict("records") == [
        {"region": "Americas", "country_count": 1}
    ]

"""Direct tests for the shared geographic analysis helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from data_helpers.analysis.geo import geography as geo


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UT": ["A", "B", "C", "D", "E"],
            "pred_countries": [
                ["USA", "BRA"],
                ["USA"],
                ["Not Applicable"],
                [],
                ["XXX"],
            ],
            "pred_threat_l0": [
                ["Pollution"],
                ["Pollution", "Climate change"],
                ["Pollution"],
                ["Climate change"],
                ["Climate change"],
            ],
        }
    )


def test_crosswalk_and_all_aggregation_levels() -> None:
    crosswalk = geo.load_crosswalk()
    assert list(crosswalk.columns) == ["iso3", "country", "region", "subregion"]

    usa = crosswalk.loc[crosswalk["iso3"].eq("USA")].iloc[0]
    source = pd.DataFrame(
        {
            "UT": ["A"],
            "pred_countries": [["USA"]],
            "pred_regions": [[usa["region"]]],
            "pred_subregions": [[usa["subregion"]]],
        }
    )

    expected = {
        "country": ("iso3", "USA"),
        "region": ("region", usa["region"]),
        "subregion": ("subregion", usa["subregion"]),
    }
    for level, (key, value) in expected.items():
        counts, audit = geo.geo_counts(source, level=level, by=None, verbose=False)
        assert counts.loc[0, key] == value
        assert counts.loc[0, "n"] == 1
        assert counts.loc[0, "share"] == pytest.approx(1.0)
        assert audit.bucket_mentions == {"mapped": 1}


def test_geo_counts_deduplicates_documents_and_audits_special_tokens() -> None:
    counts, audit = geo.geo_counts(_source(), verbose=False)
    indexed = counts.set_index(["pred_threat_l0", "iso3"])

    assert indexed.loc[("Pollution", "USA"), "n"] == 2
    assert indexed.loc[("Pollution", "BRA"), "n"] == 1
    assert indexed.loc[("Climate change", "USA"), "n"] == 1
    assert indexed.loc[("Pollution", "USA"), "group_total"] == 2
    assert audit.records_total == 5
    assert audit.records_without_geography == 1
    assert audit.bucket_mentions == {
        "mapped": 4,
        "not_applicable": 1,
        "unresolved": 1,
    }
    assert audit.unresolved_tokens == {"XXX": 1}


def test_polygon_loader_returns_the_cached_ipbes_layer() -> None:
    polygons = geo.load_polygons(
        simplify_tolerance=0.2,
        preserve_topology=False,
        use_cache=True,
    )

    assert not polygons.empty
    assert "geometry" in polygons
    assert polygons.crs is not None
    assert not polygons.geometry.isna().any()

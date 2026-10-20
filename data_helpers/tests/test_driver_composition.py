"""Tests for the income-composition analysis and its diagnostics.

These cover the helpers behind `notebooks/results/02_income_composition.ipynb`,
including the transition and missing-assignment diagnostics.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from shapely.geometry import box

from data_helpers.analysis import driver_composition as dc

GROUPS = ["Low income", "Lower middle income", "Upper middle income", "High income"]
INVASIVE = "Invasive & Other Problematic Species, Genes & Diseases"


def _classified(records: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        columns=[
            "UT",
            "publication_year",
            "current_income_group",
            "historical_income_group",
        ],
    )


def test_transition_counts_keep_only_dual_standard_assignments() -> None:
    classified = _classified(
        [
            ("a", 2001, "High income", "Upper middle income"),
            ("b", 2002, "High income", "High income"),
            ("c", 2003, "High income", "Not classified"),
            ("d", 2004, "Not classified", "High income"),
        ]
    )

    transitions = dc.income_transition_counts(classified, GROUPS)

    total = transitions["publication_country_assignments"].sum()
    assert total == 2, "Rows lacking a standard group under either scheme must drop."
    moved = transitions.loc[
        transitions["current_income_group"].ne(
            transitions["historical_income_group"]
        ),
        "publication_country_assignments",
    ].sum()
    assert moved == 1


def test_missing_by_year_counts_assignments_and_unique_publications() -> None:
    classified = _classified(
        [
            ("a", 2001, "High income", "Not classified"),
            ("a", 2001, "High income", "Not classified"),
            ("b", 2002, "High income", "High income"),
        ]
    )

    missing = dc.missing_historical_by_year(classified, GROUPS).set_index(
        "publication_year"
    )

    assert list(missing.index) == [2001]
    assert missing.loc[2001, "publication_country_assignments"] == 2
    assert missing.loc[2001, "unique_publications"] == 1


def test_build_income_threat_attributions_splits_weight_by_country_and_threat() -> None:
    linked = pd.DataFrame(
        [
            # Same income group, two countries, one threat: shares sum, no inflation.
            ("same-group", 2010, "CAN", "High income", ["T1"]),
            ("same-group", 2010, "USA", "High income", ["T1"]),
            # Different income groups, two threats: publication's unit weight
            # is split between the groups, not duplicated into each.
            ("diff-group", 2010, "SWE", "High income", ["T1", "T2"]),
            ("diff-group", 2010, "LTU", "Lower middle income", ["T1", "T2"]),
            # Single country, single threat: gets the whole unit.
            ("single", 2010, "AAA", "Low income", ["T1"]),
        ],
        columns=[
            "UT",
            "publication_year",
            "country_code",
            "historical_income_group",
            "pred_threat_l0",
        ],
    )

    attributions = dc.build_income_threat_attributions(linked)

    same_group = attributions.loc[attributions["UT"].eq("same-group")]
    assert same_group["income_group"].unique().tolist() == ["High income"]
    assert same_group["attribution_weight"].sum() == pytest.approx(1.0)

    diff_group = attributions.loc[attributions["UT"].eq("diff-group")]
    per_group_total = diff_group.groupby("income_group")[
        "attribution_weight"
    ].sum()
    assert per_group_total.to_dict() == pytest.approx(
        {"High income": 0.5, "Lower middle income": 0.5}
    )

    # Every publication's weight sums to exactly one, globally -- not per group.
    totals = attributions.groupby("UT")["attribution_weight"].sum()
    assert totals.to_numpy() == pytest.approx(np.ones(len(totals)))


def test_bootstrap_extreme_contrast_matches_composition_matrix_difference() -> None:
    linked = pd.DataFrame(
        [
            ("a", 2010, "SWE", "High income", ["T1", "T2"]),
            ("a", 2010, "LTU", "Lower middle income", ["T1", "T2"]),
            ("b", 2010, "USA", "High income", ["T1"]),
            ("c", 2010, "IND", "Low income", ["T2"]),
            ("d", 2010, "NGA", "Low income", ["T1"]),
        ],
        columns=[
            "UT",
            "publication_year",
            "country_code",
            "historical_income_group",
            "pred_threat_l0",
        ],
    )
    attributions = dc.build_income_threat_attributions(linked)
    threat_order = ["T1", "T2"]
    groups_present = ["Low income", "Lower middle income", "High income"]
    _, matrix = dc.composition_matrix(attributions, threat_order, groups_present)

    contrast = dc.bootstrap_extreme_contrast(
        attributions,
        threat_order,
        low_group="Low income",
        high_group="High income",
        n_bootstrap=5,
        seed=0,
    ).set_index("threat")

    implied = 100 * (
        matrix.loc[threat_order, "High income"]
        - matrix.loc[threat_order, "Low income"]
    )
    for threat in threat_order:
        assert contrast.loc[threat, "high_minus_low_pp"] == pytest.approx(
            implied.loc[threat]
        )


def test_bootstrap_extreme_contrast_uses_common_publication_resample(
    monkeypatch,
) -> None:
    linked = pd.DataFrame(
        [
            ("shared", 2010, "SWE", "High income", ["T1"]),
            ("shared", 2010, "NGA", "Low income", ["T1"]),
            ("low_only", 2010, "IND", "Low income", ["T1"]),
            ("high_only", 2010, "USA", "High income", ["T1"]),
        ],
        columns=[
            "UT",
            "publication_year",
            "country_code",
            "historical_income_group",
            "pred_threat_l0",
        ],
    )
    attributions = dc.build_income_threat_attributions(linked)

    class FakeRng:
        def __init__(self) -> None:
            self.calls = []

        def integers(self, low, high, size=None):
            self.calls.append((low, high, size))
            return np.array([0, 0, 2], dtype=int)

    fake_rng = FakeRng()
    monkeypatch.setattr(np.random, "default_rng", lambda seed: fake_rng)

    dc.bootstrap_extreme_contrast(
        attributions,
        ["T1"],
        low_group="Low income",
        high_group="High income",
        n_bootstrap=1,
        seed=0,
    )

    assert len(fake_rng.calls) == 1
    assert fake_rng.calls[0] == (0, 3, 3)


def test_primary_uses_all_study_designs_and_negative_direction_only(
    tmp_path,
) -> None:
    countries = [
        ("AAA", "WBA", "Alpha", "Low income"),
        ("BBB", "WBB", "Beta", "Lower middle income"),
        ("CCC", "WBC", "Gamma", "Upper middle income"),
        ("DDD", "WBD", "Delta", "High income"),
    ]
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "ipbes_iso3": iso3,
                        "wb_entity_code": entity_code,
                        "wb_entity_name": country,
                        "wb_region": "Test region",
                        "income_group": income_group,
                        "has_world_bank_economy": True,
                    }
                    for iso3, entity_code, country, income_group in countries
                ]
            }
        ),
        encoding="utf-8",
    )
    historical_path = tmp_path / "historical.parquet"
    pd.DataFrame(
        [
            {
                "wb_entity_code": entity_code,
                "classification_fy": year,
                "income": income_group,
                "gni_reference_year": year - 1,
                "effective_from": f"{year - 1}-07-01",
                "effective_to": f"{year}-06-30",
            }
            for _, entity_code, _, income_group in countries
            for year in (2001, 2002)
        ]
    ).to_parquet(historical_path, index=False)
    corpus = pd.DataFrame(
        [
            ("obs-low", "negative", ["Observational"], "AAA", "T1"),
            ("obs-lmic", "negative", ["Observational"], "BBB", "T2"),
            ("obs-umic", "negative", ["Observational"], "CCC", "T2"),
            ("obs-high", "negative", ["Observational"], "DDD", "T2"),
            ("exp-low", "negative", ["Experimental"], "AAA", "T1"),
            ("exp-high", "negative", ["Experimental"], "DDD", "T3"),
            ("positive", "positive", ["Experimental"], "AAA", "T3"),
        ],
        columns=[
            "UT",
            "s2_dir",
            "pred_study_design",
            "country",
            "threat",
        ],
    )
    corpus["publication_year"] = 2001
    corpus["pred_countries"] = corpus["country"].map(lambda value: [value])
    corpus["pred_threat_l0"] = corpus["threat"].map(lambda value: [value])

    preparation = dc.prepare_historical_income_evidence(
        corpus,
        mapping_path=mapping_path,
        historical_path=historical_path,
        income_groups=GROUPS,
        direction="negative",
        start_year=2000,
        end_year=2001,
    )

    assert set(preparation.primary["UT"]) == {
        "obs-low",
        "obs-lmic",
        "obs-umic",
        "obs-high",
        "exp-low",
        "exp-high",
    }
    audit = preparation.audit.set_index("metric")["value"]
    assert audit["Negative publications after country exclusions"] == 6


def test_build_country_article_counts_preserves_metadata_and_unique_articles() -> None:
    primary = pd.DataFrame(
        {
            "UT": ["u1", "u1", "u2", "u3"],
            "country_code": ["BBB", "BBB", "BBB", "AAA"],
            "wb_entity_name": ["Beta", "Beta", "Beta", "Alpha"],
            "wb_region": ["South", "South", "South", "North"],
        }
    )

    result = dc.build_country_article_counts(primary)

    assert list(result.columns) == [
        "iso3",
        "country",
        "region",
        "record_count",
    ]
    assert result.to_dict("records") == [
        {
            "iso3": "AAA",
            "country": "Alpha",
            "region": "North",
            "record_count": 1,
        },
        {
            "iso3": "BBB",
            "country": "Beta",
            "region": "South",
            "record_count": 2,
        },
    ]


def test_build_country_article_counts_rejects_conflicting_metadata() -> None:
    primary = pd.DataFrame(
        {
            "UT": ["u1", "u2"],
            "country_code": ["AAA", "AAA"],
            "wb_entity_name": ["Alpha", "Alternate Alpha"],
            "wb_region": ["North", "North"],
        }
    )

    with pytest.raises(ValueError, match="conflicts found for: AAA"):
        dc.build_country_article_counts(primary)


def _minimal_export_inputs():
    preparation = dc.EvidencePreparation(
        corpus=pd.DataFrame(),
        negative=pd.DataFrame(),
        world_bank_lookup=pd.DataFrame(),
        historical_classifications=pd.DataFrame(),
        world_bank_linked=pd.DataFrame(),
        classified=pd.DataFrame({"UT": ["u1"]}),
        primary=pd.DataFrame(),
        audit=pd.DataFrame({"metric": ["example"], "value": [1]}),
        transitions=pd.DataFrame(),
        missing_by_year=pd.DataFrame(),
        country_exclusions=pd.DataFrame(),
        reclassified_share=0.0,
    )
    composition = dc.CompositionAnalysis(
        attributions=pd.DataFrame(
            {
                "UT": ["u1"],
                "income_group": ["Low income"],
                "threat": ["T1"],
                "attribution_weight": [1.0],
            }
        ),
        long_composition=pd.DataFrame({"threat": ["T1"]}),
        matrix=pd.DataFrame(),
        extreme_contrast=pd.DataFrame({"threat": ["T1"]}),
        group_counts=pd.DataFrame(
            {"unique_publications": [1]}, index=pd.Index(["Low income"])
        ),
        observed_threat_order=["T1"],
    )
    return preparation, composition


def test_export_manuscript_tables_writes_core_tables(tmp_path) -> None:
    preparation, composition = _minimal_export_inputs()
    written = dc.export_manuscript_tables(
        preparation,
        composition,
        data_directory=tmp_path / "data",
        table_directory=tmp_path / "tables",
        classified_assignments_file="assignments.parquet",
    )

    assert {path.name for path in written} == {
        "assignments.parquet",
        "income_group_threat_composition.csv",
        "high_minus_low_income_bootstrap.csv",
    }


def test_country_complete_case_exclusion_drops_all_mixed_assignments(
    tmp_path,
) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "ipbes_iso3": "AAA",
                        "wb_entity_code": "WBA",
                        "wb_entity_name": "Alpha",
                        "wb_region": "Test region",
                        "income_group": "Low income",
                        "has_world_bank_economy": True,
                    },
                    {
                        "ipbes_iso3": "BBB",
                        "wb_entity_code": None,
                        "wb_entity_name": None,
                        "wb_region": None,
                        "income_group": None,
                        "has_world_bank_economy": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    historical_path = tmp_path / "historical.parquet"
    pd.DataFrame(
        [
            {
                "wb_entity_code": "WBA",
                "classification_fy": 2001,
                "income": "Low income",
                "gni_reference_year": 2000,
                "effective_from": "2000-07-01",
                "effective_to": "2001-06-30",
            }
        ]
    ).to_parquet(historical_path, index=False)
    corpus = pd.DataFrame(
        {
            "UT": ["valid", "invalid-mixed", "ineligible-mixed"],
            "publication_year": [2001, 2001, 2001],
            "s2_dir": ["negative", "negative", "negative"],
            "pred_study_design": [["Observational"]] * 3,
            "pred_countries": [
                ["aaa"],
                ["AAA", "XXX"],
                ["AAA", "BBB"],
            ],
            "pred_threat_l0": [["T1"], ["T1"], ["T1"]],
        }
    )

    preparation = dc.prepare_historical_income_evidence(
        corpus,
        mapping_path=mapping_path,
        historical_path=historical_path,
        income_groups=GROUPS,
        direction="negative",
        start_year=2000,
        end_year=2001,
    )

    assert preparation.primary[["UT", "country_code"]].values.tolist() == [
        ["valid", "AAA"]
    ]
    assert preparation.country_exclusions[
        ["UT", "normalized_token", "exclusion_reason"]
    ].to_dict("records") == [
        {
            "UT": "ineligible-mixed",
            "normalized_token": "BBB",
            "exclusion_reason": "valid_iso3_without_world_bank_economy",
        },
        {
            "UT": "invalid-mixed",
            "normalized_token": "XXX",
            "exclusion_reason": "unresolved_country_token",
        },
    ]
    audit = preparation.audit.set_index("metric")["value"]
    assert audit["Excluded publications (either reason below)"] == 2


def _plotting_composition() -> tuple[
    dc.CompositionAnalysis,
    list[str],
    dict[str, str],
    dict[str, str],
]:
    threats = [f"T{index}" for index in range(10)] + [
        "Other Options",
        "Unclear",
    ]
    shares = np.arange(len(threats), 0, -1, dtype=float)
    shares /= shares.sum()
    matrix = pd.DataFrame(
        {group: shares for group in GROUPS},
        index=threats,
    )
    contrasts = np.arange(len(threats), 0, -1, dtype=float)
    extreme = pd.DataFrame(
        {
            "threat": threats,
            "high_minus_low_pp": contrasts,
            "bootstrap_ci_low": contrasts - 0.5,
            "bootstrap_ci_high": contrasts + 0.5,
        }
    )
    group_counts = pd.DataFrame(
        {
            "unique_publications": [6_986, 23_659, 48_284, 73_385],
            "represented_countries": [70, 101, 90, 85],
        },
        index=GROUPS,
    )
    composition = dc.CompositionAnalysis(
        attributions=pd.DataFrame(),
        long_composition=pd.DataFrame(),
        matrix=matrix,
        extreme_contrast=extreme,
        group_counts=group_counts,
        observed_threat_order=threats,
    )
    color_values = plt.get_cmap("tab20")(np.linspace(0, 1, len(threats)))
    colors = {
        threat: to_hex(color)
        for threat, color in zip(threats, color_values, strict=True)
    }
    names = {threat: f"Display {threat}" for threat in threats}
    return composition, threats, colors, names


def test_income_composition_composite_has_map_counts_and_full_legend() -> None:
    geopandas = pytest.importorskip("geopandas")
    composition, threats, colors, names = _plotting_composition()
    country_counts = pd.DataFrame(
        {
            "iso3": ["AAA", "BBB"],
            "country": ["Alpha", "Beta"],
            "region": ["North", "South"],
            "record_count": [5, 50],
        }
    )
    polygons = geopandas.GeoDataFrame(
        {"ISO_3": ["AAA", "BBB", "CCC"]},
        geometry=[box(-10, 0, 0, 10), box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:4326",
    )

    figure = dc.plot_income_composition(
        composition,
        income_groups=GROUPS,
        threat_colors=colors,
        threat_names=names,
        country_counts=country_counts,
        polygons=polygons,
        contrast_excluded_threats=["Other Options", "Unclear"],
    )
    axes = {axis.get_label(): axis for axis in figure.axes}

    assert set(axes) == {
        "panel_a_map",
        "panel_a_colorbar",
        "panel_b_composition",
        "panel_c_contrast",
    }
    assert axes["panel_a_colorbar"].get_xlabel() == "Articles per country"
    assert [
        label.get_text()
        for label in axes["panel_a_colorbar"].get_xticklabels()
    ] == [
        "1",
        "3",
        "10",
        "30",
        "100",
        "300",
        "1,000",
        "3,000",
        "10,000",
        "30,000",
    ]
    assert list(axes["panel_a_colorbar"].collections[-1].cmap.colors) == list(
        dc.GEO_COUNT_COLORS
    )
    assert axes["panel_c_contrast"].get_xlabel() == (
        "Difference in threat share (percentage points)\nHIC minus LIC"
    )
    assert [
        label.get_text()
        for label in axes["panel_c_contrast"].get_yticklabels()
    ] == [names[threat] for threat in reversed(threats[:10])]
    expected_stack_order = [
        *reversed(threats[:10]),
        "Other Options",
        "Unclear",
    ]
    composition_patches = axes["panel_b_composition"].patches
    assert [
        to_hex(composition_patches[index * len(GROUPS)].get_facecolor())
        for index in range(len(expected_stack_order))
    ] == [colors[threat] for threat in expected_stack_order]
    assert [
        label.get_text()
        for label in axes["panel_b_composition"].get_xticklabels()
    ] == [
        "LIC\n($n$ = 6,986)",
        "LMIC\n($n$ = 23,659)",
        "UMIC\n($n$ = 48,284)",
        "HIC\n($n$ = 73,385)",
    ]
    assert {text.get_text() for text in axes["panel_b_composition"].texts} == {
        "b"
    }
    assert not axes["panel_a_map"].texts
    assert {text.get_text() for text in figure.texts} == {"a"}
    assert {text.get_text() for text in axes["panel_c_contrast"].texts} == {
        "c"
    }
    figure.canvas.draw()
    panel_a_label = figure.texts[0]
    panel_b_label = axes["panel_b_composition"].texts[0]
    panel_a_x = panel_a_label.get_transform().transform(
        panel_a_label.get_position()
    )[0]
    panel_b_x = panel_b_label.get_transform().transform(
        panel_b_label.get_position()
    )[0]
    assert panel_a_x == pytest.approx(panel_b_x, abs=1.0)
    expected_legend_display_order = [
        *reversed(expected_stack_order[:10]),
        *expected_stack_order[10:],
    ]
    expected_legend_handle_order = [
        threat
        for column in range(4)
        for threat in expected_legend_display_order[column::4]
    ]
    assert [
        text.get_text() for text in figure.legends[0].texts
    ] == [
        (
            "Other threats"
            if threat == "Other Options"
            else names[threat]
        )
        for threat in expected_legend_handle_order
    ]
    assert axes["panel_b_composition"].get_legend() is None
    plt.close(figure)


def test_income_composition_plot_rejects_one_map_input() -> None:
    composition, _, colors, names = _plotting_composition()

    with pytest.raises(ValueError, match="must either both be provided"):
        dc.plot_income_composition(
            composition,
            income_groups=GROUPS,
            threat_colors=colors,
            threat_names=names,
            country_counts=pd.DataFrame(),
        )

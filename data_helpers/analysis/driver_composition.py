"""Reusable calculations for the income-group threat-composition analysis.

The functions preserve the publication as the evidentiary unit: every publication
contributes exactly one unit of evidence in total, fractionally split 1/(countries x
threats) across the distinct countries and threats it carries. Multiple countries in
the same income group sum their shares within that group rather than multiplying
the publication's contribution to it; countries in different groups split the
publication's single unit of weight between them rather than each claiming a full unit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from data_helpers.analysis.geo.count_plotting import (
    add_class_colorbar,
    build_count_scale,
)
from data_helpers.analysis.geo.geography import (
    _AGGREGATE,
    _NOT_APPLICABLE,
    _UNCLEAR,
    _normalise,
)
from data_helpers.labels import parse_list_labels
from data_helpers.visualization import BIODIVERSITY

GEO_COUNT_BINS = (
    1,
    3,
    10,
    30,
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
)
GEO_COUNT_COLORS = (
    "#FFFFD9",
    "#EDF8B1",
    "#C7E9B4",
    "#7FCDBB",
    "#41B6C4",
    "#1D91C0",
    "#225EA8",
    "#253494",
    "#081D58",
)


@dataclass
class EvidencePreparation:
    """Historically classified evidence and its audit tables."""

    corpus: pd.DataFrame
    negative: pd.DataFrame
    world_bank_lookup: pd.DataFrame
    historical_classifications: pd.DataFrame
    world_bank_linked: pd.DataFrame
    classified: pd.DataFrame
    primary: pd.DataFrame
    audit: pd.DataFrame
    transitions: pd.DataFrame
    missing_by_year: pd.DataFrame
    country_exclusions: pd.DataFrame
    reclassified_share: float


@dataclass
class CompositionAnalysis:
    """Income-group threat composition and extreme-group contrast."""

    attributions: pd.DataFrame
    long_composition: pd.DataFrame
    matrix: pd.DataFrame
    extreme_contrast: pd.DataFrame
    group_counts: pd.DataFrame
    observed_threat_order: list[str]


_SPECIAL_COUNTRY_TOKENS = _NOT_APPLICABLE | _AGGREGATE | _UNCLEAR


def expand_publication_country_tokens(records_df: pd.DataFrame) -> pd.DataFrame:
    """Expand country labels once for complete-case checks and World Bank linking."""
    required_columns = {"UT", "publication_year", "pred_countries"}
    missing_columns = required_columns.difference(records_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"records_df is missing required columns: {missing}")

    columns = ["UT", "publication_year", "pred_countries"]
    if "pred_threat_l0" in records_df:
        columns.append("pred_threat_l0")
    work = records_df.loc[:, columns].copy()
    work["token"] = work["pred_countries"].map(parse_list_labels)
    work = work.explode("token", ignore_index=True).dropna(subset=["token"])
    work["token"] = work["token"].astype("string").str.strip()
    work["normalized_token"] = work["token"].map(_normalise)
    return work.drop(columns="pred_countries")


def country_complete_case_exclusions(
    country_tokens_df: pd.DataFrame,
    world_bank_lookup_df: pd.DataFrame,
    valid_ipbes_iso3: set[str],
) -> pd.DataFrame:
    """List country tokens that trigger whole-publication exclusion.

    Special non-country labels are ignored. Every other token must resolve to
    the analysis-specific World Bank lookup. A valid IPBES ISO3 without a
    separate World Bank economy is distinguished from an unrecognized token.
    """
    work = country_tokens_df
    lookup_codes = set(world_bank_lookup_df["country_code"])
    ineligible = (
        ~work["normalized_token"].isin(_SPECIAL_COUNTRY_TOKENS)
        & ~work["normalized_token"].isin(lookup_codes)
    )
    result = work.loc[
        ineligible,
        ["UT", "publication_year", "token", "normalized_token"],
    ].drop_duplicates(["UT", "normalized_token"])
    result["exclusion_reason"] = np.where(
        result["normalized_token"].isin(valid_ipbes_iso3),
        "valid_iso3_without_world_bank_economy",
        "unresolved_country_token",
    )
    return result.sort_values(
        ["UT", "normalized_token"], kind="stable"
    ).reset_index(drop=True)


def link_publications_to_world_bank(
    country_tokens_df: pd.DataFrame,
    world_bank_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per unique publication–World Bank country assignment."""
    work = country_tokens_df[
        ["UT", "publication_year", "pred_threat_l0", "normalized_token"]
    ].rename(columns={"normalized_token": "country_code"})
    work = work.loc[work["country_code"].ne("")].drop_duplicates(
        ["UT", "country_code"]
    )
    linked = work.merge(
        world_bank_lookup_df,
        on="country_code",
        how="inner",
        sort=False,
        validate="many_to_one",
    ).reset_index(drop=True)
    linked["_assignment_id"] = np.arange(len(linked), dtype=np.int64)
    assert not linked.duplicated(["UT", "country_code"]).any()
    return linked


def build_country_article_counts(primary_df: pd.DataFrame) -> pd.DataFrame:
    """Count unique primary-analysis publications for each linked country.

    The input is the publication-country frame returned as
    :attr:`EvidencePreparation.primary`.  Country labels are retained for the
    manuscript-table export, while the map itself joins on the renamed ISO3
    code.  A country code must resolve to exactly one name and one World Bank
    region; conflicting lookup metadata indicate an upstream join problem and
    fail here rather than being hidden by aggregation.
    """
    required_columns = {
        "UT",
        "country_code",
        "wb_entity_name",
        "wb_region",
    }
    missing_columns = required_columns.difference(primary_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"primary_df is missing required columns: {missing}")

    work = primary_df.loc[:, sorted(required_columns)].copy()
    if work["country_code"].isna().any():
        raise ValueError("primary_df contains missing country_code values")
    work["country_code"] = work["country_code"].astype("string").str.strip()
    if work["country_code"].eq("").any():
        raise ValueError("primary_df contains blank country_code values")

    metadata_cardinality = work.groupby(
        "country_code", observed=True, sort=False
    ).agg(
        country_names=(
            "wb_entity_name",
            lambda values: values.nunique(dropna=False),
        ),
        regions=("wb_region", lambda values: values.nunique(dropna=False)),
    )
    conflicting = metadata_cardinality.loc[
        metadata_cardinality["country_names"].ne(1)
        | metadata_cardinality["regions"].ne(1)
    ]
    if not conflicting.empty:
        codes = ", ".join(map(str, conflicting.index.tolist()))
        raise ValueError(
            "Each country_code must map to exactly one country name and region; "
            f"conflicts found for: {codes}"
        )

    country_metadata = work.drop_duplicates(
        ["country_code", "wb_entity_name", "wb_region"]
    ).rename(
        columns={
            "country_code": "iso3",
            "wb_entity_name": "country",
            "wb_region": "region",
        }
    )[
        ["iso3", "country", "region"]
    ]
    counts = (
        work.groupby("country_code", observed=True, sort=False)["UT"]
        .nunique()
        .rename("record_count")
        .rename_axis("iso3")
        .reset_index()
    )
    result = country_metadata.merge(
        counts,
        on="iso3",
        how="inner",
        validate="one_to_one",
    )
    result["record_count"] = result["record_count"].astype(int)
    return result.sort_values("iso3").reset_index(drop=True)[
        ["iso3", "country", "region", "record_count"]
    ]


def attach_historical_income(
    linked_df: pd.DataFrame,
    historical_classification_df: pd.DataFrame,
    fiscal_year_offset: int = 0,
) -> pd.DataFrame:
    """Attach one historical classification without changing assignment grain."""
    work = linked_df.copy()
    work["classification_fy"] = (
        work["publication_year"].astype("Int64") + fiscal_year_offset
    )
    result = work.merge(
        historical_classification_df,
        on=["wb_entity_code", "classification_fy"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    assert len(result) == len(work)
    assert result["_assignment_id"].is_unique
    assert np.array_equal(
        result["_assignment_id"].to_numpy(),
        work["_assignment_id"].to_numpy(),
    )
    assert not result.duplicated(["UT", "country_code"]).any()
    return result


def _country_fractional_weight(
    work: pd.DataFrame, *, country_column: str = "country_code"
) -> pd.DataFrame:
    """Weight each row 1/(countries x threats), both counted per publication.

    Both counts are computed globally per ``UT`` -- across every group a
    publication is linked to, not just the group of the current row -- so a
    publication's weight always sums to exactly one in total, however many
    distinct groups it touches. Multiple countries landing in the same
    group are not deduplicated here: their shares are meant to be summed
    by the caller's later groupby, not collapsed before weighting.
    """
    work = work.copy()
    work["attribution_weight"] = 1 / (
        work.groupby("UT")[country_column].transform("nunique")
        * work.groupby("UT")["threat"].transform("nunique")
    )
    weight_audit = work.groupby("UT", observed=True)["attribution_weight"].sum()
    assert np.allclose(weight_audit, 1.0)
    return work


def build_income_threat_attributions(
    linked_df: pd.DataFrame,
    group_column: str = "historical_income_group",
) -> pd.DataFrame:
    """Fractionally allocate each publication's unit weight 1/(countries x threats).

    A publication's weight is split evenly across the distinct countries and
    threats it carries. Countries that share an income group sum their shares
    within that group; countries in different income groups split the
    publication's single unit of weight between the groups, rather than each
    contributing a full unit.
    """
    work = linked_df[
        ["UT", "publication_year", "country_code", group_column, "pred_threat_l0"]
    ].copy()
    work = work.rename(columns={group_column: "income_group"})
    work["threat"] = work["pred_threat_l0"].map(parse_list_labels)
    work = work.explode("threat", ignore_index=True).dropna(subset=["threat"])
    work["threat"] = work["threat"].astype("string").str.strip()
    work = work.loc[work["threat"].ne("")].drop_duplicates(
        ["UT", "country_code", "threat"]
    )
    work = _country_fractional_weight(work)
    return work.reset_index(drop=True)


def composition_from_attributions(
    attribution_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate within-income-group fractional threat shares."""
    result = (
        attribution_df.groupby(["income_group", "threat"], observed=True)[
            "attribution_weight"
        ]
        .sum()
        .rename("weighted_attributions")
        .reset_index()
    )
    result["share"] = result["weighted_attributions"] / result.groupby(
        "income_group", observed=True
    )["weighted_attributions"].transform("sum")
    result["share_pct"] = 100 * result["share"]
    return result


def composition_matrix(
    attribution_df: pd.DataFrame,
    threat_order: Sequence[str],
    income_group_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long and matrix forms of the threat composition."""
    composition = composition_from_attributions(attribution_df)
    matrix = (
        composition.pivot(
            index="threat", columns="income_group", values="share"
        )
        .fillna(0)
        .reindex(
            index=threat_order,
            columns=income_group_order,
            fill_value=0,
        )
    )
    assert np.allclose(matrix.sum(axis=0), 1.0)
    return composition, matrix


def bootstrap_extreme_contrast(
    attribution_df: pd.DataFrame,
    threat_order: Sequence[str],
    low_group: str,
    high_group: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap the high-minus-low composition contrast by publication.

    The resampling unit is the publication (``UT``), not the income group. Every
    row belonging to a sampled publication stays together so that a paper linked
    to both higher- and lower-income countries keeps its within-paper structure
    intact for the same bootstrap draw. This matches the dependence structure in
    the underlying evidence and preserves the point estimate from the full
    composition matrix.
    """
    publication_totals = (
        attribution_df.groupby(["UT", "income_group", "threat"], observed=True)[
            "attribution_weight"
        ]
        .sum()
        .rename("weighted_attributions")
        .reset_index()
    )
    publication_ids = publication_totals["UT"].drop_duplicates().to_numpy()

    def weighted_share_for_group(sample: pd.DataFrame, income_group: str) -> np.ndarray:
        sample = sample.loc[sample["income_group"].eq(income_group)]
        if sample.empty:
            return np.zeros(len(threat_order), dtype=float)
        totals = (
            sample.groupby("threat", observed=True)["weighted_attributions"]
            .sum()
            .reindex(threat_order, fill_value=0)
            .to_numpy(dtype=float)
        )
        total = totals.sum()
        if total <= 0:
            return np.zeros(len(threat_order), dtype=float)
        return totals / total

    def sample_from_publications(publication_sample: np.ndarray) -> pd.DataFrame:
        sampled_counts = (
            pd.Series(publication_sample, dtype="string")
            .value_counts()
            .rename("n_samples")
            .rename_axis("UT")
            .reset_index()
        )
        sample = publication_totals.merge(sampled_counts, on="UT", how="inner")
        sample["weighted_attributions"] = (
            sample["weighted_attributions"] * sample["n_samples"]
        )
        return sample.drop(columns="n_samples")

    observed_sample = sample_from_publications(publication_ids)
    observed = 100 * (
        weighted_share_for_group(observed_sample, high_group)
        - weighted_share_for_group(observed_sample, low_group)
    )
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty((n_bootstrap, len(threat_order)))
    for index in range(n_bootstrap):
        draw = publication_ids[rng.integers(0, len(publication_ids), len(publication_ids))]
        sampled = sample_from_publications(draw)
        bootstrapped[index] = 100 * (
            weighted_share_for_group(sampled, high_group)
            - weighted_share_for_group(sampled, low_group)
        )

    result = pd.DataFrame(
        {
            "threat": threat_order,
            "high_minus_low_pp": observed,
            "bootstrap_ci_low": np.quantile(bootstrapped, 0.025, axis=0),
            "bootstrap_ci_high": np.quantile(bootstrapped, 0.975, axis=0),
        }
    )
    result["interval_excludes_zero"] = (
        result["bootstrap_ci_low"].gt(0)
        | result["bootstrap_ci_high"].lt(0)
    )
    return result.sort_values("high_minus_low_pp", ascending=False).reset_index(
        drop=True
    )


def income_transition_counts(
    classified_df: pd.DataFrame,
    income_groups: Sequence[str],
) -> pd.DataFrame:
    """Count assignments by current and publication-year income group.

    Restricted to assignments carrying a standard group under both schemes, so
    the counts are comparable; recomputable from the exported assignments file.
    """
    comparable = classified_df["historical_income_group"].isin(
        income_groups
    ) & classified_df["current_income_group"].isin(income_groups)
    return (
        classified_df.loc[
            comparable,
            ["current_income_group", "historical_income_group"],
        ]
        .value_counts()
        .rename("publication_country_assignments")
        .reset_index()
    )


def missing_historical_by_year(
    classified_df: pd.DataFrame,
    income_groups: Sequence[str],
) -> pd.DataFrame:
    """Count assignments left without a standard publication-year group."""
    return (
        classified_df.loc[
            ~classified_df["historical_income_group"].isin(income_groups)
        ]
        .groupby("publication_year", dropna=False)
        .agg(
            publication_country_assignments=("UT", "size"),
            unique_publications=("UT", "nunique"),
        )
        .reset_index()
    )


def prepare_historical_income_evidence(
    corpus_df: pd.DataFrame,
    *,
    mapping_path: str | Path,
    historical_path: str | Path,
    income_groups: Sequence[str],
    direction: str,
    start_year: int,
    end_year: int,
) -> EvidencePreparation:
    """Prepare the country-resolved evidence with historical income classifications."""
    corpus = corpus_df.copy()
    assert corpus["UT"].notna().all() and corpus["UT"].is_unique
    corpus["publication_year"] = pd.to_numeric(
        corpus["publication_year"], errors="coerce"
    ).astype("Int64")
    negative_before_country_validation = corpus.loc[
        corpus["s2_dir"].eq(direction)
    ].copy()

    with Path(mapping_path).open(encoding="utf-8") as handle:
        mapping_records = pd.DataFrame(json.load(handle)["records"])
    world_bank_lookup = (
        mapping_records.loc[
            mapping_records["ipbes_iso3"].astype("string").str.len().eq(3)
            & mapping_records["has_world_bank_economy"].eq(True),
            [
                "ipbes_iso3",
                "wb_entity_code",
                "wb_entity_name",
                "wb_region",
                "income_group",
            ],
        ]
        .rename(
            columns={
                "ipbes_iso3": "country_code",
                "income_group": "current_income_group",
            }
        )
        .copy()
    )
    assert world_bank_lookup["country_code"].is_unique

    valid_ipbes_iso3 = set(
        mapping_records.loc[
            mapping_records["ipbes_iso3"]
            .astype("string")
            .str.fullmatch(r"[A-Za-z]{3}", na=False),
            "ipbes_iso3",
        ].str.upper()
    )
    country_tokens = expand_publication_country_tokens(
        negative_before_country_validation
    )
    country_exclusions = country_complete_case_exclusions(
        country_tokens,
        world_bank_lookup,
        valid_ipbes_iso3,
    )
    excluded_ids = set(country_exclusions["UT"])
    negative = negative_before_country_validation.loc[
        ~negative_before_country_validation["UT"].isin(excluded_ids)
    ].copy()

    historical = pd.read_parquet(
        historical_path,
        columns=[
            "wb_entity_code",
            "classification_fy",
            "income",
            "gni_reference_year",
            "effective_from",
            "effective_to",
        ],
    ).rename(columns={"income": "historical_income_group"})
    historical["classification_fy"] = historical["classification_fy"].astype(
        "Int64"
    )
    assert not historical.duplicated(
        ["wb_entity_code", "classification_fy"]
    ).any()

    retained_country_tokens = country_tokens.loc[
        ~country_tokens["UT"].isin(excluded_ids)
    ]
    world_bank_linked = link_publications_to_world_bank(
        retained_country_tokens, world_bank_lookup
    )
    classified = attach_historical_income(world_bank_linked, historical)
    historical_standard = classified["historical_income_group"].isin(
        income_groups
    )
    current_standard = classified["current_income_group"].isin(income_groups)
    primary = classified.loc[
        historical_standard
        & classified["publication_year"].between(start_year, end_year)
    ].copy()

    primary_years = classified["publication_year"].between(
        start_year,
        end_year,
    )
    comparable = historical_standard & current_standard & primary_years
    reclassified = comparable & classified["historical_income_group"].ne(
        classified["current_income_group"]
    )

    token_is_special = retained_country_tokens["normalized_token"].isin(
        _SPECIAL_COUNTRY_TOKENS
    )
    dropped_before_primary = classified.loc[~(historical_standard & primary_years)]
    dropped_year_out_of_range = int((~primary_years).sum())
    dropped_unclassified_in_range = int((primary_years & ~historical_standard).sum())
    assert (
        dropped_year_out_of_range + dropped_unclassified_in_range
        == len(dropped_before_primary)
    )

    audit = pd.DataFrame(
        {
            "metric": [
                "Negative publications before country exclusions",
                "Excluded publications (either reason below)",
                "  Unresolved country token",
                "  Valid ISO3 without World Bank economy",
                "Negative publications after country exclusions",
                "Publication–country token rows",
                "  Resolvable to a World Bank country code",
                "  Special label (Unclear / Not Applicable / aggregate)",
                "World Bank-linked publication–country assignments",
                "World Bank-linked unique publications",
                f"Dropped before {start_year}–{end_year} income classification "
                "(either reason below)",
                f"  Publication year outside {start_year}–{end_year}",
                "  No standard World Bank income classification that year",
                f"Primary assignments: historical group, {start_year}–{end_year}",
                "Primary unique publications",
                "Countries in primary analysis",
                "Assignments reclassified vs current group",
            ],
            "value": [
                len(negative_before_country_validation),
                len(excluded_ids),
                country_exclusions.loc[
                    country_exclusions["exclusion_reason"].eq(
                        "unresolved_country_token"
                    ),
                    "UT",
                ].nunique(),
                country_exclusions.loc[
                    country_exclusions["exclusion_reason"].eq(
                        "valid_iso3_without_world_bank_economy"
                    ),
                    "UT",
                ].nunique(),
                len(negative),
                len(retained_country_tokens),
                int((~token_is_special).sum()),
                int(token_is_special.sum()),
                len(world_bank_linked),
                world_bank_linked["UT"].nunique(),
                len(dropped_before_primary),
                dropped_year_out_of_range,
                dropped_unclassified_in_range,
                len(primary),
                primary["UT"].nunique(),
                primary["wb_entity_code"].nunique(),
                int(reclassified.sum()),
            ],
        }
    )
    transitions = income_transition_counts(classified, income_groups)
    missing_by_year = missing_historical_by_year(classified, income_groups)
    return EvidencePreparation(
        corpus,
        negative,
        world_bank_lookup,
        historical,
        world_bank_linked,
        classified,
        primary,
        audit,
        transitions,
        missing_by_year,
        country_exclusions,
        float(reclassified.sum() / comparable.sum()),
    )


def analyze_income_composition(
    primary_df: pd.DataFrame,
    *,
    threat_order: Sequence[str],
    income_groups: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> CompositionAnalysis:
    """Build the income-group composition and high-minus-low contrast."""
    attributions = build_income_threat_attributions(primary_df)
    observed_order = [
        threat for threat in threat_order
        if threat in set(attributions["threat"])
    ]
    long_composition, matrix = composition_matrix(
        attributions, observed_order, income_groups
    )
    extreme_contrast = bootstrap_extreme_contrast(
        attributions,
        observed_order,
        low_group="Low income",
        high_group="High income",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    extreme_contrast["low_income_share_pct"] = extreme_contrast["threat"].map(
        100 * matrix["Low income"]
    )
    extreme_contrast["high_income_share_pct"] = extreme_contrast["threat"].map(
        100 * matrix["High income"]
    )
    group_counts = (
        primary_df.groupby("historical_income_group", observed=True)
        .agg(
            unique_publications=("UT", "nunique"),
            represented_countries=("wb_entity_code", "nunique"),
        )
        .reindex(income_groups)
    )
    return CompositionAnalysis(
        attributions,
        long_composition,
        matrix,
        extreme_contrast,
        group_counts,
        observed_order,
    )


def export_manuscript_tables(
    preparation: EvidencePreparation,
    composition: CompositionAnalysis,
    *,
    data_directory: str | Path,
    table_directory: str | Path,
    classified_assignments_file: str,
) -> list[Path]:
    """Write the shared publication-country assignments and the two analytical tables."""
    data_directory = Path(data_directory)
    table_directory = Path(table_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)
    assignments_path = data_directory / classified_assignments_file
    classified_export = preparation.classified.copy()
    if "pred_countries" in classified_export:
        classified_export["pred_countries"] = classified_export[
            "pred_countries"
        ].map(
            lambda value: value
            if isinstance(value, str)
            or value is None
            or (isinstance(value, float) and pd.isna(value))
            else json.dumps(list(value), ensure_ascii=False)
        )
    classified_export.to_parquet(
        assignments_path, index=False, compression="zstd"
    )
    written = [assignments_path]
    table_exports = [
        (composition.long_composition, "income_group_threat_composition.csv"),
        (composition.extreme_contrast, "high_minus_low_income_bootstrap.csv"),
    ]
    for frame, filename in table_exports:
        path = table_directory / filename
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def plot_income_composition(
    composition: CompositionAnalysis,
    *,
    income_groups: Sequence[str],
    threat_colors: dict[str, str],
    threat_names: dict[str, str],
    country_counts: pd.DataFrame | None = None,
    polygons: pd.DataFrame | None = None,
    contrast_excluded_threats: Sequence[str] = (),
    polygon_iso_column: str = "ISO_3",
    figsize: tuple[float, float] = (7.4, 7.2),
) -> plt.Figure:
    """Plot income composition, optionally with country evidence coverage.

    Supplying both ``country_counts`` and ``polygons`` builds the manuscript's
    three-panel composite (map, composition, contrast).  Omitting both retains
    the original two-panel figure for callers that have not migrated.  Passing
    only one map input is an error because it would silently produce an
    incomplete coverage panel.
    """
    if (country_counts is None) != (polygons is None):
        raise ValueError(
            "country_counts and polygons must either both be provided or both "
            "be omitted"
        )
    if country_counts is not None and polygons is not None:
        return _plot_income_composition_composite(
            composition,
            income_groups=income_groups,
            threat_colors=threat_colors,
            threat_names=threat_names,
            country_counts=country_counts,
            polygons=polygons,
            contrast_excluded_threats=contrast_excluded_threats,
            polygon_iso_column=polygon_iso_column,
            figsize=figsize,
        )

    figure, (panel_a, panel_b) = plt.subplots(
        1,
        2,
        figsize=(7.0, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.08]},
    )
    positions = np.arange(len(income_groups))
    bottoms = np.zeros(len(income_groups))
    change_magnitude = (
        composition.extreme_contrast.set_index("threat")["high_minus_low_pp"].abs()
    )
    original_position = {
        threat: position
        for position, threat in enumerate(composition.observed_threat_order)
    }
    plot_threat_order = sorted(
        composition.observed_threat_order,
        key=lambda threat: (
            -float(change_magnitude.loc[threat]),
            original_position[threat],
        ),
    )
    for threat in plot_threat_order:
        values = 100 * composition.matrix.loc[
            threat, income_groups
        ].to_numpy()
        panel_a.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.68,
            color=threat_colors[threat],
            edgecolor=BIODIVERSITY["paper"],
            linewidth=0.35,
            zorder=2,
        )
        bottoms += values
    panel_a.set(
        xticks=positions,
        xticklabels=[
            {
                "Low income": "Low\nincome",
                "Lower middle income": "Lower\nmiddle\nincome",
                "Upper middle income": "Upper\nmiddle\nincome",
                "High income": "High\nincome",
            }.get(group, group)
            for group in income_groups
        ],
        ylim=(0, 100),
        yticks=[0, 50, 100],
    )
    panel_a.set_ylabel(
        "Share of threat\nattributions (%)",
        fontsize=7.8,
    )

    plot_df = (
        composition.extreme_contrast.loc[
            lambda df: ~df["threat"].isin(contrast_excluded_threats)
        ]
        .sort_values("high_minus_low_pp")
        .reset_index(drop=True)
    )
    for position, row in plot_df.iterrows():
        panel_b.plot(
            [row["bootstrap_ci_low"], row["bootstrap_ci_high"]],
            [position, position],
            color=BIODIVERSITY["ink"],
            linewidth=0.75,
        )
        panel_b.scatter(
            row["high_minus_low_pp"],
            position,
            s=26,
            color=threat_colors[row["threat"]],
            edgecolor=BIODIVERSITY["paper"],
            linewidth=0.4,
            zorder=2,
        )
    panel_b.axvline(0, color=BIODIVERSITY["ink"], linewidth=0.7)
    panel_b.set(
        yticks=np.arange(len(plot_df)),
        yticklabels=[threat_names[threat] for threat in plot_df["threat"]],
    )
    panel_b.set_xlabel(
        "Difference in Threat Share (percentage points)\n"
        "High Income minus Low Income",
        fontsize=7.8,
    )
    for label, axis in zip("ab", [panel_a, panel_b]):
        axis.text(
            -0.075,
            1.02,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(BIODIVERSITY["neutral"])
        axis.spines[["left", "bottom"]].set_linewidth(0.6)
        axis.tick_params(
            length=0,
            colors=BIODIVERSITY["ink"],
            labelsize=6.8,
        )
    figure.legend(
        handles=[
            Patch(
                facecolor=threat_colors[threat],
                edgecolor=BIODIVERSITY["paper"],
                linewidth=0.35,
                label=threat_names[threat],
            )
            for threat in plot_threat_order
        ],
        loc="lower left",
        ncol=4,
        mode="expand",
        bbox_to_anchor=(0.08, 0.005, 0.88, 0.10),
        frameon=False,
        fontsize=6.0,
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=1.0,
        labelspacing=0.4,
        borderaxespad=0,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.98,
        bottom=0.24,
        wspace=0.68,
    )
    return figure


def _plot_income_composition_composite(
    composition: CompositionAnalysis,
    *,
    income_groups: Sequence[str],
    threat_colors: dict[str, str],
    threat_names: dict[str, str],
    country_counts: pd.DataFrame,
    polygons: pd.DataFrame,
    contrast_excluded_threats: Sequence[str],
    polygon_iso_column: str,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Build the map-plus-composition manuscript composite."""
    expected_count_columns = ["iso3", "country", "region", "record_count"]
    missing_count_columns = set(expected_count_columns).difference(
        country_counts.columns
    )
    if missing_count_columns:
        missing = ", ".join(sorted(missing_count_columns))
        raise ValueError(
            f"country_counts is missing required columns: {missing}"
        )
    if (
        country_counts["iso3"].isna().any()
        or country_counts["iso3"].duplicated().any()
    ):
        raise ValueError("country_counts.iso3 must be complete and unique")
    if polygon_iso_column not in polygons.columns:
        raise ValueError(
            f"polygons is missing ISO column {polygon_iso_column!r}"
        )

    record_counts = pd.to_numeric(
        country_counts["record_count"], errors="coerce"
    )
    if record_counts.isna().any() or record_counts.lt(0).any():
        raise ValueError(
            "country_counts.record_count must be non-negative numeric"
        )
    maximum_count = int(record_counts.max())
    if maximum_count < 1:
        raise ValueError(
            "country_counts must contain at least one publication"
        )

    figure = plt.figure(figsize=figsize)
    outer_grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.1],
        hspace=0.25,
    )
    map_grid = outer_grid[0].subgridspec(
        2,
        3,
        height_ratios=[1.0, 0.10],
        width_ratios=[0.28, 0.44, 0.28],
        hspace=0.02,
    )
    map_axis = figure.add_subplot(map_grid[0, :], label="panel_a_map")
    colorbar_axis = figure.add_subplot(
        map_grid[1, 1], label="panel_a_colorbar"
    )
    lower_grid = outer_grid[1].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.32],
        wspace=0.74,
    )
    composition_axis = figure.add_subplot(
        lower_grid[0, 0], label="panel_b_composition"
    )
    contrast_axis = figure.add_subplot(
        lower_grid[0, 1], label="panel_c_contrast"
    )

    map_counts = country_counts[["iso3", "record_count"]].copy()
    map_counts["record_count"] = record_counts.astype(int).to_numpy()
    mapped_polygons = polygons.merge(
        map_counts,
        left_on=polygon_iso_column,
        right_on="iso3",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    mapped_polygons["record_count"] = (
        mapped_polygons["record_count"].fillna(0).astype(int)
    )
    mapped_evidence = mapped_polygons.loc[
        mapped_polygons["record_count"].gt(0)
    ]
    count_cmap, count_norm, count_bins = build_count_scale(
        maximum_count,
        bins=GEO_COUNT_BINS,
        colors=GEO_COUNT_COLORS,
    )
    map_edge_color = BIODIVERSITY["neutral"]
    mapped_polygons.plot(
        ax=map_axis,
        color="#D9D9D9",
        edgecolor=map_edge_color,
        linewidth=0.15,
    )
    if not mapped_evidence.empty:
        mapped_evidence.plot(
            ax=map_axis,
            column="record_count",
            cmap=count_cmap,
            norm=count_norm,
            edgecolor=map_edge_color,
            linewidth=0.15,
        )
    map_axis.set_xlim(-170, 180)
    map_axis.set_ylim(-58, 84)
    map_axis.set_axis_off()
    add_class_colorbar(
        figure,
        colorbar_axis,
        count_cmap,
        count_norm,
        count_bins,
        "Articles per country",
    )

    excluded = set(contrast_excluded_threats)
    contrast_df = (
        composition.extreme_contrast.loc[
            lambda frame: ~frame["threat"].isin(excluded)
        ]
        .sort_values("high_minus_low_pp")
        .reset_index(drop=True)
    )
    residual_threat_order = [
        threat
        for threat in composition.observed_threat_order
        if threat in excluded
    ]
    plot_threat_order = [
        *contrast_df["threat"].tolist(),
        *residual_threat_order,
    ]

    positions = np.arange(len(income_groups))
    bottoms = np.zeros(len(income_groups))
    for threat in plot_threat_order:
        values = 100 * composition.matrix.loc[threat, income_groups].to_numpy()
        composition_axis.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.68,
            color=threat_colors[threat],
            edgecolor=BIODIVERSITY["paper"],
            linewidth=0.35,
            zorder=2,
        )
        bottoms += values

    group_publications = composition.group_counts.reindex(income_groups)[
        "unique_publications"
    ]
    if group_publications.isna().any():
        raise ValueError(
            "composition.group_counts lacks unique-publication counts for all "
            "income_groups"
        )
    group_abbreviations = {
        "Low income": "LIC",
        "Lower middle income": "LMIC",
        "Upper middle income": "UMIC",
        "High income": "HIC",
    }
    composition_axis.set(
        xticks=positions,
        xticklabels=[
            f"{group_abbreviations.get(group, group)}\n($n$ = {int(count):,})"
            for group, count in zip(
                income_groups, group_publications, strict=True
            )
        ],
        ylim=(0, 100),
        yticks=[0, 50, 100],
    )
    composition_axis.set_ylabel(
        "Share of threat attributions (%)",
        fontsize=7.8,
    )

    for position, row in contrast_df.iterrows():
        contrast_axis.plot(
            [row["bootstrap_ci_low"], row["bootstrap_ci_high"]],
            [position, position],
            color=BIODIVERSITY["ink"],
            linewidth=0.75,
        )
        contrast_axis.scatter(
            row["high_minus_low_pp"],
            position,
            s=26,
            color=threat_colors[row["threat"]],
            edgecolor=BIODIVERSITY["paper"],
            linewidth=0.4,
            zorder=2,
        )
    contrast_axis.axvline(0, color=BIODIVERSITY["ink"], linewidth=0.7)
    contrast_axis.set(
        yticks=np.arange(len(contrast_df)),
        yticklabels=[threat_names[threat] for threat in contrast_df["threat"]],
    )
    contrast_axis.set_xlabel(
        "Difference in threat share (percentage points)\n"
        "HIC minus LIC",
        fontsize=7.8,
    )

    legend_column_count = 4
    legend_display_order = [
        *reversed(contrast_df["threat"].tolist()),
        *residual_threat_order,
    ]
    # Matplotlib fills multi-column legends down columns. Interleave the
    # handles so the visible left-to-right row order runs from panel c's top
    # (most positive contrast) to its bottom, then ends with the residuals.
    legend_threat_order = [
        threat
        for column in range(legend_column_count)
        for threat in legend_display_order[column::legend_column_count]
    ]
    figure.legend(
        handles=[
            Patch(
                facecolor=threat_colors[threat],
                edgecolor=BIODIVERSITY["paper"],
                linewidth=0.35,
                label=(
                    "Other threats"
                    if threat == "Other Options"
                    else threat_names[threat]
                ),
            )
            for threat in legend_threat_order
        ],
        loc="lower left",
        ncol=legend_column_count,
        mode="expand",
        bbox_to_anchor=(0.08, 0.02, 0.88, 0.10),
        frameon=False,
        fontsize=6.0,
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=1.0,
        labelspacing=0.4,
        borderaxespad=0,
    )

    for label, axis, x_position in (
        ("b", composition_axis, -0.16),
        ("c", contrast_axis, -0.56),
    ):
        axis.text(
            x_position,
            1.02,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(BIODIVERSITY["neutral"])
        axis.spines[["left", "bottom"]].set_linewidth(0.6)
        axis.tick_params(
            length=0,
            colors=BIODIVERSITY["ink"],
            labelsize=6.8,
        )
    composition_axis.tick_params(axis="x", labelsize=6.1)
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.98,
        bottom=0.145,
    )
    figure.canvas.draw()
    panel_label_x = figure.transFigure.inverted().transform(
        composition_axis.transAxes.transform((-0.16, 0))
    )[0]
    panel_label_y = figure.transFigure.inverted().transform(
        map_axis.transAxes.transform((0, 1.02))
    )[1]
    figure.text(
        panel_label_x,
        panel_label_y,
        "a",
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )
    return figure

"""Prepare country and region summaries of geographic research attention.

The Result 05 notebook inherits an already country-complete publication base from
Result 02.  A publication linked to ``k`` resolved countries contributes ``1/k``
to each country, so country and region attention sum to 100 percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_helpers.analysis.geo.geography import _classify_country, load_crosswalk
from data_helpers.labels import parse_list_labels


_CROSSWALK_COLUMNS = ("iso3", "country", "region", "subregion")
_COUNTRY_COLUMNS = (
    "country",
    "iso3",
    "region",
    "subregion",
    "whole_publications",
    "fractional_publication_equivalent",
    "attention_share_pct",
)
_REGION_COLUMNS = (
    "region",
    "country_count",
    "whole_publications",
    "fractional_publication_equivalent",
    "attention_share_pct",
)


class GeographyAttentionError(ValueError):
    """Raised when the inherited publication base cannot be summarized safely."""


@dataclass(frozen=True)
class GeographyAttentionResult:
    """Compact Result 05 outputs and their reconciliation counts."""

    countries: pd.DataFrame
    regions: pd.DataFrame
    publication_count: int
    assignment_count: int


def _prepare_crosswalk(crosswalk: pd.DataFrame | None) -> pd.DataFrame:
    source = load_crosswalk() if crosswalk is None else crosswalk
    missing = set(_CROSSWALK_COLUMNS).difference(source.columns)
    if missing:
        raise GeographyAttentionError(
            f"IPBES crosswalk lacks required columns: {sorted(missing)}"
        )

    work = source.loc[:, list(_CROSSWALK_COLUMNS)].copy()
    for column in ("country", "region", "subregion"):
        work[column] = work[column].astype("string").str.strip()
        if work[column].isna().any() or work[column].eq("").any():
            raise GeographyAttentionError(
                f"IPBES crosswalk contains a missing or blank {column}."
            )

    work["iso3"] = work["iso3"].astype("string").fillna("").str.strip().str.upper()
    valid = work["iso3"].ne("")
    invalid_codes = valid & ~work["iso3"].str.fullmatch(r"[A-Z]{3}")
    if invalid_codes.any():
        bad = work.loc[invalid_codes, "iso3"].drop_duplicates().tolist()
        raise GeographyAttentionError(
            f"Nonblank country keys must be three ASCII letters; found {bad}."
        )

    work = work.loc[valid].copy()
    if work.empty:
        raise GeographyAttentionError("IPBES crosswalk contains no valid ISO3 keys.")
    if work["iso3"].duplicated().any():
        duplicate_codes = (
            work.loc[work["iso3"].duplicated(keep=False), "iso3"]
            .drop_duplicates()
            .tolist()
        )
        raise GeographyAttentionError(
            f"Each ISO3 must identify one IPBES place; duplicates: {duplicate_codes}."
        )
    return work.reset_index(drop=True)


def _prepare_assignments(
    evidence: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    required = {"UT", "pred_countries"}
    missing = required.difference(evidence.columns)
    if missing:
        raise GeographyAttentionError(
            f"Prepared evidence lacks required columns: {sorted(missing)}"
        )
    if evidence.empty:
        raise GeographyAttentionError("Prepared evidence cannot be empty.")
    if evidence["UT"].isna().any() or evidence["UT"].duplicated().any():
        raise GeographyAttentionError(
            "Prepared evidence must contain one non-missing row per UT."
        )

    mentions = evidence.loc[:, ["UT", "pred_countries"]].copy()
    mentions["token"] = mentions["pred_countries"].map(parse_list_labels)
    mentions = mentions[["UT", "token"]].explode("token").dropna(subset=["token"])

    valid_iso3 = set(mapping["iso3"])
    classified = mentions["token"].map(
        lambda token: _classify_country(token, valid_iso3)
    )
    mentions["bucket"] = classified.map(lambda pair: pair[0])
    mentions["iso3"] = classified.map(lambda pair: pair[1])

    unresolved = mentions.loc[mentions["bucket"].eq("unresolved"), "token"]
    if not unresolved.empty:
        examples = unresolved.astype(str).str.strip().value_counts().head(10).to_dict()
        raise GeographyAttentionError(
            "The inherited Result 02 base contains unresolved country tokens: "
            f"{examples}."
        )

    assignments = (
        mentions.loc[mentions["bucket"].eq("mapped"), ["UT", "iso3"]]
        .drop_duplicates(["UT", "iso3"])
        .reset_index(drop=True)
    )
    mapped_ids = set(assignments["UT"])
    missing_ids = set(evidence["UT"]).difference(mapped_ids)
    if missing_ids:
        raise GeographyAttentionError(
            "The inherited Result 02 base is not country-complete; "
            f"{len(missing_ids):,} publications have no mapped country."
        )

    assignments["mapped_country_count"] = assignments.groupby("UT")["iso3"].transform(
        "nunique"
    )
    assignments["fractional_weight"] = 1.0 / assignments["mapped_country_count"]
    assignments = assignments.merge(
        mapping,
        on="iso3",
        how="left",
        validate="many_to_one",
    )
    return assignments.sort_values(["UT", "iso3"], kind="stable").reset_index(drop=True)


def _validate_result(
    assignments: pd.DataFrame,
    countries: pd.DataFrame,
    regions: pd.DataFrame,
    *,
    publication_count: int,
) -> None:
    if assignments.duplicated(["UT", "iso3"]).any():
        raise GeographyAttentionError(
            "Country assignments must be unique by publication and ISO3."
        )
    per_publication = assignments.groupby("UT")["fractional_weight"].sum()
    if len(per_publication) != publication_count or not np.allclose(
        per_publication.to_numpy(), 1.0
    ):
        raise GeographyAttentionError(
            "Fractional country weights must sum to one per publication."
        )
    if not np.isclose(assignments["fractional_weight"].sum(), float(publication_count)):
        raise GeographyAttentionError(
            "Fractional publication equivalents do not reconcile to the base."
        )
    if not np.isclose(countries["attention_share_pct"].sum(), 100.0):
        raise GeographyAttentionError("Country attention does not sum to 100 percent.")
    if not np.isclose(regions["attention_share_pct"].sum(), 100.0):
        raise GeographyAttentionError("Region attention does not sum to 100 percent.")


def prepare_geography_attention(
    evidence: pd.DataFrame,
    *,
    crosswalk: pd.DataFrame | None = None,
) -> GeographyAttentionResult:
    """Build observed-country and regional fractional-attention summaries.

    The input must be the already filtered, country-complete Result 02 publication
    base. Special non-geographic tokens may accompany valid ISO3 codes, but an
    unresolved token or a publication without a mapped country fails visibly.
    """
    mapping = _prepare_crosswalk(crosswalk)
    assignments = _prepare_assignments(evidence, mapping)
    publication_count = evidence["UT"].nunique()

    countries = (
        assignments.groupby(
            ["country", "iso3", "region", "subregion"],
            observed=True,
            sort=False,
        )
        .agg(
            whole_publications=("UT", "nunique"),
            fractional_publication_equivalent=("fractional_weight", "sum"),
        )
        .reset_index()
    )
    countries["attention_share_pct"] = (
        countries["fractional_publication_equivalent"] / publication_count * 100.0
    )
    countries = (
        countries.sort_values(
            ["attention_share_pct", "country"],
            ascending=[False, True],
            kind="stable",
        )
        .loc[:, list(_COUNTRY_COLUMNS)]
        .reset_index(drop=True)
    )

    regions = (
        assignments.groupby("region", observed=True, sort=False)
        .agg(
            country_count=("iso3", "nunique"),
            whole_publications=("UT", "nunique"),
            fractional_publication_equivalent=("fractional_weight", "sum"),
        )
        .reset_index()
    )
    regions["attention_share_pct"] = (
        regions["fractional_publication_equivalent"] / publication_count * 100.0
    )
    regions = (
        regions.sort_values(
            ["attention_share_pct", "region"],
            ascending=[False, True],
            kind="stable",
        )
        .loc[:, list(_REGION_COLUMNS)]
        .reset_index(drop=True)
    )

    _validate_result(
        assignments,
        countries,
        regions,
        publication_count=publication_count,
    )
    return GeographyAttentionResult(
        countries=countries,
        regions=regions,
        publication_count=publication_count,
        assignment_count=len(assignments),
    )


__all__ = [
    "GeographyAttentionError",
    "GeographyAttentionResult",
    "prepare_geography_attention",
]

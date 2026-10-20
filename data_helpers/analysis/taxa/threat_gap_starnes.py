"""Region-level threatened-species benchmark from Starnes et al. (2026).

Compares each IPBES region's share of chordate evidence with its share of IUCN
threatened species. The benchmark is built directly from the source's IPBES
subregion counts, so no country-level matching is needed. On the evidence side,
:func:`pooled_regional_evidence` deduplicates to (article, region): an article
studying several countries in the same region counts once for that region.

Starnes, T. et al. (2026) *A new analysis of biodiversity and conservation knowledge
products to support environmental assessments*, Zenodo doi:10.5281/zenodo.18348560.
Data Table 1a gives IUCN Red List counts per IPBES subregion, built from
countries of occurrence. Its subregion names match this repository's crosswalk
(``checklists/mappings/ipbes_regions.json``) exactly; :func:`starnes_regional_threat_benchmark`
raises on any that do not. The counts cover all comprehensively assessed taxa (no
breakdown by taxonomic group is published), so the evidence side uses the broad
"Chordates" group.

Threatened = CR + EN + VU, computed from the category columns. Region totals are
the sum of their subregion counts, so a species occurring in several subregions of
one region is counted in each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class TaxaThreatGapStarnesError(ValueError):
    """Raised when the Starnes benchmark or the pooled-evidence join cannot be built."""


#: Category columns as published. CR+EN+VU is the primary "threatened" definition;
#: EX/EW/NT/LC/DD are read through but not summed into the headline count.
STARNES_CATEGORY_COLUMNS = (
    "EX_count", "EW_count", "CR_count", "EN_count", "VU_count",
    "NT_count", "LC_count", "DD_count",
)

#: Counts summed from subregions to regions.
REGIONAL_COUNT_COLUMNS = ("threatened_cr_en_vu",)


def load_starnes_subregion_counts(source: str | Path) -> pd.DataFrame:
    """Read Starnes Data Table 1a, one row per IPBES subregion.

    Drops the trailing blank row the source CSV carries, computes
    ``threatened_cr_en_vu`` from the category columns, and validates there is
    exactly one row per subregion before anything downstream can double-count one.
    """
    source = Path(source)
    if not source.exists():
        raise TaxaThreatGapStarnesError(f"Starnes source not found: {source}")

    frame = pd.read_csv(source)
    frame = frame.dropna(subset=["ipbes_region", "ipbes_subregion"]).copy()

    required = {"ipbes_region", "ipbes_subregion", *STARNES_CATEGORY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise TaxaThreatGapStarnesError(f"Starnes source is missing columns: {sorted(missing)}")

    duplicated = int(frame.duplicated(["ipbes_region", "ipbes_subregion"]).sum())
    if duplicated:
        raise TaxaThreatGapStarnesError(
            f"Starnes source has {duplicated:,} duplicate subregion rows; expected one each."
        )

    frame["threatened_cr_en_vu"] = frame[["CR_count", "EN_count", "VU_count"]].sum(axis=1)
    return frame.rename(
        columns={"ipbes_region": "starnes_region", "ipbes_subregion": "subregion"}
    ).reset_index(drop=True)


def starnes_regional_threat_benchmark(
    source: str | Path,
    crosswalk: pd.DataFrame,
    *,
    region_order: Iterable[str],
    count_columns: Iterable[str] = REGIONAL_COUNT_COLUMNS,
    primary_column: str = "threatened_cr_en_vu",
) -> pd.DataFrame:
    """Sum Starnes subregion counts to this repo's IPBES regions and score shares.

    Joins on **subregion name** against ``checklists/mappings/ipbes_regions.json``
    (via :func:`data_helpers.analysis.geo.geography.load_crosswalk`) rather than on
    Starnes' own region labels, which are spelled differently ("Asia-Pacific" vs.
    "Asia and the Pacific") but whose subregion names match this repo's crosswalk
    character-for-character — verified here, not assumed: an unmatched subregion or a
    region with zero matched subregions raises rather than silently producing a
    partial total. Antarctica is excluded because it carries no chordate evidence
    and is outside the manuscript's four-region structure.

    Every column in ``count_columns`` gets its own region-level sum and share column
    (``<column>_share_pct``, except the primary column, whose share is named
    ``threat_share_pct`` to match the schema :func:`combine_starnes_benchmark` and the
    taxonomic figure expect).
    """
    subregion_counts = load_starnes_subregion_counts(source)
    region_order = tuple(region_order)
    count_columns = tuple(count_columns)

    places = crosswalk[["region", "subregion"]].drop_duplicates()
    dup_subregion = places[places.duplicated("subregion", keep=False)]
    if not dup_subregion.empty:
        raise TaxaThreatGapStarnesError(
            f"Crosswalk subregion names are not unique across regions: "
            f"{sorted(dup_subregion['subregion'].unique())}"
        )

    merged = subregion_counts.merge(places, on="subregion", how="left", validate="one_to_one")
    unmatched = merged[merged["region"].isna()]
    if not unmatched.empty:
        raise TaxaThreatGapStarnesError(
            "Starnes subregions with no match in the IPBES crosswalk: "
            f"{sorted(unmatched['subregion'].unique())}"
        )

    scoped = merged[merged["region"].isin(region_order)].copy()
    missing_regions = set(region_order).difference(scoped["region"])
    if missing_regions:
        raise TaxaThreatGapStarnesError(f"No Starnes rows matched region(s): {sorted(missing_regions)}")

    grouped = scoped.groupby("region", as_index=False)[list(count_columns)].sum()
    for column in count_columns:
        share_name = "threat_share_pct" if column == primary_column else f"{column}_share_pct"
        grouped[share_name] = grouped[column] / grouped[column].sum() * 100.0
    grouped = grouped.rename(columns={primary_column: "threatened_species"})

    grouped = grouped.set_index("region").reindex(region_order).reset_index()
    grouped.attrs["n_subregions_matched"] = int(len(scoped))
    grouped.attrs["subregions_by_region"] = (
        scoped.groupby("region")["subregion"].apply(list).to_dict()
    )
    return grouped


def pooled_regional_evidence(
    included: pd.DataFrame,
    countries: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    group: str,
    region_order: Iterable[str],
    id_col: str = "UT",
    group_col: str = "benchmark_groups",
    country_col: str = "pred_countries",
    region_col: str = "region",
) -> pd.DataFrame:
    """Each region's pooled 2000-2025 share of one group's evidence — article-deduplicated.

    Deduplicates directly to (article, region) pairs: a paper studying Kenya, Tanzania and Uganda contributes
    one Africa occurrence, not three; a paper genuinely studying both an African and
    an Asia-Pacific country contributes once to each, which is why regional shares
    below are shares of *occurrences*, not of the article count.

    Uses the same dedup logic ``threat_gap.regional_evidence_trend`` already applies
    per year — this is that logic pooled across all years, with no year dimension.
    """
    missing = {id_col, group_col}.difference(included.columns)
    if missing:
        raise TaxaThreatGapStarnesError(f"Included frame is missing columns: {sorted(missing)}")
    missing = {id_col, country_col}.difference(countries.columns)
    if missing:
        raise TaxaThreatGapStarnesError(f"Country frame is missing columns: {sorted(missing)}")

    source = included[[id_col, group_col]].merge(countries, on=id_col, how="inner")
    source = source[source[group_col].map(lambda value: group in (value or ()))].copy()
    if source.empty:
        raise TaxaThreatGapStarnesError(f"No publications carry the group {group!r}.")

    source[country_col] = source[country_col].map(
        lambda value: list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    )
    long = source.explode(country_col).dropna(subset=[country_col])
    long[country_col] = long[country_col].astype(str).str.strip()

    places = crosswalk[["iso3", region_col]].drop_duplicates("iso3")
    long = long.merge(places, left_on=country_col, right_on="iso3", how="inner")

    region_order = tuple(region_order)
    scoped = long[long[region_col].isin(region_order)]
    region_occurrences = scoped.drop_duplicates([id_col, region_col])

    counted = (
        region_occurrences.groupby(region_col)[id_col]
        .nunique()
        .rename("n_publications")
        .reindex(region_order, fill_value=0)
        .reset_index()
    )
    total = float(counted["n_publications"].sum())
    if total <= 0:
        raise TaxaThreatGapStarnesError("No region-mapped occurrences for the requested group.")
    counted["evidence_share_pct"] = counted["n_publications"] / total * 100.0

    counted.attrs["n_articles_included"] = int(scoped[id_col].nunique())
    counted.attrs["n_region_occurrences"] = int(len(region_occurrences))
    counted.attrs["n_multi_region_articles"] = int(
        region_occurrences.groupby(id_col).size().gt(1).sum()
    )
    return counted


def combine_starnes_benchmark(
    evidence_regional: pd.DataFrame,
    threat_regional: pd.DataFrame,
    *,
    region_col: str = "region",
) -> pd.DataFrame:
    """Join the deduplicated evidence shares to the Starnes threat shares.

    Both frames are already one row per region — this is a label-preserving merge,
    not a filter, so it raises if either side is missing a region rather than
    silently dropping one.
    """
    missing = {region_col, "evidence_share_pct"}.difference(evidence_regional.columns)
    if missing:
        raise TaxaThreatGapStarnesError(f"Evidence frame is missing columns: {sorted(missing)}")
    missing = {region_col, "threat_share_pct"}.difference(threat_regional.columns)
    if missing:
        raise TaxaThreatGapStarnesError(f"Threat frame is missing columns: {sorted(missing)}")

    merged = evidence_regional.merge(threat_regional, on=region_col, how="inner", validate="one_to_one")
    if len(merged) != len(evidence_regional) or len(merged) != len(threat_regional):
        raise TaxaThreatGapStarnesError(
            "Evidence and threat regional frames do not cover the same set of regions."
        )

    merged["evidence_to_threat_ratio"] = merged["evidence_share_pct"] / merged["threat_share_pct"]
    merged["evidence_minus_threat_pp"] = merged["evidence_share_pct"] - merged["threat_share_pct"]
    return merged.sort_values("evidence_to_threat_ratio", ascending=False).reset_index(drop=True)


__all__ = [
    "REGIONAL_COUNT_COLUMNS",
    "STARNES_CATEGORY_COLUMNS",
    "TaxaThreatGapStarnesError",
    "combine_starnes_benchmark",
    "load_starnes_subregion_counts",
    "pooled_regional_evidence",
    "starnes_regional_threat_benchmark",
]

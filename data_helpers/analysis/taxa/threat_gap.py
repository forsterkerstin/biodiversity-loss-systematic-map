"""Regional trend of one taxon group's evidence against a threatened-species benchmark.

Supports the geographic comparison in ``notebooks/results/04_taxonomic_lens.ipynb``:
:func:`regional_evidence_trend` computes each IPBES region's annual share of a
group's evidence and its ratio to the region's share of threatened species (the
pooled benchmark from :mod:`threat_gap_starnes`), and
:func:`summarise_regional_trend` reduces it to start/end shares and slopes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class TaxaThreatGapError(ValueError):
    """Raised when the evidence/threat trend cannot be built as specified."""


def regional_evidence_trend(
    included: pd.DataFrame,
    countries: pd.DataFrame,
    regional: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    group: str,
    id_col: str = "UT",
    group_col: str = "benchmark_groups",
    country_col: str = "pred_countries",
    year_col: str = "publication_year",
    region_col: str = "region",
) -> pd.DataFrame:
    """Each region's share of one group's evidence, by publication year.

    The geographic counterpart of ``taxonomic_attention_trend``.

    Counting matches :func:`threat_gap_starnes.pooled_regional_evidence`: each
    publication counts once per region containing at least one of its study countries,
    so a yearly share and the pooled share are the same statistic at different
    resolutions. A publication naming two regions counts once in each, so shares are of
    region occurrences rather than of publications.

    The threat benchmark is a single snapshot and cannot vary by year; it is carried
    through as a constant so each year's ratio is against a fixed target.
    """
    for frame, columns, name in (
        (included, {id_col, group_col, year_col}, "Included"),
        (countries, {id_col, country_col}, "Country"),
        (regional, {region_col, "threat_share_pct"}, "Regional"),
    ):
        missing = columns.difference(frame.columns)
        if missing:
            raise TaxaThreatGapError(f"{name} frame is missing columns: {sorted(missing)}")

    source = included[[id_col, group_col, year_col]].merge(countries, on=id_col, how="inner")
    source = source[
        source[group_col].map(lambda value: group in (value or ()))
    ].copy()
    if source.empty:
        raise TaxaThreatGapError(f"No publications carry the group {group!r}.")

    source[country_col] = source[country_col].map(
        lambda value: list(value) if isinstance(value, (list, tuple, np.ndarray)) else []
    )
    long = source.explode(country_col).dropna(subset=[country_col])
    long[country_col] = long[country_col].astype(str).str.strip()

    places = crosswalk[["iso3", region_col]].drop_duplicates("iso3")
    long = long.merge(places, left_on=country_col, right_on="iso3", how="inner")
    # Unique document per (year, country), then summed within region -- the same
    # convention the pooled regional table uses.
    counted = (
        long.drop_duplicates([id_col, "iso3"])
        .groupby([year_col, region_col])[id_col]
        .nunique()
        .rename("n_publications")
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [sorted(counted[year_col].unique()), sorted(counted[region_col].unique())],
        names=[year_col, region_col],
    )
    counted = counted.set_index([year_col, region_col]).reindex(grid, fill_value=0).reset_index()

    totals = counted.groupby(year_col)["n_publications"].transform("sum")
    counted["evidence_share_pct"] = np.where(
        totals > 0, counted["n_publications"] / totals * 100.0, np.nan
    )
    counted["n_publications_year"] = totals
    benchmark = regional.set_index(region_col)["threat_share_pct"]
    counted["threat_share_pct"] = counted[region_col].map(benchmark)
    counted["evidence_to_threat_ratio"] = (
        counted["evidence_share_pct"] / counted["threat_share_pct"]
    )
    return counted.sort_values([year_col, region_col]).reset_index(drop=True)


def summarise_regional_trend(
    trend: pd.DataFrame,
    *,
    region_col: str = "region",
    year_col: str = "publication_year",
    share_col: str = "evidence_share_pct",
) -> pd.DataFrame:
    """Start/end shares, slope per decade, and start/end ratios for each region.

    Deliberately the same columns as ``summarise_attention_trend`` so the taxonomic and
    geographic persistence claims are read off identically shaped tables.
    """
    missing = {region_col, year_col, share_col, "threat_share_pct"}.difference(trend.columns)
    if missing:
        raise TaxaThreatGapError(f"Trend frame is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for region, block in trend.groupby(region_col):
        block = block.sort_values(year_col)
        years = block[year_col].to_numpy(dtype=float)
        shares = block[share_col].to_numpy(dtype=float)
        usable = ~np.isnan(shares)
        slope = float(np.polyfit(years[usable], shares[usable], 1)[0] * 10.0) if usable.sum() > 1 else np.nan
        threat = float(block["threat_share_pct"].iloc[0])
        rows.append(
            {
                region_col: region,
                "start_year": int(years[0]),
                "end_year": int(years[-1]),
                "start_share_pct": float(shares[0]),
                "end_share_pct": float(shares[-1]),
                "change_pp": float(shares[-1] - shares[0]),
                "slope_pp_per_decade": slope,
                "threat_share_pct": threat,
                "start_ratio": float(shares[0] / threat) if threat else np.nan,
                "end_ratio": float(shares[-1] / threat) if threat else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("end_ratio").reset_index(drop=True)


__all__ = ["TaxaThreatGapError", "regional_evidence_trend", "summarise_regional_trend"]

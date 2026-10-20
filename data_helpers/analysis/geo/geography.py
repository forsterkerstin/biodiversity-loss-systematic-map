"""Shared geographic analysis against the IPBES polygon reference.

The coding step emits three geography columns as JSON-serialized lists:

* ``pred_countries``   - ISO 3166-1 alpha-3 codes (e.g. ``["CHN", "USA"]``)
* ``pred_regions``     - IPBES region names (e.g. ``["Americas"]``)
* ``pred_subregions``  - IPBES sub-region names

This module flattens the canonical crosswalk in
``checklists/mappings/ipbes_regions.json`` (the same vocabulary used at
labelling time and in ``IPBES_Regions_Subregions2.shp``) and aggregates label
mentions into per-polygon counts - the "colsum" that a choropleth consumes.

Special label values are skipped but always reported, never dropped silently:

* ``[]``                        -> ``unclear`` (no geography assigned)
* ``["Not Applicable"]``        -> ``not_applicable``
* ``"Unclear"`` and variants    -> ``unclear``
* ``["All Regions"]`` / ``["All Countries"]`` -> ``aggregate`` (skipped)
* codes that are not valid ISO3 -> ``unresolved`` (listed in the audit)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_helpers.labels import parse_list_labels

REPO_ROOT = Path(__file__).resolve().parents[3]
CROSSWALK_PATH = REPO_ROOT / "checklists" / "mappings" / "ipbes_regions.json"
SHAPEFILE_PATH = (
    REPO_ROOT
    / "data"
    / "ipbes-polygons"
    / "ipbes_regions_subregions_shape_1.1"
    / "IPBES_Regions_Subregions2.shp"
)

# Label tokens that carry no mappable geography. Compared after normalisation.
_NOT_APPLICABLE = {"NOTAPPLICABLE", "NA", "NONE"}
_AGGREGATE = {"ALLCOUNTRIES", "ALLREGIONS", "ALLSUBREGIONS", "GLOBAL", "WORLDWIDE"}
_UNCLEAR = {
    "UNCLEAR", "UNC", "UNCL", "UNK", "UNKN", "UNKNOWN", "UNS", "UNR", "UNL", "UNP", "",
}

# Column -> crosswalk key that a level joins on.
_LEVEL_KEY = {"country": "iso3", "region": "region", "subregion": "subregion"}
_LEVEL_COL = {
    "country": "pred_countries",
    "region": "pred_regions",
    "subregion": "pred_subregions",
}


def _normalise(token: str) -> str:
    """Uppercase and strip everything but letters, for tolerant matching."""
    return re.sub(r"[^A-Za-z]", "", str(token)).upper()


def load_crosswalk() -> pd.DataFrame:
    """Flatten ``ipbes_regions.json`` to ``[iso3, country, region, subregion]``."""
    data = json.loads(CROSSWALK_PATH.read_text(encoding="utf-8"))
    rows = [
        {
            "iso3": entry["ISO_3166_alpha_3"],
            "country": entry["Country"],
            "region": region,
            "subregion": subregion,
        }
        for region, subregions in data.items()
        for subregion, entries in subregions.items()
        for entry in entries
    ]
    return pd.DataFrame(rows)


@dataclass
class GeoAudit:
    """Book-keeping for how label mentions were resolved. Printed, not dropped."""

    level: str
    records_total: int
    records_without_geography: int  # rows whose list was ``[]``
    bucket_mentions: dict[str, int] = field(default_factory=dict)
    unresolved_tokens: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"{self.level}: {self.records_total:,} records; "
            f"{self.records_without_geography:,} with no geography ([] -> unclear)",
        ]
        for bucket in ("mapped", "not_applicable", "unclear", "aggregate", "unresolved"):
            if bucket in self.bucket_mentions:
                parts.append(f"    {bucket:<15}: {self.bucket_mentions[bucket]:>8,} mentions")
        if self.unresolved_tokens:
            top = ", ".join(
                f"{tok} ({n})"
                for tok, n in sorted(
                    self.unresolved_tokens.items(), key=lambda kv: -kv[1]
                )[:15]
            )
            parts.append(f"    unresolved tokens: {top}")
        return "\n".join(parts)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"bucket": b, "mentions": n} for b, n in self.bucket_mentions.items()]
        )


def _classify_country(token: str, iso3_set: set[str]) -> tuple[str, str | None]:
    norm = _normalise(token)
    if norm in _NOT_APPLICABLE:
        return "not_applicable", None
    if norm in _AGGREGATE:
        return "aggregate", None
    if norm in _UNCLEAR:
        return "unclear", None
    if norm in iso3_set:
        return "mapped", norm
    return "unresolved", None


def _classify_name(token: str, name_set: set[str]) -> tuple[str, str | None]:
    norm = _normalise(token)
    if norm in _AGGREGATE:
        return "aggregate", None
    if norm in _UNCLEAR:
        return "unclear", None
    clean = str(token).strip()
    if clean in name_set:
        return "mapped", clean
    return "unresolved", None


def geo_counts(
    df: pd.DataFrame,
    *,
    level: str = "country",
    by: str | None = "pred_threat_l0",
    id_col: str = "UT",
    geo_col: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, GeoAudit]:
    """Count unique records per polygon, optionally split by a label column.

    Parameters
    ----------
    df : records to aggregate (one row per record, ``id_col`` is the record id).
    level : ``"country"`` | ``"region"`` | ``"subregion"``.
    by : list-valued label column to split on (e.g. ``pred_threat_l0``), or
        ``None`` for a single geography distribution.
    id_col : unique record identifier; counts use ``nunique`` of this.
    geo_col : override the geography column; defaults per ``level``.
    verbose : print the audit summary.

    Returns
    -------
    counts : long dataframe ``[<by>?, <key>, n, group_total, share]`` where
        ``key`` is ``iso3`` / ``region`` / ``subregion``. ``share`` is
        ``n / group_total``; multi-geography records let a group's shares sum
        above 1, matching the realm-matrix convention in the notebooks.
    audit : :class:`GeoAudit` describing skipped / unresolved mentions.
    """
    if level not in _LEVEL_KEY:
        raise ValueError(f"level must be one of {sorted(_LEVEL_KEY)}, got {level!r}")
    key = _LEVEL_KEY[level]
    geo_col = geo_col or _LEVEL_COL[level]
    crosswalk = load_crosswalk()

    work = df[[id_col, geo_col] + ([by] if by else [])].copy()
    work["_geo"] = work[geo_col].map(parse_list_labels)

    audit = GeoAudit(
        level=level,
        records_total=work[id_col].nunique(),
        records_without_geography=int((work["_geo"].map(len) == 0).sum()),
    )

    if level == "country":
        valid = set(crosswalk["iso3"].map(_normalise))
        classify = lambda tok: _classify_country(tok, valid)  # noqa: E731
    else:
        valid = set(crosswalk[key].dropna().astype(str))
        classify = lambda tok: _classify_name(tok, valid)  # noqa: E731

    exploded = work.explode("_geo").dropna(subset=["_geo"])
    if by:
        exploded["_by"] = exploded[by].map(parse_list_labels)
        exploded = exploded.explode("_by").dropna(subset=["_by"])
        exploded["_by"] = exploded["_by"].astype(str).str.strip()
        exploded = exploded[exploded["_by"].ne("")]

    classified = exploded["_geo"].map(classify)
    exploded["_bucket"] = classified.map(lambda pair: pair[0])
    exploded["_key"] = classified.map(lambda pair: pair[1])

    audit.bucket_mentions = exploded["_bucket"].value_counts().to_dict()
    audit.unresolved_tokens = (
        exploded.loc[exploded["_bucket"].eq("unresolved"), "_geo"]
        .astype(str).str.strip().value_counts().to_dict()
    )

    mapped = exploded[exploded["_bucket"].eq("mapped")].copy()
    mapped = mapped.rename(columns={"_key": key})

    group_cols = (["_by"] if by else []) + [key]
    mapped = mapped.drop_duplicates([id_col, *group_cols])
    counts = (
        mapped.groupby(group_cols, observed=True)[id_col]
        .nunique().rename("n").reset_index()
    )

    total_cols = ["_by"] if by else []
    if total_cols:
        totals = mapped.groupby(total_cols, observed=True)[id_col].nunique()
        counts["group_total"] = counts["_by"].map(totals)
    else:
        counts["group_total"] = mapped[id_col].nunique()
    counts["share"] = counts["n"] / counts["group_total"]

    if by:
        counts = counts.rename(columns={"_by": by})

    if verbose:
        print(audit.summary())
    return counts, audit


# Place key produced for each aggregation level (all geolocated via pred_countries).
_LQ_PLACE_KEY = {"country": "iso3", "subregion": "subregion", "region": "region"}


def load_polygons(
    simplify_tolerance: float | None = 0.2,
    preserve_topology: bool = False,
    dissolve_by: str | None = None,
    use_cache: bool = True,
) -> "pd.DataFrame":
    """Read the IPBES polygons; optionally simplify geometry for fast plotting.

    ``simplify_tolerance`` is in degrees (EPSG:4326). The full-resolution source
    has ~33M vertices across many tiny island rings, which makes a small-multiple
    world map render for minutes and export as a huge PDF.

    ``preserve_topology=False`` (the default) collapses rings smaller than the
    tolerance entirely: at ``0.2`` (~20 km) this cuts the geometry to ~12k
    vertices, so a 12-panel grid renders in seconds and the PDF is small. The
    cost is that ~70 sub-tolerance island states drop out — invisible at
    thumbnail scale. Pass ``preserve_topology=True`` to keep every country
    (heavier: ~500k vertices, slow) when the map is large enough to need them.

    ``dissolve_by`` merges the country polygons into coarser units by an
    attribute column — ``"Sub_Region"`` (17 IPBES subregions) or ``"Region"``
    (5 regions) — returning ``[<dissolve_by>, geometry]``. Dissolving also unions
    away the internal country borders, so the coarser map is clean.

    The simplified per-country layer is cached to a small GeoParquet next to the
    shapefile, keyed by tolerance and topology mode; dissolving happens on the
    cached layer (fast). Returns a ``geopandas.GeoDataFrame`` (the import is local
    so the rest of this module carries no geopandas dependency).
    """
    import geopandas as gpd

    if not simplify_tolerance:
        gdf = gpd.read_file(SHAPEFILE_PATH)
    else:
        topo_tag = "topo" if preserve_topology else "notopo"
        cache_path = SHAPEFILE_PATH.with_name(
            f"IPBES_simplified_{simplify_tolerance:g}_{topo_tag}.parquet"
        )
        if use_cache and cache_path.exists():
            gdf = gpd.read_parquet(cache_path)
        else:
            gdf = gpd.read_file(SHAPEFILE_PATH)
            gdf["geometry"] = gdf["geometry"].simplify(
                simplify_tolerance, preserve_topology=preserve_topology
            )
            # Drop rows whose geometry fully collapsed so plotting stays clean.
            gdf = gdf.loc[~(gdf.geometry.is_empty | gdf.geometry.isna())].copy()
            if use_cache:
                gdf.to_parquet(cache_path)

    if dissolve_by:
        gdf = gdf.dissolve(by=dissolve_by, as_index=False)[[dissolve_by, "geometry"]]
    return gdf

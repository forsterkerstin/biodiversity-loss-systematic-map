"""Rank-aligned taxonomic hierarchy diagnostics for Result 04.

This module builds the Kingdom -> Phylum -> Class -> Order comparison behind the
taxonomic-hierarchy sunburst figure. It keeps the article side article-balanced,
streams the same accepted species-rank GBIF source used by the broad benchmark,
and exposes taxonomy-version overlap before any comparison is made.

Research matches were resolved through GBIF's v2 match API, whose upper
classification can differ from the repository's 28-Aug-2023 GBIF backbone
snapshot.  ``canonical_research_path`` bridges only the unambiguous kingdom
changes (bacterial/archaeal domains and viral realms).  Lower-rank names are
not guessed or cross-walked: comparisons retain exact common paths and report
how much evidence that removes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

RANKS = ("kingdom", "phylum", "class", "order")
GBIF_COLUMNS = (
    "taxonID",
    "taxonRank",
    "taxonomicStatus",
    "kingdom",
    "phylum",
    "class",
    "order",
)


class TaxaHierarchyError(ValueError):
    """Raised when hierarchy evidence violates its counting contract."""


@dataclass(frozen=True)
class RankHierarchyEvidence:
    """Rank-wise research attributions, GBIF counts, and compatibility audits."""

    research_attributions: pd.DataFrame
    gbif_counts: pd.DataFrame
    resolution_audit: pd.DataFrame
    alignment_audit: pd.DataFrame
    gbif_source_audit: pd.DataFrame


@dataclass(frozen=True)
class SunburstHierarchy:
    """Selected unbalanced hierarchy and the rules/audits that produced it."""

    nodes: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True)
class SunburstGbifBenchmark:
    """Sunburst nodes rebased to independent GBIF counts and their display map."""

    nodes: pd.DataFrame
    assignments: pd.DataFrame
    audit: pd.DataFrame


DEFAULT_SUNBURST_ROOT_GROUPS: dict[str, tuple[str, ...]] = {
    "Animalia": ("Chordates", "Non-chordate animals"),
    "Plantae": ("Plants",),
    "Fungi": ("Fungi",),
    "Other kingdoms": ("Other",),
}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lineage_rank_values(lineage: Any) -> dict[str, str]:
    """Return one unambiguous name per lineage rank."""
    values: defaultdict[str, set[str]] = defaultdict(set)
    if lineage is None:
        return {}
    for node in lineage:
        rank = (_clean(node.get("rank")) or "").upper()
        name = _clean(node.get("name"))
        if rank and name:
            values[rank].add(name)
    ambiguous = {rank: names for rank, names in values.items() if len(names) > 1}
    if ambiguous:
        raise TaxaHierarchyError(
            "One accepted lineage contains multiple names at the same rank: "
            f"{ambiguous}"
        )
    return {rank: next(iter(names)) for rank, names in values.items()}


def canonical_research_path(match: Mapping[str, Any]) -> tuple[str | None, ...]:
    """Extract the fixed-rank path, applying only defensible kingdom bridges.

    The current API represents bacterial and archaeal kingdoms below their
    stable domains and virus groups below a realm.  The 2023 benchmark stores
    those three containers as ``Bacteria``, ``Archaea``, and ``Viruses`` in its
    kingdom column.  No phylum/class rename is attempted.
    """
    ranks = _lineage_rank_values(match.get("lineage"))
    kingdom = ranks.get("KINGDOM")
    domain = ranks.get("DOMAIN")
    if domain in {"Bacteria", "Archaea"}:
        kingdom = domain
    elif ranks.get("REALM") is not None:
        kingdom = "Viruses"
    return (
        kingdom,
        ranks.get("PHYLUM"),
        ranks.get("CLASS"),
        ranks.get("ORDER"),
    )


def _path_columns(rank: str) -> tuple[str, ...]:
    if rank not in RANKS:
        raise TaxaHierarchyError(f"Unsupported rank: {rank!r}.")
    return RANKS[: RANKS.index(rank) + 1]


def build_research_rank_attributions(
    matches: pd.DataFrame,
    *,
    publication_ids: Iterable[str],
    benchmark_groups: Iterable[str],
) -> pd.DataFrame:
    """Build unique, article-fractional paths independently at each rank.

    Duplicate taxon items resolving to the same path within a publication are
    removed first.  At each rank, every publication with at least one complete
    path contributes total weight one across its distinct paths.
    """
    required = {"UT", "taxa_matches"}
    missing = required.difference(matches.columns)
    if missing:
        raise TaxaHierarchyError(
            f"Prepared taxa matches lack required columns: {sorted(missing)}"
        )
    if matches["UT"].isna().any() or not matches["UT"].is_unique:
        raise TaxaHierarchyError("Prepared taxa matches must be one row per UT.")

    wanted = {str(value) for value in publication_ids}
    groups = set(benchmark_groups)
    rows: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for ut, items in zip(matches["UT"], matches["taxa_matches"]):
        ut = str(ut)
        if ut not in wanted:
            continue
        observed_ids.add(ut)
        paths_by_rank: dict[str, set[tuple[str, ...]]] = {
            rank: set() for rank in RANKS
        }
        for match in (() if items is None else items):
            if not bool(match.get("broad_group_eligible")):
                continue
            if match.get("broad_group") not in groups:
                continue
            full_path = canonical_research_path(match)
            for rank in RANKS:
                depth = len(_path_columns(rank))
                path = full_path[:depth]
                if all(path):
                    paths_by_rank[rank].add(tuple(str(value) for value in path))

        for rank, paths in paths_by_rank.items():
            if not paths:
                continue
            weight = 1.0 / len(paths)
            for path in sorted(paths):
                record: dict[str, Any] = {
                    "rank": rank,
                    "UT": ut,
                    "research_weight": weight,
                    "n_article_paths": len(paths),
                    **{rank_name: None for rank_name in RANKS},
                }
                record.update(dict(zip(_path_columns(rank), path)))
                rows.append(record)

    missing_ids = wanted.difference(observed_ids)
    if missing_ids:
        raise TaxaHierarchyError(
            "Prepared match artifact does not cover every requested publication: "
            f"missing={len(missing_ids):,}."
        )
    columns = [
        "rank",
        "UT",
        *RANKS,
        "research_weight",
        "n_article_paths",
    ]
    attributions = pd.DataFrame(rows, columns=columns)
    if attributions.empty:
        raise TaxaHierarchyError("No rank-resolved research attribution remains.")
    totals = attributions.groupby(["rank", "UT"])["research_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0):
        raise TaxaHierarchyError(
            "Research path weights do not sum to one per publication and rank."
        )
    if attributions.duplicated(["rank", "UT", *RANKS]).any():
        raise TaxaHierarchyError("Duplicate article x rank path survived de-duplication.")
    return attributions


def stream_gbif_rank_counts(
    gbif_path: str | Path,
    *,
    block_size: int = 16 << 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count accepted species-rank GBIF rows at each complete fixed-rank path."""
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as csv
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise TaxaHierarchyError(
            "PyArrow is required to stream the GBIF hierarchy benchmark."
        ) from exc

    gbif_path = Path(gbif_path)
    delimiter = "\t" if gbif_path.suffix.casefold() in {".tsv", ".txt"} else ","
    reader = csv.open_csv(
        gbif_path,
        read_options=csv.ReadOptions(block_size=block_size, use_threads=False),
        parse_options=csv.ParseOptions(
            delimiter=delimiter,
            newlines_in_values=True,
        ),
        convert_options=csv.ConvertOptions(
            include_columns=list(GBIF_COLUMNS),
            column_types={column: pa.string() for column in GBIF_COLUMNS},
            strings_can_be_null=True,
        ),
    )
    total_rows = 0
    species_rows = 0
    accepted_rows = 0
    missing_taxon_id = 0
    counts: dict[str, Counter[tuple[str, ...]]] = {
        rank: Counter() for rank in RANKS
    }
    for batch in reader:
        total_rows += batch.num_rows
        ranks = pc.utf8_lower(batch["taxonRank"])
        statuses = pc.utf8_lower(batch["taxonomicStatus"])
        species_mask = pc.fill_null(pc.equal(ranks, "species"), False)
        species_rows += int(pc.sum(pc.cast(species_mask, pa.int64())).as_py())
        accepted_mask = pc.and_(
            species_mask,
            pc.fill_null(pc.equal(statuses, "accepted"), False),
        )
        accepted = batch.filter(accepted_mask)
        accepted_rows += accepted.num_rows
        if not accepted.num_rows:
            continue
        missing_taxon_id += int(
            pc.sum(pc.cast(pc.is_null(accepted["taxonID"]), pa.int64())).as_py()
        )
        lineage = accepted.select(list(RANKS)).to_pandas()
        for raw_path, count in lineage.value_counts(dropna=False).items():
            full_path = tuple(
                None if pd.isna(value) else _clean(value) for value in raw_path
            )
            for rank in RANKS:
                depth = len(_path_columns(rank))
                path = full_path[:depth]
                if all(path):
                    counts[rank][tuple(str(value) for value in path)] += int(count)

    rows: list[dict[str, Any]] = []
    for rank in RANKS:
        for path, count in sorted(counts[rank].items()):
            record: dict[str, Any] = {
                "rank": rank,
                **{rank_name: None for rank_name in RANKS},
                "gbif_described_species_count": int(count),
            }
            record.update(dict(zip(_path_columns(rank), path)))
            rows.append(record)
    hierarchy = pd.DataFrame(rows)
    audit = pd.DataFrame(
        [
            {"metric": "GBIF backbone rows scanned", "value": total_rows},
            {"metric": "Species-rank rows", "value": species_rows},
            {"metric": "Accepted species-rank rows", "value": accepted_rows},
            {
                "metric": "Accepted species rows missing taxonID",
                "value": missing_taxon_id,
            },
        ]
    )
    for rank in RANKS:
        retained = int(sum(counts[rank].values()))
        audit = pd.concat(
            [
                audit,
                pd.DataFrame(
                    [
                        {
                            "metric": f"Accepted species complete through {rank}",
                            "value": retained,
                        },
                        {
                            "metric": f"Unique {rank} paths",
                            "value": len(counts[rank]),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
    if accepted_rows <= 0:
        raise TaxaHierarchyError("GBIF source contains no accepted species-rank rows.")
    if int(counts["kingdom"].total()) > accepted_rows:
        raise TaxaHierarchyError("GBIF kingdom counts exceed accepted species rows.")
    return hierarchy, audit


def build_rank_audits(
    research_attributions: pd.DataFrame,
    gbif_counts: pd.DataFrame,
    *,
    n_anchor_publications: int,
    n_accepted_gbif_species: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report rank resolution and exact cross-snapshot path compatibility."""
    resolution_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    previous_research = n_anchor_publications
    previous_gbif = n_accepted_gbif_species
    for rank in RANKS:
        columns = list(_path_columns(rank))
        research = research_attributions.loc[
            research_attributions["rank"].eq(rank)
        ]
        gbif = gbif_counts.loc[gbif_counts["rank"].eq(rank)]
        n_research = int(research["UT"].nunique())
        n_gbif = int(gbif["gbif_described_species_count"].sum())
        resolution_rows.append(
            {
                "rank": rank,
                "research_resolved_publications": n_research,
                "research_share_of_anchor_pct": (
                    n_research / n_anchor_publications * 100.0
                ),
                "research_newly_unresolved_from_previous_rank": (
                    previous_research - n_research
                ),
                "gbif_resolved_species": n_gbif,
                "gbif_share_of_accepted_species_pct": (
                    n_gbif / n_accepted_gbif_species * 100.0
                ),
                "gbif_newly_unresolved_from_previous_rank": previous_gbif - n_gbif,
            }
        )
        previous_research = n_research
        previous_gbif = n_gbif

        research_paths = research.groupby(columns, dropna=False)[
            "research_weight"
        ].sum()
        gbif_paths = gbif.set_index(columns)["gbif_described_species_count"]
        common = research_paths.index.intersection(gbif_paths.index)
        common_research = float(research_paths.reindex(common).sum())
        common_gbif = int(gbif_paths.reindex(common).sum())
        common_research_rows = research.merge(
            pd.DataFrame(list(common), columns=columns),
            on=columns,
            how="inner",
        )
        alignment_rows.append(
            {
                "rank": rank,
                "n_research_paths": len(research_paths),
                "n_gbif_paths": len(gbif_paths),
                "n_exact_common_paths": len(common),
                "research_attention_on_common_paths_pct": (
                    common_research / research["research_weight"].sum() * 100.0
                ),
                "gbif_species_on_common_paths_pct": (
                    common_gbif / gbif_paths.sum() * 100.0
                ),
                "research_publications_with_common_path": int(
                    common_research_rows["UT"].nunique()
                ),
            }
        )
    return pd.DataFrame(resolution_rows), pd.DataFrame(alignment_rows)


def build_rank_hierarchy_evidence(
    matches: pd.DataFrame,
    gbif_path: str | Path,
    *,
    publication_ids: Iterable[str],
    benchmark_groups: Iterable[str],
) -> RankHierarchyEvidence:
    """Build both sides once and return the rank-resolution compatibility audit."""
    publication_ids = tuple(str(value) for value in publication_ids)
    research = build_research_rank_attributions(
        matches,
        publication_ids=publication_ids,
        benchmark_groups=benchmark_groups,
    )
    gbif, source_audit = stream_gbif_rank_counts(gbif_path)
    accepted = int(
        source_audit.loc[
            source_audit["metric"].eq("Accepted species-rank rows"), "value"
        ].iloc[0]
    )
    resolution, alignment = build_rank_audits(
        research,
        gbif,
        n_anchor_publications=len(publication_ids),
        n_accepted_gbif_species=accepted,
    )
    return RankHierarchyEvidence(
        research_attributions=research,
        gbif_counts=gbif,
        resolution_audit=resolution,
        alignment_audit=alignment,
        gbif_source_audit=source_audit,
    )


def compare_exact_rank_paths(
    research_attributions: pd.DataFrame,
    gbif_counts: pd.DataFrame,
    *,
    rank: str = "class",
) -> pd.DataFrame:
    """Compare article-balanced attention on exact paths common to both snapshots.

    After exact-path filtering, each retained publication is rebalanced across
    only its retained paths so it again contributes total weight one.  The GBIF
    denominator is likewise renormalized over exactly those common paths.
    """
    columns = list(_path_columns(rank))
    research = research_attributions.loc[
        research_attributions["rank"].eq(rank), ["UT", *columns]
    ].drop_duplicates()
    gbif = gbif_counts.loc[
        gbif_counts["rank"].eq(rank),
        [*columns, "gbif_described_species_count"],
    ]
    common = research[columns].drop_duplicates().merge(
        gbif[columns].drop_duplicates(), on=columns, how="inner"
    )
    retained = research.merge(common, on=columns, how="inner")
    if retained.empty:
        raise TaxaHierarchyError(f"No exact common {rank} path remains.")
    retained["research_weight"] = 1.0 / retained.groupby("UT")["UT"].transform(
        "size"
    )
    totals = retained.groupby("UT")["research_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0):
        raise TaxaHierarchyError("Retained article weights do not sum to one.")

    research_summary = (
        retained.groupby(columns, as_index=False)
        .agg(
            research_count=("research_weight", "sum"),
            n_unique_publications=("UT", "nunique"),
        )
    )
    comparison = research_summary.merge(gbif, on=columns, how="inner")
    research_total = float(comparison["research_count"].sum())
    gbif_total = int(comparison["gbif_described_species_count"].sum())
    comparison["research_share_pct"] = (
        comparison["research_count"] / research_total * 100.0
    )
    comparison["gbif_described_species_share_pct"] = (
        comparison["gbif_described_species_count"] / gbif_total * 100.0
    )
    comparison["research_minus_gbif_pp"] = (
        comparison["research_share_pct"]
        - comparison["gbif_described_species_share_pct"]
    )
    comparison["research_to_gbif_ratio"] = (
        comparison["research_share_pct"]
        / comparison["gbif_described_species_share_pct"]
    )
    comparison["comparison_research_publications"] = int(retained["UT"].nunique())
    comparison["comparison_gbif_species"] = gbif_total
    if not np.isclose(comparison["research_share_pct"].sum(), 100.0):
        raise TaxaHierarchyError("Research comparison shares do not sum to 100%.")
    if not np.isclose(
        comparison["gbif_described_species_share_pct"].sum(), 100.0
    ):
        raise TaxaHierarchyError("GBIF comparison shares do not sum to 100%.")
    return comparison.sort_values(
        "gbif_described_species_count", ascending=False, ignore_index=True
    )


def publications_on_common_paths(
    research_attributions: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    rank: str = "order",
) -> pd.Index:
    """Return the publications retained by an exact-path comparison."""
    columns = list(_path_columns(rank))
    research = research_attributions.loc[
        research_attributions["rank"].eq(rank), ["UT", *columns]
    ]
    retained = research.merge(
        comparison[columns].drop_duplicates(), on=columns, how="inner"
    )
    return pd.Index(retained["UT"].unique(), name="UT")


def distinct_path_taxa(comparison: pd.DataFrame, *, rank: str = "order") -> pd.DataFrame:
    """Count the distinct taxa at each rank covered by the compared paths.

    A taxon is identified by its full path from kingdom, so a name reused under
    two parents counts twice, as it does in the comparison itself.
    """
    columns = list(_path_columns(rank))
    rows = [
        {"rank": level, "n_taxa": int(len(comparison[columns[: depth + 1]].drop_duplicates()))}
        for depth, level in enumerate(columns)
    ]
    return pd.DataFrame(rows)


def _label_values(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    items = [value] if isinstance(value, str) else list(value)
    return [text for text in (_clean(item) for item in items) if text]


def distinct_publication_labels(
    publications: pd.DataFrame,
    label_columns: Mapping[str, str],
    *,
    publication_ids: Iterable[str] | None = None,
    excluded_labels: Iterable[str] = ("Not Applicable", "Unclear"),
) -> pd.DataFrame:
    """Count distinct coded labels per rank over a set of publications.

    ``label_columns`` maps a rank name to a list-valued column of
    ``publications`` (for example ``{"species": "taxa_species_labels"}``).
    """
    frame = publications
    if publication_ids is not None:
        frame = frame.loc[frame["UT"].isin(set(publication_ids))]
    excluded = {label.casefold() for label in excluded_labels}
    rows = []
    for rank, column in label_columns.items():
        labels = {
            label
            for value in frame[column]
            for label in _label_values(value)
            if label.casefold() not in excluded
        }
        rows.append(
            {
                "rank": rank,
                "n_taxa": len(labels),
                "n_publications": int(frame["UT"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_research_attention_sunburst(
    order_comparison: pd.DataFrame,
    *,
    major_kingdoms: Iterable[str] = ("Animalia", "Plantae", "Fungi"),
    terminal_kingdoms: Iterable[str] = (),
    phylum_min_within_parent_pct: float = 2.0,
    class_min_within_parent_pct: float = 2.0,
    order_max_children: int = 6,
    order_max_other_within_parent_pct: float = 5.0,
    relaxed_order_expansion_min_class_research_share_pct: float | None = None,
    order_named_min_global_share_pct: float = 1.0,
) -> SunburstHierarchy:
    """Select an unbalanced research-angle hierarchy from exact Order paths.

    One Order-aligned comparison universe is aggregated upward, so every
    displayed child reconciles exactly with its parent on both measures.
    Kingdoms outside ``major_kingdoms`` and configured ``terminal_kingdoms``
    stop at Kingdom; below-threshold Phyla/Classes become terminal
    ``Remaining`` aggregates.  A named Class expands to Order only when at most
    ``order_max_children`` named Orders can leave no more than the configured
    residual share on *either* measure.  Otherwise that Class remains a named
    terminal node unless it reaches the optional relaxed-expansion threshold.
    In every expanded Class, Orders reaching the configured global share on
    either measure are named and all other Orders—including any compact-set
    residual—are combined into one terminal ``Remaining orders`` node. A Class
    with no globally nameable Order remains terminal at Class.
    """
    required = {
        *RANKS,
        "research_count",
        "gbif_described_species_count",
    }
    missing = required.difference(order_comparison.columns)
    if missing:
        raise TaxaHierarchyError(
            f"Order comparison lacks required columns: {sorted(missing)}"
        )
    if order_comparison.empty:
        raise TaxaHierarchyError("Order comparison is empty.")
    if order_comparison[list(RANKS)].isna().any().any():
        raise TaxaHierarchyError("Order comparison contains an incomplete path.")
    if min(
        phylum_min_within_parent_pct,
        class_min_within_parent_pct,
        order_max_other_within_parent_pct,
        order_named_min_global_share_pct,
    ) < 0:
        raise TaxaHierarchyError("Sunburst share thresholds cannot be negative.")
    if (
        relaxed_order_expansion_min_class_research_share_pct is not None
        and not 0
        <= relaxed_order_expansion_min_class_research_share_pct
        <= 100
    ):
        raise TaxaHierarchyError(
            "Relaxed Class expansion threshold must be in [0, 100]."
        )
    if order_named_min_global_share_pct > 100:
        raise TaxaHierarchyError(
            "Order naming threshold cannot exceed 100%."
        )
    if order_max_other_within_parent_pct >= 100:
        raise TaxaHierarchyError("Order residual threshold must be below 100%.")
    if order_max_children <= 0:
        raise TaxaHierarchyError("Order child budget must be positive.")

    work = order_comparison.copy()
    major_kingdoms = tuple(str(value) for value in major_kingdoms)
    if not major_kingdoms or len(major_kingdoms) != len(set(major_kingdoms)):
        raise TaxaHierarchyError("Major Kingdoms must be non-empty and unique.")
    absent = set(major_kingdoms).difference(work["kingdom"])
    if absent:
        raise TaxaHierarchyError(
            f"Configured major Kingdoms are absent from Order comparison: {sorted(absent)}"
        )
    terminal_kingdoms = tuple(str(value) for value in terminal_kingdoms)
    if len(terminal_kingdoms) != len(set(terminal_kingdoms)):
        raise TaxaHierarchyError("Terminal Kingdoms must be unique.")
    invalid_terminal = set(terminal_kingdoms).difference(major_kingdoms)
    if invalid_terminal:
        raise TaxaHierarchyError(
            "Terminal Kingdoms must also be configured major Kingdoms: "
            f"{sorted(invalid_terminal)}"
        )

    research_total = float(work["research_count"].sum())
    gbif_total = int(work["gbif_described_species_count"].sum())
    if research_total <= 0 or gbif_total <= 0:
        raise TaxaHierarchyError("Sunburst comparison totals must be positive.")

    def child_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        summary = (
            frame.groupby(column, as_index=False, sort=False)
            .agg(
                research_count=("research_count", "sum"),
                gbif_described_species_count=(
                    "gbif_described_species_count",
                    "sum",
                ),
                source_order_paths=("order", "size"),
            )
            .rename(columns={column: "taxon"})
        )
        summary["research_within_parent_pct"] = (
            summary["research_count"] / frame["research_count"].sum() * 100.0
        )
        summary["gbif_within_parent_pct"] = (
            summary["gbif_described_species_count"]
            / frame["gbif_described_species_count"].sum()
            * 100.0
        )
        summary["selection_score"] = summary[
            ["research_within_parent_pct", "gbif_within_parent_pct"]
        ].max(axis=1)
        return summary.sort_values(
            ["research_count", "selection_score", "taxon"],
            ascending=[False, False, True],
            kind="stable",
            ignore_index=True,
        )

    def threshold_labels(
        frame: pd.DataFrame, column: str, threshold: float
    ) -> set[str]:
        summary = child_summary(frame, column)
        retained = summary.loc[
            summary["research_within_parent_pct"].ge(threshold)
            | summary["gbif_within_parent_pct"].ge(threshold),
            "taxon",
        ]
        if retained.empty:
            retained = summary.loc[[0], "taxon"]
        return {str(value) for value in retained}

    def compact_order_labels(frame: pd.DataFrame) -> set[str] | None:
        summary = child_summary(frame, "order").sort_values(
            ["selection_score", "research_count", "taxon"],
            ascending=[False, False, True],
            kind="stable",
            ignore_index=True,
        )
        selected: list[str] = []
        for row in summary.head(order_max_children).itertuples(index=False):
            selected.append(str(row.taxon))
            selected_rows = summary["taxon"].astype(str).isin(selected)
            research_other = float(
                summary.loc[~selected_rows, "research_within_parent_pct"].sum()
            )
            gbif_other = float(
                summary.loc[~selected_rows, "gbif_within_parent_pct"].sum()
            )
            if (
                research_other <= order_max_other_within_parent_pct + 1e-12
                and gbif_other <= order_max_other_within_parent_pct + 1e-12
            ):
                return set(selected)
        return None

    def globally_named_order_labels(frame: pd.DataFrame) -> set[str]:
        summary = child_summary(frame, "order")
        global_research = summary["research_count"] / research_total * 100.0
        global_gbif = (
            summary["gbif_described_species_count"] / gbif_total * 100.0
        )
        retained = summary.loc[
            global_research.ge(order_named_min_global_share_pct)
            | global_gbif.ge(order_named_min_global_share_pct),
            "taxon",
        ]
        return {str(value) for value in retained}

    rows: list[dict[str, Any]] = []
    sibling_order: defaultdict[str | None, int] = defaultdict(int)
    expanded_classes = 0
    relaxed_expanded_classes = 0
    stopped_classes = 0

    def add_node(
        *,
        taxon: str,
        rank: str,
        parent_id: str | None,
        frame: pd.DataFrame,
        path: Mapping[str, str | None],
        is_other: bool = False,
        is_order_remainder: bool = False,
        is_relaxed_order_remainder: bool = False,
        selection_reason: str,
    ) -> str:
        node_id = " > ".join(
            str(path[value]) for value in RANKS if path.get(value) is not None
        )
        if any(row["node_id"] == node_id for row in rows):
            raise TaxaHierarchyError(f"Duplicate sunburst node id: {node_id!r}.")
        rows.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "taxon": taxon,
                "rank": rank,
                "depth": RANKS.index(rank) + 1,
                "sibling_order": sibling_order[parent_id],
                **{value: path.get(value) for value in RANKS},
                "research_count": float(frame["research_count"].sum()),
                "gbif_described_species_count": int(
                    frame["gbif_described_species_count"].sum()
                ),
                "source_order_paths": int(len(frame)),
                "is_other": bool(is_other),
                "is_order_remainder": bool(is_order_remainder),
                "is_relaxed_order_remainder": bool(
                    is_relaxed_order_remainder
                ),
                "selection_reason": selection_reason,
            }
        )
        sibling_order[parent_id] += 1
        return node_id

    for kingdom in major_kingdoms:
        kingdom_frame = work.loc[work["kingdom"].eq(kingdom)]
        kingdom_id = add_node(
            taxon=kingdom,
            rank="kingdom",
            parent_id=None,
            frame=kingdom_frame,
            path={"kingdom": kingdom},
            selection_reason="configured major Kingdom",
        )
        if kingdom in terminal_kingdoms:
            rows[-1]["selection_reason"] = (
                "configured terminal Kingdom; lower-rank segments too small"
            )
            continue
        retained_phyla = threshold_labels(
            kingdom_frame, "phylum", phylum_min_within_parent_pct
        )
        ordered_phyla = child_summary(kingdom_frame, "phylum")
        for phylum in ordered_phyla.loc[
            ordered_phyla["taxon"].astype(str).isin(retained_phyla), "taxon"
        ].astype(str):
            phylum_frame = kingdom_frame.loc[kingdom_frame["phylum"].eq(phylum)]
            phylum_id = add_node(
                taxon=phylum,
                rank="phylum",
                parent_id=kingdom_id,
                frame=phylum_frame,
                path={"kingdom": kingdom, "phylum": phylum},
                selection_reason=(
                    f"at least {phylum_min_within_parent_pct:g}% of research or "
                    "GBIF within Kingdom"
                ),
            )
            retained_classes = threshold_labels(
                phylum_frame, "class", class_min_within_parent_pct
            )
            ordered_classes = child_summary(phylum_frame, "class")
            for taxon_class in ordered_classes.loc[
                ordered_classes["taxon"].astype(str).isin(retained_classes),
                "taxon",
            ].astype(str):
                class_frame = phylum_frame.loc[
                    phylum_frame["class"].eq(taxon_class)
                ]
                class_id = add_node(
                    taxon=taxon_class,
                    rank="class",
                    parent_id=phylum_id,
                    frame=class_frame,
                    path={
                        "kingdom": kingdom,
                        "phylum": phylum,
                        "class": taxon_class,
                    },
                    selection_reason=(
                        f"at least {class_min_within_parent_pct:g}% of research or "
                        "GBIF within Phylum"
                    ),
                )
                retained_orders = compact_order_labels(class_frame)
                relaxed_expansion = False
                if retained_orders is None:
                    class_research_share_pct = (
                        float(class_frame["research_count"].sum())
                        / research_total
                        * 100.0
                    )
                    if (
                        relaxed_order_expansion_min_class_research_share_pct
                        is None
                        or class_research_share_pct
                        < relaxed_order_expansion_min_class_research_share_pct
                    ):
                        stopped_classes += 1
                        continue
                    retained_orders = globally_named_order_labels(class_frame)
                    relaxed_expansion = True
                retained_orders &= globally_named_order_labels(class_frame)
                if not retained_orders:
                    stopped_classes += 1
                    continue
                if relaxed_expansion:
                    relaxed_expanded_classes += 1
                expanded_classes += 1
                ordered_orders = child_summary(class_frame, "order")
                for order in ordered_orders.loc[
                    ordered_orders["taxon"].astype(str).isin(retained_orders),
                    "taxon",
                ].astype(str):
                    order_frame = class_frame.loc[class_frame["order"].eq(order)]
                    add_node(
                        taxon=order,
                        rank="order",
                        parent_id=class_id,
                        frame=order_frame,
                        path={
                            "kingdom": kingdom,
                            "phylum": phylum,
                            "class": taxon_class,
                            "order": order,
                        },
                        selection_reason=(
                            "named Order at or above the global share threshold "
                            "on research or GBIF"
                        ),
                    )
                omitted_orders = class_frame.loc[
                    ~class_frame["order"].isin(retained_orders)
                ]
                if not omitted_orders.empty:
                    other_order = f"Remaining {taxon_class} orders"
                    add_node(
                        taxon=other_order,
                        rank="order",
                        parent_id=class_id,
                        frame=omitted_orders,
                        path={
                            "kingdom": kingdom,
                            "phylum": phylum,
                            "class": taxon_class,
                            "order": other_order,
                        },
                        is_other=True,
                        is_order_remainder=True,
                        is_relaxed_order_remainder=relaxed_expansion,
                        selection_reason=(
                            "terminal Orders below the global threshold on both "
                            "research and GBIF, including any compact residual"
                        ),
                    )

            omitted_classes = phylum_frame.loc[
                ~phylum_frame["class"].isin(retained_classes)
            ]
            if not omitted_classes.empty:
                other_class = f"Remaining {phylum} classes"
                add_node(
                    taxon=other_class,
                    rank="class",
                    parent_id=phylum_id,
                    frame=omitted_classes,
                    path={
                        "kingdom": kingdom,
                        "phylum": phylum,
                        "class": other_class,
                    },
                    is_other=True,
                    selection_reason="terminal Remaining below-threshold Classes",
                )

        omitted_phyla = kingdom_frame.loc[
            ~kingdom_frame["phylum"].isin(retained_phyla)
        ]
        if not omitted_phyla.empty:
            other_phylum = f"Remaining {kingdom} phyla"
            add_node(
                taxon=other_phylum,
                rank="phylum",
                parent_id=kingdom_id,
                frame=omitted_phyla,
                path={"kingdom": kingdom, "phylum": other_phylum},
                is_other=True,
                selection_reason="terminal Remaining below-threshold Phyla",
            )

    other_kingdoms = work.loc[~work["kingdom"].isin(major_kingdoms)]
    if not other_kingdoms.empty:
        add_node(
            taxon="Other kingdoms",
            rank="kingdom",
            parent_id=None,
            frame=other_kingdoms,
            path={"kingdom": "Other kingdoms"},
            is_other=True,
            selection_reason="configured terminal residual Kingdoms",
        )

    nodes = pd.DataFrame(rows)
    if nodes["node_id"].duplicated().any():
        raise TaxaHierarchyError("Sunburst node ids are not unique.")
    nodes["is_terminal"] = ~nodes["node_id"].isin(nodes["parent_id"].dropna())
    nodes["research_share_pct"] = (
        nodes["research_count"] / research_total * 100.0
    )
    nodes["gbif_described_species_share_pct"] = (
        nodes["gbif_described_species_count"] / gbif_total * 100.0
    )
    parent_research = nodes.set_index("node_id")["research_count"]
    parent_gbif = nodes.set_index("node_id")["gbif_described_species_count"]
    nodes["research_within_parent_pct"] = nodes["research_share_pct"]
    nodes["gbif_within_parent_pct"] = nodes[
        "gbif_described_species_share_pct"
    ]
    descendants = nodes["parent_id"].notna()
    nodes.loc[descendants, "research_within_parent_pct"] = (
        nodes.loc[descendants, "research_count"].to_numpy()
        / nodes.loc[descendants, "parent_id"].map(parent_research).to_numpy()
        * 100.0
    )
    nodes.loc[descendants, "gbif_within_parent_pct"] = (
        nodes.loc[descendants, "gbif_described_species_count"].to_numpy()
        / nodes.loc[descendants, "parent_id"].map(parent_gbif).to_numpy()
        * 100.0
    )
    nodes["comparison_research_publications"] = int(
        order_comparison.get(
            "comparison_research_publications", pd.Series([round(research_total)])
        ).iloc[0]
    )
    nodes["comparison_gbif_species"] = gbif_total

    roots = nodes.loc[nodes["parent_id"].isna()]
    if not np.isclose(roots["research_count"].sum(), research_total):
        raise TaxaHierarchyError("Sunburst root research counts do not reconcile.")
    if int(roots["gbif_described_species_count"].sum()) != gbif_total:
        raise TaxaHierarchyError("Sunburst root GBIF counts do not reconcile.")
    for parent_id, children in nodes.loc[nodes["parent_id"].notna()].groupby(
        "parent_id", sort=False
    ):
        parent = nodes.loc[nodes["node_id"].eq(parent_id)].iloc[0]
        if not np.isclose(children["research_count"].sum(), parent["research_count"]):
            raise TaxaHierarchyError(
                f"Research children do not reconcile for {parent_id!r}."
            )
        if (
            int(children["gbif_described_species_count"].sum())
            != int(parent["gbif_described_species_count"])
        ):
            raise TaxaHierarchyError(
                f"GBIF children do not reconcile for {parent_id!r}."
            )
    other_ids = set(nodes.loc[nodes["is_other"], "node_id"])
    if nodes["parent_id"].isin(other_ids).any():
        raise TaxaHierarchyError("An Other node has plotted descendants.")

    remainder_orders = nodes.loc[nodes["is_order_remainder"]]
    max_remainder_research = (
        float(remainder_orders["research_within_parent_pct"].max())
        if not remainder_orders.empty
        else 0.0
    )
    max_remainder_gbif = (
        float(remainder_orders["gbif_within_parent_pct"].max())
        if not remainder_orders.empty
        else 0.0
    )

    audit_rows: list[dict[str, Any]] = [
        {"metric": "Comparison research publications", "value": int(nodes["comparison_research_publications"].iloc[0])},
        {"metric": "Comparison GBIF species", "value": gbif_total},
        {"metric": "Exact shared source Order paths", "value": int(len(work))},
        {"metric": "Plotted hierarchy nodes", "value": int(len(nodes))},
        {"metric": "Terminal plotted nodes", "value": int(nodes["is_terminal"].sum())},
        {"metric": "Named Classes expanded to Order", "value": expanded_classes},
        {"metric": "Large Classes using relaxed Order expansion", "value": relaxed_expanded_classes},
        {"metric": "Named Classes stopped at Class", "value": stopped_classes},
        {"metric": "Named Kingdoms stopped at Kingdom", "value": len(terminal_kingdoms)},
        {"metric": "Order maximum named children", "value": order_max_children},
        {"metric": "Compact Order prefilter maximum residual share percent", "value": order_max_other_within_parent_pct},
        {"metric": "Relaxed expansion minimum Class research share percent", "value": relaxed_order_expansion_min_class_research_share_pct},
        {"metric": "Named Order minimum global share percent", "value": order_named_min_global_share_pct},
        {"metric": "Observed maximum Remaining-orders research share percent", "value": max_remainder_research},
        {"metric": "Observed maximum Remaining-orders GBIF share percent", "value": max_remainder_gbif},
    ]
    for rank in RANKS:
        audit_rows.append(
            {
                "metric": f"Plotted {rank} nodes",
                "value": int(nodes["rank"].eq(rank).sum()),
            }
        )
    audit = pd.DataFrame(audit_rows)
    return SunburstHierarchy(nodes=nodes, audit=audit)


def apply_independent_gbif_benchmark(
    selected: SunburstHierarchy,
    gbif_counts: pd.DataFrame,
    broad_benchmark: pd.DataFrame,
    *,
    root_group_mapping: Mapping[str, Iterable[str]] | None = None,
    phylum_min_within_parent_pct: float = 2.0,
    class_min_within_parent_pct: float = 2.0,
    order_named_min_global_share_pct: float = 1.0,
) -> SunburstGbifBenchmark:
    """Rebase a research-selected sunburst to the full GBIF benchmark.

    The plotted node selection and every research count are held fixed.  Full
    accepted-species counts are then assigned independently: named nodes take
    the count for their exact fixed-snapshot path, while each displayed
    ``Remaining`` node takes its parent's exact complement.  Consequently,
    displayed children reconcile to their parent on both measures even though
    research and described diversity use different defensible denominators.

    The returned assignment table preserves every complete GBIF child taxon
    below an expanded parent and identifies its displayed destination.  Species
    missing the next rank are recorded explicitly as an unavailable-rank row;
    they are never silently dropped from the parent's complement.
    """
    required_nodes = {
        "node_id",
        "parent_id",
        "taxon",
        "rank",
        "depth",
        "sibling_order",
        *RANKS,
        "research_count",
        "gbif_described_species_count",
        "source_order_paths",
        "is_other",
        "is_order_remainder",
        "is_relaxed_order_remainder",
        "selection_reason",
    }
    missing_nodes = required_nodes.difference(selected.nodes.columns)
    if missing_nodes:
        raise TaxaHierarchyError(
            "Selected sunburst lacks required columns: "
            f"{sorted(missing_nodes)}"
        )
    required_gbif = {
        "rank",
        *RANKS,
        "gbif_described_species_count",
    }
    missing_gbif = required_gbif.difference(gbif_counts.columns)
    if missing_gbif:
        raise TaxaHierarchyError(
            f"Full GBIF hierarchy lacks columns: {sorted(missing_gbif)}"
        )
    required_benchmark = {"broad_group", "described_species_count"}
    missing_benchmark = required_benchmark.difference(broad_benchmark.columns)
    if missing_benchmark:
        raise TaxaHierarchyError(
            "Broad GBIF benchmark lacks columns: "
            f"{sorted(missing_benchmark)}"
        )
    if selected.nodes.empty:
        raise TaxaHierarchyError("Selected sunburst is empty.")
    if min(
        phylum_min_within_parent_pct,
        class_min_within_parent_pct,
        order_named_min_global_share_pct,
    ) < 0:
        raise TaxaHierarchyError("GBIF display-audit thresholds cannot be negative.")

    nodes = selected.nodes.copy().reset_index(drop=True)
    if nodes["node_id"].duplicated().any():
        raise TaxaHierarchyError("Selected sunburst node ids are not unique.")
    original = nodes.set_index("node_id").copy()
    original_expanded_ids = set(nodes["parent_id"].dropna().astype(str))
    nodes["aligned_gbif_described_species_count"] = nodes[
        "gbif_described_species_count"
    ].astype(int)
    nodes["aligned_gbif_described_species_share_pct"] = nodes.get(
        "gbif_described_species_share_pct", pd.Series(0.0, index=nodes.index)
    ).astype(float)

    mapping = {
        str(root): tuple(str(group) for group in groups)
        for root, groups in (
            DEFAULT_SUNBURST_ROOT_GROUPS
            if root_group_mapping is None
            else root_group_mapping
        ).items()
    }
    roots = nodes.loc[nodes["parent_id"].isna()]
    root_names = set(roots["taxon"].astype(str))
    if root_names != set(mapping):
        raise TaxaHierarchyError(
            "GBIF root mapping must cover the plotted roots exactly: "
            f"roots={sorted(root_names)}, mapping={sorted(mapping)}."
        )
    mapped_groups = [group for groups in mapping.values() for group in groups]
    if len(mapped_groups) != len(set(mapped_groups)):
        raise TaxaHierarchyError("A broad GBIF group is mapped to multiple roots.")
    if broad_benchmark["broad_group"].astype(str).duplicated().any():
        raise TaxaHierarchyError("Broad GBIF benchmark groups are not unique.")
    broad_counts = (
        broad_benchmark.assign(
            broad_group=broad_benchmark["broad_group"].astype(str)
        )
        .set_index("broad_group")["described_species_count"]
        .astype(int)
    )
    absent_groups = set(mapped_groups).difference(broad_counts.index)
    if absent_groups:
        raise TaxaHierarchyError(
            f"Mapped broad GBIF groups are absent: {sorted(absent_groups)}"
        )
    benchmark_total = int(broad_counts.reindex(mapped_groups).sum())
    if benchmark_total <= 0:
        raise TaxaHierarchyError("Independent GBIF benchmark total must be positive.")

    gbif_by_rank: dict[str, pd.DataFrame] = {}
    path_count_lookup: dict[str, dict[tuple[str, ...], int]] = {}
    for rank in RANKS:
        path_columns = list(_path_columns(rank))
        rank_counts = gbif_counts.loc[
            gbif_counts["rank"].eq(rank),
            [*path_columns, "gbif_described_species_count"],
        ].copy()
        if rank_counts[path_columns].isna().any().any():
            raise TaxaHierarchyError(
                f"Full GBIF {rank} rows contain an incomplete path."
            )
        if rank_counts.duplicated(path_columns).any():
            raise TaxaHierarchyError(
                f"Full GBIF {rank} paths are not unique."
            )
        rank_counts["gbif_described_species_count"] = rank_counts[
            "gbif_described_species_count"
        ].astype(int)
        gbif_by_rank[rank] = rank_counts
        path_count_lookup[rank] = {
            tuple(str(row[column]) for column in path_columns): int(
                row["gbif_described_species_count"]
            )
            for row in rank_counts.to_dict(orient="records")
        }

    root_count_by_taxon = {
        root: int(broad_counts.reindex(groups).sum())
        for root, groups in mapping.items()
    }
    for index, row in nodes.iterrows():
        taxon = str(row["taxon"])
        if pd.isna(row["parent_id"]):
            nodes.at[index, "gbif_described_species_count"] = root_count_by_taxon[
                taxon
            ]
            continue
        if bool(row["is_other"]):
            nodes.at[index, "gbif_described_species_count"] = 0
            continue
        rank = str(row["rank"])
        path = tuple(str(row[column]) for column in _path_columns(rank))
        count = path_count_lookup[rank].get(path)
        if count is None:
            raise TaxaHierarchyError(
                "A named research-selected node is absent from the full GBIF "
                f"hierarchy: {row['node_id']!r}."
            )
        nodes.at[index, "gbif_described_species_count"] = count
    nodes["gbif_described_species_count"] = nodes[
        "gbif_described_species_count"
    ].astype(int)

    assignment_rows: list[dict[str, Any]] = []
    for root, groups in mapping.items():
        root_row = roots.loc[roots["taxon"].astype(str).eq(root)].iloc[0]
        for group in groups:
            assignment_rows.append(
                {
                    "assignment_rank": "kingdom",
                    "parent_node_id": None,
                    "parent_taxon": None,
                    "actual_node_id": f"broad group: {group}",
                    "actual_taxon": group,
                    **{rank: None for rank in RANKS},
                    "source_rank_complete": True,
                    "gbif_described_species_count": int(broad_counts[group]),
                    "display_node_id": str(root_row["node_id"]),
                    "display_taxon": root,
                    "display_is_remaining": bool(root_row["is_other"]),
                    "display_assignment": (
                        "aggregated root" if len(groups) > 1 or bool(root_row["is_other"])
                        else "named root"
                    ),
                    "assignment_reason": (
                        "configured broad-benchmark-to-Kingdom display mapping"
                    ),
                    "crosses_display_threshold": False,
                }
            )

    added_remaining_nodes = 0
    for parent_id in sorted(original_expanded_ids):
        parent_index = nodes.index[nodes["node_id"].astype(str).eq(parent_id)]
        if len(parent_index) != 1:
            raise TaxaHierarchyError(f"Missing plotted parent {parent_id!r}.")
        parent_index = int(parent_index[0])
        parent = nodes.loc[parent_index]
        parent_depth = int(parent["depth"])
        if parent_depth >= len(RANKS):
            raise TaxaHierarchyError(f"Order node cannot be expanded: {parent_id!r}.")
        child_rank = RANKS[parent_depth]
        child_path_columns = list(_path_columns(child_rank))
        parent_path_columns = list(_path_columns(str(parent["rank"])))
        child_mask = nodes["parent_id"].astype(object).eq(parent_id)
        children = nodes.loc[child_mask]
        named_children = children.loc[~children["is_other"]]
        remaining_children = children.loc[children["is_other"]]
        if len(remaining_children) > 1:
            raise TaxaHierarchyError(
                f"Expanded parent has multiple Remaining children: {parent_id!r}."
            )
        named_total = int(named_children["gbif_described_species_count"].sum())
        parent_total = int(parent["gbif_described_species_count"])
        residual = parent_total - named_total
        if residual < 0:
            raise TaxaHierarchyError(
                "Named full-GBIF children exceed their parent for "
                f"{parent_id!r}: parent={parent_total:,}, named={named_total:,}."
            )

        if remaining_children.empty and residual > 0:
            lineage = {rank: None for rank in RANKS}
            for rank in parent_path_columns:
                lineage[rank] = parent[rank]
            rank_plural = {
                "phylum": "phyla",
                "class": "classes",
                "order": "orders",
            }[child_rank]
            remaining_taxon = f"Remaining {parent['taxon']} {rank_plural}"
            lineage[child_rank] = remaining_taxon
            remaining_id = " > ".join(
                str(lineage[rank]) for rank in RANKS if lineage[rank] is not None
            )
            if remaining_id in set(nodes["node_id"].astype(str)):
                raise TaxaHierarchyError(
                    f"Duplicate generated Remaining node: {remaining_id!r}."
                )
            new_row = {
                "node_id": remaining_id,
                "parent_id": parent_id,
                "taxon": remaining_taxon,
                "rank": child_rank,
                "depth": parent_depth + 1,
                "sibling_order": int(children["sibling_order"].max()) + 1,
                **lineage,
                "research_count": 0.0,
                "gbif_described_species_count": residual,
                "source_order_paths": 0,
                "is_other": True,
                "is_order_remainder": child_rank == "order",
                "is_relaxed_order_remainder": False,
                "selection_reason": (
                    "full-GBIF complement absent from the research-selected "
                    "children; zero research angle"
                ),
                "aligned_gbif_described_species_count": 0,
                "aligned_gbif_described_species_share_pct": 0.0,
            }
            nodes = pd.concat([nodes, pd.DataFrame([new_row])], ignore_index=True)
            added_remaining_nodes += 1
        else:
            if not remaining_children.empty:
                remaining_index = int(remaining_children.index[0])
                nodes.at[
                    remaining_index, "gbif_described_species_count"
                ] = residual

        children = nodes.loc[nodes["parent_id"].astype(object).eq(parent_id)]
        named_children = children.loc[~children["is_other"]]
        remaining_children = children.loc[children["is_other"]]
        named_destination = {
            str(row[child_rank]): (str(row["node_id"]), str(row["taxon"]))
            for row in named_children.to_dict(orient="records")
        }
        remaining_destination = (
            None
            if remaining_children.empty
            else (
                str(remaining_children.iloc[0]["node_id"]),
                str(remaining_children.iloc[0]["taxon"]),
            )
        )

        actual_children = gbif_by_rank[child_rank]
        actual_mask = pd.Series(True, index=actual_children.index)
        for rank in parent_path_columns:
            actual_mask &= actual_children[rank].astype(str).eq(str(parent[rank]))
        actual_children = actual_children.loc[actual_mask]
        actual_complete_total = int(
            actual_children["gbif_described_species_count"].sum()
        )
        unavailable_count = parent_total - actual_complete_total
        if unavailable_count < 0:
            raise TaxaHierarchyError(
                f"Complete {child_rank} counts exceed {parent_id!r}."
            )

        for actual in actual_children.to_dict(orient="records"):
            actual_taxon = str(actual[child_rank])
            destination = named_destination.get(actual_taxon)
            is_remaining = destination is None
            if destination is None:
                if remaining_destination is None:
                    raise TaxaHierarchyError(
                        f"No Remaining destination for unshown {child_rank} "
                        f"{actual_taxon!r} below {parent_id!r}."
                    )
                destination = remaining_destination
            actual_count = int(actual["gbif_described_species_count"])
            within_parent_pct = (
                actual_count / parent_total * 100.0 if parent_total else 0.0
            )
            global_pct = actual_count / benchmark_total * 100.0
            if child_rank == "phylum":
                crosses_threshold = (
                    within_parent_pct >= phylum_min_within_parent_pct
                )
            elif child_rank == "class":
                crosses_threshold = (
                    within_parent_pct >= class_min_within_parent_pct
                )
            else:
                crosses_threshold = (
                    global_pct >= order_named_min_global_share_pct
                )
            actual_lineage = {rank: None for rank in RANKS}
            actual_lineage.update(
                {rank: actual[rank] for rank in child_path_columns}
            )
            assignment_rows.append(
                {
                    "assignment_rank": child_rank,
                    "parent_node_id": parent_id,
                    "parent_taxon": str(parent["taxon"]),
                    "actual_node_id": " > ".join(
                        str(actual_lineage[rank])
                        for rank in RANKS
                        if actual_lineage[rank] is not None
                    ),
                    "actual_taxon": actual_taxon,
                    **actual_lineage,
                    "source_rank_complete": True,
                    "gbif_described_species_count": actual_count,
                    "display_node_id": destination[0],
                    "display_taxon": destination[1],
                    "display_is_remaining": is_remaining,
                    "display_assignment": (
                        "Remaining complement" if is_remaining else "named child"
                    ),
                    "assignment_reason": (
                        "full GBIF path is not an individually displayed child"
                        if is_remaining
                        else "exact full-GBIF path matches displayed child"
                    ),
                    "crosses_display_threshold": bool(
                        is_remaining and crosses_threshold
                    ),
                }
            )

        if unavailable_count:
            if remaining_destination is None:
                raise TaxaHierarchyError(
                    f"No Remaining destination for {unavailable_count:,} species "
                    f"without {child_rank} below {parent_id!r}."
                )
            unavailable_lineage = {rank: None for rank in RANKS}
            unavailable_lineage.update(
                {rank: parent[rank] for rank in parent_path_columns}
            )
            assignment_rows.append(
                {
                    "assignment_rank": child_rank,
                    "parent_node_id": parent_id,
                    "parent_taxon": str(parent["taxon"]),
                    "actual_node_id": f"{parent_id} > [{child_rank} unavailable]",
                    "actual_taxon": f"[{child_rank} unavailable]",
                    **unavailable_lineage,
                    "source_rank_complete": False,
                    "gbif_described_species_count": unavailable_count,
                    "display_node_id": remaining_destination[0],
                    "display_taxon": remaining_destination[1],
                    "display_is_remaining": True,
                    "display_assignment": "Remaining complement",
                    "assignment_reason": (
                        f"accepted species has no complete {child_rank} path"
                    ),
                    "crosses_display_threshold": False,
                }
            )

    assignments = pd.DataFrame(assignment_rows)
    assignments["gbif_global_share_pct"] = (
        assignments["gbif_described_species_count"] / benchmark_total * 100.0
    )
    parent_counts = nodes.set_index("node_id")["gbif_described_species_count"]
    assignments["gbif_within_display_parent_pct"] = assignments[
        "gbif_global_share_pct"
    ]
    has_parent = assignments["parent_node_id"].notna()
    assignments.loc[has_parent, "gbif_within_display_parent_pct"] = (
        assignments.loc[has_parent, "gbif_described_species_count"].to_numpy()
        / assignments.loc[has_parent, "parent_node_id"]
        .map(parent_counts)
        .to_numpy()
        * 100.0
    )
    assignment_counts = assignments.groupby("display_node_id").size()
    nodes["full_gbif_assignment_rows"] = (
        nodes["node_id"].map(assignment_counts).fillna(0).astype(int)
    )

    nodes["gbif_described_species_count"] = nodes[
        "gbif_described_species_count"
    ].astype(int)
    nodes["is_terminal"] = ~nodes["node_id"].isin(nodes["parent_id"].dropna())
    research_total = float(
        nodes.loc[nodes["parent_id"].isna(), "research_count"].sum()
    )
    if research_total <= 0:
        raise TaxaHierarchyError("Research total must remain positive after rebasing.")
    nodes["research_share_pct"] = nodes["research_count"] / research_total * 100.0
    nodes["gbif_described_species_share_pct"] = (
        nodes["gbif_described_species_count"] / benchmark_total * 100.0
    )
    parent_research = nodes.set_index("node_id")["research_count"]
    parent_gbif = nodes.set_index("node_id")["gbif_described_species_count"]
    nodes["research_within_parent_pct"] = nodes["research_share_pct"]
    nodes["gbif_within_parent_pct"] = nodes[
        "gbif_described_species_share_pct"
    ]
    descendants = nodes["parent_id"].notna()
    nodes.loc[descendants, "research_within_parent_pct"] = (
        nodes.loc[descendants, "research_count"].to_numpy()
        / nodes.loc[descendants, "parent_id"].map(parent_research).to_numpy()
        * 100.0
    )
    nodes.loc[descendants, "gbif_within_parent_pct"] = (
        nodes.loc[descendants, "gbif_described_species_count"].to_numpy()
        / nodes.loc[descendants, "parent_id"].map(parent_gbif).to_numpy()
        * 100.0
    )
    comparison_publications = int(
        original.get(
            "comparison_research_publications",
            pd.Series([round(research_total)]),
        ).iloc[0]
    )
    nodes["comparison_research_publications"] = comparison_publications
    nodes["comparison_gbif_species"] = benchmark_total

    original_ids = set(original.index.astype(str))
    rebased_original = nodes.loc[nodes["node_id"].astype(str).isin(original_ids)].set_index(
        "node_id"
    )
    if not np.allclose(
        rebased_original.loc[original.index, "research_count"].to_numpy(dtype=float),
        original["research_count"].to_numpy(dtype=float),
    ):
        raise TaxaHierarchyError("GBIF rebasing changed an existing research count.")
    if not np.array_equal(
        rebased_original.loc[original.index, "sibling_order"].to_numpy(),
        original["sibling_order"].to_numpy(),
    ):
        raise TaxaHierarchyError("GBIF rebasing changed an existing sibling order.")
    roots = nodes.loc[nodes["parent_id"].isna()]
    if not np.isclose(roots["research_count"].sum(), research_total):
        raise TaxaHierarchyError("Rebased root research counts do not reconcile.")
    if int(roots["gbif_described_species_count"].sum()) != benchmark_total:
        raise TaxaHierarchyError("Rebased GBIF roots do not reconcile.")
    for parent_id, children in nodes.loc[nodes["parent_id"].notna()].groupby(
        "parent_id", sort=False
    ):
        parent = nodes.loc[nodes["node_id"].eq(parent_id)].iloc[0]
        if not np.isclose(children["research_count"].sum(), parent["research_count"]):
            raise TaxaHierarchyError(
                f"Rebased research children do not reconcile for {parent_id!r}."
            )
        if int(children["gbif_described_species_count"].sum()) != int(
            parent["gbif_described_species_count"]
        ):
            raise TaxaHierarchyError(
                f"Rebased GBIF children do not reconcile for {parent_id!r}."
            )
    other_ids = set(nodes.loc[nodes["is_other"], "node_id"].astype(str))
    if nodes["parent_id"].isin(other_ids).any():
        raise TaxaHierarchyError("A rebased Remaining node has plotted descendants.")

    audit = selected.audit.copy()
    audit["metric"] = audit["metric"].replace(
        {
            "Comparison GBIF species": (
                "Aligned GBIF species used for display selection"
            )
        }
    )
    remainder_orders = nodes.loc[nodes["is_order_remainder"]]
    if not remainder_orders.empty:
        mask = audit["metric"].eq(
            "Observed maximum Remaining-orders GBIF share percent"
        )
        audit.loc[mask, "value"] = float(
            remainder_orders["gbif_within_parent_pct"].max()
        )
    new_audit_rows: list[dict[str, Any]] = [
        {"metric": "Independent GBIF benchmark species", "value": benchmark_total},
        {
            "metric": "Full GBIF display-assignment rows",
            "value": int(len(assignments)),
        },
        {
            "metric": "Added zero-research Remaining nodes",
            "value": added_remaining_nodes,
        },
    ]
    for rank in RANKS[1:]:
        rank_assignments = assignments.loc[assignments["assignment_rank"].eq(rank)]
        remaining = rank_assignments.loc[rank_assignments["display_is_remaining"]]
        new_audit_rows.extend(
            [
                {
                    "metric": f"Full GBIF {rank} taxa assigned to Remaining",
                    "value": int(remaining["source_rank_complete"].sum()),
                },
                {
                    "metric": f"Full GBIF species assigned to Remaining at {rank}",
                    "value": int(remaining["gbif_described_species_count"].sum()),
                },
                {
                    "metric": f"Unshown full GBIF {rank} taxa crossing display threshold",
                    "value": int(remaining["crosses_display_threshold"].sum()),
                },
                {
                    "metric": f"Full GBIF species without {rank} in expanded parents",
                    "value": int(
                        remaining.loc[
                            ~remaining["source_rank_complete"],
                            "gbif_described_species_count",
                        ].sum()
                    ),
                },
            ]
        )
    audit = pd.concat(
        [audit, pd.DataFrame(new_audit_rows)], ignore_index=True
    )
    return SunburstGbifBenchmark(
        nodes=nodes,
        assignments=assignments.sort_values(
            ["assignment_rank", "parent_node_id", "gbif_described_species_count"],
            ascending=[True, True, False],
            na_position="first",
            ignore_index=True,
        ),
        audit=audit,
    )


__all__ = [
    "RANKS",
    "RankHierarchyEvidence",
    "SunburstGbifBenchmark",
    "SunburstHierarchy",
    "TaxaHierarchyError",
    "apply_independent_gbif_benchmark",
    "build_rank_audits",
    "build_rank_hierarchy_evidence",
    "build_research_attention_sunburst",
    "build_research_rank_attributions",
    "canonical_research_path",
    "compare_exact_rank_paths",
    "stream_gbif_rank_counts",
]

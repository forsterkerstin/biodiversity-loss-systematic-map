"""Publication-level taxonomic attention relative to described diversity.

The analysis keeps one row per publication while loading the taxa, driver,
Threats L0, and ecosystem-realm coding outputs. Multi-group publications are
fractionally allocated so every included publication contributes total weight
one to the headline taxonomic composition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from data_helpers.analysis.taxa.benchmark import DescribedDiversity
from data_helpers.visualization import display_path, save_figure


class TaxaSkewError(ValueError):
    """Raised when a source or calculation violates the analysis contract."""


@dataclass(frozen=True)
class TaxaSkewEvidence:
    """Prepared article evidence, taxonomic attributions, and audits."""

    articles: pd.DataFrame
    included: pd.DataFrame
    attributions: pd.DataFrame
    inclusion_audit: pd.DataFrame
    group_audit: pd.DataFrame
    match_status_audit: pd.DataFrame
    group_reason_audit: pd.DataFrame
    special_value_audit: pd.DataFrame
    auxiliary_label_audit: pd.DataFrame


@dataclass(frozen=True)
class TaxaSkewAnalysis:
    """Headline comparison and bootstrap uncertainty."""

    comparison: pd.DataFrame
    bootstrap_intervals: pd.DataFrame


class TaxaSkewResultStore:
    """Save compact tables, article evidence, and figures consistently."""

    def __init__(
        self,
        root: str | Path,
        *,
        table_subdirectory: str = "csv",
        figure_subdirectory: str = "figures",
    ) -> None:
        self.root = Path(root)
        self.table_directory = self.root / table_subdirectory
        self.figure_directory = self.root / figure_subdirectory

    def save_table(
        self, table: pd.DataFrame, filename: str, *, index: bool = False
    ) -> Path:
        self.table_directory.mkdir(parents=True, exist_ok=True)
        path = self.table_directory / f"{filename}.csv"
        table.to_csv(path, index=index)
        print(f"Saved: {display_path(path)}")
        return path

    def save_json(self, payload: Mapping[str, Any], filename: str) -> Path:
        """Save compact analysis provenance beside the reported CSV tables."""
        self.table_directory.mkdir(parents=True, exist_ok=True)
        path = self.table_directory / f"{filename}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {display_path(path)}")
        return path

    def export_figure(self, figure: Any, filename: str) -> list[Path]:
        figure.set_layout_engine("none")
        return save_figure(
            figure,
            self.figure_directory / filename,
            formats=["pdf"],
            facecolor="white",
        )


INCLUSION_ORDER = (
    "Included: at least one benchmarkable broad group",
    "Excluded: Not applicable only",
    "Excluded: Unclear only",
    "Excluded: mixed Not applicable and Unclear",
    "Excluded: accepted taxon but broad hierarchy unresolved",
    "Excluded: reported taxon but no accepted benchmarkable match",
    "Excluded: no taxon item returned",
)


def prepare_taxa_skew_from_prepared(
    articles: pd.DataFrame,
    *,
    benchmark_groups: Iterable[str],
    unresolved_group: str,
    match_status_audit: pd.DataFrame,
    group_reason_audit: pd.DataFrame,
    special_value_audit: pd.DataFrame,
    auxiliary_label_audit: pd.DataFrame,
    direction: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> TaxaSkewEvidence:
    """Express the canonical prepared handoff as broad-skew evidence.

    Source linkage, label parsing, match accounting, and broad grouping have
    already been performed by ``taxa_analysis_prep``. This adapter retains the
    original Vector 1 estimand while keeping those processing operations out of
    the results notebook.

    ``direction`` and ``start_year``/``end_year`` restrict the analytical
    universe before any weighting. They exist so this baseline can be expressed
    on the same grain as the driver, realm, and temporal analyses, which scope to
    negative-direction evidence in complete publication years. Leaving them at
    ``None`` retains the corpus-wide universe, which spans every screened
    direction and is therefore **not** the biodiversity-loss dataset reported in
    the paper.
    """
    benchmark_groups = tuple(benchmark_groups)
    benchmark_group_set = set(benchmark_groups)
    if not benchmark_groups or len(benchmark_group_set) != len(benchmark_groups):
        raise TaxaSkewError("benchmark_groups must be unique and non-empty.")
    required = {
        "UT",
        "publication_year",
        "broad_groups_all",
        "broad_groups",
        "taxa_broad_inclusion_state",
    }
    if direction is not None:
        required.add("s2_dir")
    missing = required.difference(articles.columns)
    if missing:
        raise TaxaSkewError(
            f"Prepared taxa publications lack required columns: {sorted(missing)}"
        )
    if not articles["UT"].is_unique or articles["UT"].isna().any():
        raise TaxaSkewError("Prepared taxa publications violate one row per UT.")

    if direction is not None:
        articles = articles.loc[articles["s2_dir"].eq(direction)]
    if start_year is not None or end_year is not None:
        years = articles["publication_year"]
        lower = years.ge(start_year) if start_year is not None else True
        upper = years.le(end_year) if end_year is not None else True
        articles = articles.loc[lower & upper]
    if articles.empty:
        raise TaxaSkewError(
            "No publication remains after applying the configured analytical scope."
        )

    evidence_articles = articles.copy()
    evidence_articles["benchmark_groups"] = evidence_articles["broad_groups"].map(
        lambda labels: tuple(group for group in benchmark_groups if group in labels)
    )
    evidence_articles["n_benchmark_groups"] = evidence_articles[
        "benchmark_groups"
    ].map(len)
    evidence_articles["taxa_analysis_state"] = evidence_articles[
        "taxa_broad_inclusion_state"
    ]
    evidence_articles["taxa_included"] = evidence_articles[
        "taxa_analysis_state"
    ].eq(INCLUSION_ORDER[0])

    invalid_included = evidence_articles.loc[
        evidence_articles["taxa_included"]
        & evidence_articles["n_benchmark_groups"].eq(0)
    ]
    if not invalid_included.empty:
        raise TaxaSkewError(
            "Prepared broad inclusion state contains no benchmarkable group."
        )
    if evidence_articles["broad_groups_all"].map(
        lambda labels: unresolved_group in labels
    ).any() and unresolved_group in benchmark_group_set:
        raise TaxaSkewError("The unresolved group cannot be benchmarkable.")

    state_counts = evidence_articles["taxa_analysis_state"].value_counts()
    inclusion_audit = pd.DataFrame(
        {
            "taxa_analysis_state": INCLUSION_ORDER,
            "included_in_primary": [
                state == INCLUSION_ORDER[0] for state in INCLUSION_ORDER
            ],
            "n_publications": [
                int(state_counts.get(state, 0)) for state in INCLUSION_ORDER
            ],
        }
    )
    inclusion_audit["share_of_all_publications_pct"] = (
        inclusion_audit["n_publications"] / len(evidence_articles) * 100
    )
    if inclusion_audit["n_publications"].sum() != len(evidence_articles):
        raise TaxaSkewError("Prepared inclusion states do not sum to the source.")

    included = evidence_articles.loc[evidence_articles["taxa_included"]].copy()
    attributions = (
        included[
            ["UT", "publication_year", "benchmark_groups", "n_benchmark_groups"]
        ]
        .explode("benchmark_groups", ignore_index=True)
        .rename(columns={"benchmark_groups": "broad_group"})
    )
    attributions["fractional_publication_weight"] = (
        1.0 / attributions["n_benchmark_groups"]
    )
    article_weights = attributions.groupby("UT")[
        "fractional_publication_weight"
    ].sum()
    if len(article_weights) != len(included) or not np.allclose(
        article_weights.to_numpy(), 1.0
    ):
        raise TaxaSkewError(
            "Prepared fractional group weights do not sum to one per publication."
        )

    group_audit = pd.DataFrame({"broad_group": benchmark_groups})
    unique_counts = attributions.groupby("broad_group")["UT"].nunique()
    effective_counts = attributions.groupby("broad_group")[
        "fractional_publication_weight"
    ].sum()
    group_audit["n_unique_publications"] = (
        group_audit["broad_group"].map(unique_counts).fillna(0).astype(int)
    )
    group_audit["publication_coverage_pct"] = (
        group_audit["n_unique_publications"] / len(included) * 100
    )
    group_audit["effective_publication_count"] = (
        group_audit["broad_group"].map(effective_counts).fillna(0.0)
    )
    group_audit["fractional_attention_share_pct"] = (
        group_audit["effective_publication_count"] / len(included) * 100
    )
    return TaxaSkewEvidence(
        articles=evidence_articles,
        included=included,
        attributions=attributions,
        inclusion_audit=inclusion_audit,
        group_audit=group_audit,
        match_status_audit=match_status_audit.copy(),
        group_reason_audit=group_reason_audit.copy(),
        special_value_audit=special_value_audit.copy(),
        auxiliary_label_audit=auxiliary_label_audit.copy(),
    )


def _bootstrap_attention(
    included: pd.DataFrame,
    *,
    group_order: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    if n_bootstrap <= 0:
        raise TaxaSkewError("n_bootstrap must be positive.")
    pattern_counts = (
        included["benchmark_groups"]
        .value_counts(sort=False)
        .rename_axis("benchmark_groups")
        .rename("n_publications")
        .reset_index()
    )
    matrix = np.zeros((len(pattern_counts), len(group_order)), dtype=float)
    positions = {group: position for position, group in enumerate(group_order)}
    for row_index, groups in enumerate(pattern_counts["benchmark_groups"]):
        if not groups:
            raise TaxaSkewError("Included evidence contains an empty group pattern.")
        for group in groups:
            matrix[row_index, positions[group]] = 1.0 / len(groups)
    probabilities = (
        pattern_counts["n_publications"].to_numpy(dtype=float) / len(included)
    )
    rng = np.random.default_rng(seed)
    sampled_patterns = rng.multinomial(
        len(included), probabilities, size=n_bootstrap
    )
    return sampled_patterns @ matrix / len(included) * 100


def analyze_taxonomic_skew(
    evidence: TaxaSkewEvidence,
    benchmark: DescribedDiversity,
    *,
    benchmark_groups: Iterable[str],
    n_bootstrap: int,
    seed: int,
) -> TaxaSkewAnalysis:
    """Compare fractional publication attention with the fixed GBIF benchmark."""
    group_order = tuple(benchmark_groups)
    benchmark_counts = (
        benchmark.counts.set_index("broad_group")["described_species_count"]
        .reindex(group_order)
        .fillna(0)
        .astype(int)
    )
    if benchmark_counts.sum() <= 0 or (benchmark_counts <= 0).any():
        raise TaxaSkewError(
            "Every benchmarkable group must have a positive species count."
        )
    benchmark_share = benchmark_counts / benchmark_counts.sum() * 100

    comparison = evidence.group_audit.set_index("broad_group").reindex(group_order)
    comparison["described_species_count"] = benchmark_counts
    comparison["described_species_share_pct"] = benchmark_share
    comparison["expected_effective_publications"] = (
        len(evidence.included) * comparison["described_species_share_pct"] / 100
    )
    comparison["effective_publication_minus_expected"] = (
        comparison["effective_publication_count"]
        - comparison["expected_effective_publications"]
    )
    comparison["attention_minus_described_pp"] = (
        comparison["fractional_attention_share_pct"]
        - comparison["described_species_share_pct"]
    )
    comparison["representation_ratio"] = (
        comparison["fractional_attention_share_pct"]
        / comparison["described_species_share_pct"]
    )
    representation_ratio = comparison["representation_ratio"].to_numpy()
    comparison["log2_representation_ratio"] = np.log2(
        representation_ratio,
        out=np.full_like(representation_ratio, np.nan, dtype=float),
        where=representation_ratio > 0,
    )
    comparison["representation"] = np.where(
        comparison["representation_ratio"] >= 1,
        "Over-attended",
        "Under-attended",
    )

    bootstrap = _bootstrap_attention(
        evidence.included,
        group_order=group_order,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    intervals = pd.DataFrame(
        {
            "broad_group": group_order,
            "attention_ci_low_pct": np.quantile(bootstrap, 0.025, axis=0),
            "attention_ci_high_pct": np.quantile(bootstrap, 0.975, axis=0),
        }
    ).set_index("broad_group")
    intervals["gap_ci_low_pp"] = (
        intervals["attention_ci_low_pct"] - comparison["described_species_share_pct"]
    )
    intervals["gap_ci_high_pp"] = (
        intervals["attention_ci_high_pct"]
        - comparison["described_species_share_pct"]
    )
    comparison = comparison.join(intervals)
    comparison.insert(0, "broad_group", comparison.index)
    comparison = comparison.reset_index(drop=True)

    return TaxaSkewAnalysis(
        comparison=comparison,
        bootstrap_intervals=comparison[
            [
                "broad_group",
                "fractional_attention_share_pct",
                "attention_ci_low_pct",
                "attention_ci_high_pct",
                "attention_minus_described_pp",
                "gap_ci_low_pp",
                "gap_ci_high_pp",
            ]
        ].copy(),
    )


def taxonomic_attention_trend(
    included: pd.DataFrame,
    *,
    benchmark_groups: Iterable[str],
) -> pd.DataFrame:
    """Fractional attention share per benchmark group per publication year.

    Uses the same article-balanced weighting as ``analyze_taxonomic_skew``: each
    publication carries total weight one, split evenly across the benchmark groups
    it names, so a publication naming two groups adds 0.5 to each rather than 1.
    Shares therefore sum to 100% within every year.
    """
    benchmark_groups = tuple(benchmark_groups)
    required = {"publication_year", "benchmark_groups"}
    missing = required.difference(included.columns)
    if missing:
        raise TaxaSkewError(f"Included frame is missing columns: {sorted(missing)}")

    records: list[tuple[int, str, float]] = []
    for year, groups in zip(included["publication_year"], included["benchmark_groups"]):
        retained = [group for group in tuple(groups or ()) if group in benchmark_groups]
        if not retained:
            continue
        weight = 1.0 / len(retained)
        for group in retained:
            records.append((int(year), group, weight))
    if not records:
        raise TaxaSkewError("No publications remain after filtering; cannot build a trend.")

    long = pd.DataFrame(records, columns=["publication_year", "broad_group", "weight"])
    per_year = long.groupby(["publication_year", "broad_group"], as_index=False)[
        "weight"
    ].sum()
    per_year = per_year.rename(columns={"weight": "effective_publication_count"})
    totals = per_year.groupby("publication_year")["effective_publication_count"].transform("sum")
    per_year["fractional_attention_share_pct"] = (
        per_year["effective_publication_count"] / totals * 100.0
    )
    per_year["n_effective_publications_year"] = totals

    complete = pd.MultiIndex.from_product(
        [sorted(per_year["publication_year"].unique()), benchmark_groups],
        names=["publication_year", "broad_group"],
    )
    per_year = (
        per_year.set_index(["publication_year", "broad_group"])
        .reindex(complete)
        .fillna({"effective_publication_count": 0.0, "fractional_attention_share_pct": 0.0})
        .reset_index()
    )

    yearly_totals = per_year.groupby("publication_year")[
        "fractional_attention_share_pct"
    ].sum()
    if not np.allclose(yearly_totals.to_numpy(), 100.0):
        raise TaxaSkewError("Yearly attention shares must each sum to 100%.")
    return per_year.sort_values(["broad_group", "publication_year"]).reset_index(drop=True)


def summarise_attention_trend(
    trend: pd.DataFrame,
    *,
    described_shares: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Per-group start, end, and ordinary-least-squares slope in points per decade.

    ``described_shares`` are the static described-diversity percentages; when given,
    the ratio of attention to described diversity is reported for the first and last
    year so the reader can see whether a gap closed or widened.
    """
    rows: list[dict[str, Any]] = []
    for group, block in trend.groupby("broad_group", sort=False):
        block = block.sort_values("publication_year")
        years = block["publication_year"].to_numpy(dtype=float)
        shares = block["fractional_attention_share_pct"].to_numpy(dtype=float)
        slope = float(np.polyfit(years, shares, 1)[0]) * 10.0
        record = {
            "broad_group": group,
            "start_year": int(years[0]),
            "end_year": int(years[-1]),
            "start_share_pct": float(shares[0]),
            "end_share_pct": float(shares[-1]),
            "change_pp": float(shares[-1] - shares[0]),
            "slope_pp_per_decade": slope,
        }
        if described_shares is not None and group in described_shares:
            described = float(described_shares[group])
            record["described_species_share_pct"] = described
            record["start_representation_ratio"] = float(shares[0]) / described
            record["end_representation_ratio"] = float(shares[-1]) / described
        rows.append(record)
    return pd.DataFrame(rows)


__all__ = [
    "DescribedDiversity",
    "TaxaSkewAnalysis",
    "TaxaSkewError",
    "TaxaSkewEvidence",
    "TaxaSkewResultStore",
    "analyze_taxonomic_skew",
    "summarise_attention_trend",
    "taxonomic_attention_trend",
]

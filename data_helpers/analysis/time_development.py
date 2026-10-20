"""Reusable table calculations for temporal biodiversity-loss results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_helpers.labels import parse_list_labels


def prepare_loss_evidence(
    corpus_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    threat_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter biodiversity loss and build unique document–year–threat rows."""
    if corpus_df["UT"].isna().any() or not corpus_df["UT"].is_unique:
        raise ValueError("The merged corpus must contain one non-missing row per UT.")

    loss_df = corpus_df.loc[corpus_df["s2_dir"].eq("negative")].copy()
    loss_df["publication_year"] = pd.to_numeric(
        loss_df["publication_year"], errors="coerce"
    ).astype("Int64")
    in_window = loss_df["publication_year"].between(start_year, end_year)

    threat_long = loss_df.loc[
        in_window, ["UT", "publication_year", "pred_threat_l0"]
    ].copy()
    threat_long["pred_threat_l0"] = threat_long["pred_threat_l0"].map(
        parse_list_labels
    )
    documents_without_threat = int(
        threat_long["pred_threat_l0"].map(len).eq(0).sum()
    )
    threat_long = (
        threat_long.explode("pred_threat_l0")
        .dropna(subset=["publication_year", "pred_threat_l0"])
    )
    threat_long["pred_threat_l0"] = (
        threat_long["pred_threat_l0"].astype("string").str.strip()
    )
    threat_long = (
        threat_long.loc[threat_long["pred_threat_l0"].ne("")]
        .drop_duplicates(["UT", "publication_year", "pred_threat_l0"])
    )
    threat_long["publication_year"] = threat_long["publication_year"].astype(int)

    observed = set(threat_long["pred_threat_l0"])
    unconfigured = sorted(observed.difference(threat_order))
    if unconfigured:
        raise ValueError(f"Unconfigured pred_threat_l0 labels: {unconfigured}")

    audit = pd.DataFrame(
        {
            "metric": [
                "Eligible publications",
                "Biodiversity-loss publications",
                f"Loss publications in {start_year}–{end_year}",
                "Loss publications outside window or missing year",
                "In-window publications without usable threat",
                "Unique document–threat assignments",
                "Observed threat classes",
            ],
            "value": [
                len(corpus_df),
                len(loss_df),
                int(loss_df.loc[in_window, "UT"].nunique()),
                int((~in_window).sum()),
                documents_without_threat,
                len(threat_long),
                len(observed),
            ],
        }
    )
    return loss_df, threat_long, audit


def annual_publication_tables(
    loss_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build complete-year publication counts and their annual summary table."""
    in_window = loss_df["publication_year"].between(start_year, end_year)
    annual_counts = (
        loss_df.loc[in_window]
        .assign(publication_year=lambda frame: frame["publication_year"].astype(int))
        .groupby("publication_year")["UT"]
        .nunique()
    )
    full_years = pd.RangeIndex(start_year, end_year + 1, name="publication_year")
    annual_counts = annual_counts.reindex(full_years, fill_value=0).rename(
        "n_publications"
    )

    annual_table = annual_counts.reset_index()
    annual_table["cumulative_publications"] = annual_table[
        "n_publications"
    ].cumsum()
    annual_table["yoy_growth_pct"] = (
        annual_table["n_publications"].pct_change() * 100
    ).round(2)
    return annual_counts, annual_table


def threat_composition_tables(
    threat_long: pd.DataFrame,
    annual_counts: pd.Series,
    threat_order: list[str],
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build yearly fractional publication counts, 100% shares, and an audit.

    A publication assigned to ``k`` distinct threats contributes ``1 / k`` to
    each threat, so every publication contributes one unit in total per year.
    """
    observed_order = [
        threat
        for threat in threat_order
        if threat in set(threat_long["pred_threat_l0"])
    ]
    weighted = threat_long.copy()
    weighted["labels_per_publication"] = weighted.groupby(
        ["UT", "publication_year"], observed=True
    )["pred_threat_l0"].transform("size")
    weighted["fractional_weight"] = 1.0 / weighted["labels_per_publication"]

    counts = (
        weighted.groupby(
            ["publication_year", "pred_threat_l0"], observed=True
        )["fractional_weight"]
        .sum()
        .rename("fractional_publication_count")
        .reset_index()
        .pivot(
            index="publication_year",
            columns="pred_threat_l0",
            values="fractional_publication_count",
        )
        .reindex(
            index=range(start_year, end_year + 1),
            columns=observed_order,
            fill_value=0,
        )
        .fillna(0)
        .astype(float)
    )
    counts.index.name = "publication_year"
    annual_denominator = annual_counts.reindex(counts.index).astype(float)
    if not np.allclose(counts.sum(axis=1), annual_denominator):
        raise ValueError(
            "Fractional threat weights must sum to one per publication and year."
        )
    shares = (counts.div(annual_denominator, axis=0) * 100).round(4)
    assert np.allclose(
        shares.sum(axis=1).loc[counts.sum(axis=1) > 0], 100.0
    )

    long = (
        counts.reset_index()
        .melt(
            id_vars="publication_year",
            var_name="pred_threat_l0",
            value_name="fractional_publication_count",
        )
        .merge(
            shares.reset_index().melt(
                id_vars="publication_year",
                var_name="pred_threat_l0",
                value_name="share_pct",
            ),
            on=["publication_year", "pred_threat_l0"],
        )
        .sort_values(["publication_year", "pred_threat_l0"])
        .reset_index(drop=True)
    )
    audit = pd.DataFrame(
        {
            "publication_year": counts.index,
            "n_publications": annual_denominator.astype(int).to_numpy(),
            "n_assignments": weighted.groupby("publication_year").size()
            .reindex(counts.index, fill_value=0)
            .astype(int)
            .to_numpy(),
            "fractional_publications": counts.sum(axis=1).to_numpy(),
        }
    )
    audit["labels_per_publication"] = (
        audit["n_assignments"] / audit["n_publications"]
    ).round(3)
    return counts, shares, long, audit

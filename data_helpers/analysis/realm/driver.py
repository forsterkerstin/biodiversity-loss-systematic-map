"""Minimal document-level direct-driver summaries by ecosystem realm.

The analysis retains negative-impact publications from complete years 2000–2025
when the L4 output contains exactly one ecological realm label. The ten
ecological labels remain mutually exclusive. Administrative/aggregate labels and
outputs carrying multiple distinct realm labels are excluded and audited.

IPBES L1 drivers are multi-label. Document prevalence therefore need not sum to
one. The complementary fractional composition gives every publication total
weight one across its mapped drivers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from data_helpers.labels import parse_list_labels


CORE_REALMS = ("Terrestrial", "Freshwater", "Marine")
ANALYSIS_REALMS = (
    "Terrestrial",
    "Subterranean",
    "Subterranean-Freshwater",
    "Subterranean-Marine",
    "Freshwater-Terrestrial",
    "Freshwater",
    "Freshwater-Marine",
    "Marine",
    "Marine-Terrestrial",
    "Marine-Freshwater-Terrestrial",
)
SPECIAL_REALMS = ("Not Applicable", "All realms")
CODED_REALMS = (*ANALYSIS_REALMS, *SPECIAL_REALMS)
IPBES_DRIVERS = (
    "Land/sea use change",
    "Direct Exploitation and Resource Extraction",
    "Climate Change",
    "Pollution",
    "Invasive alien species",
)
_DRIVER_CANONICAL = {label.casefold(): label for label in IPBES_DRIVERS}
_REALM_CANONICAL = {label.casefold(): label for label in CODED_REALMS}


class DriverRealmError(ValueError):
    """Raised when input data violate the realm-analysis contract."""


@dataclass
class DriverRealmEvidence:
    """One row per included publication and its aligned binary driver matrix."""

    documents: pd.DataFrame
    driver_matrix: pd.DataFrame
    audit: pd.DataFrame
    driver_order: tuple[str, ...] = IPBES_DRIVERS
    realm_order: tuple[str, ...] = ANALYSIS_REALMS


@dataclass
class RealmEvidencePreparation:
    """Notebook-facing evidence bundle with the realm-exclusion audit."""

    evidence: DriverRealmEvidence
    primary: pd.DataFrame
    audit: pd.DataFrame
    realm_exclusions: pd.DataFrame


def _validate_driver_order(driver_order: Sequence[str]) -> tuple[str, ...]:
    order = tuple(str(label) for label in driver_order)
    if order != IPBES_DRIVERS:
        raise DriverRealmError(
            "driver_order must match the five configured IPBES L1 drivers."
        )
    return order


def _validate_realm_order(realm_order: Sequence[str]) -> tuple[str, ...]:
    order = tuple(str(label) for label in realm_order)
    if not order:
        raise DriverRealmError("realm_order must contain at least one realm.")
    if len(order) != len(set(order)):
        raise DriverRealmError("realm_order must contain unique labels.")
    unexpected = sorted(set(order).difference(ANALYSIS_REALMS))
    if unexpected:
        raise DriverRealmError(
            "realm_order contains labels outside the ecological L4 taxonomy: "
            f"{unexpected}"
        )
    return order


def _canonical_driver_labels(value: Any) -> tuple[str, ...]:
    canonical: list[str] = []
    unknown: list[str] = []
    for label in parse_list_labels(value):
        resolved = _DRIVER_CANONICAL.get(label.casefold())
        if resolved is None:
            unknown.append(label)
        elif resolved not in canonical:
            canonical.append(resolved)
    if unknown:
        raise DriverRealmError(
            f"Unexpected IPBES driver label(s): {sorted(unknown)}"
        )
    return tuple(canonical)


def _canonical_realm_labels(value: Any) -> tuple[str, ...]:
    return tuple(
        _REALM_CANONICAL.get(label.casefold(), str(label).strip())
        for label in parse_list_labels(value)
    )


def _realm_bucket(
    labels: tuple[str, ...],
    realm_order: Sequence[str],
) -> str:
    if not labels:
        return "Empty realm label"
    if len(labels) > 1:
        return "Multiple realm labels"
    if labels[0] in realm_order:
        return "Exact singleton analysis realm"
    return f"Excluded realm: {labels[0]}"


def _audit_frame(rows: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", "value"])


def prepare_driver_realm_evidence(
    corpus_df: pd.DataFrame,
    *,
    driver_order: Sequence[str] = IPBES_DRIVERS,
    realm_order: Sequence[str] = ANALYSIS_REALMS,
    start_year: int = 2000,
    end_year: int = 2025,
    direction: str = "negative",
) -> DriverRealmEvidence:
    """Create the audited exact-singleton ecological-realm evidence universe."""
    required = {"UT", "s2_dir", "publication_year", "driver", "realm"}
    missing = sorted(required.difference(corpus_df.columns))
    if missing:
        raise DriverRealmError(f"Corpus is missing required column(s): {missing}")
    if corpus_df["UT"].isna().any():
        raise DriverRealmError("The corpus has missing UT values.")
    if not corpus_df["UT"].is_unique:
        examples = (
            corpus_df.loc[corpus_df["UT"].duplicated(keep=False), "UT"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise DriverRealmError(
            "The corpus must be one row per UT; duplicated examples: "
            f"{examples}"
        )
    if start_year > end_year:
        raise DriverRealmError("start_year must not exceed end_year.")

    drivers = _validate_driver_order(driver_order)
    realms = _validate_realm_order(realm_order)
    work = corpus_df[
        ["UT", "s2_dir", "publication_year", "driver", "realm"]
    ].copy()
    work["publication_year"] = pd.to_numeric(
        work["publication_year"], errors="coerce"
    ).astype("Int64")

    direction_mask = work["s2_dir"].astype("string").str.strip().eq(direction)
    directional = work.loc[direction_mask].copy()
    year_mask = directional["publication_year"].between(start_year, end_year)
    in_window = directional.loc[year_mask].copy()
    in_window["_realm_labels"] = in_window["realm"].map(
        _canonical_realm_labels
    )
    in_window["_realm_bucket"] = in_window["_realm_labels"].map(
        lambda labels: _realm_bucket(labels, realms)
    )
    realm_bucket_counts = (
        in_window.groupby("_realm_bucket", observed=True)["UT"]
        .nunique()
        .sort_values(ascending=False)
    )

    included_realm = in_window["_realm_labels"].map(
        lambda labels: len(labels) == 1 and labels[0] in realms
    )
    analytical = in_window.loc[included_realm].copy()
    analytical["_driver_labels"] = analytical["driver"].map(
        _canonical_driver_labels
    )
    no_driver = analytical["_driver_labels"].map(len).eq(0)
    analytical = analytical.loc[~no_driver].copy()
    analytical["realm"] = analytical["_realm_labels"].map(
        lambda labels: labels[0]
    )
    analytical["publication_year"] = analytical["publication_year"].astype(int)
    analytical["driver_labels"] = analytical["_driver_labels"]
    documents = analytical[
        ["UT", "publication_year", "realm", "driver_labels"]
    ].reset_index(drop=True)
    documents["realm"] = pd.Categorical(
        documents["realm"], categories=realms, ordered=True
    )

    driver_matrix = pd.DataFrame(
        {
            driver: documents["driver_labels"].map(
                lambda labels: driver in labels
            )
            for driver in drivers
        },
        dtype=bool,
    )
    documents["n_drivers"] = driver_matrix.sum(axis=1).astype(int)

    audit_rows = [
        ("Eligible corpus publications", int(len(work))),
        (f"Publications with direction = {direction}", int(len(directional))),
        (
            f"Direction-filtered publications in {start_year}–{end_year}",
            int(len(in_window)),
        ),
        (
            "Direction-filtered publications outside window or missing year",
            int((~year_mask).sum()),
        ),
    ]
    audit_rows.extend(
        (f"Realm audit — {bucket}", int(value))
        for bucket, value in realm_bucket_counts.items()
    )
    audit_rows.extend(
        [
            (
                "Analysis-realm publications without a usable driver",
                int(no_driver.sum()),
            ),
            ("Analysis publications", int(len(documents))),
            (
                "Multi-driver analysis publications",
                int(documents["n_drivers"].gt(1).sum()),
            ),
            (
                "Unique document–driver assignments",
                int(driver_matrix.to_numpy().sum()),
            ),
        ]
    )
    evidence = DriverRealmEvidence(
        documents=documents,
        driver_matrix=driver_matrix,
        audit=_audit_frame(audit_rows),
        driver_order=drivers,
        realm_order=realms,
    )
    _validate_evidence(evidence)
    return evidence


def _validate_evidence(evidence: DriverRealmEvidence) -> None:
    if len(evidence.documents) != len(evidence.driver_matrix):
        raise DriverRealmError("Documents and driver_matrix are not row-aligned.")
    if not evidence.documents.index.equals(evidence.driver_matrix.index):
        raise DriverRealmError("Documents and driver_matrix indices are not aligned.")
    if tuple(evidence.driver_matrix.columns) != evidence.driver_order:
        raise DriverRealmError("driver_matrix columns do not match driver_order.")
    if evidence.documents["UT"].isna().any() or not evidence.documents["UT"].is_unique:
        raise DriverRealmError(
            "Prepared evidence must contain unique, non-missing UT."
        )
    if evidence.driver_matrix.sum(axis=1).lt(1).any():
        raise DriverRealmError(
            "Every prepared publication must carry at least one driver."
        )
    if not set(
        evidence.documents["realm"].astype("string").dropna()
    ).issubset(evidence.realm_order):
        raise DriverRealmError(
            "Prepared evidence contains a realm outside realm_order."
        )


def prepare_realm_evidence(
    corpus_df: pd.DataFrame,
    *,
    driver_order: Sequence[str] = IPBES_DRIVERS,
    analysis_realms: Sequence[str] = ANALYSIS_REALMS,
    direction: str = "negative",
    start_year: int = 2000,
    end_year: int = 2025,
) -> RealmEvidencePreparation:
    """Return the minimal notebook-facing evidence and exclusion audit."""
    evidence = prepare_driver_realm_evidence(
        corpus_df,
        driver_order=driver_order,
        realm_order=analysis_realms,
        direction=direction,
        start_year=start_year,
        end_year=end_year,
    )
    realm_exclusions = (
        evidence.audit.loc[
            evidence.audit["metric"].str.startswith("Realm audit — ")
            & ~evidence.audit["metric"].eq(
                "Realm audit — Exact singleton analysis realm"
            )
        ]
        .assign(
            realm_category=lambda frame: frame["metric"].str.replace(
                "Realm audit — ", "", regex=False
            )
        )[["realm_category", "value"]]
        .rename(columns={"value": "n_documents"})
        .reset_index(drop=True)
    )
    return RealmEvidencePreparation(
        evidence=evidence,
        primary=evidence.documents.copy(),
        audit=evidence.audit.copy(),
        realm_exclusions=realm_exclusions,
    )


def realm_counts(evidence: DriverRealmEvidence) -> pd.DataFrame:
    """Return one unique-publication denominator per configured realm."""
    _validate_evidence(evidence)
    counts = (
        evidence.documents.groupby("realm", observed=True)["UT"]
        .nunique()
        .reindex(evidence.realm_order, fill_value=0)
    )
    if counts.eq(0).any():
        missing = counts.index[counts.eq(0)].tolist()
        raise DriverRealmError(
            f"No publications are available for configured realm(s): {missing}"
        )
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "realm": list(evidence.realm_order),
            "n_documents": counts.to_numpy(int),
            "share_of_analysis_documents": counts.to_numpy(float) / total,
        }
    )


def driver_realm_summary(evidence: DriverRealmEvidence) -> pd.DataFrame:
    """Calculate driver prevalence and fractional composition per realm."""
    _validate_evidence(evidence)
    documents = evidence.documents.reset_index(drop=True)
    matrix = evidence.driver_matrix.astype(float).reset_index(drop=True)
    fractional = matrix.div(matrix.sum(axis=1), axis=0)

    rows: list[dict[str, Any]] = []
    for realm in evidence.realm_order:
        mask = documents["realm"].astype("string").eq(realm).to_numpy()
        n_documents = int(mask.sum())
        if n_documents == 0:
            raise DriverRealmError(
                f"No publications are available for configured realm: {realm}"
            )
        realm_matrix = matrix.loc[mask]
        realm_fractional = fractional.loc[mask]
        for driver in evidence.driver_order:
            n_driver = int(realm_matrix[driver].sum())
            prevalence = n_driver / n_documents
            fractional_share = float(realm_fractional[driver].mean())
            rows.append(
                {
                    "realm": realm,
                    "driver": driver,
                    "n_documents": n_documents,
                    "n_driver_documents": n_driver,
                    "prevalence": prevalence,
                    "prevalence_pct": 100 * prevalence,
                    "fractional_share": fractional_share,
                    "fractional_share_pct": 100 * fractional_share,
                }
            )
    return pd.DataFrame(rows)


def prepare_pollution_nameability_evidence(
    evidence: DriverRealmEvidence,
    corpus_df: pd.DataFrame,
    *,
    realm_order: Sequence[str] = CORE_REALMS,
) -> pd.DataFrame:
    """Create core-realm records for pollution, plastics, and exploitation trends."""
    _validate_evidence(evidence)
    realms = tuple(str(realm) for realm in realm_order)
    if not realms or len(realms) != len(set(realms)):
        raise DriverRealmError("realm_order must contain unique realm labels.")
    unexpected_realms = sorted(set(realms).difference(evidence.realm_order))
    if unexpected_realms:
        raise DriverRealmError(
            f"realm_order contains unconfigured realms: {unexpected_realms}"
        )
    required = {"UT", "text_available", "plastics_mention"}
    missing = sorted(required.difference(corpus_df.columns))
    if missing:
        raise DriverRealmError(
            f"Corpus is missing text-feature column(s): {missing}"
        )
    if corpus_df["UT"].isna().any() or not corpus_df["UT"].is_unique:
        raise DriverRealmError(
            "The text corpus must contain unique, non-missing UT values."
        )

    documents = evidence.documents.reset_index(drop=True)
    matrix = evidence.driver_matrix.reset_index(drop=True)
    realm_mask = documents["realm"].astype("string").isin(realms)
    selected = documents.loc[
        realm_mask, ["UT", "publication_year", "realm", "n_drivers"]
    ].copy()
    selected["realm"] = pd.Categorical(
        selected["realm"].astype("string"),
        categories=realms,
        ordered=True,
    )
    selected["pollution_present"] = matrix.loc[
        realm_mask, "Pollution"
    ].to_numpy(bool)
    selected["exploitation_present"] = matrix.loc[
        realm_mask, "Direct Exploitation and Resource Extraction"
    ].to_numpy(bool)
    selected["pollution_fractional_weight"] = (
        selected["pollution_present"].astype(float) / selected["n_drivers"]
    )
    selected["exploitation_fractional_weight"] = (
        selected["exploitation_present"].astype(float)
        / selected["n_drivers"]
    )

    text_features = corpus_df[["UT", "text_available", "plastics_mention"]].copy()
    selected = selected.merge(
        text_features,
        on="UT",
        how="left",
        validate="one_to_one",
    )
    return selected[
        [
            "UT",
            "publication_year",
            "realm",
            "n_drivers",
            "pollution_present",
            "exploitation_present",
            "pollution_fractional_weight",
            "exploitation_fractional_weight",
            "text_available",
            "plastics_mention",
        ]
    ].reset_index(drop=True)


def annual_pollution_nameability_summary(
    evidence: pd.DataFrame,
    *,
    realm_order: Sequence[str] = CORE_REALMS,
) -> pd.DataFrame:
    """Summarize annual fractional attention and plastics-within-pollution."""
    required = {
        "UT",
        "publication_year",
        "realm",
        "pollution_present",
        "pollution_fractional_weight",
        "exploitation_fractional_weight",
        "text_available",
        "plastics_mention",
    }
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise DriverRealmError(
            f"Nameability evidence is missing column(s): {missing}"
        )
    realms = tuple(str(realm) for realm in realm_order)
    outcomes = (
        ("Pollution fractional attention", "pollution_fractional_weight"),
        (
            "Direct exploitation fractional attention",
            "exploitation_fractional_weight",
        ),
    )
    rows: list[dict[str, Any]] = []
    for realm in realms:
        realm_frame = evidence.loc[
            evidence["realm"].astype("string").eq(realm)
        ]
        for year, year_frame in realm_frame.groupby(
            "publication_year", observed=True, sort=True
        ):
            n_documents = int(year_frame["UT"].nunique())
            for outcome, value_column in outcomes:
                numerator = float(year_frame[value_column].sum())
                rows.append(
                    {
                        "realm": realm,
                        "publication_year": int(year),
                        "outcome": outcome,
                        "n_denominator": n_documents,
                        "numerator": numerator,
                        "share": numerator / n_documents,
                        "share_pct": 100 * numerator / n_documents,
                    }
                )
            pollution_text = year_frame.loc[
                year_frame["pollution_present"]
                & year_frame["text_available"]
            ]
            n_pollution_text = int(pollution_text["UT"].nunique())
            if n_pollution_text:
                n_plastics = int(pollution_text["plastics_mention"].sum())
                rows.append(
                    {
                        "realm": realm,
                        "publication_year": int(year),
                        "outcome": "Plastics within pollution",
                        "n_denominator": n_pollution_text,
                        "numerator": n_plastics,
                        "share": n_plastics / n_pollution_text,
                        "share_pct": 100 * n_plastics / n_pollution_text,
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "ANALYSIS_REALMS",
    "CODED_REALMS",
    "CORE_REALMS",
    "DriverRealmError",
    "DriverRealmEvidence",
    "IPBES_DRIVERS",
    "RealmEvidencePreparation",
    "SPECIAL_REALMS",
    "driver_realm_summary",
    "annual_pollution_nameability_summary",
    "prepare_pollution_nameability_evidence",
    "prepare_driver_realm_evidence",
    "prepare_realm_evidence",
    "realm_counts",
]

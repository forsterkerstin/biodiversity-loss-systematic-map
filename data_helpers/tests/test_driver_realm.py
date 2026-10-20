"""Focused tests for the minimal direct-driver × realm analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_helpers.analysis.realm import driver as dr


LAND = "Land/sea use change"
EXPLOIT = "Direct Exploitation and Resource Extraction"
CLIMATE = "Climate Change"
POLLUTION = "Pollution"


def _record(
    ut: str,
    *,
    year: int = 2000,
    direction: str = "negative",
    realm: object = '["Terrestrial"]',
    drivers: object = '["Land/sea use change"]',
) -> dict[str, object]:
    return {
        "UT": ut,
        "s2_dir": direction,
        "publication_year": year,
        "driver": drivers,
        "realm": realm,
    }


def test_preparation_keeps_all_singleton_ecological_realms_and_audits_exits() -> None:
    records = [
        _record(
            f"realm-{index}",
            realm=[realm],
            drivers=[LAND, POLLUTION] if index == 0 else [LAND],
        )
        for index, realm in enumerate(dr.ANALYSIS_REALMS)
    ]
    records.extend(
        [
            _record("not-applicable", realm=["Not Applicable"]),
            _record("all-realms", realm=["All realms"]),
            _record("multiple", realm=["Terrestrial", "Freshwater"]),
            _record("driverless", drivers=[]),
            _record("positive", direction="positive"),
            _record("partial", year=2026),
        ]
    )

    prepared = dr.prepare_realm_evidence(pd.DataFrame(records))
    evidence = prepared.evidence

    assert evidence.realm_order == dr.ANALYSIS_REALMS
    assert evidence.documents["realm"].astype("string").tolist() == list(
        dr.ANALYSIS_REALMS
    )
    assert evidence.driver_matrix.to_numpy().sum() == 11
    audit = evidence.audit.set_index("metric")["value"]
    assert audit["Realm audit — Exact singleton analysis realm"] == 11
    assert audit["Realm audit — Excluded realm: Not Applicable"] == 1
    assert audit["Realm audit — Excluded realm: All realms"] == 1
    assert audit["Realm audit — Multiple realm labels"] == 1
    assert audit["Analysis-realm publications without a usable driver"] == 1
    assert prepared.realm_exclusions["n_documents"].sum() == 3


def test_summary_separates_prevalence_and_fractional_composition() -> None:
    evidence = dr.prepare_driver_realm_evidence(
        pd.DataFrame(
            [
                _record("T1", drivers=[LAND, POLLUTION]),
                _record("T2", drivers=[LAND]),
                _record("F1", realm=["Freshwater"], drivers=[POLLUTION]),
                _record(
                    "F2",
                    realm=["Freshwater"],
                    drivers=[POLLUTION, CLIMATE],
                ),
                _record(
                    "M1",
                    realm=["Marine"],
                    drivers=[EXPLOIT, CLIMATE],
                ),
                _record("M2", realm=["Marine"], drivers=[EXPLOIT]),
            ]
        ),
        realm_order=dr.CORE_REALMS,
    )
    summary = dr.driver_realm_summary(evidence).set_index(["realm", "driver"])

    assert summary.at[("Terrestrial", LAND), "prevalence"] == pytest.approx(1)
    assert summary.at[("Terrestrial", POLLUTION), "prevalence"] == pytest.approx(
        0.5
    )
    assert summary.at[
        ("Terrestrial", LAND), "fractional_share"
    ] == pytest.approx(0.75)
    assert summary.at[
        ("Terrestrial", POLLUTION), "fractional_share"
    ] == pytest.approx(0.25)
    totals = summary["fractional_share"].unstack("driver").sum(axis=1)
    assert np.allclose(totals, 1)


def test_realm_counts_use_unique_publication_denominators() -> None:
    evidence = dr.prepare_driver_realm_evidence(
        pd.DataFrame(
            [
                _record("T1"),
                _record("T2"),
                _record("F1", realm=["Freshwater"]),
                _record("M1", realm=["Marine"]),
            ]
        ),
        realm_order=dr.CORE_REALMS,
    )
    counts = dr.realm_counts(evidence).set_index("realm")
    assert counts.at["Terrestrial", "n_documents"] == 2
    assert counts["share_of_analysis_documents"].sum() == pytest.approx(1)


def test_preparation_rejects_duplicate_ut_and_unknown_driver() -> None:
    with pytest.raises(dr.DriverRealmError, match="one row per UT"):
        dr.prepare_driver_realm_evidence(
            pd.DataFrame([_record("A"), _record("A")])
        )
    with pytest.raises(dr.DriverRealmError, match="Unexpected IPBES driver"):
        dr.prepare_driver_realm_evidence(
            pd.DataFrame([_record("A", drivers=["Invented driver"])])
        )


def test_pollution_nameability_uses_text_and_fractional_driver_weights() -> None:
    records = [
        _record("T1", drivers=[POLLUTION]),
        _record("T2", drivers=[POLLUTION, LAND]),
        _record("T3", year=2001, drivers=[EXPLOIT]),
        _record("F1", realm=["Freshwater"], drivers=[POLLUTION]),
        _record("M1", realm=["Marine"], drivers=[LAND]),
    ]
    corpus = pd.DataFrame(records).assign(
        # Precomputed at prep time (see test_evidence_corpus_prep.py for the regex itself).
        text_available=[True, True, True, True, True],
        plastics_mention=[True, False, False, True, False],
    )
    evidence = dr.prepare_driver_realm_evidence(
        corpus,
        realm_order=dr.CORE_REALMS,
    )
    nameability = dr.prepare_pollution_nameability_evidence(
        evidence,
        corpus,
    ).set_index("UT")

    assert bool(nameability.at["T1", "plastics_mention"])
    assert not bool(nameability.at["T2", "plastics_mention"])
    assert bool(nameability.at["F1", "plastics_mention"])
    assert nameability.at[
        "T2", "pollution_fractional_weight"
    ] == pytest.approx(0.5)

    annual = dr.annual_pollution_nameability_summary(
        nameability.reset_index()
    ).set_index(["realm", "publication_year", "outcome"])
    assert annual.at[
        ("Terrestrial", 2000, "Pollution fractional attention"), "share"
    ] == pytest.approx(0.75)
    assert annual.at[
        ("Terrestrial", 2000, "Plastics within pollution"), "share"
    ] == pytest.approx(0.5)
    assert annual.at[
        (
            "Terrestrial",
            2001,
            "Direct exploitation fractional attention",
        ),
        "share",
    ] == pytest.approx(1)

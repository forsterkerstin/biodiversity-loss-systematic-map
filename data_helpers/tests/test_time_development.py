"""Tests for temporal evidence-growth calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from data_helpers.analysis import time_development


def test_threat_composition_fractionalizes_each_publication_to_one() -> None:
    threat_long = pd.DataFrame(
        {
            "UT": ["A", "A", "B", "C"],
            "publication_year": [2000, 2000, 2000, 2001],
            "pred_threat_l0": ["Threat A", "Threat B", "Threat A", "Threat B"],
        }
    )
    annual_counts = pd.Series(
        [2, 1], index=pd.Index([2000, 2001], name="publication_year")
    )

    counts, shares, long, audit = time_development.threat_composition_tables(
        threat_long,
        annual_counts,
        ["Threat A", "Threat B"],
        2000,
        2001,
    )

    assert counts.loc[2000, "Threat A"] == pytest.approx(1.5)
    assert counts.loc[2000, "Threat B"] == pytest.approx(0.5)
    assert counts.loc[2001, "Threat A"] == pytest.approx(0.0)
    assert counts.loc[2001, "Threat B"] == pytest.approx(1.0)
    assert counts.sum(axis=1).tolist() == pytest.approx([2.0, 1.0])
    assert shares.loc[2000].tolist() == pytest.approx([75.0, 25.0])
    assert shares.loc[2001].tolist() == pytest.approx([0.0, 100.0])
    assert "fractional_publication_count" in long
    assert audit["n_assignments"].tolist() == [3, 1]
    assert audit["fractional_publications"].tolist() == pytest.approx([2.0, 1.0])
    assert audit["labels_per_publication"].tolist() == pytest.approx([1.5, 1.0])

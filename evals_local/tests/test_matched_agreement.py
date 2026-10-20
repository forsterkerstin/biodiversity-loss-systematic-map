"""Record-matched LLM-versus-human coding agreement."""

from __future__ import annotations

import pandas as pd
import pytest

from evals_local import matched_agreement as ma

_THREATS_L1 = next(f for f in ma.FIELDS if f.field == "threats_l1")
_DRIVER = next(f for f in ma.FIELDS if f.field == "driver")


def _annotations(column: str, labels: list) -> pd.DataFrame:
    return pd.DataFrame({ma.UT_COLUMN: [f"WOS:{i}" for i in range(len(labels))], column: labels})


def test_records_without_a_detailed_threat_step_are_excluded() -> None:
    annotations = _annotations("threats_l1", ["['1.1 Housing & Urban Areas']"] * 3)
    predictions = pd.DataFrame(
        {"custom_id": ["WOS:0", "WOS:1"], "pred_threat_l1": ["['1.1 Housing & Urban Areas']"] * 2}
    )
    result = ma.score_field(_THREATS_L1, annotations, predictions)
    assert (result["without_output"], result["n"]) == (1, 2)
    assert result["f1"] == pytest.approx(1.0)


def test_empty_predictions_are_scored_and_missing_human_labels_skipped() -> None:
    annotations = _annotations("driver", ["['Pollution']", "['Pollution']", None])
    predictions = pd.DataFrame(
        {"custom_id": ["WOS:0", "WOS:1", "WOS:2"], "pred_driver": ["['Pollution']", "[]", "['Pollution']"]}
    )
    result = ma.score_field(_DRIVER, annotations, predictions)
    assert result["n"] == 2
    assert result["recall"] == pytest.approx(0.5)


def test_placeholder_labels_are_masked_on_either_side() -> None:
    field = next(f for f in ma.FIELDS if f.field == "threats_l0")
    annotations = _annotations("threats_l0", ["['Pollution']", "['Unclear']", "['Pollution']"])
    predictions = pd.DataFrame(
        {"custom_id": ["WOS:0", "WOS:1", "WOS:2"], "pred_threat_l0": ["['Pollution']", "['Pollution']", "['Unclear']"]}
    )
    assert ma.score_field(field, annotations, predictions)["n"] == 1


def test_fields_cover_the_five_dimensions() -> None:
    assert len(ma.FIELDS) == 13
    assert {f.dimension for f in ma.FIELDS} == {
        "Direct drivers", "Threats", "Geographic scope", "Ecosystem typology", "Taxonomy"
    }

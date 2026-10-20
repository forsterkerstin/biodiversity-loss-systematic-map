"""Pooled coding agreement with the published reference labels."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from evals_local import metrics, summary


def test_pool_weights_label_sets_equally() -> None:
    fields = pd.DataFrame(
        {
            "label_set": ["L1", "L2", "L2"],
            "field": ["a", "b", "c"],
            "precision": [1.0, 0.0, 0.5],
            "recall": [1.0, 0.0, 0.5],
            "f1": [1.0, 0.0, 0.5],
            "kappa": [1.0, 0.0, 0.5],
        }
    )
    by_label_set, overall = summary.pool(fields)
    assert by_label_set.loc["L2", "f1"] == pytest.approx(0.25)
    assert overall["f1"] == pytest.approx(0.625)


def _single_class_table() -> pd.DataFrame:
    # Every record is Chordata: its kappa is undefined. A spurious prediction of
    # a label with no reference support yields kappa 0, which must not stand in
    # for the field's kappa.
    frame = pd.DataFrame(
        {"truth": [["Chordata"]] * 4, "pred": [["Chordata"]] * 3 + [["Chordata", "Arthropoda"]]}
    )
    return metrics.label_metrics(frame, "truth", "pred", set)


def test_kappa_is_undefined_when_no_supported_label_has_one() -> None:
    table = _single_class_table()
    assert table.set_index("label").loc["_weighted", "kappa"] == pytest.approx(0.0)
    result = summary.weighted_field_metrics(table)
    assert math.isnan(result["kappa"])
    assert result["precision"] == pytest.approx(1.0)


def test_kappa_is_kept_when_a_supported_label_defines_it() -> None:
    frame = pd.DataFrame(
        {"truth": [["A"], ["A", "B"], ["B"]], "pred": [["A"], ["A"], ["A", "B"]]}
    )
    table = metrics.label_metrics(frame, "truth", "pred", set)
    weighted_kappa = table.set_index("label").loc["_weighted", "kappa"]
    assert summary.weighted_field_metrics(table)["kappa"] == pytest.approx(weighted_kappa)


def _write_test_metrics(eval_root: Path, field: summary.Field, table: pd.DataFrame) -> None:
    path = eval_root / field.run / "metrics" / field.metrics_file
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(path, index=False, engine="openpyxl")


def test_loaders_read_the_metric_workbook_layout(tmp_path: Path) -> None:
    field = summary.Field("L6", "phylum", "l6_test_run", "taxa_phylum_label_metrics.xlsx")
    table = _single_class_table()
    _write_test_metrics(tmp_path / "eval", field, table)

    row = summary.load_test_metrics(tmp_path / "eval", (field,)).iloc[0]
    assert (row["label_set"], row["field"]) == ("L6", "phylum")
    assert row["precision"] == pytest.approx(1.0)
    assert math.isnan(row["kappa"])
    assert row["n"] == 4


def test_kappa_is_undefined_when_every_record_has_the_same_reference_label() -> None:
    # As in the phylum test workbook: all records are Chordata, one is missed.
    # The per-label kappa is formally 0, but it carries no information.
    frame = pd.DataFrame(
        {"truth": [["Chordata"]] * 4, "pred": [["Chordata"]] * 3 + [["Arthropoda"]]}
    )
    table = metrics.label_metrics(frame, "truth", "pred", set)
    assert table.set_index("label").loc["chordata" if "chordata" in set(table["label"]) else "Chordata", "kappa"] == pytest.approx(0.0)
    assert math.isnan(summary.weighted_field_metrics(table)["kappa"])

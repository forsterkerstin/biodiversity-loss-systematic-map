"""Evaluation metrics against hand-computed confusion counts."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from evals_local import metrics
from evals_local.normalizers import (
    normalize_screening_label,
    normalize_threat_labels,
    to_label_set,
)


def _frame() -> pd.DataFrame:
    # The last record has no reference label and must not be scored.
    return pd.DataFrame(
        {
            "truth": [["A"], ["A", "B"], ["B"], []],
            "pred": [["A"], ["A"], ["A", "B"], ["A"]],
        }
    )


def test_label_metrics_match_hand_computed_values() -> None:
    table = metrics.label_metrics(_frame(), "truth", "pred", set).set_index("label")

    # Label A: tp=2 (rows 0, 1), fp=1 (row 2), fn=0, tn=0 over 3 scored rows.
    a = table.loc["A"]
    assert (a["tp"], a["fp"], a["fn"], a["tn"], a["n"]) == (2, 1, 0, 0, 3)
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["recall"] == pytest.approx(1.0)
    assert a["f1"] == pytest.approx(0.8)
    assert a["kappa"] == pytest.approx(0.0)

    # Label B: tp=1 (row 2), fn=1 (row 1), tn=1 (row 0).
    b = table.loc["B"]
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (1, 0, 1, 1)
    assert b["precision"] == pytest.approx(1.0)
    assert b["recall"] == pytest.approx(0.5)
    assert b["f1"] == pytest.approx(2 / 3)
    # Observed agreement 2/3, chance agreement 4/9 -> (6/9 - 4/9) / (5/9).
    assert b["kappa"] == pytest.approx(0.4)

    assert table.loc["_macro", "f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert table.loc["_weighted", "f1"] == pytest.approx((0.8 * 2 + 2 / 3 * 2) / 4)
    micro = table.loc["_micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (3, 1, 1)
    assert micro["precision"] == pytest.approx(0.75)
    assert micro["recall"] == pytest.approx(0.75)


def test_label_metrics_respects_include_mask_and_empty_truth() -> None:
    frame = _frame()
    mask = pd.Series([True, True, False, True])
    table = metrics.label_metrics(frame, "truth", "pred", set, include_mask=mask)
    assert set(table["n"].dropna()) <= {2, 4}
    per_label = table.set_index("label")
    assert per_label.loc["A", "n"] == 2
    assert per_label.loc["A", "fp"] == 0

    empty = metrics.label_metrics(frame.iloc[[3]], "truth", "pred", set)
    assert empty.empty


def test_record_confusion_skips_unlabelled_records() -> None:
    frame = metrics.record_confusion(_frame(), "truth", "pred", "x", set)
    assert frame["x_tp"].tolist()[:3] == [1, 1, 1]
    assert frame["x_fp"].tolist()[:3] == [0, 0, 1]
    assert frame["x_fn"].tolist()[:3] == [0, 1, 0]
    assert frame["x_tn"].tolist()[:3] == [1, 0, 0]
    assert frame.loc[3, "x_tp"] is None or math.isnan(frame.loc[3, "x_tp"])


def test_screening_summary_metrics_binary_counts() -> None:
    truth = ["ELIGIBLE", "ELIGIBLE", "NOT_ELIGIBLE", "NOT_ELIGIBLE", None]
    pred = ["ELIGIBLE", "NOT_ELIGIBLE", "ELIGIBLE", "NOT_ELIGIBLE", "ELIGIBLE"]
    summary, confusion = metrics.screening_summary_metrics(truth, pred)
    row = summary.iloc[0]
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 1)
    assert row["scored_rows"] == 4
    assert row["eligible_precision"] == pytest.approx(0.5)
    assert row["eligible_recall"] == pytest.approx(0.5)
    assert confusion["count"].sum() == 4


def test_screening_summary_omits_precision_for_positive_only_references() -> None:
    summary, _ = metrics.screening_summary_metrics(["ELIGIBLE"] * 2, ["ELIGIBLE", "NOT_ELIGIBLE"])
    assert list(summary.columns) == [
        "positive_label", "tp", "fp", "fn", "tn", "eligible_recall", "positive_support", "scored_rows",
    ]


def test_normalizers_canonicalise_labels() -> None:
    assert to_label_set('["Pollution", " pollution ", "Climate Change"]') == {
        "pollution",
        "climate change",
    }
    assert to_label_set(float("nan")) == set()
    assert normalize_threat_labels(["Housing & Urban Areas"]) == {"housing & urban area"}
    assert normalize_screening_label(["Not eligible"]) == "NOT_ELIGIBLE"
    assert normalize_screening_label("eligible") == "ELIGIBLE"
    assert normalize_screening_label(None) == ""

"""Tests for the UMAP embedding-map helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from data_helpers.analysis.embedding_map import (
    EmbeddingMapError,
    assign_labels,
    join_coordinates,
    plot_embedding_map,
)


def test_assign_labels_counts_unique_documents_and_picks_most_frequent():
    values = pd.Series([["A"], ["A", "B"], ["B"], ["A"], ["Not Applicable"], []])
    result = assign_labels(values)
    assert result.summary["label"].tolist() == ["A", "B"]
    assert result.summary["n_documents"].tolist() == [3, 2]
    assert result.summary["share_pct"].round(2).tolist() == [50.0, 33.33]
    # the [A, B] article takes A (more frequent); skipped/empty get code 0
    assert result.codes.tolist() == [1, 1, 2, 1, 0, 0]


def test_assign_labels_top_n_pools_the_rest_as_other_ranked_last():
    values = pd.Series([["A"], ["A"], ["B"], ["B", "C"], ["C"], ["D", "C"], ["D"], []])
    result = assign_labels(values, top_n=2, other_label="Other")
    assert result.summary["label"].tolist()[-1] == "Other"
    top = result.summary["label"].tolist()[:2]
    assert len(top) == 2 and "Other" not in top
    # an article naming a top label is not coloured as Other; empty stays 0
    assert result.codes[-1] == 0
    assert result.summary["n_documents"].iloc[-1] == int(
        pd.Series([[v for v in items if v not in top] for items in values]).map(bool).sum()
    )


def test_join_coordinates_requires_every_row_and_unique_ut():
    dataset = pd.DataFrame({"UT": ["a", "b"]})
    with pytest.raises(EmbeddingMapError):
        join_coordinates(dataset, pd.DataFrame({"UT": ["a"], "umap_x": [0.0], "umap_y": [0.0]}))
    with pytest.raises(EmbeddingMapError):
        join_coordinates(dataset, pd.DataFrame({"UT": ["a", "a"], "umap_x": [0, 0], "umap_y": [0, 0]}))


def test_plot_legend_carries_percentages_and_rejects_missing_colours():
    labelled = assign_labels(pd.Series([["A"]] * 30 + [["B"]] * 20))
    x = np.arange(50.0)
    fig = plot_embedding_map(x, x, labelled, colors={"A": "#111111", "B": "#eeeeee"})
    legend = fig.axes[0].get_legend()
    texts = [t.get_text() for t in legend.get_texts() if t.get_text()]
    assert texts == ["A (60.0%)", "B (40.0%)"]
    with pytest.raises(EmbeddingMapError):
        plot_embedding_map(x, x, labelled, colors={"A": "#111111"})

"""Tests for the shared publication-count choropleth scale."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from data_helpers.analysis.geo.count_plotting import (
    add_class_colorbar,
    build_count_scale,
    order_of_magnitude_bins,
)


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (0, [1]),
        (1, [1, 10]),
        (9, [1, 10]),
        (10, [1, 10, 100]),
        (25_690, [1, 10, 100, 1_000, 10_000, 100_000]),
    ],
)
def test_order_of_magnitude_bins_bracket_counts(
    maximum: int,
    expected: list[int],
) -> None:
    assert order_of_magnitude_bins(maximum) == expected


def test_count_scale_uses_the_fixed_ylgn_sampling() -> None:
    cmap, norm, bins = build_count_scale(25_690)

    assert bins == [1, 10, 100, 1_000, 10_000, 100_000]
    assert cmap.N == len(bins) - 1
    assert np.array_equal(norm.boundaries, np.asarray(bins))
    expected_colors = plt.get_cmap("YlGn")(np.linspace(0.12, 0.95, len(bins) - 1))
    assert np.allclose(cmap.colors, expected_colors)


def test_count_scale_accepts_explicit_boundaries_and_colors() -> None:
    bins = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000]
    colors = [
        "#FFFFD9",
        "#EDF8B1",
        "#C7E9B4",
        "#7FCDBB",
        "#41B6C4",
        "#1D91C0",
        "#225EA8",
        "#253494",
        "#081D58",
    ]

    cmap, norm, result_bins = build_count_scale(
        25_689,
        bins=bins,
        colors=colors,
    )

    assert result_bins == bins
    assert list(cmap.colors) == colors
    assert np.array_equal(norm.boundaries, np.asarray(bins))


def test_explicit_count_scale_requires_final_boundary_above_maximum() -> None:
    with pytest.raises(ValueError, match="not below the final boundary"):
        build_count_scale(
            30_000,
            bins=[1, 10, 100, 1_000, 10_000, 30_000],
        )


def test_class_colorbar_is_horizontal_uniform_and_formatted() -> None:
    cmap, norm, bins = build_count_scale(25_690)
    fig = plt.figure(figsize=(4, 1))
    cax = fig.add_axes([0.1, 0.3, 0.8, 0.3])

    colorbar = add_class_colorbar(
        fig,
        cax,
        cmap,
        norm,
        bins,
        "Articles per country",
        fontsize=8,
        tick_fontsize=5,
    )

    assert colorbar.orientation == "horizontal"
    assert colorbar.spacing == "uniform"
    assert colorbar.ax.get_xlabel() == "Articles per country"
    assert [tick.get_text() for tick in colorbar.ax.get_xticklabels()] == [
        "1",
        "10",
        "100",
        "1,000",
        "10,000",
        "100,000",
    ]
    assert all(
        tick.get_fontsize() == pytest.approx(5)
        for tick in colorbar.ax.get_xticklabels()
    )
    assert colorbar.ax.xaxis.label.get_fontsize() == pytest.approx(8)
    assert not colorbar.outline.get_visible()
    assert all(
        tick.tick1line.get_markersize() == pytest.approx(0)
        for tick in colorbar.ax.xaxis.majorTicks
    )
    plt.close(fig)

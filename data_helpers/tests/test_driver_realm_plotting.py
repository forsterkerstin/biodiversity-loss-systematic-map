"""Tests for the separated L1 heatmap and horizontal bar figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from data_helpers.analysis.realm.driver import ANALYSIS_REALMS, CORE_REALMS
from data_helpers.analysis.realm.driver_plotting import (
    plot_driver_fractional_bars,
    plot_pollution_nameability_trends,
)


DRIVERS = ("land", "exploitation", "climate", "pollution", "invasive")
REALMS = ANALYSIS_REALMS
SUPPORTS = {
    realm: np.int64(1_000 - 50 * position)
    for position, realm in enumerate(REALMS)
}
SUPPORTS["Subterranean-Marine"] = np.int64(36)
DRIVER_NAMES = {driver: driver.title() for driver in DRIVERS}
REALM_NAMES = {realm: realm for realm in REALMS}
DRIVER_COLORS = {
    "land": "#E69F00",
    "exploitation": "#009E73",
    "climate": "#0072B2",
    "pollution": "#CC79A7",
    "invasive": "#D55E00",
}
def _estimates() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = np.array((0.35, 0.15, 0.20, 0.20, 0.10))
    for realm_position, realm in enumerate(REALMS):
        fractional = np.roll(base, realm_position % len(DRIVERS))
        for driver_position, driver in enumerate(DRIVERS):
            rows.append(
                {
                    "realm": realm,
                    "driver": driver,
                    "fractional_share": fractional[driver_position],
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("realm_order", "height"),
    [(REALMS, 5.7), (CORE_REALMS, 3.1)],
)
def test_horizontal_bar_views_accept_all_and_core_realms(
    realm_order: tuple[str, ...],
    height: float,
) -> None:
    figure = plot_driver_fractional_bars(
        _estimates().loc[lambda frame: frame["realm"].isin(realm_order)],
        driver_order=DRIVERS,
        realm_order=realm_order,
        driver_names=DRIVER_NAMES,
        realm_names=REALM_NAMES,
        realm_supports=SUPPORTS,
        driver_colors=DRIVER_COLORS,
        figure_height=height,
        show_segment_labels=realm_order == CORE_REALMS,
        segment_label_decimals=1,
    )
    axis = figure.axes[0]
    assert figure.get_figheight() == pytest.approx(height)
    assert len(axis.patches) == len(realm_order) * len(DRIVERS)
    assert axis.get_xlim() == pytest.approx((0, 100))
    assert axis.get_xlabel() == ""
    assert len(axis.get_xticks()) == 0
    assert not axis.spines["bottom"].get_visible()
    assert all(
        patch.get_height() == pytest.approx(0.50)
        for patch in axis.patches
    )
    if realm_order == CORE_REALMS:
        assert axis.texts
        assert all(
            text.get_text().endswith(".0%") for text in axis.texts
        )
    else:
        assert not axis.texts
    plt.close(figure)


def test_bar_view_requires_fractional_shares_to_sum_to_one() -> None:
    estimates = _estimates()
    estimates.loc[0, "fractional_share"] = 0
    with pytest.raises(ValueError, match="sum to one"):
        plot_driver_fractional_bars(
            estimates,
            driver_order=DRIVERS,
            realm_order=REALMS,
            driver_names=DRIVER_NAMES,
            realm_names=REALM_NAMES,
            realm_supports=SUPPORTS,
            driver_colors=DRIVER_COLORS,
        )


def test_bar_labels_round_each_share_independently() -> None:
    estimates = _estimates()
    # Terrestrial driver shares of Figure 5; rounding each to one decimal gives
    # labels that sum to 99.9, which must not be adjusted to 100.
    shares = (0.32730756, 0.10529771, 0.19094703, 0.28623296, 0.09021474)
    for realm in CORE_REALMS:
        realm_rows = estimates["realm"].eq(realm)
        estimates.loc[realm_rows, "fractional_share"] = shares
    figure = plot_driver_fractional_bars(
        estimates.loc[lambda frame: frame["realm"].isin(CORE_REALMS)],
        driver_order=DRIVERS,
        realm_order=CORE_REALMS,
        driver_names=DRIVER_NAMES,
        realm_names=REALM_NAMES,
        realm_supports=SUPPORTS,
        driver_colors=DRIVER_COLORS,
        show_segment_labels=True,
        segment_label_decimals=1,
        segment_label_min_pct=0,
    )
    labels_by_realm = {
        position: [
            float(text.get_text().removesuffix("%"))
            for text in figure.axes[0].texts
            if text.get_position()[1] == position
        ]
        for position in range(len(CORE_REALMS))
    }
    for labels in labels_by_realm.values():
        assert sorted(labels) == [9.0, 10.5, 19.1, 28.6, 32.7]
    plt.close(figure)


def test_pollution_nameability_trend_figure_has_three_zero_based_panels() -> None:
    outcomes = (
        "Pollution fractional attention",
        "Plastics within pollution",
        "Direct exploitation fractional attention",
    )
    rows = []
    for realm_position, realm in enumerate(CORE_REALMS):
        for outcome_position, outcome in enumerate(outcomes):
            for year in range(2000, 2026):
                rows.append(
                    {
                        "realm": realm,
                        "publication_year": year,
                        "outcome": outcome,
                        "n_denominator": 100,
                        "share_pct": (
                            5
                            + 2 * realm_position
                            + outcome_position
                            + 0.2 * (year - 2000)
                        ),
                    }
                )
    colors = {
        "Terrestrial": "#E69F00",
        "Freshwater": "#009E73",
        "Marine": "#0072B2",
    }
    figure = plot_pollution_nameability_trends(
        pd.DataFrame(rows),
        realm_order=CORE_REALMS,
        realm_names=REALM_NAMES,
        realm_colors=colors,
        start_year=2000,
        end_year=2025,
    )
    assert len(figure.axes) == 3
    expected_titles = (
        "Pollution",
        "Plastics terminology",
        "Direct exploitation",
    )
    expected_ylabels = (
        "Pollution share of\ndriver evidence (%)",
        "Pollution-related articles\nmentioning plastics (%)",
        "Direct-exploitation share\nof driver evidence (%)",
    )
    for axis, title, ylabel in zip(
        figure.axes,
        expected_titles,
        expected_ylabels,
    ):
        assert axis.get_ylim()[0] == pytest.approx(0)
        assert len(axis.lines) == 2 * len(CORE_REALMS)
        assert axis.get_title() == title
        assert axis.get_ylabel() == ylabel
    plt.close(figure)

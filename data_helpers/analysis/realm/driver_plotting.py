"""Minimal publication figures for L1 direct drivers by ecosystem realm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_helpers.visualization import BIODIVERSITY, contrasting_text_color


LOW_SUPPORT_THRESHOLD = 50


def _validate_order(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence, not one string")
    order = tuple(values)
    if not order or len(order) != len(set(order)):
        raise ValueError(f"{name} must contain unique labels")
    if any(not isinstance(value, str) or not value for value in order):
        raise ValueError(f"{name} must contain non-empty strings")
    return order


def _validate_mapping(
    mapping: Mapping[str, object],
    labels: Sequence[str],
    *,
    name: str,
) -> None:
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise ValueError(f"{name} is missing labels: {missing}")


def _matrix(
    frame: pd.DataFrame,
    *,
    value: str,
    realms: Sequence[str],
    drivers: Sequence[str],
) -> pd.DataFrame:
    required = {"realm", "driver", value}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing column(s): {missing}")
    selected = frame[["realm", "driver", value]].copy()
    unexpected_realms = sorted(set(selected["realm"]).difference(realms))
    unexpected_drivers = sorted(set(selected["driver"]).difference(drivers))
    if unexpected_realms or unexpected_drivers:
        raise ValueError(
            "Input contains labels outside the configured orders: "
            f"realms={unexpected_realms}, drivers={unexpected_drivers}"
        )
    if selected.duplicated(["realm", "driver"]).any():
        raise ValueError("Input contains duplicate realm-driver cells")
    expected = pd.MultiIndex.from_product(
        [realms, drivers], names=["realm", "driver"]
    )
    observed = pd.MultiIndex.from_frame(selected[["realm", "driver"]])
    missing_cells = expected.difference(observed)
    if len(missing_cells):
        raise ValueError(
            f"Input is missing realm-driver cells: {list(missing_cells)}"
        )
    selected[value] = pd.to_numeric(selected[value], errors="coerce")
    return (
        selected.pivot(index="realm", columns="driver", values=value)
        .reindex(index=realms, columns=drivers)
        .astype(float)
    )


def _support_label(
    realm: str,
    *,
    realm_names: Mapping[str, str],
    realm_supports: Mapping[str, int],
) -> str:
    marker = "†" if realm_supports[realm] < LOW_SUPPORT_THRESHOLD else ""
    return f"{realm_names[realm]}{marker}\n(n={realm_supports[realm]:,})"


def _validate_common_inputs(
    *,
    driver_order: Sequence[str],
    realm_order: Sequence[str],
    driver_names: Mapping[str, str],
    realm_names: Mapping[str, str],
    realm_supports: Mapping[str, int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    drivers = _validate_order(driver_order, name="driver_order")
    realms = _validate_order(realm_order, name="realm_order")
    _validate_mapping(driver_names, drivers, name="driver_names")
    _validate_mapping(realm_names, realms, name="realm_names")
    _validate_mapping(realm_supports, realms, name="realm_supports")
    invalid_supports = [
        realm
        for realm in realms
        if not isinstance(realm_supports[realm], Integral)
        or isinstance(realm_supports[realm], bool)
        or realm_supports[realm] <= 0
    ]
    if invalid_supports:
        raise ValueError(
            f"realm_supports must contain positive integers: {invalid_supports}"
        )
    return drivers, realms


BAR_LEFT, BAR_RIGHT = 0.30, 0.985


def plot_driver_fractional_bars(
    estimates: pd.DataFrame,
    *,
    driver_order: Sequence[str],
    realm_order: Sequence[str],
    driver_names: Mapping[str, str],
    realm_names: Mapping[str, str],
    realm_supports: Mapping[str, int],
    driver_colors: Mapping[str, str],
    figure_height: float | None = None,
    show_segment_labels: bool = False,
    segment_label_min_pct: float = 6.0,
    segment_label_decimals: int = 0,
) -> plt.Figure:
    """Plot horizontal 100%-stacked driver composition bars."""
    drivers, realms = _validate_common_inputs(
        driver_order=driver_order,
        realm_order=realm_order,
        driver_names=driver_names,
        realm_names=realm_names,
        realm_supports=realm_supports,
    )
    _validate_mapping(driver_colors, drivers, name="driver_colors")
    invalid_colors = [
        driver
        for driver in drivers
        if not mcolors.is_color_like(driver_colors[driver])
    ]
    if invalid_colors:
        raise ValueError(f"driver_colors contains invalid colors: {invalid_colors}")
    if not isinstance(show_segment_labels, bool):
        raise TypeError("show_segment_labels must be boolean")
    if (
        not isinstance(segment_label_min_pct, Real)
        or isinstance(segment_label_min_pct, bool)
        or not np.isfinite(segment_label_min_pct)
        or not 0 <= segment_label_min_pct <= 100
    ):
        raise ValueError(
            "segment_label_min_pct must be between zero and 100"
        )
    if (
        not isinstance(segment_label_decimals, Integral)
        or isinstance(segment_label_decimals, bool)
        or not 0 <= segment_label_decimals <= 6
    ):
        raise ValueError("segment_label_decimals must be an integer from zero to six")
    fractional = _matrix(
        estimates,
        value="fractional_share",
        realms=realms,
        drivers=drivers,
    )
    if (
        not np.isfinite(fractional.to_numpy()).all()
        or (fractional.to_numpy() < 0).any()
        or (fractional.to_numpy() > 1).any()
    ):
        raise ValueError("fractional_share must contain finite proportions")
    if not np.allclose(
        fractional.sum(axis=1).to_numpy(), 1.0, rtol=0, atol=1e-6
    ):
        raise ValueError("fractional_share must sum to one within every realm")

    height = figure_height or max(2.4, 0.43 * len(realms) + 1.4)
    figure, axis = plt.subplots(figsize=(7.4, height))
    positions = np.arange(len(realms))
    left = np.zeros(len(realms))
    for driver in drivers:
        values = 100 * fractional[driver].to_numpy()
        axis.barh(
            positions,
            values,
            left=left,
            height=0.50,
            color=driver_colors[driver],
            edgecolor=BIODIVERSITY["paper"],
            linewidth=0.4,
            zorder=2,
        )
        if show_segment_labels:
            for position, start, width in zip(positions, left, values):
                if width >= segment_label_min_pct:
                    axis.text(
                        start + width / 2,
                        position,
                        f"{width:.{segment_label_decimals}f}%",
                        ha="center",
                        va="center",
                        fontsize=6.8,
                        color=contrasting_text_color(driver_colors[driver]),
                        zorder=3,
                    )
        left += values
    axis.set(
        xlim=(0, 100),
        ylim=(len(realms) - 0.5, -0.5),
        xticks=[],
        yticks=positions,
        yticklabels=[
            _support_label(
                realm,
                realm_names=realm_names,
                realm_supports=realm_supports,
            )
            for realm in realms
        ],
    )
    axis.tick_params(
        length=0,
        colors=BIODIVERSITY["ink"],
        labelsize=7.5,
    )
    axis.grid(False)
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)

    footer = (
        f"† Realm support < {LOW_SUPPORT_THRESHOLD} publications"
        if any(realm_supports[realm] < LOW_SUPPORT_THRESHOLD for realm in realms)
        else ""
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=driver_colors[driver],
                edgecolor=BIODIVERSITY["paper"],
                linewidth=0.4,
                label=driver_names[driver],
            )
            for driver in drivers
        ],
        loc="lower center",
        ncol=len(drivers),
        bbox_to_anchor=(0.5 * (BAR_LEFT + BAR_RIGHT), 0.035 if footer else 0.025),
        frameon=False,
        fontsize=6.8,
        handlelength=1.1,
        columnspacing=1.15,
        borderaxespad=0,
    )
    if footer:
        figure.text(
            0.30,
            0.005,
            footer,
            fontsize=6.8,
            color=BIODIVERSITY["ink"],
        )
    figure.subplots_adjust(
        left=BAR_LEFT,
        right=BAR_RIGHT,
        top=0.98,
        bottom=0.14 if not footer else 0.16,
    )
    return figure


def plot_pollution_nameability_trends(
    annual_summary: pd.DataFrame,
    *,
    realm_order: Sequence[str],
    realm_names: Mapping[str, str],
    realm_colors: Mapping[str, str],
    start_year: int,
    end_year: int,
) -> plt.Figure:
    """Plot pollution, plastics-within-pollution, and exploitation trends."""
    realms = _validate_order(realm_order, name="realm_order")
    _validate_mapping(realm_names, realms, name="realm_names")
    _validate_mapping(realm_colors, realms, name="realm_colors")
    invalid_colors = [
        realm
        for realm in realms
        if not mcolors.is_color_like(realm_colors[realm])
    ]
    if invalid_colors:
        raise ValueError(f"realm_colors contains invalid colors: {invalid_colors}")
    if start_year >= end_year:
        raise ValueError("start_year must precede end_year")
    required = {
        "realm",
        "publication_year",
        "outcome",
        "n_denominator",
        "share_pct",
    }
    missing = sorted(required.difference(annual_summary.columns))
    if missing:
        raise ValueError(f"annual_summary is missing column(s): {missing}")
    outcomes = (
        "Pollution fractional attention",
        "Plastics within pollution",
        "Direct exploitation fractional attention",
    )
    titles = (
        "Pollution",
        "Plastics terminology",
        "Direct exploitation",
    )
    y_labels = (
        "Pollution share of\ndriver evidence (%)",
        "Pollution-related articles\nmentioning plastics (%)",
        "Direct-exploitation share\nof driver evidence (%)",
    )
    selected = annual_summary.loc[
        annual_summary["realm"].isin(realms)
        & annual_summary["outcome"].isin(outcomes)
        & annual_summary["publication_year"].between(start_year, end_year)
    ].copy()
    observed = set(
        selected[["realm", "outcome"]].itertuples(index=False, name=None)
    )
    expected = {(realm, outcome) for realm in realms for outcome in outcomes}
    if observed != expected:
        raise ValueError(
            "annual_summary must contain every realm × outcome combination"
        )
    selected["publication_year"] = pd.to_numeric(
        selected["publication_year"], errors="coerce"
    )
    selected["share_pct"] = pd.to_numeric(
        selected["share_pct"], errors="coerce"
    )
    if selected[["publication_year", "share_pct"]].isna().any().any():
        raise ValueError("annual_summary contains non-numeric plotted values")

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 3.2))
    for panel, (axis, outcome, title, y_label) in enumerate(
        zip(axes, outcomes, titles, y_labels)
    ):
        panel_frame = selected.loc[selected["outcome"].eq(outcome)]
        for realm in realms:
            realm_frame = (
                panel_frame.loc[panel_frame["realm"].eq(realm)]
                .sort_values("publication_year")
                .reset_index(drop=True)
            )
            rolling = realm_frame["share_pct"].rolling(
                window=5,
                center=True,
                min_periods=3,
            ).mean()
            axis.plot(
                realm_frame["publication_year"],
                realm_frame["share_pct"],
                color=realm_colors[realm],
                linewidth=0.65,
                alpha=0.25,
                zorder=1,
            )
            axis.scatter(
                realm_frame["publication_year"],
                realm_frame["share_pct"],
                color=realm_colors[realm],
                s=5,
                alpha=0.28,
                linewidths=0,
                zorder=2,
            )
            axis.plot(
                realm_frame["publication_year"],
                rolling,
                color=realm_colors[realm],
                linewidth=1.65,
                zorder=3,
            )
        panel_max = float(panel_frame["share_pct"].max())
        upper = max(5.0, 5 * np.ceil(panel_max * 1.12 / 5))
        axis.set(
            xlim=(start_year, end_year),
            ylim=(0, upper),
            xticks=sorted({start_year, 2010, 2020, end_year}),
            title=title,
            ylabel=y_label,
        )
        axis.text(
            -0.16,
            1.08,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=BIODIVERSITY["ink"],
        )
        axis.tick_params(
            length=0,
            colors=BIODIVERSITY["ink"],
            labelsize=7,
        )
        axis.title.set_fontsize(8)
        axis.yaxis.label.set_size(7.5)
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        for spine in ("bottom", "left"):
            axis.spines[spine].set_color(BIODIVERSITY["neutral"])
            axis.spines[spine].set_linewidth(0.6)

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=realm_colors[realm],
                linewidth=1.8,
                label=realm_names[realm],
            )
            for realm in realms
        ],
        loc="upper center",
        ncol=len(realms),
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        columnspacing=1.5,
        borderaxespad=0,
    )
    figure.text(
        0.99,
        0.025,
        "Faint lines/points: annual estimates · solid lines: centered 5-year mean",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=BIODIVERSITY["ink"],
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.78,
        bottom=0.18,
        wspace=0.42,
    )
    return figure


__all__ = [
    "LOW_SUPPORT_THRESHOLD",
    "plot_driver_fractional_bars",
    "plot_pollution_nameability_trends",
]

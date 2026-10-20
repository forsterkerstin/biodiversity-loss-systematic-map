"""Radial country-level view of publication-fractional research attention."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Real
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, Wedge

from data_helpers import visualization


REGION_ORDER = (
    "Africa",
    "Asia and the Pacific",
    "Americas",
    "Europe and Central Asia",
)
REGION_COLORS = {
    "Africa": "#D9954C",
    "Asia and the Pacific": "#6F91C7",
    "Americas": "#2E7D32",
    "Europe and Central Asia": "#8C6BB1",
}

COUNTRY_DISPLAY_OVERRIDES = {
    "Akrotiri and Dhekelia": "Akrotiri–Dhekelia",
    "Bonaire, Sint Eustatius and Saba": "Bonaire–Eust.–Saba",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "British Indian Ocean Territory (the)": "BIOT",
    "Central African Republic (the)": "Central African Rep.",
    "Congo (the Democratic Republic of the)": "DR Congo",
    "Congo (the)": "Congo Republic",
    "Falkland Islands (the) [Malvinas]": "Falklands/Malvinas",
    "French Southern Territories (the)": "French S. Terr.",
    "Heard Island and McDonald Islands": "Heard–McDonald Is.",
    "Korea (the Republic of)": "South Korea",
    "Korea (the Democratic People's Republic of)": "North Korea",
    "Lao People's Democratic Republic (the)": "Lao PDR",
    "Northern Mariana Islands (the)": "N Mariana Islands",
    "Saint Helena, Ascension and Tristan da Cunha": "St Helena group",
    "Saint Kitts and Nevis": "St Kitts & Nevis",
    "Saint Pierre and Miquelon": "St Pierre & Miquelon",
    "Saint Vincent and the Grenadines": "St Vincent–Gren.",
    "Sao Tome and Principe": "São Tomé & Príncipe",
    "South Georgia and the South Sandwich Islands": "S Georgia–Sandwich",
    "Svalbard and Jan Mayen": "Svalbard–Jan Mayen",
    "Tanzania, the United Republic of": "Tanzania",
    "Turks and Caicos Islands (the)": "Turks & Caicos Is.",
    "United Kingdom of Great Britain and Northern Ireland (the)": "United Kingdom",
    "United States of America (the)": "United States",
    "United States Minor Outlying Islands (Navassa)": "Navassa Island",
    "United States Minor Outlying Islands (the)": "U.S. Outlying Is.",
    "Virgin Islands (British)": "British Virgin Is.",
    "Virgin Islands (U.S.)": "U.S. Virgin Islands",
}


class GeographicAttentionPlotError(ValueError):
    """Raised when observed country attention cannot be plotted safely."""


def _blend_with_paper(color: str, amount: float) -> str:
    foreground = np.asarray(mcolors.to_rgb(color), dtype=float)
    paper = np.asarray(mcolors.to_rgb(visualization.BIODIVERSITY["paper"]), dtype=float)
    return mcolors.to_hex((1.0 - amount) * foreground + amount * paper)


def _polar_xy(radius: float, angle_degrees: float) -> tuple[float, float]:
    angle = np.deg2rad(angle_degrees)
    return radius * float(np.cos(angle)), radius * float(np.sin(angle))


def _display_country_name(country: str) -> str:
    if country in COUNTRY_DISPLAY_OVERRIDES:
        return COUNTRY_DISPLAY_OVERRIDES[country]
    display = re.sub(r"\s*\([^)]*\)", "", country)
    display = re.sub(r"\s*\([^)]*$", "", display)
    display = re.sub(r"\s*\*+\s*$", "", display)
    return re.sub(r"\s+", " ", display).strip()


def _upright_tangential_rotation(angle_degrees: float) -> float:
    rotation = ((angle_degrees - 90 + 180) % 360) - 180
    if rotation > 90:
        rotation -= 180
    elif rotation < -90:
        rotation += 180
    return rotation


def _run_span(
    start: int,
    size: int,
    *,
    slot_width: float,
    start_angle: float,
) -> tuple[float, float, float]:
    high = start_angle - start * slot_width
    low = start_angle - (start + size) * slot_width
    return low, high, (low + high) / 2.0


def _inset_angles(
    low: float,
    high: float,
    requested_gap: float,
) -> tuple[float, float]:
    span = high - low
    gap = min(requested_gap, 0.42 * span)
    return low + gap / 2.0, high - gap / 2.0


def _attention_scale(values_pct: np.ndarray, *, floor_pct: float) -> np.ndarray:
    values = np.asarray(values_pct, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("values_pct must be finite, non-negative, and one-dimensional")
    if not isinstance(floor_pct, Real) or isinstance(floor_pct, bool):
        raise TypeError("floor_pct must be numeric")
    if not np.isfinite(floor_pct) or floor_pct <= 0:
        raise ValueError("floor_pct must be finite and positive")
    maximum = float(values.max(initial=0.0))
    if maximum == 0:
        return np.zeros_like(values)
    return np.log1p(values / floor_pct) / np.log1p(maximum / floor_pct)


def _automatic_scale_ticks(maximum_pct: float, *, limit: int = 4) -> list[float]:
    if maximum_pct <= 0:
        return []
    upper_exponent = int(np.floor(np.log10(maximum_pct)))
    ticks = [
        float(10.0**exponent)
        for exponent in range(upper_exponent - 5, upper_exponent + 1)
        if 10.0**exponent <= maximum_pct * (1 + 1e-12)
    ]
    return ticks[-limit:]


def _prepare_leaves(
    leaves: pd.DataFrame,
    *,
    region_col: str,
    country_col: str,
    count_col: str,
    share_col: str,
    region_order: Sequence[str],
) -> pd.DataFrame:
    required = {region_col, country_col, count_col, share_col}
    missing = sorted(required.difference(leaves.columns))
    if missing:
        raise GeographicAttentionPlotError(
            f"Country table is missing required column(s): {missing}"
        )
    if leaves.empty:
        raise GeographicAttentionPlotError("Country table cannot be empty")

    frame = leaves.copy().reset_index(drop=True)
    for column in (region_col, country_col):
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise GeographicAttentionPlotError(
                f"{column} must contain non-empty labels"
            )
    if frame.duplicated([region_col, country_col]).any():
        raise GeographicAttentionPlotError(
            "Country table contains duplicate region-country rows"
        )

    counts = pd.to_numeric(frame[count_col], errors="coerce")
    if (
        counts.isna().any()
        or (~np.isfinite(counts)).any()
        or counts.le(0).any()
        or not np.allclose(counts, np.round(counts), rtol=0, atol=1e-9)
    ):
        raise GeographicAttentionPlotError(
            f"{count_col} must contain positive integer counts"
        )
    frame["_count"] = counts.round().astype(np.int64)

    shares = pd.to_numeric(frame[share_col], errors="coerce")
    if (
        shares.isna().any()
        or (~np.isfinite(shares)).any()
        or shares.le(0).any()
        or shares.gt(100).any()
    ):
        raise GeographicAttentionPlotError(
            f"{share_col} must contain percentages in (0, 100]"
        )
    frame["_share_pct"] = shares.astype(float)

    configured = list(dict.fromkeys(region_order))
    if not configured:
        raise GeographicAttentionPlotError("region_order cannot be empty")
    observed = frame[region_col].drop_duplicates().tolist()
    fallback_order = [
        *configured,
        *[name for name in observed if name not in configured],
    ]
    fallback_rank = {name: index for index, name in enumerate(fallback_order)}
    region_attention = frame.groupby(region_col, observed=True)["_share_pct"].sum()
    ordered_regions = sorted(
        observed,
        key=lambda name: (
            -float(region_attention[name]),
            fallback_rank[name],
            str(name),
        ),
    )
    region_rank = {name: index for index, name in enumerate(ordered_regions)}
    frame["_region_rank"] = frame[region_col].map(region_rank)
    return frame.sort_values(
        ["_region_rank", "_share_pct", "_count", country_col],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _region_runs(
    frame: pd.DataFrame,
    region_col: str,
) -> list[tuple[str, int, int]]:
    runs: list[tuple[str, int, int]] = []
    for region, positions in frame.groupby(
        region_col, sort=False, observed=True
    ).indices.items():
        ordered = np.sort(np.asarray(positions, dtype=int))
        if ordered[-1] - ordered[0] + 1 != len(ordered):
            raise GeographicAttentionPlotError(
                f"Region is not contiguous after ordering: {region}"
            )
        runs.append((str(region), int(ordered[0]), len(ordered)))
    return sorted(runs, key=lambda item: item[1])


def _add_region_label(
    axis: plt.Axes,
    *,
    region: str,
    angle: float,
    span: float,
    radius: float,
    outer_radius: float,
    color: str,
) -> None:
    text = region.replace(" and ", " & ")
    if span >= 8.0:
        x, y = _polar_xy(radius, angle)
        axis.text(
            x,
            y,
            fill(text, width=18, break_long_words=False, break_on_hyphens=False),
            ha="center",
            va="center",
            rotation=_upright_tangential_rotation(angle),
            rotation_mode="anchor",
            fontsize=7.0,
            fontweight="bold",
            color=visualization.contrasting_text_color(color),
            linespacing=0.95,
            zorder=8,
            gid=f"region-label:{region}",
        )
        return

    anchor = _polar_xy(outer_radius, angle)
    target = _polar_xy(outer_radius + 0.34, angle)
    annotation = axis.annotate(
        text,
        xy=anchor,
        xytext=target,
        ha="left" if np.cos(np.deg2rad(angle)) >= 0 else "right",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=visualization.BIODIVERSITY["ink"],
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.5,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
        annotation_clip=False,
        zorder=10,
    )
    annotation.set_gid(f"region-label:{region}")


def _add_country_labels(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    country_col: str,
    angles: np.ndarray,
    radius: float,
    low_attention_threshold_pct: float,
    highlight_attention_threshold_pct: float,
) -> None:
    for position, row in frame.iterrows():
        angle = float(angles[position])
        x, y = _polar_xy(radius, angle)
        left_half = 90 < angle % 360 < 270
        artist = axis.text(
            x,
            y,
            _display_country_name(str(row[country_col])),
            ha="right" if left_half else "left",
            va="center",
            rotation=angle + 180 if left_half else angle,
            rotation_mode="anchor",
            fontsize=4.8,
            fontweight=(
                "bold"
                if float(row["_share_pct"]) > highlight_attention_threshold_pct
                else "normal"
            ),
            fontstyle=(
                "italic"
                if float(row["_share_pct"]) < low_attention_threshold_pct
                else "normal"
            ),
            color=visualization.BIODIVERSITY["ink"],
            clip_on=False,
            zorder=9,
        )
        artist.set_gid(f"country-label:{position}:{row[country_col]}")


def plot_geographic_attention_radial(
    leaves: pd.DataFrame,
    *,
    region_col: str = "region",
    country_col: str = "country",
    count_col: str = "record_count",
    share_col: str = "attention_share_pct",
    region_order: Sequence[str] = REGION_ORDER,
    region_colors: Mapping[str, str] | None = None,
    low_attention_threshold_pct: float = 0.1,
    highlight_attention_threshold_pct: float = 1.0,
    start_angle: float = 90.0,
    attention_scale_floor_pct: float = 0.001,
    attention_ticks_pct: Sequence[float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot observed countries in equal-angle slots ordered by attention.

    Regions and countries run clockwise from the top seam in descending
    publication-fractional attention. Radial bar length uses a labelled log scale.
    Positive shares below ``low_attention_threshold_pct`` receive pale region fills;
    labels above ``highlight_attention_threshold_pct`` are bold.
    """
    if (
        not isinstance(low_attention_threshold_pct, Real)
        or isinstance(low_attention_threshold_pct, bool)
        or not np.isfinite(low_attention_threshold_pct)
        or not 0 < low_attention_threshold_pct <= 100
    ):
        raise GeographicAttentionPlotError(
            "low_attention_threshold_pct must be a finite percentage in (0, 100]"
        )
    if (
        not isinstance(highlight_attention_threshold_pct, Real)
        or isinstance(highlight_attention_threshold_pct, bool)
        or not np.isfinite(highlight_attention_threshold_pct)
        or not 0 <= highlight_attention_threshold_pct <= 100
    ):
        raise GeographicAttentionPlotError(
            "highlight_attention_threshold_pct must be a finite percentage in [0, 100]"
        )

    colors = dict(REGION_COLORS if region_colors is None else region_colors)
    frame = _prepare_leaves(
        leaves,
        region_col=region_col,
        country_col=country_col,
        count_col=count_col,
        share_col=share_col,
        region_order=region_order,
    )
    missing_colors = sorted(set(frame[region_col]).difference(colors))
    if missing_colors:
        raise GeographicAttentionPlotError(
            f"region_colors is missing region(s): {missing_colors}"
        )
    invalid_colors = sorted(
        region
        for region in set(frame[region_col])
        if not mcolors.is_color_like(colors[region])
    )
    if invalid_colors:
        raise GeographicAttentionPlotError(
            f"region_colors contains invalid colours for: {invalid_colors}"
        )

    figure, axis = plt.subplots(figsize=figsize or (7.5, 7.5))
    axis.set_aspect("equal")
    axis.set_axis_off()

    center_radius = 0.52
    region_inner, region_width = 0.56, 0.78
    region_outer = region_inner + region_width
    country_inner, country_width = 1.40, 0.30
    country_outer = country_inner + country_width
    bar_base = country_outer + 0.08
    maximum_bar_height = 2.23
    maximum_outer_radius = bar_base + maximum_bar_height

    count = len(frame)
    slot_width = 360.0 / count
    angles = start_angle - (np.arange(count, dtype=float) + 0.5) * slot_width
    scaled_attention = _attention_scale(
        frame["_share_pct"].to_numpy(dtype=float),
        floor_pct=attention_scale_floor_pct,
    )
    bar_heights = np.maximum(maximum_bar_height * scaled_attention, 0.022)

    maximum_share = float(frame["_share_pct"].max())
    ticks = (
        _automatic_scale_ticks(maximum_share)
        if attention_ticks_pct is None
        else [float(value) for value in attention_ticks_pct]
    )
    if any(not np.isfinite(value) or value <= 0 for value in ticks):
        raise GeographicAttentionPlotError(
            "attention_ticks_pct must contain finite positive values"
        )
    ticks = sorted(set(value for value in ticks if value <= maximum_share))
    if ticks:
        tick_scaled = _attention_scale(
            np.asarray([*ticks, maximum_share], dtype=float),
            floor_pct=attention_scale_floor_pct,
        )[:-1]
        for tick, scaled in zip(ticks, tick_scaled, strict=True):
            radius = bar_base + maximum_bar_height * float(scaled)
            guide = Circle(
                (0, 0),
                radius,
                facecolor="none",
                edgecolor="#E5E5E5",
                linewidth=0.35,
                zorder=0.2,
            )
            guide.set_gid(f"attention-guide:{tick:g}")
            axis.add_patch(guide)
            x, y = _polar_xy(radius, start_angle)
            label = axis.text(
                x,
                y,
                f"{tick:g}%",
                ha="left",
                va="center",
                fontsize=6.6,
                fontweight="bold",
                color="#777777",
                bbox={
                    "boxstyle": "square,pad=0.08",
                    "facecolor": visualization.BIODIVERSITY["paper"],
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=11,
            )
            label.set_gid(f"attention-guide-label:{tick:g}")

    for region, run_start, run_size in _region_runs(frame, region_col):
        low, high, middle = _run_span(
            run_start,
            run_size,
            slot_width=slot_width,
            start_angle=start_angle,
        )
        theta1, theta2 = _inset_angles(low, high, requested_gap=1.05)
        wedge = Wedge(
            (0, 0),
            region_outer,
            theta1,
            theta2,
            width=region_width,
            facecolor=colors[region],
            edgecolor=visualization.BIODIVERSITY["paper"],
            linewidth=0.8,
            zorder=3,
        )
        wedge.set_gid(f"region:{region}")
        axis.add_patch(wedge)
        _add_region_label(
            axis,
            region=region,
            angle=middle,
            span=high - low,
            radius=region_inner + region_width / 2,
            outer_radius=region_outer,
            color=colors[region],
        )

    for position, row in frame.iterrows():
        low, high, _ = _run_span(
            position,
            1,
            slot_width=slot_width,
            start_angle=start_angle,
        )
        region_color = colors[str(row[region_col])]
        facecolor = (
            _blend_with_paper(region_color, 0.64)
            if float(row["_share_pct"]) < low_attention_threshold_pct
            else region_color
        )
        theta1, theta2 = _inset_angles(
            low, high, requested_gap=min(0.12, 0.14 * slot_width)
        )
        slot = Wedge(
            (0, 0),
            country_outer,
            theta1,
            theta2,
            width=country_width,
            facecolor=facecolor,
            edgecolor=visualization.BIODIVERSITY["paper"],
            linewidth=0.16,
            zorder=5,
        )
        slot.set_gid(f"country-slot:{position}:{row[country_col]}")
        axis.add_patch(slot)

        height = float(bar_heights[position])
        bar_theta1, bar_theta2 = _inset_angles(
            low, high, requested_gap=min(0.25, 0.24 * slot_width)
        )
        bar = Wedge(
            (0, 0),
            bar_base + height,
            bar_theta1,
            bar_theta2,
            width=height,
            facecolor=facecolor,
            edgecolor="none",
            linewidth=0,
            zorder=6,
        )
        bar.set_gid(f"attention-bar:{position}:{row[country_col]}")
        axis.add_patch(bar)

    centre = Circle(
        (0, 0),
        center_radius,
        facecolor=visualization.BIODIVERSITY["ink"],
        edgecolor=visualization.BIODIVERSITY["ink"],
        linewidth=0.4,
        zorder=20,
    )
    centre.set_gid("evidence-centre")
    axis.add_patch(centre)
    _add_country_labels(
        axis,
        frame,
        country_col=country_col,
        angles=angles,
        radius=maximum_outer_radius + 0.12,
        low_attention_threshold_pct=float(low_attention_threshold_pct),
        highlight_attention_threshold_pct=float(highlight_attention_threshold_pct),
    )

    limit = maximum_outer_radius + 0.72
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    figure.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    return figure


__all__ = [
    "GeographicAttentionPlotError",
    "REGION_COLORS",
    "REGION_ORDER",
    "plot_geographic_attention_radial",
]

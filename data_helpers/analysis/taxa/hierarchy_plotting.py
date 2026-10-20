"""Research-attention sunburst for the rank-aligned taxonomic hierarchy."""

from __future__ import annotations

import colorsys
from typing import Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

from data_helpers.analysis.taxa import clipart as taxa_clipart


SUNBURST_KINGDOM_COLORS = {
    "Animalia": "#3F6FB5",
    "Plantae": "#4E9258",
    "Fungi": "#D1843D",
    "Other kingdoms": "#AFAFAF",
}

DEFAULT_SUNBURST_RANK_LIGHTNESS = {
    "phylum": 0.58,
    "class": 0.70,
    "order": 0.83,
}
DEFAULT_SUNBURST_RANK_SATURATION = {
    "phylum": 0.64,
    "class": 0.52,
    "order": 0.40,
}

SUNBURST_RANKS = ("kingdom", "phylum", "class", "order")
DEFAULT_SUNBURST_RING_WIDTHS = {
    "kingdom": 0.30,
    "phylum": 0.34,
    "class": 0.46,
    "order": 0.72,
}
DEFAULT_SUNBURST_INNER_RADIUS = 0.44

# Nature's double-column artwork width is 180 mm.  Author the PDF at its
# intended print dimensions so LaTeX does not halve otherwise compliant text.
NATURE_DOUBLE_COLUMN_WIDTH_IN = 180.0 / 25.4
NATURE_SUNBURST_HEIGHT_IN = 170.0 / 25.4
NATURE_SUNBURST_FIGSIZE = (
    NATURE_DOUBLE_COLUMN_WIDTH_IN,
    NATURE_SUNBURST_HEIGHT_IN,
)
SUNBURST_LEGEND_RANK_INDENT = {
    1: 0.0,
    2: 0.055,
    3: 0.110,
    4: 0.165,
}

def sunburst_node_colors(
    nodes: pd.DataFrame,
    *,
    kingdom_colors: Mapping[str, str] | None = None,
    rank_lightness: Mapping[str, float] | None = None,
    rank_saturation: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Assign stable related hues down an unbalanced taxonomic hierarchy."""
    required = {"node_id", "parent_id", "depth", "sibling_order", "is_other"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"Sunburst nodes lack color columns: {sorted(missing)}")
    rank_lightness = dict(rank_lightness or DEFAULT_SUNBURST_RANK_LIGHTNESS)
    rank_saturation = dict(rank_saturation or DEFAULT_SUNBURST_RANK_SATURATION)
    expected_ranks = {"phylum", "class", "order"}
    for label, values in (
        ("lightness", rank_lightness),
        ("saturation", rank_saturation),
    ):
        if set(values) != expected_ranks or any(
            not 0 <= float(value) <= 1 for value in values.values()
        ):
            raise ValueError(
                f"Sunburst rank {label} must cover Phylum, Class, and Order "
                "with values in [0, 1]."
            )
    lightness_by_depth = {
        depth: rank_lightness[rank]
        for depth, rank in enumerate(SUNBURST_RANKS[1:], start=2)
    }
    saturation_by_depth = {
        depth: rank_saturation[rank]
        for depth, rank in enumerate(SUNBURST_RANKS[1:], start=2)
    }
    colors = {**SUNBURST_KINGDOM_COLORS, **(kingdom_colors or {})}
    assigned: dict[str, str] = {}
    roots = nodes.loc[nodes["parent_id"].isna()].sort_values("sibling_order")
    for row in roots.itertuples(index=False):
        assigned[str(row.node_id)] = colors.get(str(row.taxon), "#AFAFAF")

    for depth in sorted(nodes["depth"].unique()):
        if int(depth) <= 1:
            continue
        level = nodes.loc[nodes["depth"].eq(depth)]
        for parent_id, children in level.groupby("parent_id", sort=False):
            parent_color = assigned.get(str(parent_id), "#AFAFAF")
            red, green, blue = mcolors.to_rgb(parent_color)
            hue, _, _ = colorsys.rgb_to_hls(red, green, blue)
            children = children.sort_values("sibling_order")
            named = children.loc[~children["is_other"]]
            named_index = {
                str(node_id): index
                for index, node_id in enumerate(named["node_id"].astype(str))
            }
            named_count = max(len(named), 1)
            for row in children.itertuples(index=False):
                node_id = str(row.node_id)
                if bool(row.is_other):
                    # Residual groups are real portions of their parent, so keep
                    # them in the same rank/Kingdom palette. Placing their hue
                    # after the named siblings avoids shifting established
                    # colours for the named taxa.
                    index = len(named)
                else:
                    index = named_index[node_id]
                centered = (index - (named_count - 1) / 2) / max(named_count, 2)
                hue_span = {2: 0.10, 3: 0.055, 4: 0.032}.get(int(depth), 0.025)
                target_lightness = float(lightness_by_depth[int(depth)])
                target_saturation = float(saturation_by_depth[int(depth)])
                child_rgb = colorsys.hls_to_rgb(
                    (hue + centered * hue_span) % 1.0,
                    target_lightness,
                    target_saturation,
                )
                assigned[node_id] = mcolors.to_hex(child_rgb)
    return assigned


def _sunburst_layout(nodes: pd.DataFrame) -> pd.DataFrame:
    """Return exact research-count angular positions for every plotted node."""
    required = {
        "node_id",
        "parent_id",
        "research_count",
        "sibling_order",
        "depth",
    }
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"Sunburst nodes lack layout columns: {sorted(missing)}")
    if nodes["node_id"].duplicated().any():
        raise ValueError("Sunburst node ids must be unique.")
    roots = nodes.loc[nodes["parent_id"].isna()].sort_values("sibling_order")
    total = float(roots["research_count"].sum())
    if total <= 0:
        raise ValueError("Sunburst root research total must be positive.")
    positions: dict[str, tuple[float, float]] = {}
    start = 0.0
    for row in roots.itertuples(index=False):
        width = float(row.research_count) / total * 2 * np.pi
        positions[str(row.node_id)] = (start, width)
        start += width
    if not np.isclose(start, 2 * np.pi):
        raise ValueError("Sunburst Kingdom widths do not fill one circle.")

    for depth in sorted(nodes["depth"].unique()):
        if int(depth) <= 1:
            continue
        level = nodes.loc[nodes["depth"].eq(depth)]
        for parent_id, children in level.groupby("parent_id", sort=False):
            parent_id = str(parent_id)
            if parent_id not in positions:
                raise ValueError(f"Sunburst parent is missing: {parent_id!r}.")
            parent_start, parent_width = positions[parent_id]
            parent_count = float(
                nodes.loc[nodes["node_id"].eq(parent_id), "research_count"].iloc[0]
            )
            children = children.sort_values("sibling_order")
            child_start = parent_start
            for row in children.itertuples(index=False):
                width = parent_width * float(row.research_count) / parent_count
                positions[str(row.node_id)] = (child_start, width)
                child_start += width
            if not np.isclose(child_start, parent_start + parent_width):
                raise ValueError(
                    f"Sunburst child widths do not reconcile for {parent_id!r}."
                )

    positioned = nodes.copy()
    positioned["theta_start"] = positioned["node_id"].map(
        lambda node_id: positions[str(node_id)][0]
    )
    positioned["theta_width"] = positioned["node_id"].map(
        lambda node_id: positions[str(node_id)][1]
    )
    positioned["theta_mid"] = (
        positioned["theta_start"] + positioned["theta_width"] / 2
    )
    return positioned


def _format_sunburst_percent(value: float) -> str:
    if value < 0.005:
        return "<0.01%"
    if value < 1:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def _sunburst_contrast_color(color: str) -> str:
    """Choose a restrained light/dark foreground for a filled wedge."""
    rgb = np.asarray(mcolors.to_rgb(color))
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    return "#FAFAFA" if luminance < 0.34 else "#202020"


def _sunburst_label_candidates(
    nodes: pd.DataFrame,
    *,
    clipart_by_taxon: Mapping[str, str],
    min_angle_deg: float,
) -> pd.DataFrame:
    """Prefilter named wedges that have an asset and enough angular span."""
    if not 0 <= min_angle_deg <= 360:
        raise ValueError("Sunburst label angle threshold must be in [0, 360].")
    required = {"taxon", "theta_width", "is_other"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"Sunburst label nodes lack columns: {sorted(missing)}")
    angle_deg = np.degrees(nodes["theta_width"].astype(float))
    return nodes.loc[
        ~nodes["is_other"].astype(bool)
        & nodes["taxon"].astype(str).isin(clipart_by_taxon)
        & angle_deg.ge(min_angle_deg)
    ].copy()


def _sunburst_box_fits_wedge(
    axis: plt.Axes,
    *,
    theta: float,
    radius: float,
    theta_start: float,
    theta_width: float,
    ring_bottom: float,
    ring_top: float,
    width_px: float,
    height_px: float,
    rotation_deg: float,
    padding_px: float = 2.5,
) -> bool:
    """Test one rendered icon/glyph box against the annular wedge."""
    rotation = np.radians(rotation_deg)
    tangent = np.array([np.cos(rotation), np.sin(rotation)])
    normal = np.array([-tangent[1], tangent[0]])
    center = np.asarray(axis.transData.transform((theta, radius)))
    half_width = width_px / 2 + padding_px
    half_height = height_px / 2 + padding_px
    along = np.linspace(-half_width, half_width, 7)
    across = np.linspace(-half_height, half_height, 3)
    samples = np.asarray(
        [
            center + tangent * horizontal + normal * vertical
            for horizontal in along
            for vertical in across
        ]
    )
    polar = axis.transData.inverted().transform(samples)
    angular_offset = (polar[:, 0] - theta_start) % (2 * np.pi)
    angular_fit = np.all(angular_offset <= theta_width + 1e-9)
    radial_fit = np.all(
        (polar[:, 1] >= ring_bottom) & (polar[:, 1] <= ring_top)
    )
    return bool(angular_fit and radial_fit)


def _sunburst_curved_glyph_rotation(theta: float, reading_sign: float) -> float:
    """Return the continuous upright tangent rotation for one curved glyph."""
    rotation = -float(np.degrees(theta)) + (180.0 if reading_sign < 0 else 0.0)
    return (rotation + 180.0) % 360.0 - 180.0


def _sunburst_radial_text_rotation(theta: float) -> float:
    """Align a word with its radius while keeping it upright."""
    rotation = (90.0 - float(np.degrees(theta)) + 180.0) % 360.0 - 180.0
    if rotation > 90.0:
        rotation -= 180.0
    elif rotation < -90.0:
        rotation += 180.0
    return rotation


def _sunburst_ring_bounds(
    *,
    inner_radius: float,
    ring_widths: Mapping[str, float],
    ring_gap: float,
) -> dict[int, tuple[float, float]]:
    """Return depth-indexed radial bounds for unequal rank rings."""
    if set(ring_widths) != set(SUNBURST_RANKS):
        raise ValueError(
            "Sunburst ring widths must cover Kingdom, Phylum, Class, and Order."
        )
    widths = [float(ring_widths[rank]) for rank in SUNBURST_RANKS]
    if inner_radius <= 0 or ring_gap < 0 or any(width <= 0 for width in widths):
        raise ValueError("Sunburst radii and ring widths must be positive.")
    bounds: dict[int, tuple[float, float]] = {}
    bottom = float(inner_radius)
    for depth, width in enumerate(widths, start=1):
        bounds[depth] = (bottom, bottom + width)
        bottom += width + ring_gap
    return bounds


def _sunburst_terminal_continuation_color(
    color: str,
    *,
    target_depth: int,
    rank_lightness: Mapping[str, float],
    rank_saturation: Mapping[str, float],
) -> str:
    """Tint one terminal branch for an otherwise unused outer rank ring."""
    if target_depth not in range(2, len(SUNBURST_RANKS) + 1):
        raise ValueError("Terminal continuation depth must be Phylum through Order.")
    target_rank = SUNBURST_RANKS[target_depth - 1]
    red, green, blue = mcolors.to_rgb(color)
    hue, _, source_saturation = colorsys.rgb_to_hls(red, green, blue)
    target_lightness = float(rank_lightness[target_rank])
    # Preserve the deliberately neutral top-level Other-kingdoms branch, but
    # let coloured Remaining descendants continue in their parent's palette.
    target_saturation = (
        0.0
        if source_saturation < 1e-6
        else float(rank_saturation[target_rank])
    )
    return mcolors.to_hex(
        colorsys.hls_to_rgb(hue, target_lightness, target_saturation)
    )


def _draw_sunburst_wedge_labels(
    axis: plt.Axes,
    nodes: pd.DataFrame,
    *,
    mapping: Mapping,
    min_angle_deg: float,
    ring_bounds: Mapping[int, tuple[float, float]],
) -> tuple[str, ...]:
    """Draw readable icon/name pairs and return the taxa that passed final fit."""
    clipart_by_taxon = mapping.get("sunburst_clipart", {})
    if not clipart_by_taxon:
        return ()
    icon_paths = taxa_clipart.ensure_clipart_cached(mapping)
    candidates = _sunburst_label_candidates(
        nodes,
        clipart_by_taxon=clipart_by_taxon,
        min_angle_deg=min_angle_deg,
    )
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    labelled: list[str] = []

    for row in candidates.sort_values(["depth", "theta_start"]).itertuples(
        index=False
    ):
        asset_name = clipart_by_taxon.get(str(row.taxon))
        icon_path = icon_paths.get(str(asset_name))
        if icon_path is None:
            continue
        depth = int(row.depth)
        if depth == 1:
            font_size, icon_size_pt, gap_pt, tracking_pt = 6.4, 9.2, 4.0, 0.6
        elif depth == 2:
            font_size, icon_size_pt, gap_pt, tracking_pt = 6.0, 8.5, 3.6, 0.45
        else:
            font_size, icon_size_pt, gap_pt, tracking_pt = 5.6, 7.8, 3.2, 0.3
        font = FontProperties(size=font_size, weight="semibold")
        characters = list(str(row.taxon))
        glyph_sizes = [
            renderer.get_text_width_height_descent(character, font, ismath=False)[:2]
            for character in characters
        ]
        icon_size_px = renderer.points_to_pixels(icon_size_pt)
        gap_px = renderer.points_to_pixels(gap_pt)
        tracking_px = renderer.points_to_pixels(tracking_pt)
        text_arc_px = sum(width for width, _ in glyph_sizes) + tracking_px * max(
            len(characters) - 1, 0
        )
        pair_arc_px = icon_size_px + gap_px + text_arc_px
        ring_bottom, ring_top = ring_bounds[int(row.depth)]
        radius = (ring_bottom + ring_top) / 2
        theta_mid = float(row.theta_mid)
        epsilon = 1e-4
        before = np.asarray(axis.transData.transform((theta_mid - epsilon, radius)))
        after = np.asarray(axis.transData.transform((theta_mid + epsilon, radius)))
        pixels_per_radian = float(np.linalg.norm(after - before) / (2 * epsilon))
        theta_center = theta_mid
        theta_mod = theta_center % (2 * np.pi)
        reading_sign = (
            1.0
            if theta_mod <= np.pi / 2 or theta_mod >= 3 * np.pi / 2
            else -1.0
        )

        cursor_px = -pair_arc_px / 2
        icon_theta = theta_center + reading_sign * (
            cursor_px + icon_size_px / 2
        ) / pixels_per_radian
        cursor_px += icon_size_px + gap_px
        glyph_layout: list[tuple[str, float, float, float, float]] = []
        for index, (character, (width, height)) in enumerate(
            zip(characters, glyph_sizes)
        ):
            glyph_theta = theta_center + reading_sign * (
                cursor_px + width / 2
            ) / pixels_per_radian
            glyph_rotation = _sunburst_curved_glyph_rotation(
                glyph_theta, reading_sign
            )
            glyph_layout.append(
                (character, glyph_theta, glyph_rotation, width, height)
            )
            cursor_px += width
            if index < len(characters) - 1:
                cursor_px += tracking_px

        boxes_fit = _sunburst_box_fits_wedge(
            axis,
            theta=icon_theta,
            radius=radius,
            theta_start=float(row.theta_start),
            theta_width=float(row.theta_width),
            ring_bottom=ring_bottom,
            ring_top=ring_top,
            width_px=icon_size_px,
            height_px=icon_size_px,
            rotation_deg=0.0,
        ) and all(
            _sunburst_box_fits_wedge(
                axis,
                theta=glyph_theta,
                radius=radius,
                theta_start=float(row.theta_start),
                theta_width=float(row.theta_width),
                ring_bottom=ring_bottom,
                ring_top=ring_top,
                width_px=width,
                height_px=height,
                rotation_deg=glyph_rotation,
            )
            for _, glyph_theta, glyph_rotation, width, height in glyph_layout
        )
        if not boxes_fit:
            continue

        foreground = _sunburst_contrast_color(str(row.plot_color))
        rgba = taxa_clipart._tinted_icon(icon_path, foreground)
        icon = AnnotationBbox(
            OffsetImage(rgba, zoom=icon_size_px / max(rgba.shape[:2])),
            (icon_theta, radius),
            xycoords="data",
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0.0,
            annotation_clip=True,
            zorder=20,
        )
        icon.set_gid(f"sunburst-icon-{row.taxon}")
        axis.add_artist(icon)
        for index, (
            character,
            glyph_theta,
            glyph_rotation,
            _,
            _,
        ) in enumerate(glyph_layout):
            if character.isspace():
                continue
            label = axis.text(
                glyph_theta,
                radius,
                character,
                ha="center",
                va="center",
                rotation=glyph_rotation,
                rotation_mode="anchor",
                color=foreground,
                fontsize=font_size,
                fontweight="semibold",
                clip_on=True,
                zorder=21,
            )
            label.set_gid(f"sunburst-label-{row.taxon}-{index}")
        labelled.append(str(row.taxon))
    return tuple(labelled)


def _draw_sunburst_order_labels(
    axis: plt.Axes,
    nodes: pd.DataFrame,
    *,
    min_angle_deg: float,
    ring_bounds: Mapping[int, tuple[float, float]],
    excluded_taxa: set[str] | None = None,
    mapping: Mapping | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Draw fitted radial Order names and optional representative icons."""
    if not 0 <= min_angle_deg <= 360:
        raise ValueError("Sunburst Order-label angle threshold must be in [0, 360].")
    excluded_taxa = excluded_taxa or set()
    angle_deg = np.degrees(nodes["theta_width"].astype(float))
    candidates = nodes.loc[
        nodes["rank"].eq("order")
        & ~nodes["is_other"].astype(bool)
        & ~nodes["taxon"].astype(str).isin(excluded_taxa)
        & angle_deg.ge(min_angle_deg)
    ].copy()
    if candidates.empty:
        return (), ()

    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    order_clipart = (
        mapping.get("sunburst_order_clipart", {}) if mapping is not None else {}
    )
    icon_paths = (
        taxa_clipart.ensure_clipart_cached(mapping) if order_clipart else {}
    )
    font_size = 5.6
    font = FontProperties(size=font_size, weight="semibold")
    icon_size_px = renderer.points_to_pixels(8.0)
    icon_gap_px = renderer.points_to_pixels(2.4)
    ring_bottom, ring_top = ring_bounds[4]
    radius = (ring_bottom + ring_top) / 2
    labelled: list[str] = []
    icon_labelled: list[str] = []
    for row in candidates.sort_values("theta_start").itertuples(index=False):
        taxon = str(row.taxon)
        width, height = renderer.get_text_width_height_descent(
            taxon,
            font,
            ismath=False,
        )[:2]
        rotation = _sunburst_radial_text_rotation(float(row.theta_mid))
        foreground = _sunburst_contrast_color(str(row.plot_color))
        asset_name = order_clipart.get(taxon)
        icon_path = icon_paths.get(str(asset_name))
        if icon_path is not None:
            pair_width = icon_size_px + icon_gap_px + width
            center_display = np.asarray(
                axis.transData.transform((float(row.theta_mid), radius))
            )
            rotation_radians = np.radians(rotation)
            baseline = np.asarray(
                [np.cos(rotation_radians), np.sin(rotation_radians)]
            )
            pair_start = -pair_width / 2
            icon_display = center_display + baseline * (
                pair_start + icon_size_px / 2
            )
            text_display = center_display + baseline * (
                pair_start + icon_size_px + icon_gap_px + width / 2
            )
            icon_theta, icon_radius = axis.transData.inverted().transform(
                icon_display
            )
            text_theta, text_radius = axis.transData.inverted().transform(
                text_display
            )
            pair_fits = _sunburst_box_fits_wedge(
                axis,
                theta=float(icon_theta),
                radius=float(icon_radius),
                theta_start=float(row.theta_start),
                theta_width=float(row.theta_width),
                ring_bottom=ring_bottom,
                ring_top=ring_top,
                width_px=icon_size_px,
                height_px=icon_size_px,
                rotation_deg=0.0,
                padding_px=2.8,
            ) and _sunburst_box_fits_wedge(
                axis,
                theta=float(text_theta),
                radius=float(text_radius),
                theta_start=float(row.theta_start),
                theta_width=float(row.theta_width),
                ring_bottom=ring_bottom,
                ring_top=ring_top,
                width_px=width,
                height_px=height,
                rotation_deg=rotation,
                padding_px=2.8,
            )
            if pair_fits:
                rgba = taxa_clipart._tinted_icon(icon_path, foreground)
                icon = AnnotationBbox(
                    OffsetImage(rgba, zoom=icon_size_px / max(rgba.shape[:2])),
                    (float(icon_theta), float(icon_radius)),
                    xycoords="data",
                    box_alignment=(0.5, 0.5),
                    frameon=False,
                    pad=0.0,
                    annotation_clip=True,
                    zorder=23,
                )
                icon.set_gid(f"sunburst-order-icon-{taxon}")
                axis.add_artist(icon)
                label = axis.text(
                    float(text_theta),
                    float(text_radius),
                    taxon,
                    ha="center",
                    va="center",
                    rotation=rotation,
                    rotation_mode="anchor",
                    color=foreground,
                    fontsize=font_size,
                    fontweight="semibold",
                    clip_on=True,
                    zorder=24,
                )
                label.set_gid(f"sunburst-order-label-{taxon}")
                labelled.append(taxon)
                icon_labelled.append(taxon)
                continue
        if not _sunburst_box_fits_wedge(
            axis,
            theta=float(row.theta_mid),
            radius=radius,
            theta_start=float(row.theta_start),
            theta_width=float(row.theta_width),
            ring_bottom=ring_bottom,
            ring_top=ring_top,
            width_px=width,
            height_px=height,
            rotation_deg=rotation,
            padding_px=2.8,
        ):
            continue
        label = axis.text(
            float(row.theta_mid),
            radius,
            taxon,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            color=foreground,
            fontsize=font_size,
            fontweight="semibold",
            clip_on=True,
            zorder=22,
        )
        label.set_gid(f"sunburst-order-label-{taxon}")
        labelled.append(taxon)
    return tuple(labelled), tuple(icon_labelled)


def _draw_sunburst_class_labels(
    axis: plt.Axes,
    nodes: pd.DataFrame,
    *,
    min_angle_deg: float,
    ring_bounds: Mapping[int, tuple[float, float]],
    excluded_taxa: set[str] | None = None,
) -> tuple[str, ...]:
    """Draw full radial Class names only where their rendered boxes fit."""
    if not 0 <= min_angle_deg <= 360:
        raise ValueError(
            "Sunburst Class-label angle threshold must be in [0, 360]."
        )
    excluded_taxa = excluded_taxa or set()
    angle_deg = np.degrees(nodes["theta_width"].astype(float))
    candidates = nodes.loc[
        nodes["rank"].eq("class")
        & ~nodes["is_other"].astype(bool)
        & ~nodes["taxon"].astype(str).isin(excluded_taxa)
        & angle_deg.ge(min_angle_deg)
    ].copy()
    if candidates.empty:
        return ()

    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    font_size = 5.0
    font = FontProperties(size=font_size, weight="semibold")
    ring_bottom, ring_top = ring_bounds[3]
    radius = (ring_bottom + ring_top) / 2
    labelled: list[str] = []
    for row in candidates.sort_values("theta_start").itertuples(index=False):
        taxon = str(row.taxon)
        width, height = renderer.get_text_width_height_descent(
            taxon,
            font,
            ismath=False,
        )[:2]
        rotation = _sunburst_radial_text_rotation(float(row.theta_mid))
        fits = _sunburst_box_fits_wedge(
            axis,
            theta=float(row.theta_mid),
            radius=radius,
            theta_start=float(row.theta_start),
            theta_width=float(row.theta_width),
            ring_bottom=ring_bottom,
            ring_top=ring_top,
            width_px=width,
            height_px=height,
            rotation_deg=rotation,
            padding_px=2.0,
        ) or _sunburst_box_fits_wedge(
            axis,
            theta=float(row.theta_mid),
            radius=radius,
            theta_start=float(row.theta_start),
            theta_width=float(row.theta_width),
            ring_bottom=ring_bottom,
            ring_top=ring_top,
            width_px=width,
            height_px=height,
            rotation_deg=rotation,
            padding_px=0.5,
        )
        if not fits:
            continue
        label = axis.text(
            float(row.theta_mid),
            radius,
            taxon,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            color=_sunburst_contrast_color(str(row.plot_color)),
            fontsize=font_size,
            fontweight="semibold",
            clip_on=True,
            zorder=22,
        )
        label.set_gid(f"sunburst-class-label-{taxon}")
        labelled.append(taxon)
    return tuple(labelled)


def _legend_label(row: pd.Series) -> str:
    label = str(row["taxon"])
    if bool(row["is_other"]) and str(row["rank"]) != "kingdom":
        if bool(row.get("is_order_remainder", False)):
            return "Remaining orders"
        label = {
            "phylum": "Remaining phyla",
            "class": "Remaining classes",
            "order": "Remaining orders",
        }.get(str(row["rank"]), label)
    return label


def _descendant_rows(nodes: pd.DataFrame, root_id: str) -> pd.DataFrame:
    rows: list[pd.Series] = []

    def visit(node_id: str) -> None:
        row = nodes.loc[nodes["node_id"].eq(node_id)].iloc[0]
        rows.append(row)
        children = nodes.loc[nodes["parent_id"].eq(node_id)].sort_values(
            "sibling_order"
        )
        for child_id in children["node_id"].astype(str):
            visit(child_id)

    visit(root_id)
    return pd.DataFrame(rows).reset_index(drop=True)


def _draw_sunburst_legend_column(
    axis: plt.Axes,
    blocks: list[pd.DataFrame],
    *,
    line_step: float | None = None,
    font_size: float | None = None,
    block_gap_fraction: float = 1.0,
) -> None:
    axis.set_axis_off()
    n_rows = sum(len(block) for block in blocks) + (
        max(len(blocks) - 1, 0) * block_gap_fraction
    )
    if n_rows <= 0:
        return
    line_step = line_step or min(0.0285, 0.94 / max(n_rows - 1, 1))
    # These are final-size points because the canvas is authored at Nature's
    # 180-mm double-column width.  Dense columns remain above the journal's
    # 5-pt minimum; block headings remain below its 7-pt maximum.
    font_size = font_size or (6.2 if line_step >= 0.023 else 5.2)
    y = 0.97
    for block_index, block in enumerate(blocks):
        if block_index:
            y -= line_step * block_gap_fraction
        for _, row in block.iterrows():
            depth = int(row["depth"])
            swatch_x = 0.006 + SUNBURST_LEGEND_RANK_INDENT[depth]
            axis.add_patch(
                Rectangle(
                    (swatch_x, y - line_step * 0.27),
                    0.022,
                    line_step * 0.55,
                    transform=axis.transAxes,
                    facecolor=row["plot_color"],
                    edgecolor="white",
                    linewidth=0.25,
                    clip_on=False,
                )
            )
            is_root = depth == 1
            axis.text(
                swatch_x + 0.03,
                y,
                _legend_label(row),
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=font_size + (0.6 if is_root else 0.0),
                fontweight="bold" if is_root else "normal",
                color="#242424",
            )
            axis.text(
                0.62,
                y,
                f"({_format_sunburst_percent(float(row['research_share_pct']))}, "
                f"{_format_sunburst_percent(float(row['gbif_described_species_share_pct']))})",
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=font_size,
                fontweight="bold" if is_root else "normal",
                color="#4A4A4A",
            )
            y -= line_step


def plot_research_attention_sunburst(
    nodes: pd.DataFrame,
    *,
    kingdom_colors: Mapping[str, str] | None = None,
    rank_lightness: Mapping[str, float] | None = None,
    rank_saturation: Mapping[str, float] | None = None,
    legend_order_min_global_share_pct: float = 1.0,
    legend_show_orders: bool = True,
    mapping: Mapping | None = None,
    in_wheel_label_min_angle_deg: float = 18.0,
    ring_widths: Mapping[str, float] | None = None,
    order_label_min_angle_deg: float = 0.0,
    class_label_min_angle_deg: float = 0.0,
    extend_terminal_branches: bool = True,
    inner_radius: float = DEFAULT_SUNBURST_INNER_RADIUS,
) -> plt.Figure:
    """Plot research-angle taxonomy with a paired-percentage hierarchy legend.

    When the taxa mapping is supplied, roomy named wedges receive a curved
    in-wheel taxon label and its configured representative PhyloPic silhouette.
    Angular prefiltering is followed by rendered per-glyph and icon fit tests,
    so labels are omitted rather than squeezed into crowded sectors. Terminal
    branches can continue through unused outer rings as progressively lighter,
    unlabeled rank tints; these fills do not create artificial descendants.
    """
    required = {
        "node_id",
        "parent_id",
        "taxon",
        "rank",
        "depth",
        "sibling_order",
        "research_count",
        "research_share_pct",
        "research_within_parent_pct",
        "gbif_described_species_share_pct",
        "gbif_within_parent_pct",
        "is_other",
        "comparison_research_publications",
        "comparison_gbif_species",
    }
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"Sunburst nodes lack plotting columns: {sorted(missing)}")
    if not 0 <= legend_order_min_global_share_pct <= 100:
        raise ValueError("Sunburst Order legend threshold must be in [0, 100].")
    if not isinstance(legend_show_orders, bool):
        raise TypeError("Sunburst legend_show_orders must be boolean.")
    if not 0 <= in_wheel_label_min_angle_deg <= 360:
        raise ValueError("Sunburst label angle threshold must be in [0, 360].")
    if not 0 <= order_label_min_angle_deg <= 360:
        raise ValueError("Sunburst Order-label threshold must be in [0, 360].")
    if not 0 <= class_label_min_angle_deg <= 360:
        raise ValueError("Sunburst Class-label threshold must be in [0, 360].")
    if not isinstance(extend_terminal_branches, bool):
        raise TypeError("Sunburst terminal-branch extension must be boolean.")
    if (
        not isinstance(inner_radius, (int, float))
        or isinstance(inner_radius, bool)
        or inner_radius <= 0
    ):
        raise ValueError("Sunburst inner radius must be a positive number.")
    ring_widths = dict(ring_widths or DEFAULT_SUNBURST_RING_WIDTHS)
    rank_lightness = dict(rank_lightness or DEFAULT_SUNBURST_RANK_LIGHTNESS)
    rank_saturation = dict(rank_saturation or DEFAULT_SUNBURST_RANK_SATURATION)
    work = _sunburst_layout(nodes)
    colors = sunburst_node_colors(
        work,
        kingdom_colors=kingdom_colors,
        rank_lightness=rank_lightness,
        rank_saturation=rank_saturation,
    )
    work["plot_color"] = work["node_id"].astype(str).map(colors)

    # Explanatory prose belongs in the manuscript/notebook caption rather than
    # a footer band inside the PDF.  The canvas is authored at Nature's final
    # 180-mm double-column width; this prevents LaTeX from shrinking 5--7-pt
    # lettering to roughly 3 pt, as happened with the former 14.2-in canvas.
    figure = plt.figure(figsize=NATURE_SUNBURST_FIGSIZE)
    height_ratios = (1.52, 1.0) if legend_show_orders else (2.72, 1.0)
    outer = figure.add_gridspec(
        2,
        1,
        # A full-width wheel is easier to scan; the compact lower band keeps
        # every Phylum subtree intact while using four readable text columns.
        height_ratios=height_ratios,
        left=0.02,
        right=0.99,
        bottom=0.012,
        top=0.992,
        hspace=0.015,
    )
    axis = figure.add_subplot(outer[0, 0], projection="polar")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_axis_off()

    inner_radius = float(inner_radius)
    ring_gap = 0.025
    ring_bounds = _sunburst_ring_bounds(
        inner_radius=inner_radius,
        ring_widths=ring_widths,
        ring_gap=ring_gap,
    )
    outer_radius = ring_bounds[4][1] + ring_gap
    angles = np.linspace(0, 2 * np.pi, 720)
    for depth in range(1, 5):
        bottom, top = ring_bounds[depth]
        axis.plot(
            angles,
            np.repeat(bottom, len(angles)),
            color="#E8E8E8",
            lw=0.55,
            zorder=0,
        )
        axis.plot(
            angles,
            np.repeat(top, len(angles)),
            color="#E8E8E8",
            lw=0.55,
            zorder=0,
        )

    for row in work.sort_values(["depth", "theta_start"]).itertuples(index=False):
        bottom, top = ring_bounds[int(row.depth)]
        linewidth = 0.62 if float(row.theta_width) >= 0.01 else 0.25
        axis.bar(
            float(row.theta_mid),
            top - bottom,
            width=float(row.theta_width),
            bottom=bottom,
            color=row.plot_color,
            edgecolor="white",
            linewidth=linewidth,
            align="center",
            zorder=2 + int(row.depth),
        )

    terminal_continuations: list[dict[str, str]] = []
    if extend_terminal_branches:
        parent_ids = set(work["parent_id"].dropna().astype(str))
        terminal_nodes = work.loc[
            ~work["node_id"].astype(str).isin(parent_ids)
            & work["depth"].lt(len(SUNBURST_RANKS))
        ]
        for row in terminal_nodes.sort_values(
            ["depth", "theta_start"]
        ).itertuples(index=False):
            for target_depth in range(
                int(row.depth) + 1, len(SUNBURST_RANKS) + 1
            ):
                bottom, top = ring_bounds[target_depth]
                continuation_color = _sunburst_terminal_continuation_color(
                    str(row.plot_color),
                    target_depth=target_depth,
                    rank_lightness=rank_lightness,
                    rank_saturation=rank_saturation,
                )
                linewidth = 0.62 if float(row.theta_width) >= 0.01 else 0.25
                axis.bar(
                    float(row.theta_mid),
                    top - bottom,
                    width=float(row.theta_width),
                    bottom=bottom,
                    color=continuation_color,
                    edgecolor="white",
                    linewidth=linewidth,
                    align="center",
                    zorder=2 + target_depth,
                )
                terminal_continuations.append(
                    {
                        "taxon": str(row.taxon),
                        "source_rank": str(row.rank),
                        "filled_rank": SUNBURST_RANKS[target_depth - 1],
                        "plot_color": continuation_color,
                    }
                )
    axis.set_ylim(0, outer_radius + 0.03)

    labelled_taxa = (
        _draw_sunburst_wedge_labels(
            axis,
            work,
            mapping=mapping,
            min_angle_deg=in_wheel_label_min_angle_deg,
            ring_bounds=ring_bounds,
        )
        if mapping is not None
        else ()
    )
    class_labelled_taxa = _draw_sunburst_class_labels(
        axis,
        work,
        min_angle_deg=class_label_min_angle_deg,
        ring_bounds=ring_bounds,
        excluded_taxa=set(labelled_taxa),
    )
    order_labelled_taxa, order_icon_taxa = _draw_sunburst_order_labels(
        axis,
        work,
        min_angle_deg=order_label_min_angle_deg,
        ring_bounds=ring_bounds,
        excluded_taxa=set(labelled_taxa) | set(class_labelled_taxa),
        mapping=mapping,
    )

    legend_grid = outer[1, 0].subgridspec(1, 4, wspace=0.055)
    legend_axes = [figure.add_subplot(legend_grid[0, index]) for index in range(4)]
    roots = work.loc[work["parent_id"].isna()].sort_values("sibling_order")

    def visible_block(node_id: str) -> pd.DataFrame:
        block = _descendant_rows(work, node_id)
        if not legend_show_orders:
            return block.loc[block["rank"].ne("order")].reset_index(drop=True)
        return block.loc[
            block["rank"].ne("order")
            | block["is_other"]
            | block["research_share_pct"].ge(
                legend_order_min_global_share_pct
            )
            | block["gbif_described_species_share_pct"].ge(
                legend_order_min_global_share_pct
            )
        ].reset_index(drop=True)

    columns: list[list[pd.DataFrame]] = [[], [], [], []]
    animal_columns = {
        "Chordata": 0,
        "Mollusca": 0,
        "Arthropoda": 1,
        "Annelida": 1,
        "Remaining Animalia phyla": 1,
    }
    for row in roots.itertuples(index=False):
        taxon = str(row.taxon)
        root_id = str(row.node_id)
        root_only = visible_block(root_id).iloc[[0]].reset_index(drop=True)
        children = work.loc[work["parent_id"].eq(root_id)].sort_values(
            "sibling_order"
        )
        if taxon == "Animalia":
            columns[0].append(root_only)
            for child in children.itertuples(index=False):
                child_taxon = str(child.taxon)
                target = animal_columns.get(child_taxon, 2)
                columns[target].append(visible_block(str(child.node_id)))
        elif taxon == "Plantae":
            columns[2].append(root_only)
            for child in children.itertuples(index=False):
                if legend_show_orders:
                    target = 3 if str(child.taxon) == "Tracheophyta" else 2
                else:
                    target = (
                        3
                        if str(child.taxon) in {"Tracheophyta", "Bryophyta"}
                        else 2
                    )
                columns[target].append(visible_block(str(child.node_id)))
        elif taxon in {"Fungi", "Other kingdoms"}:
            columns[2 if legend_show_orders else 3].append(root_only)
        else:
            # Future roots remain visible without changing the established
            # Animalia/Plantae reading order.
            target = min(range(4), key=lambda index: sum(map(len, columns[index])))
            columns[target].append(visible_block(root_id))

    block_gap_fraction = 0.5
    row_units = [
        sum(len(block) for block in blocks)
        + max(len(blocks) - 1, 0) * block_gap_fraction
        for blocks in columns
    ]
    max_line_step = 0.034 if legend_show_orders else 0.052
    shared_line_step = min(
        max_line_step,
        0.94 / max(max(row_units) - 1, 1),
    )
    legend_font_size = 5.2 if legend_show_orders else 5.4
    for axis_legend, blocks in zip(legend_axes, columns):
        _draw_sunburst_legend_column(
            axis_legend,
            blocks,
            line_step=shared_line_step,
            font_size=legend_font_size,
            block_gap_fraction=block_gap_fraction,
        )

    figure._sunburst_labelled_taxa = labelled_taxa
    figure._sunburst_class_labelled_taxa = class_labelled_taxa
    figure._sunburst_order_labelled_taxa = order_labelled_taxa
    figure._sunburst_order_icon_taxa = order_icon_taxa
    figure._sunburst_ring_bounds = ring_bounds
    figure._sunburst_terminal_continuations = tuple(terminal_continuations)
    return figure


__all__ = [
    "SUNBURST_KINGDOM_COLORS",
    "plot_research_attention_sunburst",
    "sunburst_node_colors",
]

"""Shared fixed-class scales for publication-count choropleths.

The default uses powers of ten rather than quantiles or a continuous log scale.
Callers may provide finer fixed boundaries and colours while retaining the same
discrete colorbar machinery and separate grey zero-count fill.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap


def order_of_magnitude_bins(max_value: int) -> list[int]:
    """Return powers of ten that bracket ``max_value``.

    For example, ``25_690`` produces ``[1, 10, 100, 1_000, 10_000,
    100_000]``.
    """
    edges = [1]
    while edges[-1] <= max_value:
        edges.append(edges[-1] * 10)
    return edges


def build_count_scale(
    max_value: int,
    *,
    bins: Sequence[int] | None = None,
    colors: Sequence[str] | None = None,
):
    """Build a fixed-class count colormap and norm.

    The default remains the shared YlGn order-of-magnitude scale. Callers may
    instead supply explicit class boundaries and one colour per interval.
    """
    count_bins = (
        order_of_magnitude_bins(max_value)
        if bins is None
        else list(bins)
    )
    if len(count_bins) < 2:
        raise ValueError("Count scales require at least two boundaries")
    if any(
        right <= left
        for left, right in zip(count_bins, count_bins[1:])
    ):
        raise ValueError("Count-scale boundaries must be strictly increasing")
    if count_bins[0] != 1:
        raise ValueError("Count-scale boundaries must start at the value 1")
    if max_value >= count_bins[-1]:
        raise ValueError(
            f"Maximum count {max_value:,} is not below the final boundary "
            f"{count_bins[-1]:,}"
        )

    if colors is None:
        color_values = plt.get_cmap("YlGn")(
            np.linspace(0.12, 0.95, len(count_bins) - 1)
        )
    else:
        color_values = list(colors)
        if len(color_values) != len(count_bins) - 1:
            raise ValueError(
                "Count scales require exactly one colour per interval"
            )
    cmap = ListedColormap(color_values)
    norm = BoundaryNorm(count_bins, cmap.N)
    return cmap, norm, count_bins


def add_class_colorbar(
    fig,
    cax,
    cmap,
    norm,
    bins,
    label,
    fontsize=7,
    tick_fontsize=6,
):
    """Add the shared horizontal, uniformly spaced class colorbar."""
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        boundaries=bins,
        spacing="uniform",
    )
    cbar.set_ticks(bins)
    cbar.ax.set_xticklabels([f"{b:,}" for b in bins], fontsize=tick_fontsize)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize, length=0)
    cbar.outline.set_visible(False)
    return cbar

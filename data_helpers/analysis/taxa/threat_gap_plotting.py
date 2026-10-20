"""Region ordering and palette shared with the taxonomic figure."""

from __future__ import annotations

#: IPBES regions in a fixed order, so the legend does not reshuffle between runs.
REGION_ORDER = ("Africa", "Asia and the Pacific", "Americas", "Europe and Central Asia")

#: Drawn from the same house palette as the taxonomic figure's broad groups
#: (``schemes.broad.group_colors``) rather than a separate scheme, so the geographic
#: figure reads as its sibling. Africa takes the one warm hue: it carries the finding
#: and needs to separate from three cool neighbours.
REGION_COLORS = {
    "Africa": "#D9954C",
    "Asia and the Pacific": "#6F91C7",
    "Americas": "#2E7D32",
    "Europe and Central Asia": "#8C6BB1",
}


__all__ = [
    "REGION_COLORS",
    "REGION_ORDER",
]

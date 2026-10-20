"""Tests for the region ordering and palette shared with the taxonomic figure."""

from __future__ import annotations

from data_helpers.analysis.taxa import threat_gap_plotting as tgp


def test_regional_figure_uses_the_taxonomic_house_palette() -> None:
    """Sibling figures: the region colours come from the broad-group palette."""
    assert tgp.REGION_COLORS["Americas"] == "#2E7D32"
    assert tgp.REGION_COLORS["Asia and the Pacific"] == "#6F91C7"
    # Africa takes the one warm hue so it separates from three cool neighbours.
    assert tgp.REGION_COLORS["Africa"] == "#D9954C"

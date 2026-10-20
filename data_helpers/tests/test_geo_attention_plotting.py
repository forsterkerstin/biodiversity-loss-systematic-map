"""Tests for the observed-country radial attention figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Wedge
import pandas as pd
import pytest

from data_helpers.analysis.geo import attention_plotting as plotting


def _leaves() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "region": "Africa",
                "country": "Alpha",
                "record_count": 1_000,
                "attention_share_pct": 10.0,
            },
            {
                "region": "Africa",
                "country": "Beta",
                "record_count": 25,
                "attention_share_pct": 0.025,
            },
            {
                "region": "Americas",
                "country": "Delta",
                "record_count": 500,
                "attention_share_pct": 5.0,
            },
            {
                "region": "Americas",
                "country": "Epsilon",
                "record_count": 10,
                "attention_share_pct": 0.10,
            },
            {
                "region": "Americas",
                "country": "Zeta",
                "record_count": 100,
                "attention_share_pct": 1.0,
            },
            {
                "region": "Europe and Central Asia",
                "country": "Eta",
                "record_count": 50,
                "attention_share_pct": 0.5,
            },
        ]
    )


def _wedges_with_gid(axis, prefix: str) -> list[Wedge]:
    return [
        patch
        for patch in axis.patches
        if isinstance(patch, Wedge) and (patch.get_gid() or "").startswith(prefix)
    ]


def test_equal_country_slots_and_clockwise_attention_order() -> None:
    figure = plotting.plot_geographic_attention_radial(_leaves())
    slots = _wedges_with_gid(figure.axes[0], "country-slot:")

    assert len(slots) == len(_leaves())
    widths = [slot.theta2 - slot.theta1 for slot in slots]
    assert max(widths) - min(widths) == pytest.approx(0.0, abs=1e-10)
    assert [slot.get_gid().split(":", 2)[-1] for slot in slots] == [
        "Alpha",
        "Beta",
        "Delta",
        "Zeta",
        "Epsilon",
        "Eta",
    ]
    assert len(_wedges_with_gid(figure.axes[0], "attention-bar:")) == len(_leaves())
    plt.close(figure)


def test_region_and_country_emphasis_matches_thresholds() -> None:
    figure = plotting.plot_geographic_attention_radial(_leaves())
    axis = figure.axes[0]
    slots = {
        patch.get_gid().split(":", 2)[-1]: patch
        for patch in _wedges_with_gid(axis, "country-slot:")
    }
    labels = {
        text.get_gid().split(":", 2)[-1]: text
        for text in axis.texts
        if (text.get_gid() or "").startswith("country-label:")
    }

    assert (
        mcolors.to_hex(slots["Alpha"].get_facecolor())
        == plotting.REGION_COLORS["Africa"].lower()
    )
    assert mcolors.to_hex(slots["Beta"].get_facecolor()) != mcolors.to_hex(
        slots["Alpha"].get_facecolor()
    )
    assert labels["Alpha"].get_fontweight() == "bold"
    assert labels["Zeta"].get_fontweight() == "normal"
    assert labels["Beta"].get_fontstyle() == "italic"
    assert labels["Epsilon"].get_fontstyle() == "normal"

    region_labels = {
        text.get_gid().removeprefix("region-label:"): text.get_text()
        for text in axis.texts
        if (text.get_gid() or "").startswith("region-label:")
    }
    assert set(region_labels) == set(_leaves()["region"])
    assert " and " not in " ".join(region_labels.values()).lower()
    plt.close(figure)


def test_log_attention_bars_are_monotonic_and_compressed() -> None:
    figure = plotting.plot_geographic_attention_radial(_leaves())
    bars = {
        patch.get_gid().split(":", 2)[-1]: patch
        for patch in _wedges_with_gid(figure.axes[0], "attention-bar:")
    }

    assert bars["Alpha"].width > bars["Zeta"].width > bars["Beta"].width > 0
    assert bars["Alpha"].width / bars["Beta"].width < 4
    plt.close(figure)


def test_figure_has_scale_labels_country_names_and_solid_center() -> None:
    figure = plotting.plot_geographic_attention_radial(_leaves())
    axis = figure.axes[0]

    assert any(
        (text.get_gid() or "").startswith("attention-guide-label:")
        for text in axis.texts
    )
    assert {
        text.get_gid().split(":", 2)[-1]
        for text in axis.texts
        if (text.get_gid() or "").startswith("country-label:")
    } == set(_leaves()["country"])
    centers = [patch for patch in axis.patches if patch.get_gid() == "evidence-centre"]
    assert len(centers) == 1
    assert mcolors.to_hex(centers[0].get_facecolor()) == "#2b2b2b"
    assert tuple(figure.get_size_inches()) == pytest.approx((7.5, 7.5))
    plt.close(figure)


def test_invalid_or_unobserved_country_rows_fail_visibly() -> None:
    zero = _leaves()
    zero.loc[0, "record_count"] = 0
    zero.loc[0, "attention_share_pct"] = 0
    with pytest.raises(plotting.GeographicAttentionPlotError, match="positive"):
        plotting.plot_geographic_attention_radial(zero)

    missing_color = _leaves().assign(region="Unconfigured")
    with pytest.raises(plotting.GeographicAttentionPlotError, match="missing region"):
        plotting.plot_geographic_attention_radial(missing_color)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(40, -50), (150, 60), (230, -40), (310, 40)],
)
def test_tangential_region_rotation_is_upright(angle: float, expected: float) -> None:
    assert plotting._upright_tangential_rotation(angle) == pytest.approx(expected)

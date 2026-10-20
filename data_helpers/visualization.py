from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import colors as mcolors


# Shared semantic colors for biodiversity analyses.
BIODIVERSITY = {
    "primary": "#2E7D32",
    "primary_muted": "#B8CDB2",
    "primary_pale": "#DCEAD7",
    "eligible": "#67A84B",
    "ineligible": "#D95F5F",
    "unclear": "#C49A00",
    "screening_blue": "#6F91C7",
    "ink": "#2B2B2B",
    "neutral": "#D9D9D9",
    "paper": "#FFFFFF",
}

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fixed PDF metadata: omitting CreationDate makes re-exported figures
# byte-identical across runs, so reproduced outputs can be diffed.
PDF_METADATA = {"Creator": "biodiversity-change-systematic-map", "CreationDate": None}

# Embed TrueType (Type 42) fonts so text stays editable in exported PDF/PS/SVG,
# as journal figure guidelines require. Applied by apply_style and save_figure.
EDITABLE_TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}

OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def contrasting_text_color(background: str) -> str:
    """Return whichever shared text color contrasts more with a background."""

    def relative_luminance(color: str) -> float:
        channels = mcolors.to_rgb(color)
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return (
            0.2126 * linear[0]
            + 0.7152 * linear[1]
            + 0.0722 * linear[2]
        )

    background_luminance = relative_luminance(background)
    paper_contrast = (1.0 + 0.05) / (background_luminance + 0.05)
    ink_luminance = relative_luminance(BIODIVERSITY["ink"])
    ink_contrast = (background_luminance + 0.05) / (
        ink_luminance + 0.05
    )
    return (
        BIODIVERSITY["paper"]
        if paper_contrast >= ink_contrast
        else BIODIVERSITY["ink"]
    )


def apply_style() -> None:
    """Apply the clean, colorblind-safe matplotlib defaults used in results."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "figure.autolayout": False,
            "figure.constrained_layout.use": False,
            "figure.figsize": (3.5, 2.5),
            "font.size": 8,
            "font.family": "sans-serif",
            # Liberation Sans is metric-compatible with Arial, so label-fit decisions
            # (e.g. in the taxonomic sunburst) match on systems without Arial.
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "axes.linewidth": 0.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.labelweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.axisbelow": True,
            "axes.grid": False,
            "axes.prop_cycle": cycler(color=OKABE_ITO),
            "xtick.major.size": 3,
            "xtick.minor.size": 2,
            "xtick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "xtick.labelsize": 7,
            "xtick.direction": "out",
            "ytick.major.size": 3,
            "ytick.minor.size": 2,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.5,
            "ytick.labelsize": 7,
            "ytick.direction": "out",
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "lines.markeredgewidth": 0.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.loc": "best",
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "image.cmap": "viridis",
            "image.aspect": "auto",
            **EDITABLE_TEXT_RC,
        }
    )


def save_figure(
    fig: plt.Figure,
    filename: str | Path,
    formats: list[str] | tuple[str, ...] = ("pdf",),
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
    facecolor: str = "white",
    **kwargs: Any,
) -> list[Path]:
    """Save a figure in one or more publication-ready formats."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for output_format in formats:
        output_path = filename.with_suffix(f".{output_format}")
        save_kwargs = {
            "dpi": min(dpi, 300)
            if output_format in {"pdf", "eps", "svg"}
            else dpi,
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
            "facecolor": "none" if transparent else facecolor,
            "edgecolor": "none",
            "transparent": transparent,
            "format": output_format,
            **kwargs,
        }
        if output_format == "pdf":
            save_kwargs.setdefault("metadata", PDF_METADATA)
        with plt.rc_context(EDITABLE_TEXT_RC):
            fig.savefig(output_path, **save_kwargs)
        saved_paths.append(output_path)
        print(f"Saved: {display_path(output_path)}")

    return saved_paths


def display_path(path: str | Path) -> str:
    """Return a repository-relative path for logs, or the path unchanged."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)

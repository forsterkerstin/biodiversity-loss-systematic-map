"""Local cache and matplotlib embedding for PhyloPic taxon silhouette icons.

Assets are declared once in ``checklists/mappings/taxa_broad_groups.json``
(``clipart_assets`` plus each scheme's ``group_clipart`` lookup). PhyloPic
silhouettes are grayscale-with-alpha PNGs; embedding tints the shape to the
taxon's own group color so an icon reads as part of the existing palette
rather than generic clipart.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "taxa-clipart"

# PhyloPic assets use public-domain CC0 or PDM 1.0 terms; retain this note
# wherever a figure using them is published (per ``clipart_note`` in the map).
CLIPART_CREDIT = "Taxon silhouettes: PhyloPic contributors (CC0/PDM 1.0)."


def ensure_clipart_cached(
    mapping: Mapping, *, cache_dir: Path = DEFAULT_CACHE_DIR
) -> dict[str, Path]:
    """Download every referenced PhyloPic thumbnail once; return name -> local path.

    Downloads are skipped once a file exists at ``cache_dir``, so this is cheap
    to call at the top of every notebook run. The default cache,
    ``data/taxa-clipart/``, is committed so figures render offline.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, asset in mapping["clipart_assets"].items():
        thumbnail_url = asset.get("thumbnail_url")
        if not thumbnail_url:
            # E.g. the "Unresolved" placeholder is a Font Awesome SVG with no
            # raster thumbnail; skip it rather than fail, since results figures
            # never plot the Unresolved data-quality state as a taxon group.
            continue
        destination = cache_dir / f"{name}.png"
        if not destination.exists():
            urllib.request.urlretrieve(thumbnail_url, destination)
        paths[name] = destination
    return paths


def _tinted_icon(path: Path, hex_color: str) -> np.ndarray:
    """Recolor a grayscale+alpha PhyloPic PNG into a flat RGBA silhouette."""
    from PIL import Image

    image = np.array(Image.open(path).convert("LA"), dtype=float)
    alpha = image[..., 1] / 255.0
    rgba = np.zeros((*alpha.shape, 4))
    rgba[..., :3] = to_rgb(hex_color)
    rgba[..., 3] = alpha
    return rgba


def add_group_icons(
    axis: Axes,
    *,
    groups: Iterable[str],
    y_positions: Iterable[float],
    scheme: Mapping,
    group_colors: Mapping[str, str],
    icon_paths: Mapping[str, Path],
    zoom: float = 0.13,
    x_offset: float = -0.24,
) -> None:
    """Place a small tinted taxon silhouette to the left of each y position.

    ``x_offset`` is in axes-fraction units (negative = left of the plot area);
    the caller must widen the axis's left margin to make room before calling
    this, since matplotlib does not reserve space for artists placed outside
    the axes automatically.
    """
    group_clipart = scheme["group_clipart"]
    trans = blended_transform_factory(axis.transAxes, axis.transData)
    for group, y in zip(groups, y_positions):
        asset_name = group_clipart.get(group)
        if asset_name is None or asset_name not in icon_paths:
            continue
        rgba = _tinted_icon(icon_paths[asset_name], group_colors[group])
        offset_image = OffsetImage(rgba, zoom=zoom)
        annotation = AnnotationBbox(
            offset_image,
            (x_offset, y),
            xycoords=trans,
            frameon=False,
            box_alignment=(0.5, 0.5),
            annotation_clip=False,
        )
        axis.add_artist(annotation)


__all__ = [
    "CLIPART_CREDIT",
    "add_group_icons",
    "ensure_clipart_cached",
]

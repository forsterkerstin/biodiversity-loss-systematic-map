"""UMAP scatter of the corpus embeddings, coloured by a list-valued label column."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

UNLABELLED_COLOR = "#BDBDBD"
MIN_CENTROID_POINTS = 10


class EmbeddingMapError(ValueError):
    """Raised when the UMAP coordinates cannot be joined to the dataset."""


@dataclass(frozen=True)
class LabelledMap:
    """Per-article category codes plus the per-label distribution."""

    codes: np.ndarray  # 0 = no plotted label, k >= 1 = labels[k - 1]
    summary: pd.DataFrame  # label, n_documents, share_pct (of all records)
    n_records: int


def join_coordinates(dataset: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    """Attach ``umap_x``/``umap_y`` to ``dataset`` on ``UT``, one row per article."""
    if dataset["UT"].duplicated().any() or coordinates["UT"].duplicated().any():
        raise EmbeddingMapError("UT must be unique in both the dataset and the UMAP table.")
    joined = dataset.merge(
        coordinates[["UT", "umap_x", "umap_y"]], on="UT", how="left", validate="one_to_one"
    )
    missing = int(joined[["umap_x", "umap_y"]].isna().any(axis=1).sum())
    if missing:
        raise EmbeddingMapError(f"{missing:,} dataset rows have no UMAP coordinates.")
    return joined


def assign_labels(
    values: pd.Series,
    *,
    skip: Sequence[str] = ("Not Applicable",),
    top_n: int | None = None,
    other_label: str = "Other",
) -> LabelledMap:
    """Colour each article by its most frequent label, counting unique documents.

    A multi-label article takes the label with the largest document count, which
    is also the legend order. ``share_pct`` is the share of *all* records that name
    the label, so shares can sum to more than 100%. Skipped values (and empty
    lists) get code 0 and are counted in the audit, not in the legend.

    With ``top_n``, only the ``top_n`` most frequent labels keep their own entry;
    every other label is pooled into ``other_label``, which ranks last (an article
    naming a top label is coloured by it, not by "other").
    """
    skip_set = set(skip)
    cleaned = values.map(
        lambda items: [v for v in dict.fromkeys(items if items is not None else []) if v not in skip_set]
    )
    counts = cleaned.explode().dropna().value_counts()
    if top_n is not None and len(counts) > top_n:
        kept = counts.index[:top_n].tolist()
        kept_set = set(kept)
        cleaned = cleaned.map(
            lambda items: list(dict.fromkeys(v if v in kept_set else other_label for v in items))
        )
        counts = cleaned.explode().dropna().value_counts()
        labels = kept + [other_label]
        counts = counts.reindex(labels)
    else:
        labels = counts.index.tolist()
    rank = {label: position + 1 for position, label in enumerate(labels)}
    codes = cleaned.map(
        lambda items: min((rank[v] for v in items), default=0)
    ).to_numpy(dtype=np.int16)
    n_records = len(values)
    summary = pd.DataFrame(
        {
            "label": labels,
            "n_documents": counts.to_numpy(),
            "share_pct": counts.to_numpy() / n_records * 100,
        }
    )
    return LabelledMap(codes=codes, summary=summary, n_records=n_records)


def _format_share(share_pct: float) -> str:
    return "(<0.1%)" if share_pct < 0.05 else f"({share_pct:.1f}%)"


def plot_embedding_map(
    x: np.ndarray,
    y: np.ndarray,
    labelled: LabelledMap,
    *,
    colors: Mapping[str, str],
    display_names: Mapping[str, str] | None = None,
    legend_title: str | None = None,
    legend_ncols: int | None = None,
    point_size: float = 0.5,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (7.6, 7.4),
    seed: int = 0,
    label_centroids: bool = True,
) -> plt.Figure:
    """Draw the scatter with a percentage legend and optional centroid labels."""
    summary = labelled.summary
    missing = set(summary["label"]) - set(colors)
    if missing:
        raise EmbeddingMapError(f"No colour configured for: {sorted(missing)}")
    display_names = display_names or {}
    palette = np.array(
        [UNLABELLED_COLOR] + [colors[label] for label in summary["label"]], dtype=object
    )

    # Shuffled draw order so no category is hidden by file order.
    order = np.random.default_rng(seed).permutation(len(x))
    codes = labelled.codes[order]
    unlabelled_first = np.argsort(codes != 0, kind="stable")
    order = order[unlabelled_first]
    codes = labelled.codes[order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x[order],
        y[order],
        s=point_size,
        c=palette[codes].tolist(),
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()

    if label_centroids:
        # Greedy placement, largest category first: a label overlapping an
        # accepted one is dropped.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        accepted = []
        for position, row in enumerate(summary.itertuples(), start=1):
            member = labelled.codes == position
            if member.sum() < MIN_CENTROID_POINTS:
                continue
            text = ax.text(
                float(np.median(x[member])),
                float(np.median(y[member])),
                display_names.get(row.label, row.label),
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color="black",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
            )
            box = text.get_window_extent(renderer).expanded(1.15, 1.35)
            if any(box.overlaps(other) for other in accepted):
                text.remove()
            else:
                accepted.append(box)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=colors[row.label],
            markeredgecolor="none",
            label=f"{display_names.get(row.label, row.label)} {_format_share(row.share_pct)}",
        )
        for row in summary.itertuples()
    ]
    # Legend as a matrix under the map; matplotlib fills columns first, so
    # reorder the handles to read left-to-right, then top-to-bottom.
    ncols = legend_ncols or (3 if len(handles) <= 6 else 4)
    nrows = -(-len(handles) // ncols)
    padded = handles + [Line2D([], [], linestyle="", label="")] * (nrows * ncols - len(handles))
    by_column = [padded[row * ncols + col] for col in range(ncols) for row in range(nrows)]
    ax.legend(
        handles=by_column,
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=ncols,
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        alignment="center",
        columnspacing=1.6,
        handletextpad=0.3,
    )
    return fig

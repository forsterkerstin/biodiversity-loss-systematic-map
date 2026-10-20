"""Figure export contract: deterministic, editable-text PDFs and portable logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_helpers import visualization


def _figure() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(2, 1.5))
    axis.plot([0, 1, 2], [2, 0, 1], label="series")
    axis.set_title("Editable title")
    axis.legend()
    return figure


def test_pdf_export_is_byte_identical_and_uses_truetype_fonts(
    tmp_path: Path, capsys
) -> None:
    figure = _figure()
    try:
        first = visualization.save_figure(figure, tmp_path / "a", formats=["pdf"])
        second = visualization.save_figure(figure, tmp_path / "b", formats=["pdf"])
    finally:
        plt.close(figure)

    first_bytes = first[0].read_bytes()
    assert first_bytes == second[0].read_bytes()
    assert b"/CreationDate" not in first_bytes
    assert b"/Type3" not in first_bytes
    assert b"/TrueType" in first_bytes or b"/CIDFontType2" in first_bytes
    # Saving must not leave the editable-text settings changed globally.
    assert plt.rcParams["pdf.fonttype"] == matplotlib.rcParamsDefault["pdf.fonttype"]
    assert "Saved:" in capsys.readouterr().out


def test_display_path_is_repository_relative() -> None:
    inside = visualization.REPO_ROOT / "notebooks" / "results" / "figure.pdf"
    assert visualization.display_path(inside) == "notebooks/results/figure.pdf"
    outside = Path("/somewhere/else/figure.pdf")
    assert visualization.display_path(outside) == str(outside)

"""LaTeX rendering of the coding-agreement tables (Supplementary Tables S7 and S8).

Both tables share one layout: per-field weighted precision, recall, F1 and
Cohen's kappa grouped by dimension, followed by the five-dimension aggregate.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

DIMENSION_NAMES = {
    "L1": "Direct drivers",
    "L2": "Threats",
    "L3": "Geographic scope",
    "L4": "Ecosystem typology",
    "L6": "Taxonomy",
}

AGGREGATION_NOTE = (
    r"Weighted metrics were first averaged within each dimension and then macro-averaged across the five "
    r"dimensions, including both levels of the IUCN threat classification; Cohen's~$\kappa$ was averaged over "
    r"the fields for which it is defined."
)


def format_value(value: float, digits: int = 2) -> str:
    """Round half-up; undefined values print as '-', and 100.00 as '100'."""
    if pd.isna(value):
        return "-"
    text = format(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), "f")
    return "100" if text == "100.00" else text


def coding_table(fields: pd.DataFrame, pooled: pd.Series, caption: str, label: str) -> str:
    """Render one coding-agreement table.

    ``fields`` has one row per field with columns ``dimension``, ``field``, ``n``,
    ``precision``, ``recall``, ``f1``, ``kappa`` (fractions) and an optional
    boolean ``footnote`` that appends the superscript ``a`` to the label.
    """
    body = []
    for dimension, group in fields.groupby("dimension", sort=False):
        for i, row in enumerate(group.itertuples()):
            name = "" if i else (dimension if len(group) == 1 else rf"\multirow{{{len(group)}}}{{*}}{{{dimension}}}")
            mark = r"$^{a}$" if getattr(row, "footnote", False) else ""
            body.append(
                f"{name} & {row.field.replace('_', chr(92) + '_')}{mark} & {int(row.n)} & "
                f"{format_value(row.precision * 100)} & {format_value(row.recall * 100)} & "
                f"{format_value(row.f1 * 100)} & {format_value(row.kappa)} \\\\"
            )
        body.append(r"\hline")
    aggregate = (
        r"\multicolumn{2}{|l|}{\textbf{Aggregate (five dimensions)}} & -- & "
        f"\\textbf{{{format_value(pooled['precision'] * 100)}}} & \\textbf{{{format_value(pooled['recall'] * 100)}}} & "
        f"\\textbf{{{format_value(pooled['f1'] * 100)}}} & \\textbf{{{format_value(pooled['kappa'])}}} \\\\"
    )
    return rf"""\begin{{table}}[H]
\renewcommand{{\arraystretch}}{{1.3}}
\setlength{{\tabcolsep}}{{4pt}}
\centering
\scriptsize
\begin{{tabular}}{{|>{{\raggedright\arraybackslash}}p{{0.20\linewidth}}
                |>{{\raggedright\arraybackslash}}p{{0.16\linewidth}}
                |>{{\centering\arraybackslash}}p{{0.05\linewidth}}
                |>{{\centering\arraybackslash}}p{{0.11\linewidth}}
                |>{{\centering\arraybackslash}}p{{0.11\linewidth}}
                |>{{\centering\arraybackslash}}p{{0.09\linewidth}}
                |>{{\centering\arraybackslash}}p{{0.09\linewidth}}|}}
\hline
\textbf{{Dimension}} & \textbf{{Label}} & \textbf{{$n$}} & \textbf{{Precision (\%)}} & \textbf{{Recall (\%)}} & \textbf{{F1 (\%)}} & \textbf{{$\kappa$}} \\
\hline
{chr(10).join(body)}
{aggregate}
\hline
\end{{tabular}}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
"""

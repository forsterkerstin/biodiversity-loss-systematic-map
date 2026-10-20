"""Pool per-field coding agreement with the published reference labels (Supplementary Table S7).

Weighted precision, recall, F1 and Cohen's kappa of the held-out test runs are
pooled with a two-stage macro-average: fields are averaged within their
dimension (L1-L4 and L6), and the five dimension means are then averaged with
equal weight. Kappa is undefined for a field when no label varies across the
scored records in the reference (for example phylum, where every test record is
Chordata); undefined values are skipped, not counted as zero.

Inputs are the metric workbooks written by ``python -m evals_local``:

    python -m evals_local.summary
    python -m evals_local.summary --out data/labels/eval/summary
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .latex_tables import AGGREGATION_NOTE, DIMENSION_NAMES, coding_table, format_value

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "data" / "labels" / "eval"
METRICS = ("precision", "recall", "f1", "kappa")


@dataclass(frozen=True)
class Field:
    """One evaluated coding field and where its metrics are stored."""

    label_set: str
    field: str
    run: str
    metrics_file: str


# The 13 evaluated fields and the held-out test run of the configuration selected
# on the development split. "specie" is the field name
# used by the taxa evaluation for species.
FIELDS: tuple[Field, ...] = (
    Field("L1", "driver", "l1_test_010726_low_f1", "driver_driver_label_metrics.xlsx"),
    Field("L2", "threats_l0", "l2_test_010726_medium_f1", "threats_l0_threats_l0_label_metrics.xlsx"),
    Field("L2", "threats_l1", "l2_test_010726_medium_f1", "threats_l1_threats_l1_label_metrics.xlsx"),
    Field("L3", "region", "l3_test_010726_low_f1", "geography_region_label_metrics.xlsx"),
    Field("L3", "sub-region", "l3_test_010726_low_f1", "geography_sub-region_label_metrics.xlsx"),
    Field("L3", "country", "l3_test_010726_low_f1", "geography_country_label_metrics.xlsx"),
    Field("L4", "realm", "l4_test_010726_high_f1", "ecosystems_realm_realm_label_metrics.xlsx"),
    Field("L6", "kingdom", "l6_test_010726_high_f1", "taxa_kingdom_label_metrics.xlsx"),
    Field("L6", "phylum", "l6_test_010726_high_f1", "taxa_phylum_label_metrics.xlsx"),
    Field("L6", "class", "l6_test_010726_high_f1", "taxa_class_label_metrics.xlsx"),
    Field("L6", "order", "l6_test_010726_high_f1", "taxa_order_label_metrics.xlsx"),
    Field("L6", "genus", "l6_test_010726_high_f1", "taxa_genus_label_metrics.xlsx"),
    Field("L6", "specie", "l6_test_010726_high_f1", "taxa_specie_label_metrics.xlsx"),
)


def _read_excel(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def weighted_field_metrics(table: pd.DataFrame) -> dict[str, float]:
    """Return the support-weighted metrics of one field from its label table.

    ``table`` is the per-label output of ``metrics.label_metrics`` (per-label
    rows followed by ``_macro``/``_weighted``/``_micro`` rows).
    """
    labels = table["label"].astype(str)
    weighted = table.loc[labels.eq("_weighted")]
    if len(weighted) != 1:
        raise ValueError("Expected exactly one '_weighted' row in the metrics table.")
    row = weighted.iloc[0]
    result = {metric: float(row[metric]) for metric in METRICS}
    result["n"] = float(row["n"])

    # Kappa is uninformative for a label that is present in every scored record
    # (or in none): chance agreement is then perfect by construction. When no
    # label varies in the reference, the field's kappa is reported as undefined.
    per_label = table.loc[~labels.str.startswith("_")]
    support = pd.to_numeric(per_label["support"], errors="coerce")
    varies = support.gt(0) & support.lt(result["n"])
    defined = pd.to_numeric(per_label["kappa"], errors="coerce").notna()
    if not (varies & defined).any():
        result["kappa"] = math.nan
    return result


def load_test_metrics(eval_root: Path = EVAL_ROOT, fields: tuple[Field, ...] = FIELDS) -> pd.DataFrame:
    """Weighted metrics per field from the held-out test-run metric workbooks."""
    rows = []
    for field in fields:
        path = eval_root / field.run / "metrics" / field.metrics_file
        metrics = weighted_field_metrics(_read_excel(path))
        rows.append({"label_set": field.label_set, "field": field.field, **metrics,
                     "source": str(path.relative_to(eval_root))})
    return pd.DataFrame(rows)


def pool(field_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Macro-average fields within each label set, then label sets equally.

    Returns the per-label-set means and the pooled summary. Undefined (NaN)
    values are skipped at both stages.
    """
    by_label_set = field_metrics.groupby("label_set", sort=True)[list(METRICS)].mean()
    return by_label_set, by_label_set.mean()


def reference_table(fields: pd.DataFrame, pooled: pd.Series) -> str:
    """Supplementary Table S7: LLM test-split agreement with the published reference labels."""
    table = fields.assign(
        dimension=fields["label_set"].map(DIMENSION_NAMES),
        field=fields["field"].replace({"specie": "species"}),
    )
    p = {m: format_value(pooled[m] * 100) for m in ("precision", "recall", "f1")}
    caption = (
        r"\textbf{Agreement between LLM and published reference labels.} Predictions of the final LLM configuration "
        r"for the held-out test split of the Reference Coding Dataset compared with the expert full-text labels of the "
        r"published evidence syntheses. Across the five dimensions, the LLM achieved a macro-averaged weighted "
        f"precision of {p['precision']}\\%, recall of {p['recall']}\\%, F1 of {p['f1']}\\%, and Cohen's~$\\kappa$ "
        f"of {format_value(pooled['kappa'])}. "
        + AGGREGATION_NOTE
        + r" $n$ denotes records valid for scoring and varies by field because records without an expert label, and "
        r"records with unclear or inapplicable labels (e.g., \textit{Unclear}, \textit{Not applicable}), are excluded "
        r"by the scoring rules; records without a detailed-level threat prediction (broad level \textit{Unclear} or "
        r"\textit{Other options}) are excluded from threats\_l1. The \textit{phylum} test data contain only one "
        r"observed phylum (\textit{Chordata}); accordingly, performance may be overstated and Cohen's~$\kappa$ is not "
        r"applicable (-)."
    )
    return coding_table(table, pooled, caption, "tab:cc2_coding")


def _format(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.copy()
    for metric in ("precision", "recall", "f1"):
        shown[metric] = (shown[metric] * 100).round(2)
    shown["kappa"] = shown["kappa"].round(3)
    return shown


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--out", type=Path, help="Directory for the summary CSV files.")
    args = parser.parse_args(argv)

    check = "LLM vs reference labels (test split)"
    fields = load_test_metrics(args.eval_root)
    by_label_set, overall = pool(fields)
    print(f"\n{check}: weighted metrics per field")
    print(_format(fields.drop(columns="source")).to_string(index=False))
    print(f"\n{check}: macro-average within label set")
    print(_format(by_label_set.reset_index()).to_string(index=False))
    print(
        f"\n{check} pooled: precision {overall['precision'] * 100:.2f}%, "
        f"recall {overall['recall'] * 100:.2f}%, F1 {overall['f1'] * 100:.2f}%, "
        f"kappa {overall['kappa']:.2f}"
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "table_llm_reference_coding.tex").write_text(
            reference_table(fields, overall), encoding="utf-8"
        )
        fields.assign(check=check).to_csv(args.out / "coding_validation_fields.csv", index=False)
        pd.DataFrame([{"check": check, **overall.to_dict()}]).to_csv(
            args.out / "coding_validation_pooled.csv", index=False
        )
        print(f"\nWrote coding_validation_fields.csv, coding_validation_pooled.csv and "
              f"table_llm_reference_coding.tex (Supplementary Table S7) to {args.out}")


if __name__ == "__main__":
    main()

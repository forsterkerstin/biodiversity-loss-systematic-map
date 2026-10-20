"""Record-matched agreement between LLM and human coding (Supplementary Table S8).

Compares the saved predictions of the final LLM configuration with the authors'
title-and-abstract annotations of the same records. The annotations
(``annotated_coding_dataset.xlsx``, 100 records per dimension) are the
comparison labels. Predictions come from the final configuration of each
dimension: the training, held-out test and development runs at the selected
reasoning effort, which share one model, prompt, output schema and effort.

Scoring is identical to the other coding evaluations: the task's normalizer, the
placeholder masks of ``evals_local.run`` and ``metrics.label_metrics`` (records
without a human label are skipped; empty predictions are scored). The detailed
IUCN threat level is generated only when the broad level selects a category with
subcategories; records without that step are excluded from ``threats_l1``.
Fields are averaged within each dimension, and the five dimension means are
averaged with equal weight (both threat levels included).

    python -m evals_local.matched_agreement
    python -m evals_local.matched_agreement --out data/labels/eval/llm_human_matched
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .latex_tables import AGGREGATION_NOTE, coding_table, format_value
from .metrics import label_metrics
from .normalizers import to_label_set
from .run import _normalizer_for
from .summary import EVAL_ROOT, METRICS, _read_excel, weighted_field_metrics

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "consistency-check-datasets" / "data-coding" / "annotated_coding_dataset.xlsx"
DEFAULT_OUT = EVAL_ROOT / "llm_human_matched"
UT_COLUMN = "UT (Unique WOS ID)"
THREAT_PLACEHOLDERS = frozenset({"no threat_l2 candidates found", "no threat_l1 candidates found", "unclear"})
TAXA_PLACEHOLDERS = frozenset({"not applicable", "unclear"})


@dataclass(frozen=True)
class MatchedField:
    """One evaluated coding field: where its annotations and predictions live."""

    dimension: str
    field: str
    sheet: str
    task: str
    prediction_column: str
    runs: tuple[str, ...]
    placeholders: frozenset[str] | None = None

    @property
    def display_name(self) -> str:
        return "species" if self.field == "specie" else self.field


def _final_runs(level: str, effort: str) -> tuple[str, ...]:
    return tuple(f"{level}_{split}_010726_{effort}_f1" for split in ("dev", "train", "test"))


_DRIVER = _final_runs("l1", "low")
_THREATS = _final_runs("l2", "medium")
_GEOGRAPHY = _final_runs("l3", "low")
_ECOSYSTEMS = _final_runs("l4", "high")
_TAXA = _final_runs("l6", "high")

FIELDS: tuple[MatchedField, ...] = (
    MatchedField("Direct drivers", "driver", "l1_manual_sample", "driver", "pred_driver", _DRIVER),
    MatchedField("Threats", "threats_l0", "l2_manual_sample", "threats_l0", "pred_threat_l0", _THREATS, THREAT_PLACEHOLDERS),
    MatchedField("Threats", "threats_l1", "l2_manual_sample", "threats_l1", "pred_threat_l1", _THREATS, THREAT_PLACEHOLDERS),
    MatchedField("Geographic scope", "region", "l3_manual_sample", "geography", "pred_regions", _GEOGRAPHY),
    MatchedField("Geographic scope", "sub-region", "l3_manual_sample", "geography", "pred_subregions", _GEOGRAPHY),
    MatchedField("Geographic scope", "country", "l3_manual_sample", "geography", "pred_countries", _GEOGRAPHY),
    MatchedField("Ecosystem typology", "realm", "l4_manual_sample", "ecosystems_realm", "pred_realm", _ECOSYSTEMS),
    *(
        MatchedField("Taxonomy", field, "l6_manual_sample", "taxa",
                     "pred_species" if field == "specie" else f"pred_{field}", _TAXA, TAXA_PLACEHOLDERS)
        for field in ("kingdom", "phylum", "class", "order", "genus", "specie")
    ),
)


def _placeholder_mask(frame: pd.DataFrame, columns: list[str], placeholders: frozenset[str]) -> pd.Series:
    parts = [frame[c].apply(lambda value: not (to_label_set(value) & placeholders)) for c in columns]
    return pd.concat(parts, axis=1).all(axis=1)


def load_predictions(field: MatchedField, eval_root: Path = EVAL_ROOT) -> pd.DataFrame:
    """Saved final-configuration predictions of one field, one row per record."""
    frames = [
        _read_excel(eval_root / run / "data" / f"{field.task}.xlsx")[["custom_id", field.prediction_column]]
        for run in field.runs
    ]
    predictions = pd.concat(frames, ignore_index=True)
    predictions["custom_id"] = predictions["custom_id"].astype(str).str.strip()
    if predictions["custom_id"].duplicated().any():
        raise ValueError(f"{field.field}: a record has more than one final-configuration prediction.")
    return predictions


def score_field(field: MatchedField, annotations: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Weighted agreement of one field, with human annotations as comparison labels."""
    matched = annotations[[UT_COLUMN, field.field]].merge(
        predictions, left_on=UT_COLUMN, right_on="custom_id", how="left", validate="one_to_one"
    )
    without_output = matched["custom_id"].isna()
    matched = matched.loc[~without_output]
    mask = (
        _placeholder_mask(matched, [field.field, field.prediction_column], field.placeholders)
        if field.placeholders
        else None
    )
    table = label_metrics(
        matched, field.field, field.prediction_column,
        normalizer=_normalizer_for(field.task, field.field), include_mask=mask,
    )
    metrics = weighted_field_metrics(table)
    return {
        "dimension": field.dimension,
        "field": field.display_name,
        "annotated": len(annotations),
        "without_output": int(without_output.sum()),
        "n": int(metrics["n"]),
        **{metric: metrics[metric] for metric in METRICS},
    }


def matched_agreement(
    annotations_path: Path = ANNOTATIONS, eval_root: Path = EVAL_ROOT
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Per-field agreement, the five-dimension aggregate and annotation counts."""
    sheets: dict[str, pd.DataFrame] = {}
    rows = []
    for field in FIELDS:
        if field.sheet not in sheets:
            sheet = _read_excel(annotations_path, sheet_name=field.sheet)
            sheet[UT_COLUMN] = sheet[UT_COLUMN].astype(str).str.strip()
            sheets[field.sheet] = sheet
        rows.append(score_field(field, sheets[field.sheet], load_predictions(field, eval_root)))
    fields = pd.DataFrame(rows)
    by_dimension = fields.groupby("dimension", sort=False)[list(METRICS)].mean()
    articles = set().union(*(set(sheet[UT_COLUMN]) for sheet in sheets.values()))
    counts = {"assignments": sum(len(sheet) for sheet in sheets.values()), "articles": len(articles)}
    return fields, by_dimension.mean(), counts


def latex_table(fields: pd.DataFrame, pooled: pd.Series, counts: dict[str, int]) -> str:
    """Supplementary Table S8 in the manuscript's table style."""
    excluded = int(fields["without_output"].sum())
    p = {m: format_value(pooled[m] * 100) for m in ("precision", "recall", "f1")}
    caption = (
        r"\textbf{Agreement between LLM and human coding.} Saved predictions of the final LLM configuration "
        r"compared with the authors' title-and-abstract annotations of the same records, with the human annotations "
        f"as comparison labels. The annotations comprise {counts['assignments']} record--dimension assignments "
        f"(100 per dimension) covering {counts['articles']} unique articles. Across the five dimensions, the LLM and "
        f"the human annotations agreed with a macro-averaged weighted precision of {p['precision']}\\%, recall of "
        f"{p['recall']}\\%, F1 of {p['f1']}\\%, and Cohen's~$\\kappa$ of {format_value(pooled['kappa'])}. "
        + AGGREGATION_NOTE
        + r" $n$ denotes records valid for scoring and varies by field because records without a human label, and "
        r"records with unclear or inapplicable labels (e.g., \textit{Unclear}, \textit{Not applicable}), are excluded "
        r"by the scoring rules. $^{a}$The detailed threat level is generated only when the broad level selects a "
        f"category with subcategories; {excluded} records for which the broad level returned \\textit{{Unclear}} or "
        r"\textit{Other options} have no detailed-level prediction and are excluded."
    )
    return coding_table(fields.assign(footnote=fields["without_output"] > 0), pooled, caption, "tab:llm_human_coding")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--out", type=Path, help="Directory for the CSV files and the LaTeX table.")
    args = parser.parse_args(argv)

    fields, pooled, counts = matched_agreement(args.annotations, args.eval_root)
    shown = fields.assign(**{m: (fields[m] * 100).round(2) for m in ("precision", "recall", "f1")},
                          kappa=fields["kappa"].round(3))
    print(shown.to_string(index=False))
    print(
        f"\n{counts['assignments']} record-dimension assignments, {counts['articles']} unique articles. "
        f"Five-dimension aggregate: precision {pooled['precision'] * 100:.2f}%, recall {pooled['recall'] * 100:.2f}%, "
        f"F1 {pooled['f1'] * 100:.2f}%, kappa {pooled['kappa']:.2f}"
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        fields.to_csv(args.out / "llm_human_matched_fields.csv", index=False)
        pd.DataFrame([{"scope": "five dimensions", **pooled.to_dict(), **counts}]).to_csv(
            args.out / "llm_human_matched_pooled.csv", index=False
        )
        (args.out / "table_llm_human_coding.tex").write_text(latex_table(fields, pooled, counts), encoding="utf-8")
        print(f"Wrote CSV files and table_llm_human_coding.tex to {args.out}")


if __name__ == "__main__":
    main()

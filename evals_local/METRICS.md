# Evaluation metrics

This document describes the metrics produced by `evals_local` for:

- screening (`L0`)
- all other coding tasks (`L1` to `L6`)

It focuses on:

- which metrics are used
- how they are computed
- how aggregated summaries are formed
- where the files are written

## Output location

All eval outputs are written under:

`data/labels/eval/<run_name>/`

Typical structure:

```text
data/labels/eval/<run_name>/
├── data/
│   └── <task>.xlsx
└── metrics/
    ├── <task>_<col>_label_metrics.xlsx
    └── <task>_<col>_confusion.xlsx
```

Notes:

- screening writes:
  - `data/screening.xlsx`
  - `metrics/screening_eligibility_label_metrics.xlsx`
  - `metrics/screening_eligibility_confusion.xlsx`
- non-screening coding tasks write:
  - one merged `data/<task>.xlsx`
  - one metrics workbook per evaluated truth column
- confusion matrix workbooks are only written for screening

## Screening metrics

Screening is treated as a binary task with:

- positive class = `ELIGIBLE`
- negative class = `NOT_ELIGIBLE`

### Screening summary workbook

File:

`metrics/screening_eligibility_label_metrics.xlsx`

Fields:

- `positive_label`
- `tp`
- `fp`
- `fn`
- `tn`
- `eligible_precision` (only when the reference includes `NOT_ELIGIBLE` records)
- `eligible_recall`
- `eligible_f1` (only when the reference includes `NOT_ELIGIBLE` records)
- `positive_support`
- `scored_rows`

### Screening formulas

- `tp`: truth = `ELIGIBLE`, pred = `ELIGIBLE`
- `fp`: truth = `NOT_ELIGIBLE`, pred = `ELIGIBLE`
- `fn`: truth = `ELIGIBLE`, pred = `NOT_ELIGIBLE`
- `tn`: truth = `NOT_ELIGIBLE`, pred = `NOT_ELIGIBLE`

Primary metric:

- `eligible_recall = TP / (TP + FN)`

With negatives in the reference (the search-results sample):

- `eligible_precision = TP / (TP + FP)`
- `eligible_f1 = 2 * precision * recall / (precision + recall)`

Supporting counts:

- `positive_support = TP + FN`
- `scored_rows = TP + FP + FN + TN`

### Screening data workbook

File:

`data/screening.xlsx`

Contains per-record fields such as:

- `true_label`
- `pred_label`
- `tp`
- `fp`
- `fn`
- `tn`

Rows with empty truth labels are excluded from screening summary computation.

## Coding-task metrics

All non-screening tasks use a shared one-vs-rest label evaluation framework.

This applies to tasks such as:

- `driver`
- `threats_l0`
- `threats_l1`
- `geography`
- `ecosystems_realm`
- `taxa`

Notes:

- some task types may still load predictions without producing a standard truth-vs-pred metrics workbook if there is no truth-column mapping for that level
- these are `threats_l2` and `ecosystems_efg`

Some tasks produce one metrics workbook, while others produce one workbook per evaluated column.

Examples:

- `driver` -> one workbook
- `geography` -> `region`, `sub-region`, `country`
- `taxa` -> `kingdom`, `phylum`, `class`, `order`, `genus`, `specie`

## Per-label metric workbook schema

Each non-screening metrics workbook contains:

- `label`
- `tp`
- `fp`
- `fn`
- `tn`
- `n`
- `precision`
- `recall`
- `f1`
- `kappa`
- `support`

At the bottom of each workbook there are summary rows:

- `_macro`
- `_weighted`
- `_micro`

## Per-label definitions

For one label, after normalization and exclusions, the summary workbook uses a one-vs-rest row-count view:

- `tp = count(rows where the label is in truth and in pred)`
- `fp = count(rows where the label is not in truth and is in pred)`
- `fn = count(rows where the label is in truth and is not in pred)`
- `tn = total_scored_rows - tp - fp - fn`

Interpretation:

- `tp`: label truly present and predicted
- `fp`: label predicted but not truly present
- `fn`: label truly present but missed
- `tn`: label absent in both truth and prediction

This is different from the per-record merged data columns, where `*_tp`, `*_fp`, `*_fn`, and `*_tn` are set-counts computed within each row's label sets.

### Support

Support is:

- `support = tp + fn`

Meaning:

- the number of true instances of that label in the evaluated truth set

### N

For a per-label row:

- `n = tp + fp + fn + tn`

In practice this equals:

- the number of scored rows for that evaluated column after exclusions

## Core per-label metrics

### Precision

```text
precision = TP / (TP + FP)
```

If `TP + FP = 0`, precision is set to `0`.

### Recall

```text
recall = TP / (TP + FN)
```

If `TP + FN = 0`, recall is set to `0`.

### F1

```text
f1 = 2 * precision * recall / (precision + recall)
```

If `precision + recall = 0`, F1 is set to `0`.

### Cohen's kappa

Kappa is computed per label using the binary one-vs-rest confusion counts.

For one label:

- `N = TP + FP + FN + TN`
- observed agreement:

```text
Po = (TP + TN) / N
```

- expected agreement:

```text
Pe =
[(TP + FN) / N] * [(TP + FP) / N]
+ [(FP + TN) / N] * [(FN + TN) / N]
```

- final kappa:

```text
kappa = (Po - Pe) / (1 - Pe)
```

If `1 - Pe = 0`, kappa is written as `NaN`.

## Summary rows

### Macro

`_macro` gives equal weight to each label.

- macro precision = mean of per-label precision
- macro recall = mean of per-label recall
- macro F1 = mean of per-label F1
- macro kappa = mean of evaluable per-label kappa values

Notes:

- every label contributes equally
- labels with zero support still affect macro metrics if they appear in the workbook

### Weighted

`_weighted` uses support as the weight.

Weight:

```text
weight(label) = support = TP + FN
```

So:

- weighted precision = support-weighted mean of per-label precision
- weighted recall = support-weighted mean of per-label recall
- weighted F1 = support-weighted mean of per-label F1
- weighted kappa = support-weighted mean of per-label kappa

Important consequence:

- labels with `support = 0` contribute zero weight to `_weighted`

### Micro

`_micro` pools all label-wise binary decisions first, then computes one metric from the pooled counts.

If there are:

- `R` scored rows
- `L` evaluated labels in the workbook

then micro evaluates:

- `R × L` binary label decisions

Micro pooled counts:

- `tp_sum = sum(tp over labels)`
- `fp_sum = sum(fp over labels)`
- `fn_sum = sum(fn over labels)`
- `tn_sum = (R × L) - tp_sum - fp_sum - fn_sum`

Micro metrics:

- micro precision from pooled `tp_sum` and `fp_sum`
- micro recall from pooled `tp_sum` and `fn_sum`
- micro F1 from pooled precision and recall
- micro kappa from pooled `tp_sum`, `fp_sum`, `fn_sum`, `tn_sum`

Important consequence:

- `_micro n` is often much larger than the record count
- this is expected because micro counts all record-label decisions, not just records

## Exclusions before metric computation

### Generic rule

Rows are excluded from aggregate metric computation when the normalized truth set is empty.

This means:

- empty truth
- `NaN`
- empty list-like truth

do not contribute to the metrics.

### Prediction side

An empty prediction is still scored.

That is important because an empty prediction can correctly produce false negatives.

### Task-specific exclusion masks

Some tasks exclude placeholder values before computing metrics.

Examples:

- threats:
  - `unclear`
  - `no threat_l1 candidates found`
  - `no threat_l2 candidates found`
- taxa:
  - `not applicable`
  - `unclear`
- ecosystems biome:
  - `no biome candidates found`

These exclusions apply before aggregate metrics are computed.

## Per-record merged data files

For non-screening tasks, the merged `data/<task>.xlsx` files contain per-record confusion-style columns for each evaluated column.

Examples:

- `driver_tp`, `driver_fp`, `driver_fn`, `driver_tn`
- `region_tp`, `region_fp`, `region_fn`, `region_tn`
- `genus_tp`, `genus_fp`, `genus_fn`, `genus_tn`

These are per-record counts over the evaluated label universe for that column.

Rows excluded by task masks or by empty truth keep these per-record fields blank.

## Practical reading guidance

### Screening

Use:

- confusion workbook
- `eligible_recall`

This is the primary decision metric for screening.

### Coding tasks

Use:

- per-label rows when you want to inspect specific labels
- `_weighted` when you want one practical overall summary for the task
- `_macro` when you want equal treatment across labels
- `_micro` when you want pooled label-decision performance

The reported coding agreement uses the `_weighted` summaries, pooled across the five dimensions with `python -m evals_local.matched_agreement` (see [README.md](README.md)).

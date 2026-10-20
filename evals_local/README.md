# Evaluation

Compares model outputs from a labelling run with reference labels and writes agreement metrics. It reads the archived model responses in `data/artifacts/batch_outputs/<run_name>/` and makes no API calls. Run commands from the repository root.

```bash
python -m evals_local <run_name> --tasks <task>
python -m evals_local l1_test_010726_low_f1 --tasks driver     # example
```

| Task | Reference labels (`data/labels/`) | Evaluated fields |
| --- | --- | --- |
| `screening` | `l0/` | eligibility |
| `driver` | `l1/L1) driver set.csv` | IPBES direct drivers |
| `threats` (`threats_l0`, `threats_l1`) | `l2/L2) threats set.csv` | IUCN threat levels |
| `geography` | `l3/L3) geography set.csv` | region, sub-region, country |
| `ecosystems_realm` | `l4/L4) ecosystem set.csv` | realm |
| `taxa` | `l6/L6) taxa set.csv` | kingdom, phylum, class, order, genus, species |

Taxa predictions are expanded to their lineage with `checklists/mappings/gbif_lookup_cache.pkl`. Other reference files can be supplied with `--per-task-label-paths task=path` (and `--per-task-label-sheets task=sheet` for workbooks); run `python -m evals_local --help` for all options.

## Outputs

Results are written to `data/labels/eval/<run_name>/`: `data/<task>.xlsx` holds the matched reference and predicted labels per record, and `metrics/` one workbook per evaluated field with per-label precision, recall, F1 and Cohen's kappa followed by `_macro`, `_weighted` and `_micro` summaries (screening: confusion counts with `ELIGIBLE` as the positive class). Definitions and exclusions are in [METRICS.md](METRICS.md).

## Reported evaluations

| Check | Runs in `data/labels/eval/` |
| --- | --- |
| Screening against expert labels | `l0_{dev,train,test}_010726_*` |
| Screening against author annotations of search results | `l0_cc1_search-sample_010726_medium_f1` |
| Coding: LLM against published reference labels, held-out test split (Supplementary Table S7) | `summary/`, from the test runs listed in `summary.FIELDS` |
| Coding: LLM against author annotations (Supplementary Table S8) | `llm_human_matched/`, from the final-configuration runs `l{1,2,3,4,6}_{dev,train,test}_010726_*` listed in `matched_agreement.FIELDS` |

The reasoning effort of each run (`low`, `medium`, `high`) is part of its name. The coding agreement is recomputed with

```bash
python -m evals_local.matched_agreement                                            # print
python -m evals_local.matched_agreement --out data/labels/eval/llm_human_matched   # write CSV files and the LaTeX table
```

It matches the saved predictions of the final configuration of each dimension to the authors' title-and-abstract annotations (`annotated_coding_dataset.xlsx`; 500 record–dimension assignments covering 479 articles) and scores them with the same normalizers, masks and metrics as all other evaluations. Each of the 13 fields contributes its `_weighted` row; fields are averaged within their dimension and the five dimension means are averaged with equal weight. Kappa is undefined for a field when no label varies across the scored records, and undefined values are skipped.

Table S7 is recomputed from the test-split metric workbooks with the same aggregation:

```bash
python -m evals_local.summary --out data/labels/eval/summary   # writes table_llm_reference_coding.tex and the CSV files
```

The development and training runs were used to select the reasoning effort of each dimension.

# Labelling

Screens and codes bibliographic records with an OpenAI model, using the title and abstract of each record and the prompts in [`checklists/prompts/`](../checklists/prompts/). The [screening prompt](../checklists/prompts/screen.md) and the coding prompts define every decision rule.

Runs require `OPENAI_API_KEY` (in `.env` or the environment). Run commands from the repository root.

## Running a task

```bash
python labelling/orchestrator.py <input_dir> <run_name> --task <run_key>
python labelling/orchestrator.py data/labels/l1/train my_run --task driver    # example
```

Input folders contain `.csv`, `.xls` or `.xlsx` files with a record ID (`UT`, `UT (Unique WOS ID)` or `custom_id`), a title (`Article Title` or `title`) and an abstract (`Abstract` or `abstract`). Only the title and abstract are sent to the model.

| Level | Content | Run key | Prompts |
| --- | --- | --- | --- |
| L0 | Screening | `screening` | [screen](../checklists/prompts/screen.md) |
| L1 | IPBES direct drivers | `driver` | [driver](../checklists/prompts/classify_direct_driver.md) |
| L2 | IUCN threats (three levels, run in order) | `threats_l0`, `threats_l1`, `threats_l2` | [core](../checklists/prompts/classify_threats_core.md), [L0](../checklists/prompts/classify_threats.md), [L1](../checklists/prompts/classify_threats_l1.md), [L2](../checklists/prompts/classify_threats_l2.md) |
| L3 | Country, IPBES sub-region and region | `geography` | [geography](../checklists/prompts/classify_region.md) |
| L4 | Global Ecosystem Typology (three levels, run in order) | `ecosystems_realm`, `ecosystems_biome`, `ecosystems_efg` | [core](../checklists/prompts/classify_ecosystem_typology_core.md), [realm](../checklists/prompts/classify_ecosystem_typology_realm.md), [biome](../checklists/prompts/classify_ecosystem_typology_biome.md), [EFG](../checklists/prompts/classify_ecosystem_typology_efg.md) |
| L5 | Study design and methods | `study` | [study](../checklists/prompts/classify_study.md) |
| L6 | Taxa | `taxa` | [taxa](../checklists/prompts/classify_taxa.md) |

The levels of L2 and L4 use the same run name; each level restricts its labels to children of the labels chosen at the level above.

## Outputs

A run named `my_run` writes the normalised input to `data/artifacts/datasets/my_run-dataset.json`, the request files to `data/artifacts/batches/my_run/`, and the raw model responses to `data/artifacts/batch_outputs/my_run/`. Requests and responses carry the record ID as `custom_id`. Evaluate a run with [`evals_local`](../evals_local/README.md).

## Settings

Defaults are in [`checklists/mappings/repo_config.json`](../checklists/mappings/repo_config.json) under `orchestrator`.

| Setting | Default | Override |
| --- | --- | --- |
| Model | `gpt-5-nano-2025-08-07` | `--model` |
| Reasoning effort | `medium` | `--reasoning` |
| Records per request file | 10,000 | `--batch-size` |
| Submission | `batch` (Batch API); `live` for synchronous calls | `orchestrator.submission_mode` |
| Evaluate after the run | `false` | `orchestrator.run_evals` |

## From responses to tables

```bash
python labelling/process_screening.py --run-list checklists/mappings/run_names.json --run-key screening --output-dir screened-partitions
python labelling/process_coding.py --task driver
```

`process_screening.py` combines screening runs into `data/screened-partitions/` (all decisions and the eligible records). `process_coding.py` writes `data/coding-<task>/` for `driver`, `threats`, `geography`, `ecosystems`, `study`, `taxa` or `taxa-with-api`. `taxa-with-api` resolves taxon names through GBIF's species-match API and assigns broad taxonomic groups; its output is the taxon information used in the analyses.

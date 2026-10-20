# data_helpers

Reusable code behind the notebooks: building the reference-label datasets, preparing external reference data and the analysis datasets, and the calculations and figures of each result. Configuration lives in [`checklists/mappings/repo_config.json`](../checklists/mappings/repo_config.json) (datasets and paths) and [`checklists/mappings/results_config.json`](../checklists/mappings/results_config.json) (analyses and figures). Run commands from the repository root.

## Shared modules

| Module | Purpose |
| --- | --- |
| `_config.py`, `results_config.py` | Load and validate the two configuration files |
| `labels.py` | Parsing of list-valued label fields |
| `visualization.py` | Figure style, palettes and export |
| `__main__.py` | `python -m data_helpers`: train/dev/test splits of the reference labels |

## `datasets/`: reference labels and validation data

| Module | Purpose |
| --- | --- |
| `build_screening.py`, `build_coding.py` | Build the reference-label files (L0–L4, L6) in `data/labels/` from the curated workbooks in `data/consistency-check-datasets/` |
| `splitter.py` | Stratified 60/20/20 train/dev/test splits (seed 42) with split audits in `data/labels/<level>/in-process/` |
| `sample_screening.py`, `sample_coding.py` | Draw the samples for the manual title-and-abstract annotations |
| `search_recall.py` | Recall of the a priori test list against the retrieved records |

```bash
python -m data_helpers.datasets.build_screening
python -m data_helpers.datasets.build_coding
python -m data_helpers
python -m data_helpers.datasets.search_recall --start-year 2000
```

## `sources/`: external reference data

| Module | Purpose |
| --- | --- |
| `build_gbif_extract.py` | Builds and verifies the GBIF Backbone extract from the public release (see [data/README.md](../data/README.md)) |
| `build_gbif.py` | Builds the taxon-name lookup used by the taxa evaluation |
| `build_worldbank_data.py` | Builds the World Bank snapshot and IPBES–World Bank crosswalk in `data/world-bank/` |

## `prep/`: analysis datasets

| Module | Purpose |
| --- | --- |
| `corpus.py` | Joins eligible screening records to the coding outputs (one row per publication) |
| `evidence_corpus_prep.py` | Screening summary for all screened records and the integrated dataset of eligible records |
| `taxa_analysis_prep.py` | Taxon matches, broad-group assignment and the GBIF described-species benchmark |
| `_provenance.py` | Repository-relative source signatures for manifests |

These produce the prepared data in `notebooks/data_processing/outputs/`, which is the only input of the results notebooks.

## `analysis/`: results

| Module | Used by | Purpose |
| --- | --- | --- |
| `time_development.py` | `01_evidence_growth` | Annual counts and threat composition |
| `driver_composition.py` | `02_income_composition` | Income-group threat composition and bootstrap contrasts |
| `realm/driver.py`, `realm/driver_plotting.py` | `03_realm_composition` | Direct drivers by ecosystem realm and their trends |
| `taxa/skew.py`, `taxa/skew_plotting.py` | `04_taxonomic_lens` | Research attention versus described species, and its trend |
| `taxa/threat_gap_starnes.py`, `taxa/threat_gap.py`, `taxa/threat_gap_plotting.py` | `04_taxonomic_lens` | Regional chordate evidence versus threatened species, and its trend |
| `taxa/hierarchy.py`, `taxa/hierarchy_plotting.py` | `04_taxonomic_lens` | Kingdom-to-order hierarchy and sunburst figure |
| `taxa/geography.py`, `taxa/benchmark.py`, `taxa/clipart.py` | `04_taxonomic_lens` | Country assignments, benchmark groups, PhyloPic silhouettes |
| `geo/attention.py`, `geo/attention_plotting.py` | `05_geography_attention` | Fractional country and region attention |
| `geo/geography.py`, `geo/count_plotting.py` | `02_income_composition` | IPBES crosswalk, polygons and country map |
| `embedding_map.py` | `06_embedding_map` | Semantic maps from the precomputed UMAP coordinates |

## Tests

```bash
python -m pytest data_helpers
```

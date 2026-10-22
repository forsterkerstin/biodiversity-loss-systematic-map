# biodiversity-loss-systematic-map

Code and data for *Uneven research attention to biodiversity loss revealed by large language models* (Mohiuddin, Forster & Feuerriegel). An LLM pipeline screened 2.4 million Web of Science records for evidence of biodiversity change and coded the 605,199 eligible records, 253,081 of which report biodiversity loss, by geography, ecosystems, taxonomy, direct drivers, and threats. The coded map is available on OSF at [osf.io/xg8yq](https://osf.io/xg8yq/overview); the methods follow the [registered protocol](https://doi.org/10.57808/proceed.2026.31).

## Contents

```text
labelling/               LLM screening and coding pipeline
evals_local/             agreement metrics against reference labels
data_helpers/            dataset builders, corpus preparation, analysis and plotting
notebooks/results/       analyses behind the reported tables and figures
notebooks/consistency_checks/   manual-label checks of screening and ecosystem coding
notebooks/data_processing/      executed notebooks that prepared the analysis data
checklists/              prompts, mappings and configuration
data/                    reference labels, evaluation runs, external reference data (see data/README.md)
```

## Installation

Requires Python 3.12 and [Git LFS](https://git-lfs.com) (data, about 1.3 GB).

```bash
git clone https://github.com/forsterkerstin/biodiversity-loss-systematic-map.git
cd biodiversity-loss-systematic-map
git lfs pull
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Reproducing the results

Build the GBIF Backbone extract once (used by the taxonomic analyses):

```bash
curl -L -C - --create-dirs -o data/gbif/raw/backbone-2023-08-28.zip \
  https://hosted-datasets.gbif.org/datasets/backbone/2023-08-28/backbone.zip
python -m data_helpers.sources.build_gbif_extract
```

If the download stops early, rerun the `curl` command; it resumes where it left off.

Then run the results notebooks in order, interactively in Jupyter or headless:

```bash
jupyter execute --inplace notebooks/results/*.ipynb
```

| Notebook (`notebooks/results/`) | Content |
| --- | --- |
| `00_screening` | Screening outcomes and exclusion reasons |
| `01_evidence_growth` | Growth and threat composition over time |
| `02_income_composition` | Threat composition by World Bank income group |
| `03_realm_composition` | Direct drivers across terrestrial, freshwater and marine systems |
| `04_taxonomic_lens` | Research attention across taxa and regions versus described and threatened species |
| `05_geography_attention` | Research attention by country and region |
| `06_embedding_map` | Semantic maps of the evidence by realm, direct driver and taxonomic class |

Tables, figures and a manifest are written to `notebooks/results/outputs/<notebook>/`.

## Validation

The model responses of all evaluation runs are included, so the validation runs offline:

```bash
python -m evals_local l1_test_010726_low_f1 --tasks driver       # one evaluation run
python -m evals_local.summary                                    # coding agreement, LLM vs published reference labels
python -m evals_local.matched_agreement                          # coding agreement, LLM vs author annotations
python -m data_helpers.datasets.search_recall --start-year 2000  # search recall against the test list
```

The checks against the authors' manual labels (screening of a search-results sample; biome and ecosystem functional groups) are in `notebooks/consistency_checks/`. See [evals_local/README.md](evals_local/README.md).

## Labelling new records

Screening and coding new records requires an OpenAI API key (`cp .env.sample .env`):

```bash
python labelling/orchestrator.py data/labels/l1/train my_run --task driver
```

All runs use `gpt-5-nano-2025-08-07` with the prompts in `checklists/prompts/`. See [labelling/README.md](labelling/README.md) and [data_helpers/README.md](data_helpers/README.md).

## Tests

```bash
python -m pytest
```

## Licence

Code: [MIT](LICENSE). Data sources and licences: [data/README.md](data/README.md).

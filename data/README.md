# Data

Large files are stored with Git LFS; run `git lfs pull` after cloning. Paths are relative to the repository root.

## Contents

| Path | Content | Source and licence |
| --- | --- | --- |
| `data/labels/l0` … `l6` | Reference labels for screening (L0) and the coding tasks (L1–L4, L6), with train/dev/test splits | Compiled from the published evidence syntheses in `data/consistency-check-datasets/` |
| `data/labels/eval/` | Matched predictions and per-label metrics of every evaluation run, the LLM-versus-author coding agreement (`llm_human_matched/`), the agreement with the published reference labels (`summary/`) and the search-recall tables (`search_recall/`) | Derived |
| `data/artifacts/` | Inputs, requests and model responses of every evaluation run | `gpt-5-nano-2025-08-07` |
| `data/consistency-check-datasets/` | Source review workbooks, the a priori test list with the records retrieved by the search check, and the manual title-and-abstract annotations | Compiled from the cited reviews; annotations by the authors |
| `data/world-bank/snapshots/wdi-2026-06-30/` | World Bank income classifications and indicators and the IPBES–World Bank crosswalk, with checksums | World Bank, CC BY 4.0 |
| `data/ipbes-polygons/` | IPBES regions and sub-regions v1.1 | IPBES, [doi:10.5281/zenodo.3928281](https://doi.org/10.5281/zenodo.3928281) |
| `data/ipbes-species-data/` | IUCN Red List species by IPBES sub-region | Starnes et al. (2026), [doi:10.5281/zenodo.18348560](https://doi.org/10.5281/zenodo.18348560) |
| `data/taxa-clipart/` | Taxon silhouettes for the figures | [PhyloPic](https://www.phylopic.org/), CC0 / Public Domain Mark |
| `notebooks/data_processing/outputs/` | Prepared analysis data: the integrated dataset of the 605,199 eligible records (column schema in `04_dataset/manifest.json`), screening results for all 2,414,191 screened records, taxon matches, and the UMAP coordinates of the title-and-abstract embeddings used for the semantic maps (`04_dataset/umap_coordinates.parquet`, described in `04_dataset/umap_manifest.json`) | Derived |
| `notebooks/results/outputs/` | Result tables, figures and manifests | Derived |
| `checklists/mappings/gbif_lookup_cache.pkl` | Taxon-name lookup used by the taxa evaluation | Derived from the GBIF Backbone |

Web of Science Core Collection content (Clarivate), such as titles, abstracts, author, source and citation information, is not redistributed. Records are identified by their Web of Science accession number (UT) and, where available, DOI; users with Web of Science access can retrieve the text by UT. In saved model requests (`data/artifacts/batches/*/request/`), the article text is replaced by a placeholder. Rebuilding the reference-label files from the source workbooks (`data_helpers.datasets.build_screening`/`build_coding`) and labelling new records require this text; all analyses and evaluations run without it.

## GBIF Backbone

The taxonomic analyses use the GBIF Backbone Taxonomy, release 2023-08-28 (GBIF Secretariat, [doi:10.15468/39omei](https://doi.org/10.15468/39omei), CC BY 4.0). It is not stored in the repository; download it and build the extract with

```bash
curl -L -C - --create-dirs -o data/gbif/raw/backbone-2023-08-28.zip \
  https://hosted-datasets.gbif.org/datasets/backbone/2023-08-28/backbone.zip
python -m data_helpers.sources.build_gbif_extract
```

If the download stops early, rerun the `curl` command; it resumes where it left off.

The build verifies the archive (SHA-256 `2897887cc7fb6fe27188c5b0fa5ce2b14762b43c61ceecd1e3dfad25083f5770`) and the extract (`62616875cf44666bd8a53a76c873b7df29a8a2270617a56678a85542d8ee0e4d`, 7,746,724 rows). `python -m data_helpers.sources.build_gbif` rebuilds `gbif_lookup_cache.pkl` from it.

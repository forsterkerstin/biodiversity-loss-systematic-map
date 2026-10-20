"""Tests for shared broad-group rules and GBIF described-diversity benchmarking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_helpers.analysis.taxa import benchmark as td

MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "checklists"
    / "mappings"
    / "taxa_broad_groups.json"
)


@pytest.fixture(scope="module")
def mapping() -> dict:
    return td.load_broad_group_mapping(MAPPING_PATH)


def test_gbif_benchmark_counts_only_accepted_species(
    tmp_path: Path, mapping: dict
) -> None:
    gbif_path = tmp_path / "gbif.csv"
    pd.DataFrame(
        [
            ("1", "species", "accepted", "Animalia", "Chordata", "Mammalia"),
            ("2", "species", "accepted", "Animalia", "Arthropoda", "Insecta"),
            (
                "3",
                "species",
                "accepted",
                "Plantae",
                "Tracheophyta",
                "Magnoliopsida",
            ),
            ("4", "species", "accepted", "Plantae", "Bryophyta", "Bryopsida"),
            ("5", "species", "accepted", "Fungi", "Ascomycota", "Sordariomycetes"),
            (
                "6",
                "species",
                "accepted",
                "Bacteria",
                "Proteobacteria",
                "Gammaproteobacteria",
            ),
            ("7", "species", "synonym", "Animalia", "Chordata", "Mammalia"),
            ("8", "genus", "accepted", "Animalia", "Chordata", "Mammalia"),
        ],
        columns=[
            "taxonID",
            "taxonRank",
            "taxonomicStatus",
            "kingdom",
            "phylum",
            "class",
        ],
    ).to_csv(gbif_path, index=False)
    benchmark = td.build_described_diversity_benchmark(
        gbif_path, mapping, block_size=1024
    )

    assert benchmark.counts["described_species_count"].sum() == 6
    observed = benchmark.counts.set_index("broad_group")[
        "described_species_count"
    ].to_dict()
    assert observed == {
        "Chordates": 1,
        "Non-chordate animals": 1,
        "Plants": 2,
        "Fungi": 1,
        "Other": 1,
        "Unresolved": 0,
    }
    accepted = benchmark.audit.set_index("metric").at[
        "Accepted species-rank rows", "value"
    ]
    assert accepted == 6

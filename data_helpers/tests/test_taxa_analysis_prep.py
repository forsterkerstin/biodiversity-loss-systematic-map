"""Contract tests for the canonical taxa analysis-preparation bundle."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from data_helpers.analysis.taxa import skew as taxa_skew
from data_helpers.analysis.taxa.benchmark import DescribedDiversity
from data_helpers.prep import taxa_analysis_prep

ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = ROOT / "checklists" / "mappings" / "taxa_broad_groups.json"
LAND = "Land/sea use change"
CLIMATE = "Climate Change"
THREAT = "Biological Resource Use"


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UT": ["A", "B", "C", "D"],
            "publication_year": [2000, 2001, 2002, 2003],
            "s2_dir": ["negative"] * 4,
            "driver": [
                json.dumps([LAND, CLIMATE]),
                json.dumps([CLIMATE]),
                json.dumps([LAND]),
                "[]",
            ],
            "pred_threat_l0": [json.dumps([THREAT])] * 4,
            "realm": ['["Terrestrial"]'] * 4,
            "pred_study_design": ["Observational"] * 4,
        }
    )


def _write_taxa(path: Path) -> None:
    def status(name: str, state: str, reason: str) -> str:
        return json.dumps(
            [
                {
                    "canonical_name": name,
                    "status": state,
                    "group_assignments": {"broad": {"reason": reason}},
                }
            ]
        )

    frame = pd.DataFrame(
        {
            "UT": ["A", "B", "C", "D"],
            "class": [
                '["Mammalia", "Insecta"]',
                '["Magnoliopsida"]',
                "[]",
                "[]",
            ],
            "broad_taxa_groups": [
                '["Chordates", "Non-chordate animals"]',
                '["Plants"]',
                '["Unresolved"]',
                "[]",
            ],
            "taxa_analysis_groups": [
                '["Vertebrates", "Arthropods"]',
                '["Vascular plants"]',
                '["Unresolved"]',
                "[]",
            ],
            "taxa_detail_groups": [
                '["Mammals", "Arthropods"]',
                '["Vascular plants"]',
                '["Unresolved"]',
                "[]",
            ],
            "llm_taxa_json": [
                '[{"canonical_name": "Example animal"}]',
                '[{"canonical_name": "Example plant"}]',
                '[{"canonical_name": "Example unresolved"}]',
                '[{"canonical_name": "Not applicable"}]',
            ],
            "taxa_match_status_json": [
                status("Example animal", "exact", "vertebrata"),
                status("Example plant", "fuzzy_accepted", "plantae"),
                status("Example unresolved", "exact", "hierarchy_missing"),
                status("Not applicable", "special_value", "special_value"),
            ],
            "taxa_record_status": [
                "resolved",
                "resolved",
                "unresolved",
                "not_applicable",
            ],
            "n_llm_taxa": [1] * 4,
            "n_taxa_matched": [1, 1, 1, 0],
            "n_taxa_unresolved": [0, 0, 1, 0],
            "n_taxa_api_failed": [0] * 4,
        }
    )
    rank_values = {
        "domain": ['["Eukaryota"]'] * 4,
        "kingdom": ['["Animalia"]', '["Plantae"]', "[]", "[]"],
        "subkingdom": ["[]"] * 4,
        "phylum": ['["Chordata", "Arthropoda"]', '["Tracheophyta"]', "[]", "[]"],
        "order": ['["Primates", "Coleoptera"]', "[]", "[]", "[]"],
        "family": ["[]"] * 4,
        "genus": ["[]"] * 4,
        "species": ["[]"] * 4,
    }
    for column, values in rank_values.items():
        frame[column] = values
    frame.to_csv(path, index=False)


def _write_lineage(path: Path) -> None:
    statuses = ["exact", "fuzzy_accepted", "exact", "special_value"]
    rows = []
    for index, (ut, status) in enumerate(zip(["A", "B", "C", "D"], statuses)):
        resolved = status != "special_value"
        rows.append(
            {
                "UT": ut,
                "llm_taxon_index": 0,
                "llm_canonical_name": f"Taxon {ut}",
                "llm_taxon_rank": "species" if resolved else None,
                "match_status": status,
                "matched_taxon_key": str(index + 1) if resolved else None,
                "matched_name": f"Matched {ut}" if resolved else None,
                "matched_rank": "species" if resolved else None,
                "taxonomic_status": "accepted" if resolved else None,
                "match_type": "exact" if resolved else None,
                "confidence": 1.0 if resolved else None,
                "synonym": False if resolved else None,
                "broad_group": None,
                "broad_group_rule_id": None,
                "broad_group_reason": None,
                "broad_group_eligible": resolved,
                "analysis_group": None,
                "analysis_group_rule_id": None,
                "analysis_group_reason": None,
                "detail_group": None,
                "detail_group_rule_id": None,
                "detail_group_reason": None,
                "error_message": None,
                "lineage_position": 0 if resolved else None,
                "lineage_key": str(index + 1) if resolved else None,
                "lineage_rank": "species" if resolved else None,
                "lineage_name": f"Matched {ut}" if resolved else None,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture()
def prepared(tmp_path: Path):
    mapping = taxa_analysis_prep.load_taxa_mapping(MAPPING_PATH)
    taxa_path = tmp_path / "taxa.csv"
    _write_taxa(taxa_path)
    articles, audits = taxa_analysis_prep.build_taxa_publications(
        _corpus(),
        taxa_path=taxa_path,
        mapping=mapping,
        driver_order=(LAND, CLIMATE),
        threat_order=(THREAT,),
    )
    return mapping, taxa_path, articles, audits


def test_preparation_preserves_grain_and_states(prepared) -> None:
    mapping, _, articles, audits = prepared
    assert articles["UT"].tolist() == ["A", "B", "C", "D"]
    assert articles["UT"].is_unique
    assert articles.loc[0, "drivers"] == (LAND, CLIMATE)
    assert articles.loc[0, "analysis_groups"] == ("Vertebrates", "Arthropods")
    assert articles.loc[0, "class_labels"] == ("Mammalia", "Insecta")
    assert articles.loc[0, "taxa_class_labels"] == ("Mammalia", "Insecta")
    assert articles.loc[0, "taxa_phylum_labels"] == ("Chordata", "Arthropoda")
    assert "taxa_domain_labels" not in articles
    assert "taxa_subkingdom_labels" not in articles
    assert articles.loc[2, "taxa_analysis_state"] == "unresolved_only"
    assert articles["taxa_broad_inclusion_state"].tolist() == [
        taxa_analysis_prep.INCLUSION_ORDER[0],
        taxa_analysis_prep.INCLUSION_ORDER[0],
        taxa_analysis_prep.INCLUSION_ORDER[4],
        taxa_analysis_prep.INCLUSION_ORDER[1],
    ]
    assert articles["match_status_count__exact"].tolist() == [1, 0, 1, 0]
    assert sum(row["taxon_item_count"] for row in audits["match_status"]) == 4
    assert mapping["schema_version"] == 3
    assert mapping["sunburst_clipart"]["Animalia"] == "animal_cat"
    assert set(mapping["sunburst_clipart"].values()).issubset(
        mapping["clipart_assets"]
    )
    assert mapping["sunburst_order_clipart"]["Rodentia"] == "mammal_mouse"
    assert set(mapping["sunburst_order_clipart"].values()).issubset(
        mapping["clipart_assets"]
    )


def test_store_round_trip_is_four_files_and_rejects_stale_rules(
    tmp_path: Path, prepared
) -> None:
    mapping, taxa_path, articles, audits = prepared
    benchmark = DescribedDiversity(
        counts=pd.DataFrame(
                {
                    "broad_group": ["Chordates", "Non-chordate animals", "Plants"],
                    "described_species_count": [10, 80, 10],
                    "described_species_share": [0.1, 0.8, 0.1],
                    "described_species_share_pct": [10.0, 80.0, 10.0],
                }
        ),
        audit=pd.DataFrame([{"stage": "accepted species", "records": 100}]),
    )
    store = taxa_analysis_prep.TaxaPreparedStore(tmp_path / "bundle")
    lineage_path = tmp_path / "lineage.parquet"
    _write_lineage(lineage_path)
    match_summary = store.write_matches(
        lineage_path, expected_uts=articles["UT"]
    )
    manifest = taxa_analysis_prep.build_manifest(
        articles,
        benchmark,
        mapping=mapping,
        audits=audits,
        match_summary=match_summary,
        sources={"taxa": taxa_path, "lineage": lineage_path},
        repository_root=tmp_path,
    )
    paths = store.write(
        taxa_analysis_prep.TaxaPreparedBundle(articles, benchmark, manifest)
    )
    assert {path.name for path in paths} == {
        "taxa_publications.parquet",
        "taxa_matches.parquet",
        "gbif_broad_benchmark.csv",
        "manifest.json",
    }
    assert {path.name for path in store.root.iterdir()} == {path.name for path in paths}
    assert manifest["sources"]["taxa"]["path"] == "taxa.csv"
    assert manifest["sources"]["lineage"]["path"] == "lineage.parquet"

    loaded = store.load(mapping=mapping)
    assert loaded.articles.loc[0, "broad_groups"] == (
        "Chordates",
        "Non-chordate animals",
    )
    assert loaded.benchmark.counts["described_species_count"].sum() == 100
    assert loaded.manifest["taxa_matches_audit"]["taxon_items"] == 4

    changed = copy.deepcopy(mapping)
    changed["schemes"]["broad"]["rules"][0]["group"] = "Other"
    with pytest.raises(taxa_analysis_prep.TaxaAnalysisPrepError, match="stale"):
        store.load(mapping=changed)


def _match_item(index: int, name: str, status: str, lineage: list[tuple[str, str]]):
    item = {field.name: None for field in taxa_analysis_prep.TAXA_MATCH_STRUCT_TYPE}
    item.update(
        llm_taxon_index=index,
        llm_canonical_name=name,
        llm_taxon_rank="species" if status != "special_value" else "Not applicable",
        match_status=status,
        lineage=[
            {"position": position, "key": str(position), "rank": rank, "name": value}
            for position, (rank, value) in enumerate(lineage)
        ],
    )
    return item


def _match_rows() -> list[dict]:
    mammal = [
        ("KINGDOM", "Animalia"),
        ("PHYLUM", "Chordata"),
        ("SUBPHYLUM", "Vertebrata"),
        ("CLASS", "Mammalia"),
    ]
    insect = [("KINGDOM", "Animalia"), ("PHYLUM", "Arthropoda"), ("CLASS", "Insecta")]
    plant = [("KINGDOM", "Plantae"), ("PHYLUM", "Tracheophyta"), ("CLASS", "Magnoliopsida")]
    return [
        {"UT": "A", "taxa_matches": [
            _match_item(0, "Example animal", "exact", mammal),
            _match_item(1, "Example insect", "exact", insect),
        ]},
        {"UT": "B", "taxa_matches": [_match_item(0, "Example plant", "fuzzy_accepted", plant)]},
        {"UT": "C", "taxa_matches": [_match_item(0, "Example unresolved", "exact", [])]},
        {"UT": "D", "taxa_matches": [_match_item(0, "Not applicable", "special_value", [])]},
    ]


def test_regrouping_from_matches_recomputes_group_columns(
    tmp_path: Path, prepared
) -> None:
    _, _, articles, _ = prepared
    # Scramble the group columns so the comparison proves they are recomputed.
    scrambled = articles.copy()
    for scheme in taxa_analysis_prep.GROUP_SCHEMES:
        scrambled[f"{scheme}_groups_all"] = [()] * len(scrambled)
        scrambled[f"{scheme}_groups"] = [()] * len(scrambled)
    scrambled["taxa_broad_inclusion_state"] = "stale"

    regrouped, rows, audits, summary = taxa_analysis_prep.regroup_prepared_taxa(
        scrambled, _match_rows(), mapping_path=MAPPING_PATH
    )
    pd.testing.assert_frame_equal(regrouped, articles)
    first = rows[0]["taxa_matches"]
    assert [item["broad_group"] for item in first] == ["Chordates", "Non-chordate animals"]
    assert [item["detail_group"] for item in first] == ["Mammals", "Arthropods"]
    assert rows[3]["taxa_matches"][0]["broad_group_reason"] == "not_applicable"
    assert rows[3]["taxa_matches"][0]["broad_group_eligible"] is False
    assert {row["match_status"]: row["taxon_item_count"] for row in audits["match_status"]} == {
        "exact": 3, "fuzzy_accepted": 1, "special_value": 1,
    }
    assert audits["special_value"][0]["special_value"] == "Not applicable"
    assert summary == {
        "publications": 4,
        "unique_UT": 4,
        "taxon_items": 5,
        "source_lineage_rows": 4 + 3 + 3 + 1 + 1,
        "resolved_lineage_rows": 10,
        "match_status_counts": {"exact": 3, "fuzzy_accepted": 1, "special_value": 1},
    }

    path = tmp_path / "taxa_matches.parquet"
    taxa_analysis_prep.write_taxa_match_rows(rows, path)
    assert taxa_analysis_prep.read_taxa_match_rows(path) == rows

    # Streaming from and to Parquet in small batches gives the same result.
    source = tmp_path / "source_matches.parquet"
    taxa_analysis_prep.write_taxa_match_rows(_match_rows(), source)
    streamed_path = tmp_path / "streamed.parquet"
    streamed, returned_rows, streamed_audits, streamed_summary = (
        taxa_analysis_prep.regroup_prepared_taxa(
            scrambled,
            source,
            mapping_path=MAPPING_PATH,
            output_path=streamed_path,
            batch_size=2,
        )
    )
    assert returned_rows is None
    pd.testing.assert_frame_equal(streamed, regrouped)
    assert taxa_analysis_prep.read_taxa_match_rows(streamed_path) == rows
    assert streamed_audits == audits
    assert streamed_summary == summary
    assert not list(tmp_path.glob(".*regrouped*"))


def test_regrouping_follows_renamed_rules_and_rejects_misordered_matches(
    tmp_path: Path, prepared
) -> None:
    _, _, articles, _ = prepared
    renamed = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    renamed["schemes"]["broad"]["group_order"] = [
        "Animals with notochord" if group == "Chordates" else group
        for group in renamed["schemes"]["broad"]["group_order"]
    ]
    for rule in renamed["schemes"]["broad"]["rules"]:
        if rule["group"] == "Chordates":
            rule["group"] = "Animals with notochord"
    renamed["schemes"]["broad"]["group_colors"]["Animals with notochord"] = (
        renamed["schemes"]["broad"]["group_colors"].pop("Chordates")
    )
    renamed["schemes"]["broad"]["group_clipart"]["Animals with notochord"] = (
        renamed["schemes"]["broad"]["group_clipart"].pop("Chordates")
    )
    renamed_path = tmp_path / "renamed.json"
    renamed_path.write_text(json.dumps(renamed), encoding="utf-8")

    regrouped, *_ = taxa_analysis_prep.regroup_prepared_taxa(
        articles, _match_rows(), mapping_path=renamed_path
    )
    assert regrouped.loc[0, "broad_groups"] == (
        "Animals with notochord",
        "Non-chordate animals",
    )
    with pytest.raises(taxa_analysis_prep.TaxaAnalysisPrepError, match="UT order"):
        taxa_analysis_prep.regroup_prepared_taxa(
            articles, list(reversed(_match_rows())), mapping_path=MAPPING_PATH
        )


def test_rebuild_with_rules_regroups_bundle_in_place(tmp_path: Path, prepared) -> None:
    mapping, taxa_path, articles, audits = prepared
    store = taxa_analysis_prep.TaxaPreparedStore(tmp_path / "bundle")
    taxa_analysis_prep.write_taxa_match_rows(_match_rows(), store.match_path)
    old_benchmark = DescribedDiversity(
        counts=pd.DataFrame(
            {
                "broad_group": ["Chordates"],
                "described_species_count": [1],
                "described_species_share": [1.0],
                "described_species_share_pct": [100.0],
            }
        ),
        audit=pd.DataFrame([{"metric": "old", "value": 1}]),
    )
    manifest = taxa_analysis_prep.build_manifest(
        articles,
        old_benchmark,
        mapping=mapping,
        audits=audits,
        match_summary={"publications": 4, "taxon_items": 5},
        sources={"taxa": taxa_path},
        repository_root=tmp_path,
    )
    manifest["grouping_rules_sha256"] = "previous-rules"
    store.write(taxa_analysis_prep.TaxaPreparedBundle(articles, old_benchmark, manifest))
    with pytest.raises(taxa_analysis_prep.TaxaAnalysisPrepError, match="stale"):
        store.load(mapping=mapping)

    gbif = tmp_path / "gbif.csv"
    pd.DataFrame(
        {
            "taxonID": ["1", "2", "3", "4"],
            "taxonRank": ["species", "species", "species", "genus"],
            "taxonomicStatus": ["accepted", "accepted", "synonym", "accepted"],
            "kingdom": ["Animalia", "Plantae", "Plantae", "Animalia"],
            "phylum": ["Chordata", "Tracheophyta", "Tracheophyta", "Chordata"],
            "class": ["Mammalia", "Magnoliopsida", "Magnoliopsida", "Aves"],
        }
    ).to_csv(gbif, index=False)
    eml = tmp_path / "eml.xml"
    eml.write_text("<eml/>", encoding="utf-8")
    mapping_copy = tmp_path / "taxa_broad_groups.json"
    mapping_copy.write_bytes(MAPPING_PATH.read_bytes())

    written = store.rebuild_with_rules(
        mapping_path=mapping_copy,
        gbif_path=gbif,
        gbif_eml_path=eml,
        repository_root=tmp_path,
    )
    assert {path.name for path in written} == {
        "taxa_publications.parquet",
        "taxa_matches.parquet",
        "gbif_broad_benchmark.csv",
        "manifest.json",
    }
    loaded = store.load(mapping=mapping)
    assert loaded.articles.loc[0, "broad_groups"] == ("Chordates", "Non-chordate animals")
    counts = loaded.benchmark.counts.set_index("broad_group")["described_species_count"]
    assert counts["Chordates"] == 1 and counts["Plants"] == 1
    assert loaded.manifest["upstream_sources"] == {"taxa": {"path": "taxa.csv", **{
        key: value for key, value in manifest["sources"]["taxa"].items() if key != "path"
    }}}
    regrouped_from = loaded.manifest["regrouped_from"]
    assert regrouped_from["previous_grouping_rules_sha256"] == "previous-rules"
    assert set(regrouped_from["previous_artifacts"]) == {
        "taxa_publications.parquet",
        "taxa_matches.parquet",
        "manifest.json",
    }
    assert loaded.manifest["sources"]["gbif"]["path"] == "gbif.csv"
    assert loaded.manifest["audits"]["source_grain"] == audits["source_grain"]
    assert not list(store.root.glob(".*"))


def test_vector_one_adapter_preserves_article_balanced_attribution(prepared) -> None:
    _, _, articles, audits = prepared
    evidence = taxa_skew.prepare_taxa_skew_from_prepared(
        articles,
        benchmark_groups=(
            "Chordates",
            "Non-chordate animals",
            "Plants",
            "Fungi",
            "Other",
        ),
        unresolved_group="Unresolved",
        match_status_audit=pd.DataFrame(audits["match_status"]),
        group_reason_audit=pd.DataFrame(audits["broad_group_reason"]),
        special_value_audit=pd.DataFrame(audits["special_value"]),
        auxiliary_label_audit=pd.DataFrame(audits["auxiliary_labels"]),
    )
    assert evidence.included["UT"].tolist() == ["A", "B"]
    assert evidence.attributions.groupby("UT")[
        "fractional_publication_weight"
    ].sum().eq(1).all()
    assert evidence.inclusion_audit["n_publications"].sum() == 4

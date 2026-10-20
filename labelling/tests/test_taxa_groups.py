from __future__ import annotations

import json
from pathlib import Path

from labelling.taxa_groups import (
    assign_taxon_group,
    assign_taxon_groups,
    load_group_config,
    order_groups,
)

MAPPING_PATH = (
    Path(__file__).resolve().parents[2]
    / "checklists"
    / "mappings"
    / "taxa_broad_groups.json"
)


def _classification(**ranks: str) -> list[dict[str, str]]:
    return [
        {"rank": rank.removesuffix("_").upper(), "name": name}
        for rank, name in ranks.items()
    ]


def test_config_assigns_supported_hierarchies_per_taxon_item() -> None:
    config = load_group_config(MAPPING_PATH)
    cases = [
        (
            _classification(
                kingdom="Animalia",
                phylum="Chordata",
                subphylum="Vertebrata",
            ),
            "Chordates",
        ),
        (
            _classification(kingdom="Animalia", phylum="Arthropoda"),
            "Non-chordate animals",
        ),
        (
            _classification(
                kingdom="Animalia",
                phylum="Chordata",
                subphylum="Tunicata",
            ),
            "Chordates",
        ),
        (_classification(kingdom="Plantae"), "Plants"),
        (_classification(kingdom="Fungi"), "Fungi"),
        (_classification(kingdom="Chromista"), "Other"),
        (_classification(kingdom="Animalia"), "Unresolved"),
    ]

    for classification, expected in cases:
        assigned = assign_taxon_group(
            raw_rank="species",
            raw_name="Example taxon",
            match_status="exact",
            classification=classification,
            config=config,
        )
        assert assigned.group == expected
        assert assigned.eligible


def test_clipart_assets_cover_every_group_without_changing_grouping_hash(
    tmp_path: Path,
) -> None:
    config = load_group_config(MAPPING_PATH)
    referenced: set[str] = set()

    for scheme_id in config["scheme_order"]:
        scheme = config["schemes"][scheme_id]
        assert set(scheme["group_clipart"]) == set(scheme["group_order"])
        referenced.update(scheme["group_clipart"].values())
    referenced.update(config["sunburst_clipart"].values())
    referenced.update(config["sunburst_order_clipart"].values())
    referenced.update(config["retired_clipart_assets"])

    assert referenced == set(config["clipart_assets"])
    assert all(
        asset["svg_url"].startswith("https://")
        and asset["page_url"].startswith("https://")
        for asset in config["clipart_assets"].values()
    )

    changed = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    changed["clipart_assets"]["fish_carp"]["attribution"] = "Visual-only change"
    changed["sunburst_clipart"]["Chordata"] = "animal_cat"
    changed["sunburst_order_clipart"]["Rodentia"] = "primate_macaque"
    changed["sunburst_order_clipart"]["Primates"] = "mammal_mouse"
    changed_path = tmp_path / "taxa_broad_groups.json"
    changed_path.write_text(
        json.dumps(changed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    changed_config = load_group_config(changed_path)

    assert changed_config["_config_sha256"] != config["_config_sha256"]
    assert changed_config["_grouping_sha256"] == config["_grouping_sha256"]


def test_all_three_schemes_assign_meaningful_nested_labels() -> None:
    config = load_group_config(MAPPING_PATH)
    cases = [
        (
            _classification(
                kingdom="Animalia",
                phylum="Chordata",
                subphylum="Vertebrata",
                class_="Mammalia",
            ),
            ("Chordates", "Vertebrates", "Mammals"),
        ),
        (
            _classification(
                kingdom="Animalia",
                phylum="Chordata",
                subphylum="Vertebrata",
                class_="Teleostei",
            ),
            ("Chordates", "Vertebrates", "Fishes"),
        ),
        (
            _classification(kingdom="Animalia", phylum="Mollusca"),
            ("Non-chordate animals", "Other invertebrates", "Molluscs"),
        ),
        (
            _classification(
                kingdom="Animalia",
                phylum="Chordata",
                subphylum="Tunicata",
                class_="Ascidiacea",
            ),
            ("Chordates", "Other invertebrates", "Other invertebrates"),
        ),
        (
            _classification(kingdom="Plantae", phylum="Tracheophyta"),
            ("Plants", "Vascular plants", "Vascular plants"),
        ),
        (
            _classification(kingdom="Plantae"),
            (
                "Plants",
                "Other or unspecified plants",
                "Plants—phylum unspecified",
            ),
        ),
        (
            _classification(
                domain="Bacteria",
                kingdom="Pseudomonadati",
                phylum="Pseudomonadota",
            ),
            ("Other", "Bacteria & archaea", "Bacteria & archaea"),
        ),
        (
            _classification(domain="Bacteria"),
            ("Other", "Bacteria & archaea", "Bacteria & archaea"),
        ),
    ]

    for classification, expected in cases:
        assignments = assign_taxon_groups(
            raw_rank="species",
            raw_name="Example taxon",
            match_status="exact",
            classification=classification,
            config=config,
        )
        assert (
            tuple(assignments[scheme].group for scheme in config["scheme_order"])
            == expected
        )


def test_only_exact_or_accepted_fuzzy_items_are_grouped() -> None:
    config = load_group_config(MAPPING_PATH)
    plant = _classification(kingdom="Plantae")

    fuzzy = assign_taxon_group(
        raw_rank="species",
        raw_name="Example plant",
        match_status="fuzzy_accepted",
        classification=plant,
        config=config,
    )
    review = assign_taxon_group(
        raw_rank="species",
        raw_name="Example plant",
        match_status="fuzzy_review",
        classification=plant,
        config=config,
    )
    not_applicable = assign_taxon_group(
        raw_rank="Not applicable",
        raw_name="Not applicable",
        match_status="special_value",
        classification=[],
        config=config,
    )

    assert fuzzy.group == "Plants"
    assert fuzzy.eligible
    assert review.group is None
    assert not review.eligible
    assert not_applicable.group is None
    assert not_applicable.reason == "not_applicable"


def test_ut_groups_are_deduplicated_in_configured_order() -> None:
    config = load_group_config(MAPPING_PATH)

    assert order_groups(
        ["Plants", "Chordates", "Plants", "Fungi"],
        config,
    ) == ["Chordates", "Plants", "Fungi"]
    assert order_groups(
        ["Fungi", "Mammals", "Birds", "Mammals"],
        config,
        scheme="detail",
    ) == ["Mammals", "Birds", "Fungi"]

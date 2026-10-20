from __future__ import annotations

from labelling.src.tasks import EcosystemTask


def _task() -> EcosystemTask:
    return EcosystemTask(name="ecosystems", prompt_key="ecosystems")


def test_biome_candidates_use_mapping_descriptions() -> None:
    task = _task()
    mapping = {
        "Marine": {
            "biomes": [
                {
                    "name": "Marine shelves",
                    "desc": "Full marine-shelf description.",
                    "efg": [],
                }
            ]
        }
    }

    candidates, _ = task._build_biome_candidates(["Marine"], mapping)

    assert candidates == [
        {"name": "Marine shelves", "desc": "Full marine-shelf description."}
    ]


def test_efg_candidates_use_mapping_descriptions() -> None:
    task = _task()
    biome_lookup = {
        "Marine shelves": [
            {
                "name": "Seagrass meadows",
                "desc": "Full seagrass-meadow description.",
            }
        ]
    }

    candidates = task._build_efg_candidates(["Marine shelves"], biome_lookup)

    assert candidates == [
        {"name": "Seagrass meadows", "desc": "Full seagrass-meadow description."}
    ]


def test_candidate_short_description_remains_an_optional_override() -> None:
    task = _task()
    mapping = {
        "Marine": {
            "biomes": [
                {
                    "name": "Marine shelves",
                    "short_desc": "Short description.",
                    "desc": "Full description.",
                    "efg": [],
                }
            ]
        }
    }

    candidates, _ = task._build_biome_candidates(["Marine"], mapping)

    assert candidates == [{"name": "Marine shelves", "desc": "Short description."}]

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from labelling.taxa_api import (
    GBIFMatchCache,
    GBIFMatchClient,
    MatchResult,
    SCHEMA_VERSION,
    TaxonQuery,
    _CombinedOutputWriter,
    _csv_view,
    _completion_is_reusable,
    _excel_view,
    _partition_paths,
    _raw_results_from_wide,
    _read_partition_excel,
    build_partition_tables,
    evaluate_match_response,
    load_raw_taxa_results,
    resolve_queries,
)


def _api_payload(
    *,
    name: str = "Aerides multiflorum",
    rank: str = "SPECIES",
    match_type: str = "EXACT",
    confidence: int = 99,
) -> dict:
    return {
        "usage": {
            "key": "L7C82",
            "canonicalName": name,
            "rank": rank,
            "status": "ACCEPTED",
        },
        "classification": [
            {"key": "CS5HF", "name": "Eukaryota", "rank": "DOMAIN"},
            {"key": "P", "name": "Plantae", "rank": "KINGDOM"},
            {"key": "CMQ8S", "name": "Pteridobiotina", "rank": "SUBKINGDOM"},
            {"key": "TP", "name": "Tracheophyta", "rank": "PHYLUM"},
            {"key": "L2L", "name": "Liliopsida", "rank": "CLASS"},
            {"key": "SP", "name": "Asparagales", "rank": "ORDER"},
            {"key": "DPL", "name": "Orchidaceae", "rank": "FAMILY"},
            {"key": "8VTD7", "name": "Aerides", "rank": "GENUS"},
            {"key": "L7C82", "name": name, "rank": rank},
        ],
        "diagnostics": {
            "matchType": match_type,
            "confidence": confidence,
        },
        "synonym": False,
    }


def _batch_record(custom_id: str, results: object) -> dict:
    text = json.dumps({"results": results})
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ]
            }
        },
    }


def test_evaluate_match_response_acceptance_policy() -> None:
    query = TaxonQuery("Aerides multiflorum", "species")

    exact = evaluate_match_response(query, _api_payload(), attempts=1)
    assert exact.status == "exact"
    assert exact.accepted

    fuzzy = evaluate_match_response(
        query,
        _api_payload(match_type="VARIANT", confidence=91),
        attempts=1,
    )
    assert fuzzy.status == "fuzzy_accepted"
    assert fuzzy.accepted

    low_confidence = evaluate_match_response(
        query,
        _api_payload(match_type="VARIANT", confidence=89),
        attempts=1,
    )
    assert low_confidence.status == "fuzzy_review"
    assert not low_confidence.accepted

    wrong_rank = evaluate_match_response(
        query,
        _api_payload(rank="GENUS"),
        attempts=1,
    )
    assert wrong_rank.status == "rank_mismatch"
    assert not wrong_rank.accepted


def test_evaluate_match_response_distinguishes_ambiguous_and_no_match() -> None:
    query = TaxonQuery("Example", "genus")
    ambiguous = evaluate_match_response(
        query,
        {
            "diagnostics": {
                "matchType": "NONE",
                "note": "Multiple equal matches for Example",
            }
        },
        attempts=1,
    )
    no_match = evaluate_match_response(
        query,
        {"diagnostics": {"matchType": "NONE"}},
        attempts=1,
    )
    assert ambiguous.status == "ambiguous"
    assert no_match.status == "no_match"


def test_raw_loader_prefers_later_valid_retry(tmp_path: Path) -> None:
    main = tmp_path / "taxa-output.jsonl"
    retry = tmp_path / "taxa_retry_1-output.jsonl"
    malformed_retry = tmp_path / "taxa_retry_2-output.jsonl"
    main.write_text(
        json.dumps(
            _batch_record(
                "WOS:1",
                [{"taxon_rank": "genus", "canonical_name": "Aerides"}],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    retry.write_text(
        "\n".join(
            [
                "{invalid json",
                json.dumps(
                    _batch_record(
                        "WOS:1",
                        [
                            {
                                "taxon_rank": "species",
                                "canonical_name": "Aerides multiflorum",
                            }
                        ],
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    malformed_retry.write_text(
        json.dumps(
            _batch_record(
                "WOS:1",
                [{"taxon_rank": "species"}],
            )
        )
        + "\n",
        encoding="utf-8",
    )

    selected, audit = load_raw_taxa_results(tmp_path)

    assert selected["WOS:1"] == [
        {
            "taxon_rank": "species",
            "canonical_name": "Aerides multiflorum",
        }
    ]
    assert audit.response_lines == 4
    assert audit.valid_responses == 2
    assert audit.invalid_responses == 2
    assert audit.duplicate_valid_responses == 1


def test_cache_reuses_stable_results_and_retries_api_failures(
    tmp_path: Path,
) -> None:
    cache = GBIFMatchCache(tmp_path / "matches.sqlite")
    query = TaxonQuery("Aerides multiflorum", "species")
    try:
        cache.put(
            taxonomy_build_id="build",
            checklist_key="checklist",
            result=MatchResult(
                query.canonical_name,
                query.requested_rank,
                "exact",
                True,
                1,
                response=_api_payload(),
            ),
        )
        cached = cache.get(
            taxonomy_build_id="build",
            checklist_key="checklist",
            query=query,
            retry_final_failures=True,
        )
        assert cached is not None
        assert cached.accepted

        failed_query = TaxonQuery("Unavailable name", "species")
        cache.put(
            taxonomy_build_id="build",
            checklist_key="checklist",
            result=MatchResult(
                failed_query.canonical_name,
                failed_query.requested_rank,
                "api_failure",
                False,
                4,
                error_message="timeout",
            ),
        )
        assert (
            cache.get(
                taxonomy_build_id="build",
                checklist_key="checklist",
                query=failed_query,
                retry_final_failures=True,
            )
            is None
        )
        assert (
            cache.get(
                taxonomy_build_id="build",
                checklist_key="checklist",
                query=failed_query,
                retry_final_failures=False,
            )
            is not None
        )

        variant_query = TaxonQuery("Kirkaldyia deyrollii", "species")
        cache.put(
            taxonomy_build_id="build",
            checklist_key="checklist",
            result=MatchResult(
                variant_query.canonical_name,
                variant_query.requested_rank,
                "fuzzy_review",
                False,
                1,
                response=_api_payload(
                    name="Kirkaldyia deyrollei",
                    match_type="VARIANT",
                    confidence=96,
                ),
                http_status=200,
            ),
        )
        reevaluated = cache.get(
            taxonomy_build_id="build",
            checklist_key="checklist",
            query=variant_query,
            retry_final_failures=True,
        )
        assert reevaluated is not None
        assert reevaluated.status == "fuzzy_accepted"
        assert reevaluated.accepted
    finally:
        cache.close()


def test_partition_tables_include_full_hierarchy_and_record_statuses() -> None:
    base = pd.DataFrame(
        [
            {"UT": "WOS:1", "publication_year": 2020},
            {"UT": "WOS:2", "publication_year": 2021},
            {"UT": "WOS:3", "publication_year": 2022},
        ]
    )
    raw = {
        "WOS:1": [
            {
                "taxon_rank": "species",
                "canonical_name": "Aerides multiflorum",
            }
        ],
        "WOS:2": [],
    }
    query = TaxonQuery("Aerides multiflorum", "species")
    matches = {
        query.key: MatchResult(
            query.canonical_name,
            query.requested_rank,
            "exact",
            True,
            1,
            response=_api_payload(),
        )
    }

    wide, lineage, audit = build_partition_tables(
        base,
        raw,
        matches,
        partition=1,
        run_name="partition_1",
    )

    first = wide.set_index("UT").loc["WOS:1"]
    assert first["domain"] == ["Eukaryota"]
    assert first["kingdom"] == ["Plantae"]
    assert first["subkingdom"] == ["Pteridobiotina"]
    assert first["family"] == ["Orchidaceae"]
    assert first["species"] == ["Aerides multiflorum"]
    assert first["broad_taxa_groups"] == ["Plants"]
    assert first["n_broad_taxa_groups"] == 1
    assert first["taxa_analysis_groups"] == ["Vascular plants"]
    assert first["n_taxa_analysis_groups"] == 1
    assert first["taxa_detail_groups"] == ["Vascular plants"]
    assert first["n_taxa_detail_groups"] == 1
    assert first["taxa_record_status"] == "resolved"
    assert wide.set_index("UT").loc["WOS:2", "taxa_record_status"] == (
        "no_taxa_reported"
    )
    assert wide.set_index("UT").loc["WOS:3", "taxa_record_status"] == (
        "missing_llm_output"
    )
    assert len(lineage) == 9
    assert audit["accepted_taxon_items"] == 1


def test_completion_marker_requires_matching_inputs_and_outputs(
    tmp_path: Path,
) -> None:
    excel = tmp_path / "wide.xlsx"
    excel.touch()
    marker = tmp_path / "completion.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "source_run": "partition_1",
                "taxonomy_build_id": "build",
                "input_fingerprint": "fingerprint",
                "excel": str(excel),
            }
        ),
        encoding="utf-8",
    )

    reusable, _ = _completion_is_reusable(
        marker,
        run_name="partition_1",
        taxonomy_build_id="build",
        input_fingerprint="fingerprint",
        grouping_config_sha256="groups",
    )
    assert not reusable

    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    marker_payload["schema_version"] = SCHEMA_VERSION
    marker_payload["grouping_config_sha256"] = "groups"
    marker.write_text(json.dumps(marker_payload), encoding="utf-8")
    reusable, _ = _completion_is_reusable(
        marker,
        run_name="partition_1",
        taxonomy_build_id="build",
        input_fingerprint="fingerprint",
        grouping_config_sha256="groups",
    )
    assert reusable

    reusable, _ = _completion_is_reusable(
        marker,
        run_name="partition_1",
        taxonomy_build_id="different-build",
        input_fingerprint="fingerprint",
        grouping_config_sha256="groups",
    )
    assert not reusable

    reusable, _ = _completion_is_reusable(
        marker,
        run_name="partition_1",
        taxonomy_build_id="build",
        input_fingerprint="fingerprint",
        grouping_config_sha256="different-groups",
    )
    assert not reusable


def test_partition_paths_use_taxa_xlsx_only(tmp_path: Path) -> None:
    paths = _partition_paths(
        tmp_path,
        partition=2,
        max_year=2025,
        min_year=2000,
    )

    assert set(paths) == {"directory", "excel", "marker"}
    assert paths["excel"].name == "taxa_02_2025-2000.xlsx"


def test_excel_checkpoint_roundtrip_recovers_lists_and_llm_results(
    tmp_path: Path,
) -> None:
    base = pd.DataFrame(
        [
            {
                "UT": "WOS:1",
                "title": "Title",
                "authors": ["One", "Two"],
                "abstract": "Abstract",
                "source": "Journal",
                "publication_year": 2024,
                "wos_categories": ["Ecology", "Biology"],
                "doi": "10.1/example",
            }
        ]
    )
    raw = {
        "WOS:1": [
            {
                "taxon_rank": "species",
                "canonical_name": "Aerides multiflorum",
            }
        ]
    }
    query = TaxonQuery("Aerides multiflorum", "species")
    matches = {
        query.key: MatchResult(
            query.canonical_name,
            query.requested_rank,
            "exact",
            True,
            1,
            response=_api_payload(),
        )
    }
    wide, _, _ = build_partition_tables(
        base,
        raw,
        matches,
        partition=1,
        run_name="partition_1",
    )
    path = tmp_path / "taxa_01_2024-2024.xlsx"
    _excel_view(wide).to_excel(path, index=False)

    restored = _read_partition_excel(
        path,
        partition=1,
        run_name="partition_1",
    )

    assert restored.loc[0, "authors"] == ["One", "Two"]
    assert restored.loc[0, "wos_categories"] == ["Ecology", "Biology"]
    assert restored.loc[0, "species"] == ["Aerides multiflorum"]
    assert restored.loc[0, "broad_taxa_groups"] == ["Plants"]
    assert restored.loc[0, "taxa_analysis_groups"] == ["Vascular plants"]
    assert restored.loc[0, "taxa_detail_groups"] == ["Vascular plants"]
    assert restored.loc[0, "_partition"] == 1
    assert _raw_results_from_wide(restored) == raw


def test_combined_writer_streams_named_outputs(
    tmp_path: Path,
) -> None:
    base = pd.DataFrame(
        [
            {
                "UT": "WOS:1",
                "title": "Title",
                "authors": ["One"],
                "abstract": "Abstract",
                "source": "Journal",
                "publication_year": 2024,
                "wos_categories": ["Ecology"],
                "doi": "",
            }
        ]
    )
    raw = {
        "WOS:1": [
            {
                "taxon_rank": "species",
                "canonical_name": "Aerides multiflorum",
            }
        ]
    }
    query = TaxonQuery("Aerides multiflorum", "species")
    matches = {
        query.key: MatchResult(
            query.canonical_name,
            query.requested_rank,
            "exact",
            True,
            1,
            response=_api_payload(),
        )
    }
    wide, lineage, _ = build_partition_tables(
        base,
        raw,
        matches,
        partition=1,
        run_name="partition_1",
    )
    writer = _CombinedOutputWriter(tmp_path)
    try:
        writer.add(wide, lineage)
        outputs = writer.finalize()
    finally:
        writer.abort()

    assert {path.name for path in outputs.values()} == {
        "taxa_all.csv",
        "taxa_all.parquet",
        "taxa_lineages.parquet",
    }
    assert len(pd.read_csv(outputs["all_csv"])) == 1
    assert len(pd.read_parquet(outputs["all_parquet"])) == 1
    assert len(pd.read_parquet(outputs["all_lineages"])) == 9


def test_csv_view_preserves_parquet_roundtrip_rank_arrays(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "UT": ["WOS:1"],
            "authors": [["One", "Two"]],
            "domain": [["Eukaryota"]],
            "species": [["Aerides multiflorum"]],
        }
    )
    path = tmp_path / "wide.parquet"
    source.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)

    csv = _csv_view(reloaded)

    assert csv.loc[0, "authors"] == "One; Two"
    assert json.loads(csv.loc[0, "domain"]) == ["Eukaryota"]
    assert json.loads(csv.loc[0, "species"]) == ["Aerides multiflorum"]


def test_cached_only_lookup_fails_without_sending_request(
    tmp_path: Path,
) -> None:
    cache = GBIFMatchCache(tmp_path / "matches.sqlite")
    client = GBIFMatchClient(requests_per_second=1000)
    try:
        with pytest.raises(RuntimeError, match="No GBIF requests were sent"):
            resolve_queries(
                [TaxonQuery("Uncached example", "species")],
                client=client,
                cache=cache,
                taxonomy_build_id="build",
                workers=1,
                refresh_cache=False,
                retry_final_failures=False,
                session_results={},
                allow_api_requests=False,
            )
    finally:
        cache.close()


def test_client_retries_transient_http_status() -> None:
    class FakeResponse:
        def __init__(self, status_code: int, payload: dict | None = None):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}
            self.text = "temporary failure"

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self) -> dict:
            assert self._payload is not None
            return self._payload

    class FakeSession:
        def __init__(self):
            self.responses = [
                FakeResponse(503),
                FakeResponse(200, _api_payload()),
            ]

        def get(self, *args, **kwargs):
            return self.responses.pop(0)

    sleeps: list[float] = []
    client = GBIFMatchClient(
        max_attempts=2,
        retry_delay_seconds=5,
        requests_per_second=1000,
        sleep=sleeps.append,
    )
    fake_session = FakeSession()
    client._session = lambda: fake_session
    client._limiter.acquire = lambda: None

    result = client.match(TaxonQuery("Aerides multiflorum", "species"))

    assert result.status == "exact"
    assert result.attempts == 2
    assert sleeps == [5]

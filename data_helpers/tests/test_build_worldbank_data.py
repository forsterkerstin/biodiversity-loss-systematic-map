"""Tests for the World Bank data builder."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from data_helpers.sources import build_worldbank_data as wb


def _country(
    code: str,
    name: str,
    *,
    region: str = "North America",
    region_code: str = "NAC",
    income_code: str = "HIC",
    income: str = "High income",
) -> dict[str, object]:
    return {
        "id": code,
        "iso2Code": code[:2],
        "name": name,
        "region": {"id": region_code, "value": region},
        "adminregion": {"id": "", "value": ""},
        "incomeLevel": {"id": income_code, "value": income},
        "lendingType": {"id": "", "value": "Not classified"},
        "capitalCity": "",
        "longitude": "",
        "latitude": "",
    }


def _write_ipbes(path: Path) -> None:
    payload = {
        "Americas": {
            "North America": [
                {"GID_0": "USA", "ISO_3166_alpha_3": "USA", "Country": "United States"},
                {"GID_0": "AIA", "ISO_3166_alpha_3": "AIA", "Country": "Anguilla"},
                {"GID_0": "", "ISO_3166_alpha_3": "", "Country": "Hawaii (USA)"},
            ]
        },
        "Europe and Central Asia": {
            "Central and Western Europe": [
                {"GID_0": "", "ISO_3166_alpha_3": "", "Country": "Kosovo"}
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_flatten_and_crosswalk_preserve_unmatched_and_special_geographies(
    tmp_path: Path,
) -> None:
    ipbes_path = tmp_path / "ipbes.json"
    _write_ipbes(ipbes_path)
    countries = wb.normalize_countries(
        [
            _country("USA", "United States"),
            _country(
                "XKX", "Kosovo", region="Europe & Central Asia", region_code="ECS"
            ),
            _country(
                "WLD",
                "World",
                region="Aggregates",
                region_code="NA",
                income="Aggregates",
                income_code="NA",
            ),
        ]
    )

    crosswalk = wb.build_crosswalk(wb.flatten_ipbes(ipbes_path), countries)
    statuses = crosswalk.set_index("ipbes_country")["match_status"].to_dict()

    assert len(crosswalk) == 4
    assert statuses == {
        "United States": "exact_iso3",
        "Anguilla": "no_wb_economy",
        "Hawaii (USA)": "needs_review",
        "Kosovo": "wb_specific_code",
    }
    assert crosswalk.set_index("ipbes_country").at["Kosovo", "wb_entity_code"] == "XKX"
    assert not crosswalk.set_index("ipbes_country").at[
        "Hawaii (USA)", "has_world_bank_economy"
    ]
    assert countries.set_index("wb_entity_code").at["WLD", "is_aggregate"]


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.url = "https://example.test"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _FakeSession:
    def __init__(self, pages: dict[int, object]) -> None:
        self.pages = pages
        self.requested_pages: list[int] = []

    def get(
        self, url: str, *, params: dict[str, object], timeout: int
    ) -> _FakeResponse:
        page = int(params["page"])
        self.requested_pages.append(page)
        return _FakeResponse(self.pages[page])


def test_paginated_client_fetches_every_page_and_checks_total() -> None:
    session = _FakeSession(
        {
            1: [{"page": 1, "pages": 2, "total": 3}, [{"id": "A"}, {"id": "B"}]],
            2: [{"page": 2, "pages": 2, "total": 3}, [{"id": "C"}]],
        }
    )
    client = wb.WorldBankClient(session=session, per_page=2)

    records, metadata = client.paginated("country")

    assert [record["id"] for record in records] == ["A", "B", "C"]
    assert metadata["total"] == 3
    assert session.requested_pages == [1, 2]


def test_paginated_client_rejects_incomplete_result() -> None:
    session = _FakeSession({1: [{"page": 1, "pages": 1, "total": 2}, [{"id": "A"}]]})

    with pytest.raises(wb.WorldBankBuildError, match="announced 2 records"):
        wb.WorldBankClient(session=session).paginated("country")


def test_indicator_normalization_audits_exact_duplicates_and_rejects_conflicts() -> (
    None
):
    record = {
        "id": "TEST.ONE",
        "name": "Test indicator",
        "unit": "%",
        "source": {"id": "2", "value": "World Development Indicators"},
        "sourceNote": "Definition",
        "sourceOrganization": "Publisher",
        "topics": [],
    }

    normalized = wb.normalize_indicators([record, dict(record)])

    assert len(normalized) == 1
    assert normalized.attrs["api_record_count"] == 2
    assert normalized.attrs["exact_duplicate_rows_removed"] == 1

    conflicting = dict(record)
    conflicting["name"] = "Conflicting name"
    with pytest.raises(wb.WorldBankBuildError, match="conflicting metadata"):
        wb.normalize_indicators([record, conflicting])


def _write_tiny_wdi_zip(path: Path) -> None:
    data = pd.DataFrame(
        [
            {
                "Country Name": "United States",
                "Country Code": "USA",
                "Indicator Name": "Indicator one",
                "Indicator Code": "TEST.ONE",
                "2000": 1,
                "2001": None,
            },
            {
                "Country Name": "United States",
                "Country Code": "USA",
                "Indicator Name": "Indicator two",
                "Indicator Code": "TEST.TWO",
                "2000": None,
                "2001": None,
            },
            {
                "Country Name": "Kosovo",
                "Country Code": "XKX",
                "Indicator Name": "Indicator one",
                "Indicator Code": "TEST.ONE",
                "2000": 2,
                "2001": 3,
            },
            {
                "Country Name": "World",
                "Country Code": "WLD",
                "Indicator Name": "Indicator one",
                "Indicator Code": "TEST.ONE",
                "2000": 4,
                "2001": None,
            },
        ]
    )
    series = pd.DataFrame(
        [
            {"Series Code": "TEST.ONE", "Indicator Name": "Indicator one"},
            {"Series Code": "TEST.TWO", "Indicator Name": "Indicator two"},
        ]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("WDICSV.csv", data.to_csv(index=False))
        archive.writestr("WDISeries.csv", series.to_csv(index=False))


def test_wdi_zip_conversion_streams_values_and_audits_missing_cells(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "wdi.zip"
    output = tmp_path / "curated"
    _write_tiny_wdi_zip(archive)
    countries = wb.normalize_countries(
        [
            _country("USA", "United States"),
            _country(
                "XKX", "Kosovo", region="Europe & Central Asia", region_code="ECS"
            ),
            _country(
                "WLD",
                "World",
                region="Aggregates",
                region_code="NA",
                income="Aggregates",
                income_code="NA",
            ),
        ]
    )

    result = wb.convert_wdi_observations(archive, output, countries, chunksize=2)
    observations = pd.read_parquet(output / "observations.parquet")
    coverage = pd.read_parquet(output / "coverage.parquet")
    latest = pd.read_parquet(output / "latest_observations.parquet")

    assert result["source_series_rows"] == 4
    assert result["non_null_observation_rows"] == 4
    assert result["year_start"] == 2000
    assert result["year_end"] == 2001
    assert len(observations) == 4
    assert coverage["missing_observations"].sum() == 4
    assert coverage["non_null_observations"].eq(0).sum() == 1
    assert (
        latest.set_index(["wb_entity_code", "indicator_code"]).at[
            ("XKX", "TEST.ONE"), "year"
        ]
        == 2001
    )
    assert coverage.set_index(["wb_entity_code", "indicator_code"]).at[
        ("WLD", "TEST.ONE"), "is_aggregate"
    ]
    assert (output / "bulk_metadata" / "wdiseries.parquet").exists()

    indicators = wb.normalize_indicators(
        [
            {
                "id": code,
                "name": code,
                "source": {"id": "2", "value": "World Development Indicators"},
                "topics": [],
            }
            for code in ("TEST.ONE", "TEST.TWO")
        ]
    )
    crosswalk = pd.DataFrame(
        [{"ipbes_record_id": "r1", "has_world_bank_economy": True}]
    )
    current = pd.DataFrame([{"wb_entity_code": "USA"}])
    historical = pd.DataFrame([{"wb_entity_code": "USA", "classification_fy": 2027}])
    validation = wb.validate_built_artifacts(
        curated_dir=output,
        crosswalk=crosswalk,
        countries=countries,
        indicators=indicators,
        current_classification=current,
        historical_classification=historical,
        wdi_stats=result,
    )

    assert validation["status"] == "passed"
    assert validation["checks"]["observation_rows"] == 4
    assert validation["checks"]["coverage_arithmetic_valid"]


def test_public_crosswalk_contains_rules_and_status_counts(tmp_path: Path) -> None:
    ipbes_path = tmp_path / "ipbes.json"
    mapping_path = tmp_path / "mapping.json"
    _write_ipbes(ipbes_path)
    countries = wb.normalize_countries(
        [_country("USA", "United States"), _country("XKX", "Kosovo")]
    )
    crosswalk = wb.build_crosswalk(wb.flatten_ipbes(ipbes_path), countries)

    wb.save_public_crosswalk(
        mapping_path,
        crosswalk,
        generated_at="2026-07-16T00:00:00+00:00",
        api_last_updated="2026-07-13",
    )
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["record_count"] == 4
    assert payload["metadata"]["match_status_counts"]["needs_review"] == 1
    assert not payload["metadata"]["latest_indicator_values_included"]
    assert payload["indicator_catalog"] == {}
    assert len(payload["records"]) == 4
    assert all(record["world_bank_indicators"] == {} for record in payload["records"])


def test_public_crosswalk_nests_latest_indicators_by_country(tmp_path: Path) -> None:
    ipbes_path = tmp_path / "ipbes.json"
    mapping_path = tmp_path / "mapping.json"
    _write_ipbes(ipbes_path)
    countries = wb.normalize_countries(
        [_country("USA", "United States"), _country("XKX", "Kosovo")]
    )
    crosswalk = wb.build_crosswalk(wb.flatten_ipbes(ipbes_path), countries)
    indicators = wb.normalize_indicators(
        [
            {
                "id": "NY.GDP.MKTP.CD",
                "name": "GDP (current US$)",
                "unit": "US$",
                "source": {"id": "2", "value": "World Development Indicators"},
                "topics": [{"id": "3", "value": "Economy & Growth"}],
            }
        ]
    )
    latest = pd.DataFrame(
        [
            {
                "source_id": "2",
                "wb_entity_code": "USA",
                "indicator_code": "NY.GDP.MKTP.CD",
                "year": 2025,
                "value": 30_000_000_000_000.0,
            }
        ]
    )

    wb.save_public_crosswalk(
        mapping_path,
        crosswalk,
        generated_at="2026-07-16T00:00:00+00:00",
        api_last_updated="2026-07-13",
        latest_observations=latest,
        indicators=indicators,
    )
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    by_country = {record["ipbes_country"]: record for record in payload["records"]}

    assert payload["metadata"]["latest_indicator_values_included"]
    assert payload["metadata"]["indicator_catalog_count"] == 1
    assert payload["indicator_catalog"]["NY.GDP.MKTP.CD"]["name"] == "GDP (current US$)"
    assert by_country["United States"]["world_bank_indicators"]["NY.GDP.MKTP.CD"] == {
        "value": 30_000_000_000_000.0,
        "year": 2025,
    }
    assert by_country["United States"]["world_bank_indicator_count"] == 1
    assert by_country["Anguilla"]["world_bank_indicators"] == {}


def test_provenance_paths_are_repository_relative(tmp_path: Path) -> None:
    inside = wb.REPO_ROOT / "data" / "world-bank" / "manifest.json"
    assert wb.portable_path(inside) == "data/world-bank/manifest.json"
    assert wb.portable_path(wb.DEFAULT_IPBES_PATH) == (
        "checklists/mappings/ipbes_regions.json"
    )
    outside = tmp_path / "elsewhere.json"
    assert wb.portable_path(outside) == outside.resolve().as_posix()

    record = wb.download_file(
        session=None,  # an existing file is reused without any request
        url="https://example.org/file.json",
        destination=_existing(outside),
    )
    assert record.reused
    assert record.path == outside.resolve().as_posix()


def _existing(path: Path) -> Path:
    path.write_text("{}", encoding="utf-8")
    return path

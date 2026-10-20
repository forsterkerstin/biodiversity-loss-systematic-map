"""Build a reproducible World Bank WDI snapshot joined to the IPBES geography.

The builder deliberately keeps acquisition, normalization, and matching separate:

* the official WDI bulk ZIP is retained as the immutable raw data snapshot;
* World Bank API dimensions are collected with pagination and source IDs;
* the local IPBES JSON is the canonical geographic lookup;
* missing observations and unmatched territories are reported, never imputed;
* current and historical World Bank classifications are stored separately.

Usage
-----
    python -m data_helpers.sources.build_worldbank_data

The default output root is ``data/world-bank``.  Use ``--metadata-only`` for a
quick build that exercises the API, classifications, crosswalk, and reports
without downloading the large WDI observation archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IPBES_PATH = REPO_ROOT / "checklists" / "mappings" / "ipbes_regions.json"
DEFAULT_MAPPING_PATH = (
    REPO_ROOT / "checklists" / "mappings" / "ipbes_world_bank_mapping.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "world-bank"

API_ROOT = "https://api.worldbank.org/v2"
WDI_SOURCE_ID = "2"

# Versioned official files.  They are intentionally not silent "latest" URLs:
# every build manifest therefore identifies the exact upstream snapshot used.
WDI_BULK_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0037712/"
    "DR0095335/WDI_CSV_2026_06_30.zip"
)
CURRENT_CLASSIFICATION_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0037712/"
    "DR0095333/CLASS_2026_07_01.xlsx"
)
HISTORICAL_CLASSIFICATION_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0037712/"
    "DR0095347/CLASS_hist_2026_07_01.xlsx"
)
REVISION_HISTORY_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0037712/"
    "DR0095337/WDIrevisions_26-07.xlsx"
)

DEFAULT_SNAPSHOT_ID = "wdi-2026-06-30"
YEAR_COLUMN = re.compile(r"^\d{4}$")

# Only a World Bank-specific code is considered equivalent enough for an
# automatic join.  Territorial inheritance (for example Hawaii -> USA) is not
# an exact economic match and is intentionally left for explicit later review.
IPBES_WB_OVERRIDES: dict[str, dict[str, str]] = {
    "Kosovo": {
        "wb_entity_code": "XKX",
        "match_status": "wb_specific_code",
        "relationship": "same_economy_world_bank_specific_code",
        "match_note": "World Bank uses XKX; the IPBES record has no ISO3 code.",
    }
}


class WorldBankBuildError(RuntimeError):
    """Raised when an upstream response or normalized artifact is incomplete."""


@dataclass(frozen=True)
class DownloadRecord:
    url: str
    path: str
    size_bytes: int
    sha256: str
    reused: bool


def portable_path(path: str | Path) -> str:
    """Return a POSIX path relative to the repository root when possible.

    Manifests and reports are committed, so they must not record the absolute
    location of the machine that built them.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def snake_case(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_")
    return text.lower()


def json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_scalar) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_session() -> requests.Session:
    retry = Retry(
        total=7,
        connect=7,
        read=7,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "biodiversity-ipbes-world-bank-builder/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class WorldBankClient:
    """Small World Bank API client that never assumes a single response page."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout: int = 90,
        per_page: int = 1000,
    ) -> None:
        self.session = session or build_session()
        self.timeout = timeout
        self.per_page = per_page

    def get_json(self, endpoint: str, params: Mapping[str, Any] | None = None) -> Any:
        url = (
            endpoint
            if endpoint.startswith("http")
            else f"{API_ROOT}/{endpoint.lstrip('/')}"
        )
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise WorldBankBuildError(
                f"World Bank returned invalid JSON for {response.url}"
            ) from exc

    def paginated(
        self, endpoint: str, params: Mapping[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        base_params = dict(params or {})
        base_params.update({"format": "json", "per_page": self.per_page})
        page = 1
        records: list[dict[str, Any]] = []
        first_meta: dict[str, Any] | None = None
        pages = 1

        while page <= pages:
            payload = self.get_json(endpoint, {**base_params, "page": page})
            if not isinstance(payload, list) or len(payload) < 2:
                raise WorldBankBuildError(
                    f"Unexpected paginated response for {endpoint!r}: {type(payload).__name__}"
                )
            meta, rows = payload[0], payload[1]
            if not isinstance(meta, dict):
                raise WorldBankBuildError(f"Missing page metadata for {endpoint!r}")
            if rows is None:
                rows = []
            if not isinstance(rows, list):
                raise WorldBankBuildError(
                    f"Unexpected records payload for {endpoint!r}"
                )
            if first_meta is None:
                first_meta = dict(meta)
                pages = int(meta.get("pages", 1))
            elif int(meta.get("pages", pages)) != pages:
                raise WorldBankBuildError(
                    f"Page count changed while collecting {endpoint!r}: "
                    f"{pages} -> {meta.get('pages')}"
                )
            records.extend(rows)
            page += 1

        assert first_meta is not None
        announced_total = int(first_meta.get("total", len(records)))
        if len(records) != announced_total:
            raise WorldBankBuildError(
                f"Incomplete {endpoint!r}: API announced {announced_total:,} records "
                f"but {len(records):,} were collected"
            )
        return records, first_meta


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    timeout: int = 120,
    force: bool = False,
) -> DownloadRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return DownloadRecord(
            url=url,
            path=portable_path(destination),
            size_bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            reused=True,
        )

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length", 0))
            with partial.open("wb") as handle:
                for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not block:
                        continue
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
                    if size and size % (100 * 1024 * 1024) < len(block):
                        print(
                            f"    downloaded {size / 1024 / 1024:,.0f} MiB", flush=True
                        )
        if expected and size != expected:
            raise WorldBankBuildError(
                f"Incomplete download for {url}: expected {expected}, received {size} bytes"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)

    return DownloadRecord(
        url=url,
        path=portable_path(destination),
        size_bytes=size,
        sha256=digest.hexdigest(),
        reused=False,
    )


def flatten_ipbes(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorldBankBuildError("IPBES mapping must be a nested JSON object")
    rows: list[dict[str, str]] = []
    for region, subregions in data.items():
        if not isinstance(subregions, dict):
            raise WorldBankBuildError(f"IPBES region {region!r} is not an object")
        for subregion, entries in subregions.items():
            if not isinstance(entries, list):
                raise WorldBankBuildError(
                    f"IPBES subregion {region!r}/{subregion!r} is not a list"
                )
            for entry in entries:
                country = str(entry.get("Country", "")).strip()
                iso3 = str(entry.get("ISO_3166_alpha_3", "")).strip().upper()
                gid = str(entry.get("GID_0", "")).strip().upper()
                rows.append(
                    {
                        "ipbes_record_id": f"{region}|{subregion}|{country}",
                        "ipbes_region": str(region),
                        "ipbes_subregion": str(subregion),
                        "ipbes_country": country,
                        "ipbes_gid_0": gid,
                        "ipbes_iso3": iso3,
                    }
                )
    frame = pd.DataFrame(rows)
    nonblank = frame.loc[frame["ipbes_iso3"].ne(""), "ipbes_iso3"]
    duplicates = nonblank[nonblank.duplicated()].unique().tolist()
    if duplicates:
        raise WorldBankBuildError(f"Duplicate IPBES ISO3 codes: {duplicates}")
    if frame["ipbes_record_id"].duplicated().any():
        raise WorldBankBuildError("Duplicate IPBES region/subregion/country records")
    return frame


def nested_value(record: Mapping[str, Any], key: str, subkey: str) -> Any:
    value = record.get(key) or {}
    return value.get(subkey) if isinstance(value, Mapping) else None


def normalize_countries(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        region_name = nested_value(item, "region", "value")
        rows.append(
            {
                "wb_entity_code": item.get("id"),
                "wb_iso2": item.get("iso2Code"),
                "wb_entity_name": item.get("name"),
                "wb_region_code": nested_value(item, "region", "id"),
                "wb_region": region_name,
                "wb_admin_region_code": nested_value(item, "adminregion", "id"),
                "wb_admin_region": nested_value(item, "adminregion", "value"),
                "income_group_code": nested_value(item, "incomeLevel", "id"),
                "income_group": nested_value(item, "incomeLevel", "value"),
                "lending_group_code": nested_value(item, "lendingType", "id"),
                "lending_group": nested_value(item, "lendingType", "value"),
                "capital_city": item.get("capitalCity"),
                "longitude": pd.to_numeric(item.get("longitude"), errors="coerce"),
                "latitude": pd.to_numeric(item.get("latitude"), errors="coerce"),
                "is_aggregate": region_name == "Aggregates",
            }
        )
    frame = pd.DataFrame(rows)
    if (
        frame["wb_entity_code"].isna().any()
        or frame["wb_entity_code"].duplicated().any()
    ):
        raise WorldBankBuildError(
            "World Bank country endpoint returned missing/duplicate IDs"
        )
    return frame.sort_values("wb_entity_code").reset_index(drop=True)


def normalize_sources(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = [
        "id",
        "name",
        "code",
        "description",
        "url",
        "dataavailability",
        "metadataavailability",
        "lastupdated",
        "concepts",
    ]
    frame = pd.DataFrame(
        [{column: row.get(column) for column in columns} for row in records]
    )
    return frame.rename(
        columns={
            "id": "source_id",
            "name": "source_name",
            "code": "source_code",
            "dataavailability": "data_available",
            "metadataavailability": "metadata_available",
            "lastupdated": "last_updated",
        }
    ).sort_values(
        "source_id", key=lambda values: pd.to_numeric(values, errors="coerce")
    )


def normalize_indicators(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in records:
        topics = item.get("topics") or []
        rows.append(
            {
                "source_id": str(nested_value(item, "source", "id") or WDI_SOURCE_ID),
                "source_name": nested_value(item, "source", "value"),
                "indicator_code": item.get("id"),
                "indicator_name": item.get("name"),
                "unit": item.get("unit"),
                "source_note": item.get("sourceNote"),
                "source_organization": item.get("sourceOrganization"),
                "topic_ids": ";".join(str(topic.get("id", "")) for topic in topics),
                "topic_names": ";".join(
                    str(topic.get("value", "")) for topic in topics
                ),
            }
        )
    frame = pd.DataFrame(rows)
    key = ["source_id", "indicator_code"]
    if frame[key].isna().any().any():
        raise WorldBankBuildError("World Bank indicator endpoint returned missing keys")
    api_record_count = len(frame)
    frame = frame.drop_duplicates().copy()
    exact_duplicates_removed = api_record_count - len(frame)
    if frame.duplicated(key).any():
        conflicts = (
            frame.loc[frame.duplicated(key, keep=False), key]
            .drop_duplicates()
            .to_dict("records")
        )
        raise WorldBankBuildError(
            "World Bank indicator endpoint returned conflicting metadata for "
            f"the same source/indicator keys: {conflicts[:10]}"
        )
    frame = frame.sort_values(key).reset_index(drop=True)
    frame.attrs["api_record_count"] = api_record_count
    frame.attrs["exact_duplicate_rows_removed"] = exact_duplicates_removed
    return frame


def build_crosswalk(ipbes: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    economies = countries.loc[~countries["is_aggregate"]].copy()
    by_code = economies.set_index("wb_entity_code").to_dict("index")
    rows: list[dict[str, Any]] = []

    for record in ipbes.to_dict("records"):
        iso3 = record["ipbes_iso3"]
        override = IPBES_WB_OVERRIDES.get(record["ipbes_country"])
        if iso3 and iso3 in by_code:
            code = iso3
            status = "exact_iso3"
            relationship = "same_economy_iso3"
            note = "Exact ISO3 match."
        elif override and override["wb_entity_code"] in by_code:
            code = override["wb_entity_code"]
            status = override["match_status"]
            relationship = override["relationship"]
            note = override["match_note"]
        elif iso3:
            code = None
            status = "no_wb_economy"
            relationship = "no_separate_world_bank_economy"
            note = "IPBES has an ISO3 territory but the World Bank country API has no separate economy."
        else:
            code = None
            status = "needs_review"
            relationship = "non_iso_or_special_geography"
            note = (
                "IPBES special geography has no ISO3; no economic parent was inferred."
            )

        wb = by_code.get(code, {}) if code else {}
        rows.append(
            {
                **record,
                "wb_entity_code": code,
                "wb_entity_name": wb.get("wb_entity_name"),
                "wb_region_code": wb.get("wb_region_code"),
                "wb_region": wb.get("wb_region"),
                "income_group_code": wb.get("income_group_code"),
                "income_group": wb.get("income_group"),
                "lending_group_code": wb.get("lending_group_code"),
                "lending_group": wb.get("lending_group"),
                "match_status": status,
                "relationship": relationship,
                "match_note": note,
                "has_world_bank_economy": code is not None,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(ipbes) or frame["ipbes_record_id"].duplicated().any():
        raise WorldBankBuildError("Crosswalk did not preserve one row per IPBES record")
    return frame


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def save_public_crosswalk(
    path: Path,
    crosswalk: pd.DataFrame,
    *,
    generated_at: str,
    api_last_updated: str | None,
    latest_observations: pd.DataFrame | None = None,
    indicators: pd.DataFrame | None = None,
) -> None:
    if (latest_observations is None) != (indicators is None):
        raise ValueError(
            "latest_observations and indicators must either both be provided or both omitted"
        )

    counts = {
        str(key): int(value)
        for key, value in crosswalk["match_status"].value_counts().sort_index().items()
    }
    indicator_catalog: dict[str, dict[str, Any]] = {}
    values_by_entity: dict[str, dict[str, dict[str, Any]]] = {}
    if latest_observations is not None and indicators is not None:
        for row in indicators.sort_values("indicator_code").itertuples(index=False):
            indicator_catalog[str(row.indicator_code)] = {
                "name": json_scalar(row.indicator_name),
                "unit": json_scalar(row.unit),
                "topics": json_scalar(row.topic_names),
                "source_id": str(row.source_id),
                "source_name": json_scalar(row.source_name),
            }
        latest_columns = [
            "source_id",
            "wb_entity_code",
            "indicator_code",
            "year",
            "value",
        ]
        missing_columns = set(latest_columns) - set(latest_observations.columns)
        if missing_columns:
            raise WorldBankBuildError(
                f"Latest observations are missing columns: {sorted(missing_columns)}"
            )
        for row in latest_observations.sort_values(
            ["wb_entity_code", "indicator_code"]
        )[latest_columns].itertuples(index=False):
            entity_values = values_by_entity.setdefault(str(row.wb_entity_code), {})
            entity_values[str(row.indicator_code)] = {
                "value": json_scalar(row.value),
                "year": int(row.year),
            }

    records = dataframe_records(crosswalk)
    for record in records:
        entity_code = record.get("wb_entity_code")
        entity_values = (
            values_by_entity.get(str(entity_code), {}) if entity_code else {}
        )
        record["world_bank_indicator_count"] = len(entity_values)
        record["world_bank_indicators"] = entity_values

    payload = {
        "metadata": {
            "description": (
                "Analysis-ready IPBES geography to World Bank economy crosswalk "
                "with latest WDI indicator values."
            ),
            "generated_at": generated_at,
            "ipbes_source": str(DEFAULT_IPBES_PATH.relative_to(REPO_ROOT)),
            "world_bank_country_api": f"{API_ROOT}/country",
            "world_bank_source_last_updated": api_last_updated,
            "record_count": int(len(crosswalk)),
            "match_status_counts": counts,
            "latest_indicator_values_included": latest_observations is not None,
            "indicator_catalog_count": len(indicator_catalog),
            "indicator_value_semantics": (
                "Each record's world_bank_indicators object is keyed by WDI indicator "
                "code and contains the latest non-null value plus its observation year. "
                "Definitions are stored once in the top-level indicator_catalog."
            ),
            "rules": [
                "Exact ISO3 matching is preferred.",
                "Kosovo is mapped to the World Bank-specific code XKX.",
                "No sovereign/parent economy is inferred for territories or disputed areas.",
                "Missing World Bank coverage is retained explicitly.",
            ],
        },
        "indicator_catalog": indicator_catalog,
        "records": records,
    }
    write_json(path, payload)


def normalize_current_classification(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = pd.read_excel(path, sheet_name="List of economies", dtype=str).rename(
        columns=snake_case
    )
    frame = frame.rename(
        columns={"code": "wb_entity_code", "economy": "wb_entity_name"}
    )
    # An empty workbook row separates economies from aggregate groups.
    frame = frame.dropna(subset=["wb_entity_code"]).copy()
    frame["is_aggregate"] = (
        frame[["region", "income_group", "lending_category"]].isna().all(axis=1)
    )
    frame["classification_fy"] = 2027
    frame["effective_from"] = "2026-07-01"
    frame["effective_to"] = "2027-06-30"
    frame["gni_reference_year"] = 2025
    composition = pd.read_excel(path, sheet_name="composition", dtype=str).rename(
        columns=snake_case
    )
    notes = pd.read_excel(path, sheet_name="notes", header=None).iloc[:, 0].dropna()
    return (
        frame,
        composition,
        [str(value).strip() for value in notes if str(value).strip() != "Notes"],
    )


def normalize_historical_classification(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = pd.read_excel(path, sheet_name="list", dtype=str).rename(columns=snake_case)
    frame = frame.rename(
        columns={"country": "wb_entity_code", "country_name": "wb_entity_name"}
    )
    frame["classification_release_year"] = pd.to_numeric(
        frame["year"], errors="raise"
    ).astype(int)
    frame["classification_fy"] = frame["classification_release_year"] + 1
    frame["gni_reference_year"] = frame["classification_release_year"] - 1
    frame["effective_from"] = frame["classification_release_year"].map(
        lambda year: f"{year:04d}-07-01"
    )
    frame["effective_to"] = frame["classification_release_year"].map(
        lambda year: f"{year + 1:04d}-06-30"
    )
    frame = frame.drop(columns=["year"])
    counts = pd.read_excel(path, sheet_name="count", header=None).iloc[1:].copy()
    counts.columns = [
        "fiscal_year_label",
        "classification_release_year",
        "ibrd_count",
        "blend_count",
        "ida_count",
        "total_lending_count",
        "high_income_count",
        "upper_middle_income_count",
        "lower_middle_income_count",
        "low_income_count",
        "total_income_count",
    ]
    numeric_columns = counts.columns[1:]
    counts[numeric_columns] = counts[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    counts = counts.dropna(subset=["classification_release_year"]).reset_index(
        drop=True
    )
    counts["classification_release_year"] = counts[
        "classification_release_year"
    ].astype(int)
    counts["classification_fy"] = counts["classification_release_year"] + 1
    notes = pd.read_excel(path, sheet_name="notes", header=None).iloc[:, 0].dropna()
    return frame, counts, [str(value).strip() for value in notes]


def read_zip_csv(zf: zipfile.ZipFile, member: str, **kwargs: Any) -> Any:
    return pd.read_csv(
        zf.open(member), encoding="utf-8-sig", low_memory=False, **kwargs
    )


def find_wdi_data_member(zf: zipfile.ZipFile) -> str:
    candidates = []
    for name in zf.namelist():
        basename = Path(name).name.lower()
        if not basename.endswith(".csv"):
            continue
        if basename in {"wdicsv.csv", "wdidata.csv"} or (
            "wdi" in basename and "country" not in basename and "series" not in basename
        ):
            candidates.append(name)
    if not candidates:
        raise WorldBankBuildError(
            "Could not locate the main WDI data CSV inside the ZIP"
        )
    return sorted(
        candidates, key=lambda name: zf.getinfo(name).file_size, reverse=True
    )[0]


def convert_auxiliary_zip_members(
    zf: zipfile.ZipFile, data_member: str, output_dir: Path
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for member in zf.namelist():
        if member == data_member or not member.lower().endswith(".csv"):
            continue
        frame = read_zip_csv(zf, member)
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
        frame.columns = [snake_case(column) for column in frame.columns]
        destination = output_dir / f"{snake_case(Path(member).stem)}.parquet"
        frame.to_parquet(destination, index=False, compression="zstd")
        written.append(
            {
                "zip_member": member,
                "artifact": portable_path(destination),
                "rows": int(len(frame)),
                "columns": frame.columns.tolist(),
            }
        )
    return written


def _latest_from_wide(chunk: pd.DataFrame, year_columns: list[str]) -> pd.DataFrame:
    numeric = chunk[year_columns].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna()
    any_valid = valid.any(axis=1)
    if not any_valid.any():
        return pd.DataFrame(
            columns=["source_id", "wb_entity_code", "indicator_code", "year", "value"]
        )
    reversed_valid = valid.loc[any_valid, year_columns[::-1]]
    latest_year = reversed_valid.idxmax(axis=1).astype(int)
    row_indices = numeric.index[any_valid]
    values = [
        numeric.at[index, str(year)] for index, year in zip(row_indices, latest_year)
    ]
    return pd.DataFrame(
        {
            "source_id": WDI_SOURCE_ID,
            "wb_entity_code": chunk.loc[row_indices, "Country Code"].astype(str).values,
            "indicator_code": chunk.loc[row_indices, "Indicator Code"]
            .astype(str)
            .values,
            "year": latest_year.values,
            "value": values,
        }
    )


def convert_wdi_observations(
    archive: Path,
    output_dir: Path,
    countries: pd.DataFrame,
    *,
    chunksize: int = 5000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "observations.parquet"
    observations_tmp = observations_path.with_suffix(".parquet.part")
    observations_tmp.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    coverage_parts: list[pd.DataFrame] = []
    latest_parts: list[pd.DataFrame] = []
    observed_pairs: set[tuple[str, str]] = set()
    duplicate_pairs: list[tuple[str, str]] = []
    observation_rows = 0
    source_rows = 0

    aggregate_codes = set(
        countries.loc[countries["is_aggregate"], "wb_entity_code"].astype(str)
    )

    with zipfile.ZipFile(archive) as zf:
        data_member = find_wdi_data_member(zf)
        auxiliary = convert_auxiliary_zip_members(
            zf, data_member, output_dir / "bulk_metadata"
        )
        reader = read_zip_csv(zf, data_member, chunksize=chunksize)
        year_columns: list[str] | None = None

        try:
            for chunk_number, chunk in enumerate(reader, start=1):
                source_rows += len(chunk)
                required = {"Country Code", "Indicator Code"}
                if not required.issubset(chunk.columns):
                    raise WorldBankBuildError(
                        f"WDI bulk data is missing columns: {sorted(required - set(chunk.columns))}"
                    )
                if year_columns is None:
                    year_columns = sorted(
                        [
                            str(column)
                            for column in chunk.columns
                            if YEAR_COLUMN.match(str(column))
                        ]
                    )
                    if not year_columns:
                        raise WorldBankBuildError("WDI bulk data has no annual columns")

                pair_frame = chunk[["Country Code", "Indicator Code"]].astype(str)
                for pair in pair_frame.itertuples(index=False, name=None):
                    if pair in observed_pairs:
                        duplicate_pairs.append(pair)
                    observed_pairs.add(pair)

                numeric = chunk[year_columns].apply(pd.to_numeric, errors="coerce")
                non_null = numeric.notna()
                non_null_count = non_null.sum(axis=1)
                first_year = non_null.idxmax(axis=1).where(non_null_count.gt(0))
                last_year = (
                    non_null[year_columns[::-1]]
                    .idxmax(axis=1)
                    .where(non_null_count.gt(0))
                )
                coverage_parts.append(
                    pd.DataFrame(
                        {
                            "source_id": WDI_SOURCE_ID,
                            "wb_entity_code": pair_frame["Country Code"],
                            "indicator_code": pair_frame["Indicator Code"],
                            "periods_queried": len(year_columns),
                            "non_null_observations": non_null_count.astype(int),
                            "missing_observations": (
                                len(year_columns) - non_null_count
                            ).astype(int),
                            "first_available_year": pd.to_numeric(
                                first_year, errors="coerce"
                            ),
                            "last_available_year": pd.to_numeric(
                                last_year, errors="coerce"
                            ),
                        }
                    )
                )
                latest_parts.append(_latest_from_wide(chunk, year_columns))

                id_vars = ["Country Code", "Indicator Code"]
                long = chunk[id_vars + year_columns].melt(
                    id_vars=id_vars, var_name="year", value_name="value"
                )
                long["value"] = pd.to_numeric(long["value"], errors="coerce")
                long = long.dropna(subset=["value"])
                long = long.rename(
                    columns={
                        "Country Code": "wb_entity_code",
                        "Indicator Code": "indicator_code",
                    }
                )
                long.insert(0, "source_id", WDI_SOURCE_ID)
                long["year"] = long["year"].astype("int16")
                table = pa.Table.from_pandas(long, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        observations_tmp, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                observation_rows += len(long)
                print(
                    f"    WDI chunk {chunk_number}: {source_rows:,} series rows, "
                    f"{observation_rows:,} non-null observations",
                    flush=True,
                )
        finally:
            if writer is not None:
                writer.close()

        if year_columns is None or writer is None:
            raise WorldBankBuildError("WDI archive contained no observation rows")
        if duplicate_pairs:
            observations_tmp.unlink(missing_ok=True)
            raise WorldBankBuildError(
                f"WDI data contains duplicate entity/indicator rows; examples: {duplicate_pairs[:10]}"
            )
        os.replace(observations_tmp, observations_path)

    coverage = pd.concat(coverage_parts, ignore_index=True)
    coverage["is_aggregate"] = coverage["wb_entity_code"].isin(aggregate_codes)
    coverage.to_parquet(
        output_dir / "coverage.parquet", index=False, compression="zstd"
    )

    latest = pd.concat(latest_parts, ignore_index=True)
    latest["is_aggregate"] = latest["wb_entity_code"].isin(aggregate_codes)
    latest.to_parquet(
        output_dir / "latest_observations.parquet", index=False, compression="zstd"
    )

    return {
        "data_member": data_member,
        "zip_members": zf.namelist(),
        "source_series_rows": int(source_rows),
        "non_null_observation_rows": int(observation_rows),
        "entity_indicator_pairs": int(len(coverage)),
        "year_start": int(min(year_columns)),
        "year_end": int(max(year_columns)),
        "year_count": int(len(year_columns)),
        "observations_path": portable_path(observations_path),
        "coverage_path": portable_path(output_dir / "coverage.parquet"),
        "latest_path": portable_path(output_dir / "latest_observations.parquet"),
        "auxiliary_members": auxiliary,
    }


def build_ipbes_latest(
    crosswalk: pd.DataFrame,
    latest_path: Path,
    indicators: pd.DataFrame,
    output_path: Path,
) -> int:
    latest = pd.read_parquet(latest_path)
    matched = crosswalk.loc[
        crosswalk["has_world_bank_economy"],
        [
            "ipbes_record_id",
            "ipbes_region",
            "ipbes_subregion",
            "ipbes_country",
            "ipbes_iso3",
            "wb_entity_code",
            "match_status",
            "relationship",
        ],
    ]
    result = matched.merge(latest, on="wb_entity_code", how="left", validate="1:m")
    result = result.merge(
        indicators[
            ["source_id", "indicator_code", "indicator_name", "unit", "topic_names"]
        ],
        on=["source_id", "indicator_code"],
        how="left",
        validate="m:1",
    )
    result.to_parquet(output_path, index=False, compression="zstd")
    return len(result)


def collection_summary(
    *,
    crosswalk: pd.DataFrame,
    countries: pd.DataFrame,
    indicators: pd.DataFrame,
    sources: pd.DataFrame,
    wdi_stats: Mapping[str, Any] | None,
    generated_at: str,
    downloads: Sequence[DownloadRecord],
) -> dict[str, Any]:
    economies = countries.loc[~countries["is_aggregate"]]
    aggregates = countries.loc[countries["is_aggregate"]]
    unmatched = crosswalk.loc[~crosswalk["has_world_bank_economy"]]
    return {
        "generated_at": generated_at,
        "world_bank_api": {
            "source_count": int(len(sources)),
            "data_bearing_source_count": int(sources["data_available"].eq("Y").sum()),
            "entity_count": int(len(countries)),
            "economy_count": int(len(economies)),
            "aggregate_count": int(len(aggregates)),
            "wdi_indicator_count": int(len(indicators)),
            "wdi_indicator_api_record_count": int(
                indicators.attrs.get("api_record_count", len(indicators))
            ),
            "wdi_indicator_exact_duplicates_removed": int(
                indicators.attrs.get("exact_duplicate_rows_removed", 0)
            ),
            "wdi_last_updated": json_scalar(
                sources.loc[
                    sources["source_id"].astype(str).eq(WDI_SOURCE_ID), "last_updated"
                ].squeeze()
            ),
        },
        "ipbes": {
            "record_count": int(len(crosswalk)),
            "unique_nonblank_iso3": int(
                crosswalk.loc[crosswalk["ipbes_iso3"].ne(""), "ipbes_iso3"].nunique()
            ),
            "blank_iso3_count": int(crosswalk["ipbes_iso3"].eq("").sum()),
            "matched_count": int(crosswalk["has_world_bank_economy"].sum()),
            "unmatched_count": int(len(unmatched)),
            "match_status_counts": {
                str(key): int(value)
                for key, value in crosswalk["match_status"]
                .value_counts()
                .sort_index()
                .items()
            },
            "unmatched_iso3": sorted(
                unmatched.loc[unmatched["ipbes_iso3"].ne(""), "ipbes_iso3"].tolist()
            ),
            "review_required": unmatched.loc[
                unmatched["match_status"].eq("needs_review"), "ipbes_country"
            ].tolist(),
        },
        "wdi_observations": dict(wdi_stats) if wdi_stats else {"status": "not_built"},
        "downloads": [asdict(record) for record in downloads],
    }


def validate_built_artifacts(
    *,
    curated_dir: Path,
    crosswalk: pd.DataFrame,
    countries: pd.DataFrame,
    indicators: pd.DataFrame,
    current_classification: pd.DataFrame,
    historical_classification: pd.DataFrame,
    wdi_stats: Mapping[str, Any] | None,
    public_mapping_path: Path | None = None,
) -> dict[str, Any]:
    """Validate normalized keys and row-count invariants after artifacts exist."""

    checks: dict[str, Any] = {}

    if crosswalk["ipbes_record_id"].duplicated().any():
        raise WorldBankBuildError(
            "Post-build validation: duplicate IPBES crosswalk keys"
        )
    checks["crosswalk_rows"] = int(len(crosswalk))
    checks["crosswalk_unique_record_ids"] = True

    if countries["wb_entity_code"].duplicated().any():
        raise WorldBankBuildError(
            "Post-build validation: duplicate World Bank entity keys"
        )
    checks["entity_rows"] = int(len(countries))
    checks["entity_keys_unique"] = True

    indicator_key = ["source_id", "indicator_code"]
    if indicators.duplicated(indicator_key).any():
        raise WorldBankBuildError("Post-build validation: duplicate indicator keys")
    checks["indicator_rows"] = int(len(indicators))
    checks["indicator_keys_unique"] = True

    if current_classification["wb_entity_code"].duplicated().any():
        raise WorldBankBuildError(
            "Post-build validation: duplicate current classification economy keys"
        )
    checks["current_classification_rows"] = int(len(current_classification))

    historical_key = ["wb_entity_code", "classification_fy"]
    if historical_classification.duplicated(historical_key).any():
        raise WorldBankBuildError(
            "Post-build validation: duplicate historical classification keys"
        )
    checks["historical_classification_rows"] = int(len(historical_classification))
    checks["historical_fy_start"] = int(
        historical_classification["classification_fy"].min()
    )
    checks["historical_fy_end"] = int(
        historical_classification["classification_fy"].max()
    )

    if wdi_stats is not None:
        observations_path = curated_dir / "observations.parquet"
        observation_rows = pq.ParquetFile(observations_path).metadata.num_rows
        if observation_rows != int(wdi_stats["non_null_observation_rows"]):
            raise WorldBankBuildError(
                "Post-build validation: observation Parquet row count differs from build count"
            )
        checks["observation_rows"] = int(observation_rows)

        coverage = pd.read_parquet(curated_dir / "coverage.parquet")
        coverage_key = ["source_id", "wb_entity_code", "indicator_code"]
        if coverage.duplicated(coverage_key).any():
            raise WorldBankBuildError("Post-build validation: duplicate coverage keys")
        if not (
            coverage["non_null_observations"] + coverage["missing_observations"]
            == coverage["periods_queried"]
        ).all():
            raise WorldBankBuildError(
                "Post-build validation: coverage non-null/missing arithmetic failed"
            )
        if len(coverage) != int(wdi_stats["entity_indicator_pairs"]):
            raise WorldBankBuildError(
                "Post-build validation: coverage row count differs from build count"
            )
        checks["coverage_rows"] = int(len(coverage))
        checks["coverage_keys_unique"] = True
        checks["coverage_arithmetic_valid"] = True
        checks["missing_cells"] = int(coverage["missing_observations"].sum())
        checks["non_null_cells"] = int(coverage["non_null_observations"].sum())

        latest = pd.read_parquet(curated_dir / "latest_observations.parquet")
        if latest.duplicated(coverage_key).any():
            raise WorldBankBuildError(
                "Post-build validation: duplicate latest-observation keys"
            )
        expected_latest = int(coverage["non_null_observations"].gt(0).sum())
        if len(latest) != expected_latest:
            raise WorldBankBuildError(
                "Post-build validation: latest-observation coverage is incomplete"
            )
        if (
            not latest["year"]
            .between(int(wdi_stats["year_start"]), int(wdi_stats["year_end"]))
            .all()
        ):
            raise WorldBankBuildError(
                "Post-build validation: latest observation has an invalid year"
            )
        checks["latest_observation_rows"] = int(len(latest))
        checks["latest_keys_unique"] = True
        checks["latest_years_in_range"] = True

        bulk_codes = set(coverage["indicator_code"].astype(str))
        api_codes = set(indicators["indicator_code"].astype(str))
        if bulk_codes != api_codes:
            raise WorldBankBuildError(
                "Post-build validation: API and WDI bulk indicator sets differ"
            )
        checks["bulk_api_indicator_sets_equal"] = True

        if public_mapping_path is not None:
            public_mapping = json.loads(public_mapping_path.read_text(encoding="utf-8"))
            public_records = public_mapping.get("records", [])
            public_catalog = public_mapping.get("indicator_catalog", {})
            if len(public_records) != len(crosswalk):
                raise WorldBankBuildError(
                    "Post-build validation: public mapping has incomplete IPBES records"
                )
            if len(public_catalog) != len(indicators):
                raise WorldBankBuildError(
                    "Post-build validation: public mapping has incomplete indicator catalog"
                )
            nested_value_count = sum(
                len(record.get("world_bank_indicators", {}))
                for record in public_records
            )
            if nested_value_count != int(wdi_stats["ipbes_latest_rows"]):
                raise WorldBankBuildError(
                    "Post-build validation: public mapping has incomplete nested indicators"
                )
            checks["public_mapping_records"] = int(len(public_records))
            checks["public_mapping_indicator_catalog"] = int(len(public_catalog))
            checks["public_mapping_nested_indicator_values"] = int(nested_value_count)

    return {"status": "passed", "checked_at": utc_now(), "checks": checks}


def write_collection_notes(path: Path, summary: Mapping[str, Any]) -> None:
    wb = summary["world_bank_api"]
    ipbes = summary["ipbes"]
    wdi = summary["wdi_observations"]
    unmatched_codes = " ".join(ipbes["unmatched_iso3"]) or "None"
    review_names = ", ".join(ipbes["review_required"]) or "None"
    if wdi.get("status") == "not_built":
        wdi_text = "Observation archive was not built (metadata-only run)."
    else:
        wdi_text = (
            f"The WDI bulk file covered {wdi['year_start']}–{wdi['year_end']} and "
            f"produced {wdi['non_null_observation_rows']:,} non-null observations "
            f"across {wdi['entity_indicator_pairs']:,} entity-indicator rows."
        )
    text = f"""# World Bank collection notes

Generated: {summary['generated_at']}

## Scope collected

- World Bank API source registry: {wb['source_count']:,} sources ({wb['data_bearing_source_count']:,} reporting data availability).
- World Bank country registry: {wb['entity_count']:,} records, comprising {wb['economy_count']:,} economies and {wb['aggregate_count']:,} aggregates.
- World Development Indicators metadata: {wb['wdi_indicator_count']:,} source-2 indicators.
- Raw WDI indicator API records: {wb['wdi_indicator_api_record_count']:,}; exact duplicate metadata rows removed: {wb['wdi_indicator_exact_duplicates_removed']:,}.
- WDI source last-updated value reported by the API: {wb['wdi_last_updated']}.
- {wdi_text}

## IPBES matching

- IPBES records inspected: {ipbes['record_count']:,}.
- Unique nonblank IPBES ISO3 codes: {ipbes['unique_nonblank_iso3']:,}.
- Blank IPBES ISO3 records: {ipbes['blank_iso3_count']:,}.
- Records matched to a World Bank economy: {ipbes['matched_count']:,}.
- Records without a World Bank economy match: {ipbes['unmatched_count']:,}.
- Match statuses: `{json.dumps(ipbes['match_status_counts'], sort_keys=True)}`.

Unmatched ISO3 territories:

```text
{unmatched_codes}
```

Special geographies still requiring an explicit analytical decision:

```text
{review_names}
```

Kosovo is mapped to the World Bank-specific code `XKX`. No parent-economy
inheritance was inferred for territories, dependencies, or disputed areas.

## Missingness policy

- Blank WDI cells are not imputed or gap-filled.
- Only non-null values are stored in the observation fact table.
- `coverage.parquet` records all year cells inspected and counts both non-null
  and missing cells for every entity-indicator row.
- World Bank aggregates are retained and marked explicitly; they are not
  mistaken for ISO3 countries based on code length.
- Current and historical income/lending classifications are stored separately.
- Historical income groups are not reconstructed from revised GNI observations.

## Reproducibility

`manifest.json` records the exact source URLs, SHA-256 hashes, upstream update
dates, row counts, and generated artifacts. The raw versioned downloads are
retained under the snapshot's `raw/` directory.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collect_api_dimensions(
    client: WorldBankClient, raw_api_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    print("Collecting paginated World Bank API dimensions …", flush=True)
    source_rows, source_meta = client.paginated("source")
    country_rows, country_meta = client.paginated("country")
    indicator_rows, indicator_meta = client.paginated(
        f"source/{WDI_SOURCE_ID}/indicator"
    )
    raw_api_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        raw_api_dir / "sources.json", {"metadata": source_meta, "records": source_rows}
    )
    write_json(
        raw_api_dir / "countries.json",
        {"metadata": country_meta, "records": country_rows},
    )
    write_json(
        raw_api_dir / "wdi_indicators.json",
        {"metadata": indicator_meta, "records": indicator_rows},
    )
    return (
        normalize_sources(source_rows),
        normalize_countries(country_rows),
        normalize_indicators(indicator_rows),
        {
            "sources": source_meta,
            "countries": country_meta,
            "indicators": indicator_meta,
        },
    )


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = utc_now()
    output_root = Path(args.output_dir).resolve()
    snapshot_root = output_root / "snapshots" / args.snapshot_id
    raw_dir = snapshot_root / "raw"
    curated_dir = snapshot_root / "curated"
    reports_dir = snapshot_root / "reports"
    mapping_path = Path(args.mapping_output).resolve()
    for directory in (raw_dir, curated_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session = build_session()
    client = WorldBankClient(
        session=session, timeout=args.timeout, per_page=args.per_page
    )
    sources, countries, indicators, api_metadata = collect_api_dimensions(
        client, raw_dir / "api"
    )
    sources.to_parquet(curated_dir / "sources.parquet", index=False, compression="zstd")
    countries.to_parquet(
        curated_dir / "entities.parquet", index=False, compression="zstd"
    )
    indicators.to_parquet(
        curated_dir / "indicators.parquet", index=False, compression="zstd"
    )

    ipbes = flatten_ipbes(Path(args.ipbes_path).resolve())
    crosswalk = build_crosswalk(ipbes, countries)
    crosswalk.to_parquet(
        curated_dir / "ipbes_world_bank_crosswalk.parquet",
        index=False,
        compression="zstd",
    )
    crosswalk.to_csv(reports_dir / "ipbes_world_bank_crosswalk.csv", index=False)
    unmatched = crosswalk.loc[~crosswalk["has_world_bank_economy"]]
    unmatched.to_csv(reports_dir / "ipbes_unmatched.csv", index=False)

    wdi_last_updated_series = sources.loc[
        sources["source_id"].astype(str).eq(WDI_SOURCE_ID), "last_updated"
    ]
    wdi_last_updated = (
        str(wdi_last_updated_series.iloc[0])
        if not wdi_last_updated_series.empty
        else None
    )
    downloads: list[DownloadRecord] = []
    print("Downloading official classification workbooks …", flush=True)
    current_path = raw_dir / "CLASS_2026_07_01.xlsx"
    historical_path = raw_dir / "CLASS_hist_2026_07_01.xlsx"
    revision_path = raw_dir / "WDIrevisions_26-07.xlsx"
    for url, destination in (
        (args.current_classification_url, current_path),
        (args.historical_classification_url, historical_path),
        (args.revision_history_url, revision_path),
    ):
        downloads.append(
            download_file(
                session,
                url,
                destination,
                timeout=args.download_timeout,
                force=args.force_download,
            )
        )

    current, current_composition, current_notes = normalize_current_classification(
        current_path
    )
    historical, historical_counts, historical_notes = (
        normalize_historical_classification(historical_path)
    )
    current.to_parquet(
        curated_dir / "classifications_current.parquet", index=False, compression="zstd"
    )
    current_composition.to_parquet(
        curated_dir / "classification_group_composition.parquet",
        index=False,
        compression="zstd",
    )
    historical.to_parquet(
        curated_dir / "classifications_historical.parquet",
        index=False,
        compression="zstd",
    )
    historical_counts.to_parquet(
        curated_dir / "classification_historical_counts.parquet",
        index=False,
        compression="zstd",
    )
    (reports_dir / "current_classification_source_notes.txt").write_text(
        "\n\n".join(current_notes) + "\n", encoding="utf-8"
    )
    (reports_dir / "historical_classification_source_notes.txt").write_text(
        "\n\n".join(historical_notes) + "\n", encoding="utf-8"
    )

    wdi_stats: dict[str, Any] | None = None
    if not args.metadata_only:
        print("Downloading official WDI bulk archive …", flush=True)
        archive_path = raw_dir / "WDI_CSV_2026_06_30.zip"
        downloads.append(
            download_file(
                session,
                args.wdi_bulk_url,
                archive_path,
                timeout=args.download_timeout,
                force=args.force_download,
            )
        )
        print("Streaming WDI archive into curated Parquet …", flush=True)
        wdi_stats = convert_wdi_observations(
            archive_path,
            curated_dir,
            countries,
            chunksize=args.chunksize,
        )
        wdi_stats["ipbes_latest_rows"] = build_ipbes_latest(
            crosswalk,
            curated_dir / "latest_observations.parquet",
            indicators,
            curated_dir / "ipbes_wdi_latest.parquet",
        )

        bulk_coverage = pd.read_parquet(
            curated_dir / "coverage.parquet",
            columns=["indicator_code", "wb_entity_code", "non_null_observations"],
        )
        bulk_codes = set(bulk_coverage["indicator_code"].astype(str))
        registry_codes = set(indicators["indicator_code"].astype(str))
        wdi_stats["bulk_indicator_count"] = len(bulk_codes)
        wdi_stats["api_registry_indicator_count"] = len(registry_codes)
        wdi_stats["api_indicators_absent_from_bulk"] = sorted(
            registry_codes - bulk_codes
        )
        wdi_stats["bulk_indicators_absent_from_api"] = sorted(
            bulk_codes - registry_codes
        )
        wdi_stats["zero_value_series_rows"] = int(
            bulk_coverage["non_null_observations"].eq(0).sum()
        )

        print(
            "Writing analysis-ready IPBES mapping with latest WDI values …", flush=True
        )
        save_public_crosswalk(
            mapping_path,
            crosswalk,
            generated_at=generated_at,
            api_last_updated=wdi_last_updated,
            latest_observations=pd.read_parquet(
                curated_dir / "latest_observations.parquet"
            ),
            indicators=indicators,
        )
    elif not mapping_path.exists():
        save_public_crosswalk(
            mapping_path,
            crosswalk,
            generated_at=generated_at,
            api_last_updated=wdi_last_updated,
        )
    else:
        print(
            f"Metadata-only run: preserving existing enriched mapping at {mapping_path}",
            flush=True,
        )

    summary = collection_summary(
        crosswalk=crosswalk,
        countries=countries,
        indicators=indicators,
        sources=sources,
        wdi_stats=wdi_stats,
        generated_at=generated_at,
        downloads=downloads,
    )
    validation = validate_built_artifacts(
        curated_dir=curated_dir,
        crosswalk=crosswalk,
        countries=countries,
        indicators=indicators,
        current_classification=current,
        historical_classification=historical,
        wdi_stats=wdi_stats,
        public_mapping_path=mapping_path if wdi_stats is not None else None,
    )
    write_json(reports_dir / "coverage_summary.json", summary)
    write_json(reports_dir / "validation.json", validation)
    write_collection_notes(reports_dir / "collection_notes.md", summary)

    manifest = {
        "schema_version": 1,
        "snapshot_id": args.snapshot_id,
        "generated_at": generated_at,
        "builder": portable_path(__file__),
        "ipbes_source": portable_path(args.ipbes_path),
        "mapping_output": portable_path(mapping_path),
        "api_metadata": api_metadata,
        "summary": summary,
        "validation": validation,
        "artifacts": sorted(
            str(path.relative_to(snapshot_root))
            for path in snapshot_root.rglob("*")
            if path.is_file()
        ),
    }
    write_json(snapshot_root / "manifest.json", manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(
        "# World Bank data\n\n"
        f"Latest built snapshot: `{args.snapshot_id}`\n\n"
        f"See `snapshots/{args.snapshot_id}/manifest.json` and "
        f"`snapshots/{args.snapshot_id}/reports/collection_notes.md`.\n",
        encoding="utf-8",
    )
    print(f"Build complete: {snapshot_root}", flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ipbes-path", type=Path, default=DEFAULT_IPBES_PATH)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--per-page", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--download-timeout", type=int, default=300)
    parser.add_argument("--wdi-bulk-url", default=WDI_BULK_URL)
    parser.add_argument(
        "--current-classification-url", default=CURRENT_CLASSIFICATION_URL
    )
    parser.add_argument(
        "--historical-classification-url", default=HISTORICAL_CLASSIFICATION_URL
    )
    parser.add_argument("--revision-history-url", default=REVISION_HISTORY_URL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_build(parse_args(argv))
    except (WorldBankBuildError, requests.RequestException, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

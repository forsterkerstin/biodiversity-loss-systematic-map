"""Resolve standardized LLM taxa outputs through the GBIF species-match API.

This module deliberately keeps the remote-API workflow separate from the
existing local-cache taxa loader.  It provides:

* streaming batch-output parsing with successful retries preferred over failed
  or older responses;
* release-aware, per-query SQLite caching;
* bounded, rate-limited GBIF requests with transient-error retries;
* atomic, resumable per-partition outputs; and
* wide article tables plus a lossless long-form hierarchy table.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from labelling.taxa_groups import (
    GroupAssignment,
    assign_taxon_groups,
    grouping_rule_table,
    load_group_config,
    order_groups,
)

try:
    import orjson
except ImportError:  # pragma: no cover - project requirements include orjson
    orjson = None

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional fallback
    repair_json = None


GBIF_MATCH_URL = "https://api.gbif.org/v2/species/match"
COL_XR_CHECKLIST_KEY = "7ddf754f-d193-4cc9-b351-99906754a03b"
SCHEMA_VERSION = 5
DEFAULT_GROUP_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "checklists"
    / "mappings"
    / "taxa_broad_groups.json"
)

LLM_TAXON_RANKS = {
    "kingdom",
    "phylum",
    "class",
    "order",
    "genus",
    "species",
}
SPECIAL_VALUES = {"unclear", "not applicable"}
WIDE_RANKS = (
    "domain",
    "kingdom",
    "subkingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
)
TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
CORE_FIELDS = (
    "UT",
    "title",
    "authors",
    "abstract",
    "source",
    "publication_year",
    "wos_categories",
    "doi",
)
LIST_CORE_FIELDS = {"authors", "wos_categories"}
GROUP_LIST_COLUMNS = (
    "broad_taxa_groups",
    "taxa_analysis_groups",
    "taxa_detail_groups",
)
GROUP_SCHEMES = ("broad", "analysis", "detail")
LINEAGE_COLUMNS = (
    "UT",
    "_partition",
    "_source_run",
    "llm_taxon_index",
    "llm_canonical_name",
    "llm_taxon_rank",
    "match_status",
    "matched_taxon_key",
    "matched_name",
    "matched_rank",
    "taxonomic_status",
    "match_type",
    "confidence",
    "synonym",
    "broad_group",
    "broad_group_rule_id",
    "broad_group_reason",
    "broad_group_eligible",
    "analysis_group",
    "analysis_group_rule_id",
    "analysis_group_reason",
    "detail_group",
    "detail_group_rule_id",
    "detail_group_reason",
    "lineage_position",
    "lineage_key",
    "lineage_rank",
    "lineage_name",
    "error_message",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_loads(data: str | bytes) -> Any:
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data)


def _parse_json_text(text: str | None) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return _json_loads(text)
    except Exception:
        if repair_json is None:
            return None
        try:
            return repair_json(text, return_objects=True)
        except Exception:
            return None


def _normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_name(value: str) -> str:
    return _normalize_space(value).casefold()


def _normalize_rank(value: str) -> str:
    return _normalize_space(value).casefold()


def _dedupe_append(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def _collection_as_list(value: Any) -> list[Any] | None:
    """Normalize list-like cells, including arrays loaded from Parquet."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
    return None


def _serialize_cell(value: Any, *, preserve_lists: bool) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    items = _collection_as_list(value)
    if items is not None:
        if preserve_lists:
            return json.dumps(items, ensure_ascii=False)
        return "; ".join("" if item is None else str(item) for item in items)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value


def _atomic_dataframe_write(
    frame: pd.DataFrame,
    destination: Path,
    writer: Callable[[pd.DataFrame, Path], None],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    try:
        writer(frame, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_write(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _write_excel(frame: pd.DataFrame, path: Path) -> None:
    frame.to_excel(path, index=False)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


@dataclass(frozen=True)
class TaxonQuery:
    canonical_name: str
    requested_rank: str

    @property
    def normalized_name(self) -> str:
        return _normalize_name(self.canonical_name)

    @property
    def key(self) -> tuple[str, str]:
        return self.requested_rank, self.normalized_name


@dataclass
class MatchResult:
    query_name: str
    query_rank: str
    status: str
    accepted: bool
    attempts: int
    response: dict[str, Any] | None = None
    error_message: str | None = None
    http_status: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.query_rank, _normalize_name(self.query_name)

    @property
    def usage(self) -> dict[str, Any]:
        if not isinstance(self.response, dict):
            return {}
        value = self.response.get("usage")
        return value if isinstance(value, dict) else {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        if not isinstance(self.response, dict):
            return {}
        value = self.response.get("diagnostics")
        return value if isinstance(value, dict) else {}

    @property
    def classification(self) -> list[dict[str, Any]]:
        if not isinstance(self.response, dict):
            return []
        value = self.response.get("classification")
        if not isinstance(value, list):
            return []
        return [entry for entry in value if isinstance(entry, dict)]


def evaluate_match_response(
    query: TaxonQuery,
    payload: dict[str, Any],
    *,
    attempts: int,
    http_status: int = 200,
) -> MatchResult:
    """Apply the API workflow's conservative acceptance policy."""
    diagnostics = (
        payload.get("diagnostics")
        if isinstance(payload.get("diagnostics"), dict)
        else {}
    )
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    match_type = str(diagnostics.get("matchType") or "").upper()
    confidence_raw = diagnostics.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = float("-inf")

    if not usage:
        note = str(diagnostics.get("note") or "")
        status = (
            "ambiguous" if "multiple equal matches" in note.casefold() else "no_match"
        )
        return MatchResult(
            query.canonical_name,
            query.requested_rank,
            status,
            False,
            attempts,
            response=payload,
            http_status=http_status,
        )

    returned_rank = _normalize_rank(str(usage.get("rank") or ""))
    rank_matches = returned_rank == query.requested_rank
    if match_type == "EXACT":
        status = "exact" if rank_matches else "rank_mismatch"
        accepted = rank_matches
    # GBIF v2 calls fuzzy/spelling matches VARIANT; retain FUZZY for
    # compatibility with older or alternate match payloads.
    elif match_type in {"FUZZY", "VARIANT"}:
        accepted = rank_matches and confidence >= 90
        status = "fuzzy_accepted" if accepted else "fuzzy_review"
    elif match_type in {"HIGHERRANK", "HIGHER_RANK"}:
        status = "higher_rank"
        accepted = False
    else:
        status = "rank_mismatch" if not rank_matches else "fuzzy_review"
        accepted = False

    return MatchResult(
        query.canonical_name,
        query.requested_rank,
        status,
        accepted,
        attempts,
        response=payload,
        http_status=http_status,
    )


class _RateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1.0 / requests_per_second
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self._monotonic()
            delay = max(0.0, self._next_start - now)
            scheduled = max(now, self._next_start)
            self._next_start = scheduled + self._interval
        if delay:
            self._sleep(delay)


class GBIFMatchClient:
    """Thread-safe-through-thread-local-sessions GBIF match client."""

    def __init__(
        self,
        *,
        checklist_key: str = COL_XR_CHECKLIST_KEY,
        timeout_seconds: float = 5.0,
        max_attempts: int = 4,
        retry_delay_seconds: float = 5.0,
        requests_per_second: float = 8.0,
        user_agent: str = "biodiversity-evidence-synthesis/taxa-with-api",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        self.checklist_key = checklist_key
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.user_agent = user_agent
        self._sleep = sleep
        self._limiter = _RateLimiter(
            requests_per_second,
            sleep=sleep,
        )
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                }
            )
            self._thread_local.session = session
        return session

    def _retry_delay(self, response: requests.Response | None) -> float:
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(self.retry_delay_seconds, float(retry_after))
                except ValueError:
                    pass
        return self.retry_delay_seconds

    def fetch_metadata(self) -> dict[str, Any]:
        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            response: requests.Response | None = None
            try:
                self._limiter.acquire()
                response = self._session().get(
                    f"{GBIF_MATCH_URL}/metadata",
                    params={"checklistKey": self.checklist_key},
                    timeout=(self.timeout_seconds, self.timeout_seconds),
                )
                if response.status_code in TRANSIENT_HTTP_STATUSES:
                    raise requests.HTTPError(
                        f"transient HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("metadata response is not a JSON object")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if attempt < self.max_attempts:
                    self._sleep(self._retry_delay(response))
        raise RuntimeError(
            f"Could not retrieve GBIF taxonomy metadata after "
            f"{self.max_attempts} attempts: {last_error}"
        )

    def match(self, query: TaxonQuery) -> MatchResult:
        last_error = "unknown error"
        last_http_status: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            response: requests.Response | None = None
            try:
                self._limiter.acquire()
                response = self._session().get(
                    GBIF_MATCH_URL,
                    params={
                        "scientificName": query.canonical_name,
                        "taxonRank": query.requested_rank,
                        "checklistKey": self.checklist_key,
                    },
                    timeout=(self.timeout_seconds, self.timeout_seconds),
                )
                last_http_status = response.status_code
                if response.status_code in TRANSIENT_HTTP_STATUSES:
                    raise requests.HTTPError(
                        f"transient HTTP {response.status_code}",
                        response=response,
                    )
                if 400 <= response.status_code < 500:
                    return MatchResult(
                        query.canonical_name,
                        query.requested_rank,
                        "permanent_failure",
                        False,
                        attempt,
                        error_message=(
                            f"HTTP {response.status_code}: " f"{response.text[:500]}"
                        ),
                        http_status=response.status_code,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("match response is not a JSON object")
                return evaluate_match_response(
                    query,
                    payload,
                    attempts=attempt,
                    http_status=response.status_code,
                )
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if attempt < self.max_attempts:
                    self._sleep(self._retry_delay(response))
        return MatchResult(
            query.canonical_name,
            query.requested_rank,
            "api_failure",
            False,
            self.max_attempts,
            error_message=last_error,
            http_status=last_http_status,
        )


class GBIFMatchCache:
    """Release-aware SQLite cache committed after every completed lookup."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                taxonomy_build_id TEXT NOT NULL,
                checklist_key TEXT NOT NULL,
                query_rank TEXT NOT NULL,
                query_name_norm TEXT NOT NULL,
                query_name TEXT NOT NULL,
                status TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                attempts INTEGER NOT NULL,
                http_status INTEGER,
                response_json TEXT,
                error_message TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    taxonomy_build_id,
                    checklist_key,
                    query_rank,
                    query_name_norm
                )
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(
        self,
        *,
        taxonomy_build_id: str,
        checklist_key: str,
        query: TaxonQuery,
        retry_final_failures: bool,
    ) -> MatchResult | None:
        row = self.connection.execute(
            """
            SELECT status, accepted, attempts, http_status, response_json,
                   error_message, query_name
            FROM matches
            WHERE taxonomy_build_id = ?
              AND checklist_key = ?
              AND query_rank = ?
              AND query_name_norm = ?
            """,
            (
                taxonomy_build_id,
                checklist_key,
                query.requested_rank,
                query.normalized_name,
            ),
        ).fetchone()
        if row is None:
            return None
        (
            status,
            accepted,
            attempts,
            http_status,
            response_json,
            error_message,
            query_name,
        ) = row
        if retry_final_failures and status == "api_failure":
            return None
        response = json.loads(response_json) if response_json else None
        cached_result = MatchResult(
            query_name=query_name,
            query_rank=query.requested_rank,
            status=status,
            accepted=bool(accepted),
            attempts=int(attempts),
            response=response,
            error_message=error_message,
            http_status=http_status,
        )
        # The API payload is the durable fact; acceptance policy may evolve
        # independently of the taxonomy release. Re-evaluate successful cached
        # responses so a policy/schema update never requires repeat API calls.
        if isinstance(response, dict) and http_status == 200:
            current_result = evaluate_match_response(
                query,
                response,
                attempts=int(attempts),
                http_status=http_status,
            )
            if (
                current_result.status != cached_result.status
                or current_result.accepted != cached_result.accepted
            ):
                self.put(
                    taxonomy_build_id=taxonomy_build_id,
                    checklist_key=checklist_key,
                    result=current_result,
                )
            return current_result
        return cached_result

    def put(
        self,
        *,
        taxonomy_build_id: str,
        checklist_key: str,
        result: MatchResult,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO matches (
                taxonomy_build_id,
                checklist_key,
                query_rank,
                query_name_norm,
                query_name,
                status,
                accepted,
                attempts,
                http_status,
                response_json,
                error_message,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                taxonomy_build_id,
                checklist_key,
                query_rank,
                query_name_norm
            ) DO UPDATE SET
                query_name = excluded.query_name,
                status = excluded.status,
                accepted = excluded.accepted,
                attempts = excluded.attempts,
                http_status = excluded.http_status,
                response_json = excluded.response_json,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            (
                taxonomy_build_id,
                checklist_key,
                result.query_rank,
                _normalize_name(result.query_name),
                result.query_name,
                result.status,
                int(result.accepted),
                result.attempts,
                result.http_status,
                (
                    json.dumps(result.response, ensure_ascii=False)
                    if result.response is not None
                    else None
                ),
                result.error_message,
                _utc_now(),
            ),
        )
        self.connection.commit()

    def to_frame(
        self,
        *,
        taxonomy_build_id: str,
        checklist_key: str,
    ) -> pd.DataFrame:
        return pd.read_sql_query(
            """
            SELECT taxonomy_build_id, checklist_key, query_rank, query_name,
                   query_name_norm, status, accepted, attempts, http_status,
                   response_json, error_message, updated_at
            FROM matches
            WHERE taxonomy_build_id = ? AND checklist_key = ?
            ORDER BY query_rank, query_name_norm
            """,
            self.connection,
            params=(taxonomy_build_id, checklist_key),
        )


def taxonomy_build_identity(metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    main = (
        metadata.get("mainIndex") if isinstance(metadata.get("mainIndex"), dict) else {}
    )
    created = str(metadata.get("created") or "unknown")
    clb_key = str(main.get("clbDatasetKey") or "unknown")
    alias = str(main.get("datasetAlias") or "unknown")
    build_id = f"{clb_key}:{created}"
    summary = {
        "build_id": build_id,
        "created": created,
        "clb_dataset_key": clb_key,
        "dataset_key": main.get("datasetKey"),
        "dataset_alias": alias,
        "dataset_title": main.get("datasetTitle"),
        "name_usage_count": main.get("nameUsageCount"),
    }
    return build_id, summary


def _response_output_text(record: dict[str, Any]) -> str | None:
    response = record.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    if not isinstance(body, dict):
        body = record.get("body")
    if not isinstance(body, dict):
        return None
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                return part["text"]
    return None


def _output_file_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    if "retry" not in name.casefold():
        return 0, 0, name
    match = re.search(r"taxa_retry_(\d+)", name)
    number = int(match.group(1)) if match else 999_999
    return 1, number, name


@dataclass
class RawRunAudit:
    files: int = 0
    response_lines: int = 0
    valid_responses: int = 0
    invalid_responses: int = 0
    duplicate_valid_responses: int = 0
    selected_responses: int = 0
    dataset_records: int = 0
    missing_outputs: int = 0
    unknown_outputs: int = 0


def load_raw_taxa_results(
    output_directory: Path,
) -> tuple[dict[str, list[dict[str, str]]], RawRunAudit]:
    """Stream one run and select the latest valid response per custom_id."""
    paths = sorted(output_directory.glob("*.jsonl"), key=_output_file_sort_key)
    if not paths:
        raise FileNotFoundError(f"No JSONL outputs found in {output_directory}")

    selected: dict[str, list[dict[str, str]]] = {}
    audit = RawRunAudit(files=len(paths))
    for path in paths:
        with path.open("rb") as handle:
            for line in handle:
                audit.response_lines += 1
                try:
                    record = _json_loads(line)
                except Exception:
                    audit.invalid_responses += 1
                    continue
                if not isinstance(record, dict):
                    audit.invalid_responses += 1
                    continue
                custom_id = record.get("custom_id")
                text = _response_output_text(record)
                payload = _parse_json_text(text)
                results = payload.get("results") if isinstance(payload, dict) else None
                if not isinstance(custom_id, str) or not isinstance(results, list):
                    audit.invalid_responses += 1
                    continue

                cleaned: list[dict[str, str]] = []
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    rank = item.get("taxon_rank")
                    canonical = item.get("canonical_name")
                    if not isinstance(rank, str) or not isinstance(canonical, str):
                        continue
                    rank = _normalize_space(rank)
                    canonical = _normalize_space(canonical)
                    if rank and canonical:
                        cleaned.append(
                            {
                                "taxon_rank": rank,
                                "canonical_name": canonical,
                            }
                        )
                # An explicitly empty result list is valid, but a non-empty
                # partially malformed list must not overwrite an earlier good
                # response from the original batch or a previous retry.
                if len(cleaned) != len(results):
                    audit.invalid_responses += 1
                    continue
                if custom_id in selected:
                    audit.duplicate_valid_responses += 1
                selected[custom_id] = cleaned
                audit.valid_responses += 1
    audit.selected_responses = len(selected)
    return selected, audit


def _input_fingerprint(
    dataset_path: Path,
    output_directory: Path,
) -> str:
    paths = [dataset_path, *sorted(output_directory.glob("*.jsonl"))]
    details = []
    for path in paths:
        stat = path.stat()
        details.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(details, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_dataset(
    dataset_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, dict):
        raise ValueError(f"Dataset has no documents object: {dataset_path}")
    uts = list(documents)
    rows = []
    for ut in uts:
        document = documents[ut]
        if not isinstance(document, dict):
            document = {}
        rows.append(
            {
                field: (str(ut) if field == "UT" else document.get(field, ""))
                for field in CORE_FIELDS
            }
        )
    return pd.DataFrame(rows, columns=CORE_FIELDS), uts


def resolve_queries(
    queries: Iterable[TaxonQuery],
    *,
    client: GBIFMatchClient,
    cache: GBIFMatchCache,
    taxonomy_build_id: str,
    workers: int,
    refresh_cache: bool,
    retry_final_failures: bool,
    session_results: dict[tuple[str, str], MatchResult],
    allow_api_requests: bool = True,
) -> tuple[dict[tuple[str, str], MatchResult], dict[str, int]]:
    """Resolve unique queries, committing each new result immediately."""
    unique = {query.key: query for query in queries}
    results: dict[tuple[str, str], MatchResult] = {}
    pending: list[TaxonQuery] = []
    cache_hits = 0
    session_hits = 0
    for key, query in unique.items():
        # --refresh-api-cache ignores results from previous invocations, while
        # still de-duplicating a query across partitions in this invocation.
        if key in session_results:
            results[key] = session_results[key]
            session_hits += 1
            continue
        cached = None
        if not refresh_cache:
            cached = cache.get(
                taxonomy_build_id=taxonomy_build_id,
                checklist_key=client.checklist_key,
                query=query,
                retry_final_failures=retry_final_failures,
            )
        if cached is None:
            pending.append(query)
        else:
            results[key] = cached
            session_results[key] = cached
            cache_hits += 1

    print(
        f"    Taxon lookups: {len(unique):,} unique; "
        f"{cache_hits:,} cache hit(s), {session_hits:,} session hit(s), "
        f"{len(pending):,} API request(s)."
    )
    if pending and not allow_api_requests:
        sample = ", ".join(
            f"{query.canonical_name!r} ({query.requested_rank})"
            for query in pending[:5]
        )
        raise RuntimeError(
            "Cached-only mode found "
            f"{len(pending):,} uncached taxon quer{'y' if len(pending) == 1 else 'ies'}. "
            "No GBIF requests were sent. "
            f"Sample: {sample}"
        )
    status_counts: Counter[str] = Counter(result.status for result in results.values())
    if pending:
        completed = 0
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(client.match, query): query for query in pending}
            for future in as_completed(futures):
                query = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # defensive: worker must not stop the run
                    result = MatchResult(
                        query.canonical_name,
                        query.requested_rank,
                        "api_failure",
                        False,
                        client.max_attempts,
                        error_message=f"Unhandled worker error: {exc}",
                    )
                results[query.key] = result
                session_results[query.key] = result
                cache.put(
                    taxonomy_build_id=taxonomy_build_id,
                    checklist_key=client.checklist_key,
                    result=result,
                )
                status_counts[result.status] += 1
                completed += 1
                if completed == 1 or completed % 250 == 0 or completed == len(pending):
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = completed / elapsed
                    remaining = (len(pending) - completed) / rate if rate else 0
                    print(
                        f"      {completed:,}/{len(pending):,} API requests "
                        f"({rate:.1f}/s; ~{remaining / 60:.1f} min remaining)",
                        flush=True,
                    )

    audit = {
        "unique_queries": len(unique),
        "cache_hits": cache_hits,
        "session_hits": session_hits,
        "api_requests": len(pending),
        "api_failures": status_counts["api_failure"],
        **{
            f"status_{status}": count for status, count in sorted(status_counts.items())
        },
    }
    return results, audit


def _item_resolution_summary(
    item: dict[str, str],
    result: MatchResult | None,
    status: str,
    group_assignments: dict[str, GroupAssignment],
    error_message: str | None = None,
) -> dict[str, Any]:
    usage = result.usage if result else {}
    diagnostics = result.diagnostics if result else {}
    assignment_payload = {
        scheme: asdict(assignment) for scheme, assignment in group_assignments.items()
    }
    summary = {
        "taxon_rank": item.get("taxon_rank"),
        "canonical_name": item.get("canonical_name"),
        "status": status,
        "accepted": bool(result.accepted) if result else False,
        "matched_taxon_key": usage.get("key"),
        "matched_name": usage.get("canonicalName") or usage.get("name"),
        "matched_rank": usage.get("rank"),
        "match_type": diagnostics.get("matchType"),
        "confidence": diagnostics.get("confidence"),
        "group_assignments": assignment_payload,
        "error": error_message or (result.error_message if result else None),
    }
    for scheme in GROUP_SCHEMES:
        assignment = group_assignments[scheme]
        summary[f"{scheme}_group"] = assignment.group
        summary[f"{scheme}_group_rule_id"] = assignment.rule_id
        summary[f"{scheme}_group_reason"] = assignment.reason
    summary["broad_group_eligible"] = group_assignments["broad"].eligible
    return summary


def build_partition_tables(
    base: pd.DataFrame,
    raw_results: dict[str, list[dict[str, str]]],
    match_results: dict[tuple[str, str], MatchResult],
    *,
    partition: int,
    run_name: str,
    group_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create the wide article table and lossless long hierarchy table."""
    if group_config is None:
        group_config = load_group_config(DEFAULT_GROUP_CONFIG_PATH)
    wide_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    group_item_counts = {scheme: Counter() for scheme in group_config["scheme_order"]}
    group_reason_counts = {scheme: Counter() for scheme in group_config["scheme_order"]}
    total_items = 0
    accepted_items = 0

    for base_row in base.to_dict("records"):
        ut = str(base_row["UT"])
        items = raw_results.get(ut, [])
        has_llm_output = ut in raw_results
        rank_values: dict[str, list[str]] = {rank: [] for rank in WIDE_RANKS}
        item_groups: dict[str, list[str]] = {
            scheme: [] for scheme in group_config["scheme_order"]
        }
        item_summaries: list[dict[str, Any]] = []

        for item_index, item in enumerate(items):
            total_items += 1
            raw_rank = str(item.get("taxon_rank") or "")
            raw_name = str(item.get("canonical_name") or "")
            rank = _normalize_rank(raw_rank)
            normalized_name = _normalize_name(raw_name)

            result: MatchResult | None = None
            if rank in SPECIAL_VALUES or normalized_name in SPECIAL_VALUES:
                status = "special_value"
                error_message = None
            elif rank not in LLM_TAXON_RANKS:
                status = "invalid_rank"
                error_message = f"Unsupported LLM taxon rank: {raw_rank!r}"
            elif not normalized_name:
                status = "invalid_name"
                error_message = "Empty canonical name"
            else:
                result = match_results.get((rank, normalized_name))
                if result is None:
                    status = "api_failure"
                    error_message = "No lookup result available"
                else:
                    status = result.status
                    error_message = result.error_message

            status_counts[status] += 1
            if result is not None and result.accepted:
                accepted_items += 1
                for lineage in result.classification:
                    lineage_rank = _normalize_rank(str(lineage.get("rank") or ""))
                    lineage_name = str(lineage.get("name") or "").strip()
                    if lineage_rank in rank_values and lineage_name:
                        _dedupe_append(rank_values[lineage_rank], lineage_name)

            classification = (
                result.classification if result is not None and result.accepted else []
            )
            group_assignments = assign_taxon_groups(
                raw_rank=raw_rank,
                raw_name=raw_name,
                match_status=status,
                classification=classification,
                config=group_config,
            )
            for scheme, assignment in group_assignments.items():
                group_reason_counts[scheme][assignment.reason] += 1
                if assignment.group is not None:
                    item_groups[scheme].append(assignment.group)
                    group_item_counts[scheme][assignment.group] += 1

            summary = _item_resolution_summary(
                item,
                result,
                status,
                group_assignments,
                error_message,
            )
            item_summaries.append(summary)
            usage = result.usage if result else {}
            diagnostics = result.diagnostics if result else {}
            base_lineage = {
                "UT": ut,
                "_partition": partition,
                "_source_run": run_name,
                "llm_taxon_index": item_index,
                "llm_canonical_name": raw_name,
                "llm_taxon_rank": raw_rank,
                "match_status": status,
                "matched_taxon_key": usage.get("key"),
                "matched_name": usage.get("canonicalName") or usage.get("name"),
                "matched_rank": usage.get("rank"),
                "taxonomic_status": usage.get("status"),
                "match_type": diagnostics.get("matchType"),
                "confidence": diagnostics.get("confidence"),
                "synonym": (
                    result.response.get("synonym")
                    if result is not None and isinstance(result.response, dict)
                    else None
                ),
                "broad_group": group_assignments["broad"].group,
                "broad_group_rule_id": group_assignments["broad"].rule_id,
                "broad_group_reason": group_assignments["broad"].reason,
                "broad_group_eligible": group_assignments["broad"].eligible,
                "analysis_group": group_assignments["analysis"].group,
                "analysis_group_rule_id": (group_assignments["analysis"].rule_id),
                "analysis_group_reason": (group_assignments["analysis"].reason),
                "detail_group": group_assignments["detail"].group,
                "detail_group_rule_id": group_assignments["detail"].rule_id,
                "detail_group_reason": group_assignments["detail"].reason,
                "error_message": error_message,
            }
            if classification:
                for position, lineage in enumerate(classification):
                    lineage_rows.append(
                        {
                            **base_lineage,
                            "lineage_position": position,
                            "lineage_key": lineage.get("key"),
                            "lineage_rank": lineage.get("rank"),
                            "lineage_name": lineage.get("name"),
                        }
                    )
            else:
                lineage_rows.append(
                    {
                        **base_lineage,
                        "lineage_position": None,
                        "lineage_key": None,
                        "lineage_rank": None,
                        "lineage_name": None,
                    }
                )

        matched_count = sum(bool(item.get("accepted")) for item in item_summaries)
        unresolved_count = sum(
            not bool(item.get("accepted")) for item in item_summaries
        )
        api_failure_count = sum(
            item.get("status") == "api_failure" for item in item_summaries
        )
        ordered_groups = {
            scheme: order_groups(
                item_groups[scheme],
                group_config,
                scheme=scheme,
            )
            for scheme in group_config["scheme_order"]
        }
        grouped_count = sum(
            bool(item.get("broad_group"))
            and item.get("broad_group") != group_config["unresolved_group"]
            for item in item_summaries
        )
        group_unresolved_count = sum(
            item.get("broad_group") == group_config["unresolved_group"]
            for item in item_summaries
        )
        group_skipped_count = sum(
            not bool(item.get("broad_group_eligible")) for item in item_summaries
        )
        if not has_llm_output:
            record_status = "missing_llm_output"
        elif not items:
            record_status = "no_taxa_reported"
        elif api_failure_count:
            record_status = "api_failure"
        elif matched_count and unresolved_count:
            record_status = "partly_resolved"
        elif matched_count:
            record_status = "resolved"
        else:
            record_status = "unresolved"

        wide_rows.append(
            {
                **base_row,
                **rank_values,
                "llm_taxa_json": json.dumps(items, ensure_ascii=False),
                "taxa_match_status_json": json.dumps(
                    item_summaries,
                    ensure_ascii=False,
                ),
                "broad_taxa_groups": ordered_groups["broad"],
                "n_broad_taxa_groups": len(ordered_groups["broad"]),
                "taxa_analysis_groups": ordered_groups["analysis"],
                "n_taxa_analysis_groups": len(ordered_groups["analysis"]),
                "taxa_detail_groups": ordered_groups["detail"],
                "n_taxa_detail_groups": len(ordered_groups["detail"]),
                "n_taxa_items_grouped": grouped_count,
                "n_taxa_items_group_unresolved": group_unresolved_count,
                "n_taxa_items_group_skipped": group_skipped_count,
                "n_llm_taxa": len(items),
                "taxa_record_status": record_status,
                "n_taxa_matched": matched_count,
                "n_taxa_unresolved": unresolved_count,
                "n_taxa_api_failed": api_failure_count,
                "_partition": partition,
                "_source_run": run_name,
            }
        )

    wide = pd.DataFrame(wide_rows)
    lineage = pd.DataFrame(lineage_rows, columns=LINEAGE_COLUMNS)
    audit = {
        "records": len(wide),
        "llm_taxon_items": total_items,
        "accepted_taxon_items": accepted_items,
        "status_counts": dict(sorted(status_counts.items())),
        "group_item_counts_by_scheme": {
            scheme: dict(sorted(group_item_counts[scheme].items()))
            for scheme in group_config["scheme_order"]
        },
        "group_reason_counts_by_scheme": {
            scheme: dict(sorted(group_reason_counts[scheme].items()))
            for scheme in group_config["scheme_order"]
        },
        "broad_group_item_counts": dict(sorted(group_item_counts["broad"].items())),
        "broad_group_reason_counts": dict(sorted(group_reason_counts["broad"].items())),
        "grouped_taxon_items": int(sum(wide["n_taxa_items_grouped"])),
        "group_unresolved_taxon_items": int(sum(wide["n_taxa_items_group_unresolved"])),
        "group_skipped_taxon_items": int(sum(wide["n_taxa_items_group_skipped"])),
        "records_with_api_failures": int(wide["n_taxa_api_failed"].gt(0).sum()),
    }
    return wide, lineage, audit


def _year_range(frame: pd.DataFrame) -> tuple[int, int]:
    years = pd.to_numeric(frame["publication_year"], errors="coerce").dropna()
    if years.empty:
        return 0, 0
    return int(years.max()), int(years.min())


def _queries_from_raw(
    raw_results: dict[str, list[dict[str, str]]],
) -> list[TaxonQuery]:
    unique: dict[tuple[str, str], TaxonQuery] = {}
    for items in raw_results.values():
        for item in items:
            rank = _normalize_rank(str(item.get("taxon_rank") or ""))
            name = _normalize_space(str(item.get("canonical_name") or ""))
            normalized_name = _normalize_name(name)
            if (
                rank not in LLM_TAXON_RANKS
                or rank in SPECIAL_VALUES
                or normalized_name in SPECIAL_VALUES
                or not normalized_name
            ):
                continue
            query = TaxonQuery(name, rank)
            unique.setdefault(query.key, query)
    return list(unique.values())


def _completion_is_reusable(
    marker_path: Path,
    *,
    run_name: str,
    taxonomy_build_id: str,
    input_fingerprint: str,
    grouping_config_sha256: str,
) -> tuple[bool, dict[str, Any] | None]:
    if not marker_path.exists():
        return False, None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    if not isinstance(marker, dict):
        return False, None
    reusable = (
        marker.get("schema_version") == SCHEMA_VERSION
        and marker.get("status") in {"complete", "complete_with_failures"}
        and marker.get("source_run") == run_name
        and marker.get("taxonomy_build_id") == taxonomy_build_id
        and marker.get("input_fingerprint") == input_fingerprint
        and marker.get("grouping_config_sha256") == grouping_config_sha256
    )
    if reusable:
        path = marker.get("excel")
        if not isinstance(path, str) or not Path(path).exists():
            reusable = False
    return reusable, marker


def _partition_paths(
    output_base: Path,
    *,
    partition: int,
    max_year: int,
    min_year: int,
) -> dict[str, Path]:
    directory = output_base / str(partition)
    stem = f"taxa_{partition:02d}_{max_year}-{min_year}"
    return {
        "directory": directory,
        "excel": directory / f"{stem}.xlsx",
        "marker": directory / "completion.json",
    }


WIDE_OUTPUT_COLUMNS = (
    *CORE_FIELDS,
    *WIDE_RANKS,
    "llm_taxa_json",
    "taxa_match_status_json",
    "broad_taxa_groups",
    "n_broad_taxa_groups",
    "taxa_analysis_groups",
    "n_taxa_analysis_groups",
    "taxa_detail_groups",
    "n_taxa_detail_groups",
    "n_taxa_items_grouped",
    "n_taxa_items_group_unresolved",
    "n_taxa_items_group_skipped",
    "taxa_record_status",
    "n_llm_taxa",
    "n_taxa_matched",
    "n_taxa_unresolved",
    "n_taxa_api_failed",
    "_partition",
    "_source_run",
)


def _excel_view(wide: pd.DataFrame) -> pd.DataFrame:
    result = wide[
        [column for column in WIDE_OUTPUT_COLUMNS if column in wide.columns]
    ].copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda value, current=column: _serialize_cell(
                value,
                preserve_lists=(
                    current in WIDE_RANKS
                    or current in LIST_CORE_FIELDS
                    or current in GROUP_LIST_COLUMNS
                ),
            )
        )
    return result


def _list_from_excel_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        parsed = _parse_json_text(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON list in Excel, got: {value[:200]!r}")
        return [str(item) for item in parsed if item is not None]
    items = _collection_as_list(value)
    if items is not None:
        return [str(item) for item in items if item is not None]
    raise ValueError(f"Expected a list-like Excel cell, got {type(value).__name__}")


def _read_partition_excel(
    path: Path,
    *,
    partition: int,
    run_name: str,
) -> pd.DataFrame:
    """Restore a partition workbook to the in-memory wide-table representation."""
    frame = pd.read_excel(
        path,
        dtype={
            "UT": str,
            "llm_taxa_json": str,
            "taxa_match_status_json": str,
        },
    )
    missing_columns = set(WIDE_OUTPUT_COLUMNS) - set(frame.columns)
    if missing_columns - {"_partition", "_source_run"}:
        raise ValueError(
            f"Partition workbook is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    for column in (*WIDE_RANKS, *LIST_CORE_FIELDS, *GROUP_LIST_COLUMNS):
        frame[column] = frame[column].map(_list_from_excel_cell)
    for column in ("llm_taxa_json", "taxa_match_status_json"):
        frame[column] = frame[column].fillna("[]").astype(str)
    for column in (
        "n_llm_taxa",
        "n_taxa_matched",
        "n_taxa_unresolved",
        "n_taxa_api_failed",
        "n_broad_taxa_groups",
        "n_taxa_analysis_groups",
        "n_taxa_detail_groups",
        "n_taxa_items_grouped",
        "n_taxa_items_group_unresolved",
        "n_taxa_items_group_skipped",
    ):
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
        )
    frame["UT"] = frame["UT"].astype(str)
    frame["_partition"] = partition
    frame["_source_run"] = run_name
    return frame[list(WIDE_OUTPUT_COLUMNS)]


def _raw_results_from_wide(
    wide: pd.DataFrame,
) -> dict[str, list[dict[str, str]]]:
    """Recover standardized LLM results from an XLSX partition checkpoint."""
    raw_results: dict[str, list[dict[str, str]]] = {}
    columns = ["UT", "llm_taxa_json", "taxa_record_status"]
    for row in wide[columns].itertuples(index=False):
        if row.taxa_record_status == "missing_llm_output":
            continue
        payload = _parse_json_text(row.llm_taxa_json)
        if not isinstance(payload, list):
            raise ValueError(
                f"Invalid llm_taxa_json in resumed partition for UT {row.UT}"
            )
        cleaned: list[dict[str, str]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid taxon item in resumed partition for UT {row.UT}"
                )
            rank = item.get("taxon_rank")
            name = item.get("canonical_name")
            if not isinstance(rank, str) or not isinstance(name, str):
                raise ValueError(
                    f"Invalid taxon fields in resumed partition for UT {row.UT}"
                )
            cleaned.append(
                {
                    "taxon_rank": _normalize_space(rank),
                    "canonical_name": _normalize_space(name),
                }
            )
        raw_results[str(row.UT)] = cleaned
    return raw_results


def _csv_view(wide: pd.DataFrame) -> pd.DataFrame:
    result = wide.copy()
    for column in result.columns:
        if column in (*WIDE_RANKS, *GROUP_LIST_COLUMNS):
            result[column] = result[column].map(
                lambda value: json.dumps(
                    _collection_as_list(value) or [],
                    ensure_ascii=False,
                )
            )
        elif column in LIST_CORE_FIELDS:
            result[column] = result[column].map(
                lambda value: _serialize_cell(value, preserve_lists=False)
            )
    return result


_WIDE_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("UT", pa.string()),
        pa.field("title", pa.string()),
        pa.field("authors", pa.list_(pa.string())),
        pa.field("abstract", pa.string()),
        pa.field("source", pa.string()),
        pa.field("publication_year", pa.int64()),
        pa.field("wos_categories", pa.list_(pa.string())),
        pa.field("doi", pa.string()),
        *[pa.field(rank, pa.list_(pa.string())) for rank in WIDE_RANKS],
        pa.field("llm_taxa_json", pa.string()),
        pa.field("taxa_match_status_json", pa.string()),
        pa.field("broad_taxa_groups", pa.list_(pa.string())),
        pa.field("n_broad_taxa_groups", pa.int64()),
        pa.field("taxa_analysis_groups", pa.list_(pa.string())),
        pa.field("n_taxa_analysis_groups", pa.int64()),
        pa.field("taxa_detail_groups", pa.list_(pa.string())),
        pa.field("n_taxa_detail_groups", pa.int64()),
        pa.field("n_taxa_items_grouped", pa.int64()),
        pa.field("n_taxa_items_group_unresolved", pa.int64()),
        pa.field("n_taxa_items_group_skipped", pa.int64()),
        pa.field("taxa_record_status", pa.string()),
        pa.field("n_llm_taxa", pa.int64()),
        pa.field("n_taxa_matched", pa.int64()),
        pa.field("n_taxa_unresolved", pa.int64()),
        pa.field("n_taxa_api_failed", pa.int64()),
        pa.field("_partition", pa.int64()),
        pa.field("_source_run", pa.string()),
    ]
)

_LINEAGE_STRING_COLUMNS = {
    "UT",
    "_source_run",
    "llm_canonical_name",
    "llm_taxon_rank",
    "match_status",
    "matched_taxon_key",
    "matched_name",
    "matched_rank",
    "taxonomic_status",
    "match_type",
    "broad_group",
    "broad_group_rule_id",
    "broad_group_reason",
    "analysis_group",
    "analysis_group_rule_id",
    "analysis_group_reason",
    "detail_group",
    "detail_group_rule_id",
    "detail_group_reason",
    "lineage_key",
    "lineage_rank",
    "lineage_name",
    "error_message",
}
_LINEAGE_INTEGER_COLUMNS = {
    "_partition",
    "llm_taxon_index",
    "lineage_position",
}
_LINEAGE_BOOLEAN_COLUMNS = {"synonym", "broad_group_eligible"}
_LINEAGE_PARQUET_SCHEMA = pa.schema(
    [
        pa.field(
            column,
            (
                pa.string()
                if column in _LINEAGE_STRING_COLUMNS
                else (
                    pa.int64()
                    if column in _LINEAGE_INTEGER_COLUMNS
                    else pa.float64() if column == "confidence" else pa.bool_()
                )
            ),
        )
        for column in LINEAGE_COLUMNS
    ]
)


def _prepare_wide_for_parquet(wide: pd.DataFrame) -> pd.DataFrame:
    result = wide[list(WIDE_OUTPUT_COLUMNS)].copy()
    for column in (*WIDE_RANKS, *LIST_CORE_FIELDS, *GROUP_LIST_COLUMNS):
        result[column] = result[column].map(
            lambda value: [
                str(item)
                for item in (_collection_as_list(value) or [])
                if item is not None
            ]
        )
    string_columns = (set(CORE_FIELDS) - LIST_CORE_FIELDS - {"publication_year"}) | {
        "llm_taxa_json",
        "taxa_match_status_json",
        "taxa_record_status",
        "_source_run",
    }
    for column in string_columns:
        result[column] = result[column].astype("string")
    for column in (
        "publication_year",
        "n_llm_taxa",
        "n_taxa_matched",
        "n_taxa_unresolved",
        "n_taxa_api_failed",
        "n_broad_taxa_groups",
        "n_taxa_analysis_groups",
        "n_taxa_detail_groups",
        "n_taxa_items_grouped",
        "n_taxa_items_group_unresolved",
        "n_taxa_items_group_skipped",
        "_partition",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("Int64")
    return result


def _prepare_lineage_for_parquet(lineage: pd.DataFrame) -> pd.DataFrame:
    result = lineage[list(LINEAGE_COLUMNS)].copy()
    for column in _LINEAGE_STRING_COLUMNS:
        result[column] = result[column].astype("string")
    for column in _LINEAGE_INTEGER_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("Int64")
    result["confidence"] = pd.to_numeric(
        result["confidence"],
        errors="coerce",
    ).astype("Float64")
    for column in _LINEAGE_BOOLEAN_COLUMNS:
        result[column] = result[column].astype("boolean")
    return result


class _GroupingAuditAccumulator:
    """Accumulate small grouping and hierarchy audits partition by partition."""

    def __init__(self, group_config: dict[str, Any]) -> None:
        self.group_config = group_config
        self.records = 0
        self.taxonomic_items = 0
        self.group_article_counts = {
            scheme: Counter() for scheme in group_config["scheme_order"]
        }
        self.group_item_counts = {
            scheme: Counter() for scheme in group_config["scheme_order"]
        }
        self.assignment_counts: Counter[tuple[str, str, str, str, bool]] = Counter()
        self.rank_article_counts = {
            rank: Counter() for rank in ("kingdom", "phylum", "class")
        }
        self.rank_item_counts = {
            rank: Counter() for rank in ("kingdom", "phylum", "class")
        }

    def add(self, wide: pd.DataFrame, lineage: pd.DataFrame) -> None:
        self.records += len(wide)
        for row in wide.itertuples(index=False):
            for scheme in self.group_config["scheme_order"]:
                output_column = self.group_config["schemes"][scheme]["output_column"]
                for group in _collection_as_list(getattr(row, output_column)) or []:
                    self.group_article_counts[scheme][str(group)] += 1
            summaries = _parse_json_text(row.taxa_match_status_json)
            if not isinstance(summaries, list):
                raise ValueError(f"Invalid taxa_match_status_json for UT {row.UT}")
            self.taxonomic_items += len(summaries)
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                assignments = summary.get("group_assignments")
                if not isinstance(assignments, dict):
                    raise ValueError("taxa_match_status_json has no group_assignments.")
                for scheme in self.group_config["scheme_order"]:
                    assignment = assignments.get(scheme)
                    if not isinstance(assignment, dict):
                        raise ValueError(
                            "taxa_match_status_json is missing grouping "
                            f"scheme {scheme!r}."
                        )
                    group = str(assignment.get("group") or "")
                    reason = str(assignment.get("reason") or "")
                    rule_id = str(assignment.get("rule_id") or "")
                    eligible = bool(assignment.get("eligible"))
                    self.assignment_counts[
                        (scheme, group, rule_id, reason, eligible)
                    ] += 1
                    if group:
                        self.group_item_counts[scheme][group] += 1

        for rank in self.rank_article_counts:
            for values in wide[rank]:
                for value in set(_collection_as_list(values) or []):
                    self.rank_article_counts[rank][str(value)] += 1

        if lineage.empty:
            return
        inventory = lineage.loc[
            lineage["lineage_rank"]
            .astype("string")
            .str.casefold()
            .isin(self.rank_item_counts)
            & lineage["lineage_name"].notna(),
            ["UT", "llm_taxon_index", "lineage_rank", "lineage_name"],
        ].copy()
        if inventory.empty:
            return
        inventory["lineage_rank"] = inventory["lineage_rank"].astype(str).str.casefold()
        inventory["lineage_name"] = inventory["lineage_name"].astype(str)
        inventory = inventory.drop_duplicates(
            ["UT", "llm_taxon_index", "lineage_rank", "lineage_name"]
        )
        for (rank, name), count in inventory.value_counts(
            ["lineage_rank", "lineage_name"]
        ).items():
            self.rank_item_counts[str(rank)][str(name)] += int(count)

    def write(self, audit_directory: Path) -> dict[str, Path]:
        assignment_rows = [
            {
                "scheme": scheme,
                "taxa_group": group or None,
                "rule_id": rule_id or None,
                "reason": reason,
                "eligible": eligible,
                "taxon_item_count": int(count),
            }
            for (
                scheme,
                group,
                rule_id,
                reason,
                eligible,
            ), count in self.assignment_counts.items()
        ]
        assignment_audit = pd.DataFrame(
            assignment_rows,
            columns=[
                "scheme",
                "taxa_group",
                "rule_id",
                "reason",
                "eligible",
                "taxon_item_count",
            ],
        ).sort_values(
            ["scheme", "eligible", "taxon_item_count", "reason"],
            ascending=[True, False, False, True],
            ignore_index=True,
        )

        paths = {
            "group_assignment_audit": (
                audit_directory / "taxa_group_assignment_audit.csv"
            ),
            "group_rules": audit_directory / "taxa_group_rules.csv",
        }
        for scheme in self.group_config["scheme_order"]:
            scheme_config = self.group_config["schemes"][scheme]
            group_summary = pd.DataFrame(
                [
                    {
                        "taxa_group": group,
                        "article_count": int(self.group_article_counts[scheme][group]),
                        "article_share": (
                            self.group_article_counts[scheme][group] / self.records
                            if self.records
                            else 0.0
                        ),
                        "taxon_item_count": int(self.group_item_counts[scheme][group]),
                        "taxon_item_share": (
                            self.group_item_counts[scheme][group] / self.taxonomic_items
                            if self.taxonomic_items
                            else 0.0
                        ),
                    }
                    for group in scheme_config["group_order"]
                ]
            )
            key = "group_summary" if scheme == "broad" else f"{scheme}_group_summary"
            filename = (
                "taxa_group_summary.csv"
                if scheme == "broad"
                else f"taxa_{scheme}_group_summary.csv"
            )
            paths[key] = audit_directory / filename
            _atomic_dataframe_write(
                group_summary,
                paths[key],
                _write_csv,
            )
        _atomic_dataframe_write(
            assignment_audit,
            paths["group_assignment_audit"],
            _write_csv,
        )
        _atomic_dataframe_write(
            pd.DataFrame(grouping_rule_table(self.group_config)),
            paths["group_rules"],
            _write_csv,
        )

        for rank in ("kingdom", "phylum", "class"):
            labels = set(self.rank_article_counts[rank]) | set(
                self.rank_item_counts[rank]
            )
            inventory = pd.DataFrame(
                [
                    {
                        rank: label,
                        "article_count": int(self.rank_article_counts[rank][label]),
                        "taxon_item_count": int(self.rank_item_counts[rank][label]),
                    }
                    for label in labels
                ],
                columns=[rank, "article_count", "taxon_item_count"],
            ).sort_values(
                ["article_count", "taxon_item_count", rank],
                ascending=[False, False, True],
                ignore_index=True,
            )
            key = f"{rank}_inventory"
            paths[key] = audit_directory / f"taxa_{rank}_inventory.csv"
            _atomic_dataframe_write(inventory, paths[key], _write_csv)
        return paths


class _CombinedOutputWriter:
    """Incrementally build atomic combined outputs with one partition in memory."""

    def __init__(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        self.destinations = {
            "all_csv": output_directory / "taxa_all.csv",
            "all_parquet": output_directory / "taxa_all.parquet",
            "all_lineages": output_directory / "taxa_lineages.parquet",
        }
        self.temporary = {
            key: path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
            for key, path in self.destinations.items()
        }
        for path in self.temporary.values():
            if path.exists():
                path.unlink()
        self._wide_writer: pq.ParquetWriter | None = None
        self._lineage_writer: pq.ParquetWriter | None = None
        self._csv_started = False
        self._finalized = False

    def add(self, wide: pd.DataFrame, lineage: pd.DataFrame) -> None:
        csv_frame = _csv_view(wide)
        csv_frame.to_csv(
            self.temporary["all_csv"],
            mode="a",
            header=not self._csv_started,
            index=False,
        )
        self._csv_started = True

        wide_table = pa.Table.from_pandas(
            _prepare_wide_for_parquet(wide),
            schema=_WIDE_PARQUET_SCHEMA,
            preserve_index=False,
            safe=False,
        )
        lineage_table = pa.Table.from_pandas(
            _prepare_lineage_for_parquet(lineage),
            schema=_LINEAGE_PARQUET_SCHEMA,
            preserve_index=False,
            safe=False,
        )
        if self._wide_writer is None:
            self._wide_writer = pq.ParquetWriter(
                self.temporary["all_parquet"],
                _WIDE_PARQUET_SCHEMA,
                compression="zstd",
            )
        if self._lineage_writer is None:
            self._lineage_writer = pq.ParquetWriter(
                self.temporary["all_lineages"],
                _LINEAGE_PARQUET_SCHEMA,
                compression="zstd",
            )
        self._wide_writer.write_table(wide_table)
        self._lineage_writer.write_table(lineage_table)

    def _close_writers(self) -> None:
        if self._wide_writer is not None:
            self._wide_writer.close()
            self._wide_writer = None
        if self._lineage_writer is not None:
            self._lineage_writer.close()
            self._lineage_writer = None

    def finalize(self) -> dict[str, Path]:
        self._close_writers()
        if not self._csv_started:
            raise RuntimeError("No partition records were written.")
        for key, destination in self.destinations.items():
            os.replace(self.temporary[key], destination)
        self._finalized = True
        return self.destinations

    def abort(self) -> None:
        self._close_writers()
        if self._finalized:
            return
        for path in self.temporary.values():
            if path.exists():
                path.unlink()


def _load_cached_taxonomy_context(
    output_base: Path,
    cache_path: Path,
) -> tuple[str, str, dict[str, Any]]:
    """Load the build identity needed to address a release-aware local cache."""
    metadata_path = output_base / "metadata.json"
    if not metadata_path.exists() or not cache_path.exists():
        raise RuntimeError(
            "--cached-only requires an existing metadata.json and "
            f"GBIF cache at {cache_path}."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        api = metadata["api"]
        taxonomy_summary = api["taxonomy"]
        checklist_key = str(api["checklist_key"])
        taxonomy_build_id = str(taxonomy_summary["build_id"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot recover GBIF cache identity from {metadata_path}."
        ) from exc
    if not checklist_key or not taxonomy_build_id:
        raise RuntimeError(f"Incomplete GBIF cache identity in {metadata_path}.")
    return checklist_key, taxonomy_build_id, dict(taxonomy_summary)


def process_taxa_with_api(
    *,
    run_names: list[str],
    datasets_directory: Path,
    batch_outputs_directory: Path,
    output_base: Path,
    workers: int,
    requests_per_second: float,
    timeout_seconds: float,
    max_attempts: int,
    retry_delay_seconds: float,
    user_agent: str,
    force: bool,
    refresh_cache: bool,
    retry_final_failures: bool,
    skipped_missing: dict[str, list[str]],
    group_config_path: Path = DEFAULT_GROUP_CONFIG_PATH,
    cached_only: bool = False,
) -> None:
    """Run the complete resumable taxa-with-api pipeline."""
    output_base.mkdir(parents=True, exist_ok=True)
    state_directory = output_base / "state"
    audit_directory = output_base / "audit"
    all_directory = output_base / "all"
    cache_path = state_directory / "gbif_match_cache.sqlite"

    group_config = load_group_config(group_config_path)
    if cached_only:
        (
            checklist_key,
            taxonomy_build_id,
            taxonomy_summary,
        ) = _load_cached_taxonomy_context(output_base, cache_path)
        print(
            "Cached-only mode: using local GBIF matches for "
            f"{taxonomy_summary.get('dataset_alias', checklist_key)} "
            f"(build {taxonomy_build_id}); remote requests are disabled."
        )
    else:
        checklist_key = COL_XR_CHECKLIST_KEY

    client = GBIFMatchClient(
        checklist_key=checklist_key,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        requests_per_second=requests_per_second,
        user_agent=user_agent,
    )
    if not cached_only:
        print("Retrieving GBIF taxonomy-build metadata ...")
        taxonomy_metadata = client.fetch_metadata()
        taxonomy_build_id, taxonomy_summary = taxonomy_build_identity(taxonomy_metadata)
        print(
            f"GBIF taxonomy: {taxonomy_summary['dataset_alias']} "
            f"(build {taxonomy_build_id})"
        )

    cache = GBIFMatchCache(cache_path)
    combined_writer = _CombinedOutputWriter(all_directory)
    grouping_audit = _GroupingAuditAccumulator(group_config)
    session_results: dict[tuple[str, str], MatchResult] = {}
    partition_details: list[dict[str, Any]] = []
    seen_uts: set[str] = set()
    total_match_statuses: Counter[str] = Counter()
    total_record_statuses: Counter[str] = Counter()
    total_records = 0
    total_lineage_rows = 0
    total_records_with_api_failures = 0
    effective_retry_final_failures = retry_final_failures and not cached_only

    try:
        for partition, run_name in enumerate(run_names, start=1):
            print(
                f"[{partition}/{len(run_names)}] Processing '{run_name}' "
                "(task=taxa-with-api) ..."
            )
            dataset_path = datasets_directory / f"{run_name}-dataset.json"
            output_directory = batch_outputs_directory / run_name
            fingerprint = _input_fingerprint(dataset_path, output_directory)
            base, dataset_uts = _load_dataset(dataset_path)
            max_year, min_year = _year_range(base)
            paths = _partition_paths(
                output_base,
                partition=partition,
                max_year=max_year,
                min_year=min_year,
            )

            reusable, marker = _completion_is_reusable(
                paths["marker"],
                run_name=run_name,
                taxonomy_build_id=taxonomy_build_id,
                input_fingerprint=fingerprint,
                grouping_config_sha256=group_config["_grouping_sha256"],
            )
            if reusable and not force and not refresh_cache:
                assert marker is not None
                print(
                    "    [resume] completed XLSX found; rebuilding combined "
                    "outputs from cached matches."
                )
                checkpoint_wide = _read_partition_excel(
                    Path(marker["excel"]),
                    partition=partition,
                    run_name=run_name,
                )
                raw_results = _raw_results_from_wide(checkpoint_wide)
                match_results, resume_lookup_audit = resolve_queries(
                    _queries_from_raw(raw_results),
                    client=client,
                    cache=cache,
                    taxonomy_build_id=taxonomy_build_id,
                    workers=workers,
                    refresh_cache=False,
                    retry_final_failures=effective_retry_final_failures,
                    session_results=session_results,
                    allow_api_requests=not cached_only,
                )
                wide, lineage, table_audit = build_partition_tables(
                    base,
                    raw_results,
                    match_results,
                    partition=partition,
                    run_name=run_name,
                    group_config=group_config,
                )
                summary = marker.get("summary")
                detail = dict(summary) if isinstance(summary, dict) else {}
                detail["resumed"] = True
                detail["resume_lookup_audit"] = resume_lookup_audit
                detail["table_audit"] = table_audit
            else:
                raw_results, raw_audit = load_raw_taxa_results(output_directory)
                dataset_set = set(dataset_uts)
                raw_set = set(raw_results)
                unknown = raw_set - dataset_set
                missing = dataset_set - raw_set
                raw_audit.dataset_records = len(dataset_set)
                raw_audit.unknown_outputs = len(unknown)
                raw_audit.missing_outputs = len(missing)
                if unknown:
                    raise AssertionError(
                        f"[{run_name}] {len(unknown):,} batch-output UT(s) "
                        f"are absent from the dataset. Sample: {sorted(unknown)[:5]}"
                    )
                if missing:
                    print(
                        f"    [warn] {len(missing):,} dataset record(s) have "
                        "no valid batch output; they will be retained unresolved."
                    )

                queries = _queries_from_raw(raw_results)
                match_results, lookup_audit = resolve_queries(
                    queries,
                    client=client,
                    cache=cache,
                    taxonomy_build_id=taxonomy_build_id,
                    workers=workers,
                    refresh_cache=refresh_cache,
                    retry_final_failures=effective_retry_final_failures,
                    session_results=session_results,
                    allow_api_requests=not cached_only,
                )
                wide, lineage, table_audit = build_partition_tables(
                    base,
                    raw_results,
                    match_results,
                    partition=partition,
                    run_name=run_name,
                    group_config=group_config,
                )

                paths["directory"].mkdir(parents=True, exist_ok=True)
                _atomic_dataframe_write(
                    _excel_view(wide),
                    paths["excel"],
                    _write_excel,
                )
                status = (
                    "complete_with_failures"
                    if table_audit["records_with_api_failures"]
                    else "complete"
                )
                detail = {
                    "partition": partition,
                    "source_run": run_name,
                    "file": str(paths["excel"].resolve()),
                    "records": len(wide),
                    "year_range": {"min": min_year, "max": max_year},
                    "raw_batch_audit": asdict(raw_audit),
                    "lookup_audit": lookup_audit,
                    "table_audit": table_audit,
                    "resumed": False,
                }
                marker_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "status": status,
                    "completed_at": _utc_now(),
                    "source_run": run_name,
                    "partition": partition,
                    "taxonomy_build_id": taxonomy_build_id,
                    "input_fingerprint": fingerprint,
                    "grouping_config_sha256": group_config["_grouping_sha256"],
                    "grouping_config_file_sha256": group_config["_config_sha256"],
                    "grouping_config_schema_version": group_config["schema_version"],
                    "excel": str(paths["excel"].resolve()),
                    "summary": detail,
                }
                _atomic_json_write(marker_payload, paths["marker"])
                print(
                    f"    {len(wide):,} records → "
                    f"{paths['excel'].relative_to(output_base.parent.parent)}"
                )

            run_uts = set(wide["UT"].astype(str))
            overlap = run_uts & seen_uts
            if overlap:
                raise AssertionError(
                    f"Duplicate UTs across partitions: {len(overlap):,}; "
                    f"sample: {sorted(overlap)[:5]}"
                )
            seen_uts.update(run_uts)
            combined_writer.add(wide, lineage)
            grouping_audit.add(wide, lineage)
            partition_details.append(detail)
            total_records += len(wide)
            total_lineage_rows += len(lineage)
            total_match_statuses.update(table_audit["status_counts"])
            total_record_statuses.update(
                wide["taxa_record_status"].value_counts().to_dict()
            )
            total_records_with_api_failures += int(
                wide["n_taxa_api_failed"].gt(0).sum()
            )

        if not total_records:
            print("No records found. Nothing to write.")
            return

        print("Finalizing combined /all outputs ...")
        combined_paths = combined_writer.finalize()
        all_csv = combined_paths["all_csv"]
        all_parquet = combined_paths["all_parquet"]
        all_lineages = combined_paths["all_lineages"]

        cache_frame = cache.to_frame(
            taxonomy_build_id=taxonomy_build_id,
            checklist_key=client.checklist_key,
        )
        matches_path = audit_directory / "gbif_matches.parquet"
        failures_path = audit_directory / "gbif_api_failures.csv"
        _atomic_dataframe_write(cache_frame, matches_path, _write_parquet)
        failures = cache_frame.loc[
            cache_frame["status"].isin({"api_failure", "permanent_failure"})
        ].copy()
        _atomic_dataframe_write(failures, failures_path, _write_csv)
        grouping_audit_paths = grouping_audit.write(audit_directory)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "task": "taxa-with-api",
            "output_dir": str(output_base),
            "partition_format": "xlsx",
            "run_names": run_names,
            "skipped_missing": skipped_missing,
            "api": {
                "match_url": GBIF_MATCH_URL,
                "checklist_key": client.checklist_key,
                "workers": workers,
                "requests_per_second": requests_per_second,
                "timeout_seconds": timeout_seconds,
                "max_attempts": max_attempts,
                "retry_delay_seconds": retry_delay_seconds,
                "user_agent": user_agent,
                "taxonomy": taxonomy_summary,
                "cached_only": cached_only,
            },
            "cache": {
                "path": str(cache_path),
                "refresh_cache": refresh_cache,
                "retry_final_failures": retry_final_failures,
                "effective_retry_final_failures": (effective_retry_final_failures),
                "entries_for_build": len(cache_frame),
            },
            "broad_taxa_grouping": {
                "config_path": group_config["_config_path"],
                "config_sha256": group_config["_config_sha256"],
                "grouping_sha256": group_config["_grouping_sha256"],
                "config_schema_version": group_config["schema_version"],
                "eligible_match_statuses": group_config["eligible_match_statuses"],
                "group_order": group_config["group_order"],
                "group_colors": group_config["group_colors"],
                "group_clipart": group_config["group_clipart"],
                "unresolved_group": group_config["unresolved_group"],
            },
            "taxa_grouping": {
                "config_path": group_config["_config_path"],
                "config_sha256": group_config["_config_sha256"],
                "grouping_sha256": group_config["_grouping_sha256"],
                "config_schema_version": group_config["schema_version"],
                "eligible_match_statuses": group_config["eligible_match_statuses"],
                "scheme_order": group_config["scheme_order"],
                "clipart_note": group_config["clipart_note"],
                "clipart_assets": group_config["clipart_assets"],
                "schemes": {
                    scheme: {
                        "description": group_config["schemes"][scheme].get(
                            "description"
                        ),
                        "output_column": group_config["schemes"][scheme][
                            "output_column"
                        ],
                        "group_order": group_config["schemes"][scheme]["group_order"],
                        "group_colors": group_config["schemes"][scheme]["group_colors"],
                        "group_clipart": group_config["schemes"][scheme][
                            "group_clipart"
                        ],
                        "unresolved_group": group_config["schemes"][scheme][
                            "unresolved_group"
                        ],
                    }
                    for scheme in group_config["scheme_order"]
                },
            },
            "outputs": {
                "all_csv": str(all_csv),
                "all_parquet": str(all_parquet),
                "all_lineages": str(all_lineages),
                "gbif_matches": str(matches_path),
                "api_failures": str(failures_path),
                **{key: str(path) for key, path in grouping_audit_paths.items()},
            },
            "totals": {
                "records": total_records,
                "lineage_rows": total_lineage_rows,
                "partitions": len(partition_details),
                "match_statuses": dict(sorted(total_match_statuses.items())),
                "record_statuses": dict(sorted(total_record_statuses.items())),
                "records_with_api_failures": (total_records_with_api_failures),
                "taxon_items": grouping_audit.taxonomic_items,
                "broad_group_article_counts": {
                    group: int(grouping_audit.group_article_counts["broad"][group])
                    for group in group_config["group_order"]
                },
                "broad_group_taxon_item_counts": {
                    group: int(grouping_audit.group_item_counts["broad"][group])
                    for group in group_config["group_order"]
                },
                "group_article_counts_by_scheme": {
                    scheme: {
                        group: int(grouping_audit.group_article_counts[scheme][group])
                        for group in group_config["schemes"][scheme]["group_order"]
                    }
                    for scheme in group_config["scheme_order"]
                },
                "group_taxon_item_counts_by_scheme": {
                    scheme: {
                        group: int(grouping_audit.group_item_counts[scheme][group])
                        for group in group_config["schemes"][scheme]["group_order"]
                    }
                    for scheme in group_config["scheme_order"]
                },
            },
            "partitions": partition_details,
        }
        _atomic_json_write(metadata, output_base / "metadata.json")
        print(f"Combined CSV: {all_csv}")
        print(f"Combined Parquet: {all_parquet}")
        print(f"Metadata: {output_base / 'metadata.json'}")
    finally:
        combined_writer.abort()
        cache.close()

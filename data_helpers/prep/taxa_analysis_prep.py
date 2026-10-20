"""Build and validate the canonical analysis-ready taxa handoff.

Raw source linkage, serialized-label parsing, taxonomic grouping states, match
audits, and the GBIF broad benchmark are performed once in the data-processing
layer. Results notebooks consume the compact bundle and perform only their
analysis-specific filtering, estimation, inference, and plotting.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_helpers.labels import parse_list_labels
from data_helpers.analysis.taxa.benchmark import DescribedDiversity
from data_helpers.prep._provenance import source_signature


SCHEMA_VERSION = 4
INCLUSION_ORDER = (
    "Included: at least one benchmarkable broad group",
    "Excluded: Not applicable only",
    "Excluded: Unclear only",
    "Excluded: mixed Not applicable and Unclear",
    "Excluded: accepted taxon but broad hierarchy unresolved",
    "Excluded: reported taxon but no accepted benchmarkable match",
    "Excluded: no taxon item returned",
)
NOT_APPLICABLE_VALUES = {"not applicable", "non applicable"}
UNCLEAR_VALUES = {"unclear"}
SPECIAL_EXCLUSION_VALUES = NOT_APPLICABLE_VALUES | UNCLEAR_VALUES
LIST_COLUMNS = (
    "drivers",
    "threat_l0",
    "realms",
    "class_labels",
    "broad_groups_all",
    "broad_groups",
    "analysis_groups_all",
    "analysis_groups",
    "detail_groups_all",
    "detail_groups",
    "taxa_kingdom_labels",
    "taxa_phylum_labels",
    "taxa_class_labels",
    "taxa_order_labels",
    "taxa_family_labels",
    "taxa_genus_labels",
    "taxa_species_labels",
)
TAXONOMY_RANK_COLUMNS = (
    ("kingdom", "taxa_kingdom_labels"),
    ("phylum", "taxa_phylum_labels"),
    ("class", "taxa_class_labels"),
    ("order", "taxa_order_labels"),
    ("family", "taxa_family_labels"),
    ("genus", "taxa_genus_labels"),
    ("species", "taxa_species_labels"),
)
TAXA_SOURCE_COLUMNS = (
    "UT",
    *(source for source, _ in TAXONOMY_RANK_COLUMNS),
    "broad_taxa_groups",
    "taxa_analysis_groups",
    "taxa_detail_groups",
    "llm_taxa_json",
    "taxa_match_status_json",
    "taxa_record_status",
    "n_llm_taxa",
    "n_taxa_matched",
    "n_taxa_unresolved",
    "n_taxa_api_failed",
)
CORPUS_COLUMNS = (
    "UT",
    "publication_year",
    "s2_dir",
    "driver",
    "pred_threat_l0",
    "realm",
    "pred_study_design",
)
TAXA_MATCH_METADATA_COLUMNS = (
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
    "error_message",
)
TAXA_LINEAGE_COLUMNS = (
    "lineage_position",
    "lineage_key",
    "lineage_rank",
    "lineage_name",
)
TAXA_LINEAGE_STRUCT_TYPE = pa.struct(
    [
        pa.field("position", pa.int64()),
        pa.field("key", pa.string()),
        pa.field("rank", pa.string()),
        pa.field("name", pa.string()),
    ]
)
TAXA_MATCH_STRUCT_TYPE = pa.struct(
    [
        pa.field("llm_taxon_index", pa.int64()),
        pa.field("llm_canonical_name", pa.string()),
        pa.field("llm_taxon_rank", pa.string()),
        pa.field("match_status", pa.string()),
        pa.field("matched_taxon_key", pa.string()),
        pa.field("matched_name", pa.string()),
        pa.field("matched_rank", pa.string()),
        pa.field("taxonomic_status", pa.string()),
        pa.field("match_type", pa.string()),
        pa.field("confidence", pa.float64()),
        pa.field("synonym", pa.bool_()),
        pa.field("broad_group", pa.string()),
        pa.field("broad_group_rule_id", pa.string()),
        pa.field("broad_group_reason", pa.string()),
        pa.field("broad_group_eligible", pa.bool_()),
        pa.field("analysis_group", pa.string()),
        pa.field("analysis_group_rule_id", pa.string()),
        pa.field("analysis_group_reason", pa.string()),
        pa.field("detail_group", pa.string()),
        pa.field("detail_group_rule_id", pa.string()),
        pa.field("detail_group_reason", pa.string()),
        pa.field("error_message", pa.string()),
        pa.field("lineage", pa.list_(TAXA_LINEAGE_STRUCT_TYPE)),
    ]
)
TAXA_MATCH_TABLE_SCHEMA = pa.schema(
    [
        pa.field("UT", pa.string()),
        pa.field("taxa_matches", pa.list_(TAXA_MATCH_STRUCT_TYPE)),
    ]
)


class TaxaAnalysisPrepError(ValueError):
    """Raised when the prepared taxa handoff violates its contract."""


@dataclass(frozen=True)
class TaxaPreparedBundle:
    """Loaded analysis-ready publications, GBIF benchmark, and provenance."""

    articles: pd.DataFrame
    benchmark: DescribedDiversity
    manifest: dict[str, Any]

    def audit_frame(self, name: str) -> pd.DataFrame:
        rows = self.manifest.get("audits", {}).get(name)
        if rows is None:
            raise TaxaAnalysisPrepError(f"Prepared manifest has no audit: {name}")
        return pd.DataFrame(rows)


def load_taxa_mapping(path: str | Path) -> dict[str, Any]:
    """Load the current three-scheme taxa grouping configuration."""
    try:
        mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaxaAnalysisPrepError(f"Could not load taxa mapping: {exc}") from exc
    if mapping.get("schema_version") != 3:
        raise TaxaAnalysisPrepError("Taxa mapping must use schema_version 3.")
    assets = mapping.get("clipart_assets", {})
    if not isinstance(assets, dict) or not assets:
        raise TaxaAnalysisPrepError("Taxa mapping must declare clipart_assets.")
    for field in ("sunburst_clipart", "sunburst_order_clipart"):
        figure_clipart = mapping.get(field, {})
        if not isinstance(figure_clipart, dict) or not figure_clipart:
            raise TaxaAnalysisPrepError(f"Taxa mapping must declare {field}.")
        missing_sunburst_assets = set(figure_clipart.values()).difference(assets)
        if missing_sunburst_assets:
            raise TaxaAnalysisPrepError(
                f"Taxa {field} references unknown assets: "
                f"{sorted(missing_sunburst_assets)}"
            )
    schemes = mapping.get("schemes", {})
    for name in ("broad", "analysis", "detail"):
        scheme = schemes.get(name, {})
        order = scheme.get("group_order", [])
        if not order or len(order) != len(set(order)):
            raise TaxaAnalysisPrepError(
                f"Taxa {name} group order must be non-empty and unique."
            )
        if scheme.get("unresolved_group") not in order:
            raise TaxaAnalysisPrepError(
                f"Taxa {name} group order must contain its unresolved state."
            )
        missing_group_assets = set(scheme.get("group_clipart", {}).values()).difference(
            assets
        )
        if missing_group_assets:
            raise TaxaAnalysisPrepError(
                f"Taxa {name} clipart references unknown assets: "
                f"{sorted(missing_group_assets)}"
            )
    return mapping


def grouping_rules_fingerprint(mapping: Mapping[str, Any]) -> str:
    """Hash grouping semantics while allowing figure colors to change freely."""
    semantics = {
        "schema_version": mapping.get("schema_version"),
        "eligible_match_statuses": mapping.get("eligible_match_statuses"),
        "not_applicable_values": mapping.get("not_applicable_values"),
        "schemes": {
            name: {
                "output_column": scheme.get("output_column"),
                "group_order": scheme.get("group_order"),
                "unresolved_group": scheme.get("unresolved_group"),
                "rules": scheme.get("rules"),
            }
            for name, scheme in mapping.get("schemes", {}).items()
        },
    }
    canonical = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_ut(frame: pd.DataFrame, source: str) -> None:
    if "UT" not in frame:
        raise TaxaAnalysisPrepError(f"{source} has no UT column.")
    missing = int(frame["UT"].isna().sum())
    duplicated = int(frame["UT"].duplicated().sum())
    if missing or duplicated:
        raise TaxaAnalysisPrepError(
            f"{source} violates one-row-per-UT grain: "
            f"missing={missing:,}, duplicated={duplicated:,}."
        )


def _parse_json_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _list_labels(value: Any) -> tuple[str, ...]:
    """Normalize CSV-serialized or Parquet-native list values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return tuple(parse_list_labels(value))


def _read_taxa_source(path: Path) -> pd.DataFrame:
    """Read only the publication-level taxa fields required by preparation."""
    try:
        if path.suffix.casefold() == ".parquet":
            return pd.read_parquet(path, columns=list(TAXA_SOURCE_COLUMNS))
        return pd.read_csv(
            path, usecols=list(TAXA_SOURCE_COLUMNS), low_memory=False
        )
    except (ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise TaxaAnalysisPrepError(
            "Taxa source lacks current lineage/group enrichment fields."
        ) from exc


def _ordered_labels(value: Any, order: Iterable[str], source: str) -> tuple[str, ...]:
    order = tuple(order)
    labels = _list_labels(value)
    unknown = set(labels).difference(order)
    if unknown:
        raise TaxaAnalysisPrepError(
            f"{source} contains unconfigured labels: {sorted(unknown)}"
        )
    observed = set(labels)
    return tuple(label for label in order if label in observed)


def _configured_labels(
    value: Any,
    order: Iterable[str],
    *,
    excluded: Iterable[str],
    source: str,
) -> tuple[str, ...]:
    order = tuple(order)
    excluded_casefold = {label.casefold() for label in excluded}
    labels = [
        label
        for label in _list_labels(value)
        if label.casefold() not in excluded_casefold
    ]
    unknown = set(labels).difference(order)
    if unknown:
        raise TaxaAnalysisPrepError(
            f"{source} contains unconfigured labels: {sorted(unknown)}"
        )
    observed = set(labels)
    return tuple(label for label in order if label in observed)


def _broad_inclusion_state(
    broad_groups: tuple[str, ...],
    llm_items: list[dict[str, Any]],
    biological_groups: set[str],
    unresolved_group: str,
) -> str:
    if biological_groups.intersection(broad_groups):
        return INCLUSION_ORDER[0]
    if unresolved_group in broad_groups:
        return INCLUSION_ORDER[4]
    names = {
        str(item.get("canonical_name", "")).strip().casefold()
        for item in llm_items
        if str(item.get("canonical_name", "")).strip()
    }
    if not llm_items:
        return INCLUSION_ORDER[6]
    if names and names.issubset(NOT_APPLICABLE_VALUES):
        return INCLUSION_ORDER[1]
    if names and names.issubset(UNCLEAR_VALUES):
        return INCLUSION_ORDER[2]
    if names and names.issubset(SPECIAL_EXCLUSION_VALUES):
        return INCLUSION_ORDER[3]
    return INCLUSION_ORDER[5]


def _scheme_state(
    all_groups: tuple[str, ...],
    biological_groups: tuple[str, ...],
    unresolved_group: str,
) -> str:
    if biological_groups:
        return "resolved"
    if unresolved_group in all_groups:
        return "unresolved_only"
    return "no_group"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "missing"


def write_taxa_matches(
    lineage_path: str | Path,
    output_path: str | Path,
    *,
    expected_uts: Iterable[str],
    batch_size: int = 131_072,
    output_row_group_size: int = 8_192,
) -> dict[str, Any]:
    """Stream the lossless long lineage into one nested match list per UT."""
    lineage_path = Path(lineage_path)
    output_path = Path(output_path)
    expected = [str(value) for value in expected_uts]
    if len(expected) != len(set(expected)):
        raise TaxaAnalysisPrepError("Expected taxa-match UTs are not unique.")
    if batch_size <= 0 or output_row_group_size <= 0:
        raise TaxaAnalysisPrepError("Taxa-match batch sizes must be positive.")

    parquet = pq.ParquetFile(lineage_path)
    required = {
        "UT",
        *TAXA_MATCH_METADATA_COLUMNS,
        *TAXA_LINEAGE_COLUMNS,
    }
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise TaxaAnalysisPrepError(
            f"Taxa lineage source lacks required columns: {sorted(missing)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".parquet",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    writer: pq.ParquetWriter | None = None
    output_uts: list[str] = []
    output_matches: list[list[dict[str, Any]]] = []
    expected_index = 0
    current_ut: str | None = None
    current_item_index: int | None = None
    current_metadata: dict[str, Any] | None = None
    current_lineage: list[dict[str, Any]] = []
    current_matches: list[dict[str, Any]] = []
    match_status_counts: Counter[str] = Counter()
    source_rows = 0
    lineage_rows = 0
    taxon_items = 0

    def flush_output() -> None:
        nonlocal writer, output_uts, output_matches
        if not output_uts:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(output_uts, type=pa.string()),
                pa.array(output_matches, type=pa.list_(TAXA_MATCH_STRUCT_TYPE)),
            ],
            schema=TAXA_MATCH_TABLE_SCHEMA,
        )
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_path,
                TAXA_MATCH_TABLE_SCHEMA,
                compression="zstd",
            )
        writer.write_table(table)
        output_uts = []
        output_matches = []

    def finish_item() -> None:
        nonlocal current_item_index, current_metadata, current_lineage, taxon_items
        if current_metadata is None:
            return
        record = dict(current_metadata)
        record["lineage"] = list(current_lineage)
        current_matches.append(record)
        match_status_counts[str(record.get("match_status") or "<missing>")] += 1
        taxon_items += 1
        current_item_index = None
        current_metadata = None
        current_lineage = []

    def finish_ut() -> None:
        nonlocal expected_index, current_matches
        if current_ut is None:
            return
        if expected_index >= len(expected) or current_ut != expected[expected_index]:
            wanted = expected[expected_index] if expected_index < len(expected) else None
            raise TaxaAnalysisPrepError(
                "Taxa lineage UT order differs from prepared publications: "
                f"position={expected_index:,}, expected={wanted!r}, observed={current_ut!r}."
            )
        output_uts.append(current_ut)
        output_matches.append(list(current_matches))
        expected_index += 1
        current_matches = []
        if len(output_uts) >= output_row_group_size:
            flush_output()

    columns = ["UT", *TAXA_MATCH_METADATA_COLUMNS, *TAXA_LINEAGE_COLUMNS]
    try:
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            values = batch.to_pydict()
            for row_index, raw_ut in enumerate(values["UT"]):
                source_rows += 1
                ut = str(raw_ut)
                item_index = values["llm_taxon_index"][row_index]
                if item_index is None:
                    raise TaxaAnalysisPrepError(
                        f"Taxa lineage row has no llm_taxon_index for UT {ut}."
                    )
                item_index = int(item_index)
                metadata = {
                    column: values[column][row_index]
                    for column in TAXA_MATCH_METADATA_COLUMNS
                }
                metadata["llm_taxon_index"] = item_index

                if current_ut is None:
                    current_ut = ut
                elif ut != current_ut:
                    finish_item()
                    finish_ut()
                    current_ut = ut

                if current_item_index is None:
                    if item_index != len(current_matches):
                        raise TaxaAnalysisPrepError(
                            "Taxa lineage items are not contiguous and zero-based for "
                            f"UT {ut}: expected={len(current_matches)}, observed={item_index}."
                        )
                    current_item_index = item_index
                    current_metadata = metadata
                elif item_index != current_item_index:
                    finish_item()
                    if item_index != len(current_matches):
                        raise TaxaAnalysisPrepError(
                            "Taxa lineage items are not contiguous for "
                            f"UT {ut}: expected={len(current_matches)}, observed={item_index}."
                        )
                    current_item_index = item_index
                    current_metadata = metadata
                elif metadata != current_metadata:
                    raise TaxaAnalysisPrepError(
                        f"Taxa match metadata changes within item {ut}/{item_index}."
                    )

                position = values["lineage_position"][row_index]
                lineage_values = {
                    "position": int(position) if position is not None else None,
                    "key": values["lineage_key"][row_index],
                    "rank": values["lineage_rank"][row_index],
                    "name": values["lineage_name"][row_index],
                }
                if position is None:
                    if any(
                        lineage_values[key] is not None
                        for key in ("key", "rank", "name")
                    ):
                        raise TaxaAnalysisPrepError(
                            f"Taxa lineage has values without a position for {ut}/{item_index}."
                        )
                else:
                    if int(position) != len(current_lineage):
                        raise TaxaAnalysisPrepError(
                            "Taxa lineage positions are not contiguous for "
                            f"{ut}/{item_index}: expected={len(current_lineage)}, "
                            f"observed={position}."
                        )
                    current_lineage.append(lineage_values)
                    lineage_rows += 1

        finish_item()
        finish_ut()
        flush_output()
        if writer is None:
            raise TaxaAnalysisPrepError("Taxa lineage source contains no rows.")
        writer.close()
        writer = None
        if expected_index != len(expected):
            raise TaxaAnalysisPrepError(
                "Taxa lineage source does not cover every prepared publication: "
                f"expected={len(expected):,}, observed={expected_index:,}."
            )
        os.replace(temporary_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "publications": expected_index,
        "unique_UT": expected_index,
        "taxon_items": taxon_items,
        "source_lineage_rows": source_rows,
        "resolved_lineage_rows": lineage_rows,
        "match_status_counts": dict(sorted(match_status_counts.items())),
    }


GROUP_SCHEMES = ("broad", "analysis", "detail")


def read_taxa_match_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load the nested prepared matches as ``{"UT", "taxa_matches"}`` rows."""
    table = pq.read_table(path)
    if not table.schema.equals(TAXA_MATCH_TABLE_SCHEMA, check_metadata=False):
        raise TaxaAnalysisPrepError(
            "Prepared taxa matches do not have the required nested schema."
        )
    return table.to_pylist()


def _iter_match_batches(
    match_source: str | Path | Iterable[dict[str, Any]], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    if isinstance(match_source, (str, Path)):
        parquet = pq.ParquetFile(match_source)
        if not parquet.schema_arrow.equals(TAXA_MATCH_TABLE_SCHEMA, check_metadata=False):
            raise TaxaAnalysisPrepError(
                "Prepared taxa matches do not have the required nested schema."
            )
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield batch.to_pylist()
        return
    rows = list(match_source)
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def regroup_prepared_taxa(
    articles: pd.DataFrame,
    match_source: str | Path | Iterable[dict[str, Any]],
    *,
    mapping_path: str | Path,
    output_path: str | Path | None = None,
    batch_size: int = 8_192,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]] | None,
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Re-apply taxon grouping rules to an existing prepared taxa bundle.

    Every taxon item in ``taxa_matches.parquet`` keeps its LLM rank and name, its
    GBIF match status, and the accepted GBIF lineage, which are the only inputs
    of ``labelling.taxa_groups.assign_taxon_groups``. Re-running the assignment
    from those fields therefore reproduces the labelling pipeline's grouping
    without the upstream ``taxa_all``/``taxa_lineages`` files or GBIF requests.

    Only grouping-dependent fields change: the item-level ``*_group*`` fields,
    the publication-level ``*_groups_all``/``*_groups``/``n_*_groups``/
    ``taxa_*_state`` columns, ``taxa_broad_inclusion_state``, and the item
    audits. ``match_source`` is a ``taxa_matches.parquet`` path (streamed in
    batches) or an iterable of ``{"UT", "taxa_matches"}`` rows. With
    ``output_path`` the regrouped matches are written there atomically and
    ``None`` is returned in their place; otherwise they are returned as rows.
    Returns ``(articles, match_rows, audits, match_summary)``; the audits and
    summary follow ``build_taxa_publications``/``write_taxa_matches``.
    """
    from labelling.taxa_groups import (
        assign_taxon_groups,
        load_group_config,
        order_groups,
    )

    mapping = load_taxa_mapping(mapping_path)
    group_config = load_group_config(mapping_path)
    expected_uts = articles["UT"].tolist()

    schemes = mapping["schemes"]
    broad_unresolved = schemes["broad"]["unresolved_group"]
    broad_biological = set(schemes["broad"]["group_order"]).difference(
        {broad_unresolved}
    )
    group_lists: dict[str, list[tuple[str, ...]]] = {
        scheme: [] for scheme in GROUP_SCHEMES
    }
    inclusion_states: list[str] = []
    regrouped_rows: list[dict[str, Any]] | None = [] if output_path is None else None
    global_status: Counter[str] = Counter()
    global_reason: Counter[str] = Counter()
    global_special: Counter[str] = Counter()
    match_status_counts: Counter[str] = Counter()
    taxon_items = 0
    source_lineage_rows = 0
    resolved_lineage_rows = 0
    position = 0

    writer: pq.ParquetWriter | None = None
    temporary: Path | None = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.stem}.regrouped.parquet")
        writer = pq.ParquetWriter(
            temporary, TAXA_MATCH_TABLE_SCHEMA, compression="zstd"
        )
    try:
        for batch in _iter_match_batches(match_source, batch_size):
            batch_rows: list[dict[str, Any]] = []
            for row in batch:
                if position >= len(expected_uts) or row["UT"] != expected_uts[position]:
                    raise TaxaAnalysisPrepError(
                        "Prepared taxa matches do not preserve publication UT order: "
                        f"position={position:,}."
                    )
                position += 1
                regrouped_row, row_groups, row_items, row_stats = _regroup_match_row(
                    row, group_config, assign_taxon_groups
                )
                batch_rows.append(regrouped_row)
                for scheme in GROUP_SCHEMES:
                    group_lists[scheme].append(
                        tuple(order_groups(row_groups[scheme], group_config, scheme=scheme))
                    )
                inclusion_states.append(
                    _broad_inclusion_state(
                        group_lists["broad"][-1],
                        row_items,
                        broad_biological,
                        broad_unresolved,
                    )
                )
                global_status.update(row_stats["status"])
                match_status_counts.update(row_stats["raw_status"])
                global_reason.update(row_stats["reason"])
                global_special.update(row_stats["special"])
                taxon_items += row_stats["items"]
                source_lineage_rows += row_stats["source_lineage_rows"]
                resolved_lineage_rows += row_stats["resolved_lineage_rows"]
            if writer is not None:
                writer.write_table(
                    pa.Table.from_pylist(batch_rows, schema=TAXA_MATCH_TABLE_SCHEMA)
                )
            else:
                regrouped_rows.extend(batch_rows)
        if position != len(expected_uts):
            raise TaxaAnalysisPrepError(
                "Prepared taxa matches do not cover every publication: "
                f"expected={len(expected_uts):,}, observed={position:,}."
            )
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, output_path)
    finally:
        if writer is not None:
            writer.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    articles = articles.copy()
    for scheme in GROUP_SCHEMES:
        unresolved = schemes[scheme]["unresolved_group"]
        all_groups = pd.Series(group_lists[scheme], index=articles.index, dtype=object)
        biological = all_groups.map(
            lambda labels, unresolved=unresolved: tuple(
                label for label in labels if label != unresolved
            )
        )
        articles[f"{scheme}_groups_all"] = all_groups
        articles[f"{scheme}_groups"] = biological
        articles[f"n_{scheme}_groups"] = biological.map(len)
        articles[f"taxa_{scheme}_state"] = [
            _scheme_state(labels, kept, unresolved)
            for labels, kept in zip(all_groups, biological)
        ]
    articles["taxa_broad_inclusion_state"] = inclusion_states
    return _regroup_results(
        articles,
        regrouped_rows,
        mapping=mapping,
        global_status=global_status,
        global_reason=global_reason,
        global_special=global_special,
        match_summary={
            "publications": position,
            "unique_UT": position,
            "taxon_items": taxon_items,
            "source_lineage_rows": source_lineage_rows,
            "resolved_lineage_rows": resolved_lineage_rows,
            "match_status_counts": dict(sorted(match_status_counts.items())),
        },
    )


def _regroup_match_row(
    row: Mapping[str, Any], group_config: Mapping[str, Any], assign_taxon_groups
) -> tuple[dict[str, Any], dict[str, list[str]], list[dict[str, Any]], dict[str, Any]]:
    """Regroup one publication's taxon items; return the row and its tallies."""
    item_groups: dict[str, list[str]] = {scheme: [] for scheme in GROUP_SCHEMES}
    publication_items: list[dict[str, Any]] = []
    regrouped_items: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "status": Counter(),
        "raw_status": Counter(),
        "reason": Counter(),
        "special": Counter(),
        "items": 0,
        "source_lineage_rows": 0,
        "resolved_lineage_rows": 0,
    }
    for item in row["taxa_matches"] or []:
        lineage = item.get("lineage") or []
        assignments = assign_taxon_groups(
            raw_rank=item.get("llm_taxon_rank") or "",
            raw_name=item.get("llm_canonical_name") or "",
            match_status=item.get("match_status"),
            classification=[
                {"rank": entry.get("rank"), "name": entry.get("name")}
                for entry in lineage
            ],
            config=group_config,
        )
        regrouped = dict(item)
        for scheme in GROUP_SCHEMES:
            assignment = assignments[scheme]
            regrouped[f"{scheme}_group"] = assignment.group
            regrouped[f"{scheme}_group_rule_id"] = assignment.rule_id
            regrouped[f"{scheme}_group_reason"] = assignment.reason
            if assignment.group is not None:
                item_groups[scheme].append(assignment.group)
        regrouped["broad_group_eligible"] = assignments["broad"].eligible
        regrouped_items.append(regrouped)
        publication_items.append({"canonical_name": item.get("llm_canonical_name")})

        status = str(item.get("match_status") or "<missing>").strip()
        stats["status"][status] += 1
        stats["raw_status"][str(item.get("match_status") or "<missing>")] += 1
        stats["reason"][str(assignments["broad"].reason or "<missing>").strip()] += 1
        if status == "special_value":
            name = str(item.get("llm_canonical_name") or "<missing>").strip()
            stats["special"][name] += 1
        stats["items"] += 1
        stats["source_lineage_rows"] += max(len(lineage), 1)
        stats["resolved_lineage_rows"] += len(lineage)
    regrouped_row = {"UT": row["UT"], "taxa_matches": regrouped_items}
    return regrouped_row, item_groups, publication_items, stats


def _regroup_results(
    articles: pd.DataFrame,
    regrouped_rows: list[dict[str, Any]] | None,
    *,
    mapping: Mapping[str, Any],
    global_status: Counter[str],
    global_reason: Counter[str],
    global_special: Counter[str],
    match_summary: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]] | None,
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Assemble item audits with the same layout as ``build_taxa_publications``."""
    total_items = sum(global_status.values())
    eligible_statuses = set(mapping.get("eligible_match_statuses", ()))
    audits = {
        "match_status": [
            {
                "match_status": status,
                "taxon_item_count": count,
                "share_of_taxon_items_pct": count / total_items * 100,
                "eligible_for_grouping": status in eligible_statuses,
            }
            for status, count in global_status.most_common()
        ],
        "broad_group_reason": [
            {
                "broad_group_reason": reason,
                "taxon_item_count": count,
                "share_of_taxon_items_pct": count / total_items * 100,
            }
            for reason, count in global_reason.most_common()
        ],
        "special_value": [
            {
                "special_value": value,
                "taxon_item_count": count,
                "share_of_special_values_pct": (
                    count / sum(global_special.values()) * 100
                    if global_special
                    else np.nan
                ),
                "article_exclusion_value": value.casefold()
                in SPECIAL_EXCLUSION_VALUES,
            }
            for value, count in global_special.most_common()
        ],
        "auxiliary_labels": _auxiliary_inventory(articles),
    }
    return articles, regrouped_rows, audits, match_summary


def write_taxa_match_rows(
    match_rows: list[dict[str, Any]], output_path: str | Path
) -> None:
    """Atomically write nested match rows with the prepared schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(match_rows, schema=TAXA_MATCH_TABLE_SCHEMA)
    temporary = output_path.with_name(f".{output_path.stem}.regrouped.parquet")
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=8_192)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _auxiliary_inventory(articles: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, column in (
        ("driver", "drivers"),
        ("threats_l0", "threat_l0"),
        ("realm", "realms"),
    ):
        counter: Counter[str] = Counter()
        for labels in articles[column]:
            counter.update(labels)
        rows.extend(
            {
                "source": source,
                "column": column,
                "label": label,
                "n_publications": count,
                "special_for_later_crosscuts": label.casefold()
                in SPECIAL_EXCLUSION_VALUES,
            }
            for label, count in counter.most_common()
        )
    return rows


def build_taxa_publications(
    corpus_df: pd.DataFrame,
    *,
    taxa_path: str | Path,
    mapping: Mapping[str, Any],
    driver_order: Iterable[str],
    threat_order: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Build the complete one-row-per-UT analysis-ready taxa table."""
    missing = set(CORPUS_COLUMNS).difference(corpus_df.columns)
    if missing:
        raise TaxaAnalysisPrepError(
            f"Merged corpus is missing required columns: {sorted(missing)}"
        )
    _validate_ut(corpus_df, "Eligible merged corpus")
    taxa_path = Path(taxa_path)
    taxa = _read_taxa_source(taxa_path)
    _validate_ut(taxa, "Taxa-with-API source")
    corpus_keys = pd.Index(corpus_df["UT"])
    taxa_keys = pd.Index(taxa["UT"])
    corpus_only = corpus_keys.difference(taxa_keys, sort=False)
    taxa_only = taxa_keys.difference(corpus_keys, sort=False)
    if len(corpus_only) or len(taxa_only):
        raise TaxaAnalysisPrepError(
            "Taxa and corpus UT sets differ: "
            f"corpus_only={len(corpus_only):,}, taxa_only={len(taxa_only):,}."
        )

    original_keys = corpus_df["UT"].reset_index(drop=True)
    joined = corpus_df[list(CORPUS_COLUMNS)].merge(
        taxa, on="UT", how="left", sort=False, validate="one_to_one"
    )
    if len(joined) != len(corpus_df) or not joined["UT"].reset_index(
        drop=True
    ).equals(original_keys):
        raise TaxaAnalysisPrepError("Joining taxa changed the UT grain or order.")

    schemes = mapping["schemes"]
    excluded = set(mapping.get("not_applicable_values", ())) | {
        "Not Applicable",
        "Unclear",
    }
    driver_order = tuple(driver_order)
    threat_order = tuple(threat_order)
    articles = joined[
        [
            "UT",
            "publication_year",
            "s2_dir",
            "pred_study_design",
            "taxa_record_status",
            "n_llm_taxa",
            "n_taxa_matched",
            "n_taxa_unresolved",
            "n_taxa_api_failed",
        ]
    ].copy()
    articles["publication_year"] = pd.to_numeric(
        articles["publication_year"], errors="coerce"
    ).astype("Int64")
    articles["drivers"] = joined["driver"].map(
        lambda value: _configured_labels(
            value,
            driver_order,
            excluded=excluded,
            source="Driver labels",
        )
    )
    articles["threat_l0"] = joined["pred_threat_l0"].map(
        lambda value: _configured_labels(
            value,
            threat_order,
            excluded=(),
            source="Threat-L0 labels",
        )
    )
    articles["realms"] = joined["realm"].map(
        lambda value: tuple(parse_list_labels(value))
    )
    class_exclusions = {
        str(label).strip().casefold()
        for label in mapping.get("not_applicable_values", ())
    }
    articles["class_labels"] = joined["class"].map(
        lambda value: tuple(
            label
            for label in _list_labels(value)
            if label.casefold() not in class_exclusions
        )
    )

    for name, source_column in (
        ("broad", "broad_taxa_groups"),
        ("analysis", "taxa_analysis_groups"),
        ("detail", "taxa_detail_groups"),
    ):
        scheme = schemes[name]
        order = tuple(scheme["group_order"])
        unresolved = scheme["unresolved_group"]
        all_column = f"{name}_groups_all"
        biological_column = f"{name}_groups"
        articles[all_column] = joined[source_column].map(
            lambda value, order=order, name=name: _ordered_labels(
                value, order, f"{name.title()} taxon groups"
            )
        )
        articles[biological_column] = articles[all_column].map(
            lambda labels, unresolved=unresolved: tuple(
                label for label in labels if label != unresolved
            )
        )
        articles[f"n_{name}_groups"] = articles[biological_column].map(len)
        articles[f"taxa_{name}_state"] = [
            _scheme_state(all_groups, biological_groups, unresolved)
            for all_groups, biological_groups in zip(
                articles[all_column], articles[biological_column]
            )
        ]

    broad_order = tuple(schemes["broad"]["group_order"])
    broad_unresolved = schemes["broad"]["unresolved_group"]
    broad_biological = set(broad_order).difference({broad_unresolved})
    llm_items = joined["llm_taxa_json"].map(_parse_json_items)
    articles["taxa_broad_inclusion_state"] = [
        _broad_inclusion_state(groups, items, broad_biological, broad_unresolved)
        for groups, items in zip(articles["broad_groups_all"], llm_items)
    ]

    global_status: Counter[str] = Counter()
    global_reason: Counter[str] = Counter()
    global_special: Counter[str] = Counter()
    per_article_status: list[Counter[str]] = []
    for value in joined["taxa_match_status_json"]:
        status_counts: Counter[str] = Counter()
        for item in _parse_json_items(value):
            status = str(item.get("status") or "<missing>").strip()
            status_counts[status] += 1
            global_status[status] += 1
            assignments = item.get("group_assignments")
            broad = (
                (assignments.get("broad") or {})
                if isinstance(assignments, dict)
                else {}
            )
            reason = str(broad.get("reason") or "<missing>").strip()
            global_reason[reason] += 1
            if status == "special_value":
                value_name = str(item.get("canonical_name") or "<missing>").strip()
                global_special[value_name] += 1
        per_article_status.append(status_counts)
    for status in global_status:
        articles[f"match_status_count__{_slug(status)}"] = [
            counts[status] for counts in per_article_status
        ]
    articles["n_drivers"] = articles["drivers"].map(len)
    for source_column, output_column in TAXONOMY_RANK_COLUMNS:
        if source_column == "class":
            articles[output_column] = articles["class_labels"]
        else:
            articles[output_column] = joined[source_column].map(_list_labels)

    source_audit = [
        {
            "source": "eligible merged corpus",
            "rows": len(corpus_df),
            "unique_UT": corpus_df["UT"].nunique(),
            "missing_UT": int(corpus_df["UT"].isna().sum()),
            "duplicated_UT": int(corpus_df["UT"].duplicated().sum()),
            "grain": "one row per UT",
        },
        {
            "source": "taxa-with-api",
            "rows": len(taxa),
            "unique_UT": taxa["UT"].nunique(),
            "missing_UT": int(taxa["UT"].isna().sum()),
            "duplicated_UT": int(taxa["UT"].duplicated().sum()),
            "grain": "one row per UT",
        },
    ]
    total_items = sum(global_status.values())
    eligible_statuses = set(mapping.get("eligible_match_statuses", ()))
    audits = {
        "source_grain": source_audit,
        "match_status": [
            {
                "match_status": status,
                "taxon_item_count": count,
                "share_of_taxon_items_pct": count / total_items * 100,
                "eligible_for_grouping": status in eligible_statuses,
            }
            for status, count in global_status.most_common()
        ],
        "broad_group_reason": [
            {
                "broad_group_reason": reason,
                "taxon_item_count": count,
                "share_of_taxon_items_pct": count / total_items * 100,
            }
            for reason, count in global_reason.most_common()
        ],
        "special_value": [
            {
                "special_value": value,
                "taxon_item_count": count,
                "share_of_special_values_pct": (
                    count / sum(global_special.values()) * 100
                    if global_special
                    else np.nan
                ),
                "article_exclusion_value": value.casefold()
                in SPECIAL_EXCLUSION_VALUES,
            }
            for value, count in global_special.most_common()
        ],
        "auxiliary_labels": _auxiliary_inventory(articles),
    }
    return articles, audits


def build_gbif_broad_benchmark(
    gbif_path: str | Path,
    mapping_path: str | Path,
) -> DescribedDiversity:
    """Build the accepted species-rank broad benchmark once for later results."""
    from data_helpers.analysis.taxa import benchmark as taxa_driver

    broad_mapping = taxa_driver.load_broad_group_mapping(mapping_path)
    return taxa_driver.build_described_diversity_benchmark(
        gbif_path, broad_mapping
    )


def build_manifest(
    articles: pd.DataFrame,
    benchmark: DescribedDiversity,
    *,
    mapping: Mapping[str, Any],
    audits: Mapping[str, list[dict[str, Any]]],
    match_summary: Mapping[str, Any],
    sources: Mapping[str, str | Path],
    repository_root: str | Path,
) -> dict[str, Any]:
    """Create compact processing provenance and reconciliation audits."""
    if len(articles) != articles["UT"].nunique():
        raise TaxaAnalysisPrepError("Prepared publications are not one row per UT.")
    scheme_audits: dict[str, Any] = {}
    for name in ("broad", "analysis", "detail"):
        state_counts = articles[f"taxa_{name}_state"].value_counts()
        scheme_audits[name] = {
            "group_order": mapping["schemes"][name]["group_order"],
            "state_counts": {key: int(value) for key, value in state_counts.items()},
            "resolved_publications": int(articles[f"n_{name}_groups"].gt(0).sum()),
        }
    broad_inclusion = articles["taxa_broad_inclusion_state"].value_counts()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grouping_rules_sha256": grouping_rules_fingerprint(mapping),
        "sources": {
            name: source_signature(path, repository_root=repository_root)
            for name, path in sources.items()
        },
        "artifacts": {
            "taxa_publications": "taxa_publications.parquet",
            "taxa_matches": "taxa_matches.parquet",
            "gbif_broad_benchmark": "gbif_broad_benchmark.csv",
        },
        "rows": {
            "taxa_publications": len(articles),
            "unique_UT": articles["UT"].nunique(),
            "taxa_matches": int(match_summary["publications"]),
            "taxon_items": int(match_summary["taxon_items"]),
            "gbif_benchmark_groups": len(benchmark.counts),
        },
        "taxonomy_rank_columns": [
            output for _, output in TAXONOMY_RANK_COLUMNS
        ],
        "taxa_matches_audit": dict(match_summary),
        "group_schemes": scheme_audits,
        "broad_inclusion_counts": {
            state: int(broad_inclusion.get(state, 0)) for state in INCLUSION_ORDER
        },
        "benchmark_audit": benchmark.audit.to_dict(orient="records"),
        "audits": dict(audits),
    }


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Convert pandas/numpy scalar values used in compact audit records."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class TaxaPreparedStore:
    """Write and validate the four-file prepared taxa bundle."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.article_path = self.root / "taxa_publications.parquet"
        self.match_path = self.root / "taxa_matches.parquet"
        self.benchmark_path = self.root / "gbif_broad_benchmark.csv"
        self.manifest_path = self.root / "manifest.json"

    def write_matches(
        self,
        lineage_path: str | Path,
        *,
        expected_uts: Iterable[str],
    ) -> dict[str, Any]:
        """Build the nested, lossless match artifact without loading it in memory."""
        self.root.mkdir(parents=True, exist_ok=True)
        return write_taxa_matches(
            lineage_path,
            self.match_path,
            expected_uts=expected_uts,
        )

    def write(self, bundle: TaxaPreparedBundle) -> list[Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.match_path.exists():
            raise TaxaAnalysisPrepError(
                "Prepared taxa matches are missing; call write_matches first."
            )
        bundle.articles.to_parquet(
            self.article_path, index=False, compression="zstd"
        )
        bundle.benchmark.counts.to_csv(self.benchmark_path, index=False)
        self.manifest_path.write_text(
            json.dumps(
                bundle.manifest,
                indent=2,
                sort_keys=True,
                default=_json_default,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return [
            self.article_path,
            self.match_path,
            self.benchmark_path,
            self.manifest_path,
        ]

    def rebuild_with_rules(
        self,
        *,
        mapping_path: str | Path,
        gbif_path: str | Path,
        gbif_eml_path: str | Path,
        repository_root: str | Path,
    ) -> list[Path]:
        """Re-apply the current grouping rules to this bundle in place.

        Used when the upstream taxa coding files are unavailable: the grouping
        inputs are all stored in ``taxa_matches.parquet`` (see
        ``regroup_prepared_taxa``). The described-species benchmark is rebuilt
        from ``gbif_path`` under the same rules. The manifest keeps the previous
        build's upstream sources and records what was regrouped from.
        """
        mapping = load_taxa_mapping(mapping_path)
        previous = self.load(mapping=mapping, allow_stale_rules=True)
        previous_artifacts = {
            path.name: {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (self.article_path, self.match_path, self.manifest_path)
        }
        benchmark = build_gbif_broad_benchmark(gbif_path, mapping_path)
        staging = self.root / ".taxa_matches.regrouped-staging.parquet"
        try:
            articles, _, audits, match_summary = regroup_prepared_taxa(
                previous.articles,
                self.match_path,
                mapping_path=mapping_path,
                output_path=staging,
            )
            audits = {"source_grain": previous.manifest["audits"]["source_grain"], **audits}
            manifest = build_manifest(
                articles,
                benchmark,
                mapping=mapping,
                audits=audits,
                match_summary=match_summary,
                sources={
                    "gbif": gbif_path,
                    "gbif_eml": gbif_eml_path,
                    "taxa_group_mapping": mapping_path,
                },
                repository_root=repository_root,
            )
            manifest["upstream_sources"] = previous.manifest.get(
                "upstream_sources", previous.manifest["sources"]
            )
            manifest["regrouped_from"] = {
                "method": (
                    "Grouping rules re-applied to the stored LLM taxon names, GBIF "
                    "match statuses, and accepted GBIF lineages in taxa_matches.parquet "
                    "(regroup_prepared_taxa); non-grouping fields carried over unchanged."
                ),
                "previous_created_at_utc": previous.manifest.get("created_at_utc"),
                "previous_grouping_rules_sha256": previous.manifest.get(
                    "grouping_rules_sha256"
                ),
                "previous_artifacts": previous_artifacts,
            }
            os.replace(staging, self.match_path)
        finally:
            staging.unlink(missing_ok=True)
        written = self.write(TaxaPreparedBundle(articles, benchmark, manifest))
        self.load(mapping=mapping)  # round-trip verification
        return written

    def load(
        self,
        *,
        mapping: Mapping[str, Any],
        allow_stale_rules: bool = False,
    ) -> TaxaPreparedBundle:
        missing = [
            path
            for path in (
                self.article_path,
                self.match_path,
                self.benchmark_path,
                self.manifest_path,
            )
            if not path.exists()
        ]
        if missing:
            raise TaxaAnalysisPrepError(
                "Prepared taxa bundle is missing. Run "
                "notebooks/data_processing/02_taxa_analysis_prep.ipynb. "
                f"Missing: {[str(path) for path in missing]}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise TaxaAnalysisPrepError(
                "Prepared taxa schema is incompatible; rebuild the prep notebook."
            )
        expected_fingerprint = grouping_rules_fingerprint(mapping)
        if (
            not allow_stale_rules
            and manifest.get("grouping_rules_sha256") != expected_fingerprint
        ):
            raise TaxaAnalysisPrepError(
                "Prepared taxa grouping rules are stale; rebuild the prep notebook."
            )
        articles = pd.read_parquet(self.article_path)
        for column in LIST_COLUMNS:
            articles[column] = articles[column].map(
                lambda value: tuple(value) if value is not None else ()
            )
        _validate_ut(articles, "Prepared taxa publications")
        expected_rows = manifest["rows"]["taxa_publications"]
        if len(articles) != expected_rows:
            raise TaxaAnalysisPrepError(
                "Prepared taxa row count does not match manifest: "
                f"artifact={len(articles):,}, manifest={expected_rows:,}."
            )
        expected_unique = manifest["rows"]["unique_UT"]
        if articles["UT"].nunique() != expected_unique:
            raise TaxaAnalysisPrepError(
                "Prepared taxa unique-UT count does not match the manifest."
            )
        match_file = pq.ParquetFile(self.match_path)
        if not match_file.schema_arrow.equals(
            TAXA_MATCH_TABLE_SCHEMA, check_metadata=False
        ):
            raise TaxaAnalysisPrepError(
                "Prepared taxa matches do not have the required nested schema."
            )
        if match_file.metadata.num_rows != manifest["rows"]["taxa_matches"]:
            raise TaxaAnalysisPrepError(
                "Prepared taxa-match row count does not match the manifest."
            )
        match_uts = pd.read_parquet(self.match_path, columns=["UT"])["UT"]
        if not match_uts.reset_index(drop=True).equals(
            articles["UT"].reset_index(drop=True)
        ):
            raise TaxaAnalysisPrepError(
                "Prepared taxa matches do not preserve publication UT order."
            )
        counts = pd.read_csv(self.benchmark_path)
        required_benchmark = {
            "broad_group",
            "described_species_count",
            "described_species_share",
            "described_species_share_pct",
        }
        missing_benchmark = required_benchmark.difference(counts.columns)
        duplicated_groups = (
            int(counts["broad_group"].duplicated().sum())
            if "broad_group" in counts
            else 0
        )
        if missing_benchmark or duplicated_groups:
            raise TaxaAnalysisPrepError(
                "Prepared GBIF benchmark violates its column/group contract: "
                f"missing={sorted(missing_benchmark)}, "
                f"duplicated_groups={duplicated_groups}."
            )
        expected_groups = manifest["rows"]["gbif_benchmark_groups"]
        if len(counts) != expected_groups:
            raise TaxaAnalysisPrepError(
                "Prepared GBIF benchmark group count does not match the manifest."
            )
        benchmark = DescribedDiversity(
            counts=counts,
            audit=pd.DataFrame(manifest.get("benchmark_audit", [])),
        )
        return TaxaPreparedBundle(
            articles=articles,
            benchmark=benchmark,
            manifest=manifest,
        )


__all__ = [
    "INCLUSION_ORDER",
    "TaxaAnalysisPrepError",
    "TaxaPreparedBundle",
    "TaxaPreparedStore",
    "TAXA_MATCH_TABLE_SCHEMA",
    "TAXONOMY_RANK_COLUMNS",
    "build_gbif_broad_benchmark",
    "build_manifest",
    "build_taxa_publications",
    "grouping_rules_fingerprint",
    "load_taxa_mapping",
    "read_taxa_match_rows",
    "regroup_prepared_taxa",
    "write_taxa_match_rows",
    "write_taxa_matches",
]

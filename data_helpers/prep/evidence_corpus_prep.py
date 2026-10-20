"""Prepare reusable screening and eligible biodiversity-evidence handoffs.

The full screening grain and the eligible evidence grain are deliberately
separate. Screening results use compact aggregates over every screened record;
substantive results use a one-row-per-publication corpus with a stable numeric
primary key, the external UT, configured coding dimensions, and a text sidecar.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from pandas.api.types import is_bool_dtype, is_integer_dtype

from data_helpers.labels import parse_list_labels
from data_helpers.prep._provenance import repository_relative_path, source_signature


SCREENING_SCHEMA_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 18
DEFAULT_PLASTICS_PATTERN = (
    r"\b(?:micro[\s-]?plastics?|nano[\s-]?plastics?|plastics?|"
    r"plastic[\s-](?:debris|waste|litter|particles?|fibres?|fibers?|"
    r"pellets?|fragments?|pollution)|marine[\s-](?:debris|litter)|"
    r"anthropogenic[\s-]litter)\b"
)
REASON_COLUMNS = ("s1_r", "s2_r", "s3_r", "s4_r")
STAGE_NAMES = {
    "s1_r": "Q1: biodiversity change",
    "s2_r": "Q2: direction of change",
    "s3_r": "Q3: anthropogenic driver",
    "s4_r": "Q4: driver-change linkage",
}
SCREENING_ANALYSIS_COLUMNS = (
    "UT",
    "publication_year",
    "source",
    "is_eligible",
    "eligibility",
    *REASON_COLUMNS,
    "s2_dir",
)
GEOGRAPHY_AUDIT_LABEL_COLUMNS = (
    "pred_regions_audit",
    "pred_subregions_audit",
    "pred_countries_audit",
)
EVIDENCE_LIST_COLUMNS = (
    "driver",
    "pred_threat_l0",
    "pred_regions",
    "pred_subregions",
    "pred_countries",
    *GEOGRAPHY_AUDIT_LABEL_COLUMNS,
    "locales",
    "realm",
    "biome",
    "pred_study_design",
    "pred_methods_data_collection",
    "pred_methods_analysis",
    "pred_comparison_types",
    "taxa_kingdom_labels",
    "taxa_phylum_labels",
    "taxa_class_labels",
    "taxa_order_labels",
    "taxa_family_labels",
    "taxa_genus_labels",
    "taxa_species_labels",
)
GEOGRAPHY_LIST_COLUMNS = (
    "pred_regions",
    "pred_subregions",
    "pred_countries",
    "locales",
)
GEOGRAPHY_VALUE_COLUMNS = (*GEOGRAPHY_LIST_COLUMNS, "locale_coordinates")
_UNCLEAR_PRECEDENCE_COLUMNS = {
    "pred_regions",
    "pred_subregions",
    "pred_countries",
}
_NOT_APPLICABLE_LABEL = "Not Applicable"
_UNCLEAR_LABEL = "Unclear"
_NOT_APPLICABLE_TOKENS = {
    "NA",
    "NONE",
    "NOTAPPLICABLE",
}
_UNCLEAR_TOKENS = {
    "UNCLEAR",
    "UNCLEARREVIEWNEEDED",
    "UNC",
    "UNCL",
    "UNK",
    "UNKN",
    "UNKNOWN",
    "UNS",
    "UNR",
    "UNL",
    "UNP",
}
_GLOBAL_COUNTRY_TOKENS = {
    "ALLCOUNTRIES",
    "GLOBAL",
    "WORLDWIDE",
}
_WOS_UT_PATTERN = re.compile(r"^WOS:\d{15}$")
_EXCEL_MAX_EXACT_INTEGER = 9_007_199_254_740_991
_EVIDENCE_METADATA_BEFORE_AUDIT = (
    "UT",
    "title",
    "publication_year",
    "doi",
    "s1_r",
    "s2_r",
    "s3_r",
    "s4_r",
    "s1_bio",
    "s2_dir",
    "s3_drivers",
    "s4_link",
    "driver",
    "pred_threat_l0",
    "pred_regions",
    "pred_subregions",
    "pred_countries",
)
_EVIDENCE_METADATA_AFTER_AUDIT = (
    "locales",
    "locale_coordinates",
    "realm",
    "biome",
    "pred_study_design",
    "pred_methods_data_collection",
    "pred_methods_analysis",
    "pred_has_comparison",
    "pred_comparison_types",
)
EVIDENCE_METADATA_COLUMNS = (
    *_EVIDENCE_METADATA_BEFORE_AUDIT,
    *_EVIDENCE_METADATA_AFTER_AUDIT,
)
TAXONOMY_RANK_COLUMNS = (
    "taxa_kingdom_labels",
    "taxa_phylum_labels",
    "taxa_class_labels",
    "taxa_order_labels",
    "taxa_family_labels",
    "taxa_genus_labels",
    "taxa_species_labels",
)
EVIDENCE_COLUMNS = (
    "id",
    *_EVIDENCE_METADATA_BEFORE_AUDIT,
    *GEOGRAPHY_AUDIT_LABEL_COLUMNS,
    *_EVIDENCE_METADATA_AFTER_AUDIT,
    *TAXONOMY_RANK_COLUMNS,
    "taxa_record_status",
    "text_available",
    "plastics_mention",
)
EVIDENCE_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "id": "Stable integer primary key: the integer value of the 15-digit WOS UT suffix.",
    "UT": "External Web of Science record identifier (source key); format WOS:<15 digits>.",
    "title": "Article title, as retrieved from Web of Science.",
    "publication_year": "Year of publication, as retrieved from Web of Science.",
    "doi": "Digital Object Identifier, as retrieved from Web of Science.",
    "s1_r": "Q1 screening result: biodiversity change reported? 1 = Yes, -1 = Unclear, 0 = No.",
    "s2_r": "Q2 screening result: direction of change identifiable? 1 = Yes, -1 = Unclear, 0 = No.",
    "s3_r": "Q3 screening result: direct anthropogenic driver mentioned or plausibly implied? 1 = Yes, -1 = Unclear, 0 = No.",
    "s4_r": "Q4 screening result: driver-biodiversity linkage reported or plausibly implied? 1 = Yes, -1 = Unclear, 0 = No.",
    "s1_bio": "Q1 biodiversity change type(s): genetic, species, community, and/or ecosystem change.",
    "s2_dir": "Q2 direction of biodiversity change: negative, positive, mixed, or unclear (eligible records exclude 'none').",
    "s3_drivers": "Q3 driver or threat name(s) identified during screening; supports eligibility only, not the final L1/L2 coding (see driver, pred_threat_l0).",
    "s4_link": "Q4 linkage type(s) describing how the identified driver is linked to the reported biodiversity change.",
    "driver": "IPBES direct driver category or categories assigned to the record (L1 coding).",
    "pred_threat_l0": "Broadest IUCN Threats Classification level assigned to the record (L2 coding).",
    "pred_regions": "IPBES region or regions of study focus, as originally coded (L3).",
    "pred_subregions": "IPBES sub-region or sub-regions of study focus, as originally coded (L3).",
    "pred_countries": "Countries of study focus (ISO 3166-1 alpha-3 codes), as originally coded (L3).",
    "pred_regions_audit": "Reviewed region label(s): equals pred_regions except review-flagged records, which become ['Unclear - Review needed'] per the IPBES hierarchy audit.",
    "pred_subregions_audit": "Reviewed sub-region label(s); same review-flag convention as pred_regions_audit.",
    "pred_countries_audit": "Reviewed country label(s); same review-flag convention as pred_regions_audit.",
    "locales": "Free-text names of sub-national locations mentioned in the title or abstract, such as parks, reserves, cities, or mountain ranges.",
    "locale_coordinates": "[latitude, longitude] pairs (WGS84, EPSG:4326) for the named locales, where resolvable.",
    "realm": "Global Ecosystem Typology realm or realms of study focus (L4 coding).",
    "biome": "Global Ecosystem Typology biome or biomes of study focus (L4 coding).",
    "pred_study_design": "Study design category: Observational, Experimental, Modelling, Review, or Unclear.",
    "pred_methods_data_collection": "Data collection method(s) reported in the title or abstract, where applicable.",
    "pred_methods_analysis": "Analysis method(s) reported in the title or abstract, where applicable.",
    "pred_has_comparison": "True if the record explicitly states or strongly implies a comparator (e.g. pre/post, control vs. exposed, reference sites); otherwise false.",
    "pred_comparison_types": "Type or types of comparison identified for the record, populated when pred_has_comparison is true.",
    "taxa_kingdom_labels": "Kingdom(s) resolved through the GBIF Backbone Taxonomy from the article's LLM-extracted taxa mentions (L6).",
    "taxa_phylum_labels": "Phylum/phyla resolved through the GBIF Backbone Taxonomy (L6).",
    "taxa_class_labels": "Class(es) resolved through the GBIF Backbone Taxonomy (L6).",
    "taxa_order_labels": "Order(s) resolved through the GBIF Backbone Taxonomy (L6).",
    "taxa_family_labels": "Family/families resolved through the GBIF Backbone Taxonomy lineage of matched taxa; a derived rank, not directly named by the LLM coder (unlike the other taxa_*_labels ranks).",
    "taxa_genus_labels": "Genus/genera resolved through the GBIF Backbone Taxonomy (L6).",
    "taxa_species_labels": "Species resolved through the GBIF Backbone Taxonomy (L6).",
    "taxa_record_status": "Per-publication GBIF-resolution status of the LLM-extracted taxa mentions: resolved, partly_resolved, or unresolved.",
    "text_available": "True if usable title/abstract text was available for the plastics/marine-debris text-mention check.",
    "plastics_mention": "True if the title or abstract mentions plastics or marine-debris terminology (see DEFAULT_PLASTICS_PATTERN).",
}
GEOGRAPHY_HIERARCHY_AUDIT_COLUMNS = (
    "UT",
    "status",
    "primary_reason",
    "reasons",
    "warnings",
    "pred_regions",
    "pred_subregions",
    "pred_countries",
)
class EvidenceCorpusPrepError(ValueError):
    """Raised when a prepared screening/corpus artifact violates its grain."""


def _normalise_geography_token(value: object) -> str:
    """Normalize a special label or coded country value for comparison."""
    return re.sub(r"[^A-Za-z]", "", str(value)).upper()


def standardize_geography_lists(
    publications: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize existing geography lists in place without changing corpus grain.

    Empty region, subregion, country, and locale lists become
    ``("Not Applicable",)``. Empty locale-coordinate JSON lists become the
    type-preserving JSON text ``["Not Applicable"]``. Any region, subregion, or
    country list containing an ``Unclear`` label becomes exactly ``("Unclear",)``.
    Exact ``All Regions`` records whose descendants are already non-concrete are
    canonicalized to ``Not Applicable`` descendants. Other non-empty values are
    preserved for the separate IPBES hierarchy audit.

    Returns a small audit of changed cells; no corpus rows or columns are added,
    removed, or reordered.
    """
    missing = set(GEOGRAPHY_VALUE_COLUMNS).difference(publications.columns)
    if missing:
        raise EvidenceCorpusPrepError(
            f"Integrated corpus lacks geography columns: {sorted(missing)}"
        )

    original_columns = list(publications.columns)
    original_uts = publications["UT"].reset_index(drop=True).copy()
    audit_counts: Counter[tuple[str, str]] = Counter()

    for column in GEOGRAPHY_LIST_COLUMNS:
        standardized: list[tuple[str, ...]] = []
        for value in publications[column]:
            labels = tuple(parse_list_labels(value))
            reason: str | None = None
            if not labels:
                replacement = (_NOT_APPLICABLE_LABEL,)
                reason = "empty_to_not_applicable"
            elif column in _UNCLEAR_PRECEDENCE_COLUMNS and any(
                label.casefold() == "unclear" for label in labels
            ):
                replacement = (_UNCLEAR_LABEL,)
                reason = "contains_unclear_to_unclear"
            else:
                replacement = labels

            standardized.append(replacement)
            if reason is not None and replacement != labels:
                audit_counts[(column, reason)] += 1
        publications[column] = standardized

    coordinate_column = "locale_coordinates"
    standardized_coordinates: list[object] = []
    for value in publications[coordinate_column]:
        replacement = value
        changed = False
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, list) and not parsed:
                replacement = json.dumps([_NOT_APPLICABLE_LABEL])
                changed = True
        elif isinstance(value, (list, tuple)) and not value:
            replacement = (_NOT_APPLICABLE_LABEL,)
            changed = True
        standardized_coordinates.append(replacement)
        if changed:
            audit_counts[(coordinate_column, "empty_to_not_applicable")] += 1
    publications[coordinate_column] = standardized_coordinates

    regions = publications["pred_regions"]
    subregions = publications["pred_subregions"]
    countries = publications["pred_countries"]
    for index in publications.index[
        regions.map(lambda labels: labels == ("All Regions",))
    ]:
        subregion_state, _ = _hierarchy_level_state(
            subregions.at[index], "subregion"
        )
        country_state, _ = _hierarchy_level_state(countries.at[index], "country")
        non_concrete = {"empty", "not_applicable", "unclear", "global"}
        if subregion_state not in non_concrete or country_state not in non_concrete:
            continue
        for column in ("pred_subregions", "pred_countries"):
            if publications.at[index, column] != (_NOT_APPLICABLE_LABEL,):
                publications.at[index, column] = (_NOT_APPLICABLE_LABEL,)
                audit_counts[
                    (column, "all_regions_descendant_to_not_applicable")
                ] += 1

    if len(publications) != len(original_uts):
        raise EvidenceCorpusPrepError("Geography standardization changed row count.")
    if list(publications.columns) != original_columns:
        raise EvidenceCorpusPrepError("Geography standardization changed columns.")
    if not publications["UT"].reset_index(drop=True).equals(original_uts):
        raise EvidenceCorpusPrepError("Geography standardization changed UT order.")

    return pd.DataFrame(
        [
            {"column": column, "rule": reason, "changed_values": count}
            for (column, reason), count in sorted(audit_counts.items())
        ],
        columns=["column", "rule", "changed_values"],
    )


def _hierarchy_level_state(
    values: object,
    level: str,
) -> tuple[str, set[str]]:
    """Classify one hierarchy level and return its concrete values."""
    labels = tuple(parse_list_labels(values))
    if not labels:
        return "empty", set()

    kinds: set[str] = set()
    concrete: set[str] = set()
    for label in labels:
        token = _normalise_geography_token(label)
        if token in _NOT_APPLICABLE_TOKENS:
            kinds.add("not_applicable")
        elif token in _UNCLEAR_TOKENS:
            kinds.add("unclear")
        elif level == "region" and token == "ALLREGIONS":
            kinds.add("global")
        elif level == "subregion" and token == "ALLSUBREGIONS":
            kinds.add("global")
        elif level == "country" and token in _GLOBAL_COUNTRY_TOKENS:
            kinds.add("global")
        else:
            kinds.add("concrete")
            concrete.add(token if level == "country" else label)
    if len(kinds) > 1:
        return "mixed", concrete
    return next(iter(kinds)), concrete


def _load_ipbes_hierarchy(
    mapping_path: str | Path,
) -> tuple[set[str], dict[str, str], dict[str, tuple[str, str]]]:
    """Load and validate the unique IPBES region/subregion/country paths."""
    path = Path(mapping_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise EvidenceCorpusPrepError(
            f"IPBES hierarchy at {path} must be a non-empty object."
        )

    regions: set[str] = set()
    subregion_parent: dict[str, str] = {}
    country_path: dict[str, tuple[str, str]] = {}
    for region, subregions in payload.items():
        if not isinstance(region, str) or not isinstance(subregions, dict):
            raise EvidenceCorpusPrepError(
                f"IPBES hierarchy at {path} has an invalid region entry."
            )
        regions.add(region)
        for subregion, records in subregions.items():
            previous_region = subregion_parent.setdefault(subregion, region)
            if previous_region != region:
                raise EvidenceCorpusPrepError(
                    f"IPBES subregion {subregion!r} has multiple parent regions."
                )
            if not isinstance(records, list):
                raise EvidenceCorpusPrepError(
                    f"IPBES subregion {subregion!r} must contain a records list."
                )
            for record in records:
                if not isinstance(record, dict):
                    raise EvidenceCorpusPrepError(
                        f"IPBES subregion {subregion!r} contains a non-object record."
                    )
                code = _normalise_geography_token(
                    record.get("ISO_3166_alpha_3", "")
                )
                if not code:
                    continue
                previous_path = country_path.setdefault(code, (region, subregion))
                if previous_path != (region, subregion):
                    raise EvidenceCorpusPrepError(
                        f"IPBES country {code!r} has multiple hierarchy paths."
                    )
    if not regions or not subregion_parent or not country_path:
        raise EvidenceCorpusPrepError(
            f"IPBES hierarchy at {path} has no complete country paths."
        )
    return regions, subregion_parent, country_path


def audit_ipbes_geography_hierarchy(
    publications: pd.DataFrame,
    *,
    mapping_path: str | Path,
) -> pd.DataFrame:
    """Audit hierarchy paths without altering publication geography labels."""
    required = {"UT", "pred_regions", "pred_subregions", "pred_countries"}
    missing = required.difference(publications.columns)
    if missing:
        raise EvidenceCorpusPrepError(
            f"Integrated corpus lacks hierarchy columns: {sorted(missing)}"
        )
    _validate_key(publications, "Integrated biodiversity evidence corpus")
    regions, subregion_parent, country_path = _load_ipbes_hierarchy(mapping_path)
    subregions = set(subregion_parent)
    countries = set(country_path)

    statuses: list[str] = []
    primary_reasons: list[str] = []
    all_reasons: list[tuple[str, ...]] = []
    all_warnings: list[tuple[str, ...]] = []

    for row in publications[
        ["pred_regions", "pred_subregions", "pred_countries"]
    ].itertuples(index=False, name=None):
        region_values, subregion_values, country_values = row
        region_state, concrete_regions = _hierarchy_level_state(
            region_values, "region"
        )
        subregion_state, concrete_subregions = _hierarchy_level_state(
            subregion_values, "subregion"
        )
        country_state, concrete_countries = _hierarchy_level_state(
            country_values, "country"
        )
        reasons: list[str] = []
        warnings: list[str] = []

        def add_reason(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        if "mixed" in {region_state, subregion_state, country_state}:
            add_reason("mixed_special_and_concrete_or_global")

        intended_status = "review"
        pass_reason = ""
        if region_state == "global":
            if subregion_state in {"concrete", "mixed"} or country_state in {
                "concrete",
                "mixed",
            }:
                add_reason("all_regions_with_concrete_descendant")
            elif not reasons:
                intended_status = "global"
                pass_reason = "all_regions_descendants_not_applicable"
        elif region_state == "concrete":
            if concrete_regions.difference(regions):
                add_reason("unknown_region_label")
            if subregion_state == "concrete":
                known_subregions = concrete_subregions.intersection(subregions)
                if concrete_subregions.difference(subregions):
                    add_reason("unknown_subregion_label")
                if any(
                    subregion_parent[subregion] not in concrete_regions
                    for subregion in known_subregions
                ):
                    add_reason("subregion_parent_missing")

                if country_state == "concrete":
                    known_countries = concrete_countries.intersection(countries)
                    if concrete_countries.difference(countries):
                        add_reason("unknown_country_label")
                    if any(
                        country_path[country][0] not in concrete_regions
                        or country_path[country][1] not in concrete_subregions
                        for country in known_countries
                    ):
                        add_reason("country_path_missing")
                    if not reasons:
                        intended_status = "pass_complete"
                        pass_reason = "complete_paths_valid"
                        if any(
                            not any(
                                subregion_parent[subregion] == region
                                for subregion in concrete_subregions
                            )
                            for region in concrete_regions
                        ):
                            warnings.append("region_without_listed_subregion")
                        if any(
                            not any(
                                country_path[country][1] == subregion
                                for country in concrete_countries
                            )
                            for subregion in concrete_subregions
                        ):
                            warnings.append("subregion_without_listed_country")
                elif country_state != "mixed" and not reasons:
                    intended_status = "pass_partial"
                    pass_reason = "region_subregion_valid_country_non_concrete"
            elif country_state == "concrete":
                add_reason("country_present_subregion_non_concrete")
            elif subregion_state != "mixed" and not reasons:
                intended_status = "pass_region_only"
                pass_reason = "region_valid_descendants_non_concrete"
        elif subregion_state in {"concrete", "mixed"} or country_state in {
            "concrete",
            "mixed",
        }:
            add_reason("concrete_descendant_without_concrete_region")
        elif (
            region_state == "not_applicable"
            and subregion_state == "not_applicable"
            and country_state == "not_applicable"
        ):
            intended_status = "no_geography"
            pass_reason = "all_levels_not_applicable"
        else:
            add_reason("no_resolvable_geography")

        status = "review" if reasons else intended_status
        primary_reason = reasons[0] if reasons else pass_reason
        statuses.append(status)
        primary_reasons.append(primary_reason)
        all_reasons.append(tuple(reasons))
        all_warnings.append(tuple(warnings))

    result = pd.DataFrame(
        {
            "UT": publications["UT"].to_numpy(copy=True),
            "status": statuses,
            "primary_reason": primary_reasons,
            "reasons": all_reasons,
            "warnings": all_warnings,
            "pred_regions": publications["pred_regions"].map(tuple),
            "pred_subregions": publications["pred_subregions"].map(tuple),
            "pred_countries": publications["pred_countries"].map(tuple),
        }
    )
    result = result[list(GEOGRAPHY_HIERARCHY_AUDIT_COLUMNS)]
    _validate_key(result, "IPBES geography hierarchy audit")
    if not result["UT"].reset_index(drop=True).equals(
        publications["UT"].reset_index(drop=True)
    ):
        raise EvidenceCorpusPrepError(
            "IPBES geography hierarchy audit changed UT order."
        )
    return result


def summarize_geography_hierarchy_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Count mutually exclusive hierarchy outcomes for notebook review."""
    if list(audit.columns) != list(GEOGRAPHY_HIERARCHY_AUDIT_COLUMNS):
        raise EvidenceCorpusPrepError(
            "IPBES geography hierarchy audit columns do not match the contract."
        )
    return (
        audit.groupby(["status", "primary_reason"], observed=True, dropna=False)
        .size()
        .rename("n_publications")
        .reset_index()
        .sort_values(
            ["status", "n_publications", "primary_reason"],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def apply_geography_hierarchy_audit_labels(
    publications: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize reviewed geography labels without changing source labels.

    Passing, global, and no-geography records retain their prepared geography
    labels in the three ``*_audit`` columns. Records with ``status == 'review'``
    receive ``('Unclear - Review needed',)`` in all three audit-label columns.
    The original ``pred_regions``, ``pred_subregions``, and ``pred_countries``
    columns are never changed.
    """
    required = {
        "UT",
        "pred_regions",
        "pred_subregions",
        "pred_countries",
        *GEOGRAPHY_AUDIT_LABEL_COLUMNS,
    }
    missing = required.difference(publications.columns)
    if missing:
        raise EvidenceCorpusPrepError(
            f"Integrated corpus lacks geography audit-label columns: {sorted(missing)}"
        )
    if list(audit.columns) != list(GEOGRAPHY_HIERARCHY_AUDIT_COLUMNS):
        raise EvidenceCorpusPrepError(
            "IPBES geography hierarchy audit columns do not match the contract."
        )
    _validate_key(publications, "Integrated biodiversity evidence corpus")
    _validate_key(audit, "IPBES geography hierarchy audit")
    if not audit["UT"].reset_index(drop=True).equals(
        publications["UT"].reset_index(drop=True)
    ):
        raise EvidenceCorpusPrepError(
            "IPBES geography hierarchy audit does not preserve publication UT order."
        )

    allowed_statuses = {
        "pass_complete",
        "pass_partial",
        "pass_region_only",
        "global",
        "no_geography",
        "review",
    }
    unknown_statuses = set(audit["status"].dropna()).difference(allowed_statuses)
    if audit["status"].isna().any() or unknown_statuses:
        raise EvidenceCorpusPrepError(
            "IPBES geography hierarchy audit contains unsupported statuses: "
            f"{sorted(unknown_statuses)}."
        )

    original_columns = list(publications.columns)
    original_uts = publications["UT"].reset_index(drop=True).copy()
    source_columns = ("pred_regions", "pred_subregions", "pred_countries")
    original_labels = publications[list(source_columns)].copy(deep=True)
    review_mask = audit["status"].eq("review").to_numpy()
    review_label = ("Unclear - Review needed",)
    for source, target in zip(
        source_columns,
        GEOGRAPHY_AUDIT_LABEL_COLUMNS,
        strict=True,
    ):
        prepared = publications[source].map(
            lambda value: tuple(parse_list_labels(value))
        )
        publications[target] = [
            review_label if review else labels
            for labels, review in zip(prepared, review_mask, strict=True)
        ]

    if len(publications) != len(original_uts):
        raise EvidenceCorpusPrepError("Applying geography audit labels changed row count.")
    if list(publications.columns) != original_columns:
        raise EvidenceCorpusPrepError("Applying geography audit labels changed columns.")
    if not publications["UT"].reset_index(drop=True).equals(original_uts):
        raise EvidenceCorpusPrepError("Applying geography audit labels changed UT order.")
    if not publications[list(source_columns)].equals(original_labels):
        raise EvidenceCorpusPrepError(
            "Applying geography audit labels changed source geography labels."
        )

    return pd.DataFrame(
        {
            "audit_label_action": ["preserved", "review_needed"],
            "n_publications": [int((~review_mask).sum()), int(review_mask.sum())],
        }
    )


def _validate_geography_hierarchy_audit_labels(
    publications: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Require materialized audit labels to agree with hierarchy statuses."""
    review_mask = audit["status"].eq("review").to_numpy()
    review_label = ("Unclear - Review needed",)
    for source, target in zip(
        ("pred_regions", "pred_subregions", "pred_countries"),
        GEOGRAPHY_AUDIT_LABEL_COLUMNS,
        strict=True,
    ):
        prepared_source = publications[source].map(
            lambda value: tuple(parse_list_labels(value))
        ).tolist()
        audited_source = audit[source].map(
            lambda value: tuple(parse_list_labels(value))
        ).tolist()
        if audited_source != prepared_source:
            raise EvidenceCorpusPrepError(
                f"The hierarchy audit copy of {source} does not match the corpus."
            )
        expected = [
            review_label if review else tuple(parse_list_labels(value))
            for value, review in zip(
                publications[source], review_mask, strict=True
            )
        ]
        observed = publications[target].map(
            lambda value: tuple(parse_list_labels(value))
        ).tolist()
        if observed != expected:
            raise EvidenceCorpusPrepError(
                f"{target} does not agree with the geography hierarchy audit."
            )


@dataclass(frozen=True)
class ScreeningPreparedBundle:
    """Full compact screening grain, eligible metadata, aggregates, provenance."""

    screening: pd.DataFrame
    eligible: pd.DataFrame
    exclusion_overlap: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class BiodiversityEvidenceBundle:
    """Integrated one-row-per-eligible-publication corpus and provenance."""

    publications: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class BiodiversityEvidenceBuild:
    """Frames needed to write the integrated evidence artifacts."""

    publications: pd.DataFrame
    abstracts: pd.DataFrame
    manifest: dict[str, Any]


def _eligible_mask(series: pd.Series) -> pd.Series:
    if is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def _validate_key(frame: pd.DataFrame, source: str) -> None:
    if "UT" not in frame:
        raise EvidenceCorpusPrepError(f"{source} has no UT column.")
    missing = int(frame["UT"].isna().sum())
    duplicated = int(frame["UT"].duplicated().sum())
    if missing or duplicated:
        raise EvidenceCorpusPrepError(
            f"{source} violates one row per UT: "
            f"missing={missing:,}, duplicated={duplicated:,}."
        )


def derive_publication_ids(uts: pd.Series) -> pd.Series:
    """Derive stable numeric publication IDs from canonical WOS identifiers."""
    values = uts.astype("string")
    valid = values.map(
        lambda value: bool(_WOS_UT_PATTERN.fullmatch(value))
        if value is not pd.NA
        else False
    )
    if not valid.all():
        examples = values.loc[~valid].head(10).tolist()
        raise EvidenceCorpusPrepError(
            "Publication IDs require UT values in the form WOS plus a colon "
            f"and exactly 15 digits; invalid examples: {examples}."
        )

    identifiers = pd.to_numeric(values.str.slice(4), errors="raise").astype("int64")
    identifiers.name = "id"
    if identifiers.duplicated().any():
        examples = identifiers.loc[identifiers.duplicated(keep=False)].head(10).tolist()
        raise EvidenceCorpusPrepError(
            f"Derived publication IDs are not unique; examples: {examples}."
        )
    if int(identifiers.max()) > _EXCEL_MAX_EXACT_INTEGER:
        raise EvidenceCorpusPrepError(
            "A derived publication ID exceeds Excel's exact-integer limit."
        )
    reconstructed = "WOS:" + identifiers.map(lambda value: f"{value:015d}")
    if not reconstructed.eq(values).all():
        raise EvidenceCorpusPrepError(
            "Derived publication IDs cannot be reversed to the source UT values."
        )
    return identifiers


def _validate_publication_id(frame: pd.DataFrame, source: str) -> None:
    if "id" not in frame:
        raise EvidenceCorpusPrepError(f"{source} has no id column.")
    if not is_integer_dtype(frame["id"].dtype):
        raise EvidenceCorpusPrepError(f"{source} id values must use an integer dtype.")
    missing = int(frame["id"].isna().sum())
    duplicated = int(frame["id"].duplicated().sum())
    if missing or duplicated:
        raise EvidenceCorpusPrepError(
            f"{source} violates one row per id: "
            f"missing={missing:,}, duplicated={duplicated:,}."
        )
    if "UT" in frame:
        expected = derive_publication_ids(frame["UT"])
        actual = frame["id"].astype("int64").reset_index(drop=True)
        if not actual.equals(expected.reset_index(drop=True)):
            raise EvidenceCorpusPrepError(
                f"{source} id values do not match their source UT values."
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _direction_audit(screening: pd.DataFrame) -> list[dict[str, Any]]:
    eligible = screening.loc[screening["is_eligible"]]
    directions = (
        eligible["s2_dir"]
        .fillna("missing")
        .astype(str)
        .str.strip()
        .replace("", "missing")
    )
    counts = directions.value_counts(dropna=False)
    return [
        {
            "direction_of_change": direction,
            "records": int(count),
            "percent_of_eligible": count / len(eligible) * 100,
        }
        for direction, count in counts.items()
    ]


def _exclusion_overlap(screening: pd.DataFrame) -> pd.DataFrame:
    complete = screening.loc[
        ~screening["is_eligible"]
        & screening[list(REASON_COLUMNS)].notna().all(axis=1)
    ]
    zero_steps = complete[list(REASON_COLUMNS)].eq(0).rename(columns=STAGE_NAMES)
    zero_steps = zero_steps.loc[zero_steps.any(axis=1)]
    overlap = (
        zero_steps.value_counts()
        .rename("records")
        .reset_index()
        .sort_values("records", ascending=False)
        .reset_index(drop=True)
    )
    overlap.insert(0, "intersection", range(1, len(overlap) + 1))
    overlap["combination"] = overlap.apply(
        lambda row: " + ".join(
            stage for stage in STAGE_NAMES.values() if row[stage]
        ),
        axis=1,
    )
    overlap["percent"] = (overlap["records"] / len(zero_steps) * 100).round(2)
    return overlap


def build_screening_preparation(
    screening_path: str | Path,
    *,
    eligible_columns: Iterable[str],
    chunksize: int,
    repository_root: str | Path,
) -> ScreeningPreparedBundle:
    """Read the full screening CSV once and build both reusable grains."""
    screening_path = Path(screening_path)
    if chunksize <= 0:
        raise EvidenceCorpusPrepError("chunksize must be positive.")
    raw_columns = pd.read_csv(screening_path, nrows=0).columns.tolist()
    eligible_columns = tuple(dict.fromkeys([*eligible_columns, "is_eligible"]))
    required = set(SCREENING_ANALYSIS_COLUMNS).union(eligible_columns)
    missing = required.difference(raw_columns)
    if missing:
        raise EvidenceCorpusPrepError(
            f"Full screening source lacks required columns: {sorted(missing)}"
        )

    compact_chunks: list[pd.DataFrame] = []
    eligible_chunks: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            screening_path,
            usecols=list(required),
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            mask = _eligible_mask(chunk["is_eligible"])
            compact = chunk[list(SCREENING_ANALYSIS_COLUMNS)].copy()
            compact["is_eligible"] = mask.to_numpy()
            compact_chunks.append(compact)
            eligible = chunk.loc[mask, list(eligible_columns)].copy()
            eligible["is_eligible"] = True
            eligible_chunks.append(eligible)
    except ValueError as exc:
        raise EvidenceCorpusPrepError(
            f"Could not read required screening columns: {exc}"
        ) from exc

    screening = pd.concat(compact_chunks, ignore_index=True)
    eligible = pd.concat(eligible_chunks, ignore_index=True)
    _validate_key(screening, "Prepared full screening evidence")
    _validate_key(eligible, "Prepared eligible screening evidence")
    if not eligible["UT"].reset_index(drop=True).equals(
        screening.loc[screening["is_eligible"], "UT"].reset_index(drop=True)
    ):
        raise EvidenceCorpusPrepError(
            "Eligible screening artifact does not preserve the full-screening key order."
        )

    overlap = _exclusion_overlap(screening)
    eligibility_counts = screening["eligibility"].value_counts(dropna=False)
    eligible_count = int(screening["is_eligible"].sum())
    if eligible_count != len(eligible):
        raise EvidenceCorpusPrepError("Eligible screening counts do not reconcile.")
    unclear = screening[list(REASON_COLUMNS)].eq(-1).any(axis=1)
    negative = screening["s2_dir"].eq("negative")
    summary = {
        "screened_records": len(screening),
        "raw_column_count": len(raw_columns),
        "raw_columns": raw_columns,
        "eligible_records": eligible_count,
        "not_eligible_records": int((~screening["is_eligible"]).sum()),
        "eligibility_label_eligible": int(eligibility_counts.get("ELIGIBLE", 0)),
        "eligibility_label_not_eligible": int(
            eligibility_counts.get("NOT_ELIGIBLE", 0)
        ),
        "eligible_with_unclear_step": int(
            (screening["is_eligible"] & unclear).sum()
        ),
        "eligible_negative": int((screening["is_eligible"] & negative).sum()),
        "eligible_not_negative": int(
            (screening["is_eligible"] & ~negative).sum()
        ),
        "complete_ineligible_with_zero_reason": int(overlap["records"].sum()),
    }
    manifest = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_signature(
            screening_path, repository_root=repository_root
        ),
        "artifacts": {
            "screening_publications": "screening_publications.parquet",
            "eligible_screening_publications": "eligible_screening_publications.parquet",
            "screening_exclusion_overlap": "screening_exclusion_overlap.csv",
        },
        "rows": {
            "screening_publications": len(screening),
            "eligible_screening_publications": len(eligible),
            "screening_exclusion_overlap": len(overlap),
        },
        "summary": summary,
        "eligible_direction_audit": _direction_audit(screening),
        "screening_analysis_columns": list(SCREENING_ANALYSIS_COLUMNS),
        "eligible_columns": list(eligible.columns),
    }
    return ScreeningPreparedBundle(screening, eligible, overlap, manifest)


class ScreeningPreparedStore:
    """Write/load the compact full-screening and eligible-screening artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.screening_path = self.root / "screening_publications.parquet"
        self.eligible_path = self.root / "eligible_screening_publications.parquet"
        self.overlap_path = self.root / "screening_exclusion_overlap.csv"
        self.manifest_path = self.root / "manifest.json"

    def write(self, bundle: ScreeningPreparedBundle) -> list[Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        bundle.screening.to_parquet(
            self.screening_path, index=False, compression="zstd"
        )
        bundle.eligible.to_parquet(
            self.eligible_path, index=False, compression="zstd"
        )
        bundle.exclusion_overlap.to_csv(self.overlap_path, index=False)
        _write_json(self.manifest_path, bundle.manifest)
        return [
            self.screening_path,
            self.eligible_path,
            self.overlap_path,
            self.manifest_path,
        ]

    def _load_manifest(self, *, require_eligible: bool = True) -> dict[str, Any]:
        required_paths = [
            self.screening_path,
            self.overlap_path,
            self.manifest_path,
        ]
        if require_eligible:
            required_paths.append(self.eligible_path)
        missing = [
            path
            for path in required_paths
            if not path.exists()
        ]
        if missing:
            raise EvidenceCorpusPrepError(
                "Prepared screening bundle is missing. Run the screening "
                f"preparation notebook. Missing: {[str(path) for path in missing]}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCREENING_SCHEMA_VERSION:
            raise EvidenceCorpusPrepError(
                "Prepared screening schema is incompatible; rebuild it."
            )
        return manifest

    def load_analysis(self) -> tuple[dict[str, Any], pd.DataFrame]:
        """Load only the tiny aggregates required by screening results."""
        manifest = self._load_manifest(require_eligible=False)
        overlap = pd.read_csv(self.overlap_path)
        if len(overlap) != manifest["rows"]["screening_exclusion_overlap"]:
            raise EvidenceCorpusPrepError(
                "Prepared screening-overlap row count does not match manifest."
            )
        return manifest, overlap

    def remove_eligible_build_cache(self) -> bool:
        """Remove the consumed eligible-screening cache after corpus verification."""
        manifest = self._load_manifest(require_eligible=False)
        if not self.eligible_path.exists():
            return False
        self.eligible_path.unlink()
        manifest["artifacts"].pop("eligible_screening_publications", None)
        manifest["consumed_build_cache"] = {
            "filename": self.eligible_path.name,
            "removed_at_utc": datetime.now(timezone.utc).isoformat(),
            "recoverability": (
                "rerun notebooks/data_processing/03_screening_analysis_prep.ipynb"
            ),
        }
        _write_json(self.manifest_path, manifest)
        return True

    def load_eligible(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        manifest = self._load_manifest()
        eligible = pd.read_parquet(self.eligible_path)
        _validate_key(eligible, "Prepared eligible screening publications")
        if len(eligible) != manifest["rows"]["eligible_screening_publications"]:
            raise EvidenceCorpusPrepError(
                "Prepared eligible screening row count does not match manifest."
            )
        return eligible, manifest

    def load(self) -> ScreeningPreparedBundle:
        manifest = self._load_manifest()
        screening = pd.read_parquet(self.screening_path)
        eligible = pd.read_parquet(self.eligible_path)
        overlap = pd.read_csv(self.overlap_path)
        _validate_key(screening, "Prepared screening publications")
        _validate_key(eligible, "Prepared eligible screening publications")
        if len(screening) != manifest["rows"]["screening_publications"]:
            raise EvidenceCorpusPrepError(
                "Prepared screening row count does not match manifest."
            )
        return ScreeningPreparedBundle(screening, eligible, overlap, manifest)


def _compute_plastics_text_features(
    title: pd.Series,
    abstract: pd.Series,
    *,
    plastics_pattern: str = DEFAULT_PLASTICS_PATTERN,
) -> pd.DataFrame:
    """Derive text-availability and plastics-mention flags from title + abstract.

    Runs once here, at prep time, so downstream analyses never need abstract
    text themselves - only these two booleans.
    """
    title_text = title.fillna("").astype("string").str.strip()
    abstract_text = abstract.fillna("").astype("string").str.strip()
    text_available = title_text.ne("") | abstract_text.ne("")
    combined_text = title_text.str.cat(abstract_text, sep=" ")
    try:
        plastics_mention = (
            combined_text.str.contains(
                plastics_pattern,
                case=False,
                regex=True,
                na=False,
            )
            & text_available
        )
    except Exception as exc:
        raise EvidenceCorpusPrepError(
            f"plastics_pattern is not a usable regex: {exc}"
        ) from exc
    return pd.DataFrame(
        {"text_available": text_available, "plastics_mention": plastics_mention}
    )


def build_biodiversity_evidence_corpus(
    merged_corpus: pd.DataFrame,
    taxa_articles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the publication corpus and separate abstract sidecar."""
    _validate_key(merged_corpus, "Merged eligible corpus")
    _validate_key(taxa_articles, "Prepared taxa publications")
    left_keys = pd.Index(merged_corpus["UT"])
    right_keys = pd.Index(taxa_articles["UT"])
    if len(left_keys.difference(right_keys)) or len(right_keys.difference(left_keys)):
        raise EvidenceCorpusPrepError(
            "Merged corpus and prepared taxa artifacts have different UT sets."
        )

    required_metadata = set(EVIDENCE_METADATA_COLUMNS)
    missing_metadata = required_metadata.difference(merged_corpus.columns)
    if missing_metadata:
        raise EvidenceCorpusPrepError(
            f"Merged corpus lacks required evidence columns: {sorted(missing_metadata)}"
        )
    if "abstract" not in merged_corpus:
        raise EvidenceCorpusPrepError(
            "Merged corpus must contain abstract for the text sidecar."
        )
    missing_taxa = set(TAXONOMY_RANK_COLUMNS).union(
        {"taxa_record_status", "drivers", "threat_l0", "realms"}
    ).difference(taxa_articles.columns)
    if missing_taxa:
        raise EvidenceCorpusPrepError(
            f"Prepared taxa artifact lacks required fields: {sorted(missing_taxa)}"
        )

    publication_ids = derive_publication_ids(merged_corpus["UT"])
    abstracts = merged_corpus[["UT", "abstract"]].copy()
    abstracts.insert(0, "id", publication_ids.to_numpy(copy=True))
    _validate_key(abstracts, "Biodiversity evidence abstracts")
    _validate_publication_id(abstracts, "Biodiversity evidence abstracts")
    publications = merged_corpus.drop(
        columns=["abstract", "wos_categories"], errors="ignore"
    ).copy()
    publications.insert(0, "id", publication_ids.to_numpy(copy=True))
    text_features = _compute_plastics_text_features(
        merged_corpus["title"], merged_corpus["abstract"]
    )
    publications["text_available"] = text_features["text_available"].to_numpy()
    publications["plastics_mention"] = text_features["plastics_mention"].to_numpy()
    present_list_columns = [
        column for column in EVIDENCE_LIST_COLUMNS if column in publications
    ]
    for column in present_list_columns:
        publications[column] = publications[column].map(
            lambda value: tuple(parse_list_labels(value))
        )
    publications["pred_study_design"] = publications[
        "pred_study_design"
    ].map(lambda labels: labels or (_UNCLEAR_LABEL,))

    cross_checks = {
        "driver": "drivers",
        "pred_threat_l0": "threat_l0",
        "realm": "realms",
    }
    aligned_taxa = taxa_articles.set_index("UT").loc[publications["UT"]]
    for corpus_column, taxa_column in cross_checks.items():
        corpus_values = publications[corpus_column]
        if corpus_column == "driver":
            # The integrated corpus preserves the raw screening/coding state,
            # while taxa preparation deliberately excludes non-substantive
            # driver values before analysis.
            excluded = {"not applicable", "non applicable", "unclear"}
            corpus_values = corpus_values.map(
                lambda labels: tuple(
                    label for label in labels if label.casefold() not in excluded
                )
            )
        taxa_values = aligned_taxa[taxa_column]
        if corpus_column in {"driver", "pred_threat_l0"}:
            agrees = corpus_values.map(frozenset).reset_index(drop=True).equals(
                taxa_values.map(frozenset).reset_index(drop=True)
            )
        else:
            agrees = corpus_values.reset_index(drop=True).equals(
                taxa_values.reset_index(drop=True)
            )
        if not agrees:
            raise EvidenceCorpusPrepError(
                f"Prepared taxa and merged corpus disagree on {corpus_column}."
            )

    before = publications["UT"].reset_index(drop=True)
    aligned_taxa = aligned_taxa.reset_index(drop=True)
    for column in TAXONOMY_RANK_COLUMNS:
        publications[column] = aligned_taxa[column].map(
            lambda value: tuple(value) if value is not None else ()
        )
    publications["taxa_record_status"] = aligned_taxa[
        "taxa_record_status"
    ].to_numpy()
    for source, target in zip(
        ("pred_regions", "pred_subregions", "pred_countries"),
        GEOGRAPHY_AUDIT_LABEL_COLUMNS,
        strict=True,
    ):
        publications[target] = publications[source].map(tuple)
    missing_final = set(EVIDENCE_COLUMNS).difference(publications.columns)
    if missing_final:
        raise EvidenceCorpusPrepError(
            f"Integrated corpus lacks final columns: {sorted(missing_final)}"
        )
    publications = publications[list(EVIDENCE_COLUMNS)].copy()
    if len(publications) != len(merged_corpus) or not publications["UT"].reset_index(
        drop=True
    ).equals(before):
        raise EvidenceCorpusPrepError("Attaching taxa changed the UT grain or order.")
    _validate_key(publications, "Integrated biodiversity evidence corpus")
    _validate_publication_id(publications, "Integrated biodiversity evidence corpus")
    if not abstracts["UT"].reset_index(drop=True).equals(before):
        raise EvidenceCorpusPrepError("Abstract sidecar changed the UT grain or order.")
    if not abstracts["id"].reset_index(drop=True).equals(
        publications["id"].reset_index(drop=True)
    ):
        raise EvidenceCorpusPrepError("Abstract sidecar changed the id grain or order.")
    return publications, abstracts


def build_biodiversity_manifest(
    publications: pd.DataFrame,
    abstracts: pd.DataFrame,
    geography_hierarchy_audit: pd.DataFrame,
    *,
    sources: Mapping[str, str | Path],
    repository_root: str | Path,
) -> dict[str, Any]:
    """Record the integrated-corpus contract and portable source paths."""
    _validate_key(publications, "Integrated biodiversity evidence corpus")
    _validate_publication_id(publications, "Integrated biodiversity evidence corpus")
    _validate_key(abstracts, "Biodiversity evidence abstracts")
    _validate_publication_id(abstracts, "Biodiversity evidence abstracts")
    _validate_key(geography_hierarchy_audit, "IPBES geography hierarchy audit")
    if list(publications.columns) != list(EVIDENCE_COLUMNS):
        raise EvidenceCorpusPrepError(
            "Integrated corpus columns do not match the schema contract."
        )
    if set(EVIDENCE_COLUMN_DESCRIPTIONS) != set(EVIDENCE_COLUMNS):
        raise EvidenceCorpusPrepError(
            "Column-description contract does not match the schema contract."
        )
    if list(abstracts.columns) != ["id", "UT", "abstract"]:
        raise EvidenceCorpusPrepError(
            "Abstract sidecar must contain exactly id, UT, and abstract."
        )
    if not abstracts["UT"].reset_index(drop=True).equals(
        publications["UT"].reset_index(drop=True)
    ):
        raise EvidenceCorpusPrepError(
            "Abstract sidecar does not preserve publication UT order."
        )
    if not abstracts["id"].reset_index(drop=True).equals(
        publications["id"].reset_index(drop=True)
    ):
        raise EvidenceCorpusPrepError(
            "Abstract sidecar does not preserve publication id order."
        )
    if list(geography_hierarchy_audit.columns) != list(
        GEOGRAPHY_HIERARCHY_AUDIT_COLUMNS
    ):
        raise EvidenceCorpusPrepError(
            "IPBES geography hierarchy audit columns do not match the contract."
        )
    if not geography_hierarchy_audit["UT"].reset_index(drop=True).equals(
        publications["UT"].reset_index(drop=True)
    ):
        raise EvidenceCorpusPrepError(
            "IPBES geography hierarchy audit does not preserve publication UT order."
        )
    _validate_geography_hierarchy_audit_labels(
        publications,
        geography_hierarchy_audit,
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grain": "one row per eligible screening publication (id; UT source key)",
        "primary_key": "id",
        "source_key": "UT",
        "id_derivation": "integer value of the 15-digit WOS UT suffix",
        "scope": (
            "all screening-eligible publications; direction, year, threat, realm, "
            "geography, study, and taxa filters are not applied"
        ),
        "artifacts": {
            "dataset": "dataset.parquet",
            "dataset_xlsx": "dataset.xlsx",
            "dataset_abstracts": "dataset_abstracts.parquet",
        },
        "columns": list(EVIDENCE_COLUMNS),
        "column_descriptions": {
            column: EVIDENCE_COLUMN_DESCRIPTIONS[column] for column in EVIDENCE_COLUMNS
        },
        "abstract_columns": ["id", "UT", "abstract"],
        "xlsx_sheet": "dataset",
        "xlsx_list_encoding": "JSON text",
        "geography_audit_label_columns": list(GEOGRAPHY_AUDIT_LABEL_COLUMNS),
        "geography_audit_review_label": "Unclear - Review needed",
        "list_columns": [
            column for column in EVIDENCE_LIST_COLUMNS if column in publications
        ],
        "sources": {
            name: {
                "path": repository_relative_path(
                    path,
                    repository_root=repository_root,
                )
            }
            for name, path in sources.items()
        },
        "direction_counts": {
            str(key): int(value)
            for key, value in publications["s2_dir"].value_counts(
                dropna=False
            ).items()
        },
        "geography_hierarchy_status_counts": {
            str(key): int(value)
            for key, value in geography_hierarchy_audit["status"]
            .value_counts(dropna=False)
            .items()
        },
        "geography_hierarchy_primary_reason_counts": {
            str(key): int(value)
            for key, value in geography_hierarchy_audit["primary_reason"]
            .value_counts(dropna=False)
            .items()
        },
    }


def _xlsx_cell_value(worksheet: Any, value: object, *, list_value: bool) -> object:
    """Convert one Parquet-compatible value to a lossless Excel cell value."""
    if list_value:
        if value is None:
            text = "[]"
        else:
            text = json.dumps(list(value), ensure_ascii=False)
        if len(text) > 32_767 or ILLEGAL_CHARACTERS_RE.search(text):
            raise EvidenceCorpusPrepError(
                "A list-valued dataset cell cannot be represented losslessly in Excel."
            )
        return text

    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        value = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
    if isinstance(value, str):
        if len(value) > 32_767 or ILLEGAL_CHARACTERS_RE.search(value):
            raise EvidenceCorpusPrepError(
                "A text-valued dataset cell cannot be represented losslessly in Excel."
            )
        if value.startswith("="):
            cell = WriteOnlyCell(worksheet, value=value)
            cell.data_type = "s"
            return cell
    return value


def write_dataset_xlsx(publications: pd.DataFrame, path: str | Path) -> Path:
    """Stream the complete publication dataset to a single Excel worksheet."""
    _validate_key(publications, "Integrated biodiversity evidence dataset")
    _validate_publication_id(publications, "Integrated biodiversity evidence dataset")
    if list(publications.columns) != list(EVIDENCE_COLUMNS):
        raise EvidenceCorpusPrepError(
            "Cannot write dataset.xlsx with incompatible columns."
        )
    if len(publications) + 1 > 1_048_576:
        raise EvidenceCorpusPrepError(
            "dataset.xlsx would exceed Excel's 1,048,576-row worksheet limit."
        )
    if len(publications.columns) > 16_384:
        raise EvidenceCorpusPrepError(
            "dataset.xlsx would exceed Excel's 16,384-column worksheet limit."
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.xlsx")
    workbook = Workbook(write_only=True)
    worksheet = workbook.create_sheet("dataset")
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = (
        f"A1:{get_column_letter(len(publications.columns))}{len(publications) + 1}"
    )
    worksheet.sheet_view.showGridLines = False

    list_columns = set(EVIDENCE_LIST_COLUMNS)
    for index, column in enumerate(publications.columns, start=1):
        if column == "title":
            width = 50
        elif column in list_columns or column == "locale_coordinates":
            width = 34
        elif column == "id":
            width = 18
        elif column in {"UT", "doi"}:
            width = 26
        else:
            width = min(max(len(column) + 2, 12), 24)
        worksheet.column_dimensions[get_column_letter(index)].width = width

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    header: list[WriteOnlyCell] = []
    for column in publications.columns:
        cell = WriteOnlyCell(worksheet, value=column)
        cell.fill = header_fill
        cell.font = header_font
        header.append(cell)
    worksheet.append(header)

    list_positions = {
        position
        for position, column in enumerate(publications.columns)
        if column in list_columns
    }
    try:
        for row in publications.itertuples(index=False, name=None):
            worksheet.append(
                [
                    _xlsx_cell_value(
                        worksheet,
                        value,
                        list_value=position in list_positions,
                    )
                    for position, value in enumerate(row)
                ]
            )
        workbook.save(temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


class BiodiversityEvidenceStore:
    """Write and validate the id-keyed integrated corpus and abstract sidecar."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.dataset_path = self.root / "dataset.parquet"
        self.dataset_xlsx_path = self.root / "dataset.xlsx"
        self.abstracts_path = self.root / "dataset_abstracts.parquet"
        self.manifest_path = self.root / "manifest.json"

    def write(self, bundle: BiodiversityEvidenceBuild) -> list[Path]:
        """Write the one-row-per-publication evidence artifacts."""
        self.root.mkdir(parents=True, exist_ok=True)
        if list(bundle.publications.columns) != list(EVIDENCE_COLUMNS):
            raise EvidenceCorpusPrepError(
                "Cannot write biodiversity evidence with incompatible columns."
            )
        bundle.publications.to_parquet(
            self.dataset_path,
            index=False,
            compression="zstd",
        )
        write_dataset_xlsx(bundle.publications, self.dataset_xlsx_path)
        bundle.abstracts.to_parquet(
            self.abstracts_path, index=False, compression="zstd"
        )
        _write_json(self.manifest_path, bundle.manifest)
        return [
            self.dataset_path,
            self.dataset_xlsx_path,
            self.abstracts_path,
            self.manifest_path,
        ]

    def _load_manifest(self, *required: Path) -> dict[str, Any]:
        """Load the manifest after checking only the artifacts the caller reads.

        The abstract sidecar is not distributed (Web of Science terms of use), so
        loaders that never touch abstract text must not require it.
        """
        missing = [
            path
            for path in (self.manifest_path, *required)
            if not path.exists()
        ]
        if missing:
            raise EvidenceCorpusPrepError(
                "Prepared biodiversity-evidence corpus is missing. Run its "
                f"data-processing notebook. Missing: {[str(path) for path in missing]}"
            )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise EvidenceCorpusPrepError(
                "Prepared biodiversity-evidence schema is incompatible; rebuild it."
            )
        return manifest

    def load(
        self,
        *,
        columns: Iterable[str] | None = None,
    ) -> BiodiversityEvidenceBundle:
        """Load all fields or an explicit projection from the main corpus."""
        manifest = self._load_manifest(self.dataset_path)
        expected_columns = manifest["columns"]
        selected = list(columns) if columns is not None else expected_columns
        if len(selected) != len(set(selected)):
            raise EvidenceCorpusPrepError("Requested evidence columns are duplicated.")
        unknown = set(selected).difference(expected_columns)
        if unknown:
            raise EvidenceCorpusPrepError(
                f"Requested evidence columns are unavailable: {sorted(unknown)}"
            )
        publications = pd.read_parquet(self.dataset_path, columns=selected)
        for column in set(manifest.get("list_columns", ())).intersection(selected):
            publications[column] = publications[column].map(
                lambda value: tuple(value) if value is not None else ()
            )
        if "UT" in publications:
            _validate_key(publications, "Prepared biodiversity evidence corpus")
        if "id" in publications:
            _validate_publication_id(
                publications,
                "Prepared biodiversity evidence corpus",
            )
        if list(publications.columns) != selected:
            raise EvidenceCorpusPrepError(
                "Prepared biodiversity-evidence columns do not match manifest."
            )
        return BiodiversityEvidenceBundle(publications, manifest)

    def validate_xlsx(self) -> dict[str, Any]:
        """Open dataset.xlsx and validate its sheet and header contract."""
        manifest = self._load_manifest(self.dataset_xlsx_path)
        workbook = load_workbook(
            self.dataset_xlsx_path,
            read_only=True,
            data_only=True,
        )
        try:
            if workbook.sheetnames != [manifest["xlsx_sheet"]]:
                raise EvidenceCorpusPrepError(
                    "Prepared dataset.xlsx worksheet does not match the manifest."
                )
            worksheet = workbook[manifest["xlsx_sheet"]]
            header = [
                cell.value
                for cell in next(worksheet.iter_rows(min_row=1, max_row=1))
            ]
            if header != manifest["columns"]:
                raise EvidenceCorpusPrepError(
                    "Prepared dataset.xlsx columns do not match the manifest."
                )
            return {
                "sheet": manifest["xlsx_sheet"],
                "columns": len(header),
            }
        finally:
            workbook.close()

    def load_abstracts(self) -> pd.DataFrame:
        """Load publication text only for analyses that explicitly require it."""
        manifest = self._load_manifest(self.abstracts_path)
        abstracts = pd.read_parquet(self.abstracts_path)
        if list(abstracts.columns) != manifest["abstract_columns"]:
            raise EvidenceCorpusPrepError(
                "Prepared abstract columns do not match the manifest."
            )
        _validate_key(abstracts, "Prepared biodiversity evidence abstracts")
        _validate_publication_id(
            abstracts,
            "Prepared biodiversity evidence abstracts",
        )
        return abstracts

__all__ = [
    "BiodiversityEvidenceBundle",
    "BiodiversityEvidenceBuild",
    "BiodiversityEvidenceStore",
    "DEFAULT_PLASTICS_PATTERN",
    "EVIDENCE_COLUMNS",
    "EVIDENCE_COLUMN_DESCRIPTIONS",
    "EVIDENCE_LIST_COLUMNS",
    "GEOGRAPHY_AUDIT_LABEL_COLUMNS",
    "GEOGRAPHY_HIERARCHY_AUDIT_COLUMNS",
    "GEOGRAPHY_LIST_COLUMNS",
    "GEOGRAPHY_VALUE_COLUMNS",
    "EvidenceCorpusPrepError",
    "REASON_COLUMNS",
    "SCREENING_ANALYSIS_COLUMNS",
    "STAGE_NAMES",
    "ScreeningPreparedBundle",
    "ScreeningPreparedStore",
    "apply_geography_hierarchy_audit_labels",
    "audit_ipbes_geography_hierarchy",
    "build_biodiversity_evidence_corpus",
    "build_biodiversity_manifest",
    "build_screening_preparation",
    "derive_publication_ids",
    "standardize_geography_lists",
    "summarize_geography_hierarchy_audit",
    "write_dataset_xlsx",
]

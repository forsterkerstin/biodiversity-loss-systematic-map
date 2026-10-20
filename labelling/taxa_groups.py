"""Configurable multi-resolution grouping over matched GBIF hierarchies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GROUP_CONFIG_SCHEMA_VERSION = 3
SUPPORTED_OPERATORS = {"in", "not_in", "present", "missing"}
HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
HTTPS_URL = re.compile(r"^https://")
CLIPART_REQUIRED_STRING_FIELDS = {
    "provider",
    "asset_id",
    "page_url",
    "svg_url",
    "license",
    "license_url",
}


class TaxaGroupConfigError(ValueError):
    """Raised when the taxonomic-group configuration is invalid."""


@dataclass(frozen=True)
class GroupAssignment:
    """One taxon item's outcome in one configured grouping scheme."""

    group: str | None
    rule_id: str | None
    reason: str
    eligible: bool


def _require_unique_strings(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise TaxaGroupConfigError(f"{field} must be a non-empty string list.")
    cleaned = [item.strip() for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise TaxaGroupConfigError(f"{field} must not contain duplicates.")
    return cleaned


def _validate_scheme(scheme_id: str, scheme: Any) -> dict[str, Any]:
    field = f"schemes.{scheme_id}"
    if not isinstance(scheme, dict):
        raise TaxaGroupConfigError(f"{field} must be an object.")
    output_column = scheme.get("output_column")
    if not isinstance(output_column, str) or not output_column.strip():
        raise TaxaGroupConfigError(f"{field}.output_column must be a string.")

    group_order = _require_unique_strings(
        scheme.get("group_order"),
        f"{field}.group_order",
    )
    unresolved_group = scheme.get("unresolved_group")
    if unresolved_group not in group_order:
        raise TaxaGroupConfigError(
            f"{field}.unresolved_group must be present in group_order."
        )

    colors = scheme.get("group_colors")
    if not isinstance(colors, dict) or set(colors) != set(group_order):
        raise TaxaGroupConfigError(
            f"{field}.group_colors must cover every group exactly once."
        )
    for group, color in colors.items():
        if not isinstance(color, str) or not HEX_COLOR.fullmatch(color):
            raise TaxaGroupConfigError(
                f"Invalid hexadecimal color for {field} group " f"{group!r}: {color!r}"
            )

    rules = scheme.get("rules")
    if not isinstance(rules, list) or not rules:
        raise TaxaGroupConfigError(f"{field}.rules must be a non-empty list.")
    rule_ids: set[str] = set()
    for position, rule in enumerate(rules):
        rule_field = f"{field}.rules[{position}]"
        if not isinstance(rule, dict):
            raise TaxaGroupConfigError(f"{rule_field} must be an object.")
        rule_id = rule.get("id")
        group = rule.get("group")
        predicates = rule.get("all")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise TaxaGroupConfigError(f"{rule_field}.id must be a string.")
        if rule_id in rule_ids:
            raise TaxaGroupConfigError(f"Duplicate rule id in {field}: {rule_id!r}.")
        rule_ids.add(rule_id)
        if group not in group_order or group == unresolved_group:
            raise TaxaGroupConfigError(
                f"{rule_field}.group must be a configured " "non-unresolved group."
            )
        if not isinstance(predicates, list) or not predicates:
            raise TaxaGroupConfigError(
                f"{rule_field}.all must be a non-empty predicate list."
            )
        for predicate_position, predicate in enumerate(predicates):
            predicate_field = f"{rule_field}.all[{predicate_position}]"
            if not isinstance(predicate, dict):
                raise TaxaGroupConfigError(f"{predicate_field} must be an object.")
            rank = predicate.get("rank")
            operator = predicate.get("operator")
            if not isinstance(rank, str) or not rank.strip():
                raise TaxaGroupConfigError(f"{predicate_field}.rank must be a string.")
            if operator not in SUPPORTED_OPERATORS:
                raise TaxaGroupConfigError(
                    f"{predicate_field}.operator must be one of "
                    f"{sorted(SUPPORTED_OPERATORS)}."
                )
            values = predicate.get("values")
            if operator in {"in", "not_in"}:
                _require_unique_strings(
                    values,
                    f"{predicate_field}.values",
                )
            elif values is not None:
                raise TaxaGroupConfigError(
                    f"{predicate_field}.values is only valid for in/not_in."
                )

    scheme["output_column"] = output_column.strip()
    scheme["group_order"] = group_order
    return scheme


def _validate_clipart_metadata(
    config: dict[str, Any],
    schemes: dict[str, Any],
    scheme_order: list[str],
) -> None:
    note = config.get("clipart_note")
    if not isinstance(note, str) or not note.strip():
        raise TaxaGroupConfigError("clipart_note must be a non-empty string.")

    assets = config.get("clipart_assets")
    if not isinstance(assets, dict) or not assets:
        raise TaxaGroupConfigError("clipart_assets must be a non-empty object.")

    for asset_name, asset in assets.items():
        field = f"clipart_assets.{asset_name}"
        if not isinstance(asset_name, str) or not asset_name.strip():
            raise TaxaGroupConfigError("clipart asset ids must be non-empty strings.")
        if not isinstance(asset, dict):
            raise TaxaGroupConfigError(f"{field} must be an object.")
        missing = CLIPART_REQUIRED_STRING_FIELDS - set(asset)
        if missing:
            raise TaxaGroupConfigError(
                f"{field} is missing required fields: {sorted(missing)}."
            )
        for key in CLIPART_REQUIRED_STRING_FIELDS:
            value = asset[key]
            if not isinstance(value, str) or not value.strip():
                raise TaxaGroupConfigError(f"{field}.{key} must be a string.")
        for key in ("page_url", "svg_url", "license_url"):
            if not HTTPS_URL.match(asset[key]):
                raise TaxaGroupConfigError(f"{field}.{key} must use HTTPS.")
        for key in ("representative_taxon", "thumbnail_url", "attribution"):
            value = asset.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TaxaGroupConfigError(
                    f"{field}.{key} must be null or a non-empty string."
                )
        thumbnail_url = asset.get("thumbnail_url")
        if thumbnail_url is not None and not HTTPS_URL.match(thumbnail_url):
            raise TaxaGroupConfigError(
                f"{field}.thumbnail_url must be null or use HTTPS."
            )

    referenced_assets: set[str] = set()
    for scheme_id in scheme_order:
        scheme = schemes[scheme_id]
        field = f"schemes.{scheme_id}.group_clipart"
        group_clipart = scheme.get("group_clipart")
        if not isinstance(group_clipart, dict) or set(group_clipart) != set(
            scheme["group_order"]
        ):
            raise TaxaGroupConfigError(
                f"{field} must cover every configured group exactly once."
            )
        unknown = set(group_clipart.values()) - set(assets)
        if unknown:
            raise TaxaGroupConfigError(
                f"{field} references unknown assets: {sorted(unknown)}."
            )
        referenced_assets.update(group_clipart.values())

    for field in ("sunburst_clipart", "sunburst_order_clipart"):
        figure_clipart = config.get(field, {})
        if not isinstance(figure_clipart, dict) or not figure_clipart:
            raise TaxaGroupConfigError(f"{field} must be a non-empty object.")
        if any(
            not isinstance(taxon, str)
            or not taxon.strip()
            or not isinstance(asset_name, str)
            or not asset_name.strip()
            for taxon, asset_name in figure_clipart.items()
        ):
            raise TaxaGroupConfigError(
                f"{field} keys and asset ids must be non-empty strings."
            )
        unknown = set(figure_clipart.values()) - set(assets)
        if unknown:
            raise TaxaGroupConfigError(
                f"{field} references unknown assets: {sorted(unknown)}."
            )
        referenced_assets.update(figure_clipart.values())

    retired_assets = config.get("retired_clipart_assets", [])
    retired_assets = _require_unique_strings(
        retired_assets,
        "retired_clipart_assets",
    )
    unknown = set(retired_assets) - set(assets)
    if unknown:
        raise TaxaGroupConfigError(
            f"retired_clipart_assets references unknown assets: {sorted(unknown)}."
        )
    overlap = set(retired_assets) & referenced_assets
    if overlap:
        raise TaxaGroupConfigError(
            "retired_clipart_assets cannot include active assets: "
            f"{sorted(overlap)}."
        )
    referenced_assets.update(retired_assets)

    unused = set(assets) - referenced_assets
    if unused:
        raise TaxaGroupConfigError(
            f"clipart_assets contains unused assets: {sorted(unused)}."
        )


def _grouping_config_sha256(config: dict[str, Any]) -> str:
    """Fingerprint grouping semantics while ignoring figure-only clipart."""
    grouping = json.loads(json.dumps(config, ensure_ascii=False))
    grouping.pop("clipart_note", None)
    grouping.pop("clipart_assets", None)
    grouping.pop("sunburst_clipart", None)
    grouping.pop("sunburst_order_clipart", None)
    grouping.pop("retired_clipart_assets", None)
    for scheme in grouping.get("schemes", {}).values():
        scheme.pop("group_clipart", None)
    canonical = (json.dumps(grouping, indent=2, ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def load_group_config(path: str | Path) -> dict[str, Any]:
    """Load, validate, and fingerprint all configured grouping schemes."""
    config_path = Path(path)
    raw = config_path.read_bytes()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaxaGroupConfigError(
            f"Invalid JSON in taxonomic-group mapping {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise TaxaGroupConfigError("Taxonomic-group mapping must be a JSON object.")
    if config.get("schema_version") != GROUP_CONFIG_SCHEMA_VERSION:
        raise TaxaGroupConfigError(
            "Taxonomic-group mapping must use schema_version "
            f"{GROUP_CONFIG_SCHEMA_VERSION}."
        )

    eligible_statuses = _require_unique_strings(
        config.get("eligible_match_statuses"),
        "eligible_match_statuses",
    )
    not_applicable = _require_unique_strings(
        config.get("not_applicable_values"),
        "not_applicable_values",
    )
    scheme_order = _require_unique_strings(
        config.get("scheme_order"),
        "scheme_order",
    )
    if "broad" not in scheme_order:
        raise TaxaGroupConfigError(
            "scheme_order must include the primary 'broad' scheme."
        )
    schemes = config.get("schemes")
    if not isinstance(schemes, dict) or set(schemes) != set(scheme_order):
        raise TaxaGroupConfigError(
            "schemes must cover every scheme_order entry exactly once."
        )

    output_columns: set[str] = set()
    for scheme_id in scheme_order:
        scheme = _validate_scheme(scheme_id, schemes[scheme_id])
        output_column = scheme["output_column"]
        if output_column in output_columns:
            raise TaxaGroupConfigError(
                f"Duplicate scheme output_column: {output_column!r}."
            )
        output_columns.add(output_column)
    _validate_clipart_metadata(config, schemes, scheme_order)

    grouping_sha256 = _grouping_config_sha256(config)
    config["_config_path"] = str(config_path.resolve())
    config["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    config["_grouping_sha256"] = grouping_sha256
    config["eligible_match_statuses"] = eligible_statuses
    config["not_applicable_values"] = not_applicable
    config["scheme_order"] = scheme_order

    # Preserve the primary broad-scheme keys expected by existing analysis
    # consumers while exposing all schemes through ``schemes``.
    broad = schemes["broad"]
    for key in (
        "group_order",
        "group_colors",
        "group_clipart",
        "unresolved_group",
        "rules",
        "output_column",
    ):
        config[key] = broad[key]
    return config


def hierarchy_by_rank(
    classification: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Return case-preserving hierarchy names keyed by uppercase GBIF rank."""
    hierarchy: dict[str, set[str]] = {}
    for entry in classification:
        if not isinstance(entry, dict):
            continue
        rank = str(entry.get("rank") or "").strip().upper()
        name = str(entry.get("name") or "").strip()
        if rank and name:
            hierarchy.setdefault(rank, set()).add(name)
    return hierarchy


def _predicate_matches(
    hierarchy: dict[str, set[str]],
    predicate: dict[str, Any],
) -> bool:
    rank = str(predicate["rank"]).strip().upper()
    operator = predicate["operator"]
    observed = hierarchy.get(rank, set())
    if operator == "present":
        return bool(observed)
    if operator == "missing":
        return not observed
    configured = {str(value).strip().casefold() for value in predicate["values"]}
    normalized_observed = {value.casefold() for value in observed}
    if operator == "in":
        return bool(normalized_observed & configured)
    if operator == "not_in":
        return bool(observed) and normalized_observed.isdisjoint(configured)
    raise AssertionError(f"Unsupported operator after validation: {operator}")


def assign_taxon_group(
    *,
    raw_rank: str,
    raw_name: str,
    match_status: str,
    classification: list[dict[str, Any]],
    config: dict[str, Any],
    scheme: str = "broad",
) -> GroupAssignment:
    """Assign one matched taxon using one named hierarchy-rule scheme."""
    try:
        scheme_config = config["schemes"][scheme]
    except KeyError as exc:
        raise TaxaGroupConfigError(
            f"Unknown taxonomic grouping scheme: {scheme!r}."
        ) from exc
    special_values = {
        value.strip().casefold() for value in config["not_applicable_values"]
    }
    if (
        raw_rank.strip().casefold() in special_values
        or raw_name.strip().casefold() in special_values
    ):
        return GroupAssignment(None, None, "not_applicable", False)

    if match_status not in set(config["eligible_match_statuses"]):
        return GroupAssignment(
            None,
            None,
            f"ineligible_match_status:{match_status}",
            False,
        )

    hierarchy = hierarchy_by_rank(classification)
    for rule in scheme_config["rules"]:
        if all(_predicate_matches(hierarchy, predicate) for predicate in rule["all"]):
            return GroupAssignment(
                str(rule["group"]),
                str(rule["id"]),
                "matched_rule",
                True,
            )

    return GroupAssignment(
        str(scheme_config["unresolved_group"]),
        "accepted_insufficient_hierarchy",
        "accepted_insufficient_hierarchy",
        True,
    )


def assign_taxon_groups(
    *,
    raw_rank: str,
    raw_name: str,
    match_status: str,
    classification: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, GroupAssignment]:
    """Assign one taxon independently under every configured resolution."""
    return {
        scheme: assign_taxon_group(
            raw_rank=raw_rank,
            raw_name=raw_name,
            match_status=match_status,
            classification=classification,
            config=config,
            scheme=scheme,
        )
        for scheme in config["scheme_order"]
    }


def order_groups(
    groups: list[str],
    config: dict[str, Any],
    *,
    scheme: str = "broad",
) -> list[str]:
    """De-duplicate values in one scheme's deterministic display order."""
    try:
        group_order = config["schemes"][scheme]["group_order"]
    except KeyError as exc:
        raise TaxaGroupConfigError(
            f"Unknown taxonomic grouping scheme: {scheme!r}."
        ) from exc
    observed = set(groups)
    return [group for group in group_order if group in observed]


def grouping_rule_table(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return reader-facing rows for every configured grouping rule."""
    rows: list[dict[str, Any]] = []
    for scheme in config["scheme_order"]:
        scheme_config = config["schemes"][scheme]
        for position, rule in enumerate(scheme_config["rules"], start=1):
            rows.append(
                {
                    "scheme": scheme,
                    "output_column": scheme_config["output_column"],
                    "position": position,
                    "rule_id": rule["id"],
                    "taxa_group": rule["group"],
                    "all_predicates_json": json.dumps(
                        rule["all"],
                        ensure_ascii=False,
                    ),
                }
            )
        rows.append(
            {
                "scheme": scheme,
                "output_column": scheme_config["output_column"],
                "position": len(scheme_config["rules"]) + 1,
                "rule_id": "accepted_insufficient_hierarchy",
                "taxa_group": scheme_config["unresolved_group"],
                "all_predicates_json": "[]",
            }
        )
    return rows

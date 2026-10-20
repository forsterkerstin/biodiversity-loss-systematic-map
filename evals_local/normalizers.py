from __future__ import annotations

import json
import re
from functools import lru_cache
import ast
import math
from typing import Iterable

from .config import MAPPINGS_DIR


def to_label_set(value) -> set[str]:
    """Coerce list/tuple/set/str/None into a lower-cased, deduped set of strings."""
    seq: list[str] = []
    if value is None:
        seq = []
    elif isinstance(value, float) and math.isnan(value):
        seq = []
    elif isinstance(value, (list, tuple, set)):
        seq = list(value)
    elif isinstance(value, str):
        parsed = value
        # try JSON or python-literal decoding for stringified lists
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                break
            except Exception:
                parsed = value
                continue
        if isinstance(parsed, (list, tuple, set)):
            seq = list(parsed)
        else:
            seq = [value]
    else:
        seq = [str(value)]
    clean = {
        s for s in (str(item).strip().lower() for item in seq)
        if s and s != "nan"
    }
    return clean


def to_label_list(value) -> list[str]:
    return sorted(to_label_set(value))


_THREAT_ALIASES: dict[str, str] = {
    "housing & urban areas": "housing & urban area",
}


def normalize_threat_labels(value) -> set[str]:
    """Like to_label_set but collapses known label variants (e.g. singular/plural)."""
    return {_THREAT_ALIASES.get(s, s) for s in to_label_set(value)}


@lru_cache(maxsize=1)
def load_iso_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return iso3->name and name_lower->iso3 from ipbes_regions.json."""
    path = MAPPINGS_DIR / "ipbes_regions.json"
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    iso_to_name: dict[str, str] = {}
    name_to_iso: dict[str, str] = {}

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    iso = item.get("ISO_3166_alpha_3") or item.get("GID_0")
                    name = item.get("Country")
                    if iso and name:
                        iso_upper = iso.strip().upper()
                        iso_to_name.setdefault(iso_upper, name.strip())
                        name_to_iso.setdefault(name.strip().lower(), iso_upper)
                        # strip parenthetical suffixes, e.g.
                        # "Central African Republic (the)" → "central african republic"
                        clean = re.sub(r"\s*\([^)]*\)", "", name).strip().lower()
                        name_to_iso.setdefault(clean, iso_upper)
                        # also register the short name before any comma, e.g.
                        # "Tanzania, the United Republic of" → "tanzania" → TZA
                        short = clean.split(",")[0].strip()
                        name_to_iso.setdefault(short, iso_upper)
    _walk(data)
    return iso_to_name, name_to_iso


# Explicit aliases for names that don't resolve via pattern-based normalization.
# Keys must be lower-case; values are canonical lower-case region names or upper-case ISO3.
_GEO_ALIASES: dict[str, str] = {
    # Democratic Republic of the Congo
    "d.r. congo": "COD",
    "dr congo": "COD",
    "drc": "COD",
    "democratic republic of congo": "COD",
    "democratic republic of the congo": "COD",
    "congo, democratic republic of the": "COD",
    # IPBES region variants → canonical IPBES region name
    "asia-pacific": "asia and the pacific",
    "asia pacific": "asia and the pacific",
    "europe & central asia": "europe and central asia",
    # common name ≠ official name in JSON
    "russia": "RUS",
    "russian federation": "RUS",
}


_REGION_ALIASES: dict[str, str] = {
    "all regions": "all region",
}


def normalize_region_labels(values) -> set[str]:
    """Like to_label_set but collapses known region name variants."""
    return {_REGION_ALIASES.get(s, s) for s in to_label_set(values)}


def normalize_geo_labels(values) -> set[str]:
    """Normalize geography labels to ISO3 where possible; keep lower-case names fallback."""
    iso_to_name, name_to_iso = load_iso_maps()
    out: set[str] = set()
    if values is None:
        return out
    if isinstance(values, (str, bytes)):
        # try to decode stringified lists
        decoded = values
        for parser in (json.loads, ast.literal_eval):
            try:
                decoded = parser(values)
                break
            except Exception:
                decoded = values
                continue
        values = decoded if isinstance(decoded, (list, tuple, set)) else [decoded]
    if not isinstance(values, Iterable):
        values = [values]
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text or text.lower() == "nan":
            continue
        key = text.lower()
        # check explicit aliases first so e.g. "drc" routes to COD, not "DRC"
        if _GEO_ALIASES.get(key):
            out.add(_GEO_ALIASES[key])
            continue
        if len(text) == 3 and text.isalpha():
            out.add(text.upper())
            continue
        iso = name_to_iso.get(key)
        if iso:
            out.add(iso)
        else:
            out.add(key)
    return out


def normalize_screening_label(value) -> str:
    """
    Canonicalize screening labels to either ELIGIBLE or NOT_ELIGIBLE.
    Accepts single strings or single-item lists; returns empty string if missing.
    """
    labels = to_label_list(value)
    if not labels:
        return ""
    raw = labels[0].strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"eligible"}:
        return "ELIGIBLE"
    if raw in {"not_eligible", "noteligible", "ineligible"}:
        return "NOT_ELIGIBLE"
    return raw.upper()

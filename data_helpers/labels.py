"""Shared parsing for the list-valued label columns emitted by the coding step.

Coding writes multi-label fields (drivers, threats, realms, taxa, geography) as
JSON-serialized lists, occasionally as semicolon-joined text, and occasionally as
real Python lists once a frame has been round-tripped through parquet. Every
results module needs the same tolerant reader, so it lives here rather than in
any one analysis module.

Parsing only normalises shape. Deciding which label values are meaningful -
``[]``, ``"Not Applicable"``, ``"Unclear"`` and friends - is the caller's job,
and by repository convention those are skipped but always reported in an audit.
"""

from __future__ import annotations

import ast
import json

import pandas as pd


def parse_list_labels(value) -> list[str]:
    """Return a clean, de-duplicated list from a list value or serialized list string."""
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    elif pd.isna(value):
        return []
    else:
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "null"} or text == "[]":
            return []
        raw = None
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except (ValueError, SyntaxError, TypeError):
                    continue
                if isinstance(parsed, (list, tuple, set)):
                    raw = list(parsed)
                    break
        if raw is None:
            raw = text.split(";") if ";" in text else [text]
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        label = str(item).strip()
        if label and label not in out:
            out.append(label)
    return out

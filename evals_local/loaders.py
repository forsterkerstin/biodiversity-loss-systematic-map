from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    repair_json = None

from .config import BATCH_OUTPUTS_DIR, BATCHES_DIR, GBIF_CACHE_PATH, RUN_LABEL_PATH_OVERRIDES, TASK_CONFIG

# ---------------------------------------------------------------------------
# GBIF lookup helpers for taxa prediction enrichment
# ---------------------------------------------------------------------------

_GBIF_CACHE: dict | None = None

def _get_gbif_cache() -> dict:
    global _GBIF_CACHE
    if _GBIF_CACHE is None:
        if not Path(GBIF_CACHE_PATH).exists():
            print(
                "\n[evals] GBIF lookup cache not found — taxa evals were NOT run.\n"
                f"        Missing file: {GBIF_CACHE_PATH}\n"
                "        Download 'gbif_lookup_cache.pkl' from the supplementary data of the\n"
                "        protocol and place it at the path above, then re-run the taxa evals.\n"
            )
            raise SystemExit(1)
        with open(GBIF_CACHE_PATH, "rb") as f:
            _GBIF_CACHE = pickle.load(f)
    return _GBIF_CACHE


# Maps taxon_rank strings (from LLM) to pred column names
_TAXA_RANK_TO_COL: dict[str, str] = {
    "kingdom": "pred_kingdom",
    "phylum": "pred_phylum",
    "class": "pred_class",
    "order": "pred_order",
    "genus": "pred_genus",
    "species": "pred_species",
}

# Levels that the GBIF cache provides (upstream hierarchy)
_GBIF_LOOKUP_LEVELS = ["kingdom", "phylum", "class", "order", "genus"]

_TAXA_SPECIAL_VALUES = {"not applicable", "unclear"}


def _build_taxa_pred_row(results: list | None, lookup: dict) -> dict[str, list[str]]:
    """For one paper row, build pred_* lists from the LLM results and GBIF cache."""
    acc: dict[str, list[str]] = {col: [] for col in _TAXA_RANK_TO_COL.values()}

    if not isinstance(results, list):
        return acc

    for item in results:
        if not isinstance(item, dict):
            continue
        rank = (item.get("taxon_rank") or "").lower()
        canonical = item.get("canonical_name") or ""
        if not canonical:
            continue

        # Add canonical_name to its own rank column
        dest_col = _TAXA_RANK_TO_COL.get(rank)
        if dest_col and canonical not in acc[dest_col]:
            acc[dest_col].append(canonical)

        # Skip GBIF lookup for special values
        if canonical.lower() in _TAXA_SPECIAL_VALUES:
            continue

        upstream = lookup.get(canonical.lower())
        if upstream is None:
            continue

        for level in _GBIF_LOOKUP_LEVELS:
            if level == rank:
                continue  # canonical_name already added above
            col = _TAXA_RANK_TO_COL.get(level)
            if col is None:
                continue
            val = upstream.get(level)
            if val is None:
                val = "Not Applicable"
            if val not in acc[col]:
                acc[col].append(val)

    return acc


def _expand_taxa_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add pred_kingdom/phylum/class/order/genus/species columns to df."""
    lookup = _get_gbif_cache()
    expanded = df["pred_taxa"].apply(lambda r: _build_taxa_pred_row(r, lookup))
    for col in _TAXA_RANK_TO_COL.values():
        df[col] = expanded.apply(lambda d, c=col: d[c])
    return df


def load_jsonl_folder(folder: Path) -> pd.DataFrame:
    paths = sorted(Path(folder).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files in {folder}")
    frames = [pd.read_json(p, lines=True) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _output_text_from_record(record: dict) -> str | None:
    if not isinstance(record, dict):
        return None
    body = record.get("response", {}).get("body") if isinstance(record.get("response"), dict) else record.get("body")
    if not isinstance(body, dict):
        return None
    for item in body.get("output", []) or []:
        if item.get("type") == "message":
            for part in item.get("content", []) or []:
                if part.get("type") == "output_text":
                    return part.get("text")
    return None


def _safe_json(text: str | None) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return None



def parse_json_text(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if repair_json:
        try:
            return repair_json(text, return_objects=True)
        except Exception:
            pass

    cleaned = re.sub(r",\\s*([}\\]])", r"\\1", text.replace("\\r", "").replace("\\n", " "))
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def extract_output_payload(output_list):
    # Return the first JSON payload found in the model output content, else None.
    for item in output_list or []:
        content_blocks = item.get("content") or []
        for block in content_blocks:
            text = block.get("text") if isinstance(block, dict) else None
            if not text:
                continue
            parsed = parse_json_text(text)
            if parsed is not None:
                return parsed
    return None


def flatten_steps(parsed: dict) -> dict:
    flat = {}
    if not isinstance(parsed, dict):
        return flat

    # New schema: step1/step2/step3/step4 with results/biodiversity_types/direction/drivers/linkage
    if any(k in parsed for k in ("step1", "step2", "step3", "step4")):
        new_extra = {
            "step1": ("bio", "biodiversity_types"),
            "step2": ("dir", "direction"),
            "step3": ("drivers", "drivers"),
            "step4": ("link", "linkage"),
        }
        for step_key, (flat_suffix, src_key) in new_extra.items():
            step_val = parsed.get(step_key)
            if not isinstance(step_val, dict):
                continue
            snum = step_key.replace("step", "s")
            flat[f"{snum}_r"] = step_val.get("results")
            flat[f"{snum}_{flat_suffix}"] = step_val.get(src_key)
        return flat

    # Old schema: s1/s2/s3/s4 with r/bio/dir/drivers/link
    extra_fields = {"s1": "bio", "s2": "dir", "s3": "drivers", "s4": "link"}
    for step_key, step_val in parsed.items():
        if not isinstance(step_val, dict):
            continue
        flat[f"{step_key}_r"] = step_val.get("r")
        extra = extra_fields.get(step_key)
        if extra:
            flat[f"{step_key}_{extra}"] = step_val.get(extra)
    return flat


def parse_output_record(rec: dict) -> dict:
    body = (rec.get("response") or {}).get("body") or {}
    model = body.get("model")
    created_at = body.get("created_at")
    parsed = extract_output_payload(body.get("output"))
    flat = flatten_steps(parsed)
    return {
        "custom_id": rec.get("custom_id"),
        "model": model,
        "created_at": created_at,
        "raw_output": parsed,
        **flat,
    }


_SCREENING_OUTPUT_COLS = ["custom_id", "model", "created_at", "raw_output"]


def load_screening_outputs(run: str) -> pd.DataFrame:
    out_dir = BATCH_OUTPUTS_DIR / run
    if not out_dir.exists():
        return pd.DataFrame(columns=_SCREENING_OUTPUT_COLS)
    files = sorted(out_dir.glob("*.jsonl"))
    rows = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                rec = json.loads(line)
                rows.append(parse_output_record(rec))
    if not rows:
        return pd.DataFrame(columns=_SCREENING_OUTPUT_COLS)
    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], unit="s", utc=True, errors="coerce")
    return df


def load_screening_batch_records(run: str, keep_cols=("UT", "title", "abstract", "doi")) -> pd.DataFrame:
    data_dir = BATCHES_DIR / run / "data"
    frames = []
    if not data_dir.exists():
        return pd.DataFrame(columns=keep_cols)
    for fp in sorted(data_dir.glob("*.jsonl")):
        frames.append(pd.read_json(fp, lines=True))
    if not frames:
        return pd.DataFrame(columns=keep_cols)
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in keep_cols if c in df.columns]
    return df[cols]


def load_screening_predictions(run: str) -> pd.DataFrame:
    batch_df = load_screening_batch_records(run)
    outputs_df = load_screening_outputs(run)
    joined = batch_df.merge(outputs_df, left_on="UT", right_on="custom_id", how="left")

    stage_cols = ["s1_r", "s2_r", "s3_r", "s4_r"]
    for col in stage_cols:
        if col not in joined.columns:
            joined[col] = 0
        joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0).astype(int)
    joined["pred"] = joined.apply(
        lambda r: '["NOT_ELIGIBLE"]' if any(r[c] == 0 for c in stage_cols) else '["ELIGIBLE"]',
        axis=1,
    )
    return joined


def load_screening_labels(path: Path) -> pd.DataFrame:
    if path.is_file():
        frames = [pd.read_csv(path)]
    else:
        frames = [pd.read_csv(fp) for fp in sorted(path.glob("*.csv"))]
    if not frames:
        return pd.DataFrame(columns=["custom_id", "eligibility"])
    df = pd.concat(frames, ignore_index=True)
    label_col = None
    if "eligbility" in df.columns:
        label_col = "eligbility"
    elif "eligibility" in df.columns:
        label_col = "eligibility"
    if label_col is None:
        return pd.DataFrame(columns=["custom_id", "eligibility"])

    # Handle both column name formats
    id_col = None
    if "UT (Unique WOS ID)" in df.columns:
        id_col = "UT (Unique WOS ID)"
    elif "ut_unique_wos_id_" in df.columns:
        id_col = "ut_unique_wos_id_"
    elif "custom_id" in df.columns:
        id_col = "custom_id"

    keep_cols = {id_col, "article_title", "abstract", "doi", "source", label_col} if id_col else {"article_title", "abstract", "doi", "source", label_col}
    df = df[[c for c in keep_cols if c in df.columns]]

    rename_map = {
        label_col: "eligibility",
        "article_title": "Article Title",
        "abstract": "Abstract",
        "doi": "DOI",
    }
    if id_col:
        rename_map[id_col] = "custom_id"

    df = df.rename(columns=rename_map)
    return df.drop_duplicates()


def _extract_split_threat_results(row) -> list | None:
    """Extract results list from a split threat task record (live or batch mode).

    Live output records store a pre-parsed ``results_payload`` dict directly.
    Batch API output records store the raw response under ``response.body``; for
    those we fall back to extracting and parsing the output_text.
    """
    rp = row.get("results_payload")
    if isinstance(rp, dict):
        results = rp.get("results")
        if isinstance(results, list):
            return results
    # Batch mode fallback: parse from response.body output text
    rec = row.to_dict() if hasattr(row, "to_dict") else row
    text = _output_text_from_record(rec)
    if text:
        parsed = _safe_json(text)
        if isinstance(parsed, dict):
            return parsed.get("results")
    return None


def _load_split_threat_predictions(task: str, run_name: str) -> pd.DataFrame:
    """Load predictions for a single split threat level (threats_l0/l1/l2).

    Only loads files matching ``{task}-*.jsonl`` so that multiple levels stored
    in the same run folder don't bleed into each other.
    """
    folder = BATCH_OUTPUTS_DIR / run_name
    if not folder.exists():
        raise FileNotFoundError(f"Batch outputs directory not found: {folder}")
    paths = sorted(folder.glob(f"{task}-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No {task}-*.jsonl files in {folder}")
    df = pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)
    level = task.split("_")[-1]  # "l0", "l1", or "l2"
    pred_col = f"pred_threat_{level}"
    df[pred_col] = df.apply(_extract_split_threat_results, axis=1)
    return df


def _load_split_ecosystem_predictions(task: str, run_name: str) -> pd.DataFrame:
    """Load predictions for a single split ecosystem level (ecosystems_realm/biome/efg).

    Only loads files matching ``{task}-*.jsonl`` so that multiple levels stored
    in the same run folder don't bleed into each other.

    Reuses ``_extract_split_threat_results`` since the output record format is
    identical: ``results_payload`` holds the direct JSON payload with a ``results``
    key (live mode), or the raw response body (batch mode).
    """
    folder = BATCH_OUTPUTS_DIR / run_name
    if not folder.exists():
        raise FileNotFoundError(f"Batch outputs directory not found: {folder}")
    paths = sorted(folder.glob(f"{task}-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No {task}-*.jsonl files in {folder}")
    df = pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)
    level = task.split("_")[-1]  # "realm", "biome", or "efg"
    pred_col = f"pred_{level}"   # "pred_realm", "pred_biome", "pred_efg"
    df[pred_col] = df.apply(_extract_split_threat_results, axis=1)
    return df


def load_predictions(task: str, run_name: str) -> pd.DataFrame:
    cfg = TASK_CONFIG[task]
    if cfg["task_type"] == "screening":
        return load_screening_predictions(run_name)

    if cfg["task_type"] in {"threats_l0", "threats_l1", "threats_l2"}:
        return _load_split_threat_predictions(task, run_name)

    if cfg["task_type"] in {"ecosystems_realm", "ecosystems_biome", "ecosystems_efg"}:
        return _load_split_ecosystem_predictions(task, run_name)

    folder = BATCH_OUTPUTS_DIR / run_name
    df = load_jsonl_folder(folder)

    # Ensure expected columns exist to avoid KeyErrors when runs are missing fields
    if "results_payload" not in df.columns:
        df["results_payload"] = None

    def _results_field(d, key):
        if isinstance(d, dict):
            node = d.get(key)
            if isinstance(node, dict):
                return node.get("results")
        return None

    def _first_result_field(d, key):
        if not isinstance(d, dict):
            return None
        results = d.get("results")
        if not isinstance(results, list):
            return None
        for item in results:
            if isinstance(item, dict):
                return item.get(key)
        return None

    if cfg["task_type"] in {"simple", "binary"}:
        df["output_text"] = df["response"].apply(_output_text_from_record)
        df["pred"] = df["output_text"].apply(lambda t: (_safe_json(t) or {}).get("results") if t else None)
    elif cfg["task_type"] == "threats":
        df["pred_threat_l0"] = df["results_payload"].apply(lambda d: _results_field(d, "threat_l0"))
        df["pred_threat_l1"] = df["results_payload"].apply(lambda d: _results_field(d, "threat_l1"))
        df["pred_threat_l2"] = df["results_payload"].apply(lambda d: _results_field(d, "threat_l2"))
    elif cfg["task_type"] == "geo":
        df["output_text"] = df["response"].apply(_output_text_from_record)
        df["payload"] = df["output_text"].apply(_safe_json)
        df["pred_scope"] = df["payload"].apply(lambda d: d.get("scope") if isinstance(d, dict) else None)
        df["pred_regions"] = df["payload"].apply(lambda d: d.get("regions") if isinstance(d, dict) else None)
        df["pred_subregions"] = df["payload"].apply(lambda d: d.get("subregions") or d.get("sub_regions") if isinstance(d, dict) else None)
        df["pred_countries"] = df["payload"].apply(lambda d: d.get("countries_iso3") if isinstance(d, dict) else None)
        df["pred_locales"] = df["payload"].apply(lambda d: d.get("locales") if isinstance(d, dict) else None)
        df["pred_locale_coordinates"] = df["payload"].apply(lambda d: d.get("locale_coordinates") if isinstance(d, dict) else None)
    elif cfg["task_type"] == "ecosystems":
        df["pred_realm"] = df["results_payload"].apply(lambda d: _results_field(d, "realms"))
        df["pred_biome"] = df["results_payload"].apply(lambda d: _results_field(d, "biomes"))
        df["pred_efg"] = df["results_payload"].apply(lambda d: _results_field(d, "efgs"))
    elif cfg["task_type"] == "taxa":
        df["output_text"] = df["response"].apply(_output_text_from_record)
        df["payload"] = df["output_text"].apply(_safe_json)
        df["pred_taxa"] = df["payload"].apply(lambda d: d.get("results") if isinstance(d, dict) else None)
        df = _expand_taxa_predictions(df)
    return df


def load_labels(task: str, label_path: Path | None = None, label_sheet: str | None = None) -> pd.DataFrame:
    cfg = TASK_CONFIG[task]
    path: Path = Path(label_path) if label_path is not None else cfg["label_path"]
    if cfg["task_type"] == "screening":
        return load_screening_labels(path)
    # Handle both CSV and Excel files. Workbook sheet selection is opt-in so
    # existing single-sheet Excel and CSV workflows remain unchanged.
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        read_kwargs = {"sheet_name": label_sheet} if label_sheet else {}
        df = pd.read_excel(path, **read_kwargs)
    df = df.rename(columns={"UT (Unique WOS ID)": "custom_id"})
    return df


def join_truth_pred(
    task: str,
    run_name: str,
    label_path: Path | None = None,
    label_sheet: str | None = None,
) -> pd.DataFrame:
    preds = load_predictions(task, run_name)
    if label_path is None:
        label_path = RUN_LABEL_PATH_OVERRIDES.get((task, run_name))
    labels = load_labels(task, label_path=label_path, label_sheet=label_sheet)

    # Restrict labels to the records we actually have predictions for
    if "custom_id" in preds.columns:
        pred_ids = set(preds["custom_id"].dropna().astype(str))
        labels = labels[labels["custom_id"].astype(str).isin(pred_ids)]

    # Align column names to a common scheme.
    join_how = "left"  # keep prediction rows; drop extra labels with no output
    keep_common = ["Article Title", "Abstract", "DOI", "source", "custom_id"]
    label_cols = [c for c in keep_common if c in labels.columns]
    if task == "driver":
        label_cols += ["driver"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
        merged = merged.rename(columns={"pred": "pred_driver"})
    elif task == "threats":
        label_cols += ["threats_l0", "threats_l1"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "geography":
        label_cols += ["region", "sub-region", "country"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "screening":
        label_cols += ["eligibility"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
        if "pred" in merged.columns:
            merged = merged.rename(columns={"pred": "pred_eligibility"})
    elif task == "ecosystems":
        label_cols += ["realm", "biome"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "taxa":
        label_cols += [c for c in ["kingdom", "phylum", "class", "order", "genus", "specie"] if c in labels.columns]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "threats_l0":
        label_cols += ["threats_l0"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "threats_l1":
        label_cols += ["threats_l1"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "threats_l2":
        merged = preds  # no ground-truth labels for l2
    elif task == "ecosystems_realm":
        label_cols += ["realm"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "ecosystems_biome":
        label_cols += ["biome"]
        merged = preds.merge(labels[label_cols], on="custom_id", how=join_how)
    elif task == "ecosystems_efg":
        merged = preds  # no ground-truth labels for EFG
    else:
        merged = preds
    return merged


def normalize_truth_pred(task: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return df with prediction columns ready plus map truth->pred column names for metrics."""
    col_map: dict[str, str] = {}
    if task == "driver":
        col_map = {"driver": "pred_driver"}
    elif task == "threats":
        col_map = {"threats_l0": "pred_threat_l0", "threats_l1": "pred_threat_l1"}
    elif task == "geography":
        df["pred_countries"] = df["pred_countries"].apply(lambda v: [c.upper() for c in v] if isinstance(v, list) else v)
        col_map = {"region": "pred_regions", "sub-region": "pred_subregions", "country": "pred_countries"}
    elif task == "screening":
        col_map = {"eligibility": "pred_eligibility"}
    elif task == "ecosystems":
        col_map = {"realm": "pred_realm", "biome": "pred_biome"}
    elif task == "taxa":
        truth_to_pred = {
            "kingdom": "pred_kingdom",
            "phylum": "pred_phylum",
            "class": "pred_class",
            "order": "pred_order",
            "genus": "pred_genus",
            "specie": "pred_species",
        }
        col_map = {t: p for t, p in truth_to_pred.items() if t in df.columns and p in df.columns}
    elif task == "threats_l0":
        col_map = {"threats_l0": "pred_threat_l0"}
    elif task == "threats_l1":
        col_map = {"threats_l1": "pred_threat_l1"}
    elif task == "threats_l2":
        col_map = {}  # no ground-truth labels
    elif task == "ecosystems_realm":
        col_map = {"realm": "pred_realm"}
    elif task == "ecosystems_biome":
        col_map = {"biome": "pred_biome"}
    elif task == "ecosystems_efg":
        col_map = {}  # no ground-truth labels for EFG
    return df, col_map

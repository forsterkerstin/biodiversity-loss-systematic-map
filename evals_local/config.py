from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG_PATH = ROOT / "checklists" / "mappings" / "repo_config.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_repo_config() -> dict:
    """Load repo-level JSON config used for non-secret repository paths."""
    if not REPO_CONFIG_PATH.exists():
        raise ValueError(f"Required config file is missing: {REPO_CONFIG_PATH}")
    try:
        with REPO_CONFIG_PATH.open(encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {REPO_CONFIG_PATH}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config file {REPO_CONFIG_PATH} must contain a JSON object.")
    return config


REPO_CONFIG = _load_repo_config()


def _require_repo_config_section(section: str) -> dict:
    section_config = REPO_CONFIG.get(section)
    if not isinstance(section_config, dict):
        raise ValueError(f"Required config section {section!r} is missing in {REPO_CONFIG_PATH}.")
    return section_config


def _resolve_repo_config_path(section: str, key: str) -> Path:
    """Read a required path from repo_config.json; relative paths resolved from ROOT."""
    section_config = _require_repo_config_section(section)
    raw = section_config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Required config key {section}.{key} is missing in {REPO_CONFIG_PATH}.")
    p = Path(raw.strip())
    return p if p.is_absolute() else ROOT / p


def _resolve_nested_repo_config_path(section: str, nested: str, key: str) -> Path:
    section_config = _require_repo_config_section(section)
    nested_config = section_config.get(nested)
    if not isinstance(nested_config, dict):
        raise ValueError(f"Required config section {section}.{nested} is missing in {REPO_CONFIG_PATH}.")
    raw = nested_config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Required config key {section}.{nested}.{key} is missing in {REPO_CONFIG_PATH}.")
    p = Path(raw.strip())
    return p if p.is_absolute() else ROOT / p



# ── Paths ─────────────────────────────────────────────────────────────────────

MAPPINGS_DIR      = _resolve_nested_repo_config_path("orchestrator", "paths", "mappings_dir")
BATCH_OUTPUTS_DIR = _resolve_nested_repo_config_path("orchestrator", "paths", "batch_outputs_dir")
BATCHES_DIR       = _resolve_nested_repo_config_path("orchestrator", "paths", "batches_dir")
LABELS_DIR        = _resolve_repo_config_path("evals", "labels_dir")
EVAL_OUTPUT_DIR   = _resolve_repo_config_path("evals", "output_dir")
GBIF_CACHE_PATH   = _resolve_repo_config_path("evals", "gbif_cache_path")


# ── Task registry ─────────────────────────────────────────────────────────────

# Default run-name suffixes per task when using a shared base name.
DEFAULT_RUN_SUFFIXES = {
    "driver":           "",
    "screening":        "-screen",
    "geography":        "-geography",
    "threats":          "-threats",
    "threats_l0":       "-threats",
    "threats_l1":       "-threats",
    "threats_l2":       "-threats",
    "ecosystems":       "-ecosystem",
    "ecosystems_realm": "-ecosystem",
    "ecosystems_biome": "-ecosystem",
    "ecosystems_efg":   "-ecosystem",
    "taxa":             "-taxa",
}

TASK_CONFIG: dict[str, dict] = {
    "driver": {
        "label_path": LABELS_DIR / "l1" / "L1) driver set.csv",
        "truth_cols": {"driver": "driver"},
        "task_type": "simple",
    },
    "screening": {
        "label_path": LABELS_DIR / "l0",
        "truth_cols": {"eligibility": "eligibility"},
        "task_type": "screening",
    },
    "threats": {
        "label_path": LABELS_DIR / "l2" / "L2) threats set.csv",
        "truth_cols": {"threats_l0": "threats_l0", "threats_l1": "threats_l1"},
        "task_type": "threats",
    },
    "threats_l0": {
        "label_path": LABELS_DIR / "l2" / "L2) threats set.csv",
        "truth_cols": {"threats_l0": "threats_l0"},
        "task_type": "threats_l0",
    },
    "threats_l1": {
        "label_path": LABELS_DIR / "l2" / "L2) threats set.csv",
        "truth_cols": {"threats_l1": "threats_l1"},
        "task_type": "threats_l1",
    },
    "threats_l2": {
        "label_path": LABELS_DIR / "l2" / "L2) threats set.csv",
        "truth_cols": {},
        "task_type": "threats_l2",
    },
    "geography": {
        "label_path": LABELS_DIR / "l3" / "L3) geography set.csv",
        "truth_cols": {"region": "region", "sub-region": "sub-region", "country": "country"},
        "task_type": "geo",
    },
    "ecosystems": {
        "label_path": LABELS_DIR / "l4" / "L4) ecosystem set.csv",
        "truth_cols": {"realm": "realm", "biome": "biome"},
        "task_type": "ecosystems",
    },
    "ecosystems_realm": {
        "label_path": LABELS_DIR / "l4" / "L4) ecosystem set.csv",
        "truth_cols": {"realm": "realm"},
        "task_type": "ecosystems_realm",
    },
    "ecosystems_biome": {
        "label_path": LABELS_DIR / "l4" / "L4) ecosystem set.csv",
        "truth_cols": {"biome": "biome"},
        "task_type": "ecosystems_biome",
    },
    "ecosystems_efg": {
        "label_path": LABELS_DIR / "l4" / "L4) ecosystem set.csv",
        "truth_cols": {},
        "task_type": "ecosystems_efg",
    },
    "taxa": {
        "label_path": LABELS_DIR / "l6" / "L6) taxa set.csv",
        "truth_cols": {
            "kingdom": "kingdom",
            "phylum":  "phylum",
            "class":   "class",
            "order":   "order",
            "genus":   "genus",
            "specie":  "specie",
        },
        "task_type": "taxa",
    },
}


# Run-specific truth sources for consistency-check evals that do not live under
# the default data/labels task folders.
RUN_LABEL_PATH_OVERRIDES: dict[tuple[str, str], Path] = {
    (
        "screening",
        "l0_cc1_search-sample_010726_medium_f1",
    ): ROOT / "data" / "consistency-check-datasets" / "screening" / "search-sample" / "search-sample.csv",
}

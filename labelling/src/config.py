from __future__ import annotations

import json
import os
from pathlib import Path

# Repo root — internal anchor for resolving relative paths; not user-configurable.
ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_CONFIG_PATH = ROOT_DIR / "checklists" / "mappings" / "repo_config.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_repo_config() -> dict:
    """Load repo-level JSON config used for non-secret repository settings."""
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


def _require_repo_config_string(section: str, key: str) -> str:
    section_config = _require_repo_config_section(section)
    raw = section_config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Required config key {section}.{key} is missing in {REPO_CONFIG_PATH}.")
    return raw.strip()


def _require_repo_config_int(section: str, key: str) -> int:
    section_config = _require_repo_config_section(section)
    raw = section_config.get(key)
    if type(raw) is not int:
        raise ValueError(f"Required integer config key {section}.{key} is missing in {REPO_CONFIG_PATH}.")
    return raw


def _require_repo_config_optional_int(section: str, key: str) -> int | None:
    section_config = _require_repo_config_section(section)
    raw = section_config.get(key)
    if raw is None:
        return None
    if type(raw) is not int:
        raise ValueError(f"Config key {section}.{key} must be an integer or null in {REPO_CONFIG_PATH}.")
    return raw


def _require_repo_config_bool(section: str, key: str) -> bool:
    section_config = _require_repo_config_section(section)
    raw = section_config.get(key)
    if not isinstance(raw, bool):
        raise ValueError(f"Required boolean config key {section}.{key} is missing in {REPO_CONFIG_PATH}.")
    return raw


def _resolve_repo_config_path(section: str, key: str) -> Path:
    """Read a required path from repo_config.json; relative paths resolved from ROOT_DIR."""
    raw = _require_repo_config_string(section, key)
    p = Path(raw)
    return p if p.is_absolute() else ROOT_DIR / p


def _resolve_nested_repo_config_path(section: str, nested: str, key: str) -> Path:
    section_config = _require_repo_config_section(section)
    nested_config = section_config.get(nested)
    if not isinstance(nested_config, dict):
        raise ValueError(f"Required config section {section}.{nested} is missing in {REPO_CONFIG_PATH}.")
    raw = nested_config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Required config key {section}.{nested}.{key} is missing in {REPO_CONFIG_PATH}.")
    p = Path(raw.strip())
    return p if p.is_absolute() else ROOT_DIR / p


def _require_prompt_key(key: str) -> str:
    prompts_config = _require_repo_config_section("prompts")
    keys_config = prompts_config.get("keys")
    if not isinstance(keys_config, dict):
        raise ValueError(f"Required config section 'prompts.keys' is missing in {REPO_CONFIG_PATH}.")
    raw = keys_config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Required config key prompts.keys.{key} is missing in {REPO_CONFIG_PATH}.")
    return raw.strip()


def ensure_artifact_dirs() -> None:
    """Create artifact directories if they do not exist."""
    for path in (DATASETS_DIR, BATCHES_DIR, BATCH_OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _get_env_file_value(key: str) -> str:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        raw_key, _, raw_value = line.partition("=")
        if raw_key.strip() != key:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value.strip()
    return ""


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or _get_env_file_value("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Required env var 'OPENAI_API_KEY' is not set.")
    return api_key


# ── Paths ─────────────────────────────────────────────────────────────────────

MAPPINGS_DIR      = _resolve_nested_repo_config_path("orchestrator", "paths", "mappings_dir")
PROMPTS_DIR       = _resolve_repo_config_path("prompts", "dir")
DATASETS_DIR      = _resolve_nested_repo_config_path("orchestrator", "paths", "datasets_dir")
BATCHES_DIR       = _resolve_nested_repo_config_path("orchestrator", "paths", "batches_dir")
BATCH_OUTPUTS_DIR = _resolve_nested_repo_config_path("orchestrator", "paths", "batch_outputs_dir")


# ── Orchestrator defaults ─────────────────────────────────────────────────────

LIMIT_DOCS               = _require_repo_config_optional_int("orchestrator", "limit_docs")
BATCH_SIZE               = _require_repo_config_int("orchestrator", "batch_size")
DEFAULT_MODEL            = _require_repo_config_string("orchestrator", "model")
DEFAULT_REASONING_EFFORT = _require_repo_config_string("orchestrator", "reasoning")
SUBMISSION_MODE          = _require_repo_config_string("orchestrator", "submission_mode").lower()
RUN_EVALS                = _require_repo_config_bool("orchestrator", "run_evals")
if SUBMISSION_MODE not in {"live", "batch"}:
    raise ValueError(f"Config key orchestrator.submission_mode must be 'live' or 'batch' in {REPO_CONFIG_PATH}.")
DEFAULT_COLUMNS = [
    "Document Type",
    "Authors",
    "Article Title",
    "Abstract",
    "DOI",
    "WoS Categories",
    "Publisher",
    "Publication Date",
    "Publication Year",
    "UT (Unique WOS ID)",
]


# ── Prompt keys ───────────────────────────────────────────────────────────────

PROMPT_KEY_DRIVER     = _require_prompt_key("driver")
PROMPT_KEY_SCREENING  = _require_prompt_key("screening")
PROMPT_KEY_GEOGRAPHY  = _require_prompt_key("geography")
PROMPT_KEY_TAXA       = _require_prompt_key("taxa")
PROMPT_KEY_STUDY      = _require_prompt_key("study")
PROMPT_KEY_ECOSYSTEMS = _require_prompt_key("ecosystems")
PROMPT_KEY_ECOSYSTEMS_CORE  = _require_prompt_key("ecosystems_core")
PROMPT_KEY_ECOSYSTEMS_REALM = _require_prompt_key("ecosystems_realm")
PROMPT_KEY_ECOSYSTEMS_BIOME = _require_prompt_key("ecosystems_biome")
PROMPT_KEY_ECOSYSTEMS_EFG   = _require_prompt_key("ecosystems_efg")
PROMPT_KEY_THREATS    = _require_prompt_key("threats")
PROMPT_KEY_THREATS_CORE = _require_prompt_key("threats_core")
PROMPT_KEY_THREATS_L0   = _require_prompt_key("threats_l0")
PROMPT_KEY_THREATS_L1   = _require_prompt_key("threats_l1")
PROMPT_KEY_THREATS_L2   = _require_prompt_key("threats_l2")

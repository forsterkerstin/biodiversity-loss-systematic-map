from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI

from .config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    BATCH_OUTPUTS_DIR,
    MAPPINGS_DIR,
    PROMPT_KEY_DRIVER,
    PROMPT_KEY_SCREENING,
    PROMPT_KEY_GEOGRAPHY,
    PROMPT_KEY_TAXA,
    PROMPT_KEY_STUDY,
    PROMPT_KEY_ECOSYSTEMS_CORE,
    PROMPT_KEY_ECOSYSTEMS_REALM,
    PROMPT_KEY_ECOSYSTEMS_BIOME,
    PROMPT_KEY_ECOSYSTEMS_EFG,
    PROMPT_KEY_THREATS_CORE,
    PROMPT_KEY_THREATS_L0,
    PROMPT_KEY_THREATS_L1,
    PROMPT_KEY_THREATS_L2,
)
from .batch_api import build_client, format_article
from .live_api import run_live_requests, serialize_live_body, normalize_request_payload


def _coerce_text_param(structured_output: Any) -> dict | None:
    if structured_output is None:
        return None
    text_format = structured_output
    if isinstance(structured_output, dict):
        text_format = structured_output.get("text") or structured_output.get("format") or structured_output
    if not text_format:
        return None
    if isinstance(text_format, dict) and "format" in text_format:
        return text_format
    return {"format": text_format}


def _normalize_request_dest(dest: Path) -> Path:
    if dest.parent.name.lower() == "data":
        return dest.parent.parent / "request" / dest.name
    return dest


def _custom_id(doc: dict | None) -> str:
    if not doc:
        return ""
    cid = doc.get("UT") or doc.get("id") or ""
    return str(cid)


def _get_prompt_config(prompts: dict | None, key: str) -> dict:
    if not prompts or key not in prompts:
        raise KeyError(f"Prompt config '{key}' not found in loaded prompts.")
    config = prompts[key]
    if not isinstance(config, dict):
        raise ValueError(f"Prompt config '{key}' must be a mapping.")
    return config


def _compose_prompt_config(prompts: dict | None, *keys: str) -> dict:
    parts: list[str] = []
    structured_output = None
    for key in keys:
        config = _get_prompt_config(prompts, key)
        text = str(config.get("system_prompt") or "").strip()
        if text:
            parts.append(text)
        if structured_output is None and config.get("structured_output") is not None:
            structured_output = config.get("structured_output")
    return {
        "system_prompt": "\n\n".join(parts).strip(),
        "structured_output": structured_output,
    }


THREAT_L0_ALLOWED_LABELS = [
    "Residential & Commercial Development",
    "Agriculture & Aquaculture",
    "Energy Production & Mining",
    "Transportation & Service Corridors",
    "Biological Resource Use",
    "Human Intrusions & Disturbance",
    "Natural System Modifications",
    "Invasive & Other Problematic Species, Genes & Diseases",
    "Pollution",
    "Climate Change & Severe Weather",
    "Other Options",
    "Unclear",
]

ECOSYSTEM_REALM_ALLOWED_LABELS = [
    "Terrestrial",
    "Subterranean",
    "Subterranean-Freshwater",
    "Subterranean-Marine",
    "Freshwater-Terrestrial",
    "Freshwater",
    "Freshwater-Marine",
    "Marine",
    "Marine-Terrestrial",
    "Marine-Freshwater-Terrestrial",
    "Not Applicable",
    "All realms",
]


def _build_threat_text_param(
    structured_output: Any,
    allowed_labels: Sequence[str] | None = None,
) -> dict | None:
    text_param = _coerce_text_param(structured_output)
    if text_param is None or allowed_labels is None:
        return text_param
    text_param = json.loads(json.dumps(text_param))
    schema = (
        text_param.get("format", {})
        .get("schema", {})
    )
    properties = schema.get("properties", {})
    results = properties.get("results", {})
    items = results.get("items")
    if isinstance(items, dict):
        seen: set[str] = set()
        enum_values: list[str] = []
        for label in allowed_labels:
            value = str(label)
            if value in seen:
                continue
            enum_values.append(value)
            seen.add(value)
        items["enum"] = enum_values
    return text_param


def _build_ecosystem_text_param(
    structured_output: Any,
    allowed_labels: Sequence[str] | None = None,
    stop_reason_values: Sequence[str] | None = None,
) -> dict | None:
    text_param = _coerce_text_param(structured_output)
    if text_param is None:
        return None
    if allowed_labels is None and stop_reason_values is None:
        return text_param
    text_param = json.loads(json.dumps(text_param))
    schema = text_param.get("format", {}).get("schema", {})
    properties = schema.get("properties", {})

    if allowed_labels is not None:
        results = properties.get("results", {})
        items = results.get("items")
        if isinstance(items, dict):
            seen: set[str] = set()
            enum_values: list[str] = []
            for label in allowed_labels:
                value = str(label)
                if value in seen:
                    continue
                enum_values.append(value)
                seen.add(value)
            items["enum"] = enum_values

    if stop_reason_values is not None:
        stop_reason = properties.get("stop_reason")
        if isinstance(stop_reason, dict):
            seen: set[str] = set()
            enum_values: list[str] = []
            for value in stop_reason_values:
                value = str(value)
                if value in seen:
                    continue
                enum_values.append(value)
                seen.add(value)
            stop_reason["enum"] = enum_values

    return text_param


def _build_standard_request(
    doc: dict,
    prompt_config: dict,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
) -> dict:
    system_prompt = prompt_config["system_prompt"]
    text_param = _coerce_text_param(prompt_config.get("structured_output"))
    prompt = format_article(doc)
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if text_param:
        body["text"] = text_param
    return {
        "custom_id": _custom_id(doc),
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def _write_request_lines(documents: Sequence[dict], dest: Path, builder) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(builder(doc), ensure_ascii=False) + "\n")
    return dest


def _load_processed_custom_ids(out_path: Path) -> set[str]:
    processed: set[str] = set()
    if not out_path.exists():
        return processed
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = record.get("custom_id")
        if cid:
            processed.add(str(cid))
    return processed


def _extract_output_text(resp: Any, body: Any) -> str | None:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text)
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        if "output_text" in body:
            return body["output_text"]
        output = body.get("output") or body.get("outputs")
        if isinstance(output, list):
            for candidate in output:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content") or []
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text_block = block.get("text")
                    if isinstance(text_block, dict):
                        text_value = text_block.get("value") or text_block.get("content")
                        if text_value:
                            return text_value
                    if isinstance(text_block, str) and text_block:
                        return text_block
    return None


def _parse_output_payload(resp: Any, body: Any) -> tuple[dict | None, str | None]:
    output_text = _extract_output_text(resp, body)
    if not output_text:
        return None, "Missing output_text on response."
    try:
        return json.loads(str(output_text).strip()), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in response: {exc}"


@dataclass
class TaskDefinition:
    name: str
    prompt_key: str
    supports_batch: bool = True
    supports_live: bool = True
    output_prefix: str | None = None

    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        raise NotImplementedError

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        raise NotImplementedError

    def batch_metadata(self, run_name: str) -> dict:
        return {"run": run_name, "type": self.name}


@dataclass
class SimpleTask(TaskDefinition):
    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        prompt_config = _get_prompt_config(prompts, self.prompt_key)
        dest = _normalize_request_dest(Path(dest))
        return _write_request_lines(
            documents,
            dest,
            lambda doc: _build_standard_request(doc, prompt_config, model=model, reasoning_effort=reasoning_effort),
        )

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        client = client or build_client()
        prefix = self.output_prefix or self.name
        return run_live_requests(
            request_paths,
            output_dir=output_dir,
            client=client,
            default_model=default_model,
            default_reasoning=default_reasoning,
            output_prefix=prefix,
            skip_existing=skip_existing,
        )


@dataclass
class EcosystemTask(TaskDefinition):
    mapping_filename: str = "ecosystem_typology_1_3.json"

    def _load_mapping(self) -> dict:
        path = MAPPINGS_DIR / self.mapping_filename
        if not path.exists():
            raise FileNotFoundError(f"Ecosystem mapping file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _extract_labels(payload: dict | None) -> list[str]:
        if not payload:
            return []
        results = payload.get("results")
        if isinstance(results, list):
            return [str(item) for item in results if item is not None]
        return []

    @staticmethod
    def _should_continue(payload: dict | None) -> bool:
        if not payload:
            return False
        return payload.get("stop_reason") == "continue"

    def _build_biome_candidates(self, realm_labels: list[str], mapping: dict) -> tuple[list[dict], dict[str, list]]:
        candidates: list[dict] = []
        lookup: dict[str, list] = {}
        seen: set[str] = set()
        ci_mapping = {k.lower(): v for k, v in mapping.items()}
        for label in realm_labels:
            realm_info = mapping.get(label) or ci_mapping.get(label.lower(), {})
            for biome in realm_info.get("biomes", []):
                name = biome.get("name")
                if not name or name in seen:
                    continue
                desc = biome.get("short_desc") or biome.get("desc", "")
                candidates.append({"name": name, "desc": desc})
                lookup[name] = biome.get("efg", [])
                seen.add(name)
        return candidates, lookup

    def _build_efg_candidates(self, biome_labels: list[str], biome_lookup: dict[str, list]) -> list[dict]:
        candidates: list[dict] = []
        seen: set[str] = set()
        ci_biome_lookup = {k.lower(): v for k, v in biome_lookup.items()}
        for biome_name in biome_labels:
            for efg in biome_lookup.get(biome_name) or ci_biome_lookup.get(biome_name.lower(), []):
                name = efg.get("name")
                if not name or name in seen:
                    continue
                desc = efg.get("short_desc") or efg.get("desc", "")
                candidates.append({"name": name, "desc": desc})
                seen.add(name)
        return candidates


@dataclass
class ThreatTask(TaskDefinition):
    mapping_filename: str = "threats_classification.json"

    def _load_mapping(self) -> dict:
        path = MAPPINGS_DIR / self.mapping_filename
        if not path.exists():
            raise FileNotFoundError(f"Threat mapping file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _extract_labels(payload: dict | None) -> list[str]:
        if not payload:
            return []
        results = payload.get("results")
        if isinstance(results, list):
            return [str(item) for item in results if item is not None]
        return []

    @staticmethod
    def _should_continue(payload: dict | None) -> bool:
        if not payload:
            return False
        return payload.get("stop_reason") == "continue"

    def _build_l1_candidates(self, threat_l0_labels: list[str], mapping: dict) -> tuple[list[dict], dict[str, list]]:
        candidates: list[dict] = []
        l2_lookup: dict[str, list] = {}
        seen: set[str] = set()
        threats_map = mapping.get("threats", {})
        ci_threats_map = {k.lower(): v for k, v in threats_map.items()}
        for label in threat_l0_labels:
            l0_entry = threats_map.get(label) or ci_threats_map.get(label.lower(), {})
            for l1_name, l1_data in l0_entry.get("level1", {}).items():
                if not l1_name or l1_name in seen:
                    continue
                candidates.append({"name": l1_name, "desc": l1_data.get("examples", "")})
                seen.add(l1_name)
                l2_lookup[l1_name] = []
                for l2_name, l2_data in l1_data.get("level2", {}).items():
                    l2_lookup[l1_name].append({"name": l2_name, "desc": l2_data.get("examples", "")})
        # print(f"Final L1 loopkup: {l2_lookup}")
        return candidates, l2_lookup

    def _build_l2_candidates(self, threat_l1_labels: list[str], l2_lookup: dict[str, list]) -> list[dict]:
        candidates: list[dict] = []
        seen: set[str] = set()
        ci_l2_lookup = {k.lower(): v for k, v in l2_lookup.items()}
        for l1_label in threat_l1_labels:
            # print(f"Looking up L2 for L1 label: {l1_label}")
            for l2 in l2_lookup.get(l1_label) or ci_l2_lookup.get(l1_label.lower(), []):
                # print(f"  Found L2 candidate: {l2}")
                name = l2.get("name")
                if not name or name in seen:
                    continue
                candidates.append({"name": name, "desc": l2.get("desc", "")})
                seen.add(name)
        # print(f"Final L2 candidates: {candidates}")
        return candidates


# ── Split threat task helpers ──────────────────────────────────────────────────

def _normalize_split_stage_record(rec: dict) -> dict | None:
    """Normalize either a parsed live split-stage record or raw batch envelope.

    Returns a common shape:
      {custom_id, response_text, results_payload, candidates_passed, error}
    """
    if not isinstance(rec, dict):
        return None

    custom_id = rec.get("custom_id")
    if not custom_id:
        return None

    # Live-mode split tasks already write the normalized shape directly.
    if any(key in rec for key in ("results_payload", "response_text", "candidates_passed")):
        return {
            "custom_id": str(custom_id),
            "response_text": rec.get("response_text"),
            "results_payload": rec.get("results_payload"),
            "candidates_passed": rec.get("candidates_passed"),
            "error": rec.get("error"),
        }

    # Batch mode stores the raw response envelope; extract the same parsed fields.
    response = rec.get("response") if isinstance(rec.get("response"), dict) else {}
    body = response.get("body") if isinstance(response, dict) else None
    result_payload, parse_error = _parse_output_payload(None, body)
    response_text = _extract_output_text(None, body)

    error = rec.get("error")
    if error is None and parse_error:
        error = {"message": parse_error}

    return {
        "custom_id": str(custom_id),
        "response_text": response_text,
        "results_payload": result_payload,
        "candidates_passed": rec.get("candidates_passed"),
        "error": error,
    }


def _load_threat_level_outputs(run_name: str, level_prefix: str) -> dict[str, dict]:
    """Load all JSONL records from a prior threats level run, keyed by custom_id.

    Globs ``{BATCH_OUTPUTS_DIR}/{run_name}/{level_prefix}-*.jsonl`` so naming must
    follow the convention threats_l0-*, threats_l1-*, threats_l2-*.
    """
    prior_dir = BATCH_OUTPUTS_DIR / run_name
    result: dict[str, dict] = {}
    if not prior_dir.exists():
        return result
    for path in sorted(prior_dir.glob(f"{level_prefix}-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = _normalize_split_stage_record(rec)
            if normalized:
                result[normalized["custom_id"]] = normalized
    return result


def _sort_requests_by_candidates(records: list[dict]) -> list[dict]:
    """Sort request records by their candidates_passed names for cache locality.

    Groups documents that share the same candidate list (i.e. came from the same
    parent labels) so that adjacent requests in the batch have identical candidate
    text — maximising prompt-cache hits on the candidate suffix.
    """
    return sorted(
        records,
        key=lambda r: tuple(sorted(c["name"] for c in (r.get("candidates_passed") or []))),
    )


def _run_split_threat_live(
    task_name: str,
    output_prefix: str,
    request_paths: Iterable[Path | str],
    output_dir: Path,
    client: OpenAI,
    default_model: str,
    default_reasoning: str,
    skip_existing: bool,
) -> list[dict]:
    """Shared live runner for ThreatL0 / ThreatL1 / ThreatL2.

    Reads pre-built request JSONL, calls the API once per record, and writes a
    clean per-stage output record:
      {custom_id, response_text, results_payload, candidates_passed, error}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for req_path in request_paths:
        req_path = Path(req_path)
        out_path = output_dir / f"{output_prefix}-{req_path.stem}.jsonl"
        processed_ids = _load_processed_custom_ids(out_path) if skip_existing else set()
        skipped_count = 0
        rendered_count = 0
        mode = "a" if out_path.exists() else "w"

        with req_path.open("r", encoding="utf-8") as src, out_path.open(mode, encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    dst.write(json.dumps({
                        "custom_id": None,
                        "response_text": None,
                        "results_payload": None,
                        "candidates_passed": None,
                        "error": {"message": f"Invalid request JSON: {exc}"},
                    }, ensure_ascii=False) + "\n")
                    rendered_count += 1
                    continue

                custom_id = payload.get("custom_id")
                if custom_id:
                    custom_id = str(custom_id)
                # candidates_passed is stored alongside body in the request record
                candidates_passed = payload.get("candidates_passed")

                if skip_existing and custom_id and custom_id in processed_ids:
                    print(f"Skipping already processed custom_id {custom_id} from {req_path}")
                    skipped_count += 1
                    continue

                # normalize_request_payload only reads payload["body"], so
                # candidates_passed (a top-level sibling) is not submitted to the API
                _, body = normalize_request_payload(
                    payload, default_model=default_model, default_reasoning=default_reasoning
                )

                try:
                    resp = client.responses.create(**body)
                    resp_body = serialize_live_body(resp)
                    result_payload, parse_error = _parse_output_payload(resp, resp_body)
                    response_text = _extract_output_text(resp, resp_body)
                    record: dict[str, Any] = {
                        "custom_id": custom_id,
                        "response_text": response_text,
                        "results_payload": result_payload,
                        "candidates_passed": candidates_passed,
                        "error": {"message": parse_error} if parse_error else None,
                    }
                except Exception as exc:
                    record = {
                        "custom_id": custom_id,
                        "response_text": None,
                        "results_payload": None,
                        "candidates_passed": candidates_passed,
                        "error": {"message": str(exc)},
                    }

                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                rendered_count += 1

        entries.append({
            "mode": "live",
            "request_path": str(req_path),
            "output_path": str(out_path),
            "status": "completed",
            "responses": rendered_count,
            "skipped": skipped_count,
        })
        print(f"Wrote {rendered_count} {task_name} outputs to {out_path} (skipped {skipped_count})")

    return entries


# ── Split threat tasks ─────────────────────────────────────────────────────────

@dataclass
class ThreatL0Task(TaskDefinition):
    """Single-stage threat task: article_text → threat_l0 labels only.

    Supports batch mode (single-turn call, no conversation chaining).
    Output format per record: {custom_id, response_text, results_payload, error}
    """

    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        prompt_config = _get_prompt_config(prompts, self.prompt_key)
        system_prompt = prompt_config["system_prompt"]
        text_param = _build_threat_text_param(prompt_config.get("structured_output"), THREAT_L0_ALLOWED_LABELS)
        dest = _normalize_request_dest(Path(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for doc in documents:
                article_text = format_article(doc)
                body: dict[str, Any] = {
                    "model": model,
                    "instructions": system_prompt,
                    "input": [{"role": "user", "content": f"ARTICLE_TEXT:\\n{article_text}"}],
                }
                if reasoning_effort:
                    body["reasoning"] = {"effort": reasoning_effort}
                if text_param:
                    body["text"] = text_param
                record = {
                    "custom_id": _custom_id(doc),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return dest

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        return _run_split_threat_live(
            task_name=self.name,
            output_prefix=self.output_prefix or self.name,
            request_paths=request_paths,
            output_dir=output_dir,
            client=client or build_client(),
            default_model=default_model,
            default_reasoning=default_reasoning,
            skip_existing=skip_existing,
        )


@dataclass
class ThreatL1Task(ThreatTask):
    """L1-only threat task: reads threats_l0 outputs, builds L1 candidates, calls model.

    Reconstructs the L0→L1 conversation from saved L0 outputs — no server-side
    response chaining required, so batch mode is supported.

    Prior outputs are looked up automatically from
    ``{BATCH_OUTPUTS_DIR}/{run_name}/threats_l0-*.jsonl``.

    Output format per record:
      {custom_id, response_text, results_payload, candidates_passed, error}
    """

    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        prompt_config = _compose_prompt_config(prompts, PROMPT_KEY_THREATS_CORE, self.prompt_key)
        system_prompt = prompt_config["system_prompt"]
        mapping = self._load_mapping()

        dest = _normalize_request_dest(Path(dest))
        run_name = dest.parent.parent.name
        l0_outputs = _load_threat_level_outputs(run_name, "threats_l0")
        if not l0_outputs:
            print(
                f"[threats_l1] WARNING: No threats_l0 outputs found for run '{run_name}'. "
                "All documents will be skipped."
            )

        records: list[dict] = []
        skipped = 0

        for doc in documents:
            cid = _custom_id(doc)
            l0_rec = l0_outputs.get(cid)

            if not l0_rec or l0_rec.get("error") or not l0_rec.get("results_payload"):
                print(f"[threats_l1] Skipping {cid}: no valid threats_l0 output")
                skipped += 1
                continue

            l0_results = l0_rec["results_payload"]
            l0_labels = self._extract_labels(l0_results)
            if not l0_labels or not self._should_continue(l0_results):
                stop_reason = l0_results.get("stop_reason") if isinstance(l0_results, dict) else None
                print(
                    f"[threats_l1] Skipping {cid}: terminal threats_l0 output "
                    f"(labels={l0_labels}, stop_reason={stop_reason})"
                )
                skipped += 1
                continue

            l1_candidates, _l2_lookup = self._build_l1_candidates(l0_labels, mapping)
            if not l1_candidates:
                print(f"[threats_l1] Skipping {cid}: no L1 candidates for L0 labels {l0_labels}")
                skipped += 1
                continue

            article_text = format_article(doc)
            l0_response_text = l0_rec.get("response_text") or ""

            body: dict[str, Any] = {
                "model": model,
                "instructions": system_prompt,
                "input": [
                    {"role": "user", "content": f"ARTICLE_TEXT:\\n{article_text}"},
                    {"role": "assistant", "content": l0_response_text},
                    {"role": "user", "content": f"threats_l1_candidates:\\n{json.dumps(l1_candidates, indent=2)}"},
                ],
            }
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}
            text_param = _build_threat_text_param(
                prompt_config.get("structured_output"),
                [c.get("name") for c in l1_candidates if c.get("name")] + ["Unclear"],
            )
            if text_param:
                body["text"] = text_param

            # candidates_passed stored alongside body (not inside it) so it is not
            # submitted to the API but is available to run_live for the output record
            records.append({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
                "candidates_passed": l1_candidates,
            })

        # Sort by L1 candidate names so documents with identical candidate sets are
        # adjacent — maximises prompt-cache locality on the candidate suffix
        records = _sort_requests_by_candidates(records)

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if skipped:
            print(f"[threats_l1] Skipped {skipped} documents (no valid L0 output or terminal labels)")

        return dest

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        return _run_split_threat_live(
            task_name=self.name,
            output_prefix=self.output_prefix or self.name,
            request_paths=request_paths,
            output_dir=output_dir,
            client=client or build_client(),
            default_model=default_model,
            default_reasoning=default_reasoning,
            skip_existing=skip_existing,
        )


@dataclass
class ThreatL2Task(ThreatTask):
    """L2-only threat task: reads threats_l0 + threats_l1 outputs, builds L2 candidates, calls model.

    Reconstructs the full L0→L1→L2 conversation from saved outputs — batch mode supported.

    Prior outputs are looked up automatically from
    ``{BATCH_OUTPUTS_DIR}/{run_name}/threats_l0-*.jsonl`` and
    ``{BATCH_OUTPUTS_DIR}/{run_name}/threats_l1-*.jsonl``.

    Output format per record:
      {custom_id, response_text, results_payload, candidates_passed, error}
    """

    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        prompt_config = _compose_prompt_config(prompts, PROMPT_KEY_THREATS_CORE, self.prompt_key)
        system_prompt = prompt_config["system_prompt"]
        mapping = self._load_mapping()

        dest = _normalize_request_dest(Path(dest))
        run_name = dest.parent.parent.name
        l0_outputs = _load_threat_level_outputs(run_name, "threats_l0")
        l1_outputs = _load_threat_level_outputs(run_name, "threats_l1")
        if not l0_outputs:
            print(f"[threats_l2] WARNING: No threats_l0 outputs found for run '{run_name}'.")
        if not l1_outputs:
            print(f"[threats_l2] WARNING: No threats_l1 outputs found for run '{run_name}'.")

        records: list[dict] = []
        skipped = 0
        skipped_missing_l0 = 0
        skipped_terminal_l0 = 0
        skipped_missing_l1 = 0
        skipped_terminal_l1 = 0
        skipped_no_l1_candidates = 0
        skipped_no_l2_candidates = 0

        for doc in documents:
            cid = _custom_id(doc)
            l0_rec = l0_outputs.get(cid)
            l1_rec = l1_outputs.get(cid)

            if not l0_rec or l0_rec.get("error") or not l0_rec.get("results_payload"):
                skipped += 1
                skipped_missing_l0 += 1
                continue
            l0_results = l0_rec["results_payload"]
            l0_labels = self._extract_labels(l0_results)
            if not l0_labels or not self._should_continue(l0_results):
                stop_reason = l0_results.get("stop_reason") if isinstance(l0_results, dict) else None
                print(
                    f"[threats_l2] Skipping {cid}: terminal threats_l0 output "
                    f"(labels={l0_labels}, stop_reason={stop_reason})"
                )
                skipped += 1
                skipped_terminal_l0 += 1
                continue

            if not l1_rec or l1_rec.get("error") or not l1_rec.get("results_payload"):
                skipped += 1
                skipped_missing_l1 += 1
                continue
            l1_results = l1_rec["results_payload"]
            l1_labels = self._extract_labels(l1_results)
            if not l1_labels or not self._should_continue(l1_results):
                stop_reason = l1_results.get("stop_reason") if isinstance(l1_results, dict) else None
                print(
                    f"[threats_l2] Skipping {cid}: terminal threats_l1 output "
                    f"(labels={l1_labels}, stop_reason={stop_reason})"
                )
                skipped += 1
                skipped_terminal_l1 += 1
                continue

            # Re-derive L1 candidates (for history reconstruction) and L2 lookup
            l1_candidates, l2_lookup = self._build_l1_candidates(l0_labels, mapping)
            if not l1_candidates:
                skipped += 1
                skipped_no_l1_candidates += 1
                continue

            l2_candidates = self._build_l2_candidates(l1_labels, l2_lookup)
            if not l2_candidates:
                skipped += 1
                skipped_no_l2_candidates += 1
                continue

            article_text = format_article(doc)
            l0_response_text = l0_rec.get("response_text") or ""
            l1_response_text = l1_rec.get("response_text") or ""

            body: dict[str, Any] = {
                "model": model,
                "instructions": system_prompt,
                "input": [
                    {"role": "user", "content": f"ARTICLE_TEXT:\\n{article_text}"},
                    {"role": "assistant", "content": l0_response_text},
                    {"role": "user", "content": f"threats_l1_candidates:\\n{json.dumps(l1_candidates, indent=2)}"},
                    {"role": "assistant", "content": l1_response_text},
                    {"role": "user", "content": f"threats_l2_candidates:\\n{json.dumps(l2_candidates, indent=2)}"},
                ],
            }
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}
            text_param = _build_threat_text_param(
                prompt_config.get("structured_output"),
                [c.get("name") for c in l2_candidates if c.get("name")] + ["Unclear"],
            )
            if text_param:
                body["text"] = text_param

            records.append({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
                "candidates_passed": l2_candidates,
            })

        # Sort by L2 candidate names for cache locality
        records = _sort_requests_by_candidates(records)

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if skipped:
            print(f"[threats_l2] Skipped {skipped} documents (missing L0/L1 outputs or terminal labels)")
            print(
                "[threats_l2] Skip summary: "
                f"missing/bad L0={skipped_missing_l0}, "
                f"terminal L0={skipped_terminal_l0}, "
                f"missing/bad L1={skipped_missing_l1}, "
                f"terminal L1={skipped_terminal_l1}, "
                f"no L1 candidates={skipped_no_l1_candidates}, "
                f"no L2 candidates={skipped_no_l2_candidates}"
            )

        return dest

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        return _run_split_threat_live(
            task_name=self.name,
            output_prefix=self.output_prefix or self.name,
            request_paths=request_paths,
            output_dir=output_dir,
            client=client or build_client(),
            default_model=default_model,
            default_reasoning=default_reasoning,
            skip_existing=skip_existing,
        )


# ── Split ecosystem task helpers ───────────────────────────────────────────────

def _load_ecosystem_level_outputs(run_name: str, level_prefix: str) -> dict[str, dict]:
    """Load all JSONL records from a prior ecosystems level run, keyed by custom_id.

    Globs ``{BATCH_OUTPUTS_DIR}/{run_name}/{level_prefix}-*.jsonl`` so naming must
    follow the convention ecosystems_realm-*, ecosystems_biome-*, ecosystems_efg-*.
    """
    prior_dir = BATCH_OUTPUTS_DIR / run_name
    result: dict[str, dict] = {}
    if not prior_dir.exists():
        return result
    for path in sorted(prior_dir.glob(f"{level_prefix}-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = _normalize_split_stage_record(rec)
            if normalized:
                result[normalized["custom_id"]] = normalized
    return result


def _run_split_ecosystem_live(
    task_name: str,
    output_prefix: str,
    request_paths: Iterable[Path | str],
    output_dir: Path,
    client: OpenAI,
    default_model: str,
    default_reasoning: str,
    skip_existing: bool,
) -> list[dict]:
    """Shared live runner for EcosystemRealm / EcosystemBiome / EcosystemEFG.

    Reads pre-built request JSONL, calls the API once per record, and writes a
    clean per-stage output record:
      {custom_id, response_text, results_payload, candidates_passed, error}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

    for req_path in request_paths:
        req_path = Path(req_path)
        out_path = output_dir / f"{output_prefix}-{req_path.stem}.jsonl"
        processed_ids = _load_processed_custom_ids(out_path) if skip_existing else set()
        skipped_count = 0
        rendered_count = 0
        mode = "a" if out_path.exists() else "w"

        with req_path.open("r", encoding="utf-8") as src, out_path.open(mode, encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    dst.write(json.dumps({
                        "custom_id": None,
                        "response_text": None,
                        "results_payload": None,
                        "candidates_passed": None,
                        "error": {"message": f"Invalid request JSON: {exc}"},
                    }, ensure_ascii=False) + "\n")
                    rendered_count += 1
                    continue

                custom_id = payload.get("custom_id")
                if custom_id:
                    custom_id = str(custom_id)
                candidates_passed = payload.get("candidates_passed")

                if skip_existing and custom_id and custom_id in processed_ids:
                    print(f"Skipping already processed custom_id {custom_id} from {req_path}")
                    skipped_count += 1
                    continue

                _, body = normalize_request_payload(
                    payload, default_model=default_model, default_reasoning=default_reasoning
                )

                try:
                    resp = client.responses.create(**body)
                    resp_body = serialize_live_body(resp)
                    result_payload, parse_error = _parse_output_payload(resp, resp_body)
                    response_text = _extract_output_text(resp, resp_body)
                    record: dict[str, Any] = {
                        "custom_id": custom_id,
                        "response_text": response_text,
                        "results_payload": result_payload,
                        "candidates_passed": candidates_passed,
                        "error": {"message": parse_error} if parse_error else None,
                    }
                except Exception as exc:
                    record = {
                        "custom_id": custom_id,
                        "response_text": None,
                        "results_payload": None,
                        "candidates_passed": candidates_passed,
                        "error": {"message": str(exc)},
                    }

                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                rendered_count += 1

        entries.append({
            "mode": "live",
            "request_path": str(req_path),
            "output_path": str(out_path),
            "status": "completed",
            "responses": rendered_count,
            "skipped": skipped_count,
        })
        print(f"Wrote {rendered_count} {task_name} outputs to {out_path} (skipped {skipped_count})")

    return entries


# ── Split ecosystem tasks ──────────────────────────────────────────────────────

@dataclass
class EcosystemRealmTask(TaskDefinition):
    """Single-stage ecosystem task: article_text → realm labels only.

    Supports batch mode (single-turn call, no conversation chaining).
    Output format per record: {custom_id, response_text, results_payload, candidates_passed, error}
    """

    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        prompt_config = _compose_prompt_config(prompts, PROMPT_KEY_ECOSYSTEMS_CORE, self.prompt_key)
        system_prompt = prompt_config["system_prompt"]
        text_param = _build_ecosystem_text_param(
            prompt_config.get("structured_output"),
            allowed_labels=ECOSYSTEM_REALM_ALLOWED_LABELS,
            stop_reason_values=["stop", "continue"],
        )
        dest = _normalize_request_dest(Path(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for doc in documents:
                article_text = format_article(doc)
                body: dict[str, Any] = {
                    "model": model,
                    "instructions": system_prompt,
                    "input": [{"role": "user", "content": f"ARTICLE_TEXT:\\n{article_text}"}],
                }
                if reasoning_effort:
                    body["reasoning"] = {"effort": reasoning_effort}
                if text_param:
                    body["text"] = text_param
                record = {
                    "custom_id": _custom_id(doc),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return dest

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        return _run_split_ecosystem_live(
            task_name=self.name,
            output_prefix=self.output_prefix or self.name,
            request_paths=request_paths,
            output_dir=output_dir,
            client=client or build_client(),
            default_model=default_model,
            default_reasoning=default_reasoning,
            skip_existing=skip_existing,
        )


@dataclass
class EcosystemBiomeTask(EcosystemTask):
    """Biome-only ecosystem task: reads ecosystems_realm outputs, builds biome candidates, calls model.

    Reconstructs the realm→biome conversation from saved realm outputs — no server-side
    response chaining required, so batch mode is supported.

    Prior outputs are looked up automatically from
    ``{BATCH_OUTPUTS_DIR}/{run_name}/ecosystems_realm-*.jsonl``.

    Output format per record:
      {custom_id, response_text, results_payload, candidates_passed, error}
    """

    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        prompt_config = _compose_prompt_config(prompts, PROMPT_KEY_ECOSYSTEMS_CORE, self.prompt_key)
        system_prompt = prompt_config["system_prompt"]
        mapping = self._load_mapping()

        dest = _normalize_request_dest(Path(dest))
        run_name = dest.parent.parent.name
        realm_outputs = _load_ecosystem_level_outputs(run_name, "ecosystems_realm")
        if not realm_outputs:
            print(
                f"[ecosystems_biome] WARNING: No ecosystems_realm outputs found for run '{run_name}'. "
                "All documents will be skipped."
            )

        records: list[dict] = []
        skipped = 0
        skipped_missing_realm = 0
        skipped_terminal_realm = 0
        skipped_no_biome_candidates = 0

        for doc in documents:
            cid = _custom_id(doc)
            realm_rec = realm_outputs.get(cid)

            if not realm_rec or realm_rec.get("error") or not realm_rec.get("results_payload"):
                print(f"[ecosystems_biome] Skipping {cid}: no valid ecosystems_realm output")
                skipped += 1
                skipped_missing_realm += 1
                continue

            realm_results = realm_rec["results_payload"]
            realm_labels = self._extract_labels(realm_results)
            if not realm_labels or not self._should_continue(realm_results):
                stop_reason = realm_results.get("stop_reason") if isinstance(realm_results, dict) else None
                print(
                    f"[ecosystems_biome] Skipping {cid}: terminal ecosystems_realm output "
                    f"(labels={realm_labels}, stop_reason={stop_reason})"
                )
                skipped += 1
                skipped_terminal_realm += 1
                continue

            biome_candidates, _biome_lookup = self._build_biome_candidates(realm_labels, mapping)
            if not biome_candidates:
                print(f"[ecosystems_biome] Skipping {cid}: no biome candidates for realm labels {realm_labels}")
                skipped += 1
                skipped_no_biome_candidates += 1
                continue

            article_text = format_article(doc)
            realm_response_text = realm_rec.get("response_text") or ""

            body: dict[str, Any] = {
                "model": model,
                "instructions": system_prompt,
                "input": [
                    {"role": "user", "content": f"ARTICLE_TEXT:\\n{article_text}"},
                    {"role": "assistant", "content": realm_response_text},
                    {"role": "user", "content": f"BIOME_CANDIDATES:\\n{json.dumps(biome_candidates, indent=2)}"},
                ],
            }
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}
            text_param = _build_ecosystem_text_param(
                prompt_config.get("structured_output"),
                allowed_labels=[c.get("name") for c in biome_candidates if c.get("name")] + ["Unclear", "Not Applicable"],
                stop_reason_values=["stop", "continue"],
            )
            if text_param:
                body["text"] = text_param

            # candidates_passed stored alongside body (not inside it) so it is not
            # submitted to the API but is available to run_live for the output record
            records.append({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
                "candidates_passed": biome_candidates,
            })

        # Sort by biome candidate names so documents with identical candidate sets are
        # adjacent — maximises prompt-cache locality on the candidate suffix
        records = _sort_requests_by_candidates(records)

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if skipped:
            print(f"[ecosystems_biome] Skipped {skipped} documents (no valid realm output or terminal labels)")
            print(
                "[ecosystems_biome] Skip summary: "
                f"missing/bad realm={skipped_missing_realm}, "
                f"terminal realm={skipped_terminal_realm}, "
                f"no biome candidates={skipped_no_biome_candidates}"
            )

        return dest

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        return _run_split_ecosystem_live(
            task_name=self.name,
            output_prefix=self.output_prefix or self.name,
            request_paths=request_paths,
            output_dir=output_dir,
            client=client or build_client(),
            default_model=default_model,
            default_reasoning=default_reasoning,
            skip_existing=skip_existing,
        )


@dataclass
class EcosystemEFGTask(EcosystemTask):
    """EFG-only ecosystem task: reads realm + biome outputs, builds EFG candidates, calls model.

    Reconstructs the full realm→biome→EFG conversation from saved outputs — batch mode supported.

    Prior outputs are looked up automatically from
    ``{BATCH_OUTPUTS_DIR}/{run_name}/ecosystems_realm-*.jsonl`` and
    ``{BATCH_OUTPUTS_DIR}/{run_name}/ecosystems_biome-*.jsonl``.

    Output format per record:
      {custom_id, response_text, results_payload, candidates_passed, error}
    """

    def build_requests_file(
        self,
        documents: Sequence[dict],
        dest: Path,
        prompts: dict,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> Path:
        prompt_config = _compose_prompt_config(prompts, PROMPT_KEY_ECOSYSTEMS_CORE, self.prompt_key)
        system_prompt = prompt_config["system_prompt"]
        mapping = self._load_mapping()

        dest = _normalize_request_dest(Path(dest))
        run_name = dest.parent.parent.name
        realm_outputs = _load_ecosystem_level_outputs(run_name, "ecosystems_realm")
        biome_outputs = _load_ecosystem_level_outputs(run_name, "ecosystems_biome")
        if not realm_outputs:
            print(f"[ecosystems_efg] WARNING: No ecosystems_realm outputs found for run '{run_name}'.")
        if not biome_outputs:
            print(f"[ecosystems_efg] WARNING: No ecosystems_biome outputs found for run '{run_name}'.")

        records: list[dict] = []
        skipped = 0
        skipped_missing_realm = 0
        skipped_terminal_realm = 0
        skipped_missing_biome = 0
        skipped_terminal_biome = 0
        skipped_no_biome_candidates = 0
        skipped_no_efg_candidates = 0

        for doc in documents:
            cid = _custom_id(doc)
            realm_rec = realm_outputs.get(cid)
            biome_rec = biome_outputs.get(cid)

            if not realm_rec or realm_rec.get("error") or not realm_rec.get("results_payload"):
                skipped += 1
                skipped_missing_realm += 1
                continue
            realm_results = realm_rec["results_payload"]
            realm_labels = self._extract_labels(realm_results)
            if not realm_labels or not self._should_continue(realm_results):
                stop_reason = realm_results.get("stop_reason") if isinstance(realm_results, dict) else None
                print(
                    f"[ecosystems_efg] Skipping {cid}: terminal ecosystems_realm output "
                    f"(labels={realm_labels}, stop_reason={stop_reason})"
                )
                skipped += 1
                skipped_terminal_realm += 1
                continue

            if not biome_rec or biome_rec.get("error") or not biome_rec.get("results_payload"):
                skipped += 1
                skipped_missing_biome += 1
                continue
            biome_results = biome_rec["results_payload"]
            biome_labels = self._extract_labels(biome_results)
            if not biome_labels or not self._should_continue(biome_results):
                stop_reason = biome_results.get("stop_reason") if isinstance(biome_results, dict) else None
                print(
                    f"[ecosystems_efg] Skipping {cid}: terminal ecosystems_biome output "
                    f"(labels={biome_labels}, stop_reason={stop_reason})"
                )
                skipped += 1
                skipped_terminal_biome += 1
                continue

            # Re-derive biome candidates (for history reconstruction) and biome→EFG lookup
            biome_candidates, biome_lookup = self._build_biome_candidates(realm_labels, mapping)
            if not biome_candidates:
                skipped += 1
                skipped_no_biome_candidates += 1
                continue

            efg_candidates = self._build_efg_candidates(biome_labels, biome_lookup)
            if not efg_candidates:
                skipped += 1
                skipped_no_efg_candidates += 1
                continue

            article_text = format_article(doc)
            realm_response_text = realm_rec.get("response_text") or ""
            biome_response_text = biome_rec.get("response_text") or ""

            body: dict[str, Any] = {
                "model": model,
                "instructions": system_prompt,
                "input": [
                    {"role": "user", "content": f"ARTICLE_TEXT:\\n{article_text}"},
                    {"role": "assistant", "content": realm_response_text},
                    {"role": "user", "content": f"BIOME_CANDIDATES:\\n{json.dumps(biome_candidates, indent=2)}"},
                    {"role": "assistant", "content": biome_response_text},
                    {"role": "user", "content": f"EFG_CANDIDATES:\\n{json.dumps(efg_candidates, indent=2)}"},
                ],
            }
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}
            text_param = _build_ecosystem_text_param(
                prompt_config.get("structured_output"),
                allowed_labels=[c.get("name") for c in efg_candidates if c.get("name")] + ["Unclear", "Not Applicable"],
                stop_reason_values=["stop"],
            )
            if text_param:
                body["text"] = text_param

            records.append({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
                "candidates_passed": efg_candidates,
            })

        # Sort by EFG candidate names for cache locality
        records = _sort_requests_by_candidates(records)

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if skipped:
            print(f"[ecosystems_efg] Skipped {skipped} documents (missing realm/biome outputs or terminal labels)")
            print(
                "[ecosystems_efg] Skip summary: "
                f"missing/bad realm={skipped_missing_realm}, "
                f"terminal realm={skipped_terminal_realm}, "
                f"missing/bad biome={skipped_missing_biome}, "
                f"terminal biome={skipped_terminal_biome}, "
                f"no biome candidates={skipped_no_biome_candidates}, "
                f"no EFG candidates={skipped_no_efg_candidates}"
            )

        return dest

    def run_live(
        self,
        request_paths: Iterable[Path | str],
        output_dir: Path,
        client: OpenAI | None = None,
        default_model: str = DEFAULT_MODEL,
        default_reasoning: str = DEFAULT_REASONING_EFFORT,
        prompts: dict | None = None,
        skip_existing: bool = True,
    ) -> list[dict]:
        return _run_split_ecosystem_live(
            task_name=self.name,
            output_prefix=self.output_prefix or self.name,
            request_paths=request_paths,
            output_dir=output_dir,
            client=client or build_client(),
            default_model=default_model,
            default_reasoning=default_reasoning,
            skip_existing=skip_existing,
        )


DRIVER_TASK = SimpleTask(name="driver", prompt_key=PROMPT_KEY_DRIVER, output_prefix="drivers")
SCREENING_TASK = SimpleTask(name="screening", prompt_key=PROMPT_KEY_SCREENING, output_prefix="screening")
GEOGRAPHY_TASK = SimpleTask(name="geography", prompt_key=PROMPT_KEY_GEOGRAPHY, output_prefix="geography")
TAXA_TASK = SimpleTask(name="taxa", prompt_key=PROMPT_KEY_TAXA, output_prefix="taxa")
STUDY_TASK = SimpleTask(name="study", prompt_key=PROMPT_KEY_STUDY, output_prefix="study")
ECOSYSTEM_REALM_TASK = EcosystemRealmTask(name="ecosystems_realm", prompt_key=PROMPT_KEY_ECOSYSTEMS_REALM, supports_batch=True, output_prefix="ecosystems_realm")
ECOSYSTEM_BIOME_TASK = EcosystemBiomeTask(name="ecosystems_biome", prompt_key=PROMPT_KEY_ECOSYSTEMS_BIOME, supports_batch=True, output_prefix="ecosystems_biome")
ECOSYSTEM_EFG_TASK   = EcosystemEFGTask(name="ecosystems_efg",  prompt_key=PROMPT_KEY_ECOSYSTEMS_EFG, supports_batch=True, output_prefix="ecosystems_efg")
THREAT_L0_TASK = ThreatL0Task(name="threats_l0", prompt_key=PROMPT_KEY_THREATS_L0, supports_batch=True, output_prefix="threats_l0")
THREAT_L1_TASK = ThreatL1Task(name="threats_l1", prompt_key=PROMPT_KEY_THREATS_L1, supports_batch=True, output_prefix="threats_l1")
THREAT_L2_TASK = ThreatL2Task(name="threats_l2", prompt_key=PROMPT_KEY_THREATS_L2, supports_batch=True, output_prefix="threats_l2")

TASK_ALIASES = {
    "driver": DRIVER_TASK,
    "drivers": DRIVER_TASK,
    "direct_driver": DRIVER_TASK,
    "direct_drivers": DRIVER_TASK,
    "screen": SCREENING_TASK,
    "screening": SCREENING_TASK,
    "eligibility": SCREENING_TASK,
    "geography": GEOGRAPHY_TASK,
    "geo": GEOGRAPHY_TASK,
    "region": GEOGRAPHY_TASK,
    "regions": GEOGRAPHY_TASK,
    "taxa": TAXA_TASK,
    "study": STUDY_TASK,
    "studies": STUDY_TASK,
    "ecosystems_realm": ECOSYSTEM_REALM_TASK,
    "ecosystem_realm": ECOSYSTEM_REALM_TASK,
    "ecosystems_biome": ECOSYSTEM_BIOME_TASK,
    "ecosystem_biome": ECOSYSTEM_BIOME_TASK,
    "ecosystems_efg": ECOSYSTEM_EFG_TASK,
    "ecosystem_efg": ECOSYSTEM_EFG_TASK,
    "threats_l0": THREAT_L0_TASK,
    "threats_l1": THREAT_L1_TASK,
    "threats_l2": THREAT_L2_TASK,
}


def get_task(name: str) -> TaskDefinition:
    key = (name or "").lower().strip()
    if key not in TASK_ALIASES:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {', '.join(sorted(set(TASK_ALIASES)))}")
    return TASK_ALIASES[key]


def available_task_names() -> list[str]:
    return sorted({task.name for task in TASK_ALIASES.values()})

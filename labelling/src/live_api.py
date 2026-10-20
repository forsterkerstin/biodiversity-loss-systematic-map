from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Tuple
from uuid import uuid4

from .config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
from .batch_api import build_client


def normalize_request_payload(
    raw_request: dict, default_model: str = DEFAULT_MODEL, default_reasoning: str = DEFAULT_REASONING_EFFORT
) -> Tuple[str | None, dict]:
    """Prepare a single batch-style request line for live /responses calls."""
    body = dict(raw_request.get("body") or {})
    custom_id = raw_request.get("custom_id") or raw_request.get("id")
    if not body.get("input") and body.get("messages"):
        body["input"] = body.pop("messages")
    reasoning = body.get("reasoning")
    if isinstance(reasoning, str):
        body["reasoning"] = {"effort": reasoning}
    elif reasoning is None and default_reasoning:
        body["reasoning"] = {"effort": default_reasoning}
    elif isinstance(reasoning, dict) and "effort" not in reasoning and default_reasoning:
        body["reasoning"] = {"effort": reasoning.get("value") or reasoning.get("level") or default_reasoning}
    if not body.get("model"):
        body["model"] = default_model
    return custom_id, body


def serialize_live_body(resp: Any) -> dict:
    """Best-effort conversion of OpenAI response objects to plain dicts for JSONL storage."""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(resp, attr, None)
        if fn:
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    for attr in ("model_dump_json", "to_json", "json"):
        fn = getattr(resp, attr, None)
        if fn:
            try:
                return json.loads(fn())
            except Exception:
                pass
    try:
        return json.loads(str(resp))
    except Exception:
        return {"raw": str(resp)}


def _load_processed_custom_ids(out_path: Path) -> set[str]:
    """Return custom_ids already present in an existing live output file."""
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


def run_live_requests(
    request_paths: Iterable[Path | str],
    output_dir: Path,
    client=None,
    default_model: str = DEFAULT_MODEL,
    default_reasoning: str = DEFAULT_REASONING_EFFORT,
    output_prefix: str = "live",
    skip_existing: bool = True,
) -> list[dict]:
    """
    Send request JSONL lines directly to OpenAI and store batch-shaped outputs line-by-line.

    Returns metadata entries mirroring batch submission shape: one per request file with
    mode, request_path, output_path, status, and count of responses.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = client or build_client()

    live_entries: list[dict] = []
    for req_path in request_paths:
        req_path = Path(req_path)
        out_path = output_dir / f"{output_prefix}-{req_path.stem}.jsonl"
        processed_ids = _load_processed_custom_ids(out_path) if skip_existing else set()
        skipped_count = 0
        rendered_count = 0
        mode = "a" if out_path.exists() else "w"
        with req_path.open("r", encoding="utf-8") as src, out_path.open(mode, encoding="utf-8") as dest:
            for line in src:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    record = {
                        "id": f"live_req_{uuid4().hex}",
                        "custom_id": None,
                        "response": {"status_code": None, "request_id": None, "body": None},
                        "error": {"message": f"Invalid request JSON: {exc}"},
                    }
                    dest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rendered_count += 1
                    continue

                custom_id, body = normalize_request_payload(
                    payload, default_model=default_model, default_reasoning=default_reasoning
                )
                if skip_existing and custom_id and custom_id in processed_ids:
                    print(f"Skipping already processed custom_id {custom_id} from {req_path}")
                    skipped_count += 1
                    continue
                record = {
                    "id": f"live_req_{uuid4().hex}",
                    "custom_id": custom_id,
                    "response": {"status_code": None, "request_id": None, "body": None},
                    "error": None,
                }
                try:
                    resp = client.responses.create(**body)
                    resp_body = serialize_live_body(resp)
                    record["response"] = {
                        "status_code": 200,
                        "request_id": getattr(resp, "id", None) or (resp_body.get("id") if isinstance(resp_body, dict) else None),
                        "body": resp_body,
                    }
                except Exception as exc:
                    record["error"] = {"message": str(exc)}
                    record["response"] = {"status_code": None, "request_id": None, "body": None}

                dest.write(json.dumps(record, ensure_ascii=False) + "\n")
                rendered_count += 1

        live_entries.append(
            {
                "mode": "live",
                "request_path": str(req_path),
                "output_path": str(out_path),
                "status": "completed",
                "responses": rendered_count,
                "skipped": skipped_count,
            }
        )
        print(f"Wrote {rendered_count} live outputs to {out_path} (skipped {skipped_count})")

    return live_entries

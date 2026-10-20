from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from .config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, MAPPINGS_DIR, PROMPTS_DIR, PROMPT_KEY_DRIVER
from .config import get_openai_api_key


def _resolve_prompts_path(path: Path | None) -> Path:
    candidate = Path(path) if path else PROMPTS_DIR
    if not candidate.is_absolute():
        candidate = MAPPINGS_DIR / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Prompt path not found: {candidate}")
    return candidate


def _iter_prompt_files(directory: Path) -> list[Path]:
    md_files = sorted(directory.glob("*.md"))
    if md_files:
        return md_files
    # compatibility fallback for prompt directories not yet migrated
    txt_files = sorted(directory.glob("*.txt"))
    return txt_files


def _load_prompts_from_dir(directory: Path) -> dict:
    prompts: dict[str, dict] = {}
    for prompt_path in _iter_prompt_files(directory):
        key = prompt_path.stem
        system_prompt = prompt_path.read_text(encoding="utf-8")
        structured_output = None
        json_path = prompt_path.with_suffix(".json")
        if json_path.exists():
            structured_output = json.loads(json_path.read_text(encoding="utf-8"))
        prompts[key] = {"system_prompt": system_prompt, "structured_output": structured_output}
    if not prompts:
        raise ValueError(f"No prompt files found in {directory}")
    return prompts


def load_prompts(path: Path | None = None) -> dict:
    prompt_path = _resolve_prompts_path(path)
    if prompt_path.is_dir():
        return _load_prompts_from_dir(prompt_path)
    suffix = prompt_path.suffix.lower()
    if suffix == ".json":
        return json.loads(prompt_path.read_text(encoding="utf-8"))
    if suffix in {".md", ".markdown", ".txt"}:
        system_prompt = prompt_path.read_text(encoding="utf-8")
        return {prompt_path.stem: {"system_prompt": system_prompt, "structured_output": None}}
    raise ValueError(f"Unsupported prompt path type: {prompt_path}")


def format_article(doc: dict) -> str:
    parts = [doc.get("title", "").strip(), doc.get("abstract", "").strip()]
    return "\n\n".join(p for p in parts if p)


def build_direct_driver_request(
    doc: dict,
    prompts: dict,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict:
    system_prompt = prompts[PROMPT_KEY_DRIVER]["system_prompt"]
    structured_output = prompts[PROMPT_KEY_DRIVER].get("structured_output")
    text_format = None
    if isinstance(structured_output, dict):
        text_format = structured_output.get("text") or structured_output.get("format")
    prompt = format_article(doc)
    body: dict = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if text_format:
        body["text"] = {"format": text_format} if "format" not in text_format else text_format
    return {
        "custom_id": doc.get("UT") or doc.get("id") or "",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_requests_file(
    documents: Iterable[dict],
    dest: Path,
    prompts: dict | None = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Path:
    prompts = prompts or load_prompts()
    # If the caller accidentally points to the data dir, automatically switch to sibling request dir.
    if dest.parent.name.lower() == "data":
        dest = dest.parent.parent / "request" / dest.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for doc in documents:
            req = build_direct_driver_request(doc, prompts, model=model, reasoning_effort=reasoning_effort)
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    return dest


def build_client(api_key: str | None = None) -> OpenAI:
    key = api_key or get_openai_api_key()
    return OpenAI(api_key=key)


def _sanitize_batch_request_bytes(requests_path: Path) -> io.BytesIO:
    """Return a file-like object containing only Batch API supported request fields."""
    allowed_keys = {"custom_id", "method", "url", "body"}
    buf = io.BytesIO()
    with requests_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                buf.write(line.encode("utf-8"))
                continue
            sanitized = {key: payload[key] for key in allowed_keys if key in payload}
            buf.write((json.dumps(sanitized, ensure_ascii=False) + "\n").encode("utf-8"))
    buf.seek(0)
    return buf


def create_empty_output(req_path: Path, output_dir: Path) -> dict:
    out_name = req_path.name.replace("-requests.jsonl", "-output.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / out_name).touch()
    return {"mode": "skipped", "request_path": str(req_path), "output_path": str(output_dir / out_name), "status": "empty"}


def submit_batch(requests_path: Path, client: OpenAI | None = None, completion_window: str = "24h", metadata: dict | None = None) -> dict:
    client = client or build_client()
    upload_stream = _sanitize_batch_request_bytes(requests_path)
    upload = client.files.create(file=(requests_path.name, upload_stream), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata=metadata or {},
    )
    return {"batch_id": batch.id, "input_file_id": upload.id, "status": batch.status}


def poll_batch(batch_id: str, client: OpenAI | None = None) -> dict:
    client = client or build_client()
    batch = client.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "completion_window": getattr(batch, "completion_window", None),
    }


def download_batch_output(batch_id: str, dest: Path, client: OpenAI | None = None) -> Path:
    client = client or build_client()
    batch = client.batches.retrieve(batch_id)
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output file yet (status: {batch.status}).")

    content = client.files.content(output_file_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content.read())
    return dest

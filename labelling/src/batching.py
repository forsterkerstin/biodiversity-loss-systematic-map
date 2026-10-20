from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .config import BATCHES_DIR


def chunk(items: Sequence, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("Batch size must be positive.")
    batches = []
    for i in range(0, len(items), size):
        batches.append(list(items[i : i + size]))
    return batches


def write_batches(
    documents: Sequence[dict],
    batch_size: int,
    run_name: str | None = None,
    output_dir: Path = BATCHES_DIR,
) -> dict:
    """Split documents and write each batch to JSONL."""
    run_name = run_name or datetime.utcnow().strftime("run-%Y%m%dT%H%M%S")
    run_dir = Path(output_dir) / run_name
    data_dir = run_dir / "data"
    request_dir = run_dir / "request"
    data_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)

    batches = chunk(list(documents), batch_size)
    batch_entries = []
    for idx, batch in enumerate(batches, start=1):
        data_path = data_dir / f"batch-{run_name}-{idx:03d}.jsonl"
        request_path = request_dir / f"batch-{run_name}-{idx:03d}-requests.jsonl"
        with data_path.open("w", encoding="utf-8") as f:
            for doc in batch:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        batch_entries.append(
            {
                "data_path": str(data_path),
                "request_path": str(request_path),
                "path": str(data_path),  # legacy alias
                "count": len(batch),
            }
        )

    manifest = {
        "run_name": run_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "batch_size": batch_size,
        "total_documents": len(documents),
        "data_dir": str(data_dir),
        "request_dir": str(request_dir),
        "batches": batch_entries,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

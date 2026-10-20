"""Portable provenance helpers for prepared-data manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def repository_relative_path(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> str:
    """Return an existing path relative to the declared repository root."""
    root = Path(repository_root).resolve()
    source = Path(path).resolve(strict=True)
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Provenance source is outside repository root: {source} (root={root})"
        ) from exc
    return relative.as_posix()


def source_signature(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Record portable path, size, and modification time for one source."""
    source = Path(path).resolve(strict=True)
    stat = source.stat()
    return {
        "path": repository_relative_path(source, repository_root=repository_root),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


__all__ = ["repository_relative_path", "source_signature"]

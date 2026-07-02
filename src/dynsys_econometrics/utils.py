"""Small utility helpers shared across the repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Return the repository root based on the installed source layout."""
    return Path(__file__).resolve().parents[2]


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON payload to disk with deterministic formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

from __future__ import annotations

import json
from pathlib import Path

from dynsys_econometrics.utils import ensure_directory, repo_root, write_json


def test_ensure_directory_creates_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "output"
    created = ensure_directory(target)
    assert created == target
    assert target.exists()
    assert target.is_dir()


def test_write_json_writes_sorted_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.json"
    write_json({"b": 2, "a": 1}, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {"a": 1, "b": 2}


def test_repo_root_points_to_repository_root() -> None:
    root = repo_root()
    assert (root / "README.md").exists()
    assert (root / "src" / "dynsys_econometrics").exists()

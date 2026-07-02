from __future__ import annotations

from pathlib import Path

import pytest

from dynsys_econometrics.config import load_config, resolve_output_dir


def test_load_config_reads_yaml(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  output_dir: outputs/demo\n", encoding="utf-8")
    config = load_config(config_path)
    assert config["project"]["output_dir"] == "outputs/demo"


def test_resolve_output_dir_prefers_outputs_section() -> None:
    output_dir = resolve_output_dir({"outputs": {"directory": "outputs/x"}, "project": {"output_dir": "ignored"}})
    assert output_dir == Path("outputs/x")


def test_resolve_output_dir_rejects_invalid_type() -> None:
    with pytest.raises(ValueError):
        resolve_output_dir({"outputs": {"directory": 123}})

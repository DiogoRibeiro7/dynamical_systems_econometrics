"""Configuration helpers for reproducible experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.

    Assumptions
    -----------
    Config files are YAML mappings rooted at the repository or experiment
    directory.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML payload is not a mapping.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping.")
    return payload


def resolve_output_dir(config: Mapping[str, Any]) -> Path:
    """Resolve the configured output directory deterministically.

    Parameters
    ----------
    config:
        Configuration mapping containing either ``outputs.directory`` or
        ``project.output_dir``.

    Returns
    -------
    Path
        Output directory path relative to the current working directory unless
        already absolute.

    Raises
    ------
    ValueError
        If the configured output directory is blank or not path-like.
    """
    output_value = None
    outputs = config.get("outputs")
    project = config.get("project")
    if isinstance(outputs, Mapping) and "directory" in outputs:
        output_value = outputs["directory"]
    elif isinstance(project, Mapping) and "output_dir" in project:
        output_value = project["output_dir"]
    if output_value is None:
        return Path("outputs")
    if not isinstance(output_value, (str, Path)):
        raise ValueError("Configured output directory must be a string or Path-like value.")
    output_dir = Path(output_value)
    if str(output_dir).strip() == "":
        raise ValueError("Configured output directory must not be blank.")
    return output_dir

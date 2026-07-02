"""Runtime-validated data contracts used across the repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


_REQUIRED_TS_COLUMNS = ("date", "series_id", "value")
_OPTIONAL_TS_COLUMNS = (
    "country",
    "frequency",
    "unit",
    "source",
    "transformation",
    "release_version",
)
_REQUIRED_EVENT_COLUMNS = ("date", "series_id", "value", "threshold", "exceedance", "side", "run_id")


@dataclass(frozen=True)
class TimeSeriesFrame:
    """Validated long-format time-series panel.

    Parameters
    ----------
    frame:
        DataFrame containing the canonical long-format schema.
    allow_missing:
        If ``True``, the ``value`` column may contain missing entries. Non-finite
        infinite values remain disallowed.

    Returns
    -------
    TimeSeriesFrame
        Validated immutable wrapper around the sorted DataFrame.

    Assumptions
    -----------
    Rows represent observed time-series values rather than latent deterministic
    states of an economy.

    Raises
    ------
    ValueError
        If required columns are missing, duplicate keys exist, or values are
        invalid.

    Example
    -------
    >>> import pandas as pd
    >>> frame = TimeSeriesFrame(pd.DataFrame(
    ...     {"date": ["2020-01-01"], "series_id": ["x"], "value": [1.0]}
    ... ))
    >>> frame.to_frame().columns.tolist()
    ['date', 'series_id', 'value']
    """

    frame: pd.DataFrame
    allow_missing: bool = False
    _validated: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        validated = self._validate(self.frame, allow_missing=self.allow_missing)
        object.__setattr__(self, "_validated", validated)

    @staticmethod
    def _validate(frame: pd.DataFrame, *, allow_missing: bool) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        missing = set(_REQUIRED_TS_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        validated = frame.copy()
        validated["date"] = pd.to_datetime(validated["date"], errors="raise")
        validated["series_id"] = validated["series_id"].astype(str)
        if (validated["series_id"].str.strip() == "").any():
            raise ValueError("series_id must not be empty.")
        validated["value"] = pd.to_numeric(validated["value"], errors="coerce")
        if allow_missing:
            non_missing = validated["value"].dropna().to_numpy(dtype=float)
            if not np.isfinite(non_missing).all():
                raise ValueError("value contains non-finite entries.")
        else:
            if validated["value"].isna().any():
                raise ValueError("value must be numeric and finite.")
            if not np.isfinite(validated["value"].to_numpy(dtype=float)).all():
                raise ValueError("value must be finite.")
        duplicates = validated.duplicated(["series_id", "date"])
        if duplicates.any():
            raise ValueError("Duplicate rows found for date + series_id.")
        validated = validated.sort_values(["series_id", "date"]).reset_index(drop=True)
        return validated

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the validated underlying DataFrame."""
        return self._validated.copy()


@dataclass(frozen=True)
class EventFrame:
    """Validated threshold-event table.

    Parameters
    ----------
    frame:
        Event table containing exceedance metadata.

    Returns
    -------
    EventFrame
        Immutable validated event-frame wrapper.

    Assumptions
    -----------
    Event tables are descriptive outputs and do not by themselves establish
    asymptotic validity or causality.

    Raises
    ------
    ValueError
        If required columns are missing or event semantics are invalid.
    """

    frame: pd.DataFrame
    _validated: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        missing = set(_REQUIRED_EVENT_COLUMNS).difference(self.frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        validated = self.frame.copy()
        validated["date"] = pd.to_datetime(validated["date"], errors="raise")
        validated["series_id"] = validated["series_id"].astype(str)
        validated["value"] = pd.to_numeric(validated["value"], errors="raise")
        validated["threshold"] = pd.to_numeric(validated["threshold"], errors="raise")
        if validated["exceedance"].dtype != bool:
            validated["exceedance"] = validated["exceedance"].astype(bool)
        valid_sides = {"upper", "lower", "two_sided"}
        if not set(validated["side"].astype(str)).issubset(valid_sides):
            raise ValueError("side must be one of 'upper', 'lower', or 'two_sided'.")
        object.__setattr__(
            self,
            "_validated",
            validated.sort_values(["series_id", "date"]).reset_index(drop=True),
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the validated event table."""
        return self._validated.copy()


@dataclass(frozen=True)
class ExperimentResult:
    """Container for reproducible experiment outputs.

    Parameters
    ----------
    name:
        Non-empty experiment name.
    config:
        Resolved experiment configuration.
    tables:
        Mapping of output table names to DataFrames.
    figures:
        Mapping of figure names to relative or absolute paths.
    metrics:
        Mapping of scalar metric names to floats.
    warnings:
        Human-readable warning strings.

    Returns
    -------
    ExperimentResult
        Validated immutable experiment-output registry.

    Raises
    ------
    ValueError
        If names or keys are empty.
    """

    name: str
    config: dict[str, object]
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Path]
    metrics: dict[str, float]
    warnings: list[str]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty.")
        for key, value in self.tables.items():
            if not key.strip():
                raise ValueError("table keys must be non-empty.")
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"Table '{key}' must be a pandas DataFrame.")
        for key, value in self.figures.items():
            if not key.strip():
                raise ValueError("figure keys must be non-empty.")
            if str(value).strip() == "":
                raise ValueError("figure paths must be non-empty.")
        for key in self.metrics:
            if not key.strip():
                raise ValueError("metric keys must be non-empty.")
        for warning in self.warnings:
            if not warning.strip():
                raise ValueError("warning messages must be non-empty.")

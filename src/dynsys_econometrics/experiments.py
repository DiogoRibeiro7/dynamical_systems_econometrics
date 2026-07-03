"""Experiment orchestration for synthetic and empirical workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from dynsys_econometrics.config import resolve_output_dir
from dynsys_econometrics.contracts import ExperimentResult
from dynsys_econometrics.data import load_empirical_panel
from dynsys_econometrics.extremes import runs_extremal_index, threshold_exceedances
from dynsys_econometrics.plots import plot_extremal_index_bars
from dynsys_econometrics.return_times import return_times
from dynsys_econometrics.simulation import (
    simulate_ar1,
    simulate_garch11,
    simulate_intermittent_map,
    simulate_logistic_map,
)
from dynsys_econometrics.utils import ensure_directory, write_json


def _array_to_frame(values: np.ndarray, *, series_id: str, system: str) -> pd.DataFrame:
    """Convert a one-dimensional simulated array to the canonical long format."""
    dates = pd.date_range("2000-01-31", periods=values.size, freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "series_id": series_id,
            "value": values.astype(float),
            "system": system,
        }
    )


def run_synthetic_experiment(config: Mapping[str, Any]) -> ExperimentResult:
    """Run a reproducible synthetic experiment from configuration.

    Parameters
    ----------
    config:
        Configuration mapping describing synthetic systems and analysis options.

    Returns
    -------
    ExperimentResult
        Registry of tables, figures, metrics, and warnings.

    Raises
    ------
    ValueError
        If the requested synthetic mode or system configuration is invalid.
    """
    experiment_config = dict(config)
    experiment = experiment_config.get("experiment", {})
    synthetic = experiment_config.get("synthetic", {})
    analysis = experiment_config.get("analysis", {})
    name = str(experiment.get("name", "synthetic_experiment"))
    systems = synthetic.get("systems", [{"name": "logistic", "n": 5000, "r": 4.0}, {"name": "ar1", "n": 5000, "phi": 0.7}])
    if not isinstance(systems, list) or len(systems) == 0:
        raise ValueError("synthetic.systems must be a non-empty list.")
    threshold_quantile = float(analysis.get("threshold_quantile", 0.95))
    run_length = int(analysis.get("run_length", 3))

    frames: list[pd.DataFrame] = []
    for entry in systems:
        if not isinstance(entry, Mapping):
            raise ValueError("Each synthetic system entry must be a mapping.")
        system_name = str(entry.get("name", "")).strip().lower()
        if system_name == "logistic":
            frame = simulate_logistic_map(
                n=int(entry.get("n", 5000)),
                r=float(entry.get("r", 4.0)),
                x0=float(entry.get("x0", 0.123)),
                burn_in=int(entry.get("burn_in", 100)),
                noise_std=float(entry.get("noise_std", 0.0)),
                seed=entry.get("seed"),
            )
        elif system_name == "ar1":
            values = simulate_ar1(
                n_steps=int(entry.get("n", 5000)),
                phi=float(entry.get("phi", 0.7)),
                sigma=float(entry.get("sigma", 1.0)),
                seed=entry.get("seed"),
            )
            frame = _array_to_frame(values, series_id="ar1", system="ar1")
        elif system_name == "garch":
            values = np.abs(
                simulate_garch11(
                    n_steps=int(entry.get("n", 5000)),
                    omega=float(entry.get("omega", 0.01)),
                    alpha=float(entry.get("alpha", 0.05)),
                    beta=float(entry.get("beta", 0.9)),
                    seed=entry.get("seed"),
                )
            )
            frame = _array_to_frame(values, series_id="garch11", system="garch11")
        elif system_name == "intermittent":
            frame = simulate_intermittent_map(
                n=int(entry.get("n", 5000)),
                alpha=float(entry.get("alpha", 1.0)),
                x0=float(entry.get("x0", 0.123)),
                burn_in=int(entry.get("burn_in", 100)),
            )
        else:
            raise ValueError(f"Unsupported synthetic system: {system_name}")
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    events = threshold_exceedances(panel[["date", "series_id", "value"]], quantile=threshold_quantile)
    extremal_index = runs_extremal_index(events, run_length=run_length)
    durations = return_times(events)
    return ExperimentResult(
        name=name,
        config=dict(experiment_config),
        tables={
            "synthetic_panel": panel,
            "events": events,
            "extremal_index": extremal_index,
            "return_times": durations,
        },
        figures={"extremal_index_by_series": Path("figures/extremal_index_by_series.png")},
        metrics={
            "n_series": float(panel["series_id"].nunique()),
            "n_observations": float(len(panel)),
        },
        warnings=[],
    )


def run_empirical_experiment(config: Mapping[str, Any]) -> ExperimentResult:
    """Run an empirical experiment from a structured loader configuration."""
    experiment_config = dict(config)
    experiment = experiment_config.get("experiment", {})
    analysis = experiment_config.get("analysis", {})
    empirical = experiment_config.get("empirical", {})
    if not isinstance(empirical, Mapping):
        raise ValueError("empirical configuration must be a mapping.")
    name = str(experiment.get("name", "empirical_experiment"))
    panel = load_empirical_panel(empirical)
    threshold_quantile = float(analysis.get("threshold_quantile", 0.95))
    run_length = int(analysis.get("run_length", 3))
    min_observations = int(analysis.get("min_observations", 0))
    events = threshold_exceedances(panel, quantile=threshold_quantile)
    extremal_index = runs_extremal_index(events, run_length=run_length)
    durations = return_times(events)
    warnings: list[str] = []
    if min_observations > 0:
        counts = panel.groupby("series_id").size()
        underpowered = counts[counts < min_observations]
        if not underpowered.empty:
            summary = ", ".join(f"{series_id}={int(count)}" for series_id, count in underpowered.items())
            warnings.append(f"series below min_observations={min_observations}: {summary}")
    return ExperimentResult(
        name=name,
        config=dict(experiment_config),
        tables={
            "empirical_panel": panel,
            "events": events,
            "extremal_index": extremal_index,
            "return_times": durations,
        },
        figures={"extremal_index_by_series": Path("figures/extremal_index_by_series.png")},
        metrics={
            "n_series": float(panel["series_id"].nunique()),
            "n_observations": float(len(panel)),
        },
        warnings=warnings,
    )


def save_experiment_result(result: ExperimentResult, output_dir: str | Path) -> None:
    """Persist experiment tables, figures, and run metadata."""
    output_root = ensure_directory(output_dir)
    tables_dir = ensure_directory(output_root / "tables")
    figures_dir = ensure_directory(output_root / "figures")
    for name, table in result.tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    if "extremal_index_by_series" in result.figures and "extremal_index" in result.tables:
        fig, _ = plot_extremal_index_bars(result.tables["extremal_index"])
        fig.tight_layout()
        fig.savefig(figures_dir / "extremal_index_by_series.png", dpi=160)
    summary = {
        "name": result.name,
        "config": result.config,
        "metrics": result.metrics,
        "warnings": result.warnings,
        "tables": {key: str(Path("tables") / f"{key}.csv") for key in result.tables},
        "figures": {key: str(path) for key, path in result.figures.items()},
        "output_dir": str(Path(output_root)),
    }
    write_json(summary, output_root / "run_summary.json")


def default_output_dir(config: Mapping[str, Any]) -> Path:
    """Resolve the configured experiment output directory."""
    return resolve_output_dir(config)

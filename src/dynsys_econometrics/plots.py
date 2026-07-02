"""Plotting helpers for article figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _ensure_parent_dir(output_path: str | Path) -> Path:
    """Create the parent directory for an output file if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _validate_numeric_series(series: pd.Series, name: str) -> pd.Series:
    """Validate that a pandas Series is numeric and non-empty."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if series.empty:
        raise ValueError(f"{name} must not be empty.")
    if not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{name} must be numeric.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite values.")
    return series


def _validate_1d_array(values: FloatArray | IntArray, name: str) -> NDArray[np.float64]:
    """Validate that an array is one-dimensional and finite."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def plot_series_with_threshold(series: pd.Series, threshold: float, output_path: str | Path) -> None:
    """Save a time-series plot with a rare-event threshold line."""
    validated = _validate_numeric_series(series, "series")
    path = _ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    validated.plot(ax=ax)
    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_title("Rare-event threshold")
    ax.set_xlabel("date")
    ax.set_ylabel(validated.name or "value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_orbit_and_observable(
    orbit: FloatArray,
    observable: FloatArray,
    output_path: str | Path,
) -> None:
    """Plot a dynamical orbit together with its transformed observable."""
    orbit_arr = _validate_1d_array(orbit, "orbit")
    observable_arr = _validate_1d_array(observable, "observable")
    if orbit_arr.shape != observable_arr.shape:
        raise ValueError("orbit and observable must have the same shape.")

    path = _ensure_parent_dir(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(orbit_arr, color="#4c78a8", linewidth=1.0)
    axes[0].set_title("Simulated orbit")
    axes[0].set_ylabel("state")
    axes[1].plot(observable_arr, color="#f58518", linewidth=1.0)
    axes[1].set_title("Observable derived from orbit")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("observable")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_threshold_exceedances(
    series: pd.Series,
    threshold: float,
    output_path: str | Path,
    title: str = "Threshold exceedances",
) -> None:
    """Plot a series and highlight values that exceed a threshold."""
    validated = _validate_numeric_series(series, "series")
    path = _ensure_parent_dir(output_path)
    exceedances = validated > threshold

    fig, ax = plt.subplots(figsize=(10, 4))
    validated.plot(ax=ax, color="#4c78a8", linewidth=1.0)
    ax.scatter(validated.index[exceedances], validated[exceedances], color="#e45756", s=16)
    ax.axhline(threshold, linestyle="--", linewidth=1.0, color="#72b7b2")
    ax.set_title(title)
    ax.set_xlabel("date" if isinstance(validated.index, pd.DatetimeIndex) else "time")
    ax.set_ylabel(validated.name or "value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_clustered_vs_isolated_extremes(
    clustered_series: pd.Series,
    isolated_series: pd.Series,
    clustered_threshold: float,
    isolated_threshold: float,
    output_path: str | Path,
) -> None:
    """Compare clustered and isolated extreme-event patterns in two panels."""
    clustered = _validate_numeric_series(clustered_series, "clustered_series")
    isolated = _validate_numeric_series(isolated_series, "isolated_series")
    path = _ensure_parent_dir(output_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax, series, threshold, title, color in [
        (axes[0], clustered, clustered_threshold, "Clustered extremes", "#4c78a8"),
        (axes[1], isolated, isolated_threshold, "More isolated extremes", "#54a24b"),
    ]:
        exceedances = series > threshold
        series.plot(ax=ax, color=color, linewidth=1.0)
        ax.scatter(series.index[exceedances], series[exceedances], color="#e45756", s=16)
        ax.axhline(threshold, linestyle="--", linewidth=1.0, color="#72b7b2")
        ax.set_title(title)
        ax.set_ylabel(series.name or "value")

    axes[1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_return_time_distribution(
    return_times: IntArray,
    output_path: str | Path,
) -> None:
    """Plot the empirical distribution of return times."""
    values = np.asarray(return_times, dtype=np.int64)
    path = _ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    if values.size == 0:
        ax.text(0.5, 0.5, "No return times available", ha="center", va="center")
    else:
        unique, counts = np.unique(values, return_counts=True)
        ax.bar(unique.astype(str), counts, color="#f58518")
    ax.set_title("Return-time distribution")
    ax.set_xlabel("return time")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_extremal_index_by_threshold(
    quantiles: list[float],
    theta_values: list[float],
    output_path: str | Path,
) -> None:
    """Plot extremal-index estimates against threshold quantiles."""
    if len(quantiles) == 0 or len(quantiles) != len(theta_values):
        raise ValueError("quantiles and theta_values must be non-empty and have the same length.")
    if any((q <= 0.0) or (q >= 1.0) for q in quantiles):
        raise ValueError("quantiles must lie in the open interval (0, 1).")

    path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(quantiles, theta_values, marker="o", color="#72b7b2")
    ax.set_title("Extremal index by threshold")
    ax.set_xlabel("threshold quantile")
    ax.set_ylabel("theta")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_macro_financial_timeline(
    series: pd.Series,
    threshold: float,
    crisis_windows: list[tuple[pd.Timestamp, pd.Timestamp, str]],
    output_path: str | Path,
) -> None:
    """Plot a macro-financial series with highlighted crisis windows."""
    validated = _validate_numeric_series(series, "series")
    if not isinstance(validated.index, pd.DatetimeIndex):
        raise TypeError("series index must be a DatetimeIndex.")

    path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(11, 4))
    validated.plot(ax=ax, color="#4c78a8", linewidth=1.0)
    ax.axhline(threshold, linestyle="--", linewidth=1.0, color="#e45756")
    for start, end, label in crisis_windows:
        ax.axvspan(start, end, color="#f2cf5b", alpha=0.25)
        ax.text(start, float(validated.max()), label, fontsize=8, va="top")
    ax.set_title("Macro-financial crisis timeline")
    ax.set_xlabel("date")
    ax.set_ylabel(validated.name or "value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_multivariate_stress_heatmap(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot a multivariate stress heatmap from a numeric DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("frame must not be empty.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise TypeError("all frame columns must be numeric.")

    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("frame must contain only finite values.")

    path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(values.T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_title("Multivariate stress heatmap")
    ax.set_xlabel("time window")
    ax.set_ylabel("series")
    ax.set_yticks(range(frame.shape[1]))
    ax.set_yticklabels(frame.columns.tolist())
    fig.colorbar(image, ax=ax, label="stress level")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

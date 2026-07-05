"""Plotting helpers for article figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _series_label(series: pd.Series) -> str:
    """Return a printable label for a Series."""
    return str(series.name) if series.name is not None else "value"


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
    ax.set_ylabel(_series_label(validated))
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
    ax.set_ylabel(_series_label(validated))
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
        ax.set_ylabel(_series_label(series))

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
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
    independence_baseline: list[float] | None = None,
) -> None:
    """Plot extremal-index estimates against threshold quantiles."""
    if len(quantiles) == 0 or len(quantiles) != len(theta_values):
        raise ValueError("quantiles and theta_values must be non-empty and have the same length.")
    if any((q <= 0.0) or (q >= 1.0) for q in quantiles):
        raise ValueError("quantiles must lie in the open interval (0, 1).")
    if (lower_bounds is None) ^ (upper_bounds is None):
        raise ValueError("lower_bounds and upper_bounds must both be provided or both be omitted.")
    if lower_bounds is not None and (len(lower_bounds) != len(quantiles) or len(upper_bounds) != len(quantiles)):
        raise ValueError("lower_bounds and upper_bounds must match the length of quantiles.")
    if independence_baseline is not None and len(independence_baseline) != len(quantiles):
        raise ValueError("independence_baseline must match the length of quantiles.")

    path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    if lower_bounds is not None and upper_bounds is not None:
        ax.fill_between(
            quantiles,
            lower_bounds,
            upper_bounds,
            color="#9ecae9",
            alpha=0.35,
            label="90% block-bootstrap interval",
        )
    ax.plot(quantiles, theta_values, marker="o", color="#72b7b2")
    if independence_baseline is not None:
        ax.plot(
            quantiles,
            independence_baseline,
            linestyle="--",
            linewidth=1.2,
            color="#1f1f1f",
            label="i.i.d. finite-level baseline",
        )
    ax.set_title("Extremal index by threshold")
    ax.set_xlabel("threshold quantile")
    ax.set_ylabel("theta")
    if lower_bounds is not None and upper_bounds is not None or independence_baseline is not None:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cluster_adjusted_stress_by_threshold(
    quantiles: list[float],
    lambda_values: list[float],
    output_path: str | Path,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
    independence_baseline: list[float] | None = None,
) -> None:
    """Plot cluster-adjusted stress estimates against threshold quantiles."""
    if len(quantiles) == 0 or len(quantiles) != len(lambda_values):
        raise ValueError("quantiles and lambda_values must be non-empty and have the same length.")
    if any((q <= 0.0) or (q >= 1.0) for q in quantiles):
        raise ValueError("quantiles must lie in the open interval (0, 1).")
    if (lower_bounds is None) ^ (upper_bounds is None):
        raise ValueError("lower_bounds and upper_bounds must both be provided or both be omitted.")
    if lower_bounds is not None and (len(lower_bounds) != len(quantiles) or len(upper_bounds) != len(quantiles)):
        raise ValueError("lower_bounds and upper_bounds must match the length of quantiles.")
    if independence_baseline is not None and len(independence_baseline) != len(quantiles):
        raise ValueError("independence_baseline must match the length of quantiles.")

    path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    if lower_bounds is not None and upper_bounds is not None:
        ax.fill_between(
            quantiles,
            lower_bounds,
            upper_bounds,
            color="#fdd0a2",
            alpha=0.35,
            label="90% block-bootstrap interval",
        )
    ax.plot(quantiles, lambda_values, marker="o", color="#c45b12")
    if independence_baseline is not None:
        ax.plot(
            quantiles,
            independence_baseline,
            linestyle="--",
            linewidth=1.2,
            color="#1f1f1f",
            label="i.i.d. finite-level baseline",
        )
    ax.set_title("Cluster-adjusted stress by threshold")
    ax.set_xlabel("threshold quantile")
    ax.set_ylabel("lambda")
    if lower_bounds is not None and upper_bounds is not None or independence_baseline is not None:
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cluster_adjusted_stress_by_run_length(
    run_lengths: list[int],
    lambda_values: list[float],
    output_path: str | Path,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
    independence_baseline: list[float] | None = None,
) -> None:
    """Plot cluster-adjusted stress estimates against run length."""
    if len(run_lengths) == 0 or len(run_lengths) != len(lambda_values):
        raise ValueError("run_lengths and lambda_values must be non-empty and have the same length.")
    if any(r < 1 for r in run_lengths):
        raise ValueError("run_lengths must be positive integers.")
    if (lower_bounds is None) ^ (upper_bounds is None):
        raise ValueError("lower_bounds and upper_bounds must both be provided or both be omitted.")
    if lower_bounds is not None and (len(lower_bounds) != len(run_lengths) or len(upper_bounds) != len(run_lengths)):
        raise ValueError("lower_bounds and upper_bounds must match the length of run_lengths.")
    if independence_baseline is not None and len(independence_baseline) != len(run_lengths):
        raise ValueError("independence_baseline must match the length of run_lengths.")

    path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    if lower_bounds is not None and upper_bounds is not None:
        ax.fill_between(
            run_lengths,
            lower_bounds,
            upper_bounds,
            color="#fdd0a2",
            alpha=0.35,
            label="90% block-bootstrap interval",
        )
    ax.plot(run_lengths, lambda_values, marker="o", color="#8c2d04")
    if independence_baseline is not None:
        ax.plot(
            run_lengths,
            independence_baseline,
            linestyle="--",
            linewidth=1.2,
            color="#1f1f1f",
            label="i.i.d. finite-level baseline",
        )
    ax.set_title("Cluster-adjusted stress by run length")
    ax.set_xlabel("run length")
    ax.set_ylabel("lambda")
    if lower_bounds is not None and upper_bounds is not None or independence_baseline is not None:
        ax.legend(frameon=False, loc="upper left")
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
        start_num = float(mdates.date2num(start.to_pydatetime()))  # type: ignore[no-untyped-call]
        end_num = float(mdates.date2num(end.to_pydatetime()))  # type: ignore[no-untyped-call]
        ax.axvspan(start_num, end_num, color="#f2cf5b", alpha=0.25)
        ax.text(start_num, float(validated.max()), label, fontsize=8, va="top")
    ax.set_title("Macro-financial crisis timeline")
    ax.set_xlabel("date")
    ax.set_ylabel(_series_label(validated))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.90)
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
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.90)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_time_series_with_exceedances(
    series: pd.Series,
    threshold: float,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a time series with exceedances highlighted and return figure objects."""
    validated = _validate_numeric_series(series, "series")
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    exceedances = validated >= threshold
    validated.plot(ax=ax, color="#4c78a8", linewidth=1.0)
    ax.scatter(validated.index[exceedances], validated[exceedances], color="#e45756", s=16)
    ax.axhline(threshold, linestyle="--", linewidth=1.0, color="#72b7b2")
    ax.set_title("Time series with exceedances")
    ax.set_xlabel("date" if isinstance(validated.index, pd.DatetimeIndex) else "time")
    ax.set_ylabel(_series_label(validated))
    return fig, ax


def plot_extremal_index_bars(
    table: pd.DataFrame,
    *,
    series_col: str = "series_id",
    value_col: str = "extremal_index",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot extremal-index estimates as a bar chart."""
    required = {series_col, value_col}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure
    ax.bar(table[series_col].astype(str), table[value_col].astype(float), color="#72b7b2")
    ax.set_title("Extremal index by series")
    ax.set_xlabel("series")
    ax.set_ylabel("extremal index")
    return fig, ax


def plot_joint_stress_timeline(
    table: pd.DataFrame,
    *,
    date_col: str = "date",
    score_col: str = "stress_score",
    active_col: str = "joint_exceedance",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a multivariate stress score through time."""
    required = {date_col, score_col, active_col}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    dates = pd.to_datetime(table[date_col], errors="raise")
    scores = pd.to_numeric(table[score_col], errors="raise")
    active = table[active_col].astype(bool)
    ax.plot(dates, scores, color="#4c78a8", linewidth=1.0)
    ax.scatter(dates[active], scores[active], color="#e45756", s=16)
    ax.set_title("Joint stress timeline")
    ax.set_xlabel("date")
    ax.set_ylabel("stress score")
    return fig, ax


def plot_baseline_comparison(
    table: pd.DataFrame,
    *,
    x_col: str = "date",
    event_col: str = "exceedance",
    baseline_col: str = "rolling_volatility",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a baseline diagnostic against rare-event indicators."""
    required = {x_col, event_col, baseline_col}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    x_values = table[x_col]
    baseline = pd.to_numeric(table[baseline_col], errors="raise")
    active = table[event_col].astype(bool)
    ax.plot(x_values, baseline, color="#4c78a8", linewidth=1.0)
    ax.scatter(pd.Series(x_values)[active], baseline[active], color="#e45756", s=14)
    ax.set_title("Econometric baseline versus event diagnostics")
    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(baseline_col))
    return fig, ax


def plot_empirical_stress_illustration(
    panel: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot a macro-financial empirical stress panel with a joint-stress timeline."""
    required = {
        "date",
        "unrate",
        "baa10ym",
        "kcfsi",
        "unrate_threshold",
        "baa10ym_threshold",
        "kcfsi_threshold",
        "unrate_exceedance",
        "baa10ym_exceedance",
        "kcfsi_exceedance",
        "stress_score",
        "joint_exceedance",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    path = _ensure_parent_dir(output_path)
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    series_specs = [
        ("unrate", "unrate_threshold", "unrate_exceedance", "Unemployment rate", "#c45b12"),
        ("baa10ym", "baa10ym_threshold", "baa10ym_exceedance", "Baa-Treasury spread", "#4c78a8"),
        ("kcfsi", "kcfsi_threshold", "kcfsi_exceedance", "KCFSI", "#54a24b"),
    ]
    for ax, (value_col, threshold_col, exceedance_col, title, color) in zip(axes[:3], series_specs, strict=True):
        values = pd.to_numeric(frame[value_col], errors="raise")
        threshold = float(pd.to_numeric(frame[threshold_col], errors="raise").iloc[0])
        exceedance = frame[exceedance_col].astype(bool)
        ax.plot(frame["date"], values, color=color, linewidth=1.1)
        ax.scatter(frame.loc[exceedance, "date"], values[exceedance], color="#e45756", s=14)
        ax.axhline(threshold, linestyle="--", linewidth=1.0, color="#72b7b2")
        ax.set_title(title)
        ax.set_ylabel(value_col)

    joint = frame["joint_exceedance"].astype(bool)
    stress_score = pd.to_numeric(frame["stress_score"], errors="raise")
    axes[3].plot(frame["date"], stress_score, color="#14213d", linewidth=1.2)
    axes[3].scatter(frame.loc[joint, "date"], stress_score[joint], color="#e45756", s=16)
    axes[3].set_title("Joint stress score")
    axes[3].set_ylabel("score")
    axes[3].set_xlabel("date")

    fig.tight_layout(h_pad=1.0)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_empirical_lambda_robustness(
    threshold_table: pd.DataFrame,
    run_length_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot empirical Lambda robustness over thresholds and run lengths."""
    threshold_required = {
        "series_id",
        "threshold_quantile",
        "lambda_runs",
        "lambda_runs_ci_lower",
        "lambda_runs_ci_upper",
    }
    run_length_required = {
        "series_id",
        "run_length",
        "lambda_runs",
        "lambda_runs_ci_lower",
        "lambda_runs_ci_upper",
    }
    missing_threshold = threshold_required.difference(threshold_table.columns)
    missing_run_length = run_length_required.difference(run_length_table.columns)
    if missing_threshold:
        raise ValueError(f"Missing required threshold-table columns: {sorted(missing_threshold)}")
    if missing_run_length:
        raise ValueError(f"Missing required run-length-table columns: {sorted(missing_run_length)}")

    path = _ensure_parent_dir(output_path)
    threshold_frame = threshold_table.copy()
    run_length_frame = run_length_table.copy()
    palette = {
        "unrate": "#c45b12",
        "baa10ym": "#4c78a8",
        "kcfsi": "#54a24b",
    }
    labels = {
        "unrate": "Unemployment rate",
        "baa10ym": "Baa-Treasury spread",
        "kcfsi": "KCFSI",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)

    for series_id, group in threshold_frame.groupby("series_id", sort=False):
        local = group.sort_values("threshold_quantile")
        x = pd.to_numeric(local["threshold_quantile"], errors="raise")
        y = pd.to_numeric(local["lambda_runs"], errors="raise")
        lower = pd.to_numeric(local["lambda_runs_ci_lower"], errors="raise")
        upper = pd.to_numeric(local["lambda_runs_ci_upper"], errors="raise")
        color = palette.get(str(series_id), "#14213d")
        axes[0].fill_between(x, lower, upper, color=color, alpha=0.15)
        axes[0].plot(x, y, marker="o", linewidth=1.4, color=color, label=labels.get(str(series_id), str(series_id)))

    axes[0].set_title("Lambda by threshold")
    axes[0].set_xlabel("threshold quantile")
    axes[0].set_ylabel("cluster-adjusted stress")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    for series_id, group in run_length_frame.groupby("series_id", sort=False):
        local = group.sort_values("run_length")
        x = pd.to_numeric(local["run_length"], errors="raise")
        y = pd.to_numeric(local["lambda_runs"], errors="raise")
        lower = pd.to_numeric(local["lambda_runs_ci_lower"], errors="raise")
        upper = pd.to_numeric(local["lambda_runs_ci_upper"], errors="raise")
        color = palette.get(str(series_id), "#14213d")
        axes[1].fill_between(x, lower, upper, color=color, alpha=0.15)
        axes[1].plot(x, y, marker="o", linewidth=1.4, color=color, label=labels.get(str(series_id), str(series_id)))

    axes[1].set_title("Lambda by run length")
    axes[1].set_xlabel("run length")
    axes[1].set_ylabel("cluster-adjusted stress")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_conceptual_pipeline(
    stages: list[str],
    output_path: str | Path,
) -> None:
    """Render a conceptual pipeline diagram for article-facing documentation."""
    if len(stages) == 0:
        raise ValueError("stages must be non-empty.")
    path = _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    n = len(stages)
    xs = np.linspace(0.08, 0.92, n)
    for idx, (x_pos, stage) in enumerate(zip(xs, stages, strict=True)):
        ax.text(
            x_pos,
            0.5,
            stage,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ec", "edgecolor": "#c45b12"},
            transform=ax.transAxes,
        )
        if idx < n - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.06, 0.5),
                xytext=(x_pos + 0.06, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#14213d", "lw": 1.5},
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    ax.set_title("Conceptual analysis pipeline")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

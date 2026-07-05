"""Extreme-value utilities for dependent econometric sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ExtremalIndexResult:
    """Result from a runs estimator of the extremal index.

    The extremal index is near 1 when extremes behave almost independently. It is lower
    than 1 when extremes tend to cluster, which is central in dynamical-systems extreme
    value theory and highly relevant for crisis econometrics.
    """

    threshold: float
    run_length: int
    n_exceedances: int
    n_clusters: int
    theta_hat: float


@dataclass(frozen=True)
class ExceedanceExtractionResult:
    """Container with exceedance extraction outputs."""

    threshold: float
    indicator: BoolArray
    indices: IntArray
    values: FloatArray


@dataclass(frozen=True)
class PeaksOverThresholdResult:
    """Container with exceedance excesses above a threshold."""

    threshold: float
    indices: IntArray
    exceedance_values: FloatArray
    excesses: FloatArray


@dataclass(frozen=True)
class ThresholdSensitivityResult:
    """Container for extremal-index sensitivity across thresholds."""

    quantile: float
    threshold: float
    n_exceedances: int
    n_clusters: int
    theta_runs: float
    theta_cluster_mean: float
    lambda_runs: float


@dataclass(frozen=True)
class BootstrapThresholdSensitivityResult:
    """Container for bootstrap uncertainty around threshold sensitivity results."""

    quantile: float
    threshold: float
    n_exceedances: int
    n_clusters: int
    theta_runs: float
    theta_cluster_mean: float
    lambda_runs: float
    theta_runs_ci_lower: float
    theta_runs_ci_upper: float
    lambda_runs_ci_lower: float
    lambda_runs_ci_upper: float
    n_bootstrap: int
    block_size: int


@dataclass(frozen=True)
class RunLengthSensitivityResult:
    """Container for run-length sensitivity at a fixed threshold quantile."""

    run_length: int
    threshold_quantile: float
    threshold: float
    n_exceedances: int
    n_clusters: int
    theta_runs: float
    lambda_runs: float


@dataclass(frozen=True)
class BootstrapRunLengthSensitivityResult:
    """Container for bootstrap uncertainty around run-length sensitivity results."""

    run_length: int
    threshold_quantile: float
    threshold: float
    n_exceedances: int
    n_clusters: int
    theta_runs: float
    lambda_runs: float
    lambda_runs_ci_lower: float
    lambda_runs_ci_upper: float
    n_bootstrap: int
    block_size: int


def _coerce_numeric_series(values: Sequence[float] | FloatArray) -> FloatArray:
    """Validate and convert input to a finite 1-D float64 NumPy array."""
    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=np.float64)
    elif isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float64)
    else:
        raise TypeError("values must be a one-dimensional np.ndarray or pd.Series.")

    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if arr.size == 0:
        raise ValueError("values must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("values must contain only finite numbers.")

    return arr.astype(np.float64, copy=False)


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_int(value: int, name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _run_length_validation(run_length: int) -> None:
    """Validate declustering run length."""
    if not isinstance(run_length, int) or run_length < 1:
        raise ValueError("run_length must be a positive integer.")


def _moving_block_bootstrap_indices(
    n_observations: int,
    block_size: int,
    rng: np.random.Generator,
) -> IntArray:
    """Construct moving-block bootstrap indices for a one-dimensional series."""
    _validate_positive_int(n_observations, "n_observations")
    _validate_positive_int(block_size, "block_size")
    if block_size > n_observations:
        raise ValueError("block_size must not exceed n_observations.")

    n_blocks = int(np.ceil(n_observations / block_size))
    max_start = n_observations - block_size
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    pieces = [np.arange(start, start + block_size, dtype=np.int64) for start in starts]
    indices = np.concatenate(pieces)
    return indices[:n_observations].astype(np.int64, copy=False)


def _estimate_runs_extremal_index_at_threshold(
    values: FloatArray,
    threshold: float,
    run_length: int,
) -> ExtremalIndexResult:
    """Estimate the runs clustering coefficient at a fixed numeric threshold."""
    _run_length_validation(run_length)
    values_arr = _coerce_numeric_series(values)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    cluster_sizes = runs_declustering(values_arr, threshold=threshold, run_length=run_length)
    n_exceedances = int(np.flatnonzero(values_arr > threshold).size)
    if n_exceedances == 0:
        return ExtremalIndexResult(float(threshold), run_length, 0, 0, float("nan"))

    n_clusters = int(cluster_sizes.size)
    theta_hat = float(n_clusters / n_exceedances)
    return ExtremalIndexResult(
        threshold=float(threshold),
        run_length=run_length,
        n_exceedances=n_exceedances,
        n_clusters=n_clusters,
        theta_hat=theta_hat,
    )


def _basic_bootstrap_interval(
    draws: FloatArray,
    point_estimate: float,
    ci_level: float,
    lower_bound: float = 0.0,
) -> tuple[float, float]:
    """Construct a basic bootstrap interval with a finite-precision containment guard."""
    if draws.size == 0 or not np.isfinite(point_estimate):
        return float("nan"), float("nan")

    alpha = 1.0 - ci_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    lower = float(2.0 * point_estimate - np.quantile(draws, upper_q))
    upper = float(2.0 * point_estimate - np.quantile(draws, lower_q))
    if np.isfinite(lower_bound):
        lower = max(lower_bound, lower)
    if point_estimate < lower or point_estimate > upper:
        radius = max(abs(point_estimate - lower), abs(upper - point_estimate))
        lower = max(lower_bound, float(point_estimate - radius))
        upper = float(point_estimate + radius)
    return lower, upper


def exceedance_indicator(values: FloatArray, threshold: float) -> BoolArray:
    """Return a boolean indicator for exceedances above a threshold."""
    values = _coerce_numeric_series(values)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    return values > threshold


def extract_threshold_exceedances(
    values: FloatArray,
    threshold: float,
) -> ExceedanceExtractionResult:
    """Extract exceedance metadata for a fixed threshold.

    Parameters
    ----------
    values:
        One-dimensional series.
    threshold:
        Numeric threshold used to define exceedances.
    """
    indicator = exceedance_indicator(values, threshold)
    values_arr = _coerce_numeric_series(values)
    indices = np.flatnonzero(indicator)
    exceedance_values = values_arr[indicator]
    return ExceedanceExtractionResult(
        threshold=float(threshold),
        indicator=indicator,
        indices=indices.astype(np.int64, copy=False),
        values=exceedance_values,
    )


def block_maxima(
    values: FloatArray | pd.DataFrame,
    block_size: int | str = 50,
    drop_incomplete_block: bool = True,
    value_col: str = "value",
    group_col: str = "series_id",
) -> FloatArray | pd.DataFrame:
    """Compute block maxima over disjoint blocks.

    Parameters
    ----------
    values:
        One-dimensional numeric array.
    block_size:
        Number of points per block.
    drop_incomplete_block:
        If True, drop a trailing incomplete block.
    """
    if isinstance(values, pd.DataFrame):
        if value_col not in values.columns or group_col not in values.columns or "date" not in values.columns:
            raise ValueError("DataFrame input must contain date, series_id, and value columns.")
        frame = values.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.sort_values([group_col, "date"]).reset_index(drop=True)
        output_frames: list[pd.DataFrame] = []
        for series_id, group in frame.groupby(group_col, sort=True):
            local = group.reset_index(drop=True)
            if isinstance(block_size, str):
                block = (
                    local.set_index("date")[value_col]
                    .resample(block_size)
                    .max()
                    .dropna()
                    .reset_index(name="block_max")
                )
                block[group_col] = str(series_id)
                block["block"] = np.arange(len(block), dtype=np.int64)
                output_frames.append(block[[group_col, "block", "date", "block_max"]])
            else:
                _validate_positive_int(block_size, "block_size")
                n_complete = len(local) // block_size
                n_blocks = n_complete if drop_incomplete_block else int(np.ceil(len(local) / block_size))
                rows: list[dict[str, object]] = []
                for block_id in range(n_blocks):
                    start = block_id * block_size
                    stop = min((block_id + 1) * block_size, len(local))
                    chunk = local.iloc[start:stop]
                    if chunk.empty:
                        continue
                    if drop_incomplete_block and len(chunk) < block_size:
                        continue
                    rows.append(
                        {
                            group_col: str(series_id),
                            "block": block_id,
                            "date": chunk["date"].iloc[0],
                            "block_max": float(chunk[value_col].max()),
                        }
                    )
                output_frames.append(pd.DataFrame(rows))
        return pd.concat(output_frames, ignore_index=True) if output_frames else pd.DataFrame(
            columns=[group_col, "block", "date", "block_max"]
        )

    values_arr = _coerce_numeric_series(values)
    if not isinstance(block_size, int):
        raise ValueError("block_size must be an integer for array input.")
    _validate_positive_int(block_size, "block_size")

    if drop_incomplete_block:
        n_complete = values_arr.size // block_size
        if n_complete == 0:
            raise ValueError("not enough observations for a complete block.")
        usable = values_arr[: n_complete * block_size]
        reshaped = usable.reshape(n_complete, block_size)
        return np.asarray(np.max(reshaped, axis=1), dtype=np.float64)

    output = []
    start = 0
    while start < values_arr.size:
        output.append(float(np.max(values_arr[start : start + block_size])))
        start += block_size
    return np.asarray(output, dtype=np.float64)


def peaks_over_threshold(
    values: FloatArray,
    threshold: float,
) -> PeaksOverThresholdResult:
    """Compute peaks-over-threshold values and excesses."""
    result = extract_threshold_exceedances(values, threshold)
    excesses = result.values - result.threshold
    return PeaksOverThresholdResult(
        threshold=result.threshold,
        indices=result.indices,
        exceedance_values=result.values,
        excesses=excesses,
    )


def runs_declustering_from_indicator(
    exceedance_indicator: BoolArray,
    run_length: int = 5,
) -> IntArray:
    """Convert a boolean exceedance array into cluster sizes by runs rule."""
    if exceedance_indicator.ndim != 1:
        raise ValueError("exceedance_indicator must be one-dimensional.")
    if exceedance_indicator.size == 0:
        raise ValueError("exceedance_indicator must not be empty.")
    if not np.issubdtype(exceedance_indicator.dtype, np.bool_):
        raise ValueError("exceedance_indicator must be boolean.")
    _run_length_validation(run_length)

    positions = np.flatnonzero(exceedance_indicator)
    if positions.size == 0:
        return np.zeros(0, dtype=np.int64)

    cluster_size = 1
    previous = int(positions[0])
    clusters: list[int] = []

    for position in positions[1:]:
        gap = int(position) - previous
        if gap > run_length:
            clusters.append(cluster_size)
            cluster_size = 1
        else:
            cluster_size += 1
        previous = int(position)

    clusters.append(cluster_size)
    return np.asarray(clusters, dtype=np.int64)


def runs_declustering(
    values: FloatArray,
    threshold: float,
    run_length: int = 5,
) -> IntArray:
    """Estimate run-based cluster sizes for exceedances."""
    _run_length_validation(run_length)
    indicator = exceedance_indicator(values, threshold)
    return runs_declustering_from_indicator(indicator, run_length=run_length)


def estimate_cluster_size_extremal_index(
    values: FloatArray,
    threshold: float,
    run_length: int = 5,
) -> float:
    """Estimate the extremal index as inverse mean cluster size."""
    _run_length_validation(run_length)
    cluster_sizes = runs_declustering(values, threshold, run_length=run_length)
    if cluster_sizes.size == 0:
        return float("nan")

    return 1.0 / float(np.mean(cluster_sizes))


def estimate_runs_extremal_index(
    values: FloatArray,
    threshold_quantile: float = 0.95,
    run_length: int = 5,
) -> ExtremalIndexResult:
    """Estimate the extremal index using a simple runs declustering estimator.

    Parameters
    ----------
    values:
        One-dimensional time series.
    threshold_quantile:
        Quantile used to define rare events.
    run_length:
        Number of consecutive non-exceedances needed to close a cluster.

    Returns
    -------
    ExtremalIndexResult
        Contains threshold, exceedance count, cluster count, and theta estimate.
    """
    _run_length_validation(run_length)
    values_arr = _coerce_numeric_series(values)
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    threshold = float(np.quantile(values_arr, threshold_quantile))
    return _estimate_runs_extremal_index_at_threshold(values_arr, threshold, run_length)


def estimate_extremal_index(
    values: FloatArray,
    threshold_quantile: float = 0.95,
    run_length: int = 5,
    method: str = "runs",
) -> float:
    """Estimate the extremal index with a simple method selector.

    Methods:
      - ``"runs"``: number of clusters divided by exceedance count.
      - ``"cluster_size_mean"``: inverse of mean cluster size.
    """
    if method not in {"runs", "cluster_size_mean"}:
        raise ValueError("method must be 'runs' or 'cluster_size_mean'.")
    if method == "runs":
        return estimate_runs_extremal_index(
            values,
            threshold_quantile=threshold_quantile,
            run_length=run_length,
        ).theta_hat

    values_arr = _coerce_numeric_series(values)
    _validate_positive_int(run_length, "run_length")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    threshold = float(np.quantile(values_arr, threshold_quantile))
    return estimate_cluster_size_extremal_index(values_arr, threshold, run_length=run_length)


def threshold_sensitivity_analysis(
    values: FloatArray,
    threshold_quantiles: Sequence[float] = (0.8, 0.9, 0.925, 0.95, 0.975, 0.99),
    run_length: int = 5,
) -> tuple[ThresholdSensitivityResult, ...]:
    """Evaluate extremal-index estimates across a grid of quantiles.

    Returns a tuple of results sorted by threshold quantile.
    """
    values_arr = _coerce_numeric_series(values)
    _run_length_validation(run_length)

    if not isinstance(threshold_quantiles, Sequence) or len(threshold_quantiles) == 0:
        raise ValueError("threshold_quantiles must be a non-empty sequence.")

    quantiles = [float(q) for q in threshold_quantiles]
    if any((q <= 0.0) or (q >= 1.0) for q in quantiles):
        raise ValueError("all threshold_quantiles must be in (0, 1).")

    results = []
    for q in quantiles:
        runs_result = estimate_runs_extremal_index(
            values_arr,
            threshold_quantile=q,
            run_length=run_length,
        )
        if runs_result.n_exceedances == 0:
            theta_cluster = float("nan")
        else:
            threshold = runs_result.threshold
            theta_cluster = estimate_cluster_size_extremal_index(
                values_arr,
                threshold=threshold,
                run_length=run_length,
            )

        results.append(
            ThresholdSensitivityResult(
                quantile=q,
                threshold=runs_result.threshold,
                n_exceedances=runs_result.n_exceedances,
                n_clusters=runs_result.n_clusters,
                theta_runs=runs_result.theta_hat,
                theta_cluster_mean=theta_cluster,
                lambda_runs=(
                    float(runs_result.n_exceedances / values_arr.size) / float(runs_result.theta_hat)
                    if runs_result.n_exceedances and np.isfinite(runs_result.theta_hat) and runs_result.theta_hat > 0.0
                    else float("nan")
                ),
            )
        )

    return tuple(results)


def bootstrap_threshold_sensitivity_analysis(
    values: FloatArray,
    threshold_quantiles: Sequence[float] = (0.8, 0.9, 0.925, 0.95, 0.975, 0.99),
    run_length: int = 5,
    n_bootstrap: int = 250,
    block_size: int = 24,
    seed: int = 13,
    ci_level: float = 0.90,
) -> tuple[BootstrapThresholdSensitivityResult, ...]:
    """Estimate threshold sensitivity together with moving-block bootstrap intervals."""
    min_clusters_for_interval = 3
    values_arr = _coerce_numeric_series(values)
    _run_length_validation(run_length)
    _validate_positive_int(n_bootstrap, "n_bootstrap")
    _validate_positive_int(block_size, "block_size")
    if block_size > values_arr.size:
        raise ValueError("block_size must not exceed the number of observations.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie in the open interval (0, 1).")

    point_results = threshold_sensitivity_analysis(
        values_arr,
        threshold_quantiles=threshold_quantiles,
        run_length=run_length,
    )
    quantiles = [row.quantile for row in point_results]
    thresholds = {row.quantile: row.threshold for row in point_results}
    rng = np.random.default_rng(seed)
    boot_thetas: dict[float, list[float]] = {q: [] for q in quantiles}
    boot_lambdas: dict[float, list[float]] = {q: [] for q in quantiles}

    for _ in range(n_bootstrap):
        sample_idx = _moving_block_bootstrap_indices(values_arr.size, block_size, rng)
        sample = values_arr[sample_idx]
        for quantile in quantiles:
            boot_row = _estimate_runs_extremal_index_at_threshold(
                sample,
                threshold=thresholds[quantile],
                run_length=run_length,
            )
            if np.isfinite(boot_row.theta_hat):
                boot_thetas[quantile].append(float(boot_row.theta_hat))
            if (
                boot_row.n_exceedances
                and np.isfinite(boot_row.theta_hat)
                and boot_row.theta_hat > 0.0
            ):
                boot_lambdas[quantile].append(
                    float((boot_row.n_exceedances / sample.size) / boot_row.theta_hat)
                )

    summary: list[BootstrapThresholdSensitivityResult] = []
    for row in point_results:
        draws = np.asarray(boot_thetas[row.quantile], dtype=np.float64)
        lambda_draws = np.asarray(boot_lambdas[row.quantile], dtype=np.float64)
        if draws.size == 0 or row.n_clusters < min_clusters_for_interval:
            lower = float("nan")
            upper = float("nan")
        else:
            lower, upper = _basic_bootstrap_interval(
                draws=draws,
                point_estimate=float(row.theta_runs),
                ci_level=ci_level,
                lower_bound=0.0,
            )
        if lambda_draws.size == 0 or row.n_clusters < min_clusters_for_interval:
            lambda_lower = float("nan")
            lambda_upper = float("nan")
        else:
            lambda_lower, lambda_upper = _basic_bootstrap_interval(
                draws=lambda_draws,
                point_estimate=float(row.lambda_runs),
                ci_level=ci_level,
                lower_bound=0.0,
            )
        summary.append(
            BootstrapThresholdSensitivityResult(
                quantile=row.quantile,
                threshold=row.threshold,
                n_exceedances=row.n_exceedances,
                n_clusters=row.n_clusters,
                theta_runs=row.theta_runs,
                theta_cluster_mean=row.theta_cluster_mean,
                lambda_runs=row.lambda_runs,
                theta_runs_ci_lower=lower,
                theta_runs_ci_upper=upper,
                lambda_runs_ci_lower=lambda_lower,
                lambda_runs_ci_upper=lambda_upper,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
            )
        )

    return tuple(summary)


def run_length_sensitivity_analysis(
    values: FloatArray,
    threshold_quantile: float = 0.90,
    run_lengths: Sequence[int] = (1, 2, 4, 6, 8),
) -> tuple[RunLengthSensitivityResult, ...]:
    """Evaluate clustering and cluster-adjusted stress across run lengths."""
    values_arr = _coerce_numeric_series(values)
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must lie in the open interval (0, 1).")
    if not isinstance(run_lengths, Sequence) or len(run_lengths) == 0:
        raise ValueError("run_lengths must be a non-empty sequence.")

    results: list[RunLengthSensitivityResult] = []
    for run_length in run_lengths:
        _run_length_validation(int(run_length))
        runs_result = estimate_runs_extremal_index(
            values_arr,
            threshold_quantile=threshold_quantile,
            run_length=int(run_length),
        )
        lambda_runs = (
            float(runs_result.n_exceedances / values_arr.size) / float(runs_result.theta_hat)
            if runs_result.n_exceedances and np.isfinite(runs_result.theta_hat) and runs_result.theta_hat > 0.0
            else float("nan")
        )
        results.append(
            RunLengthSensitivityResult(
                run_length=int(run_length),
                threshold_quantile=float(threshold_quantile),
                threshold=runs_result.threshold,
                n_exceedances=runs_result.n_exceedances,
                n_clusters=runs_result.n_clusters,
                theta_runs=runs_result.theta_hat,
                lambda_runs=lambda_runs,
            )
        )
    return tuple(results)


def bootstrap_run_length_sensitivity_analysis(
    values: FloatArray,
    threshold_quantile: float = 0.90,
    run_lengths: Sequence[int] = (1, 2, 4, 6, 8),
    n_bootstrap: int = 250,
    block_size: int = 40,
    seed: int = 31,
    ci_level: float = 0.90,
) -> tuple[BootstrapRunLengthSensitivityResult, ...]:
    """Estimate run-length sensitivity for lambda together with bootstrap intervals."""
    min_clusters_for_interval = 3
    values_arr = _coerce_numeric_series(values)
    _validate_positive_int(n_bootstrap, "n_bootstrap")
    _validate_positive_int(block_size, "block_size")
    if block_size > values_arr.size:
        raise ValueError("block_size must not exceed the number of observations.")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie in the open interval (0, 1).")

    point_results = run_length_sensitivity_analysis(
        values_arr,
        threshold_quantile=threshold_quantile,
        run_lengths=run_lengths,
    )
    run_lengths_list = [row.run_length for row in point_results]
    fixed_threshold = float(point_results[0].threshold)
    rng = np.random.default_rng(seed)
    boot_lambdas: dict[int, list[float]] = {run_length: [] for run_length in run_lengths_list}

    for _ in range(n_bootstrap):
        sample_idx = _moving_block_bootstrap_indices(values_arr.size, block_size, rng)
        sample = values_arr[sample_idx]
        for run_length in run_lengths_list:
            boot_row = _estimate_runs_extremal_index_at_threshold(
                sample,
                threshold=fixed_threshold,
                run_length=run_length,
            )
            if (
                boot_row.n_exceedances
                and np.isfinite(boot_row.theta_hat)
                and boot_row.theta_hat > 0.0
            ):
                boot_lambdas[run_length].append(
                    float((boot_row.n_exceedances / sample.size) / boot_row.theta_hat)
                )

    summary: list[BootstrapRunLengthSensitivityResult] = []
    for row in point_results:
        draws = np.asarray(boot_lambdas[row.run_length], dtype=np.float64)
        if draws.size == 0 or row.n_clusters < min_clusters_for_interval:
            lower = float("nan")
            upper = float("nan")
        else:
            lower, upper = _basic_bootstrap_interval(
                draws=draws,
                point_estimate=float(row.lambda_runs),
                ci_level=ci_level,
                lower_bound=0.0,
            )
        summary.append(
            BootstrapRunLengthSensitivityResult(
                run_length=row.run_length,
                threshold_quantile=row.threshold_quantile,
                threshold=row.threshold,
                n_exceedances=row.n_exceedances,
                n_clusters=row.n_clusters,
                theta_runs=row.theta_runs,
                lambda_runs=row.lambda_runs,
                lambda_runs_ci_lower=lower,
                lambda_runs_ci_upper=upper,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
            )
        )

    return tuple(summary)


def threshold_exceedances(
    df: pd.DataFrame,
    value_col: str = "value",
    threshold: float | None = None,
    quantile: float | None = None,
    side: Literal["upper", "lower", "two_sided"] = "upper",
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Annotate a long-format panel with threshold exceedances.

    Parameters
    ----------
    df:
        Long-format panel containing at least ``date``, ``series_id``, and
        ``value``.
    value_col:
        Numeric value column used to define exceedances.
    threshold:
        Fixed numeric threshold shared within each series unless ``quantile`` is
        used.
    quantile:
        Per-series quantile used to compute thresholds. Exactly one of
        ``threshold`` or ``quantile`` must be provided.
    side:
        Tail direction. ``two_sided`` applies the threshold to absolute values
        and therefore expects a transformed scale where that is meaningful.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Sorted event table with added ``threshold``, ``exceedance``, and
        ``side`` columns.

    Assumptions
    -----------
    Economic series are treated as observed time series whose rare-event
    structure can be described; the function does not assume they are generated
    by deterministic maps.

    Raises
    ------
    ValueError
        If required columns are missing or threshold arguments are invalid.
    """
    required = {"date", group_col, value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if side not in {"upper", "lower", "two_sided"}:
        raise ValueError("side must be 'upper', 'lower', or 'two_sided'.")
    if (threshold is None) == (quantile is None):
        raise ValueError("Provide exactly one of threshold or quantile.")
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="raise")
    frame = frame.sort_values([group_col, "date"]).reset_index(drop=True)
    if quantile is not None and not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in the open interval (0, 1).")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    threshold_map: dict[str, float] = {}
    for series_id, group in frame.groupby(group_col, sort=True):
        series = group[value_col]
        if quantile is None:
            local_threshold = float(threshold)
        elif side == "upper":
            local_threshold = float(series.quantile(quantile))
        elif side == "lower":
            local_threshold = float(series.quantile(1.0 - quantile))
        else:
            local_threshold = float(series.abs().quantile(quantile))
        threshold_map[str(series_id)] = local_threshold

    frame["threshold"] = frame[group_col].astype(str).map(threshold_map).astype(float)
    if side == "upper":
        frame["exceedance"] = frame[value_col] >= frame["threshold"]
    elif side == "lower":
        frame["exceedance"] = frame[value_col] <= frame["threshold"]
    else:
        frame["exceedance"] = frame[value_col].abs() >= frame["threshold"]
    frame["side"] = side
    return frame


def assign_runs(
    events: pd.DataFrame,
    run_length: int,
    date_col: str = "date",
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Assign run identifiers to exceedance events in a sorted event table.

    Parameters
    ----------
    events:
        Event table containing an ``exceedance`` boolean column.
    run_length:
        Number of non-exceedance observations required to close a cluster.
    date_col:
        Event date column.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with a nullable integer ``run_id`` column.

    Assumptions
    -----------
    Clustering is defined in observation space: gaps are counted as numbers of
    observations between exceedances after sorting by ``group_col`` and
    ``date_col``.

    Raises
    ------
    ValueError
        If required columns are missing or ``run_length`` is invalid.
    """
    _run_length_validation(run_length)
    required = {date_col, group_col, "exceedance"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = events.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame = frame.sort_values([group_col, date_col]).reset_index(drop=True)
    run_ids = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for _, indexer in frame.groupby(group_col, sort=True).groups.items():
        local_index = list(indexer)
        exceed_positions = [pos for pos, idx in enumerate(local_index) if bool(frame.at[idx, "exceedance"])]
        if not exceed_positions:
            continue
        current_run = 0
        run_ids.at[local_index[exceed_positions[0]]] = current_run
        previous_position = exceed_positions[0]
        for position in exceed_positions[1:]:
            if position - previous_position > run_length:
                current_run += 1
            run_ids.at[local_index[position]] = current_run
            previous_position = position
    frame["run_id"] = run_ids
    return frame


def runs_extremal_index(
    events: pd.DataFrame,
    run_length: int,
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Summarize a runs-based extremal-index diagnostic by series.

    Parameters
    ----------
    events:
        Event table with an ``exceedance`` boolean column and one row per
        observation.
    run_length:
        Number of non-exceedance observations required to separate clusters.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Tidy summary with observation counts, exceedance counts, cluster counts,
        run length, and the runs extremal-index estimate.

    Assumptions
    -----------
    The returned extremal index is descriptive and should not be interpreted
    mechanically in short empirical samples.

    Raises
    ------
    ValueError
        If the input event table is malformed.
    """
    assigned = assign_runs(events, run_length=run_length, group_col=group_col)
    rows: list[dict[str, object]] = []
    for series_id, group in assigned.groupby(group_col, sort=True):
        n_observations = int(len(group))
        exceedances = group[group["exceedance"]]
        n_exceedances = int(len(exceedances))
        n_clusters = int(exceedances["run_id"].nunique()) if n_exceedances else 0
        extremal_index = float(n_clusters / n_exceedances) if n_exceedances else float("nan")
        rows.append(
            {
                group_col: str(series_id),
                "n_observations": n_observations,
                "n_exceedances": n_exceedances,
                "n_clusters": n_clusters,
                "run_length": run_length,
                "extremal_index": extremal_index,
            }
        )
    return pd.DataFrame(rows)


def mean_excess_table(
    df: pd.DataFrame,
    quantiles: Sequence[float],
    value_col: str = "value",
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Compute a mean-excess table across threshold quantiles.

    Parameters
    ----------
    df:
        Long-format panel containing ``series_id`` and ``value``.
    quantiles:
        Threshold quantiles at which to compute mean excesses.
    value_col:
        Numeric value column.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Summary with ``series_id``, ``quantile``, ``threshold``,
        ``n_exceedances``, and ``mean_excess``.

    Raises
    ------
    ValueError
        If the input frame or quantiles are invalid.
    """
    if not quantiles:
        raise ValueError("quantiles must be non-empty.")
    if any((q <= 0.0) or (q >= 1.0) for q in quantiles):
        raise ValueError("All quantiles must lie in the open interval (0, 1).")
    required = {group_col, value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = df.copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="raise")
    rows: list[dict[str, object]] = []
    for series_id, group in frame.groupby(group_col, sort=True):
        values = group[value_col]
        for quantile in quantiles:
            threshold = float(values.quantile(quantile))
            exceedances = values[values >= threshold]
            mean_excess = float((exceedances - threshold).mean()) if not exceedances.empty else float("nan")
            rows.append(
                {
                    group_col: str(series_id),
                    "quantile": float(quantile),
                    "threshold": threshold,
                    "n_exceedances": int(len(exceedances)),
                    "mean_excess": mean_excess,
                }
            )
    return pd.DataFrame(rows)

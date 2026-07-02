from __future__ import annotations

import numpy as np

import pandas as pd

from dynsys_econometrics.extremes import (
    block_maxima,
    estimate_extremal_index,
    estimate_runs_extremal_index,
    extract_threshold_exceedances,
    peaks_over_threshold,
    runs_declustering,
    runs_declustering_from_indicator,
    threshold_sensitivity_analysis,
)
from dynsys_econometrics.return_times import compute_return_times


def test_runs_extremal_index_is_between_zero_and_one() -> None:
    values = np.array([0.0, 1.0, 2.0, 0.1, 0.2, 3.0, 4.0, 0.0], dtype=float)
    result = estimate_runs_extremal_index(values, threshold_quantile=0.7, run_length=1)
    assert 0.0 < result.theta_hat <= 1.0
    assert result.n_exceedances > 0


def test_block_maxima_keeps_only_complete_blocks_by_default() -> None:
    values = np.arange(1, 11, dtype=float)
    maxima = block_maxima(values, block_size=4)
    assert maxima.shape == (2,)
    assert np.allclose(maxima, [4.0, 8.0])


def test_block_maxima_includes_last_incomplete_block() -> None:
    values = np.arange(1, 11, dtype=float)
    maxima = block_maxima(values, block_size=4, drop_incomplete_block=False)
    assert maxima.shape == (3,)
    assert np.allclose(maxima, [4.0, 8.0, 10.0])


def test_exceedance_extraction_and_pot() -> None:
    values = np.array([0.0, 1.0, 2.5, 3.0, 5.0], dtype=float)
    result = extract_threshold_exceedances(values, threshold=1.5)
    assert result.values.size == 3
    assert np.array_equal(result.indices, np.array([2, 3, 4]))

    pot = peaks_over_threshold(values, threshold=1.5)
    assert pot.indices.shape[0] == 3
    assert np.allclose(pot.excesses, np.array([1.0, 1.5, 3.5]))


def test_runs_declustering_handles_clustered_examples() -> None:
    values = np.array([0.0, 3.0, 4.0, 0.1, 4.5, 5.0, 0.1, 0.0, 0.0, 4.0], dtype=float)
    clusters = runs_declustering(values, threshold=1.0, run_length=1)
    assert np.array_equal(clusters, np.array([2, 2, 1]))


def test_runs_declustering_from_indicator_with_singletons() -> None:
    indicator = np.array([False, True, False, True, False, True], dtype=bool)
    clusters = runs_declustering_from_indicator(indicator, run_length=1)
    assert np.array_equal(clusters, np.array([1, 1, 1]))


def test_exceedance_methods_accept_pandas_series() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 0.0, 3.0])
    result = estimate_extremal_index(values, threshold_quantile=0.6)
    assert 0.0 <= result <= 1.0


def test_threshold_sensitivity_analysis_returns_expected_shape() -> None:
    values = np.array([0.2, 1.0, 1.3, 2.0, 2.5, 3.5, 4.0, 4.6, 5.0], dtype=float)
    results = threshold_sensitivity_analysis(values, threshold_quantiles=[0.7, 0.8], run_length=1)
    assert len(results) == 2
    assert all(0.0 <= x.theta_runs <= 1.0 for x in results)


def test_return_times() -> None:
    values = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 5.0], dtype=float)
    return_times = compute_return_times(values, threshold=1.0)
    assert return_times.tolist() == [2, 2]

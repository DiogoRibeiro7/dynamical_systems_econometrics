from __future__ import annotations

import numpy as np
import pytest

from dynsys_econometrics.return_times import (
    ReturnTimeDistribution,
    empirical_survival_curve,
    exponential_benchmark_comparison,
    ferro_segers_intervals_estimator,
    fitted_mixture_quantiles,
    first_hitting_time,
    ks_exponential_diagnostic,
    return_time_distribution,
    compute_return_times,
)


def test_first_hitting_time_found_and_missing() -> None:
    values = np.array([0.0, 0.2, 1.5, 0.1], dtype=float)
    assert first_hitting_time(values, threshold=1.0) == 2
    assert first_hitting_time(values, threshold=1.0, start_index=3) == -1


def test_return_times_no_hits_and_single_hit() -> None:
    no_hits = np.array([0.0, 0.2, 0.5], dtype=float)
    single_hit = np.array([0.0, 1.5, 0.2], dtype=float)
    assert compute_return_times(no_hits, threshold=1.0).size == 0
    assert compute_return_times(single_hit, threshold=1.0).size == 0


def test_return_time_distribution_shapes() -> None:
    values = np.array([0.0, 1.2, 0.2, 2.1, 0.1, 3.0, 0.0, 4.2], dtype=float)
    distribution = return_time_distribution(values, threshold=1.0)
    assert isinstance(distribution, ReturnTimeDistribution)
    assert distribution.return_times.tolist() == [2, 2, 2]
    assert distribution.unique_times.tolist() == [2]
    assert np.isclose(distribution.probabilities.sum(), 1.0)


def test_empirical_survival_curve() -> None:
    unique, survival = empirical_survival_curve([1, 2, 2, 3, 3, 3])
    assert unique.tolist() == [1, 2, 3]
    assert np.allclose(survival, [1.0, 5.0 / 6.0, 1.0 / 2.0])


def test_empirical_survival_curve_empty() -> None:
    unique, survival = empirical_survival_curve([])
    assert unique.size == 0
    assert survival.size == 0


def test_exponential_benchmark_invalid_empty_series() -> None:
    with pytest.raises(ValueError):
        exponential_benchmark_comparison([])


def test_ks_exponential_diagnostic_for_short_series() -> None:
    values = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 5.0], dtype=float)
    statistic, pvalue, reject, n = ks_exponential_diagnostic(values, threshold=1.0, alpha=0.5)
    assert n == 2
    assert 0.0 <= statistic <= 1.0
    assert 0.0 <= pvalue <= 1.0
    assert isinstance(reject, bool)


def test_ferro_segers_intervals_estimator_moment_variant() -> None:
    result = ferro_segers_intervals_estimator([1, 1, 2])
    assert result.estimator_variant == "moment"
    assert result.n_intervals == 3
    assert result.max_interval == 2
    assert result.theta_hat == pytest.approx(1.0)


def test_ferro_segers_intervals_estimator_bias_corrected_variant() -> None:
    result = ferro_segers_intervals_estimator([3, 5, 9])
    expected = min(1.0, 2.0 * (14.0**2) / (3.0 * ((2.0 * 1.0) + (4.0 * 3.0) + (8.0 * 7.0))))
    assert result.estimator_variant == "bias_corrected"
    assert result.n_intervals == 3
    assert result.max_interval == 9
    assert result.theta_hat == pytest.approx(expected)


def test_fitted_mixture_quantiles_respect_atom() -> None:
    quantiles = fitted_mixture_quantiles([0.10, 0.25, 0.60], theta_hat=0.75)
    assert quantiles[0] == pytest.approx(0.0)
    assert quantiles[1] == pytest.approx(0.0)
    assert quantiles[2] == pytest.approx(-np.log(0.4 / 0.75) / 0.75)

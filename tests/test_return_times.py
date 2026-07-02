from __future__ import annotations

import numpy as np
import pytest

from dynsys_econometrics.return_times import (
    ReturnTimeDistribution,
    empirical_survival_curve,
    exponential_benchmark_comparison,
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
    assert distribution.return_times.tolist() == [1, 2, 2]
    assert distribution.unique_times.tolist() == [1, 2]
    assert np.isclose(distribution.probabilities.sum(), 1.0)


def test_empirical_survival_curve() -> None:
    unique, survival = empirical_survival_curve([1, 2, 2, 3, 3, 3])
    assert unique.tolist() == [1, 2, 3]
    assert np.allclose(survival, [1.0, 2.0 / 3.0, 1.0 / 2.0])


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

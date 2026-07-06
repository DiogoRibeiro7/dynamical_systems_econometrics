from __future__ import annotations

import numpy as np

import pandas as pd

from dynsys_econometrics.extremes import (
    block_maxima,
    bootstrap_threshold_sensitivity_analysis,
    estimate_extremal_index,
    estimate_runs_extremal_index,
    extract_threshold_exceedances,
    peaks_over_threshold,
    runs_declustering,
    runs_declustering_from_indicator,
    threshold_sensitivity_analysis,
)
from dynsys_econometrics.simulation import simulate_garch11
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


def test_perturbation_bound_is_additive_and_product_form_can_fail() -> None:
    p_grid = [0.05, 0.1, 0.2, 0.35]
    theta_grid = [0.08, 0.15, 0.3, 0.6]
    eps_p_grid = [0.005, 0.01, 0.02]
    eps_theta_grid = [0.01, 0.03, 0.05]

    product_violation_found = False
    for p in p_grid:
        for theta in theta_grid:
            for eps_p in eps_p_grid:
                for eps_theta in eps_theta_grid:
                    if eps_theta >= theta:
                        continue

                    additive_bound = (eps_p / (theta - eps_theta)) + (
                        p * eps_theta / (theta * (theta - eps_theta))
                    )
                    product_bound = (eps_p / (theta - eps_theta)) * (
                        p * eps_theta / (theta * (theta - eps_theta))
                    )

                    worst_case_error = 0.0
                    for dp in (-eps_p, eps_p):
                        for dtheta in (-eps_theta, eps_theta):
                            p_tilde = p + dp
                            theta_tilde = theta + dtheta
                            if p_tilde <= 0.0 or theta_tilde <= 0.0:
                                continue
                            error = abs((p_tilde / theta_tilde) - (p / theta))
                            worst_case_error = max(worst_case_error, error)

                    assert worst_case_error <= additive_bound + 1e-12
                    if worst_case_error > product_bound + 1e-12:
                        product_violation_found = True

    assert product_violation_found


def test_clustered_bootstrap_coverage_harness_fixed_seed() -> None:
    reference_values = np.abs(simulate_garch11(n_steps=20_000, seed=611))
    reference_result = estimate_runs_extremal_index(
        reference_values,
        threshold_quantile=0.90,
        run_length=3,
    )
    reference_lambda = (
        (reference_result.n_exceedances / reference_values.size) / reference_result.theta_hat
    )

    theta_hits = 0
    lambda_hits = 0
    interpretable_theta = 0
    interpretable_lambda = 0
    n_replications = 8

    for replication in range(n_replications):
        sample = np.abs(simulate_garch11(n_steps=3000, seed=800 + replication))
        summary = bootstrap_threshold_sensitivity_analysis(
            sample,
            threshold_quantiles=[0.90],
            run_length=3,
            n_bootstrap=40,
            block_size=24,
            seed=100 + replication,
            ci_level=0.90,
        )[0]
        if np.isfinite(summary.theta_runs_ci_lower) and np.isfinite(summary.theta_runs_ci_upper):
            interpretable_theta += 1
            if summary.theta_runs_ci_lower <= reference_result.theta_hat <= summary.theta_runs_ci_upper:
                theta_hits += 1
        if np.isfinite(summary.lambda_runs_ci_lower) and np.isfinite(summary.lambda_runs_ci_upper):
            interpretable_lambda += 1
            if summary.lambda_runs_ci_lower <= reference_lambda <= summary.lambda_runs_ci_upper:
                lambda_hits += 1

    assert interpretable_theta == n_replications
    assert interpretable_lambda == n_replications
    assert theta_hits >= 5
    assert lambda_hits >= 5

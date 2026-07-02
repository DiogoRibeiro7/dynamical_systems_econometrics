from __future__ import annotations

import numpy as np

from dynsys_econometrics.extremes import estimate_runs_extremal_index
from dynsys_econometrics.return_times import compute_return_times


def test_runs_extremal_index_is_between_zero_and_one() -> None:
    values = np.array([0.0, 1.0, 2.0, 0.1, 0.2, 3.0, 4.0, 0.0], dtype=float)
    result = estimate_runs_extremal_index(values, threshold_quantile=0.7, run_length=1)
    assert 0.0 < result.theta_hat <= 1.0
    assert result.n_exceedances > 0


def test_return_times() -> None:
    values = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 5.0], dtype=float)
    return_times = compute_return_times(values, threshold=1.0)
    assert return_times.tolist() == [2, 2]

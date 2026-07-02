"""Run a minimal controlled experiment for the repository."""

from __future__ import annotations

import numpy as np

from dynsys_econometrics.extremes import estimate_runs_extremal_index
from dynsys_econometrics.return_times import compute_return_times
from dynsys_econometrics.simulation import logistic_map, simulate_ar1


def main() -> None:
    """Compare rare-event clustering in a chaotic map and AR(1) baseline."""
    chaotic = logistic_map(n_steps=5_000, x0=0.123, r=4.0)
    ar1 = simulate_ar1(n_steps=5_000, phi=0.8, sigma=1.0, seed=7)

    chaotic_observable = -np.log(np.abs(chaotic - 0.5) + 1e-12)
    ar1_observable = np.abs(ar1)

    for name, values in {"logistic_map": chaotic_observable, "ar1_abs": ar1_observable}.items():
        result = estimate_runs_extremal_index(values, threshold_quantile=0.95, run_length=5)
        return_times = compute_return_times(values, result.threshold)
        mean_return_time = float(np.mean(return_times)) if return_times.size else float("nan")
        print(
            f"{name}: threshold={result.threshold:.4f}, "
            f"exceedances={result.n_exceedances}, clusters={result.n_clusters}, "
            f"theta={result.theta_hat:.4f}, mean_return_time={mean_return_time:.2f}"
        )


if __name__ == "__main__":
    main()

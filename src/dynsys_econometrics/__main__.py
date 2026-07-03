"""Package CLI for reproducible repository workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dynsys_econometrics.config import load_config, resolve_output_dir
from dynsys_econometrics.data import materialize_catalog, validate_catalog
from dynsys_econometrics.experiments import (
    run_empirical_experiment,
    run_synthetic_experiment,
    save_experiment_result,
)
from dynsys_econometrics.extremes import estimate_runs_extremal_index
from dynsys_econometrics.return_times import compute_return_times
from dynsys_econometrics.simulation import logistic_map, simulate_ar1


def _run_smoke() -> None:
    """Run the minimal controlled rare-event comparison."""
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


def main() -> None:
    """Run the package CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m dynsys_econometrics",
        description="Run reproducible dynsys-econometrics workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("smoke", help="Run the minimal rare-event smoke experiment.")

    synthetic_parser = subparsers.add_parser("synthetic", help="Run a synthetic experiment config.")
    synthetic_parser.add_argument("--config", default="configs/synthetic.yaml", help="Path to synthetic YAML config.")

    empirical_parser = subparsers.add_parser("empirical", help="Run an empirical experiment config.")
    empirical_parser.add_argument("--config", default="configs/empirical.yaml", help="Path to empirical YAML config.")

    catalog_parser = subparsers.add_parser("validate-catalog", help="Validate a public-data catalog.")
    catalog_parser.add_argument("--catalog", default="data/catalog.example.yaml", help="Path to catalog YAML.")

    materialize_parser = subparsers.add_parser("materialize-catalog", help="Load a public-data catalog into a canonical panel.")
    materialize_parser.add_argument("--catalog", default="data/catalog.example.yaml", help="Path to catalog YAML.")
    materialize_parser.add_argument("--output", default="data/processed/catalog_panel.csv", help="Output CSV path.")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "smoke":
        _run_smoke()
        return

    if args.command == "synthetic":
        config = load_config(args.config)
        result = run_synthetic_experiment(config)
        output_dir = resolve_output_dir(config)
        save_experiment_result(result, output_dir)
        print(f"synthetic_experiment={result.name}")
        print(f"output_dir={output_dir}")
        return

    if args.command == "empirical":
        config = load_config(args.config)
        result = run_empirical_experiment(config)
        output_dir = resolve_output_dir(config)
        save_experiment_result(result, output_dir)
        print(f"empirical_experiment={result.name}")
        print(f"output_dir={output_dir}")
        return

    if args.command == "validate-catalog":
        summary = validate_catalog(Path(args.catalog))
        print(f"catalog={args.catalog}")
        print(f"n_series={summary['n_series']}")
        print(f"valid={summary['valid']}")
        return

    if args.command == "materialize-catalog":
        summary = materialize_catalog(Path(args.catalog), args.output)
        print(f"catalog={args.catalog}")
        print(f"output={summary['output_path']}")
        print(f"n_rows={summary['n_rows']}")
        print(f"n_series={summary['n_series']}")
        return


if __name__ == "__main__":
    main()

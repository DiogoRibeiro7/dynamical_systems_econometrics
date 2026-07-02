"""Run a configured synthetic or empirical experiment."""

from __future__ import annotations

import argparse

from dynsys_econometrics.config import load_config, resolve_output_dir
from dynsys_econometrics.experiments import (
    run_empirical_experiment,
    run_synthetic_experiment,
    save_experiment_result,
)


def main() -> None:
    """Run an experiment from a YAML configuration file."""
    parser = argparse.ArgumentParser(description="Run a configured dynsys-econometrics experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = config.get("experiment", {})
    mode = str(experiment.get("mode", "synthetic")).lower()
    if mode == "synthetic":
        result = run_synthetic_experiment(config)
    elif mode == "empirical":
        result = run_empirical_experiment(config)
    else:
        raise ValueError(f"Unsupported experiment mode: {mode}")
    output_dir = resolve_output_dir(config)
    save_experiment_result(result, output_dir=output_dir)
    print(f"experiment={result.name}")
    print(f"mode={mode}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

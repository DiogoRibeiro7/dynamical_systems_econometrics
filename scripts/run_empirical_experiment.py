"""Run the default empirical experiment configuration."""

from __future__ import annotations

from dynsys_econometrics.config import load_config, resolve_output_dir
from dynsys_econometrics.experiments import run_empirical_experiment, save_experiment_result


def main() -> None:
    """Run the default empirical experiment."""
    config = load_config("configs/empirical.yaml")
    result = run_empirical_experiment(config)
    output_dir = resolve_output_dir(config)
    save_experiment_result(result, output_dir)
    print(f"empirical_experiment={result.name}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

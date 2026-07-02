"""Run the default synthetic experiment configuration."""

from __future__ import annotations

from dynsys_econometrics.config import load_config, resolve_output_dir
from dynsys_econometrics.experiments import run_synthetic_experiment, save_experiment_result


def main() -> None:
    """Run the default synthetic experiment."""
    config = load_config("configs/synthetic.yaml")
    result = run_synthetic_experiment(config)
    output_dir = resolve_output_dir(config)
    save_experiment_result(result, output_dir)
    print(f"synthetic_experiment={result.name}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

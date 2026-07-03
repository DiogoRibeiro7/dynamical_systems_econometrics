"""Refresh cached remote empirical sources without running a full experiment."""

from __future__ import annotations

import argparse

from dynsys_econometrics.config import load_config
from dynsys_econometrics.data import refresh_empirical_cache


def main() -> None:
    """Refresh remote-backed empirical caches from a config file."""
    parser = argparse.ArgumentParser(
        description="Refresh cached remote empirical sources for dynsys-econometrics."
    )
    parser.add_argument(
        "--config",
        default="configs/empirical_fred_refresh.yaml",
        help="Path to the empirical YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    empirical = config.get("empirical", {})
    summary = refresh_empirical_cache(empirical)
    print(f"refresh_config={args.config}")
    print(f"n_series={summary['n_series']}")
    print(f"n_rows={summary['n_rows']}")
    if summary["targets"]:
        print(f"targets={','.join(summary['targets'])}")


if __name__ == "__main__":
    main()

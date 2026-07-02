"""Validate a public-data catalog and document offline-first usage."""

from __future__ import annotations

import argparse

from dynsys_econometrics.data import validate_catalog


def main() -> None:
    """Validate a public-data catalog file."""
    parser = argparse.ArgumentParser(description="Validate a public data catalog for dynsys-econometrics.")
    parser.add_argument(
        "--catalog",
        default="data/catalog.example.yaml",
        help="Path to the YAML catalog file.",
    )
    args = parser.parse_args()
    summary = validate_catalog(args.catalog)
    print(f"catalog={args.catalog}")
    print(f"n_series={summary['n_series']}")
    print(f"valid={summary['valid']}")


if __name__ == "__main__":
    main()

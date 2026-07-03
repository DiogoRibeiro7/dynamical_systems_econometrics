"""Validate a public-data catalog and document offline-first usage."""

from __future__ import annotations

import argparse

from dynsys_econometrics.data import materialize_catalog, validate_catalog


def main() -> None:
    """Validate a public-data catalog file."""
    parser = argparse.ArgumentParser(description="Validate a public data catalog for dynsys-econometrics.")
    parser.add_argument(
        "--catalog",
        default="data/catalog.example.yaml",
        help="Path to the YAML catalog file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. When provided, the catalog is materialized into a canonical panel.",
    )
    args = parser.parse_args()
    if args.output:
        summary = materialize_catalog(args.catalog, args.output)
        print(f"catalog={args.catalog}")
        print(f"output={summary['output_path']}")
        print(f"n_rows={summary['n_rows']}")
        print(f"n_series={summary['n_series']}")
        return

    summary = validate_catalog(args.catalog)
    print(f"catalog={args.catalog}")
    print(f"n_series={summary['n_series']}")
    print(f"valid={summary['valid']}")


if __name__ == "__main__":
    main()

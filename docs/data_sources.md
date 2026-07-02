# Data Sources

## Scope

The repository supports two data modes:

- offline mode from local CSV files
- optional online mode from public sources

Default tests do not require internet access.

## Canonical raw schema

```text
date,series_id,value,country,frequency,unit,source
```

Only `date`, `series_id`, and `value` are required.

## Supported public source families

- FRED / ALFRED
- OECD CSV exports
- World Bank indicators
- ECB Statistical Data Warehouse
- Yahoo Finance via `yfinance`

## Article caution

These sources are examples of supported input families. They are not automatically evidence used by the article unless explicitly loaded, transformed, and saved in reproducible outputs.

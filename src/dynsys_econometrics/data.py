"""Data-loading contracts for free macro-financial datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from pandas import DataFrame


_SUPPORTED_YEAR_COLUMNS = tuple(str(year) for year in range(1900, 2201))


class DataLoadFailure(ValueError):
    """Raised for invalid loader inputs."""


def _to_standard_schema(
    frame: DataFrame,
    date_series: pd.Series,
    value_series: pd.Series,
    series_id_series: pd.Series | None,
) -> DataFrame:
    """Create the standard repository schema from aligned date/value(/id) columns."""
    date = pd.to_datetime(date_series, errors="raise")
    value = pd.to_numeric(value_series, errors="raise")
    if date.isna().any():
        raise DataLoadFailure("Date column has missing or unparsable values.")
    if value.isna().any():
        raise DataLoadFailure("Value column has missing or unparsable numeric values.")

    if series_id_series is None:
        if "series_id" in frame.columns:
            series_id_series = frame["series_id"].astype(str)
        else:
            raise DataLoadFailure("A series_id is required for the returned schema.")

    standardized = pd.DataFrame(
        {
            "date": date,
            "series_id": series_id_series.astype(str),
            "value": value,
        }
    )
    return standardized.sort_values(["series_id", "date"]).reset_index(drop=True)


def _load_csv_to_frame(path: str | Path, **read_kwargs) -> DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    frame = pd.read_csv(csv_path, **read_kwargs)
    if frame.empty:
        raise DataLoadFailure("CSV file has no rows.")
    return frame


def _standardize_simple_csv(
    frame: DataFrame,
    date_col: str,
    value_col: str,
    *,
    series_id: str | None = None,
    series_col: str | None = None,
) -> DataFrame:
    missing = {date_col, value_col}.difference(frame.columns)
    if missing:
        raise DataLoadFailure(f"Missing required columns: {sorted(missing)}")

    if series_id is not None:
        series_ids = pd.Series([series_id] * len(frame))
    elif series_col is not None:
        if series_col not in frame.columns:
            raise DataLoadFailure(f"series_col '{series_col}' not found in CSV.")
        series_ids = frame[series_col].astype(str)
    elif "series_id" in frame.columns:
        series_ids = frame["series_id"].astype(str)
    else:
        raise DataLoadFailure("Either `series_id` or `series_col` must be provided.")

    return _to_standard_schema(
        frame=frame,
        date_series=frame[date_col],
        value_series=frame[value_col],
        series_id_series=series_ids,
    )


def load_time_series_csv(
    path: str | Path,
    *,
    series_id: str | None = None,
    date_col: str = "date",
    value_col: str = "value",
    series_col: str | None = None,
) -> DataFrame:
    """Load a generic CSV time-series into the standard schema.

    Parameters
    ----------
    path:
        CSV path.
    series_id:
        Explicit series id for output `series_id`.
    date_col:
        Source date column.
    value_col:
        Source value column.
    series_col:
        Optional column that contains per-row series identifiers.
    """
    frame = _load_csv_to_frame(path)
    return _standardize_simple_csv(
        frame=frame,
        date_col=date_col,
        value_col=value_col,
        series_id=series_id,
        series_col=series_col,
    )


def load_fred_csv(
    path: str | Path,
    *,
    series_id: str,
    date_col: str = "DATE",
    value_col: str = "VALUE",
) -> DataFrame:
    """Load FRED CSV exports or manually prepared FRED-formatted files."""
    frame = _load_csv_to_frame(path)
    return _standardize_simple_csv(
        frame=frame,
        date_col=date_col,
        value_col=value_col,
        series_id=series_id,
    )


def load_fred(
    series_id: str,
    *,
    csv_path: str | Path | None = None,
    api_key: str | None = None,
    api_start: str | None = None,
    api_end: str | None = None,
) -> DataFrame:
    """Load a FRED series from local CSV or FRED API (optional)."""
    if csv_path is not None:
        return load_fred_csv(csv_path, series_id=series_id)
    if api_key is None:
        raise DataLoadFailure(
            "`api_key` is required when `csv_path` is not provided."
        )

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": api_start,
        "observation_end": api_end,
    }
    params = {k: v for k, v in params.items() if v is not None}
    endpoint = f"https://api.stlouisfed.org/fred/series/observations?{urlencode(params)}"

    with urlopen(Request(endpoint, headers={"User-Agent": "dynsys-econometrics"}), timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "error_code" in payload:
        raise DataLoadFailure(f"FRED API error: {payload.get('error_message')}")

    observations = payload.get("observations", [])
    if not observations:
        raise DataLoadFailure("FRED API returned no observations.")

    frame = pd.DataFrame.from_records(observations)
    if not {"date", "value"}.issubset(set(frame.columns)):
        raise DataLoadFailure("Unexpected FRED API payload schema.")

    return _to_standard_schema(
        frame=frame,
        date_series=frame["date"],
        value_series=frame["value"].replace(".", pd.NA),
        series_id_series=pd.Series([series_id] * len(frame)),
    )


def load_ecb_sdw_csv(
    path: str | Path,
    *,
    series_id: str | None = None,
    date_col: str = "TIME_PERIOD",
    value_col: str = "OBS_VALUE",
    series_col: str = "SERIES",
) -> DataFrame:
    """Load ECB SDW CSV files into standard schema."""
    frame = _load_csv_to_frame(path)
    if date_col not in frame.columns:
        raise DataLoadFailure(f"Missing required columns: ['{date_col}']")

    if value_col not in frame.columns and series_col in frame.columns:
        raise DataLoadFailure(
            f"Missing required columns: ['{value_col}']"
        )

    return _standardize_simple_csv(
        frame=frame,
        date_col=date_col,
        value_col=value_col,
        series_id=series_id,
        series_col=series_col if series_id is None else None,
    )


def load_oecd_csv(
    path: str | Path,
    *,
    series_id: str | None = None,
    date_col: str = "TIME",
    value_col: str = "Value",
    series_col: str = "SUBJECT",
) -> DataFrame:
    """Load OECD CSV exports into standard schema."""
    frame = _load_csv_to_frame(path)
    if date_col in frame.columns and value_col in frame.columns:
        return _standardize_simple_csv(
            frame=frame,
            date_col=date_col,
            value_col=value_col,
            series_id=series_id,
            series_col=series_col if series_id is None else None,
        )

    raise DataLoadFailure("Unsupported OECD CSV format; provide TIME and Value columns.")


def load_world_bank_csv(
    path: str | Path,
    *,
    series_id: str | None = None,
    date_cols: Sequence[str] | None = None,
) -> DataFrame:
    """Load World Bank CSV exports into standard schema.

    Supports classic wide format where years are columns.
    """
    frame = _load_csv_to_frame(path)

    if "Country Name" not in frame.columns:
        raise DataLoadFailure("Expected 'Country Name' in World Bank CSV.")

    if date_cols is None:
        date_cols = tuple(c for c in frame.columns if str(c) in _SUPPORTED_YEAR_COLUMNS)

    if not date_cols:
        if "date" in frame.columns and "value" in frame.columns:
            return _standardize_simple_csv(
                frame=frame,
                date_col="date",
                value_col="value",
                series_id=series_id,
                series_col="series_id" if series_id is None and "series_id" in frame.columns else None,
            )
        raise DataLoadFailure("No year columns found for World Bank wide format.")

    inferred_series = (
        str(frame["Series Code"].iloc[0])
        if "Series Code" in frame.columns
        else (
            str(frame["Series Name"].iloc[0])
            if "Series Name" in frame.columns
            else None
        )
    )
    if inferred_series is None:
        if series_id is None:
            raise DataLoadFailure("Provide `series_id` for World Bank export.")
        inferred_series = str(series_id)

    long_frame = frame.melt(
        id_vars=[
            c
            for c in frame.columns
            if c not in date_cols and c in {"Series Code", "Series Name", "Country Name", "Country Code"}
        ],
        value_vars=list(date_cols),
        var_name="date",
        value_name="value",
    )
    if long_frame.empty:
        raise DataLoadFailure("World Bank CSV yielded no rows after reshaping.")

    long_frame["value"] = pd.to_numeric(long_frame["value"].replace("..", pd.NA), errors="coerce")
    long_frame["series_id"] = (
        long_frame["Series Code"]
        if "Series Code" in long_frame.columns
        else long_frame["Series Name"]
        if "Series Name" in long_frame.columns
        else inferred_series
    )

    long_frame["series_id"] = long_frame["series_id"].astype(str).str.cat(
        long_frame["Country Name"].astype(str).str.strip(),
        sep="|",
    )
    long_frame["date"] = long_frame["date"].astype(str) + "-01-01"

    return _to_standard_schema(
        frame=long_frame,
        date_series=long_frame["date"],
        value_series=long_frame["value"],
        series_id_series=long_frame["series_id"],
    )


def load_yfinance_series(
    symbols: str | Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    value_col: str = "Close",
) -> DataFrame:
    """Load market proxies from yfinance (optional dependency).

    Raises a descriptive error when yfinance is unavailable.
    """
    try:
        import yfinance  # type: ignore
    except ModuleNotFoundError as exc:
        raise DataLoadFailure(
            "yfinance is optional; install it with `pip install yfinance`."
        ) from exc

    symbols_list = [symbols] if isinstance(symbols, str) else list(symbols)
    if len(symbols_list) == 0:
        raise DataLoadFailure("At least one symbol is required.")

    frame = yfinance.download(
        tickers=" ".join(symbols_list),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if frame.empty:
        raise DataLoadFailure("No data returned by yfinance for the requested symbols.")

    date_index = frame.index.to_series()

    if len(symbols_list) == 1:
        if value_col not in frame.columns:
            raise DataLoadFailure(f"Column '{value_col}' not available in yfinance output.")
        return _to_standard_schema(
            frame=frame,
            date_series=date_index,
            value_series=frame[value_col],
            series_id_series=pd.Series([symbols_list[0]] * len(frame)),
        )

    if not isinstance(frame.columns, pd.MultiIndex):
        raise DataLoadFailure("Unexpected yfinance response format for multi-symbol request.")

    outputs = []
    for symbol in symbols_list:
        matches = [col for col in frame.columns if symbol in col and value_col in col]
        if not matches:
            raise DataLoadFailure(f"Symbol '{symbol}' missing '{value_col}' in result.")
        symbol_values = frame[matches[0]]
        outputs.append(
            _to_standard_schema(
                frame=frame,
                date_series=date_index,
                value_series=symbol_values,
                series_id_series=pd.Series([symbol] * len(frame)),
            )
        )

    return pd.concat(outputs, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True)

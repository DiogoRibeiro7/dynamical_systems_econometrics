"""Data-loading contracts for free macro-financial datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.errors import ParserError
import yaml

from dynsys_econometrics.contracts import TimeSeriesFrame


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
    try:
        date = pd.to_datetime(date_series, errors="raise")
        value = pd.to_numeric(value_series, errors="raise")
    except (TypeError, ValueError, ParserError) as exc:
        raise DataLoadFailure("Date or value column could not be parsed into the standard schema.") from exc
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
            "date": date.to_numpy(),
            "series_id": series_id_series.astype(str).to_numpy(),
            "value": value.to_numpy(),
        }
    )
    return cast(DataFrame, standardized.sort_values(["series_id", "date"]).reset_index(drop=True))


def _load_csv_to_frame(path: str | Path, **read_kwargs: Any) -> DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    frame = cast(DataFrame, pd.read_csv(csv_path, **read_kwargs))
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
    cache_path: str | Path | None = None,
    refresh: bool = False,
    api_key: str | None = None,
    api_start: str | None = None,
    api_end: str | None = None,
) -> DataFrame:
    """Load a FRED series from local CSV or FRED API (optional)."""
    if csv_path is not None:
        return load_fred_csv(csv_path, series_id=series_id)
    if cache_path is not None and Path(cache_path).exists() and not refresh:
        return load_time_series_csv(cache_path, series_col="series_id")
    if api_key is None:
        raise DataLoadFailure(
            "`api_key` is required when `csv_path` is not provided and no cached panel is available."
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

    standardized = _to_standard_schema(
        frame=frame,
        date_series=frame["date"],
        value_series=frame["value"].replace(".", pd.NA),
        series_id_series=pd.Series([series_id] * len(frame)),
    )
    if cache_path is not None:
        write_processed_panel(standardized, cache_path)
    return standardized


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

    inferred_series: str | None
    if series_id is not None:
        inferred_series = str(series_id)
    else:
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
            raise DataLoadFailure("Provide `series_id` for World Bank export.")

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
    if series_id is not None:
        long_frame["series_id"] = inferred_series
    else:
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
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> DataFrame:
    """Load market proxies from yfinance (optional dependency).

    Raises a descriptive error when yfinance is unavailable.
    """
    if cache_path is not None and Path(cache_path).exists() and not refresh:
        return load_time_series_csv(cache_path, series_col="series_id")
    try:
        import yfinance
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
        standardized = _to_standard_schema(
            frame=frame,
            date_series=date_index,
            value_series=frame[value_col],
            series_id_series=pd.Series([symbols_list[0]] * len(frame)),
        )
        if cache_path is not None:
            write_processed_panel(standardized, cache_path)
        return standardized

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

    combined = pd.concat(outputs, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True)
    if cache_path is not None:
        write_processed_panel(combined, cache_path)
    return cast(DataFrame, combined)


def load_timeseries_csv(path: str | Path, *, allow_missing: bool = False) -> TimeSeriesFrame:
    """Load a canonical long-format CSV into a validated time-series contract."""
    frame = _load_csv_to_frame(path)
    return TimeSeriesFrame(frame=frame, allow_missing=allow_missing)


def load_panel_from_directory(path: str | Path, pattern: str = "*.csv") -> TimeSeriesFrame:
    """Load and concatenate multiple long-format CSV files from a directory."""
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(directory.glob(pattern))
    if not files:
        raise DataLoadFailure(f"No files matched pattern '{pattern}' in {directory}.")
    frames = [load_timeseries_csv(file_path).to_frame() for file_path in files]
    combined = pd.concat(frames, ignore_index=True)
    return TimeSeriesFrame(frame=combined)


def _default_series_col(path: str | Path, *, series_id: str | None, series_col: str | None) -> str | None:
    if series_id is not None or series_col is not None:
        return series_col
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    header = pd.read_csv(csv_path, nrows=0)
    return "series_id" if "series_id" in header.columns else None


def load_empirical_panel(spec: Mapping[str, Any], *, force_refresh: bool = False) -> DataFrame:
    """Load an empirical panel from a structured loader configuration."""
    loader_config = dict(spec)
    catalog_entries = loader_config.get("catalogs", loader_config.get("catalog_paths"))
    if isinstance(catalog_entries, list):
        if len(catalog_entries) == 0:
            raise DataLoadFailure("Empirical multi-catalog loader requires a non-empty `catalogs` list.")
        frames: list[DataFrame] = []
        for entry in catalog_entries:
            merged_entry = dict(loader_config)
            merged_entry.pop("catalogs", None)
            merged_entry.pop("catalog_paths", None)
            if isinstance(entry, (str, Path)):
                merged_entry["catalog_path"] = entry
            elif isinstance(entry, Mapping):
                mapped_entry = dict(entry)
                if "catalog_path" not in mapped_entry and "catalog" not in mapped_entry and "path" in mapped_entry:
                    mapped_entry["catalog_path"] = mapped_entry.pop("path")
                merged_entry.update(mapped_entry)
            else:
                raise DataLoadFailure("Each empirical catalog entry must be a path or a mapping.")
            frames.append(load_empirical_panel(merged_entry, force_refresh=force_refresh))
        return cast(
            DataFrame,
            pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True),
        )
    series_entries = loader_config.get("series")
    if isinstance(series_entries, list):
        if len(series_entries) == 0:
            raise DataLoadFailure("Empirical multi-series loader requires a non-empty `series` list.")
        frames: list[DataFrame] = []
        for entry in series_entries:
            if not isinstance(entry, Mapping):
                raise DataLoadFailure("Each empirical series entry must be a mapping.")
            merged_entry = dict(loader_config)
            merged_entry.pop("series", None)
            merged_entry.update(dict(entry))
            frames.append(load_empirical_panel(merged_entry, force_refresh=force_refresh))
        return cast(
            DataFrame,
            pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True),
        )
    transformation = cast(str | None, loader_config.get("transformation"))
    catalog_path = loader_config.get("catalog", loader_config.get("catalog_path"))
    if isinstance(catalog_path, (str, Path)):
        materialized_output = loader_config.get("output_path")
        summary = materialize_catalog(catalog_path, materialized_output, force_refresh=force_refresh)
        output_path = summary.get("output_path")
        if isinstance(output_path, str):
            return _apply_catalog_transformation(load_time_series_csv(output_path, series_col="series_id"), transformation)
        panel = summary.get("panel")
        if isinstance(panel, pd.DataFrame):
            return _apply_catalog_transformation(cast(DataFrame, panel), transformation)
        raise DataLoadFailure("Catalog materialization did not return an output path.")
    loader = str(loader_config.get("loader", "timeseries_csv")).strip().lower()
    path = loader_config.get("path", loader_config.get("input_path"))

    if loader in {"timeseries_csv", "csv", "canonical_csv", "local_csv"}:
        if not isinstance(path, (str, Path)):
            raise DataLoadFailure("Empirical CSV loader requires `path` or `input_path`.")
        series_id_value = loader_config.get("series_id")
        series_col_value = loader_config.get("series_col")
        series_id = str(series_id_value) if series_id_value is not None else None
        series_col = str(series_col_value) if series_col_value is not None else None
        return _apply_catalog_transformation(
            load_time_series_csv(
            path,
            series_id=series_id,
            date_col=str(loader_config.get("date_col", "date")),
            value_col=str(loader_config.get("value_col", "value")),
            series_col=_default_series_col(path, series_id=series_id, series_col=series_col),
            ),
            transformation,
        )

    if loader == "fred_csv":
        if not isinstance(path, (str, Path)):
            raise DataLoadFailure("FRED CSV loader requires `path`.")
        series_id = loader_config.get("series_id")
        if series_id is None:
            raise DataLoadFailure("FRED CSV loader requires `series_id`.")
        return _apply_catalog_transformation(
            load_fred_csv(
                path,
                series_id=str(series_id),
                date_col=str(loader_config.get("date_col", "DATE")),
                value_col=str(loader_config.get("value_col", "VALUE")),
            ),
            transformation,
        )

    if loader == "fred":
        series_id = loader_config.get("series_id")
        if series_id is None:
            raise DataLoadFailure("FRED loader requires `series_id`.")
        csv_path = loader_config.get("csv_path", path)
        return _apply_catalog_transformation(
            load_fred(
                str(series_id),
                csv_path=csv_path,
                cache_path=cast(str | Path | None, loader_config.get("cache_path")),
                refresh=bool(loader_config.get("refresh", False) or force_refresh),
                api_key=cast(str | None, loader_config.get("api_key")),
                api_start=cast(str | None, loader_config.get("api_start")),
                api_end=cast(str | None, loader_config.get("api_end")),
            ),
            transformation,
        )

    if loader in {"ecb_sdw_csv", "ecb"}:
        if not isinstance(path, (str, Path)):
            raise DataLoadFailure("ECB SDW CSV loader requires `path`.")
        series_id_value = loader_config.get("series_id")
        return _apply_catalog_transformation(
            load_ecb_sdw_csv(
                path,
                series_id=str(series_id_value) if series_id_value is not None else None,
                date_col=str(loader_config.get("date_col", "TIME_PERIOD")),
                value_col=str(loader_config.get("value_col", "OBS_VALUE")),
                series_col=str(loader_config.get("series_col", "SERIES")),
            ),
            transformation,
        )

    if loader in {"oecd_csv", "oecd"}:
        if not isinstance(path, (str, Path)):
            raise DataLoadFailure("OECD CSV loader requires `path`.")
        series_id_value = loader_config.get("series_id")
        return _apply_catalog_transformation(
            load_oecd_csv(
                path,
                series_id=str(series_id_value) if series_id_value is not None else None,
                date_col=str(loader_config.get("date_col", "TIME")),
                value_col=str(loader_config.get("value_col", "Value")),
                series_col=str(loader_config.get("series_col", "SUBJECT")),
            ),
            transformation,
        )

    if loader in {"world_bank_csv", "world_bank"}:
        if not isinstance(path, (str, Path)):
            raise DataLoadFailure("World Bank CSV loader requires `path`.")
        series_id_value = loader_config.get("series_id")
        date_cols_value = loader_config.get("date_cols")
        if date_cols_value is None:
            date_cols: Sequence[str] | None = None
        elif isinstance(date_cols_value, str):
            date_cols = (date_cols_value,)
        else:
            date_cols = tuple(str(column) for column in date_cols_value)
        return _apply_catalog_transformation(
            load_world_bank_csv(
                path,
                series_id=str(series_id_value) if series_id_value is not None else None,
                date_cols=date_cols,
            ),
            transformation,
        )

    if loader in {"panel_directory", "directory"}:
        directory = loader_config.get("directory", path)
        if not isinstance(directory, (str, Path)):
            raise DataLoadFailure("Directory loader requires `directory` or `path`.")
        pattern = str(loader_config.get("pattern", "*.csv"))
        return _apply_catalog_transformation(load_panel_from_directory(directory, pattern=pattern).to_frame(), transformation)

    if loader == "yfinance":
        symbols_value = loader_config.get("symbols")
        if symbols_value is None:
            raise DataLoadFailure("yfinance loader requires `symbols`.")
        if isinstance(symbols_value, str):
            symbols: str | list[str] = symbols_value
        else:
            symbols = [str(symbol) for symbol in symbols_value]
            if len(symbols) == 0:
                raise DataLoadFailure("yfinance loader requires at least one symbol.")
        return _apply_catalog_transformation(
            load_yfinance_series(
                symbols,
                start=cast(str | None, loader_config.get("start")),
                end=cast(str | None, loader_config.get("end")),
                value_col=str(loader_config.get("value_col", "Close")),
                cache_path=cast(str | Path | None, loader_config.get("cache_path")),
                refresh=bool(loader_config.get("refresh", False) or force_refresh),
            ),
            transformation,
        )

    raise DataLoadFailure(
        "Unsupported empirical loader. Use one of: timeseries_csv, fred_csv, fred, ecb_sdw_csv, "
        "oecd_csv, world_bank_csv, panel_directory, yfinance."
    )


def _read_catalog_series(path: str | Path) -> list[dict[str, Any]]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataLoadFailure("Catalog must be a mapping.")
    series = payload.get("series", [])
    if not isinstance(series, list):
        raise DataLoadFailure("Catalog field 'series' must be a list.")
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(series):
        if not isinstance(entry, dict):
            raise DataLoadFailure(f"Entry {idx} is not a mapping.")
        normalized.append(dict(entry))
    return normalized


def _resolve_catalog_path_value(value: Any, *, base_dir: Path) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    path_value = Path(value)
    if path_value.is_absolute():
        return str(path_value)
    candidate = (base_dir / path_value).resolve()
    if candidate.exists():
        return str(candidate)
    workspace_candidate = Path(path_value).resolve()
    return str(workspace_candidate)


def _catalog_entry_to_loader_spec(entry: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    source = str(entry.get("source", "")).strip().lower()
    spec = dict(entry)
    for key in ("path", "input_path", "csv_path", "directory", "cache_path"):
        if key in spec:
            spec[key] = _resolve_catalog_path_value(spec[key], base_dir=base_dir)
    if source in {"local_csv", "canonical_csv", "csv", "timeseries_csv"}:
        spec["loader"] = "timeseries_csv"
    elif source in {"fred_csv", "ecb_sdw_csv", "oecd_csv", "world_bank_csv", "panel_directory", "directory", "yfinance"}:
        spec["loader"] = source
    elif source == "ecb":
        spec["loader"] = "ecb_sdw_csv"
    elif source == "oecd":
        spec["loader"] = "oecd_csv"
    elif source == "world_bank":
        spec["loader"] = "world_bank_csv"
    elif source == "fred":
        spec["loader"] = "fred"
    else:
        spec["loader"] = source
    return spec


def _apply_catalog_transformation(frame: DataFrame, transformation: str | None) -> DataFrame:
    transform = "level" if transformation is None else str(transformation).strip().lower()
    if transform in {"", "level"}:
        return cast(DataFrame, frame.sort_values(["series_id", "date"]).reset_index(drop=True))

    result = frame.copy()
    if transform == "diff":
        result["value"] = result.groupby("series_id")["value"].diff()
    elif transform == "pct_change":
        result["value"] = result.groupby("series_id")["value"].pct_change()
    elif transform == "log_return":
        if (result["value"] <= 0).any():
            raise DataLoadFailure("log_return transformation requires strictly positive values.")
        result["value"] = result.groupby("series_id")["value"].transform(lambda s: np.log(s).diff())
    else:
        raise DataLoadFailure(f"Unsupported catalog transformation: {transform}")

    result = result.dropna(subset=["value"]).reset_index(drop=True)
    return cast(DataFrame, result.sort_values(["series_id", "date"]).reset_index(drop=True))


def materialize_catalog(
    path: str | Path,
    output_path: str | Path | None = None,
    *,
    force_refresh: bool = False,
) -> dict[str, object]:
    """Load a validated data catalog into a canonical processed panel."""
    catalog_path = Path(path)
    summary = validate_catalog(catalog_path)
    if not bool(summary["valid"]):
        errors = cast(list[str], summary["errors"])
        raise DataLoadFailure("Catalog is invalid: " + "; ".join(errors))

    frames: list[DataFrame] = []
    base_dir = catalog_path.resolve().parent
    for entry in _read_catalog_series(catalog_path):
        spec = _catalog_entry_to_loader_spec(entry, base_dir=base_dir)
        spec.pop("transformation", None)
        frame = load_empirical_panel(spec, force_refresh=force_refresh)
        frames.append(_apply_catalog_transformation(frame, cast(str | None, entry.get("transformation"))))

    if not frames:
        raise DataLoadFailure("Catalog does not contain any series entries.")

    panel = cast(DataFrame, pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True))
    if output_path is not None:
        write_processed_panel(panel, output_path)

    return {
        "n_series": int(panel["series_id"].nunique()),
        "n_rows": int(len(panel)),
        "output_path": None if output_path is None else str(Path(output_path)),
        "panel": panel,
        "sources": summary["sources"],
        "valid": True,
        "errors": [],
    }


def refresh_empirical_cache(spec: Mapping[str, Any]) -> dict[str, object]:
    """Refresh remote-backed empirical caches without running a full experiment."""
    loader_config = dict(spec)
    catalog_path = loader_config.get("catalog", loader_config.get("catalog_path"))
    if isinstance(catalog_path, (str, Path)):
        summary = materialize_catalog(
            catalog_path,
            loader_config.get("output_path"),
            force_refresh=True,
        )
        targets = []
        if summary.get("output_path") is not None:
            targets.append(str(summary["output_path"]))
        return {
            "n_series": int(summary["n_series"]),
            "n_rows": int(summary["n_rows"]),
            "targets": targets,
        }

    panel = load_empirical_panel(loader_config, force_refresh=True)
    targets = _collect_refresh_targets(loader_config)
    return {
        "n_series": int(panel["series_id"].nunique()),
        "n_rows": int(len(panel)),
        "targets": targets,
    }


def _collect_refresh_targets(spec: Mapping[str, Any]) -> list[str]:
    targets: list[str] = []

    cache_path = spec.get("cache_path")
    if isinstance(cache_path, (str, Path)):
        targets.append(str(Path(cache_path)))

    output_path = spec.get("output_path")
    if isinstance(output_path, (str, Path)):
        targets.append(str(Path(output_path)))

    catalog_entries = spec.get("catalogs", spec.get("catalog_paths"))
    if isinstance(catalog_entries, list):
        for entry in catalog_entries:
            if isinstance(entry, Mapping):
                targets.extend(_collect_refresh_targets(entry))

    series_entries = spec.get("series")
    if isinstance(series_entries, list):
        for entry in series_entries:
            if isinstance(entry, Mapping):
                targets.extend(_collect_refresh_targets(entry))

    deduped: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if target not in seen:
            deduped.append(target)
            seen.add(target)
    return deduped


def write_processed_panel(frame: TimeSeriesFrame | DataFrame, path: str | Path) -> None:
    """Write a validated processed panel to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_frame() if isinstance(frame, TimeSeriesFrame) else TimeSeriesFrame(frame=frame).to_frame()
    data.to_csv(output_path, index=False)


def validate_catalog(path: str | Path) -> dict[str, object]:
    """Validate a YAML data catalog and return a compact summary."""
    catalog_path = Path(path)
    series = _read_catalog_series(catalog_path)
    base_dir = catalog_path.resolve().parent
    required = {"series_id", "description", "source"}
    errors: list[str] = []
    sources: set[str] = set()
    for idx, entry in enumerate(series):
        missing = required.difference(entry)
        if missing:
            errors.append(f"Entry {idx} missing fields: {sorted(missing)}")
            continue
        source = str(entry["source"]).strip().lower()
        sources.add(source)
        if source in {"local_csv", "canonical_csv", "csv", "timeseries_csv", "fred_csv", "ecb_sdw_csv", "oecd_csv", "world_bank_csv"}:
            if "path" not in entry:
                errors.append(f"Entry {idx} source '{source}' requires field 'path'.")
            else:
                resolved = Path(_resolve_catalog_path_value(entry["path"], base_dir=base_dir))
                if not resolved.exists():
                    errors.append(f"Entry {idx} path does not exist: {entry['path']}")
        elif source in {"panel_directory", "directory"}:
            if "directory" not in entry and "path" not in entry:
                errors.append(f"Entry {idx} source '{source}' requires field 'directory' or 'path'.")
            else:
                directory_value = entry.get("directory", entry.get("path"))
                resolved = Path(_resolve_catalog_path_value(directory_value, base_dir=base_dir))
                if not resolved.exists():
                    errors.append(f"Entry {idx} directory does not exist: {directory_value}")
        elif source == "fred":
            has_api_key = "api_key" in entry
            has_csv_path = "csv_path" in entry
            has_path = "path" in entry
            has_cache_path = "cache_path" in entry
            if not any((has_api_key, has_csv_path, has_path, has_cache_path)):
                errors.append(f"Entry {idx} source 'fred' requires one of: 'api_key', 'csv_path', 'path', or 'cache_path'.")
            if has_csv_path and has_path:
                errors.append(f"Entry {idx} source 'fred' cannot define both 'csv_path' and 'path'.")
            if has_api_key and (has_csv_path or has_path):
                errors.append(f"Entry {idx} source 'fred' cannot combine 'api_key' with local CSV path fields.")
            if bool(entry.get("refresh", False)) and not has_api_key:
                errors.append(f"Entry {idx} source 'fred' requires 'api_key' when 'refresh: true' is requested.")
            if has_csv_path or has_path or has_cache_path:
                csv_value = entry.get("csv_path", entry.get("path", entry.get("cache_path")))
                resolved = Path(_resolve_catalog_path_value(csv_value, base_dir=base_dir))
                if not has_api_key and not resolved.exists():
                    errors.append(f"Entry {idx} csv path does not exist: {csv_value}")
        elif source == "yfinance":
            if "symbols" not in entry:
                errors.append(f"Entry {idx} source 'yfinance' requires field 'symbols'.")
            if bool(entry.get("refresh", False)) and "cache_path" not in entry:
                errors.append(f"Entry {idx} source 'yfinance' requires 'cache_path' when 'refresh: true' is requested.")
            elif "cache_path" in entry:
                cache_value = entry["cache_path"]
                resolved = Path(_resolve_catalog_path_value(cache_value, base_dir=base_dir))
                if resolved.exists() and not resolved.is_file():
                    errors.append(f"Entry {idx} cache path is not a file path: {cache_value}")
        elif source not in {"ecb", "oecd", "world_bank"}:
            errors.append(f"Entry {idx} has unsupported source: {source}")
    return {
        "n_series": len(series),
        "sources": sorted(sources),
        "valid": len(errors) == 0,
        "errors": errors,
    }

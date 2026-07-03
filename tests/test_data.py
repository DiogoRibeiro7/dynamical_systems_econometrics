from __future__ import annotations

import json
import sys
import types

import pandas as pd

from dynsys_econometrics.data import (
    DataLoadFailure,
    load_ecb_sdw_csv,
    load_empirical_panel,
    load_fred,
    load_fred_csv,
    load_oecd_csv,
    load_panel_from_directory,
    load_time_series_csv,
    load_yfinance_series,
    load_world_bank_csv,
    materialize_catalog,
    validate_catalog,
)


def test_load_time_series_csv_standard_schema(tmp_path) -> None:
    csv_path = tmp_path / "timeseries.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "value": [1.0, 2.0],
        }
    ).to_csv(csv_path, index=False)

    frame = load_time_series_csv(csv_path, series_id="custom")
    assert frame.columns.tolist() == ["date", "series_id", "value"]
    assert frame["series_id"].iloc[0] == "custom"
    assert frame["value"].tolist() == [1.0, 2.0]


def test_load_fred_csv(tmp_path) -> None:
    csv_path = tmp_path / "fred.csv"
    pd.DataFrame({"DATE": ["2020-01-01", "2020-01-02"], "VALUE": [1.2, 2.4]}).to_csv(
        csv_path,
        index=False,
    )

    frame = load_fred_csv(csv_path, series_id="UNRATE")
    assert frame["series_id"].nunique() == 1
    assert frame["series_id"].iloc[0] == "UNRATE"


def test_load_ecb_sdw_csv(tmp_path) -> None:
    csv_path = tmp_path / "ecb.csv"
    pd.DataFrame(
        {
            "TIME_PERIOD": ["2020-01", "2020-02"],
            "SERIES": ["stress", "stress"],
            "OBS_VALUE": [5.0, 6.0],
        }
    ).to_csv(csv_path, index=False)

    frame = load_ecb_sdw_csv(csv_path)
    assert frame["series_id"].iloc[0] == "stress"
    assert frame["value"].tolist() == [5.0, 6.0]


def test_load_oecd_csv(tmp_path) -> None:
    csv_path = tmp_path / "oecd.csv"
    pd.DataFrame(
        {
            "TIME": ["2020-01", "2020-02"],
            "SUBJECT": ["C1", "C2"],
            "Value": [0.1, 0.2],
        }
    ).to_csv(csv_path, index=False)

    frame = load_oecd_csv(csv_path, series_id="oecd-stub")
    assert frame.shape[0] == 2
    assert frame["series_id"].tolist() == ["oecd-stub", "oecd-stub"]


def test_load_world_bank_csv_wide_years(tmp_path) -> None:
    csv_path = tmp_path / "wb.csv"
    pd.DataFrame(
        {
            "Country Name": ["United States", "Euro area"],
            "Series Name": ["GDP (current US$)", "GDP (current US$)"],
            "2020": [10000.0, 20000.0],
            "2021": [11000.0, 21000.0],
        }
    ).to_csv(csv_path, index=False)

    frame = load_world_bank_csv(csv_path, series_id="NY.GDP.MKTP.CD")
    assert set(frame.columns) == {"date", "series_id", "value"}
    assert frame["series_id"].nunique() == 2
    assert frame["series_id"].iloc[0].startswith("NY.GDP.MKTP.CD|")


def test_data_loader_rejects_missing_dates(tmp_path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame({"date": ["bad"], "value": [1.0]}).to_csv(csv_path, index=False)

    try:
        load_time_series_csv(csv_path, series_id="x")
        assert False
    except DataLoadFailure:
        assert True


def test_load_panel_from_directory_combines_files(tmp_path) -> None:
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "series_id": ["a", "a"],
            "value": [1.0, 2.0],
        }
    ).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "series_id": ["b", "b"],
            "value": [3.0, 4.0],
        }
    ).to_csv(tmp_path / "b.csv", index=False)

    panel = load_panel_from_directory(tmp_path).to_frame()
    assert panel.shape == (4, 3)
    assert panel["series_id"].tolist() == ["a", "a", "b", "b"]


def test_load_empirical_panel_supports_fred_csv_loader(tmp_path) -> None:
    csv_path = tmp_path / "fred.csv"
    pd.DataFrame({"DATE": ["2020-01-01", "2020-02-01"], "VALUE": [1.2, 2.4]}).to_csv(csv_path, index=False)

    frame = load_empirical_panel({"loader": "fred_csv", "path": str(csv_path), "series_id": "UNRATE"})
    assert frame["series_id"].tolist() == ["UNRATE", "UNRATE"]


def test_load_empirical_panel_supports_directory_loader(tmp_path) -> None:
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "series_id": ["a", "a"],
            "value": [1.0, 2.0],
        }
    ).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "series_id": ["b", "b"],
            "value": [3.0, 4.0],
        }
    ).to_csv(tmp_path / "b.csv", index=False)

    frame = load_empirical_panel({"loader": "panel_directory", "directory": str(tmp_path)})
    assert frame.shape == (4, 3)
    assert frame["series_id"].tolist() == ["a", "a", "b", "b"]


def test_load_empirical_panel_applies_direct_transformation(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "value": [100.0, 110.0, 121.0],
        }
    ).to_csv(csv_path, index=False)

    frame = load_empirical_panel(
        {
            "loader": "local_csv",
            "path": str(csv_path),
            "series_id": "equity",
            "transformation": "log_return",
        }
    )
    assert frame["series_id"].tolist() == ["equity", "equity"]
    assert len(frame) == 2


def test_load_empirical_panel_supports_per_series_overrides(tmp_path) -> None:
    prices_path = tmp_path / "prices.csv"
    inflation_path = tmp_path / "inflation.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "value": [100.0, 110.0, 121.0],
        }
    ).to_csv(prices_path, index=False)
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "value": [1.8, 2.0, 2.1],
        }
    ).to_csv(inflation_path, index=False)

    frame = load_empirical_panel(
        {
            "series": [
                {
                    "loader": "local_csv",
                    "path": str(prices_path),
                    "series_id": "equity",
                    "transformation": "log_return",
                },
                {
                    "loader": "local_csv",
                    "path": str(inflation_path),
                    "series_id": "inflation",
                    "transformation": "level",
                },
            ]
        }
    )
    assert frame["series_id"].tolist() == ["equity", "equity", "inflation", "inflation", "inflation"]


def test_load_empirical_panel_supports_multiple_catalogs(tmp_path) -> None:
    first_raw = tmp_path / "first.csv"
    second_raw = tmp_path / "second.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "value": [100.0, 101.0],
        }
    ).to_csv(first_raw, index=False)
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "value": [2.0, 2.1],
        }
    ).to_csv(second_raw, index=False)
    first_catalog = tmp_path / "catalog_one.yaml"
    second_catalog = tmp_path / "catalog_two.yaml"
    first_catalog.write_text(
        "series:\n"
        "  - series_id: equity\n"
        "    description: Equity levels\n"
        "    source: local_csv\n"
        f"    path: {first_raw.as_posix()}\n",
        encoding="utf-8",
    )
    second_catalog.write_text(
        "series:\n"
        "  - series_id: inflation\n"
        "    description: Inflation levels\n"
        "    source: local_csv\n"
        f"    path: {second_raw.as_posix()}\n",
        encoding="utf-8",
    )

    frame = load_empirical_panel({"catalog_paths": [str(first_catalog), str(second_catalog)]})
    assert frame["series_id"].tolist() == ["equity", "equity", "inflation", "inflation"]


def test_load_fred_uses_cached_standard_panel(tmp_path) -> None:
    cache_path = tmp_path / "fred_cache.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "series_id": ["UNRATE", "UNRATE"],
            "value": [3.5, 3.6],
        }
    ).to_csv(cache_path, index=False)

    frame = load_fred("UNRATE", cache_path=cache_path)
    assert frame["series_id"].tolist() == ["UNRATE", "UNRATE"]


def test_load_yfinance_uses_cached_standard_panel(tmp_path) -> None:
    cache_path = tmp_path / "market_cache.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "series_id": ["SPY", "SPY"],
            "value": [320.0, 330.0],
        }
    ).to_csv(cache_path, index=False)

    frame = load_yfinance_series("SPY", cache_path=cache_path)
    assert frame["series_id"].tolist() == ["SPY", "SPY"]


def test_validate_catalog_accepts_example_mapping(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: example_series\n"
        "    description: Example series\n"
        "    source: local_csv\n"
        "    path: data/raw/example.csv\n",
        encoding="utf-8",
    )
    summary = validate_catalog(catalog_path)
    assert summary["valid"] is True
    assert summary["n_series"] == 1


def test_validate_catalog_reports_missing_fields(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: broken_series\n",
        encoding="utf-8",
    )
    summary = validate_catalog(catalog_path)
    assert summary["valid"] is False
    assert len(summary["errors"]) == 1


def test_validate_catalog_requires_source_specific_fields(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: broken_series\n"
        "    description: Missing local path\n"
        "    source: local_csv\n",
        encoding="utf-8",
    )
    summary = validate_catalog(catalog_path)
    assert summary["valid"] is False
    assert "requires field 'path'" in summary["errors"][0]


def test_validate_catalog_rejects_missing_local_path(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: missing_series\n"
        "    description: Missing file\n"
        "    source: local_csv\n"
        "    path: data/raw/not_here.csv\n",
        encoding="utf-8",
    )
    summary = validate_catalog(catalog_path)
    assert summary["valid"] is False
    assert "path does not exist" in summary["errors"][0]


def test_validate_catalog_rejects_conflicting_fred_fields(tmp_path) -> None:
    csv_path = tmp_path / "fred.csv"
    csv_path.write_text("DATE,VALUE\n2020-01-01,3.5\n", encoding="utf-8")
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: UNRATE\n"
        "    description: Ambiguous fred entry\n"
        "    source: fred\n"
        "    api_key: demo\n"
        f"    path: {csv_path.as_posix()}\n",
        encoding="utf-8",
    )
    summary = validate_catalog(catalog_path)
    assert summary["valid"] is False
    assert "cannot combine 'api_key' with local CSV path fields" in summary["errors"][0]


def test_validate_catalog_rejects_refresh_without_remote_requirements(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: SPY\n"
        "    description: Refresh without cache path\n"
        "    source: yfinance\n"
        "    symbols: SPY\n"
        "    refresh: true\n",
        encoding="utf-8",
    )
    summary = validate_catalog(catalog_path)
    assert summary["valid"] is False
    assert "requires 'cache_path' when 'refresh: true' is requested" in summary["errors"][0]


def test_materialize_catalog_writes_transformed_panel(tmp_path) -> None:
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "value": [100.0, 110.0, 121.0],
        }
    ).to_csv(raw_path, index=False)
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: equity\n"
        "    description: Equity prices\n"
        "    source: local_csv\n"
        f"    path: {raw_path.as_posix()}\n"
        "    transformation: log_return\n",
        encoding="utf-8",
    )
    summary = materialize_catalog(catalog_path, output_path)
    panel = pd.read_csv(output_path)
    assert summary["n_rows"] == 2
    assert panel["series_id"].tolist() == ["equity", "equity"]


def test_materialize_catalog_resolves_paths_relative_to_catalog(tmp_path) -> None:
    data_dir = tmp_path / "nested" / "raw"
    data_dir.mkdir(parents=True)
    raw_path = data_dir / "series.csv"
    raw_path.write_text("date,value\n2020-01-01,1.0\n2020-02-01,2.0\n", encoding="utf-8")
    catalog_dir = tmp_path / "nested" / "configs"
    catalog_dir.mkdir(parents=True)
    catalog_path = catalog_dir / "catalog.yaml"
    output_path = tmp_path / "panel.csv"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: relative_series\n"
        "    description: Relative source\n"
        "    source: local_csv\n"
        "    path: ../raw/series.csv\n",
        encoding="utf-8",
    )
    summary = materialize_catalog(catalog_path, output_path)
    panel = pd.read_csv(output_path)
    assert summary["n_rows"] == 2
    assert panel["series_id"].tolist() == ["relative_series", "relative_series"]


def test_materialize_catalog_writes_fred_cache_path(tmp_path, monkeypatch) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    output_path = tmp_path / "panel.csv"
    cache_path = tmp_path / "cache" / "fred_unrate.csv"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: UNRATE\n"
        "    description: Cached unemployment example\n"
        "    source: fred\n"
        "    api_key: demo\n"
        f"    cache_path: {cache_path.as_posix()}\n",
        encoding="utf-8",
    )

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            payload = {
                "observations": [
                    {"date": "2020-01-01", "value": "3.5"},
                    {"date": "2020-02-01", "value": "3.6"},
                ]
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr("dynsys_econometrics.data.urlopen", lambda *args, **kwargs: _FakeResponse())

    summary = materialize_catalog(catalog_path, output_path)
    assert summary["n_rows"] == 2
    assert cache_path.exists()
    cached = pd.read_csv(cache_path)
    assert cached["series_id"].tolist() == ["UNRATE", "UNRATE"]


def test_materialize_catalog_writes_yfinance_cache_path(tmp_path, monkeypatch) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    output_path = tmp_path / "panel.csv"
    cache_path = tmp_path / "cache" / "spy.csv"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: SPY\n"
        "    description: Cached market example\n"
        "    source: yfinance\n"
        "    symbols: SPY\n"
        f"    cache_path: {cache_path.as_posix()}\n",
        encoding="utf-8",
    )

    fake_module = types.SimpleNamespace()

    def _download(**kwargs):
        return pd.DataFrame(
            {"Close": [320.0, 330.0]},
            index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
        )

    fake_module.download = _download
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)

    summary = materialize_catalog(catalog_path, output_path)
    assert summary["n_rows"] == 2
    assert cache_path.exists()
    cached = pd.read_csv(cache_path)
    assert cached["series_id"].tolist() == ["SPY", "SPY"]

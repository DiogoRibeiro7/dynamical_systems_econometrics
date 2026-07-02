from __future__ import annotations

import pandas as pd

from dynsys_econometrics.data import (
    DataLoadFailure,
    load_panel_from_directory,
    load_ecb_sdw_csv,
    load_fred_csv,
    load_oecd_csv,
    load_time_series_csv,
    validate_catalog,
    load_world_bank_csv,
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

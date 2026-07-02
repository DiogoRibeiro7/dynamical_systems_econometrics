from __future__ import annotations

from pathlib import Path

import pandas as pd

from dynsys_econometrics.data import load_timeseries_csv
from dynsys_econometrics.multivariate import wide_panel
from dynsys_econometrics.return_times import return_times


def _write_fixture(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_simple_series_fixture_loads_as_contract(tmp_path: Path) -> None:
    fixture_path = _write_fixture(
        tmp_path / "simple_series.csv",
        "date,series_id,value\n"
        "2020-01-01,series_a,1.0\n"
        "2020-01-02,series_a,2.0\n"
        "2020-01-03,series_a,3.5\n"
        "2020-01-04,series_a,2.5\n",
    )
    frame = load_timeseries_csv(fixture_path).to_frame()
    assert frame["series_id"].unique().tolist() == ["series_a"]
    assert frame["value"].tolist() == [1.0, 2.0, 3.5, 2.5]


def test_two_series_panel_fixture_pivots_to_wide(tmp_path: Path) -> None:
    fixture_path = _write_fixture(
        tmp_path / "two_series_panel.csv",
        "date,series_id,value\n"
        "2020-01-01,series_a,1.0\n"
        "2020-01-02,series_a,2.0\n"
        "2020-01-03,series_a,4.0\n"
        "2020-01-01,series_b,0.5\n"
        "2020-01-02,series_b,1.5\n"
        "2020-01-03,series_b,3.0\n",
    )
    frame = load_timeseries_csv(fixture_path).to_frame()
    wide = wide_panel(frame)
    assert wide.shape == (3, 2)
    assert wide.columns.tolist() == ["series_a", "series_b"]


def test_handcrafted_events_fixture_produces_expected_return_times(tmp_path: Path) -> None:
    fixture_path = _write_fixture(
        tmp_path / "handcrafted_events.csv",
        "date,series_id,exceedance\n"
        "2020-01-01,series_a,false\n"
        "2020-01-02,series_a,true\n"
        "2020-01-03,series_a,false\n"
        "2020-01-04,series_a,false\n"
        "2020-01-05,series_a,true\n"
        "2020-01-06,series_a,false\n"
        "2020-01-07,series_a,true\n",
    )
    events = pd.read_csv(fixture_path)
    durations = return_times(events)
    assert durations["duration"].tolist() == [3.0, 2.0]

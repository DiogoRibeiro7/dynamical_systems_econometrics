from __future__ import annotations

from pathlib import Path

import pandas as pd

from dynsys_econometrics.data import load_timeseries_csv
from dynsys_econometrics.multivariate import wide_panel
from dynsys_econometrics.return_times import return_times


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def test_simple_series_fixture_loads_as_contract() -> None:
    frame = load_timeseries_csv(_fixtures_dir() / "simple_series.csv").to_frame()
    assert frame["series_id"].unique().tolist() == ["series_a"]
    assert frame["value"].tolist() == [1.0, 2.0, 3.5, 2.5]


def test_two_series_panel_fixture_pivots_to_wide() -> None:
    frame = load_timeseries_csv(_fixtures_dir() / "two_series_panel.csv").to_frame()
    wide = wide_panel(frame)
    assert wide.shape == (3, 2)
    assert wide.columns.tolist() == ["series_a", "series_b"]


def test_handcrafted_events_fixture_produces_expected_return_times() -> None:
    events = pd.read_csv(_fixtures_dir() / "handcrafted_events.csv")
    durations = return_times(events)
    assert durations["duration"].tolist() == [3.0, 2.0]

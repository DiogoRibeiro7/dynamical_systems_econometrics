from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dynsys_econometrics.contracts import EventFrame, ExperimentResult, TimeSeriesFrame


def test_timeseries_frame_validates_required_schema() -> None:
    frame = TimeSeriesFrame(
        pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "series_id": ["a", "a"],
                "value": [1.0, 2.0],
            }
        )
    )
    assert frame.to_frame()["series_id"].tolist() == ["a", "a"]


def test_timeseries_frame_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        TimeSeriesFrame(
            pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-01"],
                    "series_id": ["a", "a"],
                    "value": [1.0, 2.0],
                }
            )
        )


def test_event_frame_rejects_invalid_side() -> None:
    with pytest.raises(ValueError):
        EventFrame(
            pd.DataFrame(
                {
                    "date": ["2020-01-01"],
                    "series_id": ["a"],
                    "value": [1.0],
                    "threshold": [0.5],
                    "exceedance": [True],
                    "side": ["bad"],
                    "run_id": [0],
                }
            )
        )


def test_experiment_result_validates_non_empty_name() -> None:
    with pytest.raises(ValueError):
        ExperimentResult("", {}, {}, {}, {}, [])


def test_experiment_result_accepts_registry() -> None:
    result = ExperimentResult(
        name="demo",
        config={"seed": 1},
        tables={"table": pd.DataFrame({"x": [1]})},
        figures={"fig": Path("figures/test.png")},
        metrics={"n": 1.0},
        warnings=["synthetic only"],
    )
    assert result.name == "demo"

from __future__ import annotations

import pandas as pd

from dynsys_econometrics.econometrics import compare_event_diagnostics


def test_compare_event_diagnostics_merges_on_date_and_series() -> None:
    events = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "series_id": ["a", "a"],
            "exceedance": [False, True],
        }
    )
    baselines = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "series_id": ["a", "a"],
            "rolling_volatility": [0.1, 0.2],
        }
    )
    merged = compare_event_diagnostics(events, baselines)
    assert merged.shape[0] == 2

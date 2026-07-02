from __future__ import annotations

import numpy as np
import pandas as pd

from dynsys_econometrics.multivariate import (
    joint_exceedances,
    stress_state_index,
    tail_dependence_coefficient,
    wide_panel,
)


def _long_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "series_id": ["x"] * 4 + ["y"] * 4,
            "value": [1.0, 2.0, 5.0, 6.0, 0.5, 2.5, 5.5, 1.0],
        }
    )


def test_wide_panel_converts_long_format() -> None:
    wide = wide_panel(_long_panel())
    assert wide.shape == (4, 2)


def test_joint_exceedances_detects_joint_events() -> None:
    table = joint_exceedances(_long_panel(), thresholds={"x": 4.5, "y": 2.0}, min_active=2)
    assert table["joint_exceedance"].sum() == 1


def test_tail_dependence_coefficient_on_handcrafted_data() -> None:
    wide = wide_panel(_long_panel())
    coefficient = tail_dependence_coefficient(wide, "x", "y", quantile=0.5)
    assert 0.0 <= coefficient <= 1.0


def test_stress_state_index_with_custom_weights() -> None:
    matrix = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2, freq="D"), "x": [True, False], "y": [True, True]})
    state = stress_state_index(matrix, weights={"x": 2.0, "y": 1.0})
    assert np.isclose(state["stress_score"].iloc[0], 1.0)

from __future__ import annotations

import numpy as np
import pytest

from dynsys_econometrics.simulation import logistic_map, simulate_ar1


def test_logistic_map_shape_and_bounds() -> None:
    values = logistic_map(n_steps=100, x0=0.2)
    assert values.shape == (100,)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_logistic_map_rejects_invalid_x0() -> None:
    with pytest.raises(ValueError):
        logistic_map(n_steps=10, x0=1.2)


def test_simulate_ar1_shape() -> None:
    values = simulate_ar1(n_steps=50, phi=0.5)
    assert values.shape == (50,)

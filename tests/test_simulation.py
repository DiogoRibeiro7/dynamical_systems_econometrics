from __future__ import annotations

import numpy as np
import pytest

from dynsys_econometrics.simulation import (
    logistic_map,
    simulate_ar1,
    simulate_coupled_logistic_maps,
    simulate_garch11,
    simulate_iid_gaussian,
    simulate_pomeau_manneville,
    simulate_student_t,
)


def test_logistic_map_shape_and_bounds() -> None:
    values = logistic_map(n_steps=100, x0=0.2)
    assert values.shape == (100,)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_logistic_map_rejects_invalid_x0() -> None:
    with pytest.raises(ValueError):
        logistic_map(n_steps=10, x0=1.2)


def test_simulate_iid_gaussian_reproducible() -> None:
    x = simulate_iid_gaussian(n_steps=64, mean=1.0, std=2.0, seed=13)
    y = simulate_iid_gaussian(n_steps=64, mean=1.0, std=2.0, seed=13)
    assert np.allclose(x, y)


def test_simulate_iid_gaussian_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        simulate_iid_gaussian(n_steps=0)
    with pytest.raises(ValueError):
        simulate_iid_gaussian(n_steps=10, std=0.0)


def test_simulate_student_t_shape() -> None:
    values = simulate_student_t(n_steps=50, df=5, seed=7)
    assert values.shape == (50,)


def test_simulate_student_t_invalid_df() -> None:
    with pytest.raises(ValueError):
        simulate_student_t(n_steps=10, df=0)


def test_simulate_ar1_shape() -> None:
    values = simulate_ar1(n_steps=50, phi=0.5)
    assert values.shape == (50,)


def test_simulate_ar1_reproducible() -> None:
    x = simulate_ar1(n_steps=50, phi=0.2, seed=7)
    y = simulate_ar1(n_steps=50, phi=0.2, seed=7)
    assert np.array_equal(x, y)


def test_simulate_ar1_invalid_phi() -> None:
    with pytest.raises(ValueError):
        simulate_ar1(n_steps=20, phi=1.5)


def test_simulate_garch11_shape_and_stationary_variance() -> None:
    values = simulate_garch11(n_steps=200, seed=5)
    assert values.shape == (200,)


def test_simulate_garch11_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        simulate_garch11(n_steps=20, alpha=0.6, beta=0.5)


def test_simulate_pomeau_manneville_shape() -> None:
    values = simulate_pomeau_manneville(n_steps=128, x0=0.2)
    assert values.shape == (128,)


def test_simulate_pomeau_manneville_bounds() -> None:
    values = simulate_pomeau_manneville(n_steps=100, x0=0.2)
    assert np.all((values >= 0.0) & (values < 1.0))


def test_simulate_pomeau_manneville_invalid_z() -> None:
    with pytest.raises(ValueError):
        simulate_pomeau_manneville(n_steps=20, x0=0.2, z=1.0)


def test_simulate_coupled_logistic_maps_shape() -> None:
    values = simulate_coupled_logistic_maps(n_steps=128, n_series=3, coupling=0.1, seed=4)
    assert values.shape == (128, 3)


def test_simulate_coupled_logistic_maps_reproducible() -> None:
    values_a = simulate_coupled_logistic_maps(
        n_steps=10,
        n_series=2,
        coupling=0.08,
        seed=11,
        x0=np.array([0.2, 0.4]),
    )
    values_b = simulate_coupled_logistic_maps(
        n_steps=10,
        n_series=2,
        coupling=0.08,
        seed=11,
        x0=np.array([0.2, 0.4]),
    )
    assert np.array_equal(values_a, values_b)


def test_simulate_coupled_logistic_maps_invalid_r_vector_shape() -> None:
    with pytest.raises(ValueError):
        simulate_coupled_logistic_maps(n_steps=10, n_series=2, r=np.array([3.9]))

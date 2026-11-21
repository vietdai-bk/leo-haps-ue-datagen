import os
import sys

import numpy as np
import pymap3d as pm
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import haps_ue as H  # noqa: E402


@pytest.mark.parametrize("P", [
    [[0.5, 0.5], [0.5, 0.5], [1, 0]],          # not square
    [[1.1, -0.1, 0], [0, 1, 0], [0, 0, 1]],     # negative
    [[0.5, 0.4, 0], [0, 1, 0], [0, 0, 1]],      # row sum 0.9
])
def test_invalid_transition_matrix(P):
    with pytest.raises(ValueError):
        H.validate_transition_matrix(P)


def test_default_matrix_valid_and_stationary():
    P = H.validate_transition_matrix(H.DEFAULT_P)
    pi = H.stationary_distribution(P)
    assert np.allclose(pi @ P, pi) and np.isclose(pi.sum(), 1) and np.all(pi > 0)


def test_state_frequencies_converge_to_stationary():
    sim = H.simulate_ue(n_ue=200, n_steps=3000, seed=1)
    freq = np.bincount(sim["state"].ravel(), minlength=3) / sim["state"].size
    assert np.allclose(freq, H.stationary_distribution(H.DEFAULT_P), atol=0.03)


def test_empirical_transitions_match_P():
    sim = H.simulate_ue(n_ue=200, n_steps=3000, seed=2)
    a, b = sim["state"][:-1].ravel(), sim["state"][1:].ravel()
    C = np.zeros((3, 3))
    np.add.at(C, (a, b), 1)
    assert np.allclose(C / C.sum(1, keepdims=True), H.DEFAULT_P, atol=0.005)


def test_stays_in_area_and_speed_matches_state():
    sim = H.simulate_ue(n_ue=100, n_steps=2000, radius_m=500.0, seed=3)  # small disc -> many reflections
    assert np.all(np.linalg.norm(sim["pos"], axis=-1) <= 500.0 + 1e-6)
    speed = np.linalg.norm(sim["vel"], axis=-1)
    for i, name in enumerate(H.STATES):
        lo, hi = H.SPEED_RANGE[name]
        m = sim["state"] == i
        assert np.all((speed[m] >= lo - 1e-9) & (speed[m] <= hi + 1e-9))


def test_motion_is_consistent_with_velocity():
    sim = H.simulate_ue(n_ue=50, n_steps=200, radius_m=1e6, seed=4)  # huge disc -> no reflection
    step = sim["pos"][1:] - sim["pos"][:-1]
    assert np.allclose(step, sim["vel"][:-1] * 1.0)


def test_reflection_keeps_step_length():
    sim = H.simulate_ue(n_ue=100, n_steps=1000, radius_m=300.0, seed=5)
    step = np.linalg.norm(sim["pos"][1:] - sim["pos"][:-1], axis=-1)
    assert np.all(step <= np.linalg.norm(sim["vel"][:-1], axis=-1) + 1e-6)


def test_reproducible_with_seed():
    a, b = H.simulate_ue(seed=7), H.simulate_ue(seed=7)
    c = H.simulate_ue(seed=8)
    assert np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["user_type"], b["user_type"])
    assert not np.array_equal(a["pos"], c["pos"])


def test_user_types_and_qos():
    sim = H.simulate_ue(n_ue=50, p_critical=0.2, seed=0)
    assert (sim["user_type"] == "critical").sum() == 10
    tab = H.ue_table(sim, H.HAPS())
    crit = tab["user_type"] == "critical"
    assert np.all(tab["priority"][crit] == 1) and np.all(tab["max_latency_ms"][~crit] == 100.0)
    assert tab["ue_id"].size == 50 * sim["state"].shape[0]


def test_geodetic_conversion_round_trip():
    haps = H.HAPS()
    pos = np.array([[1000.0, -2000.0], [0.0, 0.0]])
    lat, lon = H.ue_geodetic(pos, haps)
    e, n, _ = pm.geodetic2enu(lat, lon, 0.0, haps.lat, haps.lon, 0.0)
    assert np.allclose(e, pos[:, 0], atol=0.01) and np.allclose(n, pos[:, 1], atol=0.01)
    assert np.isclose(lat[1], haps.lat) and np.isclose(lon[1], haps.lon)
    # HAPS directly above its nadir UE at alt_m
    d = np.linalg.norm(haps.ecef() - H.ue_ecef(pos[1:], haps)[0])
    assert np.isclose(d, haps.alt_m, atol=0.01)


def test_bad_params():
    with pytest.raises(ValueError):
        H.simulate_ue(p_critical=1.5)
    with pytest.raises(ValueError):
        H.simulate_ue(radius_m=10.0, dt_s=1.0)

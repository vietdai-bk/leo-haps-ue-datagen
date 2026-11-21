import os
import sys

import numpy as np
import pymap3d as pm
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import orbit as O  # noqa: E402
from leo_haps_ue_datagen import rst as R  # noqa: E402

T_REF = "2025-08-10T06:00:00"
SITE = (35.6762, 139.6503, 20000.0)
nan = np.nan


def test_rst_entering_leaving_and_censored():
    m = np.array([0, 1, 1, 0, 1, 1], bool)  # enters at 1, leaves after 2, re-enters, still serving at end
    r, s, lb = R.rst(m, dt_s=2.0)
    assert np.allclose(r, [nan, 4, 2, nan, nan, nan], equal_nan=True)
    assert s.tolist() == [R.NOT_SERVING, R.SERVING, R.SERVING, R.NOT_SERVING, R.CENSORED, R.CENSORED]
    assert np.allclose(lb, [nan, 4, 2, nan, 4, 2], equal_nan=True)


def test_rst_never_visible_and_always_visible():
    r, s, _ = R.rst(np.zeros(5, bool))
    assert np.all(np.isnan(r)) and np.all(s == R.NOT_SERVING)
    r, s, lb = R.rst(np.ones(5, bool))
    assert np.all(np.isnan(r)) and np.all(s == R.CENSORED) and lb.tolist() == [5, 4, 3, 2, 1]


def test_rst_vectorised_over_leading_axes():
    rng = np.random.default_rng(0)
    m = rng.random((3, 4, 50)) > 0.4
    r, s, lb = R.rst(m)
    for i in range(3):
        for j in range(4):
            r1, s1, lb1 = R.rst(m[i, j])
            assert np.array_equal(s[i, j], s1) and np.allclose(r[i, j], r1, equal_nan=True)


def test_intervals():
    assert R.intervals([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert R.intervals([0, 0]) == []


def test_serving_mask_combines_conditions():
    m = R.serving_mask([40, 40, 40, 10, nan], 30, snr_dB=[5, 15, 15, 15, 15], snr_min_dB=10,
                       dist_m=[1, 1, 9, 1, 1], dist_max_m=5)
    assert m.tolist() == [False, True, False, False, False]


def test_elevation_from_matches_pymap3d():
    rng = np.random.default_rng(1)
    lat, lon = rng.uniform(-60, 60, 20), rng.uniform(-180, 180, 20)
    obs = np.stack(pm.geodetic2ecef(lat, lon, np.zeros(20)), -1)
    tgt = obs + rng.normal(0, 5e5, (20, 3))
    el, d = R.elevation_from(tgt, obs, lat, lon)
    _, el_ref, d_ref = pm.ecef2aer(*tgt.T, lat, lon, np.zeros(20))
    assert np.allclose(el, el_ref, atol=1e-6) and np.allclose(d, d_ref)


def test_moving_ue_exit_time_analytic():
    # target fixed 1 km above origin; UE moves east at 20 m/s -> elev 30 deg at x = 1000/tan30 = 1732 m
    lat0, lon0 = 10.0, 20.0
    tgt = np.array(pm.geodetic2ecef(lat0, lon0, 1000.0))
    t = np.arange(200.0)
    lat, lon, _ = pm.enu2geodetic(20.0 * t, 0 * t, 0 * t, lat0, lon0, 0.0)
    ue = np.stack(pm.geodetic2ecef(lat, lon, 0 * t), -1)
    out = R.rst_sat_ue(np.broadcast_to(tgt, (1, t.size, 3)), ue[:, None, :], lat[:, None], lon[:, None], 1.0)
    t_cross = 1000 / np.tan(np.radians(30)) / 20.0  # 86.6 s
    assert out["status"][0, 0, 0] == R.SERVING
    assert t_cross <= out["rst_s"][0, 0, 0] < t_cross + 1.0
    # static UE at origin never leaves -> censored
    static = R.rst_sat_ue(np.broadcast_to(tgt, (1, t.size, 3)), np.broadcast_to(ue[0], (t.size, 1, 3)),
                          np.full((t.size, 1), lat[0]), np.full((t.size, 1), lon[0]), 1.0)
    assert np.all(static["status"] == R.CENSORED)


@pytest.fixture(scope="module")
def real_pass():
    cat, _ = O.validate_catalog(O.load_catalog(os.path.join(ROOT, "data", "STARLINK_20250810.txt"))[:300], T_REF)
    p = O.compute_passes(cat, SITE, T_REF, 3600, 1.0, 30.0)
    sid = np.unique(p["sat_id"][p["visible"]])[:5]
    return cat, sid, p["times"]


def test_rst_real_pass_sat_haps(real_pass):
    cat, sid, times = real_pass
    tr = R.dense_tracks(cat, sid, times)
    out = R.rst_sat_haps(tr, *SITE, dt_s=1.0, elev_min_deg=30.0)
    assert np.all(out["available"] <= out["geometric_visible"])  # availability implies visibility
    for i in range(len(sid)):
        for k0, k1 in R.intervals(out["available"][i]):
            seg = out["rst_s"][i, k0:k1 + 1]
            if k1 == times.size - 1:
                assert np.all(out["status"][i, k0:k1 + 1] == R.CENSORED)
                continue
            assert seg[0] == k1 + 1 - k0 and np.all(np.diff(seg) == -1.0)  # pass length, decreases with t
            assert out["elev"][i, k1] >= 30 > out["elev"][i, k1 + 1]


def test_snr_condition_shortens_service(real_pass):
    cat, sid, times = real_pass
    tr = R.dense_tracks(cat, sid, times)
    geo = R.rst_sat_haps(tr, *SITE, 1.0, 30.0)
    snr = R.rst_sat_haps(tr, *SITE, 1.0, 30.0, snr_fn=lambda d, e: e, snr_min_dB=60.0)  # stricter
    assert np.all(snr["available"] <= geo["available"])
    assert snr["available"].sum() < geo["available"].sum()

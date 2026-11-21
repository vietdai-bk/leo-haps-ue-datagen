import os
import sys

import numpy as np
import pymap3d as pm
import pytest
from sgp4.api import Satrec, jday

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import visibility as V  # noqa: E402

TLE = os.path.join(ROOT, "data", "STARLINK_20250810.txt")
START = (2025, 8, 17, 9, 0, 0)
OBS = (35.6762, 139.6503, 20000.0)


@pytest.fixture(scope="module")
def small_tle(tmp_path_factory):
    blocks = V.parse_tle(TLE)[:150]
    p = tmp_path_factory.mktemp("tle") / "small.txt"
    p.write_text("".join(f"{n}\n{a}\n{b}\n" for n, a, b in blocks))
    return str(p)


# ---------- run lengths / RST ----------
def test_run_lengths_basic_and_censoring():
    m = np.array([0, 1, 1, 0, 1, 1, 1], bool)
    vt, c = V.run_lengths(m)
    assert vt.tolist() == [0, 2, 1, 0, 3, 2, 1]
    assert c.tolist() == [False, False, False, False, True, True, True]


def test_run_lengths_edges():
    vt, c = V.run_lengths(np.zeros(4, bool))
    assert vt.tolist() == [0] * 4 and not c.any()
    vt, c = V.run_lengths(np.ones(3, bool))
    assert vt.tolist() == [3, 2, 1] and c.all()
    vt, c = V.run_lengths(np.zeros(0, bool))
    assert vt.size == 0


def test_run_lengths_matches_naive_loop():
    rng = np.random.default_rng(0)
    m = rng.random(500) > 0.3
    vt, _ = V.run_lengths(m)
    for t in range(m.size):  # the notebook's original O(T^2) definition
        n = 0
        for k in range(t, m.size):
            if not m[k]:
                break
            n += 1
        assert vt[t] == n


# ---------- timeline ----------
def test_time_grid_exact_step():
    times, jd, fr = V.make_time_grid(START, 86400, 1.0)
    assert times.size == 86400 and np.all(np.diff(times) == 1.0)
    jd0, fr0 = jday(*START)
    assert jd == jd0 and fr[0] == fr0
    assert np.isclose((fr[-1] - fr[0]) * 86400, 86399, atol=1e-4)


# ---------- coordinates vs independent reference (skyfield) ----------
def test_teme_to_ecef_and_elevation_vs_skyfield():
    pytest.importorskip("skyfield.api")
    from skyfield.api import EarthSatellite, load, wgs84
    from skyfield.framelib import itrs

    name, l1, l2 = V.parse_tle(TLE)[0]
    ts = load.timescale(builtin=True)
    sat_sf = EarthSatellite(l1, l2, name, ts)
    sat = Satrec.twoline2rv(l1, l2)
    times, jd, fr = V.make_time_grid(START, 3600, 60.0)
    e, r, _ = sat.sgp4_array(np.full_like(fr, jd), fr)
    assert np.all(e == 0)
    ours = V.teme_to_ecef(r, jd + fr)

    t = ts.utc(*START[:5], START[5] + times)
    ref = sat_sf.at(t).frame_xyz(itrs).m.T
    err = np.linalg.norm(ours - ref, axis=1)
    assert err.max() < 500, err.max()  # GMST-only rotation vs full IAU: polar motion/UT1

    hap = wgs84.latlon(OBS[0], OBS[1], elevation_m=OBS[2])
    alt_ref, _, dist_ref = (sat_sf - hap).at(t).altaz()
    east, north, up = pm.ecef2enu(*ours.T, *OBS)
    elev = np.degrees(np.arctan2(up, np.hypot(east, north)))
    assert np.max(np.abs(elev - alt_ref.degrees)) < 0.05
    assert np.max(np.abs(np.linalg.norm([east, north, up], axis=0) - dist_ref.m)) < 500


def test_zenith_geometry():
    lat, lon, alt = 10.0, 20.0, 20000.0
    x, y, z = pm.geodetic2ecef(lat, lon, alt + 550e3)
    e, n, u = pm.ecef2enu(x, y, z, lat, lon, alt)
    assert np.isclose(np.degrees(np.arctan2(u, np.hypot(e, n))), 90, atol=1e-6)
    assert np.isclose(np.linalg.norm([e, n, u]), 550e3, rtol=1e-9)


# ---------- index mapping with invalid SGP4 samples ----------
def test_invalid_samples_do_not_shift_timeline(small_tle, monkeypatch):
    kw = dict(start_utc=START, total_seconds=7200, dt=10.0, use_parallel=False)
    good = V.compute_visibility(small_tle, **kw)
    name, sv = next((n, s) for n, s in good.sats.items() if s.idx.size > 5)
    bad_t = int(sv.idx[0]) - 1 if sv.idx[0] > 0 else int(sv.idx[-1]) + 1

    orig = Satrec.sgp4_array

    def patched(self, jd, fr):
        e, r, v = orig(self, jd, fr)
        e = e.copy()
        e[: max(bad_t, 1)] = 1  # invalidate samples before the pass
        return e, r, v

    monkeypatch.setattr(Satrec, "sgp4_array", patched)
    bad = V.compute_visibility(small_tle, **kw)
    sv2 = bad.sats[name]
    keep = sv.idx >= max(bad_t, 1)
    assert np.array_equal(sv2.idx, sv.idx[keep])
    assert np.allclose(sv2.enu, sv.enu[keep])
    assert bad.invalid_samples[name] == max(bad_t, 1)


# ---------- visibility pipeline ----------
def test_visibility_consistency_and_parallel_equal(small_tle):
    kw = dict(start_utc=START, total_seconds=7200, dt=10.0)
    seq = V.compute_visibility(small_tle, use_parallel=False, **kw)
    par = V.compute_visibility(small_tle, use_parallel=True, **kw)
    assert not seq.failed and seq.sats.keys() == par.sats.keys()
    assert np.array_equal(seq.cover_counts, par.cover_counts)
    assert seq.sats, "expected at least one visible satellite in 2 h"
    for sv in seq.sats.values():
        assert np.all(sv.elev > 30) and np.all(sv.slant > 0)
        assert np.allclose(sv.slant, np.linalg.norm(sv.enu, axis=1))
        # RST decreases by one per consecutive visible sample
        consec = np.diff(sv.idx) == 1
        assert np.all(sv.vt[:-1][consec] == sv.vt[1:][consec] + 1)
        assert np.all(sv.vt[:-1][~consec] == 1)
        last_t = seq.times.size - 1
        assert np.all(sv.censored == (sv.idx + sv.vt - 1 == last_t))


def test_legacy_results_format(small_tle):
    vis = V.compute_visibility(small_tle, start_utc=START, total_seconds=600, dt=10.0)
    rps = V.results_per_step(vis, cfso_fn=lambda s, z: 1.0)
    assert sorted(rps) == list(range(60))
    for rows in rps.values():
        for name, elev, vt, slant, c, enu in rows:  # unpacking used by model/HAP.py
            assert isinstance(vt, int) and vt >= 1 and c == 1.0 and len(enu) == 3

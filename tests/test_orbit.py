import json
import os
import sys

import numpy as np
import pytest
from sgp4 import exporter
from sgp4.api import Satrec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import orbit as O  # noqa: E402
from leo_haps_ue_datagen.visibility import parse_tle  # noqa: E402

TLE = os.path.join(ROOT, "data", "STARLINK_20250810.txt")
T_REF = "2025-08-10T06:00:00"  # near this TLE set's epoch (2025-08-10)
SITE = (35.6762, 139.6503, 20000.0)


@pytest.fixture(scope="module")
def blocks():
    return parse_tle(TLE)[:200]


@pytest.fixture(scope="module")
def omm_file(blocks, tmp_path_factory):
    p = tmp_path_factory.mktemp("omm") / "s.json"
    p.write_text(json.dumps([exporter.export_omm(Satrec.twoline2rv(a, b), n) for n, a, b in blocks]))
    return str(p)


@pytest.fixture(scope="module")
def catalog(omm_file):
    kept, _ = O.validate_catalog(O.load_catalog(omm_file), T_REF)
    return kept


def test_omm_equals_tle(blocks, omm_file):
    cat = O.load_catalog(omm_file)
    times = O.time_grid(T_REF, 3600, 60)
    for rec, (n, a, b) in zip(cat[:20], blocks[:20]):
        r1, v1, _ = O.propagate_ecef(rec.satrec, times)
        r2, v2, _ = O.propagate_ecef(Satrec.twoline2rv(a, b), times)
        assert rec.name == n
        # OMM epoch is rounded to 1 us -> ~mm differences
        assert np.nanmax(np.abs(r1 - r2)) < 1e-2 and np.nanmax(np.abs(v1 - v2)) < 1e-5


def test_malformed_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('[{"OBJECT_NAME": "X"')  # truncated download
    with pytest.raises(json.JSONDecodeError):
        O.load_catalog(str(p))


def test_validate_epoch_age_and_duplicates(omm_file):
    cat = O.load_catalog(omm_file)
    kept, rej = O.validate_catalog(cat, "2025-08-20T00:00:00", max_age_days=3)
    assert not kept and all("epoch age" in r for r in rej.values())
    dup = cat[:3] + [cat[0]]
    dup[0] = O.OrbitRecord(cat[0].sat_id, "old", cat[0].satrec, cat[0].epoch - np.timedelta64(1, "D"))
    kept, rej = O.validate_catalog(dup, T_REF)
    assert len(kept) == 3 and "duplicate" in rej[cat[0].sat_id]
    assert next(k for k in kept if k.sat_id == cat[0].sat_id).name != "old"


def test_epoch_matches_tle(blocks):
    rec = O.load_catalog(TLE)[0]
    assert str(rec.epoch).startswith("2025-08-10T03:18:4")  # 25222.13798611 d


def test_time_grid():
    t = O.time_grid("2026-11-21T00:00:00", 10, 0.5)
    assert t.size == 20 and np.all(np.diff(t) == np.timedelta64(500, "ms"))


def test_state_vs_skyfield(catalog):
    pytest.importorskip("skyfield")
    from skyfield.api import EarthSatellite, load, wgs84
    from skyfield.framelib import itrs

    ts = load.timescale(builtin=True)
    rec = catalog[0]
    times = O.time_grid(T_REF, 5400, 30)
    r, v, e = O.propagate_ecef(rec.satrec, times)
    secs = (times - times[0]) / np.timedelta64(1, "s")
    t = ts.utc(2025, 8, 10, 6, 0, secs)
    sf = EarthSatellite.from_satrec(rec.satrec, ts)
    pos, vel = sf.at(t).frame_xyz_and_velocity(itrs)
    assert np.max(np.linalg.norm(r - pos.m.T, axis=1)) < 500
    assert np.max(np.linalg.norm(v - vel.m_per_s.T, axis=1)) < 1.0
    alt, az, dist = (sf - wgs84.latlon(*SITE[:2], elevation_m=SITE[2])).at(t).altaz()
    d, el, _ = O.look_angles(r, *SITE)
    assert np.max(np.abs(el - alt.degrees)) < 0.05
    assert np.max(np.abs(d - dist.m)) < 500


def test_velocity_is_derivative_of_position(catalog):
    times = O.time_grid(T_REF, 4, 1)
    r, v, _ = O.propagate_ecef(catalog[0].satrec, times)
    fd = (r[2] - r[0]) / 2.0
    assert np.linalg.norm(fd - v[1]) < 0.05


def test_max_range_known_geometry():
    ro, rs = 6371e3, 6371e3 + 550e3
    assert np.isclose(O.max_range_for_elevation(90, ro, rs), 550e3)
    assert np.isclose(O.max_range_for_elevation(0, ro, rs), np.sqrt(rs**2 - ro**2))


def test_coarse_to_fine_equals_brute_force(catalog):
    kw = dict(start_utc=T_REF, duration_s=7200, dt_s=1.0, min_elev_deg=30.0)
    fast = O.compute_passes(catalog, SITE, coarse_dt_s=60.0, **kw)
    brute = O.compute_passes(catalog, SITE, coarse_dt_s=1.0, margin_m=1e9, **kw)  # every sat, every second
    vis = lambda p: set(zip(p["sat_id"][p["visible"]].tolist(), p["t_idx"][p["visible"]].tolist()))
    assert vis(brute), "expected visible samples"
    assert vis(fast) == vis(brute)
    assert np.all(fast["elev_deg"][fast["visible"]] >= 30)

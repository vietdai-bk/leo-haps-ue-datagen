"""Starlink orbit data (CelesTrak OMM JSON / TLE) -> SGP4 -> ECEF state, range, elevation.

Units: time UTC (numpy datetime64[ms]), distance m, velocity m/s, angles deg.
Frames: SGP4 outputs TEME. TEME -> ECEF(PEF) by R3(GMST) (Vallado et al. 2006,
"Revisiting Spacetrack Report #3", AIAA 2006-6753, sec. on TEME); polar motion
and UT1-UTC neglected (< ~0.5 km here, checked against skyfield in tests).
ECEF velocity: v_ecef = R3(GMST) v_teme - omega_E x r_ecef.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pymap3d as pm
from sgp4 import omm
from sgp4.api import Satrec

from leo_haps_ue_datagen.visibility import gst_from_jd_array, parse_tle

OMEGA_EARTH = 7.2921158553e-5  # rad/s, IERS Conventions 2010 nominal (for v_ecef only)
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=JSON"
_UNIX_JD = 2440587.5


@dataclass
class OrbitRecord:
    sat_id: int        # NORAD catalog number
    name: str
    satrec: Satrec
    epoch: np.datetime64  # UTC


def _omm_fields(rec):
    f = {k: str(v) for k, v in rec.items()}
    if "." not in f["EPOCH"]:  # sgp4.omm expects fractional seconds
        f["EPOCH"] += ".000000"
    return f


def load_catalog(path):
    """OMM JSON (preferred) or TLE text -> [OrbitRecord]. Raises on malformed JSON."""
    out = []
    if path.endswith(".json"):
        with open(path) as fh:
            data = json.load(fh)
        for rec in data:
            sat = Satrec()
            omm.initialize(sat, _omm_fields(rec))
            out.append(OrbitRecord(int(rec["NORAD_CAT_ID"]), rec["OBJECT_NAME"], sat, _satrec_epoch(sat)))
    else:
        for name, l1, l2 in parse_tle(path):
            sat = Satrec.twoline2rv(l1, l2)
            out.append(OrbitRecord(int(sat.satnum), name, sat, _satrec_epoch(sat)))
    return out


def _satrec_epoch(sat):
    days = (sat.jdsatepoch - _UNIX_JD) + sat.jdsatepochF
    return np.datetime64(int(round(days * 86400e3)), "ms")


def validate_catalog(catalog, t_ref, max_age_days=3.0):
    """Split into (kept, rejected{sat_id: reason}). Rejects stale epochs, SGP4 init/propagation
    errors at t_ref, non-LEO mean motion and duplicate IDs (keeps newest epoch)."""
    t_ref = np.datetime64(t_ref, "ms")
    rejected, by_id = {}, {}
    for rec in catalog:
        age = abs((t_ref - rec.epoch) / np.timedelta64(1, "D"))
        if age > max_age_days:
            rejected[rec.sat_id] = f"epoch age {age:.2f} d > {max_age_days} d"
            continue
        if rec.satrec.error != 0:
            rejected[rec.sat_id] = f"sgp4 init error {rec.satrec.error}"
            continue
        jd, fr = _jd(np.array([t_ref]))
        e, r, _ = rec.satrec.sgp4_array(jd, fr)
        if e[0] != 0:
            rejected[rec.sat_id] = f"sgp4 error {int(e[0])} at t_ref"
            continue
        alt_km = np.linalg.norm(r[0]) - 6378.137
        if not 150 < alt_km < 2000:
            rejected[rec.sat_id] = f"altitude {alt_km:.0f} km not LEO"
            continue
        prev = by_id.get(rec.sat_id)
        if prev is not None:
            rejected[rec.sat_id] = "duplicate id (kept newest epoch)"
            if prev.epoch >= rec.epoch:
                continue
        by_id[rec.sat_id] = rec
    return list(by_id.values()), rejected


def time_grid(start_utc, duration_s, dt_s):
    start = np.datetime64(start_utc, "ms")
    n = int(round(duration_s / dt_s))
    return start + (np.arange(n) * dt_s * 1000).astype("timedelta64[ms]")


def _jd(times):
    """datetime64 UTC -> (jd, fr) split for SGP4 precision."""
    days = (times - np.datetime64("1970-01-01T00:00:00", "ms")) / np.timedelta64(1, "D")
    whole = np.floor(days)
    return _UNIX_JD + whole, days - whole


def propagate_ecef(satrec, times):
    """-> (r_ecef m (n,3), v_ecef m/s (n,3), err (n,)). Invalid samples are NaN."""
    jd, fr = _jd(times)
    e, r, v = satrec.sgp4_array(jd, fr)
    g = gst_from_jd_array(jd + fr)
    c, s = np.cos(g), np.sin(g)
    rot = lambda a: np.stack([a[:, 0] * c + a[:, 1] * s, -a[:, 0] * s + a[:, 1] * c, a[:, 2]], 1)
    r_e = rot(r) * 1e3
    v_e = rot(v) * 1e3 - np.cross([0.0, 0.0, OMEGA_EARTH], r_e)
    bad = e != 0
    r_e[bad] = np.nan
    v_e[bad] = np.nan
    return r_e, v_e, e


def look_angles(r_ecef, lat, lon, alt_m):
    """-> (distance m, elevation deg, azimuth deg) from a WGS84 geodetic site."""
    az, el, rng = pm.ecef2aer(r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2], lat, lon, alt_m)
    return rng, el, az


def max_range_for_elevation(min_elev_deg, site_radius_m, sat_radius_m):
    """Slant range at elevation e on a spherical Earth (law of cosines):
    rho = sqrt(Rs^2 - Ro^2 cos^2 e) - Ro sin e."""
    e = np.radians(min_elev_deg)
    return np.sqrt(sat_radius_m**2 - (site_radius_m * np.cos(e)) ** 2) - site_radius_m * np.sin(e)


def compute_passes(catalog, site, start_utc, duration_s, dt_s=1.0, min_elev_deg=30.0,
                   coarse_dt_s=60.0, margin_m=50e3):
    """Fine-resolution states only around candidate passes.

    Coarse step: sample every coarse_dt_s. A satellite can be within range R_max at some
    instant in [t_k, t_k+coarse] only if range(t_k or t_k+coarse) < R_max + |v|*coarse/2
    (range changes at most |v_ecef| m/s; site fixed in ECEF). Those intervals are refined at
    dt_s. margin_m covers spherical-Earth R_max approximation.

    Returns dict of 1-D columns (one row per sat per fine sample inside refined windows):
    sat_id, timestamp, x,y,z [m], vx,vy,vz [m/s], dist_m, elev_deg, az_deg, visible.
    """
    lat, lon, alt = site
    fine = time_grid(start_utc, duration_s, dt_s)
    n_fine = fine.size
    step = int(round(coarse_dt_s / dt_s))
    coarse_idx = np.arange(0, n_fine, step)
    if coarse_idx[-1] != n_fine - 1:
        coarse_idx = np.append(coarse_idx, n_fine - 1)
    site_ecef = np.array(pm.geodetic2ecef(lat, lon, alt))
    ro = np.linalg.norm(site_ecef)

    cols = {k: [] for k in ("sat_id", "t_idx", "r", "v", "dist", "elev", "az")}
    for rec in catalog:
        r, v, e = propagate_ecef(rec.satrec, fine[coarse_idx])
        ok = e == 0
        if not ok.any():
            continue
        rs = np.nanmax(np.linalg.norm(r, axis=1))
        rmax = max_range_for_elevation(min_elev_deg, ro, rs) + margin_m
        d = np.linalg.norm(r - site_ecef, axis=1)
        vmax = np.nanmax(np.linalg.norm(v, axis=1))
        near = ok & (d < rmax + vmax * coarse_dt_s / 2)
        if not near.any():
            continue
        # fine samples covering [coarse_k - step, coarse_k + step] for every near coarse k
        sel = np.zeros(n_fine, bool)
        for k in coarse_idx[near]:
            sel[max(0, k - step): min(n_fine, k + step + 1)] = True
        idx = np.nonzero(sel)[0]
        rf, vf, ef = propagate_ecef(rec.satrec, fine[idx])
        good = ef == 0
        idx, rf, vf = idx[good], rf[good], vf[good]
        if idx.size == 0:
            continue
        dist, el, az = look_angles(rf, lat, lon, alt)
        cols["sat_id"].append(np.full(idx.size, rec.sat_id))
        cols["t_idx"].append(idx)
        cols["r"].append(rf)
        cols["v"].append(vf)
        cols["dist"].append(dist)
        cols["elev"].append(el)
        cols["az"].append(az)

    if not cols["sat_id"]:
        empty = np.zeros(0)
        return dict(sat_id=empty.astype(int), timestamp=empty.astype("datetime64[ms]"),
                    **{k: empty for k in ("x", "y", "z", "vx", "vy", "vz", "dist_m", "elev_deg", "az_deg")},
                    visible=empty.astype(bool), t_idx=empty.astype(int), times=fine)
    r = np.concatenate(cols["r"])
    v = np.concatenate(cols["v"])
    t_idx = np.concatenate(cols["t_idx"])
    elev = np.concatenate(cols["elev"])
    return dict(sat_id=np.concatenate(cols["sat_id"]), t_idx=t_idx, timestamp=fine[t_idx],
                x=r[:, 0], y=r[:, 1], z=r[:, 2], vx=v[:, 0], vy=v[:, 1], vz=v[:, 2],
                dist_m=np.concatenate(cols["dist"]), elev_deg=elev, az_deg=np.concatenate(cols["az"]),
                visible=elev >= min_elev_deg, times=fine)


def snapshot_name(now=None):
    now = now or datetime.now(timezone.utc)
    return f"starlink_{now:%Y%m%dT%H%M%SZ}.json"

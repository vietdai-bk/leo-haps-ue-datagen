"""Per-step satellite visibility in the legacy format of the original SAGIN project notebook.

Same physics/geometry as the notebook; fixes indexing, timeline, RST censoring,
error handling and memory. Units: time [s], distance [m] unless suffixed _km,
angles [deg] unless suffixed _rad.
"""
import concurrent.futures
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pymap3d as pm
from sgp4.api import Satrec, jday

EARTH_R_KM = 6371.0


def parse_tle(path):
    """Return [(name, line1, line2)] from a 2- or 3-line TLE file (same rules as notebook)."""
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    sats, i = [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            sats.append((f"NO_NAME_{i}", lines[i], lines[i + 1]))
            i += 2
            continue
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            sats.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
            continue
        i += 1
    return sats


def gst_from_jd_array(jd_full):
    """GMST [rad] (Meeus 1998 eq. 12.4, unchanged from notebook). jd_full is UT1≈UTC."""
    T = (jd_full - 2451545.0) / 36525.0
    gst_deg = (280.46061837 + 360.98564736629 * (jd_full - 2451545.0)
               + 0.000387933 * T**2 - (T**3) / 38710000.0)
    return np.deg2rad(np.mod(gst_deg, 360.0))


def teme_to_ecef(r_teme_km, jd_full):
    """TEME [km] -> ECEF [m] by R3(GMST); polar motion and UT1-UTC ignored (~tens of m)."""
    g = gst_from_jd_array(jd_full)
    c, s = np.cos(g), np.sin(g)
    x, y, z = r_teme_km[:, 0], r_teme_km[:, 1], r_teme_km[:, 2]
    return np.stack([x * c + y * s, -x * s + y * c, z], axis=1) * 1000.0


def make_time_grid(start_utc, total_seconds, dt=1.0):
    """times[k] = k*dt for k < total_seconds/dt. Notebook used linspace(0,T,T) -> dt=T/(T-1)."""
    n = int(round(total_seconds / dt))
    times = np.arange(n, dtype=np.float64) * dt
    jd, fr = jday(*start_utc)
    return times, jd, fr + times / 86400.0


def run_lengths(mask):
    """vt[t] = consecutive True samples starting at t (0 where False);
    censored[t] = that run is still True at the last sample (exit not observed)."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    vt = np.zeros(n, dtype=np.int64)
    if n == 0:
        return vt, np.zeros(0, dtype=bool)
    # index of next False at or after t (n if none), vectorised via reverse min-accumulate
    idx = np.where(~mask, np.arange(n), n)
    next_false = np.minimum.accumulate(idx[::-1])[::-1]
    vt = np.where(mask, next_false - np.arange(n), 0)
    censored = mask & (next_false == n)
    return vt, censored


@dataclass
class SatVisibility:
    name: str
    idx: np.ndarray        # timeline indices where visible
    elev: np.ndarray       # [deg]
    slant: np.ndarray      # [m]
    enu: np.ndarray        # (k,3) [m]
    vt: np.ndarray         # samples until exit (incl. current)
    censored: np.ndarray   # bool, exit not observed inside window


@dataclass
class VisibilityResult:
    times: np.ndarray
    dt: float
    cover_counts: np.ndarray
    sats: dict = field(default_factory=dict)       # name -> SatVisibility (visible ones only)
    n_loaded: int = 0
    failed: dict = field(default_factory=dict)     # name -> reason
    invalid_samples: dict = field(default_factory=dict)  # name -> #SGP4 error samples


def _process_one_sat(block, jd, fr_array, obs, cos_beam, min_elev_deg):
    name, l1, l2 = block
    lat, lon, alt = obs
    sat = Satrec.twoline2rv(l1, l2)
    e, r_km, _v = sat.sgp4_array(np.full_like(fr_array, jd), fr_array)
    ok = e == 0
    n_bad = int((~ok).sum())
    if not ok.any():
        return name, None, n_bad
    # keep full-length timeline: invalid samples -> NaN (fixes index shift of r_km[ok])
    r_ecef = np.full((fr_array.size, 3), np.nan)
    r_ecef[ok] = teme_to_ecef(r_km[ok], jd + fr_array[ok])

    obs_ecef = np.array(pm.geodetic2ecef(lat, lon, alt))
    with np.errstate(invalid="ignore"):
        cos_ang = (r_ecef / np.linalg.norm(r_ecef, axis=1, keepdims=True)) @ (obs_ecef / np.linalg.norm(obs_ecef))
    in_beam = ok & (cos_ang >= cos_beam)
    if not in_beam.any():
        return name, None, n_bad
    sel = np.nonzero(in_beam)[0]
    east, north, up = pm.ecef2enu(r_ecef[sel, 0], r_ecef[sel, 1], r_ecef[sel, 2], lat, lon, alt)
    elev = np.degrees(np.arctan2(up, np.hypot(east, north)))
    keep = elev > min_elev_deg
    if not keep.any():
        return name, None, n_bad
    mask = np.zeros(fr_array.size, dtype=bool)
    mask[sel[keep]] = True
    vt, cens = run_lengths(mask)
    idx = sel[keep]
    enu = np.stack([east[keep], north[keep], up[keep]], axis=1)
    return name, SatVisibility(name, idx, elev[keep], np.linalg.norm(enu, axis=1), enu,
                               vt[idx], cens[idx]), n_bad


def compute_visibility(tle_file, lat_obs=35.6762, lon_obs=139.6503, alt_obs_m=20000,
                       beam_radius_km=500, min_elev_deg=30,
                       start_utc=(2025, 8, 17, 9, 0, 0), total_seconds=86400, dt=1.0,
                       use_parallel=True, max_workers=None):
    """Satellites inside beam (geocentric angle) and above min_elev_deg, per time step."""
    blocks = parse_tle(tle_file)
    times, jd, fr_array = make_time_grid(start_utc, total_seconds, dt)
    cos_beam = np.cos(beam_radius_km / EARTH_R_KM)
    obs = (lat_obs, lon_obs, alt_obs_m)
    res = VisibilityResult(times, dt, np.zeros(times.size, dtype=np.int64), n_loaded=len(blocks))

    def collect(block, fut_or_val):
        try:
            name, sv, n_bad = fut_or_val() if callable(fut_or_val) else fut_or_val
        except Exception as exc:  # report, don't swallow
            res.failed[block[0]] = repr(exc)
            return
        if n_bad:
            res.invalid_samples[name] = n_bad
        if sv is not None:
            res.sats[name] = sv
            res.cover_counts[sv.idx] += 1

    args = (jd, fr_array, obs, cos_beam, min_elev_deg)
    if use_parallel:
        workers = max_workers or min(32, os.cpu_count() or 1)
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_process_one_sat, b, *args): b for b in blocks}
            for f in concurrent.futures.as_completed(futs):
                collect(futs[f], f.result)
    else:
        for b in blocks:
            collect(b, lambda b=b: _process_one_sat(b, *args))
    if res.failed:
        warnings.warn(f"{len(res.failed)} satellites failed to propagate; see result.failed")
    return res


def results_per_step(vis, cfso_fn=None, with_censored=False):
    """Legacy per-step format of the original SAGIN project (model/HAP.py there):
    {t: [(name, elev_deg, vt, slant_m, cfso, (E, N, U))]} for every t (empty list if none).
    cfso_fn(slant_m, zenith_deg) -> value; None leaves NaN. with_censored adds a 7th field."""
    out = {t: [] for t in range(vis.times.size)}
    for sv in vis.sats.values():
        for k, t in enumerate(sv.idx):
            c = cfso_fn(sv.slant[k], 90.0 - sv.elev[k]) if cfso_fn else np.nan
            row = (sv.name, sv.elev[k], int(sv.vt[k]), sv.slant[k], c, tuple(sv.enu[k]))
            out[int(t)].append(row + (bool(sv.censored[k]),) if with_censored else row)
    return out

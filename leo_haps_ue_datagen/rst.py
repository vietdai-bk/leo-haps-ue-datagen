"""Remaining Service Time (RST).

RST(t_k) = t_exit - t_k, t_exit = end of the current continuous serving interval.
Sampled at dt: if samples k0..k1 serve and k1+1 does not, the true exit lies in
(t_k1, t_k1+1]; we take t_exit = t_{k1+1} (first failing sample) -> error in [0, dt).
If the interval still serves at the last sample, t_exit is unobserved: RST = NaN,
status CENSORED and rst_lower_bound = t_last + dt - t_k (exit is at least that far).
Not serving -> RST = NaN, status NOT_SERVING (never 0, which would read as 'about to exit').

Geometric visibility = elevation >= 0 deg (above local horizon).
Communication availability = elevation >= elev_min AND snr >= snr_min AND dist <= dist_max (each optional).
"""
import numpy as np

NOT_SERVING, SERVING, CENSORED = 0, 1, 2


def serving_mask(elev_deg, elev_min_deg=None, snr_dB=None, snr_min_dB=None, dist_m=None, dist_max_m=None):
    m = np.ones(np.shape(elev_deg), bool)
    e = np.asarray(elev_deg)
    m &= np.isfinite(e)
    if elev_min_deg is not None:
        m &= e >= elev_min_deg
    if snr_min_dB is not None:
        m &= np.asarray(snr_dB) >= snr_min_dB
    if dist_max_m is not None:
        m &= np.asarray(dist_m) <= dist_max_m
    return m


def intervals(mask):
    """[(start, end_inclusive)] of consecutive True samples along the last axis (1-D)."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def rst(mask, dt_s=1.0):
    """Vectorised along the LAST axis (time). Returns (rst_s, status, rst_lower_bound_s)."""
    mask = np.asarray(mask, bool)
    n = mask.shape[-1]
    k = np.arange(n)
    idx = np.where(~mask, k, n)
    nxt = np.flip(np.minimum.accumulate(np.flip(idx, -1), axis=-1), -1)
    status = np.where(mask, np.where(nxt == n, CENSORED, SERVING), NOT_SERVING).astype(np.int8)
    span = (nxt - k) * dt_s  # t_exit - t_k, t_exit = first non-serving sample
    rst_s = np.where(status == SERVING, span, np.nan)
    lb = np.where(mask, span, np.nan)
    return rst_s, status, lb


def elevation_from(r_target, r_obs, lat_deg, lon_deg):
    """Elevation [deg] and range [m] of target from (possibly moving) observers.
    r_* (...,3) ECEF m; lat/lon geodetic deg of the observer (broadcastable). Up = WGS84 normal."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    d = np.asarray(r_target) - np.asarray(r_obs)
    rng = np.linalg.norm(d, axis=-1)
    return np.degrees(np.arcsin(np.sum(d * up, -1) / rng)), rng


def rst_sat_haps(r_sat, haps_lat, haps_lon, haps_alt, dt_s, elev_min_deg=30.0, snr_fn=None, snr_min_dB=None,
                 dist_max_m=None):
    """r_sat (n_sat, T, 3) ECEF on the full grid (NaN = invalid). snr_fn(dist, elev) -> dB (optional).
    Returns dict: elev, dist, geometric_visible, available, rst_s, status, rst_lb_s, each (n_sat, T)."""
    import pymap3d as pm
    r_h = np.array(pm.geodetic2ecef(haps_lat, haps_lon, haps_alt))
    elev, dist = elevation_from(r_sat, r_h, haps_lat, haps_lon)
    snr = snr_fn(dist, elev) if snr_fn else None
    avail = serving_mask(elev, elev_min_deg, snr, snr_min_dB, dist, dist_max_m)
    r, s, lb = rst(avail, dt_s)
    return dict(elev=elev, dist=dist, snr=snr, geometric_visible=np.isfinite(elev) & (elev >= 0),
                available=avail, rst_s=r, status=s, rst_lb_s=lb)


def rst_sat_ue(r_sat, r_ue, ue_lat, ue_lon, dt_s, elev_min_deg=30.0, snr_fn=None, snr_min_dB=None,
               dist_max_m=None):
    """r_sat (n_sat, T, 3); r_ue (T, n_ue, 3) with lat/lon (T, n_ue) -> arrays (n_sat, n_ue, T).
    UE motion is handled because the observer position/normal changes every sample."""
    rs = np.asarray(r_sat)[:, None, :, :]                    # (S,1,T,3)
    ru = np.moveaxis(np.asarray(r_ue), 0, 1)[None]           # (1,U,T,3)
    lat = np.moveaxis(np.asarray(ue_lat), 0, 1)[None]
    lon = np.moveaxis(np.asarray(ue_lon), 0, 1)[None]
    elev, dist = elevation_from(rs, ru, lat, lon)
    snr = snr_fn(dist, elev) if snr_fn else None
    avail = serving_mask(elev, elev_min_deg, snr, snr_min_dB, dist, dist_max_m)
    r, s, lb = rst(avail, dt_s)
    return dict(elev=elev, dist=dist, snr=snr, geometric_visible=np.isfinite(elev) & (elev >= 0),
                available=avail, rst_s=r, status=s, rst_lb_s=lb)


def dense_tracks(catalog, sat_ids, times):
    """ECEF (n_sat, T, 3) on the full grid for the chosen satellites (NaN where SGP4 fails)."""
    from leo_haps_ue_datagen.orbit import propagate_ecef
    by_id = {r.sat_id: r for r in catalog}
    return np.stack([propagate_ecef(by_id[i].satrec, times)[0] for i in sat_ids])

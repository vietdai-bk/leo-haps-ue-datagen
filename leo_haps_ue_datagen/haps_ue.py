"""HAPS node and Markov-chain UE mobility.

ALL numeric defaults below (speeds, transition matrix, QoS, critical ratio) are
SIMULATION ASSUMPTIONS, not measured Starlink/HAPS data.
Units: m, m/s, s, deg. UE plane coords are local ENU (east, north) [m] centred on
the HAPS ground nadir point.
"""
from dataclasses import dataclass

import numpy as np
import pymap3d as pm

STATES = ("stationary", "pedestrian", "vehicle")
# speed drawn U[lo, hi] m/s when entering a state (assumption)
SPEED_RANGE = {"stationary": (0.0, 0.0), "pedestrian": (1.0, 2.0), "vehicle": (10.0, 20.0)}
# per-step (dt = 1 s) transition probabilities, rows = from-state (assumption)
DEFAULT_P = np.array([[0.980, 0.015, 0.005],
                      [0.020, 0.970, 0.010],
                      [0.010, 0.020, 0.970]])

# QoS profiles: assumptions loosely following 3GPP TS 22.261 (URLLC vs eMBB) / TS 23.501 5QI.
QOS_PROFILES = {
    "critical": dict(max_latency_ms=10.0, min_rate_bps=1e6, reliability=0.99999, priority=1),
    "normal": dict(max_latency_ms=100.0, min_rate_bps=5e6, reliability=0.99, priority=5),
}


@dataclass
class HAPS:
    lat: float = 35.6762
    lon: float = 139.6503
    alt_m: float = 20000.0

    def position(self, t=None):
        """Geodetic (lat, lon, alt_m) at time t. Fixed for now; override for moving HAPS."""
        return self.lat, self.lon, self.alt_m

    def ecef(self, t=None):
        return np.array(pm.geodetic2ecef(*self.position(t)))


def validate_transition_matrix(P, atol=1e-9):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"transition matrix must be square, got {P.shape}")
    if np.any(P < 0):
        raise ValueError("transition matrix has negative entries")
    sums = P.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=atol):
        raise ValueError(f"rows must sum to 1, got {sums}")
    return P


def stationary_distribution(P):
    """pi with pi P = pi, sum(pi)=1 (left eigenvector for eigenvalue 1)."""
    w, v = np.linalg.eig(np.asarray(P).T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def _draw_speed(rng, states):
    lo = np.array([SPEED_RANGE[STATES[s]][0] for s in states])
    hi = np.array([SPEED_RANGE[STATES[s]][1] for s in states])
    return rng.uniform(lo, hi)


def simulate_ue(n_ue=50, n_steps=600, dt_s=1.0, radius_m=10e3, P=DEFAULT_P, p_critical=0.2,
                heading_sigma_deg=10.0, seed=0):
    """Discrete-time Markov mobility in a disc of radius radius_m.

    Each step: state_{k+1} ~ P[state_k, :]; on a state change a new speed is drawn
    from SPEED_RANGE; heading performs a random walk N(0, heading_sigma); position
    p += v*dt; at the boundary the UE is specularly reflected (position mirrored
    radially, velocity mirrored about the boundary normal).

    Returns dict: ue_id (n,), user_type (n,) str, state (T,n) int, pos (T,n,2) m,
    vel (T,n,2) m/s, plus 'config'.
    """
    P = validate_transition_matrix(P)
    if P.shape[0] != len(STATES):
        raise ValueError(f"P must be {len(STATES)}x{len(STATES)}")
    if not 0 <= p_critical <= 1:
        raise ValueError("p_critical must be in [0, 1]")
    vmax = max(hi for _, hi in SPEED_RANGE.values())
    if vmax * dt_s >= radius_m:
        raise ValueError("dt too large for area: one step can cross the disc")
    rng = np.random.default_rng(seed)

    n_crit = int(round(p_critical * n_ue))
    user_type = np.array(["critical"] * n_crit + ["normal"] * (n_ue - n_crit))
    rng.shuffle(user_type)

    r = radius_m * np.sqrt(rng.random(n_ue))  # uniform over the disc area
    phi = rng.uniform(0, 2 * np.pi, n_ue)
    p = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    s = rng.choice(len(STATES), size=n_ue, p=stationary_distribution(P))
    speed = _draw_speed(rng, s)
    heading = rng.uniform(0, 2 * np.pi, n_ue)
    cumP = np.cumsum(P, axis=1)

    state = np.empty((n_steps, n_ue), dtype=np.int8)
    pos = np.empty((n_steps, n_ue, 2))
    vel = np.empty((n_steps, n_ue, 2))
    sigma = np.radians(heading_sigma_deg)
    for k in range(n_steps):
        v = speed[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=1)
        state[k], pos[k], vel[k] = s, p, v
        new_s = (rng.random(n_ue)[:, None] > cumP[s]).sum(axis=1)
        new_s = np.minimum(new_s, len(STATES) - 1)  # guards float round-off in cumsum
        changed = new_s != s
        if changed.any():
            speed[changed] = _draw_speed(rng, new_s[changed])
        s = new_s
        heading = heading + rng.normal(0, sigma, n_ue)
        p = p + v * dt_s
        out = np.linalg.norm(p, axis=1) > radius_m
        if out.any():
            d = np.linalg.norm(p[out], axis=1, keepdims=True)
            nrm = p[out] / d
            p[out] = nrm * (2 * radius_m - d)
            dirv = np.stack([np.cos(heading[out]), np.sin(heading[out])], axis=1)
            dirv = dirv - 2 * np.sum(dirv * nrm, axis=1, keepdims=True) * nrm
            heading[out] = np.arctan2(dirv[:, 1], dirv[:, 0])

    return dict(ue_id=np.arange(n_ue), user_type=user_type, state=state, pos=pos, vel=vel,
                config=dict(n_ue=n_ue, n_steps=n_steps, dt_s=dt_s, radius_m=radius_m, P=P.tolist(),
                            p_critical=p_critical, heading_sigma_deg=heading_sigma_deg, seed=seed,
                            speed_range=SPEED_RANGE, qos=QOS_PROFILES, assumptions="simulation"))


def ue_geodetic(pos, haps, t=None):
    """Local ENU (e, n) on the ground around HAPS nadir -> (lat, lon) deg; UE alt = 0 m."""
    lat0, lon0, _ = haps.position(t)
    lat, lon, _ = pm.enu2geodetic(pos[..., 0], pos[..., 1], 0.0, lat0, lon0, 0.0)
    return lat, lon


def ue_ecef(pos, haps, t=None):
    lat, lon = ue_geodetic(pos, haps, t)
    return np.stack(pm.geodetic2ecef(lat, lon, np.zeros_like(lat)), axis=-1)


def ue_table(sim, haps, times=None):
    """Long-format columns: ue_id, t_idx, timestamp, user_type, state, e, n, lat, lon, vx, vy, speed,
    plus QoS columns from QOS_PROFILES."""
    T, n = sim["state"].shape
    lat, lon = ue_geodetic(sim["pos"], haps)
    ut = np.tile(sim["user_type"], T)
    cols = dict(ue_id=np.tile(sim["ue_id"], T), t_idx=np.repeat(np.arange(T), n), user_type=ut,
                state=np.array(STATES)[sim["state"].ravel()],
                e=sim["pos"][..., 0].ravel(), n=sim["pos"][..., 1].ravel(),
                lat=lat.ravel(), lon=lon.ravel(),
                vx=sim["vel"][..., 0].ravel(), vy=sim["vel"][..., 1].ravel(),
                speed=np.linalg.norm(sim["vel"], axis=-1).ravel())
    if times is not None:
        cols["timestamp"] = np.repeat(np.asarray(times)[:T], n)
    for key in QOS_PROFILES["normal"]:
        cols[key] = np.array([QOS_PROFILES[u][key] for u in ut])
    return cols

"""Build the LEO-HAPS-UE time-series dataset.

Tables (long format, Parquet or CSV), joinable on (t_idx | timestamp, sat_id, ue_id):
  sat_haps : satellite -> HAPS (geometry, RST, FSO + RF backhaul)        rows: sat elev from HAPS >= row_min_elev
  sat_ue   : satellite -> UE   (geometry, RF SNR, RST, bistatic ISAC)    rows: sat elev from UE >= row_min_elev
  haps_ue  : HAPS -> UE        (geometry, RF SNR, monostatic ISAC)       rows: every UE, every step
  ue       : UE trajectory, mobility state, user type, QoS
  metadata.json : config, seed, orbit snapshot + sha256, units, assumptions, validation results
Missing values are NaN (never 0); RST status 0/1/2 = not serving / serving / censored.

Usage:  python -m leo_haps_ue_datagen.build_dataset --orbit data/STARLINK_20250810.txt --start 2025-08-10T06:00:00 \
          --duration 86400 --n-ue 50 --chunk 600 --out output/day
        python -m leo_haps_ue_datagen.build_dataset --quick      (5 min, 10 UE: end-to-end smoke test)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pymap3d as pm

from leo_haps_ue_datagen import fso_sat_haps as FSO
from leo_haps_ue_datagen import haps_ue as HU
from leo_haps_ue_datagen import isac as ISAC
from leo_haps_ue_datagen import rf_link as RF
from leo_haps_ue_datagen import orbit as O
from leo_haps_ue_datagen import rst as RST

DEFAULTS = dict(orbit="data/STARLINK_20250810.txt", start="2025-08-10T06:00:00", duration_s=3600, dt_s=1.0,
                n_ue=50, ue_radius_m=10e3, p_critical=0.2, seed=0,
                haps=dict(lat=35.6762, lon=139.6503, alt_m=20000.0),
                elev_min_haps_deg=30.0, elev_min_ue_deg=30.0, row_min_elev_deg=20.0,
                max_epoch_age_days=3.0, snr_min_comm_dB=0.0, fmt="parquet",
                chunk_s=600, lookahead_s=600)

UNITS = dict(timestamp="UTC", distance="m", velocity="m/s", angle="deg", snr="dB", power="dBm",
             capacity="bit/s", bandwidth="Hz", rst="s", channel_gain="linear (Gtx*Grx/L)")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(table, path, fmt):
    import pyarrow as pa
    t = pa.table({k: np.asarray(v) for k, v in table.items()})
    if fmt == "parquet":
        import pyarrow.parquet as pq
        pq.write_table(t, path + ".parquet")
    else:
        import pyarrow.csv as pc
        pc.write_csv(t, path + ".csv")


def read_table(out_dir, name, columns=None, t_max=None):
    """Read one output table (`name`.parquet or `name`.csv, whichever exists) as a pyarrow Table.
    t_max keeps only rows with t_idx < t_max; rows are filtered while streaming, so a long run can be
    previewed without loading the whole file into RAM."""
    import pyarrow.dataset as ds
    path = os.path.join(out_dir, name)
    if os.path.exists(path + ".parquet"):
        d = ds.dataset(path + ".parquet", format="parquet")
    elif os.path.exists(path + ".csv"):
        import pyarrow.csv as pc
        # null_values=[]: keep "nan" as float NaN (pyarrow would otherwise turn it into null -> None)
        fmt = ds.CsvFileFormat(convert_options=pc.ConvertOptions(null_values=[]))
        d = ds.dataset(path + ".csv", format=fmt)
    else:
        raise FileNotFoundError(f"{path}.parquet / {path}.csv not found")
    flt = (ds.field("t_idx") < t_max) if t_max is not None else None
    return d.to_table(columns=columns, filter=flt)


def _fso_rows(dist, elev, log=None):
    """FSO metrics only where the link is above the HAPS elevation mask (else NaN)."""
    n = dist.size
    cap, msnr, gmax = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    idx = np.nonzero(np.isfinite(dist))[0]
    for j, i in enumerate(idx):
        if log and j and j % max(1, idx.size // 4) == 0:
            log(f"    FSO capacity {j}/{idx.size} rows")
        z = 90.0 - elev[i]
        cap[i] = FSO.ergodic_capacity_vec(dist[i], z)
        msnr[i] = 10 * np.log10(FSO.mean_snr(dist[i], z))
        gmax[i] = 10 * np.log10(FSO.gamma_max(dist[i], z))
    return cap, msnr, gmax


class _Writer:
    """Appends table chunks to one Parquet / CSV file per table (keeps only one chunk in RAM)."""

    def __init__(self, out_dir, fmt):
        self.out_dir, self.fmt, self.w, self.schema, self.rows = out_dir, fmt, {}, {}, {}

    def write(self, name, table):
        import pyarrow as pa
        t = pa.table({k: np.asarray(v) for k, v in table.items()})
        if name not in self.w:
            path = os.path.join(self.out_dir, f"{name}.{self.fmt}")
            if self.fmt == "parquet":
                import pyarrow.parquet as pq
                self.w[name] = pq.ParquetWriter(path, t.schema)
            else:
                import pyarrow.csv as pc
                self.w[name] = pc.CSVWriter(path, t.schema)
            self.schema[name], self.rows[name] = t.schema, 0
        if t.num_rows:
            self.w[name].write_table(t.cast(self.schema[name]))
            self.rows[name] += t.num_rows

    def close(self):
        for w in self.w.values():
            w.close()


def _slice_sim(sim, a, b):
    out = dict(sim)
    for k in ("state", "pos", "vel"):
        out[k] = sim[k][a:b]
    return out


def _window(cfg, kept, haps, sim, times, a, b, b_ext, log):
    """Tables for steps [a, b) (global t_idx). Geometry/RST are computed on [a, b_ext) so that the
    remaining service time of a link that is still serving at b - 1 is resolved by the look-ahead."""
    site, dt, U = haps.position(), cfg["dt_s"], cfg["n_ue"]
    T_all, Tw, n_out = times.size, b_ext - a, b - a
    tw = times[a:b_ext]
    passes = O.compute_passes(kept, site, tw[0], Tw * dt, dt,
                              min_elev_deg=cfg["row_min_elev_deg"], margin_m=50e3 + cfg["ue_radius_m"])
    sat_ids = np.unique(passes["sat_id"])
    log(f"  pass search: {sat_ids.size} candidate satellites")
    by_id = {r.sat_id: r for r in kept}
    states = [O.propagate_ecef(by_id[i].satrec, tw) for i in sat_ids]
    tracks = np.stack([st[0] for st in states]) if states else np.zeros((0, Tw, 3))
    vel = np.stack([st[1] for st in states]) if states else np.zeros((0, Tw, 3))
    simw = _slice_sim(sim, a, b_ext)
    ue_lat, ue_lon = HU.ue_geodetic(simw["pos"], haps)
    ue_xyz = HU.ue_ecef(simw["pos"], haps)

    def check_lookahead(status, t_local, what):
        if b_ext < T_all and np.any((status == RST.CENSORED) & (t_local < n_out)):
            raise RuntimeError(f"{what}: a link is still serving at the end of the look-ahead window; "
                               f"increase lookahead_s (now {cfg['lookahead_s']} s)")

    sh = RST.rst_sat_haps(tracks, *site, dt_s=dt, elev_min_deg=cfg["elev_min_haps_deg"])
    keep = np.isfinite(sh["elev"]) & (sh["elev"] >= cfg["row_min_elev_deg"])
    keep[:, n_out:] = False
    si, ti = np.nonzero(keep)
    check_lookahead(sh["status"][si, ti], ti, "sat->HAPS")
    d, e = sh["dist"][si, ti], sh["elev"][si, ti]
    _, _, az = pm.ecef2aer(*tracks[si, ti].T, *site)
    avail = sh["available"][si, ti]
    log(f"  sat->HAPS: {si.size} rows, {int(avail.sum())} available -> FSO capacity")
    cap, msnr, gmax = _fso_rows(np.where(avail, d, np.nan), e, log)
    rf = RF.sat_haps_rf_link(d, e)
    haps_xyz = haps.ecef()
    sat_haps = dict(
        t_idx=ti + a, timestamp=tw[ti], sat_id=sat_ids[si],
        sat_x=tracks[si, ti, 0], sat_y=tracks[si, ti, 1], sat_z=tracks[si, ti, 2],
        sat_vx=vel[si, ti, 0], sat_vy=vel[si, ti, 1], sat_vz=vel[si, ti, 2],
        haps_x=np.full(si.size, haps_xyz[0]), haps_y=np.full(si.size, haps_xyz[1]),
        haps_z=np.full(si.size, haps_xyz[2]),
        sat_haps_distance_m=d, elev_deg=e, az_deg=az,
        geometric_visible=sh["geometric_visible"][si, ti], available=avail,
        rst_sat_haps_s=sh["rst_s"][si, ti], rst_status=sh["status"][si, ti], rst_lb_s=sh["rst_lb_s"][si, ti],
        fso_capacity_bps=cap, fso_mean_snr_dB=msnr, fso_gamma_max_dB=gmax,
        rf_snr_dB=rf["snr_dB"], rf_capacity_bps=RF.capacity_bps(RF.SAT_HAPS_RF["B_hz"], rf["snr_dB"]),
        rf_channel_gain=rf["channel_gain"], rf_noise_power_dBm=np.full(si.size, rf["noise_dBm"]),
        rf_bandwidth_hz=np.full(si.size, RF.SAT_HAPS_RF["B_hz"]),
        rf_p_max_dBm=np.full(si.size, RF.SAT_HAPS_RF["P_tx_dBm"]))

    el_hu, d_hu = RST.elevation_from(np.broadcast_to(haps_xyz, ue_xyz.shape), ue_xyz, ue_lat, ue_lon)
    lb_hu = RF.haps_ue_link(d_hu[:n_out], el_hu[:n_out])
    isac_m = ISAC.sensing_metrics(ISAC.snr_monostatic_db(d_hu[:n_out]), ISAC.HAPS_MONO["B_hz"])
    ut = np.broadcast_to(sim["user_type"], (n_out, U))
    N = n_out * U
    haps_ue = dict(
        t_idx=np.repeat(np.arange(a, b), U), timestamp=np.repeat(times[a:b], U), ue_id=np.tile(sim["ue_id"], n_out),
        user_type=ut.ravel(), haps_ue_distance_m=d_hu[:n_out].ravel(), elev_deg=el_hu[:n_out].ravel(),
        snr_haps_ue_dB=lb_hu["snr_dB"].ravel(), channel_gain=lb_hu["channel_gain"].ravel(),
        noise_power_dBm=np.full(N, lb_hu["noise_dBm"]), bandwidth_hz=np.full(N, RF.HAPS_UE["B_hz"]),
        p_max_dBm=np.full(N, RF.HAPS_UE["P_tx_dBm"]),
        capacity_bps=RF.capacity_bps(RF.HAPS_UE["B_hz"], lb_hu["snr_dB"]).ravel(),
        available=(lb_hu["snr_dB"] >= cfg["snr_min_comm_dB"]).ravel(),
        isac_snr_dB=isac_m["isac_snr_dB"].ravel(), isac_pd=isac_m["pd_swerling1"].ravel(),
        isac_range_crlb_m=isac_m["range_crlb_m"].ravel(), sensing_feasible=isac_m["sensing_feasible"].ravel())

    # sat -> UE one satellite at a time to bound memory
    cols = {k: [] for k in ("t_idx", "sat_id", "ue_id", "user_type", "sat_ue_distance_m", "elev_deg",
                            "snr_sat_ue_dB", "channel_gain", "capacity_bps", "geometric_visible", "available",
                            "rst_sat_ue_s", "rst_status", "rst_lb_s", "isac_snr_dB", "isac_pd",
                            "isac_range_crlb_m", "sensing_feasible")}
    snr_fn = lambda dd, ee: RF.sat_ue_link(dd, ee)["snr_dB"]
    for s in range(sat_ids.size):
        if s and s % max(1, sat_ids.size // 10) == 0:
            log(f"    sat->UE {s}/{sat_ids.size} satellites")
        o = RST.rst_sat_ue(tracks[s:s + 1], ue_xyz, ue_lat, ue_lon, dt, cfg["elev_min_ue_deg"],
                           snr_fn=snr_fn, snr_min_dB=cfg["snr_min_comm_dB"])
        el = o["elev"][0]
        ok = np.isfinite(el) & (el >= cfg["row_min_elev_deg"])
        ok[:, n_out:] = False
        uu, tt = np.nonzero(ok)
        if uu.size == 0:
            continue
        check_lookahead(o["status"][0, uu, tt], tt, "sat->UE")
        dist = o["dist"][0, uu, tt]
        snr = o["snr"][0, uu, tt]
        bi = ISAC.sensing_metrics(ISAC.snr_bistatic_db(dist, d_hu[tt, uu]), ISAC.SAT_HAPS_BI["B_hz"])
        lb = RF.sat_ue_link(dist, el[uu, tt])
        for k, v in dict(t_idx=tt + a, sat_id=np.full(uu.size, sat_ids[s]), ue_id=uu, user_type=sim["user_type"][uu],
                         sat_ue_distance_m=dist, elev_deg=el[uu, tt], snr_sat_ue_dB=snr,
                         channel_gain=lb["channel_gain"], capacity_bps=RF.capacity_bps(RF.SAT_UE["B_hz"], snr),
                         geometric_visible=o["geometric_visible"][0, uu, tt], available=o["available"][0, uu, tt],
                         rst_sat_ue_s=o["rst_s"][0, uu, tt], rst_status=o["status"][0, uu, tt],
                         rst_lb_s=o["rst_lb_s"][0, uu, tt], isac_snr_dB=bi["isac_snr_dB"], isac_pd=bi["pd_swerling1"],
                         isac_range_crlb_m=bi["range_crlb_m"], sensing_feasible=bi["sensing_feasible"]).items():
            cols[k].append(v)
    empty = dict(t_idx=np.int64, sat_id=np.int64, ue_id=np.int64, user_type=str, geometric_visible=bool,
                 available=bool, rst_status=np.int8, sensing_feasible=bool)
    sat_ue = {k: np.concatenate(v) if v else np.zeros(0, empty.get(k, float)) for k, v in cols.items()}
    n = sat_ue["t_idx"].size
    sat_ue["timestamp"] = times[sat_ue["t_idx"].astype(int)]
    sat_ue["noise_power_dBm"] = np.full(n, RF.noise_dbm(RF.SAT_UE["B_hz"], RF.SAT_UE["NF_dB"]))
    sat_ue["bandwidth_hz"] = np.full(n, RF.SAT_UE["B_hz"])
    sat_ue["p_max_dBm"] = np.full(n, RF.SAT_UE["P_tx_dBm"])
    log(f"  sat->UE: {n} rows")

    simo = _slice_sim(sim, a, b)
    ue_tab = HU.ue_table(simo, haps, times[a:b])
    ue_tab["t_idx"] = ue_tab["t_idx"] + a
    tables = dict(sat_haps=sat_haps, sat_ue=sat_ue, haps_ue=haps_ue, ue=ue_tab)
    return tables, dict(times=times[a:b], sim=simo)


def _merge_reports(reps):
    """Combine per-window validation reports: booleans with all(), numbers with max(), lists/dicts summed."""
    out = {}
    for k in reps[0]:
        vals = [r[k] for r in reps if k in r]
        if all(isinstance(v, bool) for v in vals):
            out[k] = all(vals)
        elif all(isinstance(v, (int, float)) or v is None for v in vals):
            nums = [v for v in vals if v is not None]
            out[k] = max(nums) if nums else None
        elif all(isinstance(v, list) for v in vals):
            out[k] = sorted(set(sum(vals, [])))
        elif all(isinstance(v, dict) for v in vals):
            out[k] = {}
            for v in vals:
                for kk, c in v.items():
                    out[k][kk] = out[k].get(kk, 0) + c
    return out


def build(cfg, out_dir, verbose=True):
    """Generate the dataset into out_dir.

    cfg["chunk_s"] (None = whole duration at once) splits the time axis into windows that are computed,
    validated and appended to the output files one at a time, so RAM depends on chunk_s, not on the
    total duration. Each window is extended by cfg["lookahead_s"] so RST is not cut at chunk edges.
    Returns (tables, meta, extra); tables/extra hold the whole run only when it fits in one chunk,
    otherwise they hold the first chunk (used for validation.png).
    """
    t0 = time.time()
    cfg = dict(DEFAULTS, **cfg)

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:7.1f} s] {msg}", flush=True)

    dt = cfg["dt_s"]
    times = O.time_grid(cfg["start"], cfg["duration_s"], dt)
    T = times.size
    step = T if not cfg.get("chunk_s") else max(1, int(round(cfg["chunk_s"] / dt)))
    look = int(round(cfg["lookahead_s"] / dt))
    bounds = [(a, min(a + step, T)) for a in range(0, T, step)]
    log(f"start: {cfg['duration_s']:g} s @ dt={dt:g} s, {cfg['n_ue']} UE, {len(bounds)} chunk(s), "
        f"orbit={cfg['orbit']}")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):                    # stale files from a run in the other format
        if f.split(".")[0] in ("sat_haps", "sat_ue", "haps_ue", "ue") and f.endswith((".parquet", ".csv")):
            os.remove(os.path.join(out_dir, f))
    haps = HU.HAPS(**cfg["haps"])

    catalog = O.load_catalog(cfg["orbit"])
    kept, rejected = O.validate_catalog(catalog, cfg["start"], cfg["max_epoch_age_days"])
    log(f"orbits: {len(catalog)} loaded, {len(kept)} kept, {len(rejected)} rejected")
    sim = HU.simulate_ue(cfg["n_ue"], T, dt, cfg["ue_radius_m"], p_critical=cfg["p_critical"], seed=cfg["seed"])
    log(f"UE mobility simulated ({cfg['n_ue']} UE x {T} steps)")

    writer, reports, first = _Writer(out_dir, cfg["fmt"]), [], None
    try:
        for i, (a, b) in enumerate(bounds):
            log(f"chunk {i + 1}/{len(bounds)}: t_idx {a}..{b - 1}")
            tables, extra = _window(cfg, kept, haps, sim, times, a, b, min(T, b + look), log)
            reports.append(validate(tables, extra["sim"], dt))
            for name, tab in tables.items():
                writer.write(name, tab)
            log(f"  validation {'OK' if reports[-1]['ok'] else 'FAILED'}; chunk written")
            if first is None:
                first = (tables, extra)
            del tables, extra
    finally:
        writer.close()
    report = _merge_reports(reports)
    for name, rows in writer.rows.items():
        path = os.path.join(out_dir, f"{name}.{cfg['fmt']}")
        log(f"wrote {name}.{cfg['fmt']}: {rows} rows, {os.path.getsize(path) / 1e6:.1f} MB")
    plot(*first, out_dir)
    meta = dict(
        config=cfg, units=UNITS, created_utc=str(np.datetime64("now")),
        orbit_snapshot=dict(path=cfg["orbit"], sha256=_sha256(cfg["orbit"]), n_loaded=len(catalog),
                            n_kept=len(kept), rejected={str(k): v for k, v in rejected.items()}),
        seed=cfg["seed"], ue_config={k: v for k, v in sim["config"].items()},
        presets=dict(HAPS_UE=RF.HAPS_UE, SAT_UE=RF.SAT_UE, SAT_HAPS_RF=RF.SAT_HAPS_RF,
                     FSO=FSO.FSO_PARAMS, ISAC_MONO=ISAC.HAPS_MONO, ISAC_BI=ISAC.SAT_HAPS_BI,
                     isac_rcs_m2=ISAC.RCS_M2, isac_p_fa=ISAC.P_FA, isac_snr_min_dB=ISAC.SNR_MIN_DB),
        rst_status={"0": "not_serving", "1": "serving", "2": "censored (exit not observed)"},
        assumptions="All radio, QoS, mobility and ISAC parameters are simulation assumptions; only orbits "
                    "come from CelesTrak/TLE data via SGP4.",
        chunks=len(bounds), validation_png="first chunk only" if len(bounds) > 1 else "whole run",
        row_counts=dict(writer.rows), validation=report, build_seconds=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"done: validation {'OK' if report['ok'] else 'FAILED'}")
    return first[0], meta, first[1]


def _rst_ok(t_idx, key, rst_s, status, dt):
    """Within each link, serving RST must drop by exactly dt per consecutive step."""
    order = np.lexsort((t_idx, key))
    t, k, r, s = t_idx[order], key[order], rst_s[order], status[order]
    nxt = (k[1:] == k[:-1]) & (t[1:] == t[:-1] + 1) & (s[:-1] == RST.SERVING) & (s[1:] == RST.SERVING)
    return bool(np.allclose(r[:-1][nxt] - r[1:][nxt], dt))


def validate(tables, sim, dt):
    rep = {}
    for name, tab in tables.items():
        bad = {}
        for k, v in tab.items():
            a = np.asarray(v)
            if a.dtype.kind == "f":
                nonfinite = int((~np.isfinite(a)).sum())
                if nonfinite:
                    bad[k] = nonfinite
        rep[f"{name}_nonfinite_counts"] = bad
    sh, su, hu = tables["sat_haps"], tables["sat_ue"], tables["haps_ue"]
    allowed_nan = ("rst_", "fso_")
    rep["unexpected_nonfinite"] = [f"{n}.{k}" for n in ("sat_haps", "sat_ue", "haps_ue")
                                   for k in rep[f"{n}_nonfinite_counts"] if not k.startswith(allowed_nan)]
    # orbit continuity: |r(t+1) - r(t)| ~ |v| dt
    r = np.stack([sh["sat_x"], sh["sat_y"], sh["sat_z"]], 1)
    v = np.linalg.norm(np.stack([sh["sat_vx"], sh["sat_vy"], sh["sat_vz"]], 1), axis=1)
    o = np.lexsort((sh["t_idx"], sh["sat_id"]))
    same = (sh["sat_id"][o][1:] == sh["sat_id"][o][:-1]) & (sh["t_idx"][o][1:] == sh["t_idx"][o][:-1] + 1)
    step = np.linalg.norm(np.diff(r[o], axis=0), axis=1)[same]
    rep["orbit_step_rel_err_max"] = float(np.max(np.abs(step / (v[o][:-1][same] * dt) - 1))) if same.any() else None
    # SNR vs distance: free-space part must be exactly -20log10(d) (+ atm) for sat links
    k = RF.sat_ue_link(1.0, 90.0)["snr_dB"]  # value at d=1 m, elev 90 (atm at zenith)
    if su["t_idx"].size:
        resid = su["snr_sat_ue_dB"] + 20 * np.log10(su["sat_ue_distance_m"]) + RF.atm_loss_db(
            su["elev_deg"], RF.SAT_UE["A_zenith_dB"]) - (k + RF.SAT_UE["A_zenith_dB"])
        rep["sat_ue_snr_vs_distance_max_resid_dB"] = float(np.max(np.abs(resid)))
    rep["haps_ue_snr_decreases_with_distance"] = bool(
        np.all(np.diff(hu["snr_haps_ue_dB"][np.argsort(hu["haps_ue_distance_m"])]) <= 1e-3))  # equal d can differ ~1e-6 dB via the local normal
    rep["sat_haps_rst_consistent"] = _rst_ok(sh["t_idx"], sh["sat_id"], sh["rst_sat_haps_s"], sh["rst_status"], dt)
    if su["t_idx"].size:
        key = su["sat_id"].astype(np.int64) * 100000 + su["ue_id"]
        rep["sat_ue_rst_consistent"] = _rst_ok(su["t_idx"], key, su["rst_sat_ue_s"], su["rst_status"], dt)
    rep["available_implies_visible"] = bool(np.all(~sh["available"] | sh["geometric_visible"]))
    rep["ue_inside_area"] = bool(np.all(np.linalg.norm(sim["pos"], axis=-1) <= sim["config"]["radius_m"] + 1e-6))
    rep["ok"] = (not rep["unexpected_nonfinite"] and rep["haps_ue_snr_decreases_with_distance"]
                 and rep["sat_haps_rst_consistent"] and rep.get("sat_ue_rst_consistent", True)
                 and rep["available_implies_visible"] and rep["ue_inside_area"]
                 and (rep["orbit_step_rel_err_max"] is None or rep["orbit_step_rel_err_max"] < 1e-3)
                 and rep.get("sat_ue_snr_vs_distance_max_resid_dB", 0) < 1e-6)
    return rep


def plot(tables, extra, out_dir):
    from matplotlib.figure import Figure  # no pyplot: does not change the caller's backend (Jupyter/Colab)
    sh, su, hu = tables["sat_haps"], tables["sat_ue"], tables["haps_ue"]
    T = extra["times"].size
    t_min = np.arange(T) * (extra["times"][1] - extra["times"][0]) / np.timedelta64(60, "s") if T > 1 else [0]
    fig = Figure(figsize=(12, 11))
    ax = fig.subplots(3, 2)
    ax[0, 0].plot(t_min, np.bincount(sh["t_idx"][sh["available"]], minlength=T), lw=1)
    ax[0, 0].set(title="Satellites available to HAPS (elev ≥ mask)", xlabel="time [min]", ylabel="count")
    top = np.unique(sh["sat_id"][sh["available"]])[:8]
    for s in top:
        m = sh["sat_id"] == s
        ax[0, 1].plot(sh["t_idx"][m] / 60, sh["sat_haps_distance_m"][m] / 1e3, lw=1)
    ax[0, 1].set(title="Satellite–HAPS distance (8 sats)", xlabel="time [min]", ylabel="km")
    for s in top[:4]:
        m = (sh["sat_id"] == s) & sh["available"]
        ax[1, 0].plot(sh["t_idx"][m] / 60, sh["fso_mean_snr_dB"][m], ".", ms=1, label=f"FSO {s}")
        ax[1, 0].plot(sh["t_idx"][m] / 60, sh["rf_snr_dB"][m], "-", lw=1, label=f"RF {s}")
    ax[1, 0].set(title="Sat–HAPS SNR (FSO mean / RF)", xlabel="time [min]", ylabel="dB")
    ax[1, 0].legend(fontsize=6)
    for s in top[:6]:
        m = (sh["sat_id"] == s) & (sh["rst_status"] > 0)
        rr = np.where(sh["rst_status"][m] == RST.SERVING, sh["rst_sat_haps_s"][m], sh["rst_lb_s"][m])
        ax[1, 1].plot(sh["t_idx"][m] / 60, rr, ".", ms=1)
    ax[1, 1].set(title="RST sat–HAPS (censored: lower bound)", xlabel="time [min]", ylabel="s")
    pos = extra["sim"]["pos"]
    for u in range(min(pos.shape[1], 20)):
        c = "tab:red" if extra["sim"]["user_type"][u] == "critical" else "tab:blue"
        ax[2, 0].plot(pos[:, u, 0] / 1e3, pos[:, u, 1] / 1e3, lw=0.7, color=c)
    ax[2, 0].set(title="UE trajectories (red = critical)", xlabel="east [km]", ylabel="north [km]", aspect="equal")
    for typ, c in (("critical", "tab:red"), ("normal", "tab:blue")):
        if su["t_idx"].size:
            m = (su["user_type"] == typ) & su["available"]
            ax[2, 1].hist(su["snr_sat_ue_dB"][m], bins=40, alpha=0.5, color=c, density=True, label=f"sat–UE {typ}")
        m = hu["user_type"] == typ
        ax[2, 1].hist(hu["snr_haps_ue_dB"][m], bins=40, alpha=0.5, density=True, histtype="step", color=c,
                      label=f"HAPS–UE {typ}")
    ax[2, 1].set(title="SNR distribution by user type", xlabel="dB", yscale="log")
    ax[2, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation.png"), dpi=120)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbit", default=DEFAULTS["orbit"])
    ap.add_argument("--start", default=DEFAULTS["start"])
    ap.add_argument("--duration", type=float, default=DEFAULTS["duration_s"])
    ap.add_argument("--n-ue", type=int, default=DEFAULTS["n_ue"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--fmt", choices=["parquet", "csv"], default=DEFAULTS["fmt"])
    ap.add_argument("--chunk", type=float, default=DEFAULTS["chunk_s"],
                    help="seconds per chunk; RAM depends on this, not on --duration (0 = no chunking)")
    ap.add_argument("--out", default="output/run")
    ap.add_argument("--quick", action="store_true", help="5 min, 10 UE smoke test")
    a = ap.parse_args(argv)
    cfg = dict(DEFAULTS, orbit=a.orbit, start=a.start, duration_s=a.duration, n_ue=a.n_ue, seed=a.seed, fmt=a.fmt,
               chunk_s=a.chunk or None)
    if a.quick:
        cfg.update(duration_s=300, n_ue=10)
    _, meta, _ = build(cfg, a.out)
    print(json.dumps(dict(rows=meta["row_counts"], validation=meta["validation"],
                          seconds=meta["build_seconds"]), indent=1, default=str))
    return 0 if meta["validation"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

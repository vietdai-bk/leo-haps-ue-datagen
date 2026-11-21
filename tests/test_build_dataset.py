import json
import os
import sys

import numpy as np
import pytest
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import build_dataset as B  # noqa: E402


def test_end_to_end_small(tmp_path):
    cfg = dict(B.DEFAULTS, orbit=os.path.join(ROOT, "data", "STARLINK_20250810.txt"), duration_s=120, n_ue=5)
    tables, meta, _ = B.build(cfg, str(tmp_path))
    assert meta["validation"]["ok"], meta["validation"]
    for name in ("sat_haps", "sat_ue", "haps_ue", "ue"):
        t = pq.read_table(tmp_path / f"{name}.parquet")
        assert t.num_rows == meta["row_counts"][name] > 0
    md = json.loads((tmp_path / "metadata.json").read_text())
    assert md["seed"] == 0 and len(md["orbit_snapshot"]["sha256"]) == 64
    sh = tables["sat_haps"]
    # FSO only where available; never 0-filled
    assert np.all(np.isnan(sh["fso_capacity_bps"][~sh["available"]]))
    assert np.all(sh["fso_capacity_bps"][sh["available"]] > 0)
    # required power-allocation inputs present
    for k in ("channel_gain", "noise_power_dBm", "bandwidth_hz", "p_max_dBm"):
        assert k in tables["sat_ue"] and k in tables["haps_ue"]
    assert set(tables["haps_ue"]["user_type"]) <= {"critical", "normal"}


def test_reproducible(tmp_path):
    cfg = dict(B.DEFAULTS, orbit=os.path.join(ROOT, "data", "STARLINK_20250810.txt"), duration_s=60, n_ue=3)
    a, _, _ = B.build(cfg, str(tmp_path / "a"))
    b, _, _ = B.build(cfg, str(tmp_path / "b"))
    for name in a:
        for k in a[name]:
            x, y = np.asarray(a[name][k]), np.asarray(b[name][k])
            assert (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)), k


def test_csv_output_round_trip(tmp_path):
    cfg = dict(B.DEFAULTS, orbit=os.path.join(ROOT, "data", "STARLINK_20250810.txt"), duration_s=60, n_ue=3)
    tables, meta, _ = B.build(dict(cfg, fmt="csv"), str(tmp_path))
    for name in ("sat_haps", "sat_ue", "haps_ue", "ue"):
        assert (tmp_path / f"{name}.csv").exists() and not (tmp_path / f"{name}.parquet").exists()
        back = B.read_table(str(tmp_path), name).to_pydict()
        assert set(back) == set(tables[name])
        for k, v in tables[name].items():
            a, b = np.asarray(v), np.asarray(back[k])
            if a.dtype.kind != "M":  # to_pydict returns timestamps as datetime objects
                assert b.dtype != object, (name, k, "NaN read back as None")
            if a.dtype.kind == "f":
                # whole-number floats (e.g. 4e8 Hz) are written as "400000000" and read back as int
                assert b.dtype.kind in "fi" and np.allclose(a, b, equal_nan=True, rtol=1e-12), (name, k)
            elif a.dtype.kind == "M":
                assert np.array_equal(a.astype("datetime64[ms]"), b.astype("datetime64[ms]")), (name, k)
            else:
                assert np.array_equal(a, b.astype(a.dtype)), (name, k)


def _sorted(out_dir, name):
    t = {k: np.asarray(v) for k, v in B.read_table(out_dir, name).to_pydict().items()}
    keys = [t[k] for k in ("ue_id", "sat_id", "t_idx") if k in t]
    o = np.lexsort(keys)
    return {k: v[o] for k, v in t.items()}


def test_chunked_equals_single_run(tmp_path):
    cfg = dict(B.DEFAULTS, orbit=os.path.join(ROOT, "data", "STARLINK_20250810.txt"), duration_s=900, n_ue=3,
               lookahead_s=600)
    _, m1, _ = B.build(dict(cfg, chunk_s=None), str(tmp_path / "one"), verbose=False)
    _, m2, _ = B.build(dict(cfg, chunk_s=200), str(tmp_path / "chunks"), verbose=False)
    assert m1["chunks"] == 1 and m2["chunks"] == 5
    assert m1["validation"]["ok"] and m2["validation"]["ok"]
    assert m1["row_counts"] == m2["row_counts"]
    for name in ("sat_haps", "sat_ue", "haps_ue", "ue"):
        a, b = _sorted(str(tmp_path / "one"), name), _sorted(str(tmp_path / "chunks"), name)
        for k in a:
            if a[k].dtype.kind == "f":
                assert np.array_equal(a[k], b[k], equal_nan=True), (name, k)
            else:
                assert np.array_equal(a[k], b[k]), (name, k)
    # RST rows that are censored only at the global end, never at an internal chunk edge
    sh = _sorted(str(tmp_path / "chunks"), "sat_haps")
    cens = sh["rst_status"] == 2
    assert np.all(sh["t_idx"][cens] + sh["rst_lb_s"][cens] == 900)


def test_too_short_lookahead_is_an_error(tmp_path):
    cfg = dict(B.DEFAULTS, orbit=os.path.join(ROOT, "data", "STARLINK_20250810.txt"), duration_s=600, n_ue=2,
               chunk_s=100, lookahead_s=5)
    with pytest.raises(RuntimeError, match="lookahead"):
        B.build(cfg, str(tmp_path), verbose=False)


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_read_table_t_max(tmp_path, fmt):
    cfg = dict(B.DEFAULTS, orbit=os.path.join(ROOT, "data", "STARLINK_20250810.txt"), duration_s=120, n_ue=2,
               fmt=fmt)
    B.build(cfg, str(tmp_path), verbose=False)
    full = B.read_table(str(tmp_path), "sat_ue")
    part = B.read_table(str(tmp_path), "sat_ue", columns=["t_idx", "snr_sat_ue_dB"], t_max=60)
    t = np.asarray(full.column("t_idx"))
    assert part.num_rows == int((t < 60).sum()) > 0 and part.column_names == ["t_idx", "snr_sat_ue_dB"]

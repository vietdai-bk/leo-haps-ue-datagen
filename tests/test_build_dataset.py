import json
import os
import sys

import numpy as np
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

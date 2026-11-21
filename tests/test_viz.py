import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import build_dataset as B  # noqa: E402
from leo_haps_ue_datagen import viz  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("viz")
    B.build(dict(B.DEFAULTS, orbit=os.path.join(ROOT, "data", "STARLINK_20250810.txt"), duration_s=300, n_ue=5), str(out))
    return viz.load(str(out))


def test_greedy_selection_uses_available_and_switches_only_on_loss(data):
    sh, sel = data["sat_haps"], viz.greedy_selection(data["sat_haps"])
    avail = set(zip(sh["t_idx"][sh["available"]].tolist(), sh["sat_id"][sh["available"]].tolist()))
    assert all((t, s) in avail for t, s in enumerate(sel) if s >= 0)
    for h in viz.handovers(sel):          # a handover only happens when the old satellite is gone
        assert (h, sel[h - 1]) not in avail


def test_animations_render(data, tmp_path):
    sel = viz.greedy_selection(data["sat_haps"])
    for anim in (viz.sky_animation(data, sel, step=60), viz.ue_animation(data, step=60)):
        p = tmp_path / "a.gif"
        viz.save(anim, str(p))
        assert p.stat().st_size > 1000


def test_timeline_and_3d(data):
    sel = viz.greedy_selection(data["sat_haps"])
    fig = viz.timeline(data, sel)
    assert fig.axes
    pytest.importorskip("plotly")
    f3 = viz.orbit_3d(data, sel, step=60)
    assert len(f3.frames) == 5 and f3.data[0].type == "surface"

"""Visualisations of a generated dataset (reads the Parquet or CSV output of build_dataset).

    from leo_haps_ue_datagen import viz
    d = viz.load("output/run")
    sel = viz.greedy_selection(d["sat_haps"])            # which satellite the HAPS uses at each t
    anim = viz.sky_animation(d, sel)                     # polar az/el from HAPS
    viz.show(anim)                                       # inline player (Jupyter/Colab)
    viz.ue_animation(d); viz.orbit_3d(d, sel).show(); viz.timeline(d, sel)

Animations are matplotlib FuncAnimation; viz.save(anim, "x.gif") writes a GIF (Pillow).
"""
import numpy as np

EARTH_R_KM = 6371.0
UE_STATES = ("stationary", "pedestrian", "vehicle")


def load(out_dir, ue_cols=None):
    """Read sat_haps / ue tables as dicts of numpy arrays (sat_ue is large and not needed here)."""
    from leo_haps_ue_datagen.build_dataset import read_table
    rd = lambda n, cols=None: {k: np.asarray(v) for k, v in read_table(out_dir, n, cols).to_pydict().items()}
    sh = rd("sat_haps", ["t_idx", "sat_id", "sat_x", "sat_y", "sat_z", "haps_x", "haps_y", "haps_z",
                         "elev_deg", "az_deg", "available", "rst_sat_haps_s", "rst_status", "rst_lb_s",
                         "fso_capacity_bps"])
    ue = rd("ue", ue_cols or ["t_idx", "ue_id", "user_type", "state", "e", "n"])
    return dict(sat_haps=sh, ue=ue, T=int(sh["t_idx"].max()) + 1)


def greedy_selection(sh):
    """Example HAPS association policy (not part of the dataset): keep the current satellite while it
    is available; on loss pick the available satellite with the largest remaining service time
    (censored rows use their lower bound). Returns sat_id per t (-1 = none)."""
    T = int(sh["t_idx"].max()) + 1
    rst = np.where(sh["rst_status"] == 2, sh["rst_lb_s"], sh["rst_sat_haps_s"])
    av = sh["available"]
    order = np.argsort(sh["t_idx"][av], kind="stable")
    t_av, s_av, r_av = sh["t_idx"][av][order], sh["sat_id"][av][order], rst[av][order]
    bounds = np.searchsorted(t_av, np.arange(T + 1))
    sel, cur = np.full(T, -1, np.int64), -1
    for t in range(T):
        sats, rr = s_av[bounds[t]:bounds[t + 1]], r_av[bounds[t]:bounds[t + 1]]
        if cur not in sats:
            cur = int(sats[np.nanargmax(rr)]) if sats.size else -1
        sel[t] = cur
    return sel


def handovers(sel):
    """Indices t where the selected satellite changes (excluding the first association)."""
    return np.nonzero((sel[1:] != sel[:-1]) & (sel[:-1] >= 0))[0] + 1


def show(anim):
    """Inline player for Jupyter/Colab: H.264 video if ffmpeg exists (small), else a JS frame player."""
    import shutil
    import matplotlib as mpl
    from IPython.display import HTML
    if shutil.which("ffmpeg"):
        return HTML(anim.to_html5_video())
    mpl.rcParams["animation.embed_limit"] = 100  # MB; jshtml embeds every frame as PNG
    return HTML(anim.to_jshtml())


def save(anim, path, fps=10):
    if path.endswith(".gif"):
        from matplotlib.animation import PillowWriter
        anim.save(path, writer=PillowWriter(fps=fps))
    else:
        anim.save(path, fps=fps)  # .mp4 needs ffmpeg (available on Colab)


def _frames(T, step, max_frames):
    step = max(step, int(np.ceil(T / max_frames)))
    return np.arange(0, T, step)


def sky_animation(d, sel=None, step=10, trail_s=120, max_frames=200, elev_mask=30.0):
    """Polar sky plot seen from the HAPS: azimuth (N up, clockwise), radius = 90 - elevation.
    Green = available (elev >= mask), grey = above horizon only; star = satellite selected by `sel`.
    *Below ~20 deg only candidate satellites are in the dataset (see docs), so the grey ring is not complete."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    sh = d["sat_haps"]
    t, az, r, av, sid = sh["t_idx"], np.radians(sh["az_deg"]), 90 - sh["elev_deg"], sh["available"], sh["sat_id"]
    frames = _frames(d["T"], step, max_frames)
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0°"])
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 90 - elev_mask), "g--", lw=0.8, label=f"elev = {elev_mask:g}°")
    trail = ax.scatter([], [], s=2, c="0.75")
    vis = ax.scatter([], [], s=10, c="0.6", label="above horizon (candidates*)")
    use = ax.scatter([], [], s=22, c="tab:green", label="available")
    star = ax.scatter([], [], s=180, marker="*", c="tab:red", zorder=5, label="selected by HAPS")
    title = ax.set_title("")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.12, -0.12), fontsize=8)
    order = np.argsort(t, kind="stable")
    bounds = np.searchsorted(t[order], np.arange(d["T"] + 1))
    rows = lambda a, b: order[bounds[a]:bounds[b]]
    n_ho = np.cumsum(np.isin(np.arange(d["T"]), handovers(sel))) if sel is not None else None

    def update(k):
        now = rows(k, k + 1)
        tr = rows(max(0, k - trail_s), k)
        tr = tr[av[tr] & (t[tr] % 5 == 0)]  # trail every 5 s, available satellites only
        trail.set_offsets(np.c_[az[tr], r[tr]] if tr.size else np.empty((0, 2)))
        vis.set_offsets(np.c_[az[now], r[now]] if now.size else np.empty((0, 2)))
        a = now[av[now]]
        use.set_offsets(np.c_[az[a], r[a]] if a.size else np.empty((0, 2)))
        s = now[sid[now] == sel[k]] if sel is not None else now[:0]
        star.set_offsets(np.c_[az[s], r[s]] if s.size else np.empty((0, 2)))
        txt = f"t = {k} s   available: {a.size}"
        if sel is not None:
            txt += f"   selected: {sel[k] if sel[k] >= 0 else '-'}   handovers: {n_ho[k]}"
        title.set_text(txt)
        return trail, vis, use, star, title

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.close(fig)
    return anim


def ue_animation(d, step=10, trail_s=300, max_frames=200, radius_m=None):
    """UEs moving in the local ENU plane (km). Red = critical, blue = normal;
    marker: o stationary, ^ pedestrian, s vehicle."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    ue = d["ue"]
    t, e, n = ue["t_idx"], ue["e"] / 1e3, ue["n"] / 1e3
    st, crit = ue["state"], ue["user_type"] == "critical"
    R = (radius_m / 1e3) if radius_m else np.ceil(np.hypot(e, n).max())
    frames = _frames(d["T"], step, max_frames)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", color="0.5"))
    ax.plot(0, 0, "k+", ms=12, label="HAPS nadir")
    ax.set(xlim=(-R * 1.05, R * 1.05), ylim=(-R * 1.05, R * 1.05), aspect="equal", xlabel="east [km]", ylabel="north [km]")
    trail = ax.scatter([], [], s=1, c=[], cmap="coolwarm", vmin=0, vmax=1, alpha=0.35)
    marks = {s: ax.scatter([], [], s=40, marker=m, c=[], cmap="coolwarm", vmin=0, vmax=1, edgecolors="k",
                           linewidths=0.4, label=s) for s, m in zip(UE_STATES, ("o", "^", "s"))}
    ax.scatter([], [], c="tab:red", label="critical")
    ax.scatter([], [], c="tab:blue", label="normal")
    ax.legend(loc="upper right", fontsize=8)
    title = ax.set_title("")
    order = np.argsort(t, kind="stable")
    bounds = np.searchsorted(t[order], np.arange(d["T"] + 1))
    rows = lambda a, b: order[bounds[a]:bounds[b]]

    def update(k):
        tr = rows(max(0, k - trail_s), k)[::3]
        trail.set_offsets(np.c_[e[tr], n[tr]] if tr.size else np.empty((0, 2)))
        trail.set_array(crit[tr].astype(float))
        now = rows(k, k + 1)
        for s, sc in marks.items():
            m = now[st[now] == s]
            sc.set_offsets(np.c_[e[m], n[m]] if m.size else np.empty((0, 2)))
            sc.set_array(crit[m].astype(float))
        counts = {s: int((st[now] == s).sum()) for s in UE_STATES}
        title.set_text(f"t = {k} s   " + "  ".join(f"{s}: {c}" for s, c in counts.items()))
        return (trail, title, *marks.values())

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.close(fig)
    return anim


def orbit_3d(d, sel=None, step=20, max_frames=150, only_available=True):
    """Interactive plotly globe: satellite tracks near the HAPS, animated positions, and the HAPS link."""
    import plotly.graph_objects as go
    sh = d["sat_haps"]
    m = sh["available"] if only_available else np.ones(sh["t_idx"].size, bool)
    t, sid = sh["t_idx"][m], sh["sat_id"][m]
    xyz = np.c_[sh["sat_x"][m], sh["sat_y"][m], sh["sat_z"][m]] / 1e3
    haps = np.array([sh["haps_x"][0], sh["haps_y"][0], sh["haps_z"][0]]) / 1e3

    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    earth = go.Surface(x=EARTH_R_KM * np.cos(u) * np.sin(v), y=EARTH_R_KM * np.sin(u) * np.sin(v),
                       z=EARTH_R_KM * np.cos(v), colorscale=[[0, "#1f4e79"], [1, "#4f8fc0"]],
                       showscale=False, opacity=0.9, hoverinfo="skip")
    tracks = []
    for s in np.unique(sid):
        k = np.nonzero(sid == s)[0]
        k = k[np.argsort(t[k])]
        brk = np.r_[False, np.diff(t[k]) > 1]  # separate passes
        x, y, z = (np.insert(xyz[k, i].astype(float), np.nonzero(brk)[0], np.nan) for i in range(3))
        tracks.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=2, color="rgba(200,200,200,0.5)"),
                                   name=str(s), showlegend=False, hoverinfo="name"))
    haps_tr = go.Scatter3d(x=[haps[0]], y=[haps[1]], z=[haps[2]], mode="markers",
                           marker=dict(size=6, color="orange"), name="HAPS")
    frames_t = _frames(d["T"], step, max_frames)

    def frame_data(k):
        i = np.nonzero(t == k)[0]
        sats = go.Scatter3d(x=xyz[i, 0], y=xyz[i, 1], z=xyz[i, 2], mode="markers",
                            marker=dict(size=4, color="limegreen"), text=[str(s) for s in sid[i]], name="satellites")
        if sel is not None and sel[k] >= 0 and np.any(sid[i] == sel[k]):
            p = xyz[i[sid[i] == sel[k]][0]]
            link = go.Scatter3d(x=[haps[0], p[0]], y=[haps[1], p[1]], z=[haps[2], p[2]], mode="lines",
                                line=dict(width=5, color="red"), name=f"HAPS link ({sel[k]})")
        else:
            link = go.Scatter3d(x=[], y=[], z=[], mode="lines", name="HAPS link")
        return [sats, link]

    n_static = 1 + len(tracks) + 1
    fig = go.Figure(data=[earth, *tracks, haps_tr, *frame_data(frames_t[0])],
                    frames=[go.Frame(data=frame_data(k), name=str(k), traces=[n_static, n_static + 1])
                            for k in frames_t])
    c = haps / np.linalg.norm(haps)
    fig.update_layout(
        title="Starlink passes over the HAPS (available, elev ≥ mask)",
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   camera=dict(eye=dict(x=c[0] * 1.2, y=c[1] * 1.2, z=c[2] * 1.2))),
        updatemenus=[dict(type="buttons", x=0.05, y=0.05, buttons=[
            dict(label="▶", method="animate", args=[None, dict(frame=dict(duration=100), fromcurrent=True)]),
            dict(label="❚❚", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(steps=[dict(label=str(k), method="animate",
                                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0))])
                             for k in frames_t], currentvalue=dict(prefix="t [s] = "))],
        margin=dict(l=0, r=0, t=40, b=0), height=650)
    return fig


def timeline(d, sel=None, t0=0, window_s=600, max_sats=40):
    """Satellites available to the HAPS during [t0, t0 + window_s): one row each, bars coloured by mean
    FSO capacity; red = link selected by `sel`, vertical lines = handovers. Shows at most `max_sats` rows:
    the selected satellites first, then those available longest in the window."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from leo_haps_ue_datagen.rst import intervals
    sh = d["sat_haps"]
    T = d["T"]
    av = sh["available"]
    t1 = min(T, t0 + window_s)
    win = av & (sh["t_idx"] >= t0) & (sh["t_idx"] < t1)
    sats, dur = np.unique(sh["sat_id"][win], return_counts=True)
    win_n = sats.size
    chosen = set(sel[t0:t1].tolist()) if sel is not None else set()
    rank = sorted(range(sats.size), key=lambda i: (sats[i] not in chosen, -dur[i]))[:max_sats]
    sats = sats[rank]
    first = {s: sh["t_idx"][win & (sh["sat_id"] == s)].min() for s in sats}
    sats = sorted(sats, key=first.get)
    cap = sh["fso_capacity_bps"] / 1e9
    norm = colors.Normalize(np.nanmin(cap[av]), np.nanmax(cap[av]))
    fig, ax = plt.subplots(figsize=(11, 0.16 * len(sats) + 1.6))
    for row, s in enumerate(sats):
        m = av & (sh["sat_id"] == s)
        mask = np.zeros(T, bool)
        mask[sh["t_idx"][m]] = True
        c = np.zeros(T)
        c[sh["t_idx"][m]] = cap[m]
        for a, b in intervals(mask):
            ax.broken_barh([(a, b - a + 1)], (row - 0.4, 0.8), color=cm.viridis(norm(c[a:b + 1].mean())))
        if sel is not None:
            for a, b in intervals(sel == s):
                ax.broken_barh([(a, b - a + 1)], (row - 0.12, 0.24), color="tab:red")
    if sel is not None:
        for h in handovers(sel):
            if t0 <= h < t1:
                ax.axvline(h, color="k", lw=0.6, alpha=0.7)
    ax.set_yticks(range(len(sats)), [str(s) for s in sats], fontsize=6)
    ax.set(xlim=(t0, t1), ylim=(-1, len(sats)), xlabel="t [s]", ylabel="sat_id (NORAD)",
           title=f"Satellite availability for the HAPS, {len(sats)} of {win_n} satellites (colour = mean FSO capacity, red = selected link)")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="FSO capacity [Gbps]", pad=0.01)
    fig.tight_layout()
    return fig

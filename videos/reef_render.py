"""Render 1920x1080 frames of the Indonesian reef explosion.

Usage: python reef_render.py START STOP [--preview INDEX]

Every coloured pixel of the water is a computed overpressure value on the
signed logarithmic MPa scale of the colour bar (identical to the
repository animations).  The pale veil marks water held at the vapour
pressure (cavitated).  The glowing disc is the gas bubble at its computed
Rayleigh-Plesset radius; the ship is a silhouette at the 5 m keel line
and is not coupled to the water.  The lower strip is the computed
overpressure at the keel depth directly under the ship (r = 40 m).
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "outputs", "cache", "video")
sys.path.insert(0, ROOT)

from legacymine import ocean  # noqa: E402
from legacymine.anim import (BG, FG, SIGNED, draw_bubble,  # noqa: E402
                             draw_seabed, pressure_colorbar, pressure_norm)
from legacymine.bubble import radius_history  # noqa: E402
from legacymine.scenario import CHARGE, H_WATER, Z_KEEL, Z_MINE  # noqa: E402

matplotlib.use("Agg")

OUT = os.path.join(WORK, "reef_png")
NORM = pressure_norm(vmin=-1.0, vmax=30.0, lin=0.02)
TICKS = [-1, -0.1, 0, 0.1, 1, 10, 30]
VEIL = np.array([0.80, 0.92, 1.00, 0.40])
SKY = LinearSegmentedColormap.from_list("sky", ["#0b1526", BG])
HULL = "#cfd6e2"
ACCENT = "#ff8a1f"
MUTED = "#8793a8"

d = np.load(os.path.join(WORK, "reef_frames.npz"))
frames, times = d["frames"], d["times"]
shape = tuple(int(v) for v in d["cav_shape"])
cav = np.unpackbits(d["cav"], axis=-1, count=shape[-1]).astype(bool)
t_keel, p_keel = d["t"], d["keel"] / 1e6
h, R_SHIP = float(d["h"]), float(d["r_ship"])
nz, nr = frames.shape[1:]
R_EDGE, Z_EDGE = nr * h, nz * h
EXT = [-R_EDGE, R_EDGE, Z_EDGE, 0.0]
Z_SKY = -16.0

col = ocean.column(ocean.REGIMES["indo_reef"], np.array([Z_MINE]))
tb, Rb = radius_history(CHARGE.W_eff, float(col["rho"][0]),
                        float(col["p_h"][0]) - 2.3e3)


def ship(ax, xc, L, draft):
    """Side silhouette of a small cargo vessel, keel at ``draft``."""
    x = xc + L * np.array([-0.50, -0.47, -0.40, 0.36, 0.46, 0.515, 0.50,
                           -0.50])
    z = np.array([-3.2, 0.3 * draft, draft, draft, 0.55 * draft, -2.2,
                  -3.6, -3.6])
    ax.add_patch(Polygon(np.c_[x, z], closed=True, fc=HULL, ec="none",
                         zorder=8))
    for x0, x1, z0, z1 in ((-0.42, -0.20, -3.6, -9.2),
                           (-0.40, -0.23, -9.2, -11.6),
                           (-0.33, -0.31, -11.6, -14.6)):
        ax.add_patch(Rectangle((xc + L * x0, z1), L * (x1 - x0), z0 - z1,
                               fc=HULL, ec="none", zorder=8))
    for xh in (-0.05, 0.12, 0.29):
        ax.add_patch(Rectangle((xc + L * xh, -5.4), L * 0.13, 1.8,
                               fc="#9aa6b8", ec="none", zorder=8))
    ax.plot([xc - 0.5 * L, xc + 0.5 * L], [0.0, 0.0], color="#2a3a52",
            lw=0.6, zorder=9)


def render(n, path):
    """Draw frame ``n`` and save it to ``path``."""
    tn = float(times[n])
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0.045, 0.285, 0.91, 0.585])
    ax.set_facecolor(BG)
    ax.imshow(np.linspace(0, 1, 64)[:, None], cmap=SKY, aspect="auto",
              extent=[-R_EDGE, R_EDGE, 0.0, Z_SKY], zorder=0)
    f = frames[n].astype(float)
    full = np.concatenate((f[:, ::-1], f), axis=1)
    im = ax.imshow(full, cmap=SIGNED, norm=NORM, extent=EXT,
                   interpolation="bicubic", aspect="equal", zorder=1)
    c = cav[n]
    veil = np.zeros(c.shape[:1] + (2 * c.shape[1], 4))
    veil[np.concatenate((c[:, ::-1], c), axis=1)] = VEIL
    ax.imshow(veil, extent=EXT, interpolation="bilinear", aspect="equal",
              zorder=2)
    draw_seabed(ax, -R_EDGE, R_EDGE, H_WATER, Z_EDGE, "carbonate", seed=11)
    ax.plot([-R_EDGE, R_EDGE], [0, 0], color="#3b5a80", lw=0.8, zorder=5)
    ax.axhline(Z_KEEL, color=MUTED, lw=0.5, ls=(0, (2, 4)), zorder=5,
               alpha=0.7)
    ship(ax, R_SHIP, 62.0, Z_KEEL)
    draw_bubble(ax, 0.0, Z_MINE, float(np.interp(tn, tb, Rb)))
    ax.set_xlim(-R_EDGE, R_EDGE)
    ax.set_ylim(Z_EDGE, Z_SKY)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xticks(np.arange(-100, 101, 25))
    ax.tick_params(colors=MUTED, labelsize=9, length=3, direction="out")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("depth (m)", color=MUTED, fontsize=10)
    ax.set_xlabel("distance from the charge axis (m)", color=MUTED,
                  fontsize=10, labelpad=2)
    ax.text(3.0, Z_MINE + 6.5, "charge, 3 m above the reef",
            color=FG, fontsize=8.5, alpha=0.85, zorder=9)

    fig.text(0.045, 0.955, "A legacy naval mine on an Indonesian reef",
             color=FG, fontsize=20, va="top")
    fig.text(0.045, 0.912, "300 kg TNT equivalent at 47 m depth, 50 m of "
             "warm tropical water over carbonate rock", color=MUTED,
             fontsize=11, va="top")
    fig.text(0.955, 0.955, f"{1e3 * tn:6.1f} ms", color=FG, fontsize=20,
             va="top", ha="right", family="monospace")
    fig.text(0.955, 0.912, "after detonation", color=MUTED, fontsize=11,
             va="top", ha="right")

    st = fig.add_axes([0.045, 0.075, 0.44, 0.13])
    st.set_facecolor(BG)
    show = t_keel <= tn
    st.axhline(0, color="#2a3346", lw=0.6)
    st.plot(1e3 * t_keel, p_keel, color="#2a3346", lw=0.8)
    st.fill_between(1e3 * t_keel[show], 0, p_keel[show],
                    color=ACCENT, alpha=0.18, lw=0)
    st.plot(1e3 * t_keel[show], p_keel[show], color=ACCENT, lw=1.3)
    k = int(np.searchsorted(t_keel, tn)) - 1
    if k >= 0:
        st.plot([1e3 * t_keel[k]], [p_keel[k]], "o", ms=4, color="#fff4d6")
    st.set_xlim(0, 1e3 * t_keel[-1])
    st.set_ylim(min(-0.3, 1.1 * p_keel.min()), 1.15 * p_keel.max())
    st.tick_params(colors=MUTED, labelsize=8, length=2)
    for s in ("top", "right"):
        st.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        st.spines[s].set_color("#3a4152")
    st.set_xlabel("time (ms)", color=MUTED, fontsize=9, labelpad=1)
    st.set_ylabel("MPa", color=MUTED, fontsize=9)
    st.set_title("overpressure at the keel, 40 m from the axis",
                 color=FG, fontsize=10, loc="left", pad=4)

    cax = fig.add_axes([0.56, 0.155, 0.395, 0.022])
    pressure_colorbar(fig, im, cax, TICKS, "", "horizontal")
    cax.tick_params(colors=MUTED, labelsize=8.5)
    fig.text(0.56, 0.205, "overpressure (MPa), signed logarithmic scale",
             color=FG, fontsize=10)
    fig.text(0.56, 0.085, "pale veil: cavitated water at vapour pressure"
             "     glow: gas bubble at its computed radius",
             color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=120, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--preview" in sys.argv:
        n = int(sys.argv[sys.argv.index("--preview") + 1])
        render(n, os.path.join(WORK, f"preview_{n}.png"))
        print("preview", n, f"t = {1e3 * times[n]:.1f} ms")
    else:
        a, b = int(sys.argv[1]), min(int(sys.argv[2]), len(times))
        for n in range(a, b):
            render(n, os.path.join(OUT, f"f{n:04d}.png"))
        print(f"rendered {a}..{b - 1} of {len(times)}")

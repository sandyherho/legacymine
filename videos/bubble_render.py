"""Render 1920x1080 frames of the pulsating mine bubble on the reef.

Usage: python bubble_render.py START STOP | --preview INDEX | --count

Field.  Pressure radiated by the bubble (far-field monopole, retarded
time) plus its images in the sea surface (reflection -1) and the reef
(normal-incidence coefficient R_0 of the carbonate seabed), up to two
bounces.  The primary detonation shock and growth-phase radiation are
omitted, so the rings are the bubble pulses emitted at each collapse.
Colours follow the same signed logarithmic MPa scale as the repository
animations.  Pale veil: water where the linear field would demand
tension below vapour pressure (cavitation).

Bubble.  Keller-Miksis radius R(t) from bubble_km.npz, drawn to scale as
a sphere clipped by the reef, coloured by its computed gas pressure.

Time.  Frames are spaced non-uniformly: slower around each collapse.
The clock shows real time and the playback rate relative to real time.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "outputs", "cache", "video")
sys.path.insert(0, ROOT)

from legacymine import ocean  # noqa: E402
from legacymine.anim import (BG, FG, SIGNED, draw_seabed,  # noqa: E402
                             pressure_colorbar, pressure_norm)
from legacymine.scenario import H_WATER, Z_KEEL, Z_MINE  # noqa: E402

matplotlib.use("Agg")

OUT = os.path.join(WORK, "bubble_png")
FPS = 30
N_BASE = 190          # frames if time ran uniformly
SLOW = 11.0           # extra frame density around each collapse
WIN = (-0.010, 0.050)  # slow-motion window around a collapse, s
X0, X1, ZT, ZB = -70.0, 70.0, -13.0, 60.0
H = 0.125
X_SHIP, Z_REC = 40.0, Z_KEEL
P_V, P_ATM, G = 3.8e3, 101325.0, 9.81
NORM = pressure_norm(vmin=-1.0, vmax=30.0, lin=0.02)
TICKS = [-1, -0.1, 0, 0.1, 1, 10, 30]
GAS = LinearSegmentedColormap.from_list("gas", [
    (0.0, "#2b0a3d"), (0.25, "#7a1d4a"), (0.5, "#e2572b"),
    (0.72, "#ffb347"), (0.88, "#fff1c9"), (1.0, "#dff6ff")])
GAS_NORM = LogNorm(vmin=0.02, vmax=100.0)
MUTED, ACCENT, HULL = "#8793a8", "#ff8a1f", "#cfd6e2"
SKY = LinearSegmentedColormap.from_list("sky", ["#0b1526", BG])

b = np.load(os.path.join(WORK, "bubble_km.npz"))
T, RAD, RDOT, SRC, PG = b["t"], b["R"], b["Rd"], b["src"], b["pg"]
C, RHO = float(b["c"]), float(b["rho"])
T_MIN = b["t_min"][b["t_min"] < T[-1]]

reg = ocean.REGIMES["indo_reef"]
RB = float(ocean.reflection(reg))
# image sources: (depth, amplitude); surface z = 0, reef z = H_WATER
SOURCES = [(Z_MINE, 1.0), (-Z_MINE, -1.0),
           (2 * H_WATER - Z_MINE, RB), (-(2 * H_WATER - Z_MINE), -RB),
           (2 * H_WATER + Z_MINE, -RB), (-(2 * H_WATER + Z_MINE), RB)]

x = np.arange(X0 + H / 2, X1, H)
z = np.arange(H / 2, H_WATER, H)
XX, ZZ = np.meshgrid(x, z)
DIST = [np.hypot(XX, ZZ - zs) for zs, _ in SOURCES]
P_H = P_ATM + RHO * G * ZZ
INSIDE = None


def field(t):
    """Radiated overpressure (Pa) on the display grid at time t.

    Linear superposition, capped below at the cavitation floor
    p_v - p_h(z): water cannot carry more tension than that.
    """
    p = np.zeros_like(XX)
    for d, (_, a) in zip(DIST, SOURCES):
        p += a * np.interp(t - d / C, T, SRC, left=0.0) / d
    return p


def keel_series():
    """Radiated overpressure at the keel under the ship, Pa."""
    p = np.zeros_like(T)
    for zs, a in SOURCES:
        d = np.hypot(X_SHIP, Z_REC - zs)
        p += a * np.interp(T - d / C, T, SRC, left=0.0) / d
    floor = P_V - (P_ATM + RHO * G * Z_REC)
    return np.maximum(p, floor)


def frame_times():
    """Non-uniform frame clock, dense in a window around each collapse."""
    tt = np.linspace(0.0, T[-1], 200001)
    w = np.ones_like(tt)
    for tm in T_MIN:
        a, b_ = tm + WIN[0], tm + WIN[1]
        ramp = 0.004
        up = 0.5 * (1 + np.tanh((tt - a) / ramp))
        dn = 0.5 * (1 - np.tanh((tt - b_) / ramp))
        w += SLOW * up * dn
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(tt))))
    n = int(round(N_BASE * cum[-1] / T[-1]))
    return np.interp(np.linspace(0.0, cum[-1], n), cum, tt)


TF = frame_times()
KEEL = keel_series() / 1e6


def ship(ax, xc, L, draft):
    """Side silhouette of a small cargo vessel, keel at ``draft``."""
    xs = xc + L * np.array([-0.50, -0.47, -0.40, 0.36, 0.46, 0.515, 0.50,
                            -0.50])
    zs = np.array([-3.2, 0.3 * draft, draft, draft, 0.55 * draft, -2.2,
                   -3.6, -3.6])
    ax.add_patch(Polygon(np.c_[xs, zs], closed=True, fc=HULL, ec="none",
                         zorder=8))
    for x0, x1, z0, z1 in ((-0.42, -0.20, -3.6, -9.2),
                           (-0.40, -0.23, -9.2, -11.6),
                           (-0.33, -0.31, -11.6, -14.6)):
        ax.add_patch(Rectangle((xc + L * x0, z1), L * (x1 - x0), z0 - z1,
                               fc=HULL, ec="none", zorder=8))
    for xh in (-0.05, 0.12, 0.29):
        ax.add_patch(Rectangle((xc + L * xh, -5.4), L * 0.13, 1.8,
                               fc="#9aa6b8", ec="none", zorder=8))


def bubble(ax, R, pg):
    """Shaded sphere of radius R (m) coloured by gas pressure pg (Pa)."""
    base = np.array(GAS(GAS_NORM(np.clip(pg / 1e6, 0.02, 100.0))))
    glow = float(np.clip(GAS_NORM(pg / 1e6), 0.0, 1.0))
    for f, a in ((3.2, 0.05), (2.3, 0.09), (1.6, 0.16)):
        ax.add_patch(Circle((0.0, Z_MINE), R * (1 + (f - 1) * (0.35 + glow)),
                            fc=base[:3], ec="none", alpha=a * (0.4 + glow),
                            zorder=6))
    n = 14
    for k in range(n):
        q = k / (n - 1)
        col = (1 - q) * base[:3] * 0.75 + q * np.minimum(
            1.0, base[:3] * 0.7 + 0.5)
        rr = R * (1.0 - 0.72 * q)
        off = 0.32 * R * q
        ax.add_patch(Circle((-off, Z_MINE - off), rr, fc=col, ec="none",
                            zorder=7))
    ax.add_patch(Circle((0.0, Z_MINE), R, fc="none", ec="#ffffff",
                        lw=0.6, alpha=0.55, zorder=7))


def trail(ax, xs, ys, color, lw=1.6):
    """Line whose opacity fades toward its start."""
    if xs.size < 2:
        return
    seg = np.stack([np.c_[xs[:-1], ys[:-1]], np.c_[xs[1:], ys[1:]]], 1)
    alpha = np.linspace(0.08, 1.0, seg.shape[0]) ** 1.5
    rgba = np.tile(np.array(matplotlib.colors.to_rgba(color)),
                   (seg.shape[0], 1))
    rgba[:, 3] = alpha
    ax.add_collection(LineCollection(seg, colors=rgba, linewidths=lw))


def style(ax, title):
    """Dark small-multiple style."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#3a4152")
    ax.set_title(title, color=FG, fontsize=10, loc="left", pad=4)


def render(n, path):
    """Draw frame ``n`` and save it."""
    tn = float(TF[n])
    k = int(np.searchsorted(T, tn))
    k = min(max(k, 1), T.size - 1)
    R, pg = float(RAD[k]), float(PG[k])
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)

    ax = fig.add_axes([0.035, 0.20, 0.60, 0.585])
    ax.set_facecolor(BG)
    ax.imshow(np.linspace(0, 1, 64)[:, None], cmap=SKY, aspect="auto",
              extent=[X0, X1, 0.0, ZT], zorder=0)
    p_lin = field(tn)
    cav = p_lin < (P_V - P_H)
    p = np.where(cav, P_V - P_H, p_lin)
    im = ax.imshow(p / 1e6, cmap=SIGNED, norm=NORM, aspect="equal",
                   extent=[X0, X1, H_WATER, 0.0], interpolation="bicubic",
                   zorder=1)
    veil = np.zeros(cav.shape + (4,))
    veil[cav] = (0.80, 0.92, 1.00, 0.45)
    ax.imshow(veil, aspect="equal", extent=[X0, X1, H_WATER, 0.0],
              interpolation="bilinear", zorder=2)
    bubble(ax, R, pg)
    draw_seabed(ax, X0, X1, H_WATER, ZB, "carbonate", seed=11)
    ax.plot([X0, X1], [0, 0], color="#3b5a80", lw=0.8, zorder=5)
    ship(ax, X_SHIP, 50.0, Z_KEEL)
    ax.plot([X_SHIP], [Z_REC], "o", ms=3.5, color=ACCENT, zorder=9)
    ax.set_xlim(X0, X1)
    ax.set_ylim(ZB, ZT)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xticks(np.arange(-60, 61, 20))
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("depth (m)", color=MUTED, fontsize=10)
    ax.set_xlabel("distance from the charge axis (m)", color=MUTED,
                  fontsize=10, labelpad=2)
    ax.text(X0 + 2, 57.5, f"bubble radius  {R:5.2f} m", color=FG,
            fontsize=11, family="monospace", zorder=9)

    fig.text(0.035, 0.955, "The breathing bubble of a legacy naval mine",
             color=FG, fontsize=20, va="top")
    fig.text(0.035, 0.912, "300 kg TNT equivalent, 3 m above an "
             "Indonesian carbonate reef at 47 m depth", color=MUTED,
             fontsize=11, va="top")
    dt_f = (TF[min(n + 1, TF.size - 1)] - TF[max(n - 1, 0)]) / 2.0
    rate = dt_f * FPS
    fig.text(0.965, 0.955, f"{tn:6.3f} s", color=FG, fontsize=20,
             va="top", ha="right", family="monospace")
    fig.text(0.965, 0.912, f"after detonation, playback {rate:.3f}"
             "× real time", color=MUTED, fontsize=11, va="top",
             ha="right")

    a1 = fig.add_axes([0.70, 0.645, 0.265, 0.19])
    style(a1, "bubble radius (m)")
    a1.plot(T, RAD, color="#2a3346", lw=1.0)
    past = T <= tn
    a1.plot(T[past], RAD[past], color="#ffd27a", lw=1.4)
    a1.plot([tn], [R], "o", ms=4, color="#fff4d6")
    a1.set_xlim(0, T[-1])
    a1.set_ylim(0, 6.5)
    a1.set_xlabel("time (s)", color=MUTED, fontsize=9, labelpad=1)

    a2 = fig.add_axes([0.70, 0.385, 0.265, 0.19])
    style(a2, "phase portrait: wall speed (m/s) against radius (m)")
    a2.plot(RAD, RDOT, color="#2a3346", lw=0.8)
    j0 = max(0, k - 16000)
    trail(a2, RAD[j0:k + 1:20], RDOT[j0:k + 1:20], "#ffd27a")
    a2.plot([R], [RDOT[k]], "o", ms=4, color="#fff4d6")
    a2.set_yscale("symlog", linthresh=10.0)
    a2.set_xlim(0, 6.5)
    a2.set_ylim(-200, 200)
    a2.set_yticks([-100, -10, 0, 10, 100])
    a2.set_yticklabels(["-100", "-10", "0", "10", "100"])

    a3 = fig.add_axes([0.70, 0.13, 0.265, 0.19])
    style(a3, "pulse pressure at the keel, 40 m from the axis (MPa)")
    a3.axhline(0, color="#2a3346", lw=0.6)
    a3.plot(T, KEEL, color="#2a3346", lw=0.8)
    a3.plot(T[past], KEEL[past], color=ACCENT, lw=1.2)
    a3.set_xlim(0, T[-1])
    lo, hi = KEEL.min(), KEEL.max()
    a3.set_ylim(1.15 * lo, 1.15 * hi)
    a3.set_xlabel("time (s)", color=MUTED, fontsize=9, labelpad=1)

    cax = fig.add_axes([0.035, 0.085, 0.28, 0.018])
    pressure_colorbar(fig, im, cax, TICKS, "", "horizontal")
    cax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.035, 0.115, "radiated overpressure (MPa)", color=FG,
             fontsize=9.5)
    sm = matplotlib.cm.ScalarMappable(norm=GAS_NORM, cmap=GAS)
    gax = fig.add_axes([0.355, 0.085, 0.28, 0.018])
    gb = fig.colorbar(sm, cax=gax, orientation="horizontal")
    gb.set_ticks([0.1, 1, 10, 100])
    gb.set_ticklabels(["0.1", "1", "10", "100"])
    gb.outline.set_edgecolor("#3a4152")
    gax.tick_params(colors=MUTED, labelsize=8, length=2)
    fig.text(0.355, 0.115, "gas pressure inside the bubble (MPa)",
             color=FG, fontsize=9.5)
    fig.text(0.035, 0.03, "Keller-Miksis bubble with surface and reef "
             "reflections. Primary shock not shown. Pale veil: water "
             "at the cavitation limit. Idealized model; spherical "
             "bubble clipped by the reef.", color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=120, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--count" in sys.argv:
        print(TF.size, "frames,", TF.size / FPS, "s")
        print("keel pulse range (MPa):", KEEL.min(), KEEL.max())
    elif "--preview" in sys.argv:
        n = int(sys.argv[sys.argv.index("--preview") + 1])
        render(n, os.path.join(WORK, f"bprev_{n}.png"))
        print("preview", n, f"t = {TF[n]:.4f} s")
    else:
        a, b_ = int(sys.argv[1]), min(int(sys.argv[2]), TF.size)
        for n in range(a, b_):
            render(n, os.path.join(OUT, f"f{n:04d}.png"))
        print(f"rendered {a}..{b_ - 1} of {TF.size}")

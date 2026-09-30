"""Animation 2. Six seas, one charge.

The overpressure field of the reference charge in all six regimes, shown
over the half-plane 0 <= r <= 112 m with the charge on the left edge.
Colour scale, cavitation veil and bubble as in animation 1, identical in
every panel and frame.  The regimes differ only in their water column
and seabed.  Requires the caches written by fig02_fields.py.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from legacymine import ocean
from legacymine.anim import (SIGNED, dark, draw_bubble, draw_seabed,
                             fig_to_rgb, pressure_colorbar, pressure_norm,
                             write_gif)
from legacymine.bubble import radius_history
from legacymine.experiment import simulate, unpack_cav
from legacymine.scenario import CHARGE, DECIM, DX, H_WATER, Z_KEEL, Z_MINE

NORM = pressure_norm(vmin=-1.0, vmax=30.0, lin=0.02)
TICKS = [-1, -0.1, 0, 0.1, 1, 10, 30]
VEIL = np.array([0.80, 0.92, 1.00, 0.42])

runs = {k: simulate(k, "surface") for k in ocean.ORDER}
cavs = {k: unpack_cav(runs[k]) for k in ocean.ORDER}
rads = {}
for k in ocean.ORDER:
    col = ocean.column(ocean.REGIMES[k], np.array([Z_MINE]))
    tb, Rb = radius_history(CHARGE.W_eff, float(col["rho"][0]),
                            float(col["p_h"][0]) - 2.3e3)
    rads[k] = (tb, Rb)
t_show = runs[ocean.ORDER[0]]["times"][::2]
h = DX * DECIM
nz, nr = runs[ocean.ORDER[0]]["frames"].shape[1:]
R_EDGE, Z_EDGE = nr * h, nz * h
ext = [0.0, R_EDGE, Z_EDGE, 0.0]

dark()
frames = []
for tn in t_show:
    fig = plt.figure(figsize=(8.4, 4.4), dpi=85)
    for q, k in enumerate(ocean.ORDER):
        row, cl = divmod(q, 3)
        ax = fig.add_axes([0.05 + 0.315 * cl, 0.575 - 0.40 * row, 0.30,
                           0.36])
        rn = runs[k]
        m = int(np.argmin(np.abs(rn["times"] - tn)))
        im = ax.imshow(rn["frames"][m].astype(float), cmap=SIGNED,
                       norm=NORM, extent=ext, interpolation="bilinear",
                       aspect="equal")
        veil = np.zeros(cavs[k][m].shape + (4,))
        veil[cavs[k][m]] = VEIL
        ax.imshow(veil, extent=ext, interpolation="nearest",
                  aspect="equal", zorder=2)
        reg = ocean.REGIMES[k]
        draw_seabed(ax, 0.0, R_EDGE, H_WATER, Z_EDGE, reg.bed, seed=q)
        draw_bubble(ax, 0.0, Z_MINE, float(np.interp(tn, *rads[k])))
        ax.axhline(Z_KEEL, color="#8793a8", lw=0.5, ls=(0, (3, 3)),
                   zorder=5)
        ax.set_xlim(0.0, R_EDGE)
        ax.set_ylim(Z_EDGE, 0.0)
        ax.tick_params(labelsize=6, length=2)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 25, 50])
        if cl:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("depth (m)", fontsize=7)
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$r$ (m)", fontsize=7)
        ax.text(0.97, 0.94, reg.label, transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5)
    fig.text(0.05, 0.99, f"t = {1e3 * tn:5.1f} ms", va="top",
             fontsize=9, family="monospace")
    cax = fig.add_axes([0.30, 0.075, 0.42, 0.028])
    pressure_colorbar(fig, im, cax, TICKS, "overpressure (MPa)",
                      "horizontal")
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 10
print(write_gif("anim02_six_seas", frames, fps=12))

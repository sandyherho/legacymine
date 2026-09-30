"""Animation 1. One legacy charge, two seas.

The overpressure field of the same 300 kg TNT-equivalent charge, 3 m
above the seabed at 47 m depth, in the Baltic in summer (left, soft mud)
and on an Indonesian carbonate reef (right).  The axisymmetric solution
is mirrored about the charge axis.  Colours follow the signed
logarithmic scale on the colour bar, in MPa, identical in both panels and
in every frame.  Pale veils mark cavitated water at the vapour-pressure
floor.  The glowing disc is the gas bubble drawn at its computed
Rayleigh-Plesset radius; the bubble is not part of the acoustic solution,
whose source is the equivalent point monopole.  The ship silhouette
marks the 5 m keel line and is not coupled to the water.
Requires the caches written by fig02_fields.py (or run_all.py).
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from legacymine import ocean
from legacymine.anim import (SIGNED, dark, draw_bubble, draw_seabed,
                             draw_ship, fig_to_rgb, pressure_colorbar,
                             pressure_norm, write_gif)
from legacymine.bubble import radius_history
from legacymine.experiment import simulate, unpack_cav
from legacymine.scenario import CHARGE, DECIM, DX, H_WATER, Z_KEEL, Z_MINE

KEYS = ("baltic_summer", "indo_reef")
NORM = pressure_norm(vmin=-1.0, vmax=30.0, lin=0.02)
TICKS = [-1, -0.1, 0, 0.1, 1, 10, 30]
VEIL = np.array([0.80, 0.92, 1.00, 0.42])


def bubble_radius(key):
    """Callable t -> bubble radius (m) for the regime ``key``."""
    col = ocean.column(ocean.REGIMES[key], np.array([Z_MINE]))
    tb, Rb = radius_history(CHARGE.W_eff, float(col["rho"][0]),
                            float(col["p_h"][0]) - 2.3e3)
    return lambda t: float(np.interp(t, tb, Rb))


runs = {k: simulate(k, "surface") for k in KEYS}
cavs = {k: unpack_cav(runs[k]) for k in KEYS}
rads = {k: bubble_radius(k) for k in KEYS}
t_show = runs[KEYS[0]]["times"][::2]
h = DX * DECIM
nz, nr = runs[KEYS[0]]["frames"].shape[1:]
R_EDGE, Z_EDGE = nr * h, nz * h
ext = [-R_EDGE, R_EDGE, Z_EDGE, 0.0]

dark()
frames = []
for n, tn in enumerate(t_show):
    fig = plt.figure(figsize=(8.0, 6.3), dpi=90)
    axs = [fig.add_axes([0.075, 0.555 - 0.415 * q, 0.9, 0.40])
           for q in range(2)]
    for ax, k in zip(axs, KEYS):
        rn = runs[k]
        m = int(np.argmin(np.abs(rn["times"] - tn)))
        f = rn["frames"][m].astype(float)
        full = np.concatenate((f[:, ::-1], f), axis=1)
        cav = cavs[k][m]
        cfull = np.concatenate((cav[:, ::-1], cav), axis=1)
        im = ax.imshow(full, cmap=SIGNED, norm=NORM, extent=ext,
                       interpolation="bilinear", aspect="equal")
        veil = np.zeros(cfull.shape + (4,))
        veil[cfull] = VEIL
        ax.imshow(veil, extent=ext, interpolation="nearest",
                  aspect="equal", zorder=2)
        reg = ocean.REGIMES[k]
        draw_seabed(ax, -R_EDGE, R_EDGE, H_WATER, Z_EDGE, reg.bed)
        draw_ship(ax, 38.0, 52.0, Z_KEEL)
        draw_bubble(ax, 0.0, Z_MINE, rads[k](tn))
        ax.set_xlim(-R_EDGE, R_EDGE)
        ax.set_ylim(Z_EDGE, -9.0)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.tick_params(labelsize=7, length=2)
        ax.set_ylabel("depth (m)", fontsize=8)
        ax.text(0.015, 0.97, f"{reg.label}  ({reg.bed})",
                transform=ax.transAxes, va="top", fontsize=9)
    axs[0].set_xticklabels([])
    axs[1].set_xlabel(r"distance from the charge axis, $r$ (m)", fontsize=8)
    axs[0].text(0.985, 0.97, f"t = {1e3 * tn:5.1f} ms",
                transform=axs[0].transAxes, va="top", ha="right",
                fontsize=9, family="monospace")
    cax = fig.add_axes([0.25, 0.055, 0.52, 0.022])
    pressure_colorbar(fig, im, cax, TICKS, "overpressure (MPa)",
                      "horizontal")
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 10
print(write_gif("anim01_diptych", frames, fps=12))

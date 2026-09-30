"""Figure 2. Peak compression and cavitation in the six regimes.

Left column: largest overpressure reached at each point over the 105 ms
run, on a logarithmic scale in MPa.  Right column: total time each water
parcel spent at the vapour-pressure floor (cavitated), in ms; episodes
totalling less than 0.1 ms (a few time steps) are not drawn.  Same
charge (300 kg TNT equivalent, eta = 1) 3 m above the seabed at 47 m
depth, same grid; only the water column and seabed change.  The dashed
line marks the keel depth (5 m) and the solid line the seabed.  The
field is axisymmetric about the charge axis at r = 0.  Percentiles of
the cavitated area are written to outputs/data/fig02_cavitation.csv.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from legacymine import ocean
from legacymine.experiment import simulate
from legacymine.io_utils import write_csv
from legacymine.plotting import panel_label, save, setup
from legacymine.scenario import DX, H_WATER, Z_KEEL

RMAX, ZMAX = 110.0, 60.0
TC_MIN = 0.1           # ms; shorter episodes are not counted
setup()
fig, axes = plt.subplots(6, 2, figsize=(7.0, 8.6), sharex=True,
                         sharey=True)
labels = iter("abcdefghijkl")
tab = {"regime_index": np.arange(6), "cav_area_m2": [], "cav_depth_m": [],
       "max_cav_ms": []}
for row, k in enumerate(ocean.ORDER):
    s = simulate(k, "surface")
    r, z = s["r"], s["z"]
    nr = int(RMAX / DX)
    nz = int(ZMAX / DX)
    ext = [0, r[nr - 1] + DX / 2, z[nz - 1] + DX / 2, 0]
    pk = np.clip(s["pmax"][:nz, :nr] / 1e6, 1e-3, None)
    tc = s["tcav"][:nz, :nr] * 1e3
    ax = axes[row, 0]
    im1 = ax.imshow(pk, norm=LogNorm(vmin=0.3, vmax=30.0), cmap="inferno",
                    extent=ext, aspect="auto", interpolation="nearest",
                    rasterized=True)
    ax = axes[row, 1]
    im2 = ax.imshow(np.where(tc >= TC_MIN, tc, np.nan), cmap="viridis",
                    vmin=0, vmax=80, extent=ext, aspect="auto",
                    interpolation="nearest", rasterized=True)
    for ax in axes[row]:
        ax.axhline(H_WATER, color="0.75", lw=0.6)
        ax.axhline(Z_KEEL, color="0.75", lw=0.6, ls="--")
        ax.set_facecolor("0.93")
        panel_label(ax, f"({next(labels)})")
    axes[row, 1].text(1.02, 0.5, ocean.REGIMES[k].label,
                      transform=axes[row, 1].transAxes, rotation=270,
                      va="center", ha="left", fontsize=8)
    cav = tc >= TC_MIN
    tab["cav_area_m2"].append(cav.sum() * DX * DX)
    tab["cav_depth_m"].append(z[:nz][cav.any(axis=1)].max()
                              if cav.any() else 0.0)
    tab["max_cav_ms"].append(tc.max())
for ax in axes[:, 0]:
    ax.set_ylabel("depth (m)")
for ax in axes[-1]:
    ax.set_xlabel(r"$r$ (m)")
axes[0, 0].set_ylim(ZMAX, 0)
fig.tight_layout(h_pad=0.5, w_pad=1.6)
fig.subplots_adjust(bottom=0.10)
cax1 = fig.add_axes([0.10, 0.035, 0.36, 0.012])
cax2 = fig.add_axes([0.56, 0.035, 0.36, 0.012])
cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
cb1.set_ticks([0.3, 1, 3, 10, 30])
cb1.set_ticklabels(["0.3", "1", "3", "10", "30"])
cb1.set_label("peak overpressure (MPa)")
cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
cb2.set_label("time cavitated (ms)")
write_csv("fig02_cavitation", {k: np.asarray(v, dtype=float)
                               for k, v in tab.items()})
for k, a, d, m in zip(ocean.ORDER, tab["cav_area_m2"], tab["cav_depth_m"],
                      tab["max_cav_ms"]):
    print(f"{k:14s} cavitated area {a:8.1f} m2 (half-plane), deepest "
          f"{d:5.2f} m, longest {m:5.1f} ms")
print(save(fig, "fig02_fields", dpi=400))

"""Figure 5. The six idealized regimes.

(a) Temperature, (b) Absolute Salinity, and (c) TEOS-10 sound speed
against depth over the 50 m water column; the seabed sound speed of each
regime is marked at the bottom of panel (c) (an arrow gives the value
for carbonate, which lies off scale).  (d) Seabed image strength
A = (rho_b - rho_w)/(rho_b + rho_w), which sets how strongly a
collapsing bubble is drawn to the bed, and the normal-incidence pressure
reflection coefficient R_0 of the seabed.  Profiles are tabulated in
outputs/data/fig05_profiles.csv and seabed constants in
outputs/data/fig05_seabed.csv.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from legacymine import ocean
from legacymine.io_utils import write_csv
from legacymine.plotting import (REGIME_COLOR, REGIME_STYLE, outside_legend,
                                 panel_label, regime_handles, save, setup)

z = np.linspace(0.0, 50.0, 501)
cols = {k: ocean.column(ocean.REGIMES[k], z) for k in ocean.ORDER}

setup()
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5),
                         gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
data = {"z_m": z}
for k in ocean.ORDER:
    reg, col = ocean.REGIMES[k], cols[k]
    kw = {"color": REGIME_COLOR[k], "ls": REGIME_STYLE[k]}
    axes[0].plot(col["T"], z, **kw)
    axes[1].plot(col["S"], z, **kw)
    axes[2].plot(col["c"], z, **kw)
    _, c_b = ocean.seabed(reg)
    if c_b < 1720.0:
        axes[2].plot([c_b], [49.2], marker="v", ms=4,
                     color=REGIME_COLOR[k])
    else:
        axes[2].annotate(f"$c_b$ = {c_b:.0f}", xy=(1717, 47.0),
                         xytext=(1700, 38.0), fontsize=6.5,
                         color=REGIME_COLOR[k], ha="right",
                         arrowprops={"arrowstyle": "->", "lw": 0.6,
                                     "color": REGIME_COLOR[k]})
    for q in ("T", "S", "rho", "c"):
        data[f"{k}_{q}"] = col[q]
for ax in axes[:3]:
    ax.set_ylim(50, 0)
axes[0].set_xlabel(r"$T$ ($^\circ$C)")
axes[1].set_xlabel(r"$S_A$ (g kg$^{-1}$)")
axes[2].set_xlabel(r"$c$ (m s$^{-1}$)")
axes[2].set_xlim(1400, 1720)
axes[0].set_ylabel("depth (m)")
for ax in axes[1:3]:
    ax.set_yticklabels([])

A = np.array([ocean.image_strength(ocean.REGIMES[k]) for k in ocean.ORDER])
R0 = np.array([ocean.reflection(ocean.REGIMES[k]) for k in ocean.ORDER])
y = np.arange(len(ocean.ORDER))
ax = axes[3]
ax.barh(y + 0.18, A, height=0.34,
        color=[REGIME_COLOR[k] for k in ocean.ORDER])
ax.barh(y - 0.18, R0, height=0.34, color="none",
        edgecolor=[REGIME_COLOR[k] for k in ocean.ORDER], lw=1.0)
ax.set_yticks(y)
ax.set_yticklabels([ocean.REGIMES[k].bed for k in ocean.ORDER])
ax.invert_yaxis()
ax.set_xlim(0, 0.8)
ax.set_xlabel(r"$A$ (filled), $R_0$ (open)", fontsize=7.5)
ax.tick_params(axis="y", length=0)
for a, lab in zip(axes, "abcd"):
    panel_label(a, f"({lab})")
fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.34,
                    wspace=0.30)
axes[3].set_position([0.83, 0.34, 0.16, 0.57])
axes[0].set_position(axes[0].get_position().shrunk(0.9, 1.0))
h, lab = regime_handles(ocean.ORDER,
                        [ocean.REGIMES[k].label for k in ocean.ORDER])
outside_legend(fig, h, lab, ncol=3, y=0.0)

write_csv("fig05_profiles", data)
rows = {"index": y, "rho_b": [], "c_b": [], "A": A, "R0": R0,
        "theta_c_deg": []}
for k in ocean.ORDER:
    reg = ocean.REGIMES[k]
    rb, cb = ocean.seabed(reg)
    rows["rho_b"].append(rb)
    rows["c_b"].append(cb)
    rows["theta_c_deg"].append(ocean.critical_angle(reg))
write_csv("fig05_seabed", {k: np.asarray(v, dtype=float)
                           for k, v in rows.items()})
for k, a, r in zip(ocean.ORDER, A, R0):
    print(f"{k:14s} A = {a:6.3f}  R0 = {r:6.3f}  "
          f"theta_c = {ocean.critical_angle(ocean.REGIMES[k]):6.2f} deg")
print(save(fig, "fig05_regimes"))

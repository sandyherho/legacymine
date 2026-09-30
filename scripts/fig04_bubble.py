"""Figure 4. The gas bubble: size, period, and jet direction.

(a) Maximum radius and (b) first period against charge depth for the
six water columns, fresh charge (solid) and eta = 0.2 (thin); the black
dashed line in (b) is the tabulated TNT period law
T = 2.11 W^(1/3) / (z + 10)^(5/6).  (c, d) Vertical anisotropy
parameter zeta (positive: jet up, toward a ship; negative: jet down,
into the seabed) for a fresh 300 kg charge, against water depth H and
height above the seabed h_b, over Baltic soft mud (c) and Indonesian
carbonate (d).  Black contours mark zeta = 0 for the regime shown; the
coloured contours in (d) mark zeta = 0 for all six seabeds.  Hatched
cells have gamma_b < 1 or gamma_s < 1, where the bubble reaches a
boundary and the leading-order Kelvin-impulse estimate does not apply.
Values are in outputs/data/fig04_*.csv.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from legacymine import ocean
from legacymine.bubble import anisotropy, max_radius, period
from legacymine.io_utils import write_csv
from legacymine.plotting import (REGIME_COLOR, REGIME_STYLE, handles,
                                 panel_label, regime_handles, save, setup)
from legacymine.scenario import CHARGE

P_V = 2.3e3
W = CHARGE.W
z = np.linspace(2.0, 100.0, 197)
Hs = np.linspace(10.0, 100.0, 181)
hb_frac = np.linspace(0.005, 0.995, 199)


def zeta_map(reg):
    """Anisotropy over (h_b / H, H) for regime ``reg``."""
    A = ocean.image_strength(reg)
    out = np.empty((hb_frac.size, Hs.size))
    ok = np.empty_like(out, dtype=bool)
    for j, H in enumerate(Hs):
        hb = hb_frac * H
        zz = H - hb
        col = ocean.column(reg, zz[::-1])
        rho, dp = col["rho"][::-1], col["p_h"][::-1] - P_V
        out[:, j] = anisotropy(W, rho, dp, zz, hb, A)
        Rm = max_radius(W, dp)
        ok[:, j] = (hb > Rm) & (zz > Rm)
    return out, ok


setup()
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5),
                         gridspec_kw={"width_ratios": [1, 1, 1.15, 1.15]})
data = {"z_m": z}
for k in ocean.ORDER:
    col = ocean.column(ocean.REGIMES[k], z)
    dp = col["p_h"] - P_V
    for eta, lw in ((1.0, 1.1), (0.2, 0.6)):
        Rm = max_radius(eta * W, dp)
        T = period(eta * W, col["rho"], dp)
        kw = {"color": REGIME_COLOR[k], "ls": REGIME_STYLE[k], "lw": lw}
        axes[0].plot(Rm, z, **kw)
        axes[1].plot(T, z, **kw)
        data[f"{k}_eta{eta:g}_Rmax_m"] = Rm
        data[f"{k}_eta{eta:g}_T_s"] = T
T_tab = 2.11 * np.cbrt(W) / (z + 10.0) ** (5.0 / 6.0)
axes[1].plot(T_tab, z, color="k", ls="--", lw=0.9)
data["T_tabulated_s"] = T_tab
for ax in axes[:2]:
    ax.set_ylim(100, 0)
axes[0].set_ylabel("charge depth (m)")
axes[0].set_xlabel(r"$R_m$ (m)")
axes[1].set_xlabel(r"$T$ (s)")
axes[1].set_yticklabels([])

maps = {}
for ax, key in ((axes[2], "baltic_summer"), (axes[3], "indo_reef")):
    zt, ok = zeta_map(ocean.REGIMES[key])
    maps[key] = zt
    im = ax.pcolormesh(Hs, hb_frac, zt, cmap="PuOr_r",
                       norm=TwoSlopeNorm(0.0, -0.3, 0.3), shading="auto",
                       rasterized=True)
    ax.contourf(Hs, hb_frac, (~ok).astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["////"])
    ax.contour(Hs, hb_frac, zt, levels=[0.0], colors="k", linewidths=1.0)
    ax.set_xlabel(r"water depth $H$ (m)")
    write_csv(f"fig04_zeta_{key}", {f"H{H:.1f}": zt[:, j]
                                    for j, H in enumerate(Hs)})
for k in ocean.ORDER:
    zt, _ = zeta_map(ocean.REGIMES[k])
    axes[3].contour(Hs, hb_frac, zt, levels=[0.0],
                    colors=[REGIME_COLOR[k]], linewidths=0.8,
                    linestyles=[REGIME_STYLE[k]])
axes[2].set_ylabel(r"$h_b/H$")
axes[3].set_yticklabels([])
for a, lab in zip(axes, "abcd"):
    panel_label(a, f"({lab})")
fig.tight_layout(w_pad=0.6)
fig.subplots_adjust(bottom=0.36)
cax = fig.add_axes([0.60, 0.13, 0.34, 0.025])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label(r"$\zeta$ (up $>0$)")
h, lab = regime_handles(ocean.ORDER,
                        [ocean.REGIMES[k].label for k in ocean.ORDER])
h2, l2 = handles(["TNT period law"], ["k"], ["--"])
fig.legend(h + h2, lab + l2, loc="upper left", bbox_to_anchor=(0.03, 0.2),
           ncol=2, frameon=False, fontsize=7, handlelength=1.8)
write_csv("fig04_size_period", data)
col = ocean.column(ocean.REGIMES["baltic_summer"], np.array([47.0]))
ratio = float(period(W, col["rho"][0], col["p_h"][0] - P_V)
              / (2.11 * np.cbrt(W) / 57.0 ** (5 / 6)))
print(f"Rayleigh / tabulated period at 47 m: {ratio:.4f}")
for k in ("baltic_summer", "indo_reef"):
    j = int(np.argmin(np.abs(Hs - 50.0)))
    zc = maps[k][:, j]
    cross = hb_frac[np.where(np.diff(np.sign(zc)) != 0)[0]] * 50.0
    print(f"{k}: zeta = 0 at h_b = {np.array2string(cross, precision=2)}"
          " m for H = 50 m")
print(save(fig, "fig04_bubble"))

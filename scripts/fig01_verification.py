"""Figure 1. Verification of the solver and of the plate integrator.

(a) Free-field convergence: maximum error of the axisymmetric solver
against the exact spherical solution for a Gaussian pulse at 10 m,
relative to the peak, for four grid spacings.  The exact solution carries
the same Gaussian source smoothing (width 1.5 h), which lengthens the
pulse from tau to sqrt(tau^2 + 2 sigma^2 / c^2).  (b) Lloyd mirror: a
source 8 m below the pressure-release surface, received at r = 12 m,
z = 4 m, against the exact direct-minus-image solution.  (c) Seabed
reflection on the axis for the six seabeds, numerical reflected peak
over R_0 times the image prediction (BS, BW Baltic summer and winter; NS
North Sea; AD Adriatic; JS Java Sea; IR Indonesian reef).  (d) Taylor
plate: exponential integrator against the closed form for the
reference pulse and 12 mm steel.  The cavitation-closure sensitivity to
the water Courant number is written to outputs/data/fig01e_courant.csv
and quoted in the report.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from legacymine import ocean
from legacymine.acoustics import make_grid, run
from legacymine.experiment import profile_of, _source_density
from legacymine.hull import plate_response, taylor_closed
from legacymine.io_utils import load_cache, save_cache, write_csv
from legacymine.plotting import (OKABE_ITO, REGIME_COLOR, outside_legend,
                                 handles, panel_label, save, setup)
from legacymine.scenario import CHARGE, PLATE_M, Z_MINE
from legacymine.source import decay_time, peak_pressure, volume_rate

RHO, C = 1025.0, 1500.0
TAU, T0, RREF = 5.0e-4, 2.5e-3, 10.0
HS = [0.4, 0.2, 0.1, 0.05]


def homogeneous(z):
    """Uniform water column."""
    z = np.asarray(z, dtype=float)
    return (np.full_like(z, RHO), np.full_like(z, C),
            np.full_like(z, ocean.P_ATM))


def Q_gauss(t):
    """Volume rate radiating a unit Gaussian pulse at RREF."""
    k = 4.0 * np.pi * RREF / RHO
    return k * TAU * np.sqrt(np.pi) / 2.0 * (1.0 + erf((t - T0) / TAU))


def p_exact(t, R, h):
    """Exact filtered free-field Gaussian pulse at range R."""
    te = np.sqrt(TAU ** 2 + 2.0 * (1.5 * h) ** 2 / C ** 2)
    return TAU / te * np.exp(-((t - R / C - T0) / te) ** 2) * RREF / R


def compute():
    """Run every verification case and cache the results."""
    out = {}
    errs = []
    for h in HS:
        g = make_grid(homogeneous, h, 16.0, -12.0, 12.0, 1e9, RHO, C,
                      free_surface=False, sponge_w=int(6 / h))
        i, j = g.index(10.0, 0.0)
        r = run(g, Q_gauss, 0.0, 0.0125, cavitation=False,
                receivers=[(i, j)], track_cavitation=False)
        R = np.hypot(g.r[j], g.z[i])
        ex = p_exact(r["t"], R, h)
        errs.append(np.max(np.abs(r["hist"][0] - ex)) / ex.max())
    out["h"], out["err"] = np.array(HS), np.array(errs)
    # Lloyd mirror
    h = 0.1
    g = make_grid(homogeneous, h, 20.0, 0.0, 20.0, 1e9, RHO, C,
                  free_surface=True, sponge_w=60)
    i, j = g.index(12.0, 4.0)
    r = run(g, Q_gauss, 8.0, 0.0200, cavitation=False, receivers=[(i, j)],
            track_cavitation=False)
    R1 = np.hypot(g.r[j], g.z[i] - 8.0)
    R2 = np.hypot(g.r[j], g.z[i] + 8.0)
    out["lm_t"], out["lm_num"] = r["t"], r["hist"][0]
    out["lm_ex"] = p_exact(r["t"], R1, h) - p_exact(r["t"], R2, h)
    # seabed reflection on the axis
    ratios = []
    for k in ocean.ORDER:
        reg = ocean.REGIMES[k]
        rho_b, c_b = reg.rho_b, reg.nu_b * C
        g = make_grid(homogeneous, 0.1, 30.0, -10.0, 34.0, 20.0, rho_b,
                      c_b, free_surface=False, sponge_w=60)
        i, j = g.index(0.0, 5.0)
        r = run(g, Q_gauss, 10.0, 0.0260, cavitation=False,
                receivers=[(i, j)], track_cavitation=False)
        t = r["t"]
        late = t > (5.0 / C + T0 + 6 * TAU)
        R2 = 25.0 + 0.0 * g.r[j]
        R0 = ((rho_b * c_b - RHO * C) / (rho_b * c_b + RHO * C))
        ref = R0 * p_exact(t, R2, 0.1)
        ratios.append(np.max(r["hist"][0][late]) / np.max(ref))
    out["refl"] = np.array(ratios)
    # Courant sensitivity of the cavitating keel peak (reduced domain)
    reg = ocean.REGIMES["baltic_summer"]
    rb, cb = ocean.seabed(reg)
    cfls, pk = [0.6, 0.45, 0.3], []
    for cw in cfls:
        g = make_grid(profile_of(reg), 0.125, 40.0, -12.0, 62.0, 50.0, rb,
                      cb, free_surface=True, sponge_w=60)
        rec = [g.index(0.0, 5.0), g.index(20.0, 5.0)]
        r = run(g, volume_rate(CHARGE, _source_density(reg)), Z_MINE, 0.07,
                receivers=rec, cfl_water=cw)
        pk.append(r["hist"].max(axis=1))
    out["cfl"], out["cfl_pk"] = np.array(cfls), np.array(pk)
    save_cache("verification", **out)
    return load_cache("verification")


v = load_cache("verification")
v = compute() if v is None else v

# Taylor plate: integrator against the closed form
P = float(peak_pressure(CHARGE, 45.0))
th = float(decay_time(CHARGE, 45.0))
Z = RHO * C
tt = np.linspace(0.0, 6 * th, 6001)
pi = P * np.exp(-tt / th)
v_num, t_sep, v_sep = plate_response(tt, pi, PLATE_M, Z)
psi = Z * th / PLATE_M
v_ex = 2 * P * th / PLATE_M * (np.exp(-tt / th)
                               - np.exp(-psi * tt / th)) / (psi - 1)
t_star, v_star = taylor_closed(P, th, PLATE_M, Z)

setup()
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
ax = axes[0]
ax.loglog(v["h"], v["err"], "o-", color=OKABE_ITO["black"], ms=3.5,
          mfc="none")
ax.loglog(v["h"], v["err"][-1] * (v["h"] / v["h"][-1]) ** 2, "--",
          color=OKABE_ITO["grey"], lw=0.9)
ax.set_xticks(HS)
ax.set_xticklabels([f"{x:g}" for x in HS])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel(r"$h$ (m)")
ax.set_ylabel("max relative error")
ax = axes[1]
ms = 1e3 * v["lm_t"]
ax.plot(ms, v["lm_ex"], color=OKABE_ITO["grey"], lw=2.4)
ax.plot(ms, v["lm_num"], color=OKABE_ITO["black"], lw=0.9)
ax.set_xlim(8, 20)
ax.set_xlabel(r"$t$ (ms)")
ax.set_ylabel(r"$p/p_{\mathrm{ref}}$")
ax = axes[2]
y = np.arange(len(ocean.ORDER))
ax.bar(y, v["refl"], color=[REGIME_COLOR[k] for k in ocean.ORDER],
       width=0.7)
ax.axhline(1.0, color=OKABE_ITO["grey"], lw=0.8, ls="--")
ax.set_ylim(0.9, 1.1)
ax.set_xticks(y)
ax.set_xticklabels(["BS", "BW", "NS", "AD", "JS", "IR"], fontsize=7)
ax.set_ylabel("reflected / theory")
ax = axes[3]
ax.plot(tt / th, v_ex * PLATE_M / (2 * P * th), color=OKABE_ITO["grey"],
        lw=2.4)
ax.plot(tt / th, v_num * PLATE_M / (2 * P * th), color=OKABE_ITO["black"],
        lw=0.9)
ax.plot([t_star / th], [v_star * PLATE_M / (2 * P * th)], "o",
        color=OKABE_ITO["vermil"], ms=3.5, mfc="none")
ax.set_xlim(0, 3)
ax.set_xlabel(r"$t/\theta$")
ax.set_ylabel(r"$m v / (2P\theta)$")
for a, lab in zip(axes, "abcd"):
    panel_label(a, f"({lab})")
fig.tight_layout(w_pad=0.6)
hh, ll = handles(["numerical", "exact", r"$h^2$ reference",
                  "closed-form peak"],
                 [OKABE_ITO["black"], OKABE_ITO["grey"], OKABE_ITO["grey"],
                  OKABE_ITO["vermil"]], ["-", "-", "--", "none"],
                 [None, None, None, "o"])
hh[1].set_linewidth(2.4)
outside_legend(fig, hh, ll, y=0.0)

write_csv("fig01a_convergence", {"h_m": v["h"], "max_rel_err": v["err"]})
write_csv("fig01b_lloyd", {"t_s": v["lm_t"], "numerical": v["lm_num"],
                           "exact": v["lm_ex"]})
write_csv("fig01c_reflection", {"regime_index": y, "ratio": v["refl"]})
write_csv("fig01d_taylor", {"t_s": tt, "v_numerical": v_num,
                            "v_exact": v_ex})
write_csv("fig01e_courant", {"cfl_water": v["cfl"],
                             "keel_peak_r0_Pa": v["cfl_pk"][:, 0],
                             "keel_peak_r20_Pa": v["cfl_pk"][:, 1]})
e = v["err"]
print("convergence errors", np.array2string(e, precision=4))
print("observed orders", np.array2string(np.log2(e[:-1] / e[1:]),
                                         precision=2))
print("Lloyd max err / peak",
      np.max(np.abs(v["lm_num"] - v["lm_ex"])) / np.max(np.abs(v["lm_ex"])))
print("reflection ratios", np.array2string(v["refl"], precision=4))
print("Taylor: max |v_num - v_ex| / v* =",
      np.max(np.abs(v_num - v_ex)) / v_star, " t_sep/t* =", t_sep / t_star)
print("Courant keel peaks (MPa)", np.array2string(v["cfl_pk"] / 1e6,
                                                  precision=3))
print(save(fig, "fig01_verification"))

"""Figure 3. Loading along the keel line at 5 m depth.

(a) Peak overpressure in the full field (surface run), with the Cole
similitude P_m(R) for the same charge (black dashed) and the linear
free-field peak of the numerical monopole (grey).  (b) Positive-phase
impulse of the full field, which includes the reloading by cavitation
closure.  (c) Environmental gain: impulse of the incident field
(absorbing surface) over the numerical free-field impulse at the same
slant range.  Values above one are energy returned by the seabed.
(d) Taylor kick, the velocity of a 12 mm air-backed steel plate element
when the water first separates from it, driven by the incident field.
(e) Normalized arrival entropy of the incident energy in 0.5 ms bins over
20 ms from first arrival: 0 for a single arrival, 1 for a uniformly
spread one.  (f) Incident waveforms at r = 60 m.  Receivers closer than
10 m to the sponge are excluded.  Values are in
outputs/data/fig03_keel.csv.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from legacymine import ocean
from legacymine.experiment import freefield, simulate
from legacymine.hull import arrival_entropy, impulse, plate_response
from legacymine.io_utils import write_csv
from legacymine.plotting import (OKABE_ITO, REGIME_COLOR, REGIME_STYLE,
                                 outside_legend, panel_label, handles,
                                 regime_handles, save, setup)
from legacymine.scenario import CHARGE, PLATE_M, Z_KEEL, Z_MINE
from legacymine.source import peak_pressure

R_LIM = 100.0
R_SHOW = 60.0
ff = freefield()
t_ff, p_ff, R_ff = ff["t"], ff["p"], float(ff["R"])
I_ff_ref = float(impulse(t_ff, p_ff)) * R_ff
P_ff_ref = float(p_ff.max()) * R_ff

res = {}
for k in ocean.ORDER:
    s = simulate(k, "surface")
    inc = simulate(k, "incident")
    reg = ocean.REGIMES[k]
    col = ocean.column(reg, np.array([Z_KEEL]))
    Z = float(col["rho"][0] * col["c"][0])
    ph = float(col["p_h"][0])
    r = s["r_keel"]
    sel = r <= R_LIM
    R = np.hypot(r, Z_MINE - Z_KEEL)[sel]
    hs, hi = s["hist"][sel], inc["hist"][sel]
    ts, ti = s["t"], inc["t"]
    kick, ent = [], []
    for p in hi:
        _, _, vs = plate_response(ti, p, PLATE_M, Z, p_h=ph, p_v=2.3e3)
        kick.append(vs)
        t0 = ti[np.argmax(np.abs(p) > 0.01 * np.abs(p).max())]
        ent.append(arrival_entropy(ti, p, t0, 0.020, 40))
    res[k] = {"r": r[sel], "R": R, "peak": hs.max(axis=1),
              "imp": impulse(ts, hs), "imp_inc": impulse(ti, hi),
              "gain": impulse(ti, hi) * R / I_ff_ref,
              "kick": np.array(kick), "entropy": np.array(ent),
              "t": ti, "w60": hi[np.argmin(np.abs(r[sel] - R_SHOW))]}

setup()
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
ax = axes.ravel()
rr = res[ocean.ORDER[0]]["r"]
RR = res[ocean.ORDER[0]]["R"]
ax[0].plot(rr, peak_pressure(CHARGE, RR) / 1e6, color="k", ls="--",
           lw=0.9)
ax[0].plot(rr, P_ff_ref / RR / 1e6, color=OKABE_ITO["grey"], lw=2.2)
for k in ocean.ORDER:
    d = res[k]
    kw = {"color": REGIME_COLOR[k], "ls": REGIME_STYLE[k], "lw": 1.1}
    ax[0].plot(d["r"], d["peak"] / 1e6, **kw)
    ax[1].plot(d["r"], d["imp"] / 1e3, **kw)
    ax[2].plot(d["r"], d["gain"], **kw)
    ax[3].plot(d["r"], d["kick"], **kw)
    ax[4].plot(d["r"], d["entropy"], **kw)
    ax[5].plot(1e3 * (d["t"] - d["t"][np.argmax(d["w60"] > 0.01
                                                * d["w60"].max())]),
               d["w60"] / 1e6, **kw)
ax[0].set_ylabel("peak overpressure (MPa)")
ax[1].set_ylabel(r"impulse (kPa s)")
ax[2].set_ylabel(r"$I_{\mathrm{inc}}/I_{\mathrm{free}}$")
ax[2].axhline(1.0, color=OKABE_ITO["grey"], lw=0.7, ls=":")
ax[3].set_ylabel(r"Taylor kick (m s$^{-1}$)")
ax[4].set_ylabel("arrival entropy")
ax[4].set_ylim(0, 1)
ax[5].set_xlim(-1, 20)
ax[5].set_xlabel(r"$t - t_{\mathrm{arr}}$ (ms)")
ax[5].set_ylabel(r"$p_{\mathrm{inc}}$ (MPa), $r=60$ m")
for a in ax[:5]:
    a.set_xlim(0, R_LIM)
for a in ax[3:5]:
    a.set_xlabel(r"$r$ (m)")
for a, lab in zip(ax, "abcdef"):
    panel_label(a, f"({lab})")
fig.tight_layout(h_pad=1.0, w_pad=0.8)
h, lab = regime_handles(ocean.ORDER,
                        [ocean.REGIMES[k].label for k in ocean.ORDER])
h2, l2 = handles(["similitude", "linear free field"],
                 ["k", OKABE_ITO["grey"]], ["--", "-"])
h2[1].set_linewidth(2.2)
outside_legend(fig, h + h2, lab + l2, ncol=4, y=0.0)

cols = {"r_m": rr, "R_m": RR,
        "P_similitude_Pa": peak_pressure(CHARGE, RR),
        "P_linear_free_Pa": P_ff_ref / RR}
for k in ocean.ORDER:
    for q in ("peak", "imp", "imp_inc", "gain", "kick", "entropy"):
        cols[f"{k}_{q}"] = res[k][q]
write_csv("fig03_keel", cols)
i60 = int(np.argmin(np.abs(rr - R_SHOW)))
for k in ocean.ORDER:
    d = res[k]
    print(f"{k:14s} r=60 m: peak {d['peak'][i60] / 1e6:5.2f} MPa, "
          f"I {d['imp'][i60] / 1e3:5.2f} kPa s, gain {d['gain'][i60]:4.2f},"
          f" kick {d['kick'][i60]:5.2f} m/s, H {d['entropy'][i60]:4.2f}")
print(save(fig, "fig03_keel"))

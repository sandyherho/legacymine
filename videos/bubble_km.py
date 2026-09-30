"""Multi-cycle gas bubble of the Indonesian reef case (Keller-Miksis).

Physics, in SI units throughout.  The bubble wall R(t) obeys the
Keller-Miksis equation, which adds first-order water compressibility
(acoustic radiation damping) to Rayleigh-Plesset:

    (1 - Rd/c) R Rdd + 3/2 (1 - Rd/(3c)) Rd^2
        = (1 + Rd/c) (p_g - dp) / rho + R/(rho c) dp_g/dt,

with dp = p_h(z) - p_v the driving pressure at the charge depth and a
polytropic gas p_g = eps dp (R_m / R)^(3 kappa).  The maximum radius R_m
comes from the TNT bubble-energy coefficient used in the repository.  The
single free constant eps is fixed so that the first period reproduces the
tabulated TNT period law T = 2.11 W^(1/3) / (z + 10)^(5/6).  Energy is
lost only by acoustic radiation; turbulence, jetting and heat loss are
not modelled, so later cycles decay more slowly than in reality.

The growth phase before the first maximum is taken as the mirror image of
the first collapse (lossless approximation), so t = 0 is the detonation.
Radiation is ramped on after about 60 ms: the primary shock and its
immediate afterflow belong to the shock model, not to this one.

Radiated pressure.  The far-field monopole pressure at distance d is

    p(d, t) = rho [R^2 Rdd + 2 R Rd^2](t - d/c) / d,

sampled here on a uniform 5 us grid for the display script.
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "outputs", "cache", "video")
sys.path.insert(0, ROOT)

from legacymine import ocean  # noqa: E402
from legacymine.bubble import max_radius  # noqa: E402
from legacymine.scenario import CHARGE, Z_MINE  # noqa: E402

KAPPA = 1.25          # polytropic exponent of detonation products
P_V = 3.8e3           # vapour pressure of water near 28 C, Pa
N_CYCLES = 4          # bubble cycles integrated after the first maximum
DT_OUT = 5e-6         # output sampling, s

col = ocean.column(ocean.REGIMES["indo_reef"], np.array([Z_MINE]))
RHO = float(col["rho"][0])
C = float(col["c"][0])
DP = float(col["p_h"][0]) - P_V
W = CHARGE.W_eff
RM = float(max_radius(W, DP))
TC = RM * np.sqrt(RHO / DP)
M = np.sqrt(DP / RHO) / C
T_TAB = 2.11 * np.cbrt(W) / (Z_MINE + 10.0) ** (5.0 / 6.0)


def rhs(s, y, eps):
    """Dimensionless Keller-Miksis; r = R/R_m, s = t/t_c."""
    r, v = y
    pg = eps * r ** (-3.0 * KAPPA)
    dpg = -3.0 * KAPPA * pg * v / r
    num = (-1.5 * (1.0 - M * v / 3.0) * v * v
           + (1.0 + M * v) * (pg - 1.0) + M * r * dpg)
    return [v, num / ((1.0 - M * v) * r)]


def integrate(eps, s_end, dense=False):
    """Integrate from the first maximum; return solution and minima."""
    def turn_min(s, y, eps):
        return y[1]
    turn_min.direction = 1.0

    def turn_max(s, y, eps):
        return y[1]
    turn_max.direction = -1.0
    return solve_ivp(rhs, (0.0, s_end), [1.0, 0.0], args=(eps,),
                     method="Radau", rtol=1e-10, atol=1e-12,
                     events=(turn_min, turn_max), dense_output=dense,
                     first_step=1e-8)


def first_collapse(eps):
    """Dimensionless time from maximum to first minimum."""
    sol = integrate(eps, 3.0)
    return float(sol.t_events[0][0])


def calibrate():
    """Gas constant eps giving the tabulated first period."""
    target = T_TAB / (2.0 * TC)
    return brentq(lambda e: first_collapse(e) - target, 1e-4, 0.3,
                  xtol=1e-12)


def solve():
    """Return a dict with the sampled history and radiated source."""
    eps = calibrate()
    sol = integrate(eps, 2.2 * N_CYCLES, dense=True)
    s_c1 = float(sol.t_events[0][0])
    s_end = float(sol.t_events[0][min(N_CYCLES, len(sol.t_events[0])) - 1])
    s_end += 0.35 * s_c1
    t_half = s_c1 * TC                     # detonation to first maximum
    t = np.arange(0.0, t_half + s_end * TC, DT_OUT)
    s = (t - t_half) / TC
    R = np.empty_like(t)
    V = np.empty_like(t)
    grow = s < 0.0
    y = sol.sol(np.abs(s[grow]))           # mirror of the first collapse
    R[grow], V[grow] = y[0], -y[1]
    y = sol.sol(s[~grow])
    R[~grow], V[~grow] = y[0], y[1]
    A = np.array([rhs(0.0, (r, v), eps)[1] for r, v in zip(R, V)])
    Rd, Rdd, Rr = V * RM / TC, A * RM / TC ** 2, R * RM
    src = RHO * (Rr ** 2 * Rdd + 2.0 * Rr * Rd ** 2)   # Pa m
    # the primary shock and the first 60 ms of afterflow are not
    # represented by this model; ramp the radiation on smoothly after them
    src *= 0.5 * (1.0 + np.tanh((t - 0.06) / 0.015))
    pg = eps * DP * R ** (-3.0 * KAPPA)
    t_min = t_half + np.array(sol.t_events[0]) * TC
    keep = sol.t_events[1] > 1e-6          # drop the start, where Rd = 0
    t_max = t_half + np.concatenate(([0.0], sol.t_events[1][keep])) * TC
    r_max = np.concatenate(([1.0], sol.y_events[1][keep, 0])) * RM
    r_min = sol.y_events[0][:, 0] * RM
    return {"t": t, "R": Rr, "Rd": Rd, "src": src, "pg": pg,
            "t_min": t_min, "t_max": t_max, "r_min": r_min,
            "r_max": r_max, "eps": np.array(eps), "rho": np.array(RHO),
            "c": np.array(C), "dp": np.array(DP), "T_tab": np.array(T_TAB)}


if __name__ == "__main__":
    out = solve()
    os.makedirs(WORK, exist_ok=True)
    np.savez_compressed(os.path.join(WORK, "bubble_km.npz"), **out)
    print(f"rho = {RHO:.2f} kg/m3, c = {C:.1f} m/s, dp = {DP / 1e5:.3f}"
          f" bar, R_m = {RM:.3f} m, t_c = {TC * 1e3:.2f} ms, M = {M:.2e}")
    print(f"eps = {float(out['eps']):.5f}, tabulated period "
          f"{T_TAB:.4f} s")
    print("minima t (s):", np.round(out["t_min"], 4),
          " R_min (m):", np.round(out["r_min"], 3))
    print("maxima t (s):", np.round(out["t_max"], 4),
          " R_max (m):", np.round(out["r_max"], 3))
    print("periods (s):", np.round(np.diff(out["t_min"]), 4))
    print(f"peak gas pressure {out['pg'].max() / 1e6:.1f} MPa, "
          f"peak wall speed {np.abs(out['Rd']).max():.1f} m/s")
    for d in (10.0, 40.0):
        print(f"peak radiated pressure at {d:.0f} m: "
              f"{out['src'].max() / d / 1e6:.3f} MPa")
    print("samples", out["t"].size, "duration", out["t"][-1])

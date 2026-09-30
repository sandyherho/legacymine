"""Gas bubble of a legacy charge: size, period, and jet direction.

Size and period.  A bubble holding energy E_b against the ambient
driving pressure dp = p_h(z) - p_v reaches the maximum radius

    R_m = (3 E_b / (4 pi dp))^(1/3),

and an empty Rayleigh cavity of that radius collapses in 0.91468
R_m sqrt(rho/dp), so the first period is T = 1.82936 R_m sqrt(rho/dp).
With E_b = e_b eta W, a fixed bubble energy per kilogram, R_m and T carry
the familiar depth scalings (W / Z)^(1/3) and W^(1/3) Z^(-5/6), where Z is
the depth measured from a level p_atm / (rho g) above the sea surface.
The constant e_b is fixed so that R_m reproduces the TNT coefficient
J = 3.38 m kg^(-1/3) m^(1/3) of the similitude literature in water of
density 1025 kg m^-3; the regime enters through rho(z) and p_h(z).  The
Rayleigh period is smaller than the tabulated TNT period constant by the
factor that the residual gas pressure contributes, which is reported in
the verification output rather than tuned away.

Jet direction.  To leading order in the inverse standoffs, the Kelvin
impulse of a collapsing bubble is the sum of independent contributions
(Blake and Gibson 1987; Supponen et al. 2016), which, expressed through
the anisotropy parameter with positive values pointing up, read

    zeta = rho g R_m / dp  -  0.195 A gamma_b^(-2)  -  0.195 gamma_s^(-2),

with gamma_b = h_b / R_m the standoff from the seabed and gamma_s = z /
R_m that from the free surface.  Buoyancy drives the jet up; the free
surface repels the bubble and the seabed attracts it, both pushing the
jet down.  A = (rho_b - rho_w)/(rho_b + rho_w) is the image strength of a
fluid seabed in incompressible potential flow: A = 1 is a rigid wall and
A = -1 a free surface.  Soft mud with rho_b near rho_w barely attracts
the bubble, carbonate attracts it almost as a wall does.  The expression
is an asymptotic estimate for gamma well above one, not a boundary-
integral solution, and is used only to map where the sign of zeta turns.
"""

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["J_TNT", "E_B", "RAYLEIGH", "max_radius", "period",
           "anisotropy", "radius_history"]

G = 9.81
J_TNT = 3.38                    # m kg^-1/3 m^1/3 (similitude coefficient)
E_B = 4.0 / 3.0 * np.pi * 1025.0 * G * J_TNT ** 3   # J kg^-1
RAYLEIGH = 0.914681             # Rayleigh collapse-time coefficient
ZETA_COEF = 0.195               # Kelvin-impulse coefficient of a wall


def max_radius(W_eff, dp):
    """Maximum bubble radius (m) for effective mass ``W_eff`` (kg)."""
    return np.cbrt(3.0 * E_B * W_eff / (4.0 * np.pi * np.asarray(dp)))


def period(W_eff, rho, dp):
    """First bubble period 2 x Rayleigh collapse time, s."""
    Rm = max_radius(W_eff, dp)
    return 2.0 * RAYLEIGH * Rm * np.sqrt(rho / np.asarray(dp))


def anisotropy(W_eff, rho, dp, z, h_b, A):
    """Vertical anisotropy parameter zeta, positive for an upward jet.

    ``z`` is the depth of the bubble centre and ``h_b`` its height above
    the seabed, both in metres; ``A`` is the seabed image strength.
    """
    Rm = max_radius(W_eff, dp)
    gs = np.asarray(z) / Rm
    gb = np.asarray(h_b) / Rm
    return (rho * G * Rm / dp - ZETA_COEF * A / gb ** 2
            - ZETA_COEF / gs ** 2)


def radius_history(W_eff, rho, dp, gas=0.02, kappa=1.25, n=2000):
    """Radius R(t) over one lossless Rayleigh-Plesset cycle.

    The bubble is started at R_m with zero wall speed and residual gas
    pressure ``gas`` dp, obeying p_g = gas dp (R_m / R)^(3 kappa), and
    integrated to its minimum.  Because the lossless equation is
    reversible, the growth phase is the mirror image of the collapse.
    Returns times (s, starting at the minimum) and radii (m).
    """
    Rm = float(max_radius(W_eff, dp))
    tc = Rm * np.sqrt(rho / dp)

    def rhs(s, y):
        r, v = y
        pg = gas * r ** (-3.0 * kappa)
        return [v, (pg - 1.0 - 1.5 * v * v) / r]

    def bottom(s, y):
        return y[1]
    bottom.direction = 1.0
    bottom.terminal = True
    sol = solve_ivp(rhs, (0.0, 5.0), [1.0, 0.0], method="DOP853",
                    rtol=1e-10, atol=1e-12, events=bottom,
                    dense_output=True, first_step=1e-6)
    s_min = float(sol.t_events[0][0])
    s = np.linspace(0.0, s_min, n)
    r = sol.sol(s)[0]
    t = np.concatenate((s_min - s[::-1], s_min + s[1:]))
    R = np.concatenate((r[::-1], r[1:]))
    return t * tc, R * Rm

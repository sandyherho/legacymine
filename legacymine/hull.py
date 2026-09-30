"""Air-backed hull plating under an incident pulse (Taylor plate).

An air-backed plate of areal mass m, struck at normal incidence by an
incident pressure p_i(t) in water of impedance Z = rho c, moves as

    m dv/dt = 2 p_i(t) - Z v,

the factor 2 being the rigid-wall doubling and -Z v the radiation
loading of the moving plate.  The wet-face pressure p_i + p_r =
2 p_i - Z v falls to the vapour level, and the water separates from the
plate, when 2 p_i - Z v + p_h <= p_v.  For the exponential pulse
p_i = P exp(-t/theta) and with psi = Z theta / m the closed form is

    v(t) = (2 P theta / m) (exp(-t/theta) - exp(-psi t/theta)) / (psi - 1),

peaking at t* = theta ln(psi) / (psi - 1) with
v* = (2 P theta / m) psi^(-psi / (psi - 1)).  Neglecting p_h - p_v, the
net pressure vanishes at t*, so separation and peak coincide (Taylor
1941).  For an arbitrary recorded waveform the equation is advanced by
the exponential integrator, exact for piecewise-constant forcing.

The metrics describe a single rigid plate element with no stiffness or
frame support, which is the classical first estimate of the kick
delivered to shell plating; they are not structural damage predictions.
"""

import numpy as np
from scipy.integrate import trapezoid

__all__ = ["taylor_closed", "plate_response", "peak", "impulse", "efd",
           "arrival_entropy"]


def taylor_closed(P, theta, m, Z):
    """Peak time (s) and peak velocity (m/s) of the Taylor plate."""
    psi = Z * theta / m
    if abs(psi - 1.0) < 1e-12:
        return theta, 2.0 * P * theta / (m * np.e)
    t_star = theta * np.log(psi) / (psi - 1.0)
    v_star = 2.0 * P * theta / m * psi ** (-psi / (psi - 1.0))
    return t_star, v_star


def plate_response(t, p_i, m, Z, p_h=0.0, p_v=0.0):
    """Integrate the plate for a sampled incident waveform.

    Returns (v, t_sep, v_sep) with ``v`` the velocity history, ``t_sep``
    the first time at which the wet-face absolute pressure reaches the
    vapour pressure (NaN if never), and ``v_sep`` the velocity there.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p_i, dtype=float)
    beta = Z / m
    v = np.zeros_like(t)
    t_sep, v_sep = np.nan, np.nan
    for n in range(1, t.size):
        dt = t[n] - t[n - 1]
        e = np.exp(-beta * dt)
        pm = 0.5 * (p[n] + p[n - 1])
        v[n] = v[n - 1] * e + 2.0 * pm / Z * (1.0 - e)
        net = 2.0 * p[n] - Z * v[n] + p_h
        if np.isnan(t_sep) and v[n] > 0.0 and net <= p_v:
            # linear interpolation of the crossing between samples
            net0 = 2.0 * p[n - 1] - Z * v[n - 1] + p_h
            w = (net0 - p_v) / (net0 - net) if net0 != net else 1.0
            t_sep = t[n - 1] + w * dt
            v_sep = v[n - 1] + w * (v[n] - v[n - 1])
    return v, t_sep, v_sep


def peak(p):
    """Largest overpressure along the last axis, Pa."""
    return np.max(p, axis=-1)


def impulse(t, p):
    """Positive-phase impulse int max(p, 0) dt along the last axis, Pa s."""
    return trapezoid(np.clip(p, 0.0, None), t, axis=-1)


def efd(t, p, rho, c):
    """Energy flux density int p^2 / (rho c) dt, J m^-2."""
    return trapezoid(np.asarray(p) ** 2, t, axis=-1) / (rho * c)


def arrival_entropy(t, p, t0, window, n_bins):
    """Return the normalized Shannon entropy of the arrival energy.

    The energy p^2 dt received in [t0, t0 + window] is binned into
    ``n_bins`` equal intervals, q_k is its fraction in bin k, and
    H = -sum q_k ln q_k / ln n_bins.  H = 0 when all energy arrives in
    one bin, H = 1 when it is spread uniformly.
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(p, dtype=float) ** 2
    edges = t0 + np.linspace(0.0, window, n_bins + 1)
    idx = np.digitize(t, edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    E = np.bincount(idx[ok], weights=e[ok], minlength=n_bins)
    q = E / E.sum()
    q = q[q > 0]
    return float(-np.sum(q * np.log(q)) / np.log(n_bins))

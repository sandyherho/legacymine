"""Aged-charge source: similitude pulse and its equivalent monopole.

The incident shock of a TNT-equivalent charge W (kg) at slant range R (m)
is represented by the similitude form attributed to Cole (1948),

    p_i(t) = P_m exp(-t / theta),
    P_m = K_P (W^(1/3) / R)^A_P,   theta = K_T W^(1/3) (W^(1/3) / R)^(-A_T).

A legacy charge whose explosive has partly degraded is described by an
effective yield fraction eta in (0, 1], entering only through W -> eta W.
Every similitude length therefore scales as (eta W)^(1/3): a criterion
met at range R_h by a fresh charge is met at eta^(1/3) R_h by an aged
one, whatever the criterion.

For the wave solver the pulse is represented by a point monopole.  In a
homogeneous liquid a volume source rate Q(t) radiates

    p(r, t) = rho Q'(t - r/c) / (4 pi r),

so the choice Q(t) = (4 pi R_ref / rho) int_0^t p_ref dt reproduces the
reference waveform p_ref exactly at r = R_ref and propagates it with
spherical spreading.  Linear spreading decays as 1/R, the similitude
peak as R^(-1.13); calibrating at a mid-domain R_ref keeps the mismatch
below about 15 percent over the ranges analysed, and the mismatch is
reported rather than hidden.  The front is smoothed by a rise time
t_r, with the waveform

    f(t) = (exp(-t/theta) - exp(-t/t_r)) / n,

normalized so that its maximum is one, which removes the unresolved part
of the spectrum without changing the peak.
"""

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Charge", "peak_pressure", "decay_time", "pulse_norm",
           "pulse", "pulse_integral", "volume_rate", "free_field"]


@dataclass(frozen=True)
class Charge:
    """TNT-equivalent charge, its degradation, and pulse smoothing."""

    W: float = 300.0          # nominal TNT-equivalent mass, kg
    eta: float = 1.0          # effective yield fraction, dimensionless
    R_ref: float = 40.0       # calibration range of the monopole, m
    t_rise: float = 1.5e-4    # rise time of the smoothed front, s
    K_P: float = 52.16e6      # similitude peak constant, Pa
    A_P: float = 1.13         # similitude peak exponent
    K_T: float = 92.5e-6      # similitude decay constant, s kg^-1/3
    A_T: float = 0.22         # similitude decay exponent

    @property
    def W_eff(self):
        """Effective TNT-equivalent mass eta W, kg."""
        return self.eta * self.W

    def with_(self, **kw):
        """Return a copy with the given fields replaced."""
        return replace(self, **kw)


def peak_pressure(ch, R):
    """Similitude peak overpressure at slant range ``R``, Pa."""
    s = np.cbrt(ch.W_eff) / np.asarray(R, dtype=float)
    return ch.K_P * s ** ch.A_P


def decay_time(ch, R):
    """Similitude decay constant at slant range ``R``, s."""
    s = np.cbrt(ch.W_eff) / np.asarray(R, dtype=float)
    return ch.K_T * np.cbrt(ch.W_eff) * s ** (-ch.A_T)


def pulse_norm(theta, t_rise):
    """Time of the maximum and normalizing constant of ``f``."""
    tm = theta * t_rise / (theta - t_rise) * np.log(theta / t_rise)
    return tm, np.exp(-tm / theta) - np.exp(-tm / t_rise)


def pulse(t, P, theta, t_rise):
    """Smoothed reference waveform with maximum ``P``, Pa."""
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    _, n = pulse_norm(theta, t_rise)
    f = (np.exp(-tp / theta) - np.exp(-tp / t_rise)) / n
    return np.where(t >= 0.0, P * f, 0.0)


def pulse_integral(t, P, theta, t_rise):
    """Integrate :func:`pulse` from 0 to ``t``, Pa s."""
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    _, n = pulse_norm(theta, t_rise)
    F = (theta * -np.expm1(-tp / theta)
         - t_rise * -np.expm1(-tp / t_rise)) / n
    return P * F


def volume_rate(ch, rho):
    """Return Q(t), the monopole volume rate (m^3 s^-1) of the charge.

    ``rho`` is the water density at the source.  The returned callable
    radiates the smoothed similitude waveform of range ``ch.R_ref``.
    """
    P = float(peak_pressure(ch, ch.R_ref))
    th = float(decay_time(ch, ch.R_ref))
    k = 4.0 * np.pi * ch.R_ref / rho

    def Q(t):
        return k * pulse_integral(t, P, th, ch.t_rise)
    return Q


def free_field(ch, R, t):
    """Linear free-field waveform of the monopole at range ``R``, Pa.

    ``t`` is measured from the arrival time R / c.
    """
    P = float(peak_pressure(ch, ch.R_ref))
    th = float(decay_time(ch, ch.R_ref))
    return pulse(t, P, th, ch.t_rise) * ch.R_ref / np.asarray(R, float)

"""Idealized ocean regimes: water column state and a fluid seabed.

Each regime is a horizontally uniform column of depth H over a fluid
half-space.  Temperature and salinity follow smooth step profiles,

    T(z) = T_top + (T_bot - T_top) [1 + tanh((z - z_T) / d_T)] / 2,

and likewise for S, with depth z positive downward.  Density and sound
speed are evaluated from TEOS-10 through the ``gsw`` package, with the
salinity input read as Absolute Salinity in g/kg and the temperature as
in-situ temperature.  The composition anomaly of Baltic seawater is
ignored, which changes density by far less than the regime contrasts
studied here.  Hydrostatic pressure is integrated from the TEOS-10
density at the local pressure, so the column is self-consistent.

The seabed is a lossless fluid with density rho_b and sound speed
c_b = nu_b c_w(H), where c_w(H) is the sound speed of the bottom water.
Shear rigidity is neglected, which is a fair leading-order description
of unconsolidated mud and sand and a crude one for carbonate rock; the
head wave in a rock bottom is present but its shear partner is not.

All values are round, illustrative numbers chosen to sit inside the
ranges usually reported for each setting.  They are not observations at
any site and carry no seasonal or interannual information beyond the
single profile that each regime represents.
"""

from dataclasses import dataclass, replace

import gsw
import numpy as np

__all__ = ["Regime", "REGIMES", "ORDER", "G", "P_ATM", "column",
           "hydrostatic", "bottom_water", "seabed", "reflection",
           "image_strength", "critical_angle"]

G = 9.81            # gravitational acceleration, m s^-2
P_ATM = 101325.0    # atmospheric pressure, Pa


def _step(z, top, bot, z_mid, width):
    """Smooth step from ``top`` at the surface to ``bot`` at depth."""
    return top + (bot - top) * 0.5 * (1.0 + np.tanh((z - z_mid) / width))


@dataclass(frozen=True)
class Regime:
    """One idealized water column over a fluid seabed.

    Temperatures in degrees Celsius, salinities in g/kg, depths in
    metres, seabed density in kg m^-3; ``nu_b`` is the seabed-to-bottom-
    water sound-speed ratio.
    """

    key: str
    label: str
    T_top: float
    T_bot: float
    z_T: float
    d_T: float
    S_top: float
    S_bot: float
    z_S: float
    d_S: float
    rho_b: float
    nu_b: float
    bed: str
    depth: float = 50.0

    def T(self, z):
        """In-situ temperature at depth ``z``, degrees Celsius."""
        return _step(np.asarray(z, dtype=float), self.T_top, self.T_bot,
                     self.z_T, self.d_T)

    def S(self, z):
        """Absolute Salinity at depth ``z``, g/kg."""
        return _step(np.asarray(z, dtype=float), self.S_top, self.S_bot,
                     self.z_S, self.d_S)

    def with_(self, **kw):
        """Return a copy with the given fields replaced."""
        return replace(self, **kw)


def column(reg, z):
    """TEOS-10 state of the water column at depths ``z`` (m, ascending).

    Returns a dict with density ``rho`` (kg m^-3), sound speed ``c``
    (m s^-1), absolute hydrostatic pressure ``p_h`` (Pa), temperature
    ``T`` and salinity ``S``.  The hydrostatic integral is evaluated on
    an auxiliary grid of 0.05 m spacing from the surface, with the
    pressure argument of TEOS-10 updated in two fixed-point sweeps.
    """
    z = np.asarray(z, dtype=float)
    zmax = max(float(np.max(z)), 0.0)
    zf = np.linspace(0.0, zmax, max(int(np.ceil(zmax / 0.05)) + 1, 2))
    SA, t = reg.S(zf), reg.T(zf)
    p_dbar = np.zeros_like(zf)
    for _ in range(3):
        CT = gsw.CT_from_t(SA, t, p_dbar)
        rho = gsw.rho(SA, CT, p_dbar)
        dp = 0.5 * (rho[1:] + rho[:-1]) * G * np.diff(zf)
        p_abs = P_ATM + np.concatenate(([0.0], np.cumsum(dp)))
        p_dbar = (p_abs - P_ATM) / 1.0e4
    c = gsw.sound_speed(SA, CT, p_dbar)
    zc = np.clip(z, 0.0, None)
    return {"rho": np.interp(zc, zf, rho), "c": np.interp(zc, zf, c),
            "p_h": np.interp(zc, zf, p_abs), "T": reg.T(zc),
            "S": reg.S(zc)}


def hydrostatic(reg, z):
    """Absolute hydrostatic pressure at depth ``z``, Pa."""
    return column(reg, z)["p_h"]


def bottom_water(reg):
    """Density and sound speed of the water just above the seabed."""
    col = column(reg, np.array([reg.depth]))
    return float(col["rho"][0]), float(col["c"][0])


def seabed(reg):
    """Seabed density (kg m^-3) and sound speed (m s^-1)."""
    _, c_w = bottom_water(reg)
    return reg.rho_b, reg.nu_b * c_w


def reflection(reg):
    """Normal-incidence pressure reflection coefficient of the seabed."""
    rho_w, c_w = bottom_water(reg)
    rho_b, c_b = seabed(reg)
    zw, zb = rho_w * c_w, rho_b * c_b
    return (zb - zw) / (zb + zw)


def image_strength(reg):
    """Incompressible image strength A = (rho_b - rho_w)/(rho_b + rho_w).

    For potential flow near a plane interface between two inviscid
    liquids, a source of strength m in the water has an image of
    strength A m.  A = 1 recovers the rigid wall and A = -1 the free
    surface, so A measures how strongly the seabed attracts a collapsing
    bubble.
    """
    rho_w, _ = bottom_water(reg)
    return (reg.rho_b - rho_w) / (reg.rho_b + rho_w)


def critical_angle(reg):
    """Critical grazing-complement angle in degrees, or NaN if none.

    Returned as the angle from the vertical, arcsin(c_w / c_b).
    """
    _, c_w = bottom_water(reg)
    _, c_b = seabed(reg)
    return float(np.degrees(np.arcsin(c_w / c_b))) if c_b > c_w else np.nan


# Canonical regimes.  Profiles represent a stratified season where one
# exists; the Baltic is given in both summer and winter to isolate the
# thermocline.  Seabed pairs (rho_b, nu_b) are round values inside the
# ranges tabulated for each sediment class.
REGIMES = {
    "baltic_summer": Regime(
        "baltic_summer", "Baltic, summer",
        17.0, 4.0, 20.0, 3.0, 7.0, 7.4, 40.0, 5.0,
        1450.0, 0.99, "soft mud"),
    "baltic_winter": Regime(
        "baltic_winter", "Baltic, winter",
        2.5, 4.0, 25.0, 8.0, 7.0, 7.4, 40.0, 5.0,
        1450.0, 0.99, "soft mud"),
    "north_sea": Regime(
        "north_sea", "North Sea",
        12.0, 10.5, 30.0, 8.0, 34.8, 34.8, 25.0, 5.0,
        1900.0, 1.12, "fine sand"),
    "adriatic": Regime(
        "adriatic", "Adriatic, summer",
        24.0, 14.0, 18.0, 4.0, 38.0, 38.5, 20.0, 6.0,
        1700.0, 1.03, "silt"),
    "java_sea": Regime(
        "java_sea", "Java Sea",
        29.5, 28.5, 35.0, 6.0, 32.0, 32.8, 30.0, 8.0,
        1600.0, 1.01, "silty mud"),
    "indo_reef": Regime(
        "indo_reef", "Indonesian reef",
        29.0, 27.5, 40.0, 6.0, 33.8, 34.2, 35.0, 8.0,
        2400.0, 1.95, "carbonate"),
}

ORDER = ["baltic_summer", "baltic_winter", "north_sea", "adriatic",
         "java_sea", "indo_reef"]

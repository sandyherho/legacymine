"""Axisymmetric acoustics with a bilinear cavitating liquid.

Derivation.  Let s = rho'/rho be the condensation and u the particle
velocity.  Linearized mass and momentum balance with a point volume
source of rate Q(t) at x_s read

    ds/dt = -div u + Q(t) delta(x - x_s),      rho du/dt = -grad p,

with depth z positive downward and cylindrical symmetry about the
vertical axis through the source,

    div u = (1/r) d(r u_r)/dr + du_z/dz.

The liquid is closed by the bilinear law of the cavitating-acoustics
literature (Newton 1978),

    p = max(K s, p_v - p_h(z)),       K = rho c^2,

where p is the overpressure, p_h(z) the absolute hydrostatic pressure
and p_v the vapour pressure.  Water cannot carry an absolute pressure
below p_v; once the incident compression and the surface-reflected
rarefaction together demand one, the pressure is held at p_v while s
keeps evolving, so the cavitated volume, its growth, and its closure
follow from mass conservation alone.  Closure regenerates pressure as a
water-hammer pulse.  In the sediment the law stays linear.

Discretization.  s and p live at cell centres (r_j, z_i) = ((j + 1/2) h,
z_0 + (i + 1/2) h); u_r and u_z live on cell faces.  Spatial derivatives
use the fourth-order staggered stencil

    df/dx|_{k+1/2} = [9/8 (f_{k+1} - f_k) - 1/24 (f_{k+2} - f_{k-1})] / h,

applied to p for the gradient and to r u_r and u_z for the divergence,
with leapfrog time stepping (second order in time).  Symmetry about the
axis is imposed by ghost cells with p and r u_r even in r and u_r = 0 on
r = 0.  The pressure-release surface z = 0 lies on a face and is imposed
by ghost cells with p odd and u_z even in z.  The remaining boundaries
are Cerjan sponges.  Buoyancy 1/rho is averaged arithmetically onto
faces.  The stable step for the fourth-order stencil in two dimensions
is dt <= h / (sqrt(2) (9/8 + 1/24) c_max).
"""

import numpy as np

__all__ = ["Grid", "make_grid", "run", "sponge_profile", "stable_dt", "C1",
           "C2"]

C1, C2 = 9.0 / 8.0, -1.0 / 24.0


class Grid:
    """Material fields and geometry of one axisymmetric domain."""

    def __init__(self, h, r, z, rho, c, p_h, water, free_surface, sponge):
        """Store the fields; see :func:`make_grid`."""
        self.h = h
        self.r = r                  # cell-centre radii, m
        self.z = z                  # cell-centre depths, m
        self.rho = rho              # density, kg m^-3
        self.c = c                  # sound speed, m s^-1
        self.p_h = p_h              # absolute hydrostatic pressure, Pa
        self.water = water          # boolean mask of cavitable cells
        self.free_surface = free_surface
        self.sponge = sponge        # multiplicative damping at centres

    @property
    def shape(self):
        """Number of cells (nz, nr)."""
        return self.rho.shape

    def index(self, r, z):
        """Nearest cell indices (i, j) to radius ``r`` and depth ``z``."""
        j = int(np.clip(np.round(r / self.h - 0.5), 0, self.r.size - 1))
        i = int(np.clip(np.round((z - self.z[0]) / self.h), 0,
                        self.z.size - 1))
        return i, j


def sponge_profile(n, width, strength):
    """Cerjan ramp exp(-(a (w - k))^2) on the last ``width`` cells."""
    f = np.ones(n)
    if width > 0:
        k = np.arange(width)
        f[n - width:] = np.exp(-(strength * (k + 1)) ** 2)
    return f


def make_grid(profile, h, r_max, z_top, z_bottom, z_bed, rho_b, c_b,
              free_surface=True, sponge_w=60, sponge_a=0.012):
    """Build a grid for one regime.

    ``profile(z)`` returns (rho, c, p_h) arrays of the water column at
    depths ``z``.  Cells deeper than ``z_bed`` are seabed.  When
    ``free_surface`` is false the domain starts at ``z_top < 0`` with a
    sponge in the uppermost cells, so that the upper boundary absorbs.
    Sponge width ``sponge_w`` is in cells, and sponges are appended
    beyond ``r_max`` and below ``z_bottom``.
    """
    nr = int(round(r_max / h)) + sponge_w
    z0 = 0.0 if free_surface else z_top - sponge_w * h
    nz = int(round((z_bottom - z0) / h)) + sponge_w
    r = (np.arange(nr) + 0.5) * h
    z = z0 + (np.arange(nz) + 0.5) * h
    rho_w, c_w, p_h = profile(np.clip(z, 0.0, z_bed))
    bed = z >= z_bed
    rho_c = np.where(bed, rho_b, rho_w)
    c_c = np.where(bed, c_b, c_w)
    rho = np.repeat(rho_c[:, None], nr, axis=1)
    c = np.repeat(c_c[:, None], nr, axis=1)
    ph = np.repeat(p_h[:, None], nr, axis=1)
    water = np.repeat(((~bed) & (z > 0.0))[:, None], nr, axis=1)
    fz = sponge_profile(nz, sponge_w, sponge_a)
    if not free_surface:
        fz = fz * sponge_profile(nz, sponge_w, sponge_a)[::-1]
    fr = sponge_profile(nr, sponge_w, sponge_a)
    return Grid(h, r, z, rho, c, ph, water, free_surface,
                fz[:, None] * fr[None, :])


def _source_weights(grid, z_s):
    """Discrete delta at (0, z_s) normalized by the axisymmetric volume."""
    h = grid.h
    sig = 1.5 * h
    i0 = int(np.round((z_s - grid.z[0]) / h - 0.5))
    ii = np.arange(max(i0 - 8, 0), min(i0 + 9, grid.z.size))
    jj = np.arange(0, 9)
    g = np.exp(-(grid.r[jj][None, :] ** 2
                 + (grid.z[ii][:, None] - z_s) ** 2) / (2.0 * sig ** 2))
    vol = 2.0 * np.pi * grid.r[jj][None, :] * h * h
    g /= np.sum(g * vol)
    return (slice(ii[0], ii[-1] + 1), slice(0, 9)), g


def stable_dt(grid, cfl=0.9):
    """Largest stable step times ``cfl`` for the fourth-order stencil."""
    lam = np.sqrt(2.0) * (abs(C1) + abs(C2))
    return cfl * grid.h / (lam * float(grid.c.max()))


def run(grid, Q, z_s, t_end, cfl=0.9, cavitation=True, p_v=2.3e3,
        n_frames=0, decim=2, receivers=(), track_cavitation=True,
        cfl_water=None):
    """Integrate the system and collect diagnostics.

    ``Q`` is the source volume rate (callable, m^3 s^-1) at depth
    ``z_s`` on the axis.  ``receivers`` is a sequence of (i, j) cells
    whose overpressure is recorded at every step.  Returns a dict with
    decimated frames of overpressure (MPa, float32) and of the
    cavitation mask, their times, the running maximum and minimum of
    overpressure, the total cavitated time per cell, receiver histories,
    the time step, and the step count.
    """
    h = grid.h
    nz, nr = grid.shape
    dt = stable_dt(grid, cfl)
    if cfl_water is not None:
        lam = np.sqrt(2.0) * (abs(C1) + abs(C2))
        c_w = float(grid.c[grid.water].max())
        dt = min(dt, cfl_water * h / (lam * c_w))
    nt = int(np.ceil(t_end / dt))
    K = grid.rho * grid.c ** 2
    b = 1.0 / grid.rho
    bR = np.empty((nz, nr + 1))
    bR[:, 1:-1] = 0.5 * (b[:, 1:] + b[:, :-1])
    bR[:, 0], bR[:, -1] = b[:, 0], b[:, -1]
    bZ = np.empty((nz + 1, nr))
    bZ[1:-1] = 0.5 * (b[1:] + b[:-1])
    bZ[0], bZ[-1] = b[0], b[-1]
    r_face = np.arange(nr + 1) * h
    inv_r = 1.0 / grid.r
    floor = np.where(grid.water & cavitation, p_v - grid.p_h, -np.inf)
    damp = grid.sponge
    dampR = np.concatenate((damp, damp[:, -1:]), axis=1)
    dampZ = np.concatenate((damp, damp[-1:]), axis=0)
    fs = grid.free_surface
    (si, sj), g = _source_weights(grid, z_s)

    P = np.zeros((nz + 4, nr + 4))
    p = P[2:-2, 2:-2]
    S = np.zeros((nz, nr))
    UR = np.zeros((nz, nr + 1))
    UZ = np.zeros((nz + 1, nr))
    RUp = np.zeros((nz, nr + 3))
    UZp = np.zeros((nz + 3, nr))
    div = np.empty((nz, nr))
    tmp = np.empty((nz, nr))
    pmax = np.zeros((nz, nr))
    pmin = np.zeros((nz, nr))
    tcav = np.zeros((nz, nr))
    rec = np.asarray(receivers, dtype=int).reshape(-1, 2)
    hist = np.zeros((rec.shape[0], nt))
    every = max(nt // n_frames, 1) if n_frames else 0
    frames, cavs, times = [], [], []
    a1, a2 = C1 / h, C2 / h

    for n in range(nt):
        # ghost cells: pressure release (odd) or absorbing (zero) on top,
        # symmetry (even) on the axis
        if fs:
            P[1, 2:-2] = -P[2, 2:-2]
            P[0, 2:-2] = -P[3, 2:-2]
        P[:, 1] = P[:, 2]
        P[:, 0] = P[:, 3]
        Pi = P[2:-2]
        UR -= dt * bR * (a1 * (Pi[:, 2:nr + 3] - Pi[:, 1:nr + 2])
                         + a2 * (Pi[:, 3:nr + 4] - Pi[:, 0:nr + 1]))
        UR[:, 0] = 0.0
        Pc = P[:, 2:-2]
        UZ -= dt * bZ * (a1 * (Pc[2:nz + 3] - Pc[1:nz + 2])
                         + a2 * (Pc[3:nz + 4] - Pc[0:nz + 1]))
        UR *= dampR
        UZ *= dampZ
        # divergence of (u_r, u_z)
        np.multiply(r_face, UR, out=RUp[:, 1:nr + 2])
        RUp[:, 0] = -RUp[:, 2]
        np.multiply(a1, RUp[:, 2:nr + 2] - RUp[:, 1:nr + 1], out=div)
        div += a2 * (RUp[:, 3:nr + 3] - RUp[:, 0:nr])
        div *= inv_r
        UZp[1:nz + 2] = UZ
        UZp[0] = UZ[1] if fs else 0.0
        np.multiply(a1, UZp[2:nz + 2] - UZp[1:nz + 1], out=tmp)
        tmp += a2 * (UZp[3:nz + 3] - UZp[0:nz])
        div += tmp
        S -= dt * div
        S[si, sj] += dt * Q((n + 0.5) * dt) * g
        S *= damp
        np.multiply(K, S, out=p)
        if cavitation:
            cav = p < floor
            np.maximum(p, floor, out=p)
            if track_cavitation:
                tcav += cav * dt
        np.maximum(pmax, p, out=pmax)
        np.minimum(pmin, p, out=pmin)
        if rec.size:
            hist[:, n] = p[rec[:, 0], rec[:, 1]]
        if every and n % every == 0:
            frames.append((p[::decim, ::decim] * 1e-6).astype(np.float32))
            if cavitation:
                cavs.append(cav[::decim, ::decim].copy())
            times.append((n + 1) * dt)
    return {"frames": np.array(frames), "cav": np.array(cavs),
            "times": np.array(times), "pmax": pmax, "pmin": pmin,
            "tcav": tcav, "hist": hist, "dt": dt, "nt": nt,
            "t": (np.arange(nt) + 1) * dt}

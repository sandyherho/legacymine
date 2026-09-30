"""Simulation driver shared by the figure and animation scripts.

Each regime is run twice with identical source, grid and geometry:

``surface``   pressure-release sea surface with bilinear cavitation.
              This is the physical field in the water column, used for
              maps, cavitation, animations, and the loading metrics on
              the keel line;
``incident``  absorbing upper boundary, no cavitation.  At the keel
              depth this retains the direct and seabed-borne arrivals
              and removes the surface reflection, so it is the
              incident waveform that the Taylor plate requires.

A homogeneous, unbounded ``freefield`` run on the same grid provides
the numerical free-field waveform, carrying exactly the same source
smoothing and grid filtering, against which environmental gains are
measured.  Results are cached in ``outputs/cache``.
"""

import time

import numpy as np

from . import ocean
from .acoustics import make_grid, run
from .io_utils import load_cache, save_cache
from .scenario import (CFL_WATER, CHARGE, DECIM, DX, N_FRAMES, R_KEEL,
                       R_MAX, SPONGE, T_END, Z_BOTTOM, Z_KEEL, Z_MINE)
from .source import volume_rate

__all__ = ["profile_of", "grid_for", "simulate", "unpack_cav", "freefield",
           "R_FF"]

R_FF = 40.0              # range of the free-field reference receiver, m
FRAME_R, FRAME_Z = 112.0, 60.0   # extent of stored animation frames, m


def profile_of(reg):
    """Callable z -> (rho, c, p_h) for the water column of ``reg``."""
    def prof(z):
        col = ocean.column(reg, z)
        return col["rho"], col["c"], col["p_h"]
    return prof


def grid_for(reg, free_surface):
    """Grid of one regime in the reference geometry."""
    rho_b, c_b = ocean.seabed(reg)
    return make_grid(profile_of(reg), DX, R_MAX, -12.0, Z_BOTTOM,
                     reg.depth, rho_b, c_b, free_surface=free_surface,
                     sponge_w=SPONGE)


def _source_density(reg):
    return float(ocean.column(reg, np.array([Z_MINE]))["rho"][0])


def simulate(key, mode, charge=CHARGE, verbose=True):
    """Run (or load) regime ``key`` in mode ``surface`` or ``incident``."""
    stem = f"run_{key}_{mode}"
    cached = load_cache(stem)
    if cached is not None:
        return cached
    reg = ocean.REGIMES[key]
    fs = mode == "surface"
    grid = grid_for(reg, fs)
    rec = [grid.index(r, Z_KEEL) for r in R_KEEL]
    t0 = time.time()
    res = run(grid, volume_rate(charge, _source_density(reg)), Z_MINE,
              T_END, cavitation=fs, n_frames=N_FRAMES if fs else 0,
              decim=DECIM, receivers=rec, track_cavitation=fs,
              cfl_water=CFL_WATER)
    if verbose:
        print(f"{key:14s} {mode:8s} {res['nt']:5d} steps, "
              f"dt = {res['dt']:.3e} s, {time.time() - t0:6.1f} s",
              flush=True)
    i_keel = rec[0][0]
    out = {"t": res["t"].astype(np.float64), "hist": res["hist"],
           "r": grid.r, "z": grid.z, "dt": res["dt"],
           "z_keel": grid.z[i_keel], "r_keel": grid.r[[j for _, j in rec]]}
    if fs:
        nr = int(round(FRAME_R / DX)) // DECIM
        nz = int(round(FRAME_Z / DX)) // DECIM
        out.update({
            "frames": res["frames"][:, :nz, :nr].astype(np.float16),
            "cav": np.packbits(res["cav"][:, :nz, :nr], axis=-1),
            "cav_shape": np.array(res["cav"][:, :nz, :nr].shape),
            "times": res["times"],
            "pmax": res["pmax"].astype(np.float32),
            "pmin": res["pmin"].astype(np.float32),
            "tcav": res["tcav"].astype(np.float32),
            "p_h": grid.p_h[:, 0], "rho": grid.rho[:, 0], "c": grid.c[:, 0]})
    save_cache(stem, **out)
    return load_cache(stem)


def unpack_cav(run_):
    """Boolean cavitation frames from a cached surface run."""
    n = tuple(int(v) for v in run_["cav_shape"])
    return np.unpackbits(run_["cav"], axis=-1, count=n[-1]).astype(bool)


def freefield(charge=CHARGE, verbose=True):
    """Numerical free-field waveform at range ``R_FF`` on the same grid."""
    stem = "run_freefield"
    cached = load_cache(stem)
    if cached is not None:
        return cached
    rho0, c0 = 1025.0, 1500.0

    def prof(z):
        z = np.asarray(z, dtype=float)
        return (np.full_like(z, rho0), np.full_like(z, c0),
                np.full_like(z, ocean.P_ATM))
    grid = make_grid(prof, DX, R_FF + 6.0, -20.0, 20.0, 1e9, rho0, c0,
                     free_surface=False, sponge_w=SPONGE)
    i, j = grid.index(R_FF, 0.0)
    t0 = time.time()
    res = run(grid, volume_rate(charge, rho0), 0.0, R_FF / c0 + 0.02,
              cavitation=False, receivers=[(i, j)], track_cavitation=False,
              cfl_water=CFL_WATER)
    if verbose:
        print(f"{'free field':23s} {res['nt']:5d} steps, "
              f"{time.time() - t0:6.1f} s", flush=True)
    R = float(np.hypot(grid.r[j], grid.z[i]))
    save_cache(stem, t=res["t"], p=res["hist"][0], R=np.array(R),
               c=np.array(c0), rho=np.array(rho0))
    return load_cache(stem)

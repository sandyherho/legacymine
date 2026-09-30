"""Reference configuration shared by every figure and animation.

One idealized geometry is used for every regime so that only the
environment changes between runs: a 50 m water column, a legacy ground
charge 3 m above the seabed on the axis, and a keel line at 5 m depth.
The nominal 300 kg TNT-equivalent mass sits in the range usually quoted
for twentieth-century naval mines; every similitude length scales with
(eta W)^(1/3), so the choice fixes units rather than conclusions.
"""

import numpy as np

from .source import Charge

__all__ = ["CHARGE", "Z_MINE", "Z_KEEL", "H_WATER", "Z_BOTTOM", "DX",
           "R_MAX", "T_END", "SPONGE", "PLATE_M", "R_KEEL", "ETAS",
           "N_FRAMES", "DECIM", "CFL_WATER"]

CHARGE = Charge(W=300.0, eta=1.0, R_ref=40.0, t_rise=1.5e-4)
H_WATER = 50.0          # water depth, m
Z_MINE = 47.0           # charge depth, m (3 m above the seabed)
Z_KEEL = 5.0            # keel depth, m
Z_BOTTOM = 62.0         # bottom of the modelled seabed layer, m
DX = 0.125              # grid spacing, m
R_MAX = 120.0           # horizontal extent before the sponge, m
SPONGE = 60             # sponge width, cells
T_END = 0.105           # simulated time, s
PLATE_M = 7850.0 * 0.012   # areal mass of 12 mm steel plating, kg m^-2
R_KEEL = np.arange(0.0, 110.0 + 1e-9, 1.0)   # keel receivers, m
ETAS = np.array([0.05, 0.1, 0.2, 0.5, 1.0])  # effective yield fractions
N_FRAMES = 140          # animation frames per run
DECIM = 2               # spatial decimation of stored frames
CFL_WATER = 0.45        # Courant cap in the water, all runs

"""Re-run the Indonesian reef case with dense frame output for video.

Uses the unchanged legacymine package and the reference scenario (same
charge, grid, geometry and water Courant cap as the published run); only
the number of stored frames differs, so the video is smoother than the
GIF.  Output: outputs/cache/video/reef_frames.npz.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "outputs", "cache", "video")
sys.path.insert(0, ROOT)

from legacymine import ocean  # noqa: E402
from legacymine.acoustics import run  # noqa: E402
from legacymine.experiment import _source_density, grid_for  # noqa: E402
from legacymine.scenario import (CFL_WATER, CHARGE, DX, T_END,  # noqa: E402
                                 Z_KEEL, Z_MINE)
from legacymine.source import volume_rate  # noqa: E402

KEY = "indo_reef"
N_FRAMES = 300
DECIM = 2
R_SHIP = 40.0          # keel receiver under the ship, m
FRAME_R, FRAME_Z = 112.0, 60.0

reg = ocean.REGIMES[KEY]
grid = grid_for(reg, free_surface=True)
rec = [grid.index(R_SHIP, Z_KEEL)]
t0 = time.time()
res = run(grid, volume_rate(CHARGE, _source_density(reg)), Z_MINE, T_END,
          cavitation=True, n_frames=N_FRAMES, decim=DECIM, receivers=rec,
          track_cavitation=True, cfl_water=CFL_WATER)
print(f"{res['nt']} steps, dt = {res['dt']:.3e} s, "
      f"{time.time() - t0:.1f} s", flush=True)
os.makedirs(WORK, exist_ok=True)
nr = int(round(FRAME_R / DX)) // DECIM
nz = int(round(FRAME_Z / DX)) // DECIM
np.savez_compressed(
    os.path.join(WORK, "reef_frames.npz"),
    frames=res["frames"][:, :nz, :nr].astype(np.float16),
    cav=np.packbits(res["cav"][:, :nz, :nr], axis=-1),
    cav_shape=np.array(res["cav"][:, :nz, :nr].shape),
    times=res["times"], t=res["t"], keel=res["hist"][0],
    h=np.array(DX * DECIM), r_ship=np.array(R_SHIP))
print("saved", flush=True)

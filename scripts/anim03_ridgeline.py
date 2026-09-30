"""Animation 3. What the keel line hears.

Incident overpressure (absorbing surface, so only direct and seabed
arrivals) at 5 m depth, drawn as a ridgeline of traces every 4 m in
range from r = 0 (front) to r = 100 m (back), for the Baltic in summer
(left) and the Indonesian reef (right).  Each trace is revealed in real
time as the recording clock advances; the white bar is 2 MPa.  The
later ridges behind the first arrival over the reef are energy trapped
by the carbonate seabed.  Requires the caches written by fig02_fields.py.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from legacymine import ocean
from legacymine.anim import BG, FG, dark, fig_to_rgb, write_gif
from legacymine.experiment import simulate
from legacymine.plotting import REGIME_COLOR

KEYS = ("baltic_summer", "indo_reef")
STEP = 4
R_LIM = 100.0
LIFT = 0.55          # MPa of vertical offset per trace
N = 90

runs = {k: simulate(k, "incident") for k in KEYS}
T_MAX = 0.098
clock = np.linspace(0.012, T_MAX, N)

dark()
frames = []
for tc in clock:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=85, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13, top=0.92,
                        wspace=0.04)
    for ax, k in zip(axs, KEYS):
        rn = runs[k]
        t = rn["t"]
        idx = np.where(rn["r_keel"] <= R_LIM)[0][::STEP][::-1]
        show = t <= tc
        n_tr = idx.size
        for q, i in enumerate(idx):
            base = (n_tr - 1 - q) * LIFT
            p = rn["hist"][i] / 1e6
            y = base + p
            ax.fill_between(1e3 * t[show], base - 1, y[show], color=BG,
                            zorder=q * 2)
            shade = 0.35 + 0.65 * (q + 1) / n_tr
            ax.plot(1e3 * t[show], y[show], color=REGIME_COLOR[k],
                    lw=0.8, alpha=shade, zorder=q * 2 + 1)
        ax.set_xlim(10, 1e3 * T_MAX)
        ax.set_ylim(-0.8, n_tr * LIFT + 5.0)
        ax.set_xlabel("time after detonation (ms)", fontsize=8)
        ax.tick_params(labelsize=7, length=2)
        ax.set_yticks([])
        ax.text(0.02, 0.97, ocean.REGIMES[k].label, transform=ax.transAxes,
                va="top", fontsize=9)
        ax.plot([13, 13], [n_tr * LIFT + 1.0, n_tr * LIFT + 3.0], color=FG,
                lw=2.0)
        ax.text(14, n_tr * LIFT + 2.0, "2 MPa", va="center", fontsize=7)
    axs[0].set_ylabel(r"range, $r$ = 0 m (front) to 100 m (back)",
                      fontsize=8)
    axs[1].text(0.98, 0.97, f"t = {1e3 * tc:5.1f} ms",
                transform=axs[1].transAxes, ha="right", va="top",
                fontsize=9, family="monospace")
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 12
print(write_gif("anim03_ridgeline", frames, fps=14))

"""Dark style, pressure colour scale, scene elements, and GIF export.

Animations are rendered frame by frame with Matplotlib into RGB arrays and
written as palette GIFs with Pillow.  Every coloured pixel is a computed
model value; the only display nonlinearity is the signed logarithmic
colour scale of :func:`pressure_norm`, which is drawn on every colour bar
with ticks in megapascals, so colours can be read back as numbers.

The colour scale is asymmetric by construction.  Compression near the
charge reaches tens of megapascals, while tension in water is capped at
the local hydrostatic pressure (a few tenths of a megapascal) by
cavitation.  Each sign therefore has its own range, with zero at the
centre of the map and a linear core of half-width ``lin`` around it.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import FuncNorm, LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon
from PIL import Image

matplotlib.use("Agg")

__all__ = ["dark", "SIGNED", "BED_COLOR", "pressure_norm", "fig_to_rgb",
           "write_gif", "ANIMDIR", "draw_seabed", "draw_ship",
           "draw_bubble", "pressure_colorbar", "BG", "FG"]

_HERE = os.path.dirname(os.path.abspath(__file__))
ANIMDIR = os.path.join(os.path.dirname(_HERE), "outputs", "animations")

BG = "#05070d"
FG = "#d9dee8"

# tension: pale cyan to deep blue; zero: background; compression: ember
# red through amber to white
SIGNED = LinearSegmentedColormap.from_list("signed", [
    (0.00, "#e8fbff"), (0.16, "#44d0ff"), (0.36, "#1446a0"),
    (0.50, BG), (0.62, "#6e1206"), (0.78, "#ff6a13"), (0.92, "#ffd27a"),
    (1.00, "#fffaf0"),
])

BED_COLOR = {
    "soft mud": ("#1d1a16", "#3a3329"),
    "silty mud": ("#221d16", "#463b2c"),
    "silt": ("#241f18", "#4b4231"),
    "fine sand": ("#2e281c", "#6a5a3c"),
    "carbonate": ("#2d1e1d", "#7d5b55"),
}


def dark():
    """Apply the animation style."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#3a4152",
        "xtick.color": FG, "ytick.color": FG, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.linewidth": 0.6,
    })


def pressure_norm(vmin=-1.0, vmax=30.0, lin=0.02):
    """Signed logarithmic norm, zero at mid-scale, values in MPa."""
    ln_n = np.log1p(-vmin / lin)
    ln_p = np.log1p(vmax / lin)

    def fwd(x):
        x = np.asarray(x, dtype=float)
        pos = 0.5 + 0.5 * np.log1p(np.clip(x, 0, None) / lin) / ln_p
        neg = 0.5 - 0.5 * np.log1p(np.clip(-x, 0, None) / lin) / ln_n
        return np.where(x >= 0.0, pos, neg)

    def inv(y):
        y = np.asarray(y, dtype=float)
        pos = lin * np.expm1(np.clip(y - 0.5, 0, None) / 0.5 * ln_p)
        neg = -lin * np.expm1(np.clip(0.5 - y, 0, None) / 0.5 * ln_n)
        return np.where(y >= 0.5, pos, neg)
    return FuncNorm((fwd, inv), vmin=vmin, vmax=vmax)


def pressure_colorbar(fig, mappable, cax, ticks, label, orientation):
    """Colour bar with explicit MPa ticks for :func:`pressure_norm`."""
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.set_label(label)
    cb.outline.set_edgecolor("#3a4152")
    cb.ax.tick_params(labelsize=7, length=2)
    return cb


def fig_to_rgb(fig):
    """Rasterise a figure to an (h, w, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def write_gif(stem, frames, fps=18, colors=224):
    """Write a looping, palette-quantised GIF and return its path.

    One palette is computed from a mosaic of evenly spaced frames and
    applied to all frames, so colours do not flicker between frames.
    """
    os.makedirs(ANIMDIR, exist_ok=True)
    path = os.path.join(ANIMDIR, f"{stem}.gif")
    pick = frames[::max(len(frames) // 12, 1)]
    mosaic = Image.fromarray(np.concatenate(pick, axis=0))
    pal = mosaic.quantize(colors=colors, method=Image.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=pal,
                                        dither=Image.Dither.NONE)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(round(1000 / fps)), optimize=True, disposal=2)
    return path


def draw_seabed(ax, x0, x1, z_bed, z_max, bed, seed=3):
    """Speckled seabed from ``z_bed`` to ``z_max`` between x0 and x1."""
    lo, hi = BED_COLOR.get(bed, ("#222222", "#555555"))
    cmap = LinearSegmentedColormap.from_list("bed", [lo, hi])
    rng = np.random.default_rng(seed)
    nx = 400
    nz = max(int(nx * (z_max - z_bed) / (x1 - x0)), 8)
    tex = rng.random((nz, nx)) ** 2.2
    tex = 0.55 * tex + 0.45 * np.linspace(1.0, 0.2, nz)[:, None]
    ax.imshow(tex, cmap=cmap, vmin=0, vmax=1, extent=[x0, x1, z_max, z_bed],
              interpolation="bilinear", zorder=3, aspect="auto")
    ax.plot([x0, x1], [z_bed, z_bed], color=hi, lw=0.8, zorder=4)


def draw_ship(ax, x_c, length, draft, freeboard=3.0, color="#c9d1de"):
    """Side silhouette of a generic displacement hull at the surface."""
    L, d, f = length, draft, freeboard
    xs = x_c + np.array([-0.50, -0.46, -0.30, 0.34, 0.47, 0.52, 0.50,
                         -0.50]) * L
    zs = np.array([-f, 0.55 * d, d, d, 0.6 * d, -0.2 * f, -f, -f])
    ax.add_patch(Polygon(np.c_[xs, zs], closed=True, fc=color, ec="none",
                         zorder=6))
    ax.add_patch(Polygon(np.c_[x_c + np.array([-0.15, 0.10, 0.08, -0.13])
                               * L, [-f, -f, -f - 4.0, -f - 4.0]],
                         closed=True, fc=color, ec="none", zorder=6))


def draw_bubble(ax, x_c, z_c, radius, zorder=7):
    """Glowing gas bubble of the given radius (m) at (x_c, z_c)."""
    arts = []
    for k, (f, a, col) in enumerate(((2.2, 0.06, "#ff7a1a"),
                                     (1.6, 0.12, "#ffab4a"),
                                     (1.2, 0.22, "#ffd9a0"))):
        arts.append(ax.add_patch(Circle((x_c, z_c), f * radius, fc=col,
                                        ec="none", alpha=a,
                                        zorder=zorder + k)))
    arts.append(ax.add_patch(Circle((x_c, z_c), radius, fc="#fff6e6",
                                    ec="#ffffff", lw=0.5, alpha=0.92,
                                    zorder=zorder + 3)))
    return arts

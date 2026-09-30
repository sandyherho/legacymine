"""Figure style, palette, and multi-format export.

Conventions follow the usual physics-journal layout.  Panel labels sit outside
the axes frame, above its top-left corner.  Legends sit below the axes as a
single unframed figure-level legend, so that no curve is ever obscured and no
explanatory text competes with the data.  Figures carry no titles and no
in-panel annotation: numerical values belong in the reports and the manuscript
captions, not inside the axes.

Axis quantities are either dimensionless ratios or SI values with the unit
in the label.  Animations use a separate dark style defined in
:mod:`legacymine.anim`.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")

__all__ = ["OKABE_ITO", "SEQ", "REGIME_COLOR", "REGIME_STYLE", "setup",
           "panel_label", "outside_legend", "handles", "save", "DIVERGING",
           "sci", "regime_handles"]

# Okabe and Ito colorblind-safe qualitative palette
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#8C8C8C",
}

# ordered sequence for families of curves, used consistently across figures
SEQ = [
    OKABE_ITO["black"], OKABE_ITO["blue"], OKABE_ITO["green"],
    OKABE_ITO["orange"], OKABE_ITO["vermil"], OKABE_ITO["purple"],
]

# diverging map for signed pressure: compression red, tension blue
DIVERGING = "RdBu_r"

# one colour per regime, kept identical in every figure and animation;
# cold seas in blues, the temperate control in purple and green, the
# tropical seas in orange and vermilion
REGIME_COLOR = {
    "baltic_summer": OKABE_ITO["blue"],
    "baltic_winter": OKABE_ITO["skyblue"],
    "north_sea": OKABE_ITO["green"],
    "adriatic": OKABE_ITO["purple"],
    "java_sea": OKABE_ITO["orange"],
    "indo_reef": OKABE_ITO["vermil"],
}
REGIME_STYLE = {
    "baltic_summer": "-", "baltic_winter": "--", "north_sea": "-.",
    "adriatic": ":", "java_sea": "-", "indo_reef": "-",
}

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(_HERE), "outputs", "figures")


def setup():
    """Apply the figure style used by every script in this repository."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sci(x):
    """Mathtext for a positive number as m x 10^k, dropping m = 1."""
    k = int(np.floor(np.log10(x) + 1e-9))
    m = x / 10.0 ** k
    if abs(m - 1.0) < 1e-9:
        return rf"10^{{{k}}}"
    return rf"{m:g}\times10^{{{k}}}"


def panel_label(ax, text, dx=0.0, dy=1.03):
    """Panel label outside the axes, above its top-left corner."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")


def handles(labels, colors, styles=None, markers=None):
    """Legend proxies for a family of curves."""
    styles = styles or ["-"] * len(labels)
    markers = markers or [None] * len(labels)
    return ([Line2D([], [], color=c, ls=s, lw=1.3, marker=m, ms=3.8,
                    mfc="none")
             for c, s, m in zip(colors, styles, markers)], list(labels))


def regime_handles(keys, labels):
    """Legend proxies for a set of regimes."""
    return handles(labels, [REGIME_COLOR[k] for k in keys],
                   [REGIME_STYLE[k] for k in keys])


def outside_legend(fig, handles, labels, ncol=None, y=0.0):
    """Single unframed figure-level legend below the axes."""
    ncol = ncol or len(labels)
    return fig.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                      handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
                      borderaxespad=0.0)


def save(fig, stem, dpi=600):
    """Write a vector PDF and a high-resolution PNG.

    The PDF is the submission artifact; fonts are embedded as Type 42 so that
    text remains selectable and editable.  The PNG is rendered at ``dpi`` for
    screen use and for preview in the repository.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    paths = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": dpi})):
        path = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(path, **kw)
        paths.append(path)
    plt.close(fig)
    return paths[0]

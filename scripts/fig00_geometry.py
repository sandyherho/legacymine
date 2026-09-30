"""Figure 0. Model geometry, boundary conditions, and loads.

(a) The axisymmetric domain to scale: a legacy moored mine 3 m above the
seabed on the symmetry axis r = 0, a frigate-type surface combatant whose
keel lies on the 5 m receiver line, the three principal wave paths to
the keel element at r = 40 m (direct, surface-reflected, seabed-
reflected), and the boundary conditions of the solver.  The water column
carries TEOS-10 density and sound speed from the regime profiles; the
seabed is a fluid half-space.  Absorbing sponges lie beyond r = 120 m and
below z = 62 m.  (b) Detail of the mine and its mooring.  (c) Free-body
diagram of the air-backed hull plate element used for the Taylor kick.
All lengths in metres and to scale except the mine and plate details.
The vessel silhouette is generic and illustrative; the ship is not
coupled to the water, so the free surface z = 0 is flat everywhere,
including beneath the hull, where the surface-reflected path turns.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, Polygon,
                                Rectangle, Wedge)

from legacymine.plotting import OKABE_ITO, panel_label, save, setup
from legacymine.scenario import (CHARGE, DX, H_WATER, PLATE_M, R_MAX,
                                 SPONGE, Z_BOTTOM, Z_KEEL, Z_MINE)

WATER_TOP, WATER_BOT = "#eaf4fb", "#c9e0f0"
SEABED, SEABED_EDGE = "#e3d5bb", "#9c8660"
SPONGE_C = "#f4f4f4"
SHIP, MINE = "#2e3440", "#3a3a3a"
INK = "#1f2328"
C_DIR, C_SUR, C_BED = OKABE_ITO["blue"], OKABE_ITO["vermil"], \
    OKABE_ITO["green"]
R_SHIP = 40.0              # keel element range, m
L_SHIP = 90.0              # vessel length, m
W_SPONGE = SPONGE * DX     # sponge width, m


def warship(ax, xc, L, draft, color=SHIP, z=6):
    """Side silhouette of a generic frigate, keel at depth ``draft``.

    Depth is positive downward, so the superstructure has negative z.
    Proportions are generic: raked bow, transom stern, main gun forward,
    bridge and mast amidships, funnel, hangar and flight deck aft.
    """
    def P(pts, **kw):
        pts = np.asarray(pts, dtype=float)
        ax.add_patch(Polygon(np.c_[xc + L * pts[:, 0], pts[:, 1]],
                             closed=True, fc=kw.get("fc", color), ec="none",
                             zorder=kw.get("zorder", z)))

    fb = 6.5                                     # freeboard at the stem
    P([(-0.500, -4.6), (-0.500, 1.2), (-0.470, 3.8), (-0.420, draft),
       (0.300, draft), (0.400, 3.6), (0.470, 0.6), (0.515, -fb),
       (0.380, -5.9), (0.100, -5.2), (-0.500, -4.6)])
    # hangar and flight deck aft
    P([(-0.34, -4.9), (-0.34, -9.6), (-0.12, -10.4), (-0.10, -5.1)])
    # main superstructure and bridge
    P([(-0.10, -5.1), (-0.10, -11.2), (0.05, -11.8), (0.08, -13.6),
       (0.19, -13.6), (0.21, -11.0), (0.22, -5.4)])
    # funnel, raked aft
    P([(-0.07, -11.2), (-0.055, -15.0), (0.015, -15.0), (0.035, -11.5)])
    # mast with yard and radar
    xm = xc + 0.13 * L
    ax.plot([xm, xm + 0.3], [-13.6, -24.0], color=color, lw=1.4,
            solid_capstyle="butt", zorder=z)
    ax.plot([xm - 3.0, xm + 3.2], [-20.5, -20.5], color=color, lw=0.9,
            zorder=z)
    ax.add_patch(Rectangle((xm - 1.6, -18.2), 3.4, 1.1, fc=color,
                           ec="none", zorder=z))
    # main gun: turret and barrel
    P([(0.285, -5.7), (0.285, -7.6), (0.305, -8.3), (0.345, -8.3),
       (0.36, -5.9)])
    ax.plot([xc + 0.35 * L, xc + 0.43 * L], [-7.4, -8.4], color=color,
            lw=1.6, solid_capstyle="round", zorder=z)
    # close-in weapon system on the hangar roof and a forward launcher
    P([(-0.22, -10.0), (-0.22, -11.6), (-0.185, -12.0), (-0.165, -10.2)])
    ax.plot([xc - 0.19 * L, xc - 0.14 * L], [-11.4, -12.0], color=color,
            lw=1.0, zorder=z)
    P([(0.23, -5.5), (0.23, -6.8), (0.265, -6.8), (0.265, -5.6)])


def moored_mine(ax, x0, zc, radius, bed, lw=0.8):
    """Contact-horned moored mine with cable and sinker on the seabed."""
    ax.add_patch(Circle((x0, zc), radius, fc=MINE, ec=INK, lw=lw,
                        zorder=6))
    for ang in (-90, -45, -135, 0, 180):
        a = np.radians(ang)
        x1, z1 = x0 + radius * np.cos(a), zc + radius * np.sin(a)
        x2 = x0 + 1.32 * radius * np.cos(a)
        z2 = zc + 1.32 * radius * np.sin(a)
        ax.plot([x1, x2], [z1, z2], color=MINE, lw=2.2 * lw,
                solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((x0 - 0.3 * radius, zc - 0.3 * radius),
                        0.22 * radius, fc="#8a8f98", ec="none", zorder=7))
    sw, sh = 2.2 * radius, 1.0 * radius
    ax.plot([x0, x0], [zc + radius, bed - sh], color=INK, lw=0.7 * lw,
            zorder=5)
    ax.add_patch(Polygon([(x0 - sw / 2, bed), (x0 + sw / 2, bed),
                          (x0 + 0.35 * sw, bed - sh),
                          (x0 - 0.35 * sw, bed - sh)], closed=True,
                         fc="#5b5147", ec=INK, lw=0.6 * lw, zorder=6))


def dim(ax, p0, p1, text, off=(0, 0), color=INK, fs=7, rot=0, **kw):
    """Double-headed dimension arrow with a label."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>",
                                 mutation_scale=6, lw=0.6, color=color,
                                 shrinkA=0, shrinkB=0, zorder=9))
    xm, zm = 0.5 * (p0[0] + p1[0]) + off[0], 0.5 * (p0[1] + p1[1]) + off[1]
    ax.text(xm, zm, text, fontsize=fs, color=color, ha="center",
            va="center", rotation=rot, zorder=10,
            bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.85},
            **kw)


def ray(ax, pts, color, ls):
    """Polyline wave path with an arrowhead on its last leg."""
    pts = np.asarray(pts, dtype=float)
    ax.plot(pts[:-1, 0].tolist() + [pts[-1, 0]], pts[:, 1], color=color,
            lw=1.0, ls=ls, zorder=7)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=8, lw=1.0, color=color,
                                 ls=ls, shrinkA=0, shrinkB=1.5, zorder=7))


def ray_key(ax, x, z, dz):
    """Key for the three wave paths."""
    for k, (c, ls, lab) in enumerate(((C_DIR, "-", "direct"),
                                      (C_SUR, "--", "surface-reflected"),
                                      (C_BED, "-.", "seabed-reflected"))):
        zk = z + k * dz
        ax.plot([x, x + 8.0], [zk, zk], color=c, ls=ls, lw=1.0, zorder=8)
        ax.text(x + 10.0, zk, lab, color=c, fontsize=7, va="center",
                zorder=8)


setup()
fig = plt.figure(figsize=(7.2, 4.35))
ax = fig.add_axes([0.055, 0.10, 0.63, 0.84])
X0, X1 = -14.0, R_MAX + W_SPONGE + 3.0
ZT, ZB = -27.0, Z_BOTTOM + W_SPONGE + 1.0

# water with a gentle vertical tint, seabed, sponges
ax.imshow(np.linspace(0, 1, 128)[:, None], aspect="auto",
          extent=[X0, R_MAX + W_SPONGE, H_WATER, 0.0], zorder=0,
          cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
              "w", [WATER_TOP, WATER_BOT]))
ax.add_patch(Rectangle((X0, H_WATER), R_MAX + W_SPONGE - X0,
                       ZB - H_WATER, fc=SEABED, ec="none", zorder=1))
ax.add_patch(Rectangle((X0, H_WATER), R_MAX + W_SPONGE - X0,
                       Z_BOTTOM - H_WATER, fc="none", ec=SEABED_EDGE,
                       lw=0, hatch="....", zorder=1))
ax.plot([X0, R_MAX + W_SPONGE], [H_WATER, H_WATER], color=SEABED_EDGE,
        lw=1.0, zorder=2)
ax.add_patch(Rectangle((R_MAX, 0.0), W_SPONGE, Z_BOTTOM + W_SPONGE,
                       fc=SPONGE_C, ec="0.55", lw=0.5, hatch="////",
                       alpha=0.85, zorder=3))
ax.add_patch(Rectangle((X0, Z_BOTTOM), R_MAX - X0, W_SPONGE, fc=SPONGE_C,
                       ec="0.55", lw=0.5, hatch="////", alpha=0.85,
                       zorder=3))
ax.plot([X0, R_MAX + W_SPONGE], [0, 0], color="#3b6a95", lw=1.2, zorder=4)

# expanding shock fronts, drawn only in the water
for rr, a in ((14.0, 0.55), (26.0, 0.4), (38.0, 0.28)):
    w = Wedge((0.0, Z_MINE), rr, 180.0, 360.0, width=0.35, fc=C_DIR,
              ec="none", alpha=a, zorder=2)
    ax.add_patch(w)
    w.set_clip_path(Rectangle((X0, 0.0), R_MAX - X0, H_WATER,
                              transform=ax.transData))

# symmetry axis and computational region
ax.plot([0, 0], [ZT + 2, Z_BOTTOM + W_SPONGE], color=INK, lw=0.8,
        ls=(0, (6, 2, 1, 2)), zorder=5)
ax.text(1.2, 58.5, r"symmetry axis $r=0$", fontsize=7, ha="left",
        va="center", color=INK, zorder=10,
        bbox={"fc": SEABED, "ec": "none", "pad": 0.5})

# keel receivers and the vessel
rk = np.arange(0.0, 110.1, 10.0)
ax.plot(rk, np.full_like(rk, Z_KEEL), ls="none", marker="o", ms=2.2,
        mfc="white", mec=INK, mew=0.5, zorder=8)
ax.plot([0, 110], [Z_KEEL, Z_KEEL], color=INK, lw=0.4, ls=":", zorder=4)
warship(ax, R_SHIP, L_SHIP, Z_KEEL)
ax.plot([R_SHIP], [Z_KEEL], marker="o", ms=4.0, mfc=OKABE_ITO["orange"],
        mec=INK, mew=0.6, zorder=9)

# wave paths to the keel element (image construction)
xs = R_SHIP * Z_MINE / (Z_MINE + Z_KEEL)
zb_img = 2 * H_WATER - Z_MINE
xb = R_SHIP * (zb_img - H_WATER) / (zb_img - Z_KEEL)
ray(ax, [(0, Z_MINE), (R_SHIP, Z_KEEL)], C_DIR, "-")
ray(ax, [(0, Z_MINE), (xs, 0.0), (R_SHIP, Z_KEEL)], C_SUR, "--")
ray(ax, [(0, Z_MINE), (xb, H_WATER), (R_SHIP, Z_KEEL)], C_BED, "-.")
ray_key(ax, 62.0, 15.0, 3.6)
moored_mine(ax, 0.0, Z_MINE, 1.1, H_WATER, lw=0.5)

# labels of media and boundaries
kw = {"fontsize": 7, "color": INK, "va": "center"}
ax.text(88.0, -4.2, "sea surface\n" r"$p=0$ (pressure release)",
        ha="left", **kw)
ax.text(60.0, 29.5, "water column, TEOS-10\n"
        r"$\rho(z)$, $c(z)$ from $T(z)$, $S(z)$", ha="left", **kw)
ax.text(60.0, 43.0, "linear acoustics with\nbilinear cavitation "
        r"$p \geq p_v - p_h$", ha="left", **kw)
ax.text(60.0, 56.0, r"seabed: fluid half-space $(\rho_b, c_b)$",
        ha="left", **kw, zorder=10,
        bbox={"fc": SEABED, "ec": "none", "pad": 0.5})
ax.text(R_MAX + W_SPONGE / 2, 30.0, "absorbing sponge", rotation=90,
        ha="center", **kw)
ax.text(60.0, Z_BOTTOM + W_SPONGE / 2, "absorbing sponge", ha="center",
        **kw, zorder=10, bbox={"fc": SPONGE_C, "ec": "none", "pad": 0.4})
ax.text(64.0, 8.6, "keel receivers", ha="left", **kw)
ax.text(R_SHIP + 1.5, Z_KEEL + 2.6, "keel element", ha="left",
        fontsize=7, color=OKABE_ITO["vermil"])

# dimensions
dim(ax, (-7.5, 0.0), (-7.5, H_WATER), r"$H=50$", rot=90)
dim(ax, (-11.5, 0.0), (-11.5, Z_MINE), r"$z_c=47$", rot=90)
dim(ax, (0.0, 9.6), (R_SHIP, 9.6), r"$r=40$", off=(8.0, 0))
dim(ax, (106.0, 0.0), (106.0, Z_KEEL), r"$5$", off=(3.2, 0))

ax.set_xlim(X0, X1)
ax.set_ylim(ZB, ZT)
ax.set_aspect("equal")
ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
ax.set_xlabel(r"$r$ (m)")
ax.set_ylabel(r"depth $z$ (m)")
panel_label(ax, "(a)")

# (b) mine detail
bx = fig.add_axes([0.735, 0.555, 0.25, 0.385])
bx.add_patch(Rectangle((-3, 43), 6, H_WATER - 43, fc=WATER_BOT,
                       ec="none", zorder=0))
bx.add_patch(Rectangle((-3, H_WATER), 6, 2.0, fc=SEABED, ec="none",
                       zorder=0))
bx.add_patch(Rectangle((-3, H_WATER), 6, 2.0, fc="none", ec=SEABED_EDGE,
                       lw=0, hatch="....", zorder=1))
bx.plot([-3, 3], [H_WATER, H_WATER], color=SEABED_EDGE, lw=1.0)
bx.plot([0, 0], [43, 52], color=INK, lw=0.6, ls=(0, (6, 2, 1, 2)))
moored_mine(bx, 0.0, Z_MINE, 0.6, H_WATER, lw=0.9)
dim(bx, (2.1, Z_MINE), (2.1, H_WATER), r"$h_b=3$", rot=90)
bx.plot([0.7, 2.3], [Z_MINE, Z_MINE], color=INK, lw=0.4, ls=":")
bx.text(0.0, 43.6, f"$W$ = {CHARGE.W:.0f} kg TNT eq.", fontsize=7,
        ha="center", va="center", color=INK,
        bbox={"fc": WATER_BOT, "ec": "none", "pad": 0.5})
bx.text(-2.85, 45.2, "contact\nhorns", fontsize=6.5, color=INK,
        va="center")
bx.add_patch(FancyArrowPatch((-1.7, 45.6), (-0.62, 46.35),
                             arrowstyle="-|>", mutation_scale=5, lw=0.5,
                             color=INK))
bx.text(-2.85, 48.2, "mooring", fontsize=6.5, color=INK, va="center")
bx.text(-2.85, 49.45, "sinker", fontsize=6.5, color=INK, va="center")
bx.set_xlim(-3, 3)
bx.set_ylim(52, 43)
bx.set_aspect("equal")
bx.set_xticks([-2, 0, 2])
bx.set_yticks([44, 46, 48, 50])
bx.tick_params(labelsize=7)
bx.set_xlabel(r"$r$ (m)", fontsize=8, labelpad=1)
panel_label(bx, "(b)")

# (c) free-body diagram of the hull plate element
cx = fig.add_axes([0.735, 0.10, 0.25, 0.34])
cx.set_xlim(0, 10)
cx.set_ylim(0, 8.0)
cx.axis("off")
cx.add_patch(Rectangle((0, 0), 10, 3.1, fc=WATER_BOT, ec="none"))
cx.add_patch(Rectangle((0, 3.6), 10, 3.0, fc="#f3f4f6", ec="none"))
cx.add_patch(Rectangle((1.0, 3.1), 8.0, 0.5, fc="#6b7280", ec=INK, lw=0.6))
cx.text(0.3, 6.3, "air (hull interior)", ha="left", va="top",
        fontsize=6.5, color=INK)
cx.text(9.8, 0.25, "water", ha="right", va="bottom", fontsize=6.5,
        color=INK)
cx.text(1.0, 4.0, rf"steel plate, $m={PLATE_M:.0f}$ kg m$^{{-2}}$",
        ha="left", fontsize=6.5, color=INK)
for xa in (2.0, 3.2, 4.4):
    cx.add_patch(FancyArrowPatch((xa, 0.9), (xa, 3.05), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=C_SUR))
cx.text(3.2, 0.55, r"$2\,p_i(t)$", ha="center", fontsize=7.5,
        color=C_SUR)
cx.add_patch(FancyArrowPatch((6.6, 3.05), (6.6, 1.0), arrowstyle="-|>",
                             mutation_scale=7, lw=1.0, color=C_DIR))
cx.text(6.6, 0.55, r"$\rho_w c_w\, v$", ha="center", fontsize=7.5,
        color=C_DIR)
cx.add_patch(FancyArrowPatch((8.6, 3.7), (8.6, 5.9), arrowstyle="-|>",
                             mutation_scale=7, lw=1.2, color=INK))
cx.text(8.95, 5.2, r"$v$", fontsize=8, color=INK)
cx.text(5.0, 7.35, r"$m\,\dot v = 2\,p_i - \rho_w c_w\, v$",
        ha="center", fontsize=8.5, color=INK)
panel_label(cx, "(c)", dy=1.0)

print(save(fig, "fig00_geometry"))

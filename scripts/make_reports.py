"""Generate the plain-text reports.

Numbers are recomputed here from the package or read back from the CSV
files and caches written by the figure scripts, so every value quoted in
a report can be traced to a file under outputs/.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import os

import numpy as np

from legacymine import ocean
from legacymine.bubble import E_B, J_TNT, max_radius, period
from legacymine.io_utils import DATADIR, load_cache, write_report
from legacymine.scenario import (CFL_WATER, CHARGE, DX, H_WATER, PLATE_M,
                                 R_MAX, SPONGE, T_END, Z_KEEL, Z_MINE)
from legacymine.source import decay_time, peak_pressure


def csv(stem):
    """Read a CSV written by io_utils.write_csv as a structured array."""
    return np.genfromtxt(os.path.join(DATADIR, f"{stem}.csv"),
                         delimiter=",", names=True)


def wrap(text, width=68):
    """Wrap prose to report lines indented two spaces."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append("  " + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append("  " + cur)
    return lines


# ------------------------------------------------------------- parameters
rows = ["  regime          | T top/bot (C) | S top/bot (g/kg) | bed       |"
        " rho_b  | c_b    | A     | R_0   | theta_c",
        "  " + "-" * 100]
for k in ocean.ORDER:
    r = ocean.REGIMES[k]
    rb, cb = ocean.seabed(r)
    tc = ocean.critical_angle(r)
    rows.append(f"  {k:15s} | {r.T_top:5.1f} / {r.T_bot:5.1f} |"
                f" {r.S_top:6.2f} / {r.S_bot:6.2f}  | {r.bed:9s} |"
                f" {rb:6.0f} | {cb:6.1f} | {ocean.image_strength(r):5.3f} |"
                f" {ocean.reflection(r):5.3f} | "
                + ("  none" if np.isnan(tc) else f"{tc:6.2f}"))
P40 = float(peak_pressure(CHARGE, CHARGE.R_ref))
th40 = float(decay_time(CHARGE, CHARGE.R_ref))
write_report("parameters", "Symbols, units, and reference values", [
    ("units", wrap(
        "All quantities are SI except where a figure axis states MPa, kPa"
        " s, or ms. Depth z is positive downward from the sea surface; r"
        " is the horizontal distance from the vertical axis through the"
        " charge.")),
    ("geometry and grid", [
        f"  water depth H = {H_WATER} m, charge depth {Z_MINE} m, keel"
        f" line {Z_KEEL} m",
        f"  grid spacing h = {DX} m, domain r <= {R_MAX} m plus {SPONGE}"
        " sponge cells, simulated time", f"  {T_END} s, water Courant"
        f" number <= {CFL_WATER}"]),
    ("charge", [
        f"  W = {CHARGE.W} kg TNT equivalent, eta = {CHARGE.eta}",
        f"  similitude: K_P = {CHARGE.K_P / 1e6} MPa, A_P = {CHARGE.A_P},"
        f" K_T = {CHARGE.K_T * 1e6} us kg^-1/3, A_T = {CHARGE.A_T}",
        f"  monopole calibrated at R_ref = {CHARGE.R_ref} m: P = "
        f"{P40 / 1e6:.4f} MPa, theta = {th40 * 1e3:.4f} ms, rise"
        f" {CHARGE.t_rise * 1e3} ms",
        f"  bubble: J = {J_TNT} m kg^-1/3 m^1/3, e_b = {E_B / 1e6:.4f}"
        " MJ/kg"]),
    ("hull element", [f"  areal mass m = {PLATE_M:.1f} kg m^-2 (12 mm"
                      " steel)"]),
    ("regimes (illustrative, not observations)", rows),
    ("provenance", wrap(
        "Similitude constants are the TNT values widely tabulated after"
        " Cole (1948). Water properties are TEOS-10 through gsw, with"
        " salinity read as Absolute Salinity and the Baltic composition"
        " anomaly ignored. Temperature and salinity profiles and seabed"
        " constants are round values inside the ranges usually quoted for"
        " each setting. None is a measurement at a site."))])

# ----------------------------------------------------------- verification
v = load_cache("verification")
lines = []
if v is not None:
    e = v["err"]
    lines += ["  free field, max relative error vs h:"]
    lines += [f"    h = {h:5.3f} m   {x:.4e}" for h, x in zip(v["h"], e)]
    lines += ["  observed orders: " + ", ".join(
        f"{o:.2f}" for o in np.log2(e[:-1] / e[1:]))]
    lm = np.max(np.abs(v["lm_num"] - v["lm_ex"])) / np.max(
        np.abs(v["lm_ex"]))
    lines += [f"  Lloyd mirror, max error / peak: {lm:.4e}"]
    lines += ["  seabed reflection, numerical / theory:"]
    lines += [f"    {k:15s} {x:.4f}"
              for k, x in zip(ocean.ORDER, v["refl"])]
    lines += ["  cavitating keel peak vs water Courant number (reduced"
              " domain):"]
    lines += [f"    C = {c:4.2f}: r = 0 m {p[0] / 1e6:.3f} MPa, r = 20 m"
              f" {p[1] / 1e6:.3f} MPa"
              for c, p in zip(v["cfl"], v["cfl_pk"])]
td = csv("fig01d_taylor")
lines += [f"  Taylor plate integrator, max |v - v_exact|: "
          f"{np.max(np.abs(td['v_numerical'] - td['v_exact'])):.3e} m/s"]
col = ocean.column(ocean.REGIMES["baltic_summer"], np.array([Z_MINE]))
rat = float(period(CHARGE.W, col["rho"][0], col["p_h"][0] - 2.3e3)
            / (2.11 * np.cbrt(CHARGE.W) / (Z_MINE + 10.0) ** (5 / 6)))
lines += [f"  Rayleigh period / tabulated TNT period at {Z_MINE} m:"
          f" {rat:.4f}"]
write_report("verification", "Verification", [
    ("results", lines),
    ("notes", wrap(
        "The exact free-field solution includes the Gaussian source"
        " smoothing of width 1.5 h. The observed order approaches two,"
        " the order of the leapfrog step; the fourth-order spatial"
        " stencil is not the limiting error. The axis flux is written"
        " in conservative form, so the injected volume is conserved to"
        " round-off; a symmetric ghost for r u_r lost 1.65 percent of the"
        " source at every resolution and was rejected. At water Courant"
        " numbers near 0.9, closure of cavitated water produced growing"
        " water-hammer oscillations and a ringing precursor ahead of"
        " sharp fronts; all runs therefore cap the water Courant number"
        " at 0.45. The remaining spread of the cavitating keel peak, a"
        " few percent between Courant numbers 0.3 and 0.6, is the"
        " uncertainty attached to closure pulses."))])

# ---------------------------------------------------------------- results
kd = csv("fig03_keel")
i60 = int(np.argmin(np.abs(kd["r_m"] - 60.0)))
res = ["  regime          | peak (MPa) | I (kPa s) | I_inc/I_free |"
       " kick (m/s) | entropy", "  " + "-" * 76]
for k in ocean.ORDER:
    res.append(f"  {k:15s} | {kd[k + '_peak'][i60] / 1e6:10.3f} |"
               f" {kd[k + '_imp'][i60] / 1e3:9.3f} |"
               f" {kd[k + '_gain'][i60]:12.3f} |"
               f" {kd[k + '_kick'][i60]:10.3f} |"
               f" {kd[k + '_entropy'][i60]:7.3f}")
cv = csv("fig02_cavitation")
cav = ["  regime          | cavitated area (m^2, half-plane) |"
       " longest (ms)", "  " + "-" * 64]
for k, a, m in zip(ocean.ORDER, cv["cav_area_m2"], cv["max_cav_ms"]):
    cav.append(f"  {k:15s} | {a:32.1f} | {m:12.2f}")
bub = []
for k in ocean.ORDER:
    c = ocean.column(ocean.REGIMES[k], np.array([Z_MINE]))
    dp = float(c["p_h"][0]) - 2.3e3
    bub.append(f"  {k:15s} R_m = {float(max_radius(CHARGE.W, dp)):.3f} m,"
               f" T = {float(period(CHARGE.W, c['rho'][0], dp)):.4f} s")
write_report("results", "Key results", [
    ("keel line at r = 60 m (fresh charge)", res),
    ("cavitation", cav),
    ("bubble at the charge depth", bub),
    ("aged charge", wrap(
        "Every similitude length scales with (eta W)^(1/3). A charge that"
        " retains 20 percent of its yield has 0.585 times the hazard"
        " radius, bubble radius, and bubble period scale of a fresh one,"
        " and 5 percent retains 0.368 times. Large losses of yield"
        " therefore shrink the hazard only modestly."))])

# ------------------------------------------------------------ open items
write_report("open_items", "Limitations and open items", [
    ("physics", wrap(
        "Propagation is linear acoustics with a bilinear cavitation law;"
        " the finite-amplitude steepening of the shock and the similitude"
        " decay R^(-1.13) are not represented, and the linear model"
        " exceeds the similitude peak by up to about 15 percent at the"
        " far end of the keel line. The seabed is a lossless fluid"
        " without shear, which is a crude description of carbonate rock."
        " The ship is not coupled to the water; keel metrics are field"
        " quantities and a single rigid Taylor element. The gas bubble is"
        " not part of the acoustic source, and its jet direction is a"
        " leading-order Kelvin-impulse estimate that is invalid where the"
        " bubble reaches a boundary, which includes the reference charge"
        " 3 m above the bed.")),
    ("numerics", wrap(
        "Cavitation closure near the axis leaves grid-scale speckle in"
        " the cavitated region, visible in the animations over the reef."
        " Receivers within 10 m of the sponge are excluded. The axis"
        " focuses any sponge reflection; for the reference domain these"
        " arrive after the simulated time.")),
    ("inputs", wrap(
        "Regime profiles and seabed constants are illustrative. Charge"
        " ageing enters only through eta; no corrosion or degradation"
        " kinetics are modelled. The DOIs listed in the README other than"
        " Supponen et al. (2016) have not yet been checked against the"
        " publisher records and must be verified before citation."))])
print("reports written")

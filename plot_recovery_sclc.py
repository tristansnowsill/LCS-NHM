#!/usr/bin/env python3
"""Visualise SCLC recovery fits (sclc-model.stan) against the true values.

The SCLC counterpart of plot_recovery.py. For each simulated set (a truth JSON
from simulate_sclc.py plus one or more CmdStan CSV files, one per chain) this
writes:

  <label>-density-<group>.png   univariate posterior densities, one figure per
                                parameter group, truth marked
  <label>-corner.png            all pairwise posteriors of the 14 global
                                parameters: scatter plots below the diagonal
                                (truth marked), densities on the diagonal,
                                and the correlation of each pair above it;
                                sensitivities on the logit scale

and, across all sets given:

  recovery-z.png                z = (posterior mean - truth) / posterior sd for
                                every global parameter, one marker per set
  recovery-z.csv                the same numbers as a table, plus the posterior
                                rank of the truth and the 90% interval

Correlations are on the unconstrained scale (logit for sensitivities, log for
positive parameters). Densities are estimated so they don't leak past the
bounds: reflected at 0 and 1 for sensitivities, on the log scale for positive
parameters.

Stage 1 is limited SCLC and stage 2 extensive SCLC.

Examples:
  python plot_recovery_sclc.py -o sclc-plots \\
      --set sclc-recovery-truth-1.json sclc-fit-1.csv \\
      --set sclc-recovery-truth-2.json sclc-fit-2-chain1.csv sclc-fit-2-chain2.csv
"""

import argparse
import csv
import json
import os
import re

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Palette as in plot_recovery.py (validated: categorical slots 1-4 pass CVD
# and normal-vision checks on the light surface; slots 3 and 4 are below 3:1
# contrast, so sets also differ by marker shape and the z-scores are written
# as a table).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
POST = "#2a78d6"        # posterior (series slot 1)
POST_LIGHT = "#b7d3f6"  # posterior 90% interval fill (blue ramp step 150)
TRUTH = "#eb6834"       # truth (series slot 2)
SET_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SET_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9,
})

ARM = {1: "arm 1", 2: "arm 2"}
STAGE = {1: "limited", 2: "extensive"}

# The 14 global parameters, in plotting order
PARAMS = ["m_SCLC", "c_SCLC", "b_SCLC_plco", "b_SCLC_cons.1", "b_SCLC_cons.2",
          "mu", "phi.1", "phi.2",
          "sens_SCLC.1.1", "sens_SCLC.1.2", "sens_SCLC.2.1", "sens_SCLC.2.2",
          "log_or_sens_final.1", "log_or_sens_final.2"]

# Univariate figure groups: (file suffix, title, columns, ncols)
GROUPS = [
    ("natural_history", "Incidence, baseline prevalence and progression",
     ["m_SCLC", "c_SCLC", "b_SCLC_plco", "b_SCLC_cons.1", "b_SCLC_cons.2",
      "mu", "phi.1", "phi.2"], 4),
    ("sens", "Screen sensitivity and final-screen effect (rows: arm)",
     ["sens_SCLC.1.1", "sens_SCLC.1.2", "log_or_sens_final.1",
      "sens_SCLC.2.1", "sens_SCLC.2.2", "log_or_sens_final.2"], 3),
]
DERIVED_GROUP = (
    "sens_final", "Final-screen sensitivity (rows: arm)",
    [f"sens_SCLC_final.{g}.{k}" for g in (1, 2) for k in (1, 2)], 2)


def label(name, short=False):
    """Readable mathtext label for a CSV column name (short for the corner plot)."""
    p = name.split(".")
    base, idx = p[0], [int(i) for i in p[1:]]
    if base == "m_SCLC":
        return "$m$" if short else "$m$ (Gompertz shape)"
    if base == "c_SCLC":
        return "$c$" if short else "$c$ (Gompertz log rate)"
    if base == "b_SCLC_plco":
        return r"$b_\mathrm{plco}$"
    if base == "b_SCLC_cons":
        st = "L" if idx[0] == 1 else "E"
        return (rf"$b_\mathrm{{cons,{st}}}$" if short
                else rf"$b_\mathrm{{cons}}$, {STAGE[idx[0]]}")
    if base == "mu":
        return r"$\mu$" if short else r"$\mu$ (limited $\to$ extensive)"
    if base == "phi":
        st = "L" if idx[0] == 1 else "E"
        return (rf"$\phi_\mathrm{{{st}}}$" if short
                else rf"$\phi_{idx[0]}$ (presentation, {STAGE[idx[0]]})")
    if base == "log_or_sens_final":
        return (f"log OR final, arm {idx[0]}" if short
                else f"log OR final screen, {ARM[idx[0]]}")
    if base in ("sens_SCLC", "sens_SCLC_final"):
        st = "L" if idx[1] == 1 else "E"
        pre = "final sens" if base == "sens_SCLC_final" else "sens"
        return (f"{pre} {st}, arm {idx[0]}" if short
                else f"{pre}, {ARM[idx[0]]}, {STAGE[idx[1]]}")
    return name


def scale(name):
    """'logit', 'log' or 'identity': the unconstrained scale of a parameter."""
    if name.startswith(("sens_SCLC.", "sens_SCLC_final.")):
        return "logit"
    if name.startswith(("b_SCLC_plco", "log_or_sens_final", "mu", "phi", "m_SCLC")):
        return "log"
    return "identity"


def to_unc(name, x):
    s = scale(name)
    if s == "logit":
        x = np.clip(x, 1e-15, 1 - 1e-15)   # extensive sens can round to 1
        return np.log(x) - np.log1p(-x)
    if s == "log":
        return np.log(x)
    return x


def read_cmdstan_csv(path):
    """Post-warmup draws from one CmdStan CSV (drops saved warmup)."""
    num_warmup, save_warmup = 1000, False
    header, rows = None, []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                m = re.match(r"#\s*num_warmup = (\d+)", line)
                if m:
                    num_warmup = int(m.group(1))
                m = re.match(r"#\s*save_warmup = (\S+)", line)
                if m:
                    save_warmup = m.group(1) in ("1", "true")
                continue
            if header is None:
                header = next(csv.reader([line]))
            elif line.strip():
                r = line.rstrip("\n").split(",")
                if len(r) == len(header):   # skip a partly written last row
                    rows.append(r)
    draws = np.array(rows, dtype=float)
    if save_warmup:
        draws = draws[num_warmup:]
    return header, draws


def load_set(truth_path, csv_paths):
    header, parts = None, []
    for p in csv_paths:
        h, d = read_cmdstan_csv(p)
        if header is None:
            header = h
        elif h != header:
            raise SystemExit(f"{p}: columns differ from {csv_paths[0]}")
        parts.append(d)
    draws = np.vstack(parts)
    if len(draws) < 10:
        raise SystemExit(f"{csv_paths}: only {len(draws)} post-warmup draws")
    with open(truth_path) as f:
        truth_json = json.load(f)
    truth = {}

    def flat(key, v):
        if isinstance(v, list):
            for i, x in enumerate(v, 1):
                flat(f"{key}.{i}", x)
        else:
            truth[key] = float(v)
    for k, v in truth_json.items():
        if k != "seed":
            flat(k, v)
    col = {name: draws[:, i] for i, name in enumerate(header)}
    missing = [n for n in PARAMS if n not in col or n not in truth]
    if missing:
        raise SystemExit(f"{truth_path} / {csv_paths[0]}: missing {missing}")
    return col, truth, len(parts)


def default_label(truth_path):
    m = re.search(r"(\d+)\.json$", truth_path)
    return f"set{m.group(1)}" if m else os.path.splitext(os.path.basename(truth_path))[0]


def legend_label(lab):
    m = re.fullmatch(r"set(\d+)", lab)
    return f"Set {m.group(1)}" if m else lab


def density(name, x, grid, kind=None):
    """Posterior density estimate that respects the parameter's bounds.

    Probabilities: KDE on the natural scale, reflected at 0 and 1 (a KDE on the
    logit scale turns single draws near 1 into spikes via the Jacobian).
    Positive parameters: KDE on the log scale, transformed back.
    """
    s = kind or scale(name)
    if s == "logit":
        h = gaussian_kde(x).factor * x.std(ddof=1)   # Scott bandwidth of x
        xr = np.concatenate([x, -x, 2 - x])
        kde = gaussian_kde(xr, bw_method=h / xr.std(ddof=1))
        return 3 * kde(grid)
    if s == "log":
        return gaussian_kde(np.log(x))(np.log(grid)) / grid
    return gaussian_kde(x)(grid)


def summarise(x, t):
    sd = x.std(ddof=1)
    return {"mean": x.mean(), "sd": sd, "z": (x.mean() - t) / sd,
            "rank": np.mean(x < t), "q05": np.quantile(x, 0.05),
            "q95": np.quantile(x, 0.95)}


def plot_range(name, x, t, kind=None):
    """Axis limits covering the bulk of the draws and the truth, and the
    narrower range over which to draw the density."""
    q_lo, q_hi = np.quantile(x, [0.002, 0.998])
    lo, hi = min(q_lo, t), max(q_hi, t)
    pad = 0.06 * (hi - lo) if hi > lo else 0.01
    lo, hi = lo - pad, hi + pad
    s = kind or scale(name)
    if s == "logit":
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    elif s == "log":
        lo = max(lo, 0.0)
    return lo, hi, q_lo, q_hi


def draw_density(ax, name, x, t, show_interval=True, kind=None):
    """kind overrides the parameter's scale ('identity' for logit values)."""
    st = summarise(x, t)
    lo, hi, q_lo, q_hi = plot_range(name, x, t, kind)
    if q_hi > q_lo:
        grid = np.linspace(q_lo, q_hi, 400)
        dens = density(name, x, grid, kind)
        if show_interval:
            inside = (grid >= st["q05"]) & (grid <= st["q95"])
            ax.fill_between(grid, dens, where=inside, color=POST_LIGHT, lw=0)
        ax.plot(grid, dens, color=POST, lw=2 if show_interval else 1.5)
    ax.axvline(t, color=TRUTH, lw=2 if show_interval else 1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    return st


def plot_density_group(col, truth, lab, group, outdir, fmt):
    suffix, title, names, ncols = group
    names = [n for n in names if n in col and n in truth]
    if not names:
        return None
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.1 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        st = draw_density(ax, name, col[name], truth[name])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(label(name), loc="left")
        ax.set_title(f"z {st['z']:+.2f}", loc="right", color=INK_2, fontsize=8)
    handles = [plt.Line2D([], [], color=POST, lw=2, label="posterior"),
               plt.Rectangle((0, 0), 1, 1, color=POST_LIGHT, label="90% interval"),
               plt.Line2D([], [], color=TRUTH, lw=2, label="truth")]
    fig.suptitle(f"{lab}: {title}", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06 if nrows == 1 else 0.04, 1, 0.95))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    path = os.path.join(outdir, f"{lab}-density-{suffix}.{fmt}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_corner(col, truth, lab, outdir, fmt, max_points):
    names = PARAMS
    k = len(names)
    U = np.array([to_unc(n, col[n]) for n in names])
    C = np.corrcoef(U)
    n = U.shape[1]
    idx = np.random.default_rng(1).choice(n, min(n, max_points), replace=False)
    # Sensitivities on the logit scale here: extensive-stage draws pile up
    # against 1 on the probability scale, hiding any relationship
    on_logit = {nm: scale(nm) == "logit" for nm in names}
    val = {nm: to_unc(nm, col[nm]) if on_logit[nm] else col[nm] for nm in names}
    tru = {nm: float(to_unc(nm, np.array(truth[nm]))) if on_logit[nm] else truth[nm]
           for nm in names}
    kind = {nm: "identity" if on_logit[nm] else None for nm in names}
    lims = {nm: plot_range(nm, val[nm], tru[nm], kind[nm])[:2] for nm in names}
    corner_label = lambda nm: ("logit " if on_logit[nm] else "") + label(nm, short=True)

    fig, axes = plt.subplots(k, k, figsize=(1.35 * k, 1.35 * k))
    for i, yn in enumerate(names):
        for j, xn in enumerate(names):
            ax = axes[i, j]
            ax.tick_params(labelsize=6, length=2)
            if i == j:
                draw_density(ax, xn, val[xn], tru[xn], show_interval=False,
                             kind=kind[xn])
            elif i > j:
                ax.scatter(val[xn][idx], val[yn][idx], s=2, color=POST,
                           alpha=0.3, lw=0, rasterized=True)
                ax.axvline(tru[xn], color=TRUTH, lw=0.8, alpha=0.6)
                ax.axhline(tru[yn], color=TRUTH, lw=0.8, alpha=0.6)
                ax.scatter([tru[xn]], [tru[yn]], s=28, marker="X",
                           color=TRUTH, edgecolor=SURFACE, lw=0.8, zorder=4)
                ax.set_xlim(*lims[xn])
                ax.set_ylim(*lims[yn])
            else:
                r = C[i, j]
                ax.axis("off")
                ax.text(0.5, 0.5, f"{r:+.2f}", transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=9 if abs(r) >= 0.5 else 7.5,
                        fontweight="bold" if abs(r) >= 0.5 else "normal",
                        color=INK if abs(r) >= 0.3 else MUTED)
                continue
            ax.xaxis.set_major_locator(MaxNLocator(3))
            if i != j:
                ax.yaxis.set_major_locator(MaxNLocator(3))
            if i == k - 1:
                ax.set_xlabel(corner_label(xn), fontsize=8)
                for t in ax.get_xticklabels():
                    t.set_rotation(45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(corner_label(yn), fontsize=8)
            elif i != j:
                ax.set_yticklabels([])
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=4, color=POST, alpha=0.6,
                   label="posterior draws / density"),
        plt.Line2D([], [], ls="", marker="X", ms=8, color=TRUTH,
                   markeredgecolor=SURFACE, label="truth")]
    fig.suptitle(f"{lab}: all pairwise posteriors. Above the diagonal: correlation "
                 "on the unconstrained scale (bold if |r| ≥ 0.5). "
                 "Sensitivities on the logit scale. L = limited, E = extensive",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(0.99, 0.975))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.95,
                        wspace=0.12, hspace=0.12)
    fig.align_ylabels(axes[:, 0])
    path = os.path.join(outdir, f"{lab}-corner.{fmt}")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_z(sets, outdir, fmt):
    names = PARAMS
    k = len(sets)
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(names) + 1.4))
    ax.axvspan(-1, 1, color="#f0efec", lw=0, zorder=0)
    for v in (-2, 2):
        ax.axvline(v, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ypos = np.arange(len(names))[::-1]
    rows = []
    for s, (lab, col, truth) in enumerate(sets):
        zs, ys = [], []
        for y, n in zip(ypos, names):
            st = summarise(col[n], truth[n])
            zs.append(st["z"])
            ys.append(y)
            rows.append({"set": lab, "parameter": n, "truth": truth[n], **st})
        ax.scatter(zs, ys, s=40, marker=SET_MARKERS[s % 8],
                   color=SET_COLOURS[s % 8], edgecolor=SURFACE, lw=1,
                   label=legend_label(lab), zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label(n) for n in names], fontsize=8)
    ax.set_ylim(-0.8, len(names) - 0.2)
    lim = max(3, np.ceil(max(abs(r["z"]) for r in rows) + 0.3))
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("z = (posterior mean − truth) / posterior sd")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("SCLC recovery (shaded: |z| < 1; dashed: |z| = 2)",
                 loc="left", fontsize=10, pad=20 if k > 1 else 6)
    if k > 1:
        # one row above the axes, so it never covers a point
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=k,
                  frameon=False, borderaxespad=0.3, handletextpad=0.2,
                  columnspacing=1.2)
    fig.tight_layout()
    path = os.path.join(outdir, f"recovery-z.{fmt}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    tpath = os.path.join(outdir, "recovery-z.csv")
    with open(tpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "parameter", "truth", "mean", "sd", "q05", "q95",
                    "z", "rank_of_truth"])
        for r in rows:
            w.writerow([r["set"], r["parameter"], f"{r['truth']:.6g}",
                        f"{r['mean']:.6g}", f"{r['sd']:.6g}", f"{r['q05']:.6g}",
                        f"{r['q95']:.6g}", f"{r['z']:.3f}", f"{r['rank']:.3f}"])
    return path, tpath, rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="+", action="append", required=True,
                    metavar=("TRUTH_JSON", "CSV"),
                    help="truth JSON followed by the CmdStan CSV(s) of that fit "
                         "(one per chain); repeat for each set")
    ap.add_argument("--label", action="append",
                    help="label for each --set, in order (default: set<N> "
                         "from the truth file name)")
    ap.add_argument("-o", "--outdir", default="sclc-recovery-plots")
    ap.add_argument("--derived", action="store_true",
                    help="also plot sens_SCLC_final densities")
    ap.add_argument("--max-points", type=int, default=1500,
                    help="maximum draws shown in each scatter plot")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    for s in args.set:
        if len(s) < 2:
            ap.error("--set needs a truth JSON and at least one CSV")
    if args.label and len(args.label) != len(args.set):
        ap.error("give one --label per --set")
    os.makedirs(args.outdir, exist_ok=True)

    groups = GROUPS + ([DERIVED_GROUP] if args.derived else [])
    sets = []
    for i, (truth_path, *csvs) in enumerate(args.set):
        lab = args.label[i] if args.label else default_label(truth_path)
        col, truth, nchains = load_set(truth_path, csvs)
        ndraws = len(next(iter(col.values())))
        print(f"{lab}: {ndraws} post-warmup draws from {nchains} chain(s)")
        for g in groups:
            p = plot_density_group(col, truth, lab, g, args.outdir, args.format)
            if p:
                print("  wrote", p)
        print("  wrote", plot_corner(col, truth, lab, args.outdir, args.format,
                                     args.max_points))
        sets.append((lab, col, truth))

    zpath, tpath, rows = plot_z(sets, args.outdir, args.format)
    print("wrote", zpath, "and", tpath)
    for lab, _, _ in sets:
        z = np.array([r["z"] for r in rows if r["set"] == lab])
        rk = np.array([r["rank"] for r in rows if r["set"] == lab])
        print(f"{lab}: {len(z)} parameters, |z| > 1: {np.sum(abs(z) > 1)}, "
              f"|z| > 2: {np.sum(abs(z) > 2)}, truth outside 90% interval: "
              f"{np.sum((rk < 0.05) | (rk > 0.95))}, mean z {z.mean():+.2f}")


if __name__ == "__main__":
    main()

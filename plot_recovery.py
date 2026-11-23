#!/usr/bin/env python3
"""Visualise NSCLC recovery fits against the true parameter values.

For each simulated set (a truth JSON plus one or more CmdStan CSV files, one per
chain) this writes:

  <label>-density-<group>.png   univariate posterior densities, one figure per
                                parameter group, truth marked
  <label>-pairs.png             pairwise scatter plots, truth marked

and, across all sets given:

  recovery-z.png                z = (posterior mean - truth) / posterior sd for
                                every global parameter, one marker per set
  recovery-z.csv                the same numbers as a table, plus the posterior
                                rank of the truth and the 90% interval

Pairs: four fixed pairs with a joint role in the model (m and c of the Gompertz
hazard; lambda[1] and sigma_re, since z multiplies lambda; stage-1 sensitivity
against stage-1 progression, the sensitivity / sojourn-time trade-off; lambda[1]
and xi[1], the two exits from stage 1), then the most correlated remaining pairs
in that set's posterior (correlation on the logit scale for sens, log scale for
positive parameters), up to --n-pairs in total. Add pairs with --pair a:b.

Densities are kernel density estimates on the unconstrained scale (logit for
sens, log for positive parameters), transformed back, so they don't leak past
the bounds.

Examples:
  python plot_recovery.py --set nsclc-recovery-truth-1.json simulated-nsclc-recovery-1.thinned.csv
  python plot_recovery.py -o figs \\
      --set nsclc-recovery-truth-1.json simulated-nsclc-recovery-1.thinned.csv \\
      --set nsclc-recovery-truth-2.json set2-chain1.csv set2-chain2.csv
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

# Palette (validated: categorical slots 1-4 pass CVD and normal-vision checks on
# the light surface; slots 3 and 4 are below 3:1 contrast, so sets also differ
# by marker shape and the z-scores are written as a table).
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

# Univariate figure groups: (file suffix, title, columns, ncols)
GROUPS = [
    ("scalars", "Incidence, PLCO effect, random effect and final-screen effect",
     ["m_NSCLC", "c_NSCLC", "b_NSCLC_plco", "sigma_re",
      "log_or_sens_final.1", "log_or_sens_final.2"], 3),
    ("b_cons", "Prevalence-screen stage intercepts b_NSCLC_cons",
     [f"b_NSCLC_cons.{k}" for k in range(1, 8)], 4),
    ("lambda", "Progression rates lambda (stage k to k+1)",
     [f"lambda.{k}" for k in range(1, 7)], 3),
    ("xi", "Clinical presentation rates xi",
     [f"xi.{k}" for k in range(1, 8)], 4),
    ("sens", "Screen sensitivity by stage (rows: arm)",
     [f"sens.{g}.{k}" for g in (1, 2) for k in range(1, 8)], 7),
]
DERIVED_GROUP = (
    "sens_final", "Final-screen sensitivity by stage (rows: arm)",
    [f"sens_final.{g}.{k}" for g in (1, 2) for k in range(1, 8)], 7)

FIXED_PAIRS = [("m_NSCLC", "c_NSCLC"), ("lambda.1", "sigma_re"),
               ("lambda.1", "sens.1.1"), ("lambda.1", "xi.1")]

GLOBAL_PREFIXES = ("m_NSCLC", "c_NSCLC", "b_NSCLC_plco", "b_NSCLC_cons",
                   "sigma_re", "log_or_sens_final", "lambda", "xi", "sens.")


def label(name):
    """Readable mathtext label for a CSV column name."""
    p = name.split(".")
    base, idx = p[0], [int(i) for i in p[1:]]
    if base == "m_NSCLC":
        return "$m$ (Gompertz shape)"
    if base == "c_NSCLC":
        return "$c$ (Gompertz log rate)"
    if base == "b_NSCLC_plco":
        return r"$b_\mathrm{plco}$"
    if base == "sigma_re":
        return r"$\sigma_\mathrm{re}$"
    if base == "log_or_sens_final":
        return f"log OR final screen, {ARM[idx[0]]}"
    if base == "b_NSCLC_cons":
        return rf"$b_\mathrm{{cons}}$, stage {idx[0]}"
    if base == "lambda":
        return rf"$\lambda_{idx[0]}$ (stage {idx[0]}$\to${idx[0] + 1})"
    if base == "xi":
        return rf"$\xi_{idx[0]}$ (stage {idx[0]})"
    if base == "sens":
        return f"sens, {ARM[idx[0]]}, stage {idx[1]}"
    if base == "sens_final":
        return f"final sens, {ARM[idx[0]]}, stage {idx[1]}"
    return name


def scale(name):
    """'logit', 'log' or 'identity': the unconstrained scale of a parameter."""
    if name.startswith(("sens.", "sens_final.")):
        return "logit"
    if name.startswith(("b_NSCLC_plco", "sigma_re", "log_or_sens_final",
                        "lambda", "xi", "m_NSCLC")):
        return "log"
    return "identity"


def to_unc(name, x):
    s = scale(name)
    if s == "logit":
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
        if k not in ("eta", "seed"):
            flat(k, v)
    col = {name: draws[:, i] for i, name in enumerate(header)}
    return col, truth, len(parts)


def default_label(truth_path):
    m = re.search(r"(\d+)\.json$", truth_path)
    return f"set{m.group(1)}" if m else os.path.splitext(os.path.basename(truth_path))[0]


def legend_label(lab):
    m = re.fullmatch(r"set(\d+)", lab)
    return f"Set {m.group(1)}" if m else lab


def density(name, x, grid):
    """Posterior density estimate that respects the parameter's bounds.

    Probabilities: KDE on the natural scale, reflected at 0 and 1 (a KDE on the
    logit scale turns single draws near 1 into spikes via the Jacobian).
    Positive parameters: KDE on the log scale, transformed back.
    """
    s = scale(name)
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


def plot_density_group(col, truth, lab, group, outdir, fmt):
    suffix, title, names, ncols = group
    names = [n for n in names if n in col and n in truth]
    if not names:
        return None
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        x, t = col[name], truth[name]
        st = summarise(x, t)
        # Density only over the bulk of the draws: beyond them a KDE on the
        # logit/log scale shows bumps from single draws, magnified by the
        # Jacobian near a bound. The axis still extends to show the truth.
        q_lo, q_hi = np.quantile(x, [0.002, 0.998])
        lo, hi = min(q_lo, t), max(q_hi, t)
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if scale(name) == "logit":
            lo, hi = max(lo, 0.0), min(hi, 1.0)
        elif scale(name) == "log":
            lo = max(lo, 0.0)
        grid = np.linspace(q_lo, q_hi, 400)
        dens = density(name, x, grid)
        inside = (grid >= st["q05"]) & (grid <= st["q95"])
        ax.fill_between(grid, dens, where=inside, color=POST_LIGHT, lw=0)
        ax.plot(grid, dens, color=POST, lw=2)
        ax.axvline(t, color=TRUTH, lw=2)
        ax.set_title(label(name), loc="left")
        ax.set_title(f"z {st['z']:+.2f}", loc="right", color=INK_2, fontsize=8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
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


def choose_pairs(col, truth, n_pairs, extra):
    names = [n for n in col if n.startswith(GLOBAL_PREFIXES)
             and not n.startswith("sens_final") and n in truth]
    chosen = []
    for a, b in list(extra) + FIXED_PAIRS:
        if a in col and b in col and (a, b) not in chosen and (b, a) not in chosen:
            chosen.append((a, b))
    U = np.array([to_unc(n, col[n]) for n in names])
    C = np.corrcoef(U)
    ranked = sorted(((abs(C[i, j]), names[i], names[j])
                     for i in range(len(names)) for j in range(i + 1, len(names))),
                    reverse=True)
    for _, a, b in ranked:
        if len(chosen) >= max(n_pairs, len(extra)):
            break
        if (a, b) not in chosen and (b, a) not in chosen:
            chosen.append((a, b))
    return chosen[:max(n_pairs, len(extra))]


def plot_pairs(col, truth, lab, pairs, outdir, fmt, max_points):
    ncols = min(4, len(pairs))
    nrows = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    n = len(next(iter(col.values())))
    rng = np.random.default_rng(1)
    idx = rng.choice(n, min(n, max_points), replace=False)
    for ax, (a, b) in zip(axes.flat, pairs):
        x, y = col[a], col[b]
        r = np.corrcoef(to_unc(a, x), to_unc(b, y))[0, 1]
        ax.scatter(x[idx], y[idx], s=8, color=POST, alpha=0.35, lw=0,
                   rasterized=True)
        ax.scatter([x.mean()], [y.mean()], s=60, color=POST,
                   edgecolor=SURFACE, lw=2, zorder=3)
        ax.axvline(truth[a], color=TRUTH, lw=1, alpha=0.6)
        ax.axhline(truth[b], color=TRUTH, lw=1, alpha=0.6)
        ax.scatter([truth[a]], [truth[b]], s=90, marker="X", color=TRUTH,
                   edgecolor=SURFACE, lw=1.5, zorder=4)
        ax.set_xlabel(label(a))
        ax.set_ylabel(label(b))
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"r = {r:+.2f}", loc="left", color=INK_2, fontsize=9)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=4, color=POST, alpha=0.6,
                   label="posterior draws"),
        plt.Line2D([], [], ls="", marker="o", ms=7, color=POST,
                   markeredgecolor=SURFACE, label="posterior mean"),
        plt.Line2D([], [], ls="", marker="X", ms=9, color=TRUTH,
                   markeredgecolor=SURFACE, label="truth")]
    fig.suptitle(f"{lab}: pairwise posteriors "
                 "(r = correlation on the unconstrained scale)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    path = os.path.join(outdir, f"{lab}-pairs.{fmt}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_z(sets, outdir, fmt):
    order = [n for g in GROUPS for n in g[2]]
    names = [n for n in order if any(n in c and n in t for _, c, t in sets)]
    k = len(sets)
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * len(names) + 1.4))
    ax.axvspan(-1, 1, color="#f0efec", lw=0, zorder=0)
    for v in (-2, 2):
        ax.axvline(v, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ypos = np.arange(len(names))[::-1]
    rows = []
    for s, (lab, col, truth) in enumerate(sets):
        zs, ys = [], []
        for y, n in zip(ypos, names):
            if n in col and n in truth:
                st = summarise(col[n], truth[n])
                zs.append(st["z"])
                ys.append(y)
                rows.append({"set": lab, "parameter": n, "truth": truth[n], **st})
        ax.scatter(zs, ys, s=36, marker=SET_MARKERS[s % 8],
                   color=SET_COLOURS[s % 8], edgecolor=SURFACE, lw=1,
                   label=legend_label(lab), zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label(n) for n in names], fontsize=7.5)
    ax.set_ylim(-0.8, len(names) - 0.2)
    lim = max(3, np.ceil(max(abs(r["z"]) for r in rows) + 0.3))
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("z = (posterior mean − truth) / posterior sd")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Recovery of global parameters (shaded: |z| < 1; dashed: |z| = 2)",
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
    ap.add_argument("-o", "--outdir", default="recovery-plots")
    ap.add_argument("--n-pairs", type=int, default=8,
                    help="number of pairwise plots per set (default 8)")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B",
                    help="extra pair of CSV column names, e.g. xi.1:sens.2.1")
    ap.add_argument("--derived", action="store_true",
                    help="also plot sens_final densities")
    ap.add_argument("--max-points", type=int, default=2000,
                    help="maximum draws shown in each scatter plot")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    for s in args.set:
        if len(s) < 2:
            ap.error("--set needs a truth JSON and at least one CSV")
    if args.label and len(args.label) != len(args.set):
        ap.error("give one --label per --set")
    extra = []
    for p in args.pair:
        a, _, b = p.partition(":")
        if not b:
            ap.error(f"--pair {p}: expected A:B")
        extra.append((a, b))
    os.makedirs(args.outdir, exist_ok=True)

    groups = GROUPS + ([DERIVED_GROUP] if args.derived else [])
    sets = []
    for i, (truth_path, *csvs) in enumerate(args.set):
        lab = args.label[i] if args.label else default_label(truth_path)
        col, truth, nchains = load_set(truth_path, csvs)
        for a, b in extra:
            for n in (a, b):
                if n not in col:
                    raise SystemExit(f"--pair: no column {n} in {csvs[0]}")
        ndraws = len(next(iter(col.values())))
        print(f"{lab}: {ndraws} post-warmup draws from {nchains} chain(s)")
        for g in groups:
            p = plot_density_group(col, truth, lab, g, args.outdir, args.format)
            if p:
                print("  wrote", p)
        pairs = choose_pairs(col, truth, args.n_pairs, extra)
        print("  pairs:", ", ".join(f"{a}/{b}" for a, b in pairs))
        print("  wrote", plot_pairs(col, truth, lab, pairs, args.outdir,
                                    args.format, args.max_points))
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

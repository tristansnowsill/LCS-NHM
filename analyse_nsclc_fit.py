#!/usr/bin/env python3
"""Analyse a multi-chain fit of nsclc-model-marginal.stan to real data.

Writes to the output directory:

  summary.csv            posterior summary for every global parameter, the
                         final-screen sensitivities and the derived quantities
                         below: mean, sd, 5%/50%/95% quantiles, rank-normalised
                         split R-hat, bulk and tail ESS (Vehtari et al. 2021),
                         and the prior mean/90% interval for comparison
  density-<group>.png    posterior densities with the prior overlaid
  pairs.png              the most correlated pairs of global parameters (on
                         the unconstrained scale), plus (m, c)
  correlation.png        correlation matrix of the global parameters on the
                         unconstrained scale
  trace.png              trace plots of the global parameters, by chain

and prints per-chain sampler diagnostics (divergences, tree depth, E-BFMI,
step size), a summary of the random effects eta (worst R-hat and ESS), and
the summary table.

Derived quantities (natural history without screening, at the posterior
draws, averaged over the random effect z = exp(sigma_re * e - sigma_re^2 / 2),
e ~ N(0, 1), by Gauss-Hermite quadrature; a cancer starts in stage IA1):
  p_clin.k       probability that it presents clinically in stage k
  sojourn_total  mean years from onset (IA1) to clinical presentation

Priors follow nsclc-model-marginal.stan and are summarised by simulation,
including the ordering constraints: sens (ordered within each arm) and xi
(positive_ordered) by rejection from the independent priors, and m > 0 for
the bivariate normal prior on (m, c). The ordering pulls the implied prior
for early-stage sensitivity well below the nominal Beta(14, 6) mean of 0.7.

Stages 1-7 are labelled IA1, IA2, IA3, IB, II, III, IV; arm 1 is LDCT and
arm 2 is CXR.

With --psis-weights (the psis-weights.csv of psis_reweight.py for this fit),
the posterior summaries (mean, sd, quantiles), densities, correlations and
pair plots are PSIS-weighted (importance-weighted to the full data); see
psis_weights.py for the definitions (quantiles: np.quantile with
method="inverted_cdf" and weights). The chains are matched to the weights by
lp__. Sampler diagnostics, R-hat, ESS and trace plots stay unweighted: they
describe the MCMC run (see psis_reweight.py for the PSIS diagnostics).

Usage:
  python analyse_nsclc_fit.py -o nsclc-real-fit output/hpc/empirical-nsclc.thinned_[1-4].csv
  python analyse_nsclc_fit.py -o nsclc-real-fit-psis --psis-weights psis-nsclc/psis-weights.csv \\
      output/hpc/empirical-nsclc.thinned_[1-4].csv
"""

import argparse
import csv
import os

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import plot_recovery as pr        # palette, scales, CSV reader
import psis_weights as pw
from analyse_sclc_fit import diagnostics

SURFACE, INK, INK_2, MUTED, GRID = pr.SURFACE, pr.INK, pr.INK_2, pr.MUTED, pr.GRID
POST, POST_LIGHT = pr.POST, pr.POST_LIGHT
CHAIN_COLOURS = pr.SET_COLOURS[:4]   # validated categorical slots 1-4
PRIOR = INK_2

STAGES = ["IA1", "IA2", "IA3", "IB", "II", "III", "IV"]
ARMS = {1: "LDCT", 2: "CXR"}

GLOBALS = (["m_NSCLC", "c_NSCLC", "b_NSCLC_plco", "sigma_re",
            "log_or_sens_final.1", "log_or_sens_final.2"]
           + [f"b_NSCLC_cons.{k}" for k in range(1, 8)]
           + [f"lambda.{k}" for k in range(1, 7)]
           + [f"xi.{k}" for k in range(1, 8)]
           + [f"sens.{g}.{k}" for g in (1, 2) for k in range(1, 8)])
SENS_FINAL = [f"sens_final.{g}.{k}" for g in (1, 2) for k in range(1, 8)]
DERIVED = [f"p_clin.{k}" for k in range(1, 8)] + ["sojourn_total"]

GROUPS = [
    ("scalars", "Incidence, PLCO effect, random effect and final-screen effect",
     GLOBALS[:6], 3),
    ("b_cons", "Prevalence-screen stage intercepts", [f"b_NSCLC_cons.{k}" for k in range(1, 8)], 4),
    ("lambda", "Progression rates", [f"lambda.{k}" for k in range(1, 7)], 3),
    ("xi", "Clinical presentation rates", [f"xi.{k}" for k in range(1, 8)], 4),
    ("sens", "Screen sensitivity by stage (rows: LDCT, CXR)",
     [f"sens.{g}.{k}" for g in (1, 2) for k in range(1, 8)], 7),
    ("sens_final", "Final-screen sensitivity by stage (rows: LDCT, CXR)", SENS_FINAL, 7),
    ("derived", "Natural history without screening (averaged over the random effect)", DERIVED, 4),
]

# Prior means as in nsclc-model-marginal.stan (corrected 2026-11-23; the fits
# before then used 1.56143 / 0.80634 and 0.08692 / 0.41317 for IA and III)
LAMBDA_MEAN = np.array([1.480541, 1.480541, 1.480541, 1.375, 2.021739, 0.855147])
XI_MEAN = np.array([0.082418, 0.082418, 0.082418, 0.1875, 0.152174, 0.438179, 1.351351])
SENS_BETA = [(14.0, 6.0)] * 4 + [(16.0, 4.0), (18.0, 2.0), (19.0, 1.0)]


def label(name, short=False):
    p = name.split(".")
    base, idx = p[0], [int(i) for i in p[1:]]
    if base == "m_NSCLC":
        return "$m$" if short else "$m$ (Gompertz shape)"
    if base == "c_NSCLC":
        return "$c$" if short else "$c$ (Gompertz log rate)"
    if base == "b_NSCLC_plco":
        return r"$b_\mathrm{plco}$"
    if base == "sigma_re":
        return r"$\sigma_\mathrm{re}$"
    if base == "log_or_sens_final":
        return f"log OR final, {ARMS[idx[0]]}" if short else f"log OR final screen, {ARMS[idx[0]]}"
    if base == "b_NSCLC_cons":
        return rf"$b_\mathrm{{cons}}$ {STAGES[idx[0] - 1]}"
    if base == "lambda":
        return rf"$\lambda$ {STAGES[idx[0] - 1]}$\to${STAGES[idx[0]]}"
    if base == "xi":
        return rf"$\xi$ {STAGES[idx[0] - 1]}"
    if base == "sens":
        return f"sens {ARMS[idx[0]]} {STAGES[idx[1] - 1]}"
    if base == "sens_final":
        return f"final sens {ARMS[idx[0]]} {STAGES[idx[1] - 1]}"
    if base == "p_clin":
        return f"P(present at {STAGES[idx[0] - 1]})"
    if base == "sojourn_total":
        return "mean preclinical years"
    return name


def scale(name):
    if name.startswith("p_clin"):
        return "logit"
    if name == "sojourn_total":
        return "log"
    return pr.scale(name)


def density(name, x, grid):
    return pr.density(name, x, grid) if scale(name) != "identity" else gaussian_kde(x)(grid)


# ---- priors (by simulation, with the ordering constraints)

def prior_draws(n=100_000, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    # (m, c): bivariate normal, m > 0
    mc = rng.multivariate_normal([2.0, -7.5], [[1.0, -0.8], [-0.8, 25.0 / 16.0]], size=2 * n)
    mc = mc[mc[:, 0] > 0][:n]
    out["m_NSCLC"], out["c_NSCLC"] = mc[:, 0], mc[:, 1]
    out["b_NSCLC_plco"] = rng.gamma(10.0, 1 / 10.0, n)
    out["sigma_re"] = rng.gamma(16.0, 0.607 / 16.0, n)
    for g in (1, 2):
        out[f"log_or_sens_final.{g}"] = rng.gamma(4.0, 0.082 / 4.0, n)
    for k in range(7):
        out[f"b_NSCLC_cons.{k + 1}"] = rng.normal(-8.0, 2.0, n)
    for k in range(6):
        out[f"lambda.{k + 1}"] = rng.gamma(4.0, LAMBDA_MEAN[k] / 4.0, n)
    # xi: independent gammas restricted to increasing values
    xs = []
    while sum(len(x) for x in xs) < n:
        x = rng.gamma(4.0, XI_MEAN / 4.0, size=(n, 7))
        xs.append(x[(np.diff(x, axis=1) > 0).all(1)])
    xi = np.concatenate(xs)[:n]
    for k in range(7):
        out[f"xi.{k + 1}"] = xi[:, k]
    # sens: independent betas restricted to increasing values, per arm
    for g in (1, 2):
        ss = []
        while sum(len(s) for s in ss) < n:
            s = np.column_stack([rng.beta(a, b, n) for a, b in SENS_BETA])
            ss.append(s[(np.diff(s, axis=1) > 0).all(1)])
        s = np.concatenate(ss)[:n]
        lor = out[f"log_or_sens_final.{g}"]
        for k in range(7):
            out[f"sens.{g}.{k + 1}"] = s[:, k]
            lo = np.log(s[:, k]) - np.log1p(-s[:, k]) + lor
            out[f"sens_final.{g}.{k + 1}"] = 1 / (1 + np.exp(-lo))
    add_derived(out)
    return out


def add_derived(col, n_gh=40):
    """Natural-history quantities, averaged over the random effect."""
    e, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    lam = np.array([col[f"lambda.{k}"] for k in range(1, 7)])          # 6 x S
    xi = np.array([col[f"xi.{k}"] for k in range(1, 8)])               # 7 x S
    sig = np.asarray(col["sigma_re"])
    p_clin = np.zeros_like(xi)
    soj = np.zeros_like(sig)
    for ej, wj in zip(e, w):
        z = np.exp(sig * ej - 0.5 * sig ** 2)
        reach = np.ones_like(sig)
        for k in range(7):
            out_rate = xi[k] + (z * lam[k] if k < 6 else 0.0)
            p_clin[k] += wj * reach * xi[k] / out_rate
            soj += wj * reach / out_rate
            if k < 6:
                reach = reach * z * lam[k] / out_rate
    for k in range(7):
        col[f"p_clin.{k + 1}"] = p_clin[k]
    col["sojourn_total"] = soj


# ---- plots

def plot_density_groups(col, pdraws, outdir, w=None):
    """w: normalised PSIS weights for the pooled draws, or None (unweighted)."""
    paths = []
    for suffix, title, names, ncols in GROUPS:
        nrows = -(-len(names) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.2 * nrows), squeeze=False)
        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        for ax, name in zip(axes.flat, names):
            x = col[name]
            if w is None:
                q05, q95 = np.quantile(x, [0.05, 0.95])
                q_lo, q_hi = np.quantile(x, [0.002, 0.998])
            else:
                q05, q95 = pw.wquantile(x, [0.05, 0.95], w)
                q_lo, q_hi = pw.wquantile(x, [0.002, 0.998], w)
            span = q_hi - q_lo
            lo, hi = q_lo - 0.35 * span, q_hi + 0.35 * span
            if scale(name) == "logit":
                lo, hi = max(lo, 0.0), min(hi, 1.0)
            elif scale(name) == "log":
                lo = max(lo, 0.0)
            grid = np.linspace(q_lo, q_hi, 400)
            dens = density(name, x, grid) if w is None else pw.wdensity(scale(name), x, grid, w)
            ax.fill_between(grid, dens, where=(grid >= q05) & (grid <= q95), color=POST_LIGHT, lw=0)
            ax.plot(grid, dens, color=POST, lw=2)
            p = pdraws[name]
            sub = p[np.random.default_rng(2).choice(len(p), 20_000, replace=False)]
            g2 = np.linspace(max(lo, 1e-9), min(hi, 1 - 1e-9) if scale(name) == "logit" else hi, 300)
            inside = sub[(sub > g2[0] - 0.1 * (g2[-1] - g2[0])) & (sub < g2[-1] + 0.1 * (g2[-1] - g2[0]))]
            if len(inside) > 200:
                pd_ = density(name, inside, g2) * len(inside) / len(sub)
                ax.plot(g2, pd_, color=PRIOR, lw=1.2, ls=(0, (4, 3)))
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, dens.max() * 1.12)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_title(label(name), loc="left", fontsize=9)
            a, b = np.quantile(p, [0.05, 0.95])
            ax.set_title(f"prior {a:.2g}–{b:.2g}", loc="right", color=INK_2, fontsize=7)
        handles = [plt.Line2D([], [], color=POST, lw=2, label="posterior"),
                   plt.Rectangle((0, 0), 1, 1, color=POST_LIGHT, label="90% interval"),
                   plt.Line2D([], [], color=PRIOR, lw=1.2, ls=(0, (4, 3)), label="prior")]
        fig.suptitle(f"Real NSCLC data: {title}" + ("" if w is None else " (PSIS-weighted)"),
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0.1 if nrows == 1 else 0.06, 1, 0.94))
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        path = os.path.join(outdir, f"density-{suffix}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def unc(name, x):
    return pr.to_unc(name, x)


def plot_pairs(col, outdir, n_pairs=12, max_points=2000, w=None):
    """With PSIS weights w, correlations are weighted and the plotted points are
    a systematic resample by weight."""
    U = np.array([unc(n, col[n]) for n in GLOBALS])
    C = np.corrcoef(U) if w is None else pw.wcorr(U, w)
    ranked = sorted(((abs(C[i, j]), i, j) for i in range(len(GLOBALS))
                     for j in range(i + 1, len(GLOBALS))), reverse=True)
    pairs = [(0, 1)] + [(i, j) for _, i, j in ranked if (i, j) != (0, 1)][:n_pairs - 1]
    ncols = 4
    nrows = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows), squeeze=False)
    n = len(col[GLOBALS[0]])
    if w is None:
        idx = np.random.default_rng(1).choice(n, min(n, max_points), replace=False)
    else:
        idx = pw.resample(w, min(n, max_points), seed=1)
    for ax, (i, j) in zip(axes.flat, pairs):
        a, b = GLOBALS[i], GLOBALS[j]
        ax.scatter(col[a][idx], col[b][idx], s=6, color=POST, alpha=0.35, lw=0, rasterized=True)
        ax.set_xlabel(label(a))
        ax.set_ylabel(label(b))
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"r = {C[i, j]:+.2f}", loc="left", color=INK_2, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle("Real NSCLC data: most correlated pairs of global parameters, plus (m, c) "
                 "(r on the unconstrained scale)" + ("" if w is None else " (PSIS-weighted)"),
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, "pairs.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_correlation(col, outdir, w=None):
    U = np.array([unc(n, col[n]) for n in GLOBALS])
    C = np.corrcoef(U) if w is None else pw.wcorr(U, w)
    k = len(GLOBALS)
    fig, ax = plt.subplots(figsize=(11, 10))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#2a78d6", "#f1f0ed", "#eb6834"])
    im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
    labs = [label(n, short=True) for n in GLOBALS]
    ax.set_xticks(range(k))
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_yticks(range(k))
    ax.set_yticklabels(labs, fontsize=7)
    for i in range(k):
        for j in range(k):
            if i != j and abs(C[i, j]) >= 0.5:
                ax.text(j, i, f"{C[i, j]:+.1f}".replace("+", ""), ha="center", va="center",
                        fontsize=5, color=INK)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="correlation (unconstrained scale)")
    ax.set_title("Real NSCLC data: posterior correlations of global parameters "
                 "(|r| ≥ 0.5 written)" + ("" if w is None else " (PSIS-weighted)"),
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "correlation.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_trace(chains, outdir):
    names = GLOBALS
    ncols = 4
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 1.15 * nrows), squeeze=False, sharex=True)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        for c, (h, d) in enumerate(chains):
            ax.plot(np.arange(1, d.shape[0] + 1), d[:, h.index(name)], color=CHAIN_COLOURS[c],
                    lw=0.4, alpha=0.75)
        ax.set_title(label(name), loc="left", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(MaxNLocator(3))
    handles = [plt.Line2D([], [], color=CHAIN_COLOURS[c], lw=1.5, label=f"chain {c + 1}")
               for c in range(len(chains))]
    fig.suptitle("Real NSCLC data: post-warmup traces by chain", x=0.01, ha="left", color=INK,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.legend(handles=handles, loc="lower center", ncol=len(chains), frameon=False)
    path = os.path.join(outdir, "trace.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CmdStan CSV files, one per chain")
    ap.add_argument("-o", "--outdir", default="nsclc-real-fit")
    ap.add_argument("--psis-weights", help="psis-weights.csv from psis_reweight.py for this fit: "
                    "PSIS-weighted summaries, densities and correlations (see psis_weights.py)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    chains = [pr.read_cmdstan_csv(p) for p in args.csv]
    h0 = chains[0][0]
    if any(h != h0 for h, _ in chains):
        raise SystemExit("chains have different columns")
    if args.psis_weights:
        chain_w = pw.align(args.psis_weights, [d[:, h.index("lp__")] for h, d in chains], args.csv)
    n = min(d.shape[0] for _, d in chains)
    chains = [(h, d[:n]) for h, d in chains]
    print(f"{len(chains)} chains, {n} post-warmup draws each")
    psis_w = pw.normalise(np.concatenate([cw[:n] for cw in chain_w])) if args.psis_weights else None

    print("\nchain  step size  divergent  max depth hits  depth range  mean leapfrog  accept  E-BFMI")
    for c, (p, (h, d)) in enumerate(zip(args.csv, chains), 1):
        step = open(p).read().split("Step size = ")[1].split("\n")[0]
        col = lambda k: d[:, h.index(k)]
        td, en = col("treedepth__"), col("energy__")
        bfmi = np.sum(np.diff(en) ** 2) / np.sum((en - en.mean()) ** 2)
        print(f"{c:5d}  {float(step):9.4f}  {int(col('divergent__').sum()):9d}  "
              f"{int((td >= 10).sum()):14d}  {int(td.min())}-{int(td.max()):<9d}  "
              f"{col('n_leapfrog__').mean():13.1f}  {col('accept_stat__').mean():6.3f}  {bfmi:6.2f}")

    stack = {k: np.array([d[:, h.index(k)] for h, d in chains]) for k in ["lp__"] + GLOBALS + SENS_FINAL}
    add_derived(stack)
    pooled = {k: v.ravel() for k, v in stack.items()}
    pdraws = prior_draws()

    # Random effects: worst diagnostics
    eta_names = [k for k in h0 if k.startswith("eta.")]
    worst = []
    for k in eta_names:
        x = np.array([d[:, h0.index(k)] for _, d in chains])
        worst.append(diagnostics(x))
    worst = np.array(worst)
    print(f"\neta ({len(eta_names)} random effects): max R-hat {worst[:, 0].max():.3f}, "
          f"min bulk ESS {worst[:, 1].min():.0f}, min tail ESS {worst[:, 2].min():.0f}, "
          f"R-hat > 1.01: {(worst[:, 0] > 1.01).sum()}")

    if psis_w is not None:
        print(f"\nPSIS-weighted posterior summaries ({args.psis_weights}; Kish ESS of the weights "
              f"{pw.kish_ess(psis_w):.0f} of {len(psis_w)}); R-hat and ESS are of the unweighted chains")
    rows = []
    for k in ["lp__"] + GLOBALS + SENS_FINAL + DERIVED:
        x = stack[k]
        rhat, bulk, tail = diagnostics(x)
        v = x.ravel()
        if psis_w is None:
            q = np.quantile(v, [0.05, 0.5, 0.95])
            mean, sd = v.mean(), v.std(ddof=1)
        else:
            q = pw.wquantile(v, [0.05, 0.5, 0.95], psis_w)
            mean, sd = pw.wmean(v, psis_w), pw.wsd(v, psis_w)
        p = pdraws.get(k)
        rows.append({
            "parameter": k, "label": label(k).replace("$", ""), "mean": mean, "sd": sd,
            "q05": q[0], "q50": q[1], "q95": q[2],
            "rhat": rhat, "ess_bulk": bulk, "ess_tail": tail,
            "prior_mean": p.mean() if p is not None else np.nan,
            "prior_q05": np.quantile(p, 0.05) if p is not None else np.nan,
            "prior_q95": np.quantile(p, 0.95) if p is not None else np.nan,
        })
    path = os.path.join(args.outdir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"\n{'parameter':22s} {'mean':>9s} {'sd':>8s} {'5%':>9s} {'50%':>9s} {'95%':>9s} "
          f"{'R-hat':>6s} {'bulk':>6s} {'tail':>6s}   prior mean [90%]")
    for r in rows:
        pm = (f"{r['prior_mean']:.3g} [{r['prior_q05']:.3g}, {r['prior_q95']:.3g}]"
              if np.isfinite(r["prior_mean"]) else "")
        print(f"{r['parameter']:22s} {r['mean']:9.4g} {r['sd']:8.3g} {r['q05']:9.4g} {r['q50']:9.4g} "
              f"{r['q95']:9.4g} {r['rhat']:6.3f} {r['ess_bulk']:6.0f} {r['ess_tail']:6.0f}   {pm}")

    print("\nwrote", path)
    for p in plot_density_groups(pooled, pdraws, args.outdir, w=psis_w):
        print("wrote", p)
    print("wrote", plot_pairs(pooled, args.outdir, w=psis_w))
    print("wrote", plot_correlation(pooled, args.outdir, w=psis_w))
    print("wrote", plot_trace(chains, args.outdir))


if __name__ == "__main__":
    main()

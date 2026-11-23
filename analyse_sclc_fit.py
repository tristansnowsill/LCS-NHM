#!/usr/bin/env python3
"""Analyse a multi-chain fit of sclc-model.stan to real data.

Writes to the output directory:

  summary.csv            posterior summary for every global parameter and the
                         derived quantities below: mean, sd, 5%/50%/95%
                         quantiles, rank-normalised split R-hat, bulk and tail
                         ESS (Vehtari et al. 2021), and the prior mean/90%
                         interval for comparison
  density-<group>.png    posterior densities with the prior overlaid
  corner.png             all pairwise posteriors (sensitivities on the logit
                         scale), correlations above the diagonal
  trace.png              trace plots of every global parameter, by chain

and prints per-chain sampler diagnostics (divergences, tree depth, E-BFMI,
step size) and the summary table.

Derived quantities (natural history, at the posterior draws):
  p_limited_clinical = phi[1] / (mu + phi[1])   share of limited SCLC that
                       presents clinically while still limited (no screening)
  sojourn_limited    = 1 / (mu + phi[1])        mean years in limited stage
  sojourn_extensive  = 1 / phi[2]               mean years in extensive stage

Prior densities follow sclc-model.stan, including the truncation m > 0
(about 2.3% of the mass) of the bivariate normal prior of (m, c): m is a
truncated normal, and c has the exact marginal under that truncation
(TruncatedMCMarginalC); prior draws of (m, c) are joint. They ignore the
small truncation from the ordering of logit_sens_SCLC (about 2.6% of the
mass) and of phi (phi[1] < phi[2], about 0.06%); prior draws respect the
phi ordering, and prior summaries of derived quantities are by simulation.

With --psis-weights (the psis-weights.csv of psis_reweight.py for this fit),
the posterior summaries (mean, sd, quantiles), densities, correlations and
corner-plot points are PSIS-weighted (importance-weighted to the full data);
see psis_weights.py for the definitions (quantiles: np.quantile with
method="inverted_cdf" and weights). The chains are matched to the weights by
lp__. Sampler diagnostics, R-hat, ESS and trace plots stay unweighted: they
describe the MCMC run (see psis_reweight.py for the PSIS diagnostics).

Usage:
  python analyse_sclc_fit.py -o sclc-real-fit empirical-sclc.thinned_[1-4].csv
  python analyse_sclc_fit.py -o sclc-real-fit-psis --psis-weights psis-sclc/psis-weights.csv \\
      empirical-sclc.thinned_[1-4].csv
"""

import argparse
import csv
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import plot_recovery_sclc as prs   # palette, labels, scales, density, CSV reader
import psis_weights as pw

SURFACE, INK, INK_2, MUTED, GRID = prs.SURFACE, prs.INK, prs.INK_2, prs.MUTED, prs.GRID
POST, POST_LIGHT = prs.POST, prs.POST_LIGHT
CHAIN_COLOURS = prs.SET_COLOURS[:4]   # validated categorical slots 1-4
PRIOR = INK_2

PARAMS = prs.PARAMS
DERIVED = ["p_limited_clinical", "sojourn_limited", "sojourn_extensive"]
DERIVED_LABEL = {
    "p_limited_clinical": "share presenting while limited",
    "sojourn_limited": "mean years in limited stage",
    "sojourn_extensive": "mean years in extensive stage",
}

# Priors from sclc-model.stan, as scipy distributions on the natural scale
# where possible; sensitivities are normal on the logit scale.
LOGIT_SENS_PRIOR = {1: (0.0, np.sqrt(2.0)), 2: (5.220108, 2.286569)}
MC_MEAN = np.array([2.0, -10.5])
MC_COV = np.array([[1.0, -0.8], [-0.8, 25.0 / 16.0]])


def gamma_prior(shape, mean):
    return stats.gamma(shape, scale=mean / shape)


class TruncatedMCMarginalC:
    """Marginal prior of c when (m, c) is bivariate normal (MC_MEAN, MC_COV)
    truncated to m > 0: the normal density of c times P(m > 0 | c), divided
    by P(m > 0)."""

    def __init__(self, mean, cov):
        self.mu_m, self.mu_c = mean
        self.sd_c = np.sqrt(cov[1, 1])
        self.slope = cov[0, 1] / cov[1, 1]                       # E[m | c] slope
        self.sd_m_c = np.sqrt(cov[0, 0] - cov[0, 1] ** 2 / cov[1, 1])
        self.p_keep = stats.norm.sf(-self.mu_m / np.sqrt(cov[0, 0]))

    def pdf(self, x):
        mean_m = self.mu_m + self.slope * (np.asarray(x) - self.mu_c)
        return (stats.norm.pdf(x, self.mu_c, self.sd_c)
                * stats.norm.sf(-mean_m / self.sd_m_c) / self.p_keep)


def truncated_mc_draws(rng, n):
    """n draws of (m, c) from the bivariate normal prior truncated to m > 0."""
    out = np.empty((0, 2))
    while len(out) < n:
        x = rng.multivariate_normal(MC_MEAN, MC_COV, size=n)
        out = np.concatenate([out, x[x[:, 0] > 0]])
    return out[:n]


PRIOR_DIST = {
    "m_SCLC": stats.truncnorm(-2.0, np.inf, loc=2.0, scale=1.0),
    "c_SCLC": TruncatedMCMarginalC(MC_MEAN, MC_COV),
    "b_SCLC_plco": gamma_prior(10.0, 1.0),
    "b_SCLC_cons.1": stats.norm(-7.0, 2.0),
    "b_SCLC_cons.2": stats.norm(-7.0, 2.0),
    # corrected 2026-11-23 (fits before then used means 0.277591 and 0.149588)
    "mu": gamma_prior(4.0, 0.262766),
    "phi.1": gamma_prior(4.0, 0.141599),
    "phi.2": gamma_prior(4.0, 1.960784),
    "log_or_sens_final.1": gamma_prior(4.0, np.exp(-1.5)),
    "log_or_sens_final.2": gamma_prior(4.0, np.exp(-1.5)),
}


def prior_pdf(name, x):
    if name.startswith("sens_SCLC."):
        mu, sd = LOGIT_SENS_PRIOR[int(name.split(".")[2])]
        xc = np.clip(x, 1e-12, 1 - 1e-12)
        return stats.norm.pdf(np.log(xc / (1 - xc)), mu, sd) / (xc * (1 - xc))
    if name in PRIOR_DIST:
        return PRIOR_DIST[name].pdf(x)
    return None


def prior_draws(n=200_000, seed=1):
    """Draws from the priors (marginals as in prior_pdf), for prior summaries."""
    rng = np.random.default_rng(seed)
    out = {k: d.rvs(n, random_state=rng) for k, d in PRIOR_DIST.items()
           if k not in ("m_SCLC", "c_SCLC", "phi.1", "phi.2")}
    phi = np.empty((0, 2))                                          # phi[1] < phi[2]
    while len(phi) < n:
        x = np.column_stack([PRIOR_DIST["phi.1"].rvs(n, random_state=rng),
                             PRIOR_DIST["phi.2"].rvs(n, random_state=rng)])
        phi = np.concatenate([phi, x[x[:, 0] < x[:, 1]]])
    out["phi.1"], out["phi.2"] = phi[:n].T
    out["m_SCLC"], out["c_SCLC"] = truncated_mc_draws(rng, n).T     # jointly, m > 0
    for g in (1, 2):
        for k in (1, 2):
            mu, sd = LOGIT_SENS_PRIOR[k]
            out[f"sens_SCLC.{g}.{k}"] = 1 / (1 + np.exp(-rng.normal(mu, sd, n)))
    add_derived(out)
    return out


def add_derived(col):
    mu, phi1, phi2 = col["mu"], col["phi.1"], col["phi.2"]
    col["p_limited_clinical"] = phi1 / (mu + phi1)
    col["sojourn_limited"] = 1 / (mu + phi1)
    col["sojourn_extensive"] = 1 / phi2


# ---- convergence diagnostics (Vehtari, Gelman, Simpson, Carpenter, Bürkner 2021)

def _split(x):
    """chains x draws -> 2*chains x draws/2"""
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, -n:]])


def _rhat(x):
    m, n = x.shape
    w = x.var(axis=1, ddof=1).mean()
    b = n * x.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * w + b / n) / w)


def _ess(x):
    """Multi-chain ESS with Geyer's initial monotone sequence (as in Stan)."""
    m, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(xc, 2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    w = x.var(axis=1, ddof=1).mean()
    if w == 0:
        return np.nan
    var_plus = (n - 1) / n * w + n * x.mean(axis=1).var(ddof=1) / n if m > 1 else w
    rho = 1 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    t, sums = 0, []
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        sums.append(p)
        t += 2
    sums = np.minimum.accumulate(sums)          # monotone sequence
    tau = -1 + 2 * np.sum(sums)
    return m * n / max(tau, 1.0 / np.log10(m * n))


def _z_rank(x):
    r = stats.rankdata(x, method="average").reshape(x.shape)
    return stats.norm.ppf((r - 3 / 8) / (x.size + 1 / 4))


def diagnostics(x):
    """x: chains x draws. Returns (rank-normalised split R-hat, bulk ESS, tail ESS)."""
    s = _split(x)
    z = _z_rank(s)
    zf = _z_rank(np.abs(s - np.median(s)))
    rhat = max(_rhat(z), _rhat(zf))
    bulk = _ess(z)
    q05, q95 = np.quantile(x, [0.05, 0.95])
    tail = min(_ess(_split((x <= q05).astype(float))), _ess(_split((x <= q95).astype(float))))
    return rhat, bulk, tail


# ---- plots

def plot_range_w(name, x, t, kind, w):
    """prs.plot_range with PSIS-weighted quantiles."""
    q_lo, q_hi = pw.wquantile(x, [0.002, 0.998], w)
    lo, hi = min(q_lo, t), max(q_hi, t)
    pad = 0.06 * (hi - lo) if hi > lo else 0.01
    lo, hi = lo - pad, hi + pad
    s = kind or prs.scale(name)
    if s == "logit":
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    elif s == "log":
        lo = max(lo, 0.0)
    return lo, hi, q_lo, q_hi


def density_w(name, x, grid, kind, w):
    """prs.density with PSIS weights."""
    return pw.wdensity(kind or prs.scale(name), x, grid, w)


def plot_density_groups(col, outdir, w=None):
    """w: normalised PSIS weights for the pooled draws, or None (unweighted)."""
    groups = prs.GROUPS + [("derived", "Derived natural-history quantities", DERIVED, 3)]
    paths = []
    pdraws = prior_draws()
    for suffix, title, names, ncols in groups:
        nrows = -(-len(names) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.1 * nrows), squeeze=False)
        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        for ax, name in zip(axes.flat, names):
            x = col[name]
            if w is None:
                q05, q95 = np.quantile(x, [0.05, 0.95])
                lo, hi, q_lo, q_hi = prs.plot_range(name, x, np.median(x),
                                                    kind="identity" if name in DERIVED else None)
            else:
                q05, q95 = pw.wquantile(x, [0.05, 0.95], w)
                lo, hi, q_lo, q_hi = plot_range_w(name, x, pw.wquantile(x, 0.5, w),
                                                  "identity" if name in DERIVED else None, w)
            # widen to show the prior around the posterior
            span = hi - lo
            lo, hi = lo - 0.15 * span, hi + 0.15 * span
            if prs.scale(name) == "logit":
                lo, hi = max(lo, 0.0), min(hi, 1.0)
            elif prs.scale(name) == "log" or name in DERIVED:
                lo = max(lo, 0.0)
            grid = np.linspace(q_lo, q_hi, 400)
            if w is None:
                dens = prs.density(name, x, grid, kind="log" if name in DERIVED else None)
            else:
                dens = density_w(name, x, grid, "log" if name in DERIVED else None, w)
            inside = (grid >= q05) & (grid <= q95)
            ax.fill_between(grid, dens, where=inside, color=POST_LIGHT, lw=0)
            ax.plot(grid, dens, color=POST, lw=2)
            g2 = np.linspace(max(lo, 1e-9), hi, 400)
            pd = prior_pdf(name, g2)
            if pd is None and name in pdraws:      # derived: prior by simulation
                pr = pdraws[name]
                pr = pr[(pr > g2[0]) & (pr < g2[-1])]
                pd = stats.gaussian_kde(np.log(pr))(np.log(g2)) / g2 * (len(pr) / len(pdraws[name]))
            if pd is not None:
                ax.plot(g2, pd, color=PRIOR, lw=1.2, ls=(0, (4, 3)))
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, dens.max() * 1.1)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_title(DERIVED_LABEL.get(name, prs.label(name)), loc="left")
            if name in pdraws:
                a, b = np.quantile(pdraws[name], [0.05, 0.95])
                ax.set_title(f"prior {a:.2g} to {b:.2g}", loc="right", color=INK_2, fontsize=7)
        handles = [plt.Line2D([], [], color=POST, lw=2, label="posterior"),
                   plt.Rectangle((0, 0), 1, 1, color=POST_LIGHT, label="90% interval"),
                   plt.Line2D([], [], color=PRIOR, lw=1.2, ls=(0, (4, 3)), label="prior")]
        fig.suptitle(f"Real SCLC data: {title}" + ("" if w is None else " (PSIS-weighted)"),
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0.1 if nrows == 1 else 0.05, 1, 0.95))
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        path = os.path.join(outdir, f"density-{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_corner(col, outdir, max_points=1500, w=None):
    """With PSIS weights w: weighted correlations, densities and axis ranges,
    and the scatter points are a systematic resample by weight."""
    names = PARAMS
    k = len(names)
    on_logit = {nm: prs.scale(nm) == "logit" for nm in names}
    val = {nm: prs.to_unc(nm, col[nm]) if on_logit[nm] else col[nm] for nm in names}
    kind = {nm: "identity" if on_logit[nm] else None for nm in names}
    n = len(col[names[0]])
    if w is None:
        C = np.corrcoef([prs.to_unc(n, col[n]) for n in names])
        idx = np.random.default_rng(1).choice(n, min(n, max_points), replace=False)
        rng_ = lambda nm: prs.plot_range(nm, val[nm], np.median(val[nm]), kind[nm])
    else:
        C = pw.wcorr(np.array([prs.to_unc(nm, col[nm]) for nm in names]), w)
        idx = pw.resample(w, min(n, max_points), seed=1)
        rng_ = lambda nm: plot_range_w(nm, val[nm], pw.wquantile(val[nm], 0.5, w), kind[nm], w)
    lims = {nm: rng_(nm)[:2] for nm in names}
    lab = lambda nm: ("logit " if on_logit[nm] else "") + prs.label(nm, short=True)
    fig, axes = plt.subplots(k, k, figsize=(1.35 * k, 1.35 * k))
    for i, yn in enumerate(names):
        for j, xn in enumerate(names):
            ax = axes[i, j]
            ax.tick_params(labelsize=6, length=2)
            if i < j:
                r = C[i, j]
                ax.axis("off")
                ax.text(0.5, 0.5, f"{r:+.2f}", transform=ax.transAxes, ha="center", va="center",
                        fontsize=9 if abs(r) >= 0.5 else 7.5,
                        fontweight="bold" if abs(r) >= 0.5 else "normal",
                        color=INK if abs(r) >= 0.3 else MUTED)
                continue
            if i == j:
                lo, hi, q_lo, q_hi = rng_(xn)
                grid = np.linspace(q_lo, q_hi, 300)
                dens = (prs.density(xn, val[xn], grid, kind[xn]) if w is None
                        else density_w(xn, val[xn], grid, kind[xn], w))
                ax.plot(grid, dens, color=POST, lw=1.5)
                ax.set_xlim(lo, hi)
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            else:
                ax.scatter(val[xn][idx], val[yn][idx], s=2, color=POST, alpha=0.3, lw=0,
                           rasterized=True)
                ax.set_xlim(*lims[xn])
                ax.set_ylim(*lims[yn])
                ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            if i == k - 1:
                ax.set_xlabel(lab(xn), fontsize=8)
                for t in ax.get_xticklabels():
                    t.set_rotation(45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(lab(yn), fontsize=8)
            elif i != j:
                ax.set_yticklabels([])
    fig.suptitle("Real SCLC data: all pairwise posteriors. Above the diagonal: correlation on the "
                 "unconstrained scale (bold if |r| ≥ 0.5). Sensitivities on the logit scale. "
                 "L = limited, E = extensive" + ("" if w is None else " (PSIS-weighted)"),
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.12, hspace=0.12)
    fig.align_ylabels(axes[:, 0])
    path = os.path.join(outdir, "corner.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_trace(chains, outdir):
    names = PARAMS
    ncols = 2
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 1.25 * nrows), squeeze=False, sharex=True)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        for c, (h, d) in enumerate(chains):
            ax.plot(np.arange(1, d.shape[0] + 1), d[:, h.index(name)], color=CHAIN_COLOURS[c],
                    lw=0.5, alpha=0.8)
        ax.set_title(prs.label(name), loc="left", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(3))
    handles = [plt.Line2D([], [], color=CHAIN_COLOURS[c], lw=1.5, label=f"chain {c + 1}")
               for c in range(len(chains))]
    fig.suptitle("Real SCLC data: post-warmup traces by chain", x=0.01, ha="left", color=INK,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.legend(handles=handles, loc="lower center", ncol=len(chains), frameon=False)
    path = os.path.join(outdir, "trace.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CmdStan CSV files, one per chain")
    ap.add_argument("-o", "--outdir", default="sclc-real-fit")
    ap.add_argument("--psis-weights", help="psis-weights.csv from psis_reweight.py for this fit: "
                    "PSIS-weighted summaries, densities and correlations (see psis_weights.py)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    chains = [prs.read_cmdstan_csv(p) for p in args.csv]
    h0 = chains[0][0]
    if any(h != h0 for h, _ in chains):
        raise SystemExit("chains have different columns")
    if args.psis_weights:
        chain_w = pw.align(args.psis_weights, [d[:, h.index("lp__")] for h, d in chains], args.csv)
    n = min(d.shape[0] for _, d in chains)
    chains = [(h, d[:n]) for h, d in chains]
    print(f"{len(chains)} chains, {n} post-warmup draws each")
    psis_w = pw.normalise(np.concatenate([cw[:n] for cw in chain_w])) if args.psis_weights else None

    # Per-chain sampler diagnostics
    print("\nchain  step size  divergent  max depth hits  depth range  mean leapfrog  accept  E-BFMI")
    for c, (p, (h, d)) in enumerate(zip(args.csv, chains), 1):
        step = open(p).read().split("Step size = ")[1].split("\n")[0]
        col = lambda k: d[:, h.index(k)]
        td, en = col("treedepth__"), col("energy__")
        maxd = 10
        bfmi = np.sum(np.diff(en) ** 2) / np.sum((en - en.mean()) ** 2)
        print(f"{c:5d}  {float(step):9.4f}  {int(col('divergent__').sum()):9d}  "
              f"{int((td >= maxd).sum()):14d}  {int(td.min())}-{int(td.max()):<9d}  "
              f"{col('n_leapfrog__').mean():13.1f}  {col('accept_stat__').mean():6.3f}  {bfmi:6.2f}")

    # Pooled draws and diagnostics
    stack = {k: np.array([d[:, h.index(k)] for h, d in chains]) for k in PARAMS + ["lp__"]}
    sc = {k: v.copy() for k, v in stack.items()}
    add_derived(sc)
    pooled = {k: v.ravel() for k, v in sc.items()}
    pdraws = prior_draws()

    if psis_w is not None:
        print(f"\nPSIS-weighted posterior summaries ({args.psis_weights}; Kish ESS of the weights "
              f"{pw.kish_ess(psis_w):.0f} of {len(psis_w)}); R-hat and ESS are of the unweighted chains")
    rows = []
    for k in ["lp__"] + PARAMS + DERIVED:
        x = sc[k]
        rhat, bulk, tail = diagnostics(x)
        v = x.ravel()
        if psis_w is None:
            q = np.quantile(v, [0.05, 0.5, 0.95])
            mean, sd = v.mean(), v.std(ddof=1)
        else:
            q = pw.wquantile(v, [0.05, 0.5, 0.95], psis_w)
            mean, sd = pw.wmean(v, psis_w), pw.wsd(v, psis_w)
        pr = pdraws.get(k)
        rows.append({
            "parameter": k, "mean": mean, "sd": sd,
            "q05": q[0], "q50": q[1], "q95": q[2],
            "rhat": rhat, "ess_bulk": bulk, "ess_tail": tail,
            "prior_mean": pr.mean() if pr is not None else np.nan,
            "prior_q05": np.quantile(pr, 0.05) if pr is not None else np.nan,
            "prior_q95": np.quantile(pr, 0.95) if pr is not None else np.nan,
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
    for p in plot_density_groups(pooled, args.outdir, w=psis_w):
        print("wrote", p)
    print("wrote", plot_corner(pooled, args.outdir, w=psis_w))
    print("wrote", plot_trace(chains, args.outdir))


if __name__ == "__main__":
    main()

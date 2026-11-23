"""PSIS importance weights for the downstream analyses (analyse_nsclc_fit.py,
analyse_sclc_fit.py, nsclc_ppc.py, sclc_ppc.py, make_tables.py).

Weights file
------------
One CSV per fit, written by psis_reweight.py, with one row per post-warmup
draw:

  chain, draw, lp__, log_ratio, weight

chain and draw are 1-based; lp__ is the sampler's log density at the draw
(as in the chain CSV); log_ratio = lp_full - lp__; weight is the
Pareto-smoothed importance weight, normalised to sum to 1 over the file.

Every consumer matches its chains to the file by lp__ (each input chain must
equal one of the file's chains draw by draw), so the weights can't be applied
in the wrong order: a chain that matches none of the file's chains, or a
different number of draws, is an error. Generated-quantities CSVs contain no
lp__, so their weights are matched through the fit CSV named as
fitted_params in the GQ file's header. Weights are renormalised over the
draws a script uses.

Weighted summaries
------------------
With weights w (normalised, w_s >= 0, sum 1):

  mean          sum_s w_s x_s
  sd            sqrt(sum_s w_s (x_s - mean)^2)   (no small-sample correction;
                with S = 8,000 draws the correction would be negligible)
  quantiles     np.quantile(x, q, method="inverted_cdf", weights=w): the
                smallest draw x with weighted empirical CDF >= q. This is the
                only weighted quantile definition used. It depends only on the
                weighted empirical CDF, so it does not depend on the order of
                tied draws (many weighted interpolating definitions do; the
                integer predictive counts have heavy ties). See
                https://www.practicalsignificance.com/posts/weighted-quantile-weirdness/
                Unweighted results keep their own definitions (np.quantile's
                default "linear" in most places), so weighted and unweighted
                quantiles differ slightly by definition as well as by the
                weights. Measured on the corrected NSCLC fit (8,000 draws,
                unweighted): linear and inverted_cdf differ by at most 0.004
                posterior SD for continuous parameters and not at all for the
                predictive cell counts, well below the Monte Carlo error.
  proportions   sum_s w_s 1{condition}   (e.g. posterior predictive p-values)
  correlations  weighted Pearson correlation (np.cov with aweights)
  densities     scipy.stats.gaussian_kde with weights (Scott bandwidth from
                the Kish effective sample size, as scipy computes it)

The quantile function needs NumPy >= 2.0 (for weights=).

When no weights file is given, the scripts make their original unweighted
calls; nothing here is called with equal weights.
"""

import csv
import os
import re

import numpy as np
from scipy.stats import gaussian_kde

LP_RTOL, LP_ATOL = 1e-10, 1e-6


def check_numpy():
    major = int(np.__version__.split(".")[0])
    if major < 2:
        raise SystemExit(f"PSIS weights need NumPy >= 2.0 for weighted quantiles "
                         f"(np.quantile(..., weights=)); this is NumPy {np.__version__}")


def read_weights(path):
    """{chain: (lp__ array, weight array)} in draw order."""
    check_numpy()
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r["chain"]), []).append(
                (int(r["draw"]), float(r["lp__"]), float(r["weight"])))
    out = {}
    for c, v in rows.items():
        v.sort()
        draws = np.array([d for d, _, _ in v])
        if not np.array_equal(draws, np.arange(1, len(v) + 1)):
            raise SystemExit(f"{path}: chain {c} draws are not 1..{len(v)}")
        out[c] = (np.array([lp for _, lp, _ in v]), np.array([w for _, _, w in v]))
    return out


def match_chains(path, lp_chains, labels):
    """The weights-file chain matched to each input chain by lp__.

    lp_chains: list of lp__ arrays (one per input chain, in the script's
    order); labels: names for error messages. Returns (chain ids, weight
    arrays as in the file, not renormalised). A chain matches only if every
    lp__ equals the file chain's at the same position; weights are then
    taken by position."""
    file_chains = read_weights(path)
    used, ids, out = set(), [], []
    for lp, lab in zip(lp_chains, labels):
        lp = np.asarray(lp, float)
        match = [c for c, (flp, _) in file_chains.items()
                 if c not in used and len(flp) == len(lp)
                 and np.allclose(flp, lp, rtol=LP_RTOL, atol=LP_ATOL)]
        if not match:
            sizes = {c: len(v[0]) for c, v in file_chains.items()}
            raise SystemExit(f"{lab}: its {len(lp)} draws match no chain in {path} "
                             f"(draws per chain {sizes}, chains already matched {sorted(used)}) "
                             "by lp__; check that the weights were computed for this fit")
        used.add(match[0])
        ids.append(match[0])
        out.append(file_chains[match[0]][1])
        print(f"PSIS weights: {lab} matched to chain {match[0]} of {path}")
    return ids, out


def align(path, lp_chains, labels):
    """Weights for each input chain, matched to the weights file by lp__
    (see match_chains). Returns a list of weight arrays (as in the file, not
    renormalised)."""
    return match_chains(path, lp_chains, labels)[1]


def normalise(w):
    w = np.asarray(w, float)
    if (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
        raise SystemExit("PSIS weights must be finite, non-negative and not all zero")
    return w / w.sum()


def kish_ess(w):
    w = normalise(w)
    return 1.0 / np.sum(w ** 2)


# ---- lp__ of a fit CSV (for generated quantities, via fitted_params)

def fitted_params_path(gq_path):
    """The fitted_params CSV named in a GQ CSV header (resolved as sclc_ppc.warmup_rows does)."""
    fitted = None
    with open(gq_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*fitted_params = (\S+)", line)
            if m:
                fitted = m.group(1)
    if fitted is None:
        raise SystemExit(f"{gq_path}: no fitted_params in the header, so its draws can't be "
                         "matched to PSIS weights")
    if not os.path.isabs(fitted) and not os.path.exists(fitted):
        fitted = os.path.join(os.path.dirname(gq_path), fitted)
    return fitted


def read_lp(path):
    """Post-warmup lp__ of a CmdStan CSV (drops saved warmup; skips a partial last row)."""
    num_warmup, save_warmup = 1000, False
    header, lp = None, []
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
                header = line.rstrip("\n").split(",")
                i = header.index("lp__")
                continue
            v = line.rstrip("\n").split(",")
            if len(v) == len(header):
                lp.append(float(v[i]))
    lp = np.array(lp)
    return lp[num_warmup:] if save_warmup else lp


def gq_weights(path, gq_paths):
    """Weights for generated-quantities CSVs, matched through their fitted_params fits."""
    fits = [fitted_params_path(p) for p in gq_paths]
    return align(path, [read_lp(p) for p in fits], [f"{g} (fit {f})" for g, f in zip(gq_paths, fits)])


# ---- weighted statistics (w normalised)

def wmean(x, w):
    return float(np.sum(w * x))


def wsd(x, w):
    m = np.sum(w * x)
    return float(np.sqrt(np.sum(w * (x - m) ** 2)))


def wquantile(x, q, w, axis=None):
    return np.quantile(x, q, method="inverted_cdf", weights=w, axis=axis)


def wprop(cond, w):
    return float(np.sum(w * cond))


def wcorr(U, w):
    """Weighted correlation matrix of the rows of U."""
    C = np.cov(U, aweights=w)
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def wdensity(kind, x, grid, w):
    """Weighted counterpart of plot_recovery(_sclc).density: reflected KDE for
    probabilities, log-scale KDE for positive parameters, plain KDE otherwise."""
    if kind == "logit":
        h = gaussian_kde(x, weights=w).factor * wsd(x, w)
        xr = np.concatenate([x, -x, 2 - x])
        wr = np.tile(w, 3) / 3
        kde = gaussian_kde(xr, bw_method=h / wsd(xr, wr), weights=wr)
        return 3 * kde(grid)
    if kind == "log":
        return gaussian_kde(np.log(x), weights=w)(np.log(grid)) / grid
    return gaussian_kde(x, weights=w)(grid)


def systematic(w, n, rng):
    """n draw indices by systematic resampling (one uniform from rng), in index order."""
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), u), len(w) - 1)


def resample(w, n, seed):
    """n draw indices by systematic resampling (for scatter plots only)."""
    return systematic(w, n, np.random.default_rng(seed))

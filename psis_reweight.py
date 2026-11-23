#!/usr/bin/env python3
"""Pareto-smoothed importance sampling (PSIS) from a fit to thinned data to
the full data, for nsclc-model-marginal.stan or sclc-model.stan.

Inputs: the CSVs written by full-log-density.R (every column of the chain
CSV plus lp_full, the full-data log density at the draw), named
<anything>_<chain>.<chunk>.csv or <anything>_<chain>.csv, e.g.
output/empirical-nsclc.psis-input_3.17.csv. Within a chain, chunks are taken
in chunk order and must hold consecutive draws; with --fit (the chain CSVs of
the fit, in chain order) every draw is checked against the fit by lp__, which
also catches a missing or misordered chunk.

Log ratios: log r = lp_full - lp__. Both log densities include the prior,
the Jacobian and (NSCLC) the cancer patients' eta prior, which cancel; any
constant offset is removed by normalising the weights.

PSIS (Vehtari, Simpson, Gelman, Yao, Gabry 2024, "Pareto smoothed importance
sampling", JMLR 25(72)):
  r_eff   relative efficiency of the ratios: ESS of exp(log r) over split
          chains (Geyer's initial monotone sequence; equal to
          posterior::ess_mean, which loo::relative_eff uses) divided by S;
          per-chain ESS summed if chain lengths differ
  M       tail size ceiling(min(0.2 S, 3 sqrt(S / r_eff))) (as loo's n_pareto)
  k-hat   generalised Pareto shape fitted to the M largest ratios (Zhang and
          Stephens 2009, with the weakly informative prior adjustment, as
          loo::psis); the M largest ratios are replaced by the fitted
          distribution's expected order statistics, truncated at the largest
          raw ratio. Reliable if k-hat < min(1 - 1/log10 S, 0.7)
  ESS     r_eff / sum(w^2) (PSIS effective sample size)
  MCSE    of a weighted mean: sqrt(sum w^2 (x - mean)^2 / r_eff)

Writes to --outdir:
  psis-weights.csv      chain, draw, lp__, log_ratio, weight (normalised over
                        all draws), for --psis-weights in analyse_nsclc_fit.py,
                        analyse_sclc_fit.py, nsclc_ppc.py, sclc_ppc.py and
                        make_tables.py
  psis-summary.csv      for each parameter, transformed parameter and derived
                        quantity: the thinned-data posterior (unweighted) and
                        the reweighted (full-data) posterior: mean, sd,
                        5%/50%/95% quantiles; the shift of the mean in
                        thinned-posterior SDs and its MCSE in the same units
  psis-diagnostics.txt  the diagnostics printed to the console

Quantiles in psis-summary.csv are np.quantile(..., method="inverted_cdf") for
both posteriors (with weights for the reweighted one), so the shifts are like
for like; see psis_weights.py for the weighted summaries.

Usage (--fit takes several files, so put it after the inputs):
  python psis_reweight.py -o psis-nsclc output/empirical-nsclc.psis-input_*.csv \\
      --fit output/hpc/empirical-nsclc.thinned_[1-4].csv
"""

import argparse
import csv
import os
import re

import numpy as np
from scipy.special import logsumexp

import psis_weights as pw
from analyse_sclc_fit import _ess, _split


# ---- PSIS

def gpdfit(x):
    """Generalised Pareto fit (Zhang and Stephens 2009) to sorted exceedances
    x, with the weakly informative adjustment of k, as loo's gpdfit.
    Returns (k, sigma)."""
    n = len(x)
    if x[-1] <= 0:                         # all tail ratios equal
        return -np.inf, 0.0
    prior, m = 3, 30 + int(np.sqrt(n))
    jj = np.arange(1, m + 1)
    xstar = x[int(n / 4 + 0.5) - 1]        # first quartile
    theta = 1 / x[-1] + (1 - np.sqrt(m / (jj - 0.5))) / prior / xstar
    a = -theta
    kk = np.array([np.mean(np.log1p(ai * x)) for ai in a])
    l_theta = n * (np.log(a / kk) - kk - 1)
    wt = np.exp(l_theta - logsumexp(l_theta))
    th = np.sum(theta * wt)
    k = np.mean(np.log1p(-th * x))
    sigma = -k / th
    k = (k * n + 0.5 * 10) / (n + 10)
    return k, sigma


def relative_eff(lr, chain_lengths):
    r = np.exp(lr - lr.max())
    S = len(r)
    if len(set(chain_lengths)) == 1:
        return _ess(_split(r.reshape(len(chain_lengths), -1))) / S
    ends = np.cumsum(chain_lengths)
    return sum(_ess(_split(r[e - n:e][None, :])) for n, e in zip(chain_lengths, ends)) / S


def psis(lr, r_eff):
    """Pareto-smoothed normalised weights and k-hat for log ratios lr."""
    S = len(lr)
    lr = lr - lr.max()
    M = int(np.ceil(min(0.2 * S, 3 * np.sqrt(S / r_eff))))
    order = np.argsort(lr, kind="stable")
    tail = order[-M:]
    cut = lr[order[-M - 1]]
    k, sigma = gpdfit(np.sort(np.exp(lr[tail]) - np.exp(cut)))
    lw = lr.copy()
    if np.isfinite(k):
        p = (np.arange(1, M + 1) - 0.5) / M
        q = (sigma * np.expm1(-k * np.log1p(-p)) / k) if abs(k) > 1e-12 else -sigma * np.log1p(-p)
        lw[tail] = np.minimum(np.log(q + np.exp(cut)), 0.0)    # tail is in ascending order
    return np.exp(lw - logsumexp(lw)), k, M


# ---- inputs

def chain_chunk(path):
    m = re.search(r"_(\d+)(?:\.(\d+))?\.csv$", os.path.basename(path))
    if not m:
        raise SystemExit(f"{path}: name must end _<chain>.<chunk>.csv or _<chain>.csv")
    return int(m.group(1)), int(m.group(2) or 1)


def read_input(path):
    with open(path) as f:
        r = csv.reader(l for l in f if not l.startswith("#"))
        header = next(r)
        rows = [v for v in r if v]
    # full-log-density.R records a failed evaluation as NA (written by fwrite as an empty field)
    d = np.array([[np.nan if x in ("", "NA") else float(x) for x in v] for v in rows])
    for k in ("lp__", "lp_full"):
        if k not in header:
            raise SystemExit(f"{path}: no {k} column")
    return header, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="full-log-density.R outputs, _<chain>.<chunk>.csv")
    ap.add_argument("--fit", nargs="+", help="the fit's chain CSVs, in chain order, to check every draw "
                    "(give it after the inputs)")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out_lines = []

    def say(s=""):
        print(s)
        out_lines.append(s)

    # Read and order the chunks
    parts = {}
    for p in args.inputs:
        c, k = chain_chunk(p)
        if (c, k) in parts:
            raise SystemExit(f"{p}: chain {c} chunk {k} given twice")
        parts[(c, k)] = p
    chains = sorted({c for c, _ in parts})
    header, by_chain = None, {}
    for c in chains:
        ks = sorted(k for cc, k in parts if cc == c)
        if ks != list(range(1, len(ks) + 1)):
            say(f"warning: chain {c} has chunks {ks}, not 1..{len(ks)} (a chunk missing?)")
        ds = []
        for k in ks:
            h, d = read_input(parts[(c, k)])
            if header is None:
                header = h
            elif h != header:
                raise SystemExit(f"{parts[(c, k)]}: columns differ from the other inputs")
            ds.append(d)
        by_chain[c] = np.vstack(ds)
    col_i = {k: i for i, k in enumerate(header)}
    lengths = [len(by_chain[c]) for c in chains]
    say(f"{len(args.inputs)} input files; chains {chains}; draws per chain {lengths}")

    for c in chains:
        bad = ~np.isfinite(by_chain[c][:, col_i["lp_full"]])
        if bad.any():
            raise SystemExit(f"chain {c}: lp_full is missing or not finite at draws "
                             f"{(np.nonzero(bad)[0] + 1)[:10].tolist()} (rerun those chunks)")

    if args.fit:
        if len(args.fit) != len(chains):
            raise SystemExit(f"--fit has {len(args.fit)} CSVs but the inputs have {len(chains)} chains")
        for c, f in zip(chains, args.fit):
            lp = pw.read_lp(f)
            mine = by_chain[c][:, col_i["lp__"]]
            if len(lp) != len(mine) or not np.allclose(lp, mine, rtol=pw.LP_RTOL, atol=pw.LP_ATOL):
                raise SystemExit(f"chain {c}: its {len(mine)} draws do not match the {len(lp)} "
                                 f"post-warmup draws of {f} by lp__")
        say(f"every draw matches --fit by lp__ ({', '.join(args.fit)})")
    else:
        say("no --fit given: chunk order and completeness not checked against the fit")

    d = np.vstack([by_chain[c] for c in chains])
    chain_id = np.concatenate([[c] * n for c, n in zip(chains, lengths)])
    draw_id = np.concatenate([np.arange(1, n + 1) for n in lengths])
    lp, lr = d[:, col_i["lp__"]], d[:, col_i["lp_full"]] - d[:, col_i["lp__"]]
    S = len(lr)

    # PSIS
    r_eff = relative_eff(lr, lengths)
    w, k, M = psis(lr, r_eff)
    thr = min(1 - 1 / np.log10(S), 0.7)
    ess = r_eff / np.sum(w ** 2)
    say(f"\nlog ratio: mean {lr.mean():.3f}, sd {lr.std(ddof=1):.3f}, range {lr.min():.2f} to "
        f"{lr.max():.2f} (a constant offset is irrelevant: weights are normalised)")
    say(f"r_eff {r_eff:.3f}; tail size M = {M}; k-hat {k:.3f} (reliable below {thr:.2f}"
        f"{'' if k < thr else '; NOT RELIABLE'})")
    say(f"PSIS ESS {ess:.0f} of {S} draws; Kish ESS {1 / np.sum(w ** 2):.0f}; "
        f"largest weight {w.max():.5f} (equal weights: {1 / S:.5f})")
    for c in chains:
        s = chain_id == c
        r_c = relative_eff(lr[s], [s.sum()])
        _, k_c, _ = psis(lr[s], r_c)
        say(f"  chain {c} alone: k-hat {k_c:.3f}")

    path = os.path.join(args.outdir, "psis-weights.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["chain", "draw", "lp__", "log_ratio", "weight"])
        for row in zip(chain_id, draw_id, lp, lr, w):
            wr.writerow([int(row[0]), int(row[1]), repr(float(row[2])), repr(float(row[3])),
                         repr(float(row[4]))])
    say(f"\nwrote {path}")

    # Parameter shifts
    col = {k: d[:, i] for k, i in col_i.items()}
    if "m_NSCLC" in col:
        import analyse_nsclc_fit as A
        names = A.GLOBALS + A.SENS_FINAL + A.DERIVED
        model = "NSCLC"
    elif "m_SCLC" in col:
        import analyse_sclc_fit as A
        names = A.PARAMS + A.DERIVED
        model = "SCLC"
    else:
        raise SystemExit("inputs are neither NSCLC (m_NSCLC) nor SCLC (m_SCLC) fits")
    A.add_derived(col)
    qs = [0.05, 0.5, 0.95]
    rows = []
    for n in names:
        x = col[n]
        m0, s0 = x.mean(), x.std(ddof=1)
        q0 = np.quantile(x, qs, method="inverted_cdf")
        m1, s1 = pw.wmean(x, w), pw.wsd(x, w)
        q1 = pw.wquantile(x, qs, w)
        mcse = np.sqrt(np.sum(w ** 2 * (x - m1) ** 2) / r_eff)
        rows.append({"parameter": n, "thinned_mean": m0, "thinned_sd": s0, "thinned_q05": q0[0],
                     "thinned_q50": q0[1], "thinned_q95": q0[2], "full_mean": m1, "full_sd": s1,
                     "full_q05": q1[0], "full_q50": q1[1], "full_q95": q1[2],
                     "shift_sd": (m1 - m0) / s0, "mcse_shift_sd": mcse / s0})
    path = os.path.join(args.outdir, "psis-summary.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    say(f"\n{model}: thinned-data posterior vs reweighted to the full data "
        "(shift of the mean in thinned-posterior SDs, with its MCSE)")
    say(f"{'parameter':22s} {'thinned mean':>12s} {'sd':>9s} {'full mean':>12s} {'sd':>9s} "
        f"{'full 5%':>10s} {'full 95%':>10s} {'shift':>7s} {'MCSE':>6s}")
    for r in rows:
        say(f"{r['parameter']:22s} {r['thinned_mean']:12.5g} {r['thinned_sd']:9.3g} {r['full_mean']:12.5g} "
            f"{r['full_sd']:9.3g} {r['full_q05']:10.4g} {r['full_q95']:10.4g} {r['shift_sd']:+7.2f} "
            f"{r['mcse_shift_sd']:6.3f}")
    say(f"\nwrote {path}")
    path = os.path.join(args.outdir, "psis-diagnostics.txt")
    with open(path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Equal-weight resample of a fit's draws using its PSIS weights.

For users who want a set of draws from the (approximate) full-data posterior
without having to apply weights. The main analyses use the weights directly
(--psis-weights), which loses less information; this is an additional output.

  python psis_resample.py output/hpc/empirical-nsclc.thinned_[1-4].csv \\
      --psis-weights psis-nsclc/psis-weights.csv -o nsclc-psis-resample/draws.csv \\
      [--gq nsclc-real-fit-corrected/ppc-gq_[1-4].csv --gq-output nsclc-psis-resample/ppc-gq.csv] \\
      [--method systematic-shuffle|systematic|multinomial] [-n 8000] [--seed 1]

Inputs: the fit's CmdStan chain CSVs (post-warmup draws are used; saved warmup
is dropped) and its psis-weights.csv. Chains are matched to the weights file
by lp__ (psis_weights.match_chains), so the weights can't be applied to the
wrong draws; weights are renormalised over the chains given.

Methods (n draws, indices into the input draws):
  systematic-shuffle  systematic resampling (one uniform), then the n draws are
                      put in random order (default)
  systematic          systematic resampling, draws left in input order (chain
                      by chain), so neighbouring rows are correlated
  multinomial         independent sampling with replacement, probability = weight
Systematic resampling gives each draw floor(n w) or ceil(n w) copies, so it adds
less Monte Carlo noise than multinomial sampling.

Outputs:
  -o FILE             CmdStan-format CSV with every column of the fit, the
                      header of the first chain (num_samples set to n,
                      save_warmup to false) plus comment lines describing the
                      resample. The adaptation block is chain 1's; sampler
                      columns (accept_stat__ etc.) are those of the source draws.
  FILE stem + .index.csv
                      one row per output draw: row, chain and draw of the
                      source (as in the weights file), weight (normalised over
                      the inputs).
  --gq-output FILE    (with --gq) the same draws, in the same order, from the
                      standalone generated-quantities CSVs of this fit (matched
                      through the fitted_params named in each GQ header). Its
                      fitted_params is set to the -o file, so nsclc_ppc.py /
                      sclc_ppc.py can be run on it unweighted.

Caveats (also written into the output header): the resample carries no more
information than the weighted draws. It contains repeated draws: its
effective sample size is about the PSIS ESS of the weights (e.g. about 2,000
for NSCLC), not n. It is one set of draws, not Markov chains, so R-hat and
ESS must not be computed from it, and it should not be passed back through
--psis-weights.
"""

import argparse
import datetime
import os
import re
import sys

import numpy as np

import psis_weights as pw

METHODS = ("systematic-shuffle", "systematic", "multinomial")


def read_cmdstan(path):
    """Raw lines of a CmdStan CSV: (comments before the column header, column
    header, comments between header and draws, post-warmup draw lines,
    num_warmup, save_warmup). Partial (truncated) rows are dropped."""
    pre, mid, rows = [], [], []
    header = None
    num_warmup, save_warmup = 1000, False
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if header is None:
                    pre.append(line)
                    m = re.match(r"#\s*num_warmup = (\d+)", line)
                    if m:
                        num_warmup = int(m.group(1))
                    m = re.match(r"#\s*save_warmup = (\S+)", line)
                    if m:
                        save_warmup = m.group(1) in ("1", "true")
                elif not rows or (save_warmup and len(rows) <= num_warmup):
                    mid.append(line)      # adaptation block (after any saved warmup)
                continue
            if header is None:
                header = line
                ncol = len(header.split(","))
                continue
            if len(line.split(",")) == ncol:
                rows.append(line)
    if header is None:
        raise SystemExit(f"{path}: no column header")
    if save_warmup:
        rows = rows[num_warmup:]
    return pre, header, mid, rows, num_warmup, save_warmup


def lp_of(header, rows, path):
    cols = header.split(",")
    if "lp__" not in cols:
        raise SystemExit(f"{path}: no lp__ column (give the fit's chain CSVs, not GQ output)")
    i = cols.index("lp__")
    return np.array([float(r.split(",")[i]) for r in rows])


def set_header_value(lines, key, value):
    """Set `# ... key = value` in CmdStan header comment lines (dropping '(Default)')."""
    out, done = [], False
    for line in lines:
        m = re.match(rf"(#\s*{key} = ).*$", line)
        if m and not done:
            line, done = m.group(1) + str(value), True
        out.append(line)
    return out


def pick(w, n, method, rng):
    if method == "multinomial":
        return rng.choice(len(w), size=n, replace=True, p=w)
    idx = pw.systematic(w, n, rng)
    if method == "systematic-shuffle":
        idx = idx[rng.permutation(n)]
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fits", nargs="+", help="the fit's CmdStan chain CSVs")
    ap.add_argument("--psis-weights", required=True, help="psis-weights.csv from psis_reweight.py")
    ap.add_argument("-o", "--output", required=True, help="output CSV of resampled draws")
    ap.add_argument("--method", choices=METHODS, default="systematic-shuffle")
    ap.add_argument("-n", type=int, help="number of draws (default: number of input draws)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gq", nargs="+", help="standalone generated-quantities CSVs of this fit")
    ap.add_argument("--gq-output", help="output CSV of the resampled generated quantities")
    a = ap.parse_args()
    if bool(a.gq) != bool(a.gq_output):
        ap.error("--gq and --gq-output go together")
    pw.check_numpy()

    # fit draws and their weights
    fits = [read_cmdstan(p) for p in a.fits]
    headers = {f[1] for f in fits}
    if len(headers) != 1:
        raise SystemExit("the fit CSVs have different columns")
    ids, wts = pw.match_chains(a.psis_weights, [lp_of(f[1], f[3], p) for f, p in zip(fits, a.fits)], a.fits)
    rows = [r for f in fits for r in f[3]]
    src = [(c, d + 1) for c, f in zip(ids, fits) for d in range(len(f[3]))]
    w = pw.normalise(np.concatenate(wts))
    S = len(w)
    n = a.n or S
    if n < 1:
        raise SystemExit("-n must be positive")

    # generated quantities, lined up with the fit draws by (chain, draw)
    gq_rows = None
    if a.gq:
        gq = [read_cmdstan(p) for p in a.gq]
        if len({g[1] for g in gq}) != 1:
            raise SystemExit("the GQ CSVs have different columns")
        fitted = [pw.fitted_params_path(p) for p in a.gq]
        gids, _ = pw.match_chains(a.psis_weights, [pw.read_lp(f) for f in fitted],
                                  [f"{g} (fit {f})" for g, f in zip(a.gq, fitted)])
        if sorted(gids) != sorted(ids):
            raise SystemExit(f"the GQ files cover chains {sorted(gids)} of the weights file "
                             f"but the fits cover {sorted(ids)}")
        by_chain = {}
        for c, g, f, p in zip(gids, gq, fitted, a.gq):
            r = g[3]
            nfit = len(pw.read_lp(f))
            if len(r) != nfit:
                # GQ run on a fit with saved warmup writes the warmup draws too
                _, _, _, _, nw, sw = read_cmdstan(f)
                if sw and len(r) == nw + nfit:
                    r = r[nw:]
                else:
                    raise SystemExit(f"{p}: {len(r)} draws but its fit {f} has {nfit} post-warmup draws")
            by_chain[c] = r
        gq_rows = [by_chain[c][d - 1] for c, d in src]

    rng = np.random.default_rng(a.seed)
    idx = pick(w, n, a.method, rng)

    counts = np.bincount(idx, minlength=S)
    distinct, most = int((counts > 0).sum()), int(counts.max())
    kish = pw.kish_ess(w)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    notes = [
        f"# PSIS resample (psis_resample.py, {stamp}): {n} draws from {S} input draws "
        f"(chains {', '.join(map(str, ids))} of the weights file)",
        f"#   inputs: {' '.join(a.fits)}",
        f"#   weights: {a.psis_weights}; method: {a.method}; seed: {a.seed}",
        f"#   distinct input draws: {distinct}; most copies of one draw: {most}; "
        f"Kish ESS of the weights: {kish:.0f}",
        "#   Equal-weight draws for convenience. Not Markov chains: do not compute R-hat or ESS "
        "from them, and do not reweight them again. Sampler columns are those of the source "
        "draws; the adaptation block below is the first input chain's.",
    ]

    out_dir = os.path.dirname(a.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pre, header, mid = fits[0][0], fits[0][1], fits[0][2]
    pre = set_header_value(set_header_value(pre, "num_samples", n), "save_warmup", "false")
    with open(a.output, "w") as f:
        f.write("\n".join(pre + notes + [header] + mid + [rows[i] for i in idx]) + "\n")
    print(f"wrote {a.output}")

    index_path = re.sub(r"\.csv$", "", a.output) + ".index.csv"
    with open(index_path, "w") as f:
        f.write("row,chain,draw,weight\n")
        for k, i in enumerate(idx, 1):
            f.write(f"{k},{src[i][0]},{src[i][1]},{float(w[i])!r}\n")
    print(f"wrote {index_path}")

    if gq_rows is not None:
        gpre, gheader = gq[0][0], gq[0][1]
        gq_dir = os.path.dirname(a.gq_output)
        if gq_dir:
            os.makedirs(gq_dir, exist_ok=True)
        rel = os.path.relpath(a.output, gq_dir or ".")
        gpre = set_header_value(gpre, "fitted_params", rel)
        gnotes = notes[:1] + [f"#   generated quantities from: {' '.join(a.gq)}; "
                              f"rows match {a.output} (fitted_params) row by row"] + notes[2:]
        with open(a.gq_output, "w") as f:
            f.write("\n".join(gpre + gnotes + [gheader] + [gq_rows[i] for i in idx]) + "\n")
        print(f"wrote {a.gq_output}")

    print(f"{a.method}: {n} draws from {S}; {distinct} distinct draws, at most {most} copies of one; "
          f"Kish ESS of the weights {kish:.0f}")


if __name__ == "__main__":
    sys.exit(main())

"""Sensitivity of one screen for prevalent NSCLC, PSIS-weighted.

For each posterior draw, the sensitivity for prevalent cancers is the
stage-specific sensitivity averaged over the stage mix of prevalent
(undiagnosed at baseline) NSCLC:

    sens_prev = sum_k p_k * sens[arm, k],   p = softmax(b_NSCLC_cons)

Within the NSCLC sub-model the stage mix does not depend on PLCOm2012,
because all seven NSCLC stages share one PLCOm2012 slope (it cancels when
normalising over stages). This holds only for NSCLC alone: for NSCLC and
SCLC together, the NSCLC/SCLC share would vary with PLCOm2012 (the two
sub-models have different slopes), and so would the combined sensitivity.
With the final-screen effect, sens[arm, k] is replaced by
inv_logit(logit_sens[arm, k] + log_or_sens_final[arm]).

Draws are weighted by their PSIS weights (matched by chain, draw and lp__).
Quantiles use the weighted inverted-CDF definition of psis_weights.py.

  python prevalent_sensitivity.py
  python prevalent_sensitivity.py --unweighted
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psis_weights as pw

STAGES = 7
ARMS = {1: "LDCT", 2: "CXR"}


def load(fits, weights_path):
    cols = lambda n: n.startswith(("b_NSCLC_cons", "logit_sens", "log_or_sens_final", "lp__"))
    parts = []
    for c, path in enumerate(fits, 1):
        f = pd.read_csv(path, comment="#", usecols=cols)
        f["chain"] = c
        f["draw"] = np.arange(1, len(f) + 1)
        parts.append(f)
    d = pd.concat(parts, ignore_index=True)
    if weights_path is None:
        return d, np.full(len(d), 1 / len(d))
    w = pd.read_csv(weights_path)
    d = d.merge(w, on=["chain", "draw"], suffixes=("", "_w"), validate="one_to_one")
    if len(d) != len(w) or not np.allclose(d.lp__, d.lp___w, rtol=1e-10, atol=1e-6):
        sys.exit("PSIS weights do not match the fit draws (chain/draw/lp__)")
    return d, d.weight.to_numpy() / d.weight.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fits", nargs=4, metavar="CSV",
                    default=[f"output/hpc/empirical-nsclc.thinned_{c}.sampling.csv" for c in range(1, 5)],
                    help="the 4 NSCLC chain CSVs without warmup, in chain order")
    ap.add_argument("--psis-weights", default="psis-nsclc/psis-weights.csv")
    ap.add_argument("--unweighted", action="store_true", help="ignore the PSIS weights")
    args = ap.parse_args()

    d, w = load(args.fits, None if args.unweighted else args.psis_weights)
    b = d[[f"b_NSCLC_cons.{k}" for k in range(1, STAGES + 1)]].to_numpy()
    p = np.exp(b - b.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    inv_logit = lambda x: 1 / (1 + np.exp(-x))

    print(f"{len(d)} draws, {'unweighted' if args.unweighted else 'PSIS-weighted'}")
    print(f"prevalent stage mix (mean): " + ", ".join(f"{x:.3f}" for x in (w[:, None] * p).sum(0)))
    for g, arm in ARMS.items():
        ls = d[[f"logit_sens.{g}.{k}" for k in range(1, STAGES + 1)]].to_numpy()
        lor = d[f"log_or_sens_final.{g}"].to_numpy()[:, None]
        for label, sens in (("without final-screen effect", inv_logit(ls)),
                            ("with final-screen effect", inv_logit(ls + lor))):
            x = (p * sens).sum(axis=1)
            m = np.sum(w * x)
            sd = np.sqrt(np.sum(w * (x - m) ** 2))
            q10, q50, q90 = pw.wquantile(x, [0.1, 0.5, 0.9], w)
            print(f"{arm}, {label}: mean {m:.3f} (SD {sd:.3f}), median {q50:.3f}, "
                  f"80% CrI {q10:.3f}-{q90:.3f}")


if __name__ == "__main__":
    main()

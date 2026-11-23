"""Scatter of the full-data log density (lp_full) against the thinned-data log
density (lp__) at each draw, coloured by chain (the PSIS weights are not shown).

Reads a psis_reweight.py weights file (chain, draw, lp__, log_ratio, weight),
where lp_full = lp__ + log_ratio. The dashed line has slope 1 through the mean
log ratio; the scatter about it is what determines the weights.

  python psis_lp_scatter.py                      # psis-nsclc/ and psis-sclc/
  python psis_lp_scatter.py psis-nsclc/psis-weights.csv -o out.png --label NSCLC
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(weights, out, label):
    d = pd.read_csv(weights)
    d["lp_full"] = d.lp__ + d.log_ratio
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sh = d.sample(frac=1, random_state=1)          # random plotting order so no chain hides another
    ax.scatter(sh.lp__, sh.lp_full, s=3, alpha=0.4, c=[f"C{c - 1}" for c in sh.chain], rasterized=True)
    for c in sorted(d.chain.unique()):
        ax.scatter([], [], s=12, color=f"C{c - 1}", label=f"chain {c}")
    x = np.array([d.lp__.min(), d.lp__.max()])
    off = d.log_ratio.mean()
    ax.plot(x, x + off, "k--", lw=0.8, label=f"slope 1, offset {off:.2f}")
    r = np.corrcoef(d.lp__, d.lp_full)[0, 1]
    ax.set_xlabel("thinned-data log density (lp__)")
    ax.set_ylabel("full-data log density (lp_full)")
    ax.set_title(f"{label}: full vs thinned log density (r = {r:.3f}; SD of difference {d.log_ratio.std():.2f})",
                 fontsize=10)
    ax.legend(markerscale=1.5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{label}: r {r:.4f}, SD lp__ {d.lp__.std():.2f}, SD lp_full {d.lp_full.std():.2f}, "
          f"SD log ratio {d.log_ratio.std():.3f}; wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="?", help="psis-weights.csv (default: both psis-nsclc/ and psis-sclc/)")
    ap.add_argument("-o", "--output", help="output PNG (default: psis-lp-scatter.png next to the weights file)")
    ap.add_argument("--label", help="title label (default: from the directory name)")
    a = ap.parse_args()
    if a.weights is None:
        if a.output or a.label:
            ap.error("-o/--label need a weights file")
        jobs = [("psis-nsclc/psis-weights.csv", None, "NSCLC"), ("psis-sclc/psis-weights.csv", None, "SCLC")]
    else:
        jobs = [(a.weights, a.output, a.label)]
    for weights, out, label in jobs:
        out = out or os.path.join(os.path.dirname(weights), "psis-lp-scatter.png")
        label = label or os.path.basename(os.path.dirname(os.path.abspath(weights))).replace("psis-", "").upper()
        plot(weights, out, label)


if __name__ == "__main__":
    main()

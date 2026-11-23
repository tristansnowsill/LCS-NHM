"""Screening sensitivity (non-final screen) of LDCT and CXR by stage, NSCLC and
SCLC, from the PSIS-reweighted posteriors.

For each stage, LDCT and CXR are side by side: dark band = central 50% of the
posterior, light band = central 90%, thin white line = posterior median
(the same layout as the posterior predictive figures). NSCLC stages IA1-IV
come first, then, after a gap, SCLC limited and extensive.

The quantities are sens[arm, stage] (NSCLC) and sens_SCLC[arm, stage] (SCLC),
arm 1 = LDCT and arm 2 = CXR: sensitivity at non-final screens, not
sens_final. Quantiles are PSIS-weighted (psis_weights.wquantile, i.e.
np.quantile(..., method="inverted_cdf", weights=w)); each fit's chains are
matched to its weights by lp__. With --unweighted the plain np.quantile of
the thinned-data draws is used instead.

  python plot_psis_sensitivity.py                       # default inputs below
  python plot_psis_sensitivity.py -o figures/psis-sensitivity.pdf
  python plot_psis_sensitivity.py --unweighted -o figures/sensitivity-unweighted.png

Writes the figure and a CSV of the plotted quantiles (same stem, .csv).
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import plot_recovery_sclc as prs      # palette and rcParams of the other figures
import psis_weights as pw

# LDCT and CXR: series slots 1 and 2 of the validated palette (blue/orange,
# CVD-safe pair), each with a light tint for the 90% band
COLOURS = {"LDCT": (prs.POST, prs.POST_LIGHT), "CXR": (prs.TRUTH, "#f7c3aa")}
GREY, GREY_LIGHT = "#6f6e69", "#cfcdc8"

NSCLC_STAGES = ["IA1", "IA2", "IA3", "IB", "II", "III", "IV"]
SCLC_STAGES = ["Limited", "Extensive"]
Q = [0.05, 0.25, 0.5, 0.75, 0.95]


def load(fits, weights, columns):
    """Draws of `columns` from the chain CSVs, and normalised PSIS weights (or None)."""
    parts = [pd.read_csv(p, comment="#", usecols=["lp__"] + columns) for p in fits]
    d = pd.concat(parts, ignore_index=True)
    if weights is None:
        return d, None
    w = pw.align(weights, [p.lp__.values for p in parts], fits)
    return d, pw.normalise(np.concatenate(w))


def quantiles(x, w):
    if w is None:
        return np.quantile(x, Q)
    return pw.wquantile(x, Q, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsclc-fit", nargs="+", default=sorted(glob.glob("output/hpc/empirical-nsclc.thinned_[1-4].sampling.csv")))
    ap.add_argument("--sclc-fit", nargs="+", default=sorted(glob.glob("output/hpc/empirical-sclc.thinned_[1-4].sampling.csv")))
    ap.add_argument("--nsclc-psis-weights", default="psis-nsclc/psis-weights.csv")
    ap.add_argument("--sclc-psis-weights", default="psis-sclc/psis-weights.csv")
    ap.add_argument("--unweighted", action="store_true", help="plain quantiles of the thinned-data draws")
    ap.add_argument("-o", "--output", default="figures/psis-sensitivity.png")
    a = ap.parse_args()
    if not a.nsclc_fit or not a.sclc_fit:
        ap.error("fit CSVs not found; give --nsclc-fit / --sclc-fit")
    if not a.unweighted:
        pw.check_numpy()

    arms = [("LDCT", 1), ("CXR", 2)]
    groups = [("NSCLC", a.nsclc_fit, a.nsclc_psis_weights, "sens", NSCLC_STAGES),
              ("SCLC", a.sclc_fit, a.sclc_psis_weights, "sens_SCLC", SCLC_STAGES)]
    rows = []
    for model, fits, weights, var, stages in groups:
        cols = [f"{var}.{g}.{k}" for _, g in arms for k in range(1, len(stages) + 1)]
        d, w = load(fits, None if a.unweighted else weights, cols)
        for k, st in enumerate(stages, 1):
            for arm, g in arms:
                q = quantiles(d[f"{var}.{g}.{k}"].values, w)
                rows.append(dict(model=model, stage=st, test=arm, parameter=f"{var}[{g},{k}]",
                                 q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4]))
    t = pd.DataFrame(rows)

    # x positions: NSCLC stages 0..6, a gap, then SCLC
    gap = 0.7
    centre = {}
    for i, st in enumerate(NSCLC_STAGES):
        centre[("NSCLC", st)] = i
    for i, st in enumerate(SCLC_STAGES):
        centre[("SCLC", st)] = len(NSCLC_STAGES) + gap + i
    off, width = {"LDCT": -0.19, "CXR": 0.19}, 0.32

    fig, ax = plt.subplots(figsize=(8, 4))
    for r in t.itertuples():
        x = centre[(r.model, r.stage)] + off[r.test]
        dark, light = COLOURS[r.test]
        ax.bar(x, r.q95 - r.q05, bottom=r.q05, width=width, color=light, lw=0)
        ax.bar(x, r.q75 - r.q25, bottom=r.q25, width=width, color=dark, lw=0)
        ax.hlines(r.q50, x - width / 2, x + width / 2, color=prs.SURFACE, lw=1.5)

    ticks = [centre[k] for k in centre]
    ax.set_xticks(ticks)
    ax.set_xticklabels([k[1] for k in centre])
    # model labels under the stage labels
    for model, stages in (("NSCLC", NSCLC_STAGES), ("SCLC", SCLC_STAGES)):
        mid = np.mean([centre[(model, s)] for s in stages])
        ax.annotate(model, xy=(mid, 0), xycoords=("data", "axes fraction"), xytext=(0, -26),
                    textcoords="offset points", ha="center", va="top", fontsize=9, color=prs.INK_2)
    sep = len(NSCLC_STAGES) - 1 + (1 + gap) / 2
    ax.axvline(sep, color=prs.GRID, lw=0.8)
    ax.set_xlim(-0.6, centre[("SCLC", SCLC_STAGES[-1])] + 0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color=prs.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Stage of lung cancer", labelpad=20)
    ax.set_ylabel("Sensitivity (non-final screen)")

    handles = [Patch(color=COLOURS["LDCT"][0]), Patch(color=COLOURS["CXR"][0]),
               Patch(color=GREY_LIGHT), Patch(color=GREY),
               (Patch(color=GREY), Line2D([0.15, 0.85], [0.5, 0.5], color=prs.SURFACE, lw=1.5))]
    labels = ["LDCT", "CXR", "90% interval", "50% interval", "median"]
    ax.legend(handles, labels, loc="upper left", ncol=3, frameon=True, facecolor=prs.SURFACE,
              edgecolor=prs.GRID, handlelength=1.6)
    fig.tight_layout()

    out_dir = os.path.dirname(a.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(a.output, dpi=200)
    csv_path = re.sub(r"\.[^.]+$", "", a.output) + ".csv"
    t.to_csv(csv_path, index=False, float_format="%.5g")
    print(f"wrote {a.output} and {csv_path}")


if __name__ == "__main__":
    main()

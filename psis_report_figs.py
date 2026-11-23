"""Figures 1-3 and extra numbers (weight concentration, log-ratio correlations, per-chain log-ratio mean/SD) for docs/psis-report.md.

  python psis_report_figs.py   (from /workspace; writes psis-nsclc/psis-{log-ratios,weights,shifts}.png)
  python psis_report_figs.py --tag -2026-11   (psis-{nsclc,sclc}-2026-11/ and the
      output/hpc/empirical-{nsclc,sclc}-2026-11.thinned_* fits; figures in psis-nsclc-2026-11/)
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psis_weights as pw

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--tag', default='', help='suffix of the fit and PSIS directory names, e.g. -2026-11')
TAG = ap.parse_args().tag

def load(model):
    w = pd.read_csv(f"psis-{model}{TAG}/psis-weights.csv")
    fits = [pd.read_csv(f"output/hpc/empirical-{model}{TAG}.thinned_{c}.sampling.csv", comment="#") for c in (1, 2, 3, 4)]
    for c, f in enumerate(fits, 1):
        f["chain"] = c; f["draw"] = np.arange(1, len(f) + 1)
    d = pd.concat(fits).merge(w, on=["chain", "draw"], suffixes=("", "_w"))
    assert np.allclose(d.lp__, d.lp___w, rtol=1e-10, atol=1e-6)
    return d, pd.read_csv(f"psis-{model}{TAG}/psis-summary.csv")

out = {}
for model in ("nsclc", "sclc"):
    d, s = load(model)
    out[model] = (d, s)
    wt = d.weight.values
    srt = np.sort(wt)[::-1]
    print(f"== {model}: S={len(d)}, weight share of top 1% draws {srt[:len(d)//100].sum():.3f}, top 10% {srt[:len(d)//10].sum():.3f}")
    print(f"   max/min weight ratio {wt.max()/wt.min():.1f}; draws with weight > 5/S: {(wt > 5/len(d)).sum()}")
    par = [p for p in s.parameter if p.replace("[",".") in d.columns or p in d.columns]
    cors = {p: np.corrcoef(d[p], d.log_ratio)[0, 1] for p in s.parameter if p in d.columns}
    top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("   corr(log ratio, parameter):", ", ".join(f"{p} {r:+.2f}" for p, r in top))
    # per-chain log-ratio sd and mean
    print("   per chain log ratio mean/sd:", d.groupby("chain").log_ratio.agg(["mean", "std"]).round(3).values.tolist())

# Figure 1: log-ratio distributions and NSCLC log ratio vs m
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, model, label in ((ax[0], "nsclc", "NSCLC"), (ax[1], "sclc", "SCLC")):
    d, _ = out[model]
    lr = d.log_ratio - d.log_ratio.mean()
    a.hist(lr, bins=60, color="0.6")
    a.set_title(f"{label}: log ratio (centred), SD {d.log_ratio.std():.2f}")
    a.set_xlabel("log p_full − log p_thinned − mean")
d, _ = out["nsclc"]
ax[2].scatter(d.m_NSCLC, d.log_ratio, s=2, alpha=0.3, color="k")
ax[2].set_xlabel("m_NSCLC"); ax[2].set_ylabel("log ratio")
ax[2].set_title(f"NSCLC: log ratio vs m (r = {np.corrcoef(d.m_NSCLC, d.log_ratio)[0,1]:+.2f})")
fig.tight_layout(); fig.savefig(f"psis-nsclc{TAG}/psis-log-ratios.png", dpi=130)

# Figure 2: weights
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for model, label, col in (("nsclc", "NSCLC", "C0"), ("sclc", "SCLC", "C1")):
    d, _ = out[model]
    S = len(d)
    ax[0].plot(np.arange(1, S + 1) / S, np.cumsum(np.sort(d.weight.values)[::-1]), color=col, label=label)
    ax[1].plot(np.arange(S), d.weight.values * S, ".", ms=1.5, color=col, alpha=0.5, label=label)
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8, label="equal weights")
ax[0].set_xlabel("fraction of draws (largest weights first)"); ax[0].set_ylabel("share of total weight")
ax[0].legend(); ax[0].set_title("Concentration of the PSIS weights")
ax[1].set_xlabel("draw (chains 1–4 in order)"); ax[1].set_ylabel("weight × S"); ax[1].set_yscale("log")
ax[1].set_title("Normalised weights by draw"); ax[1].legend(markerscale=6)
fig.tight_layout(); fig.savefig(f"psis-nsclc{TAG}/psis-weights.png", dpi=130)

# Figure 3: shifts
fig, ax = plt.subplots(1, 2, figsize=(11, 9), gridspec_kw={"width_ratios": [3, 1.3]})
for a, model, label in ((ax[0], "nsclc", "NSCLC"), (ax[1], "sclc", "SCLC")):
    _, s = out[model]
    s = s.iloc[::-1]
    y = np.arange(len(s))
    mc = s.mcse_shift_sd if "mcse_shift_sd" in s else 0
    a.errorbar(s.shift_sd, y, xerr=2 * mc, fmt="o", ms=3, color="k", ecolor="0.5", lw=1)
    a.axvline(0, color="0.3", lw=0.8)
    a.set_yticks(y); a.set_yticklabels(s.parameter, fontsize=7)
    if model == "sclc":  # just wide enough for the largest |shift| + 2 MCSE (c_SCLC)
        lim = 1.05 * (s.shift_sd.abs() + 2 * mc).max()
    else:
        lim = 0.9
    a.set_xlim(-lim, lim); a.set_ylim(-1, len(s))
    a.set_title(label)
fig.supxlabel("Change in posterior mean after PSIS reweighting, in thinned-fit posterior SDs (bars: ±2 MCSE)", fontsize=10)
fig.tight_layout(); fig.savefig(f"psis-nsclc{TAG}/psis-shifts.png", dpi=130)
print("figures written")

#!/usr/bin/env python3
"""Posterior predictive check of NSCLC diagnosis counts, with a bootstrap
comparator (the NSCLC counterpart of sclc_ppc.py).

Inputs:
  generated-quantities CSVs from nsclc-model-marginal.stan run on the fitted
  draws with the cohort from nsclc_ppc_data.py (one per chain; saved warmup
  draws are dropped using the fitted_params CSV named in each file's
  header), and the observed-diagnoses CSV from nsclc_ppc_data.py --observed.

For each arm (LDCT, CXR), stage (IA1, IA2, IA3, IB, II, III, IV) and
category (pre-screening, T0 screen, T0-T1 interval, T1 screen, T1-T2
interval, T2 screen, post-screening) it compares:
  - the posterior predictive distribution of the count (pp_count),
  - a non-parametric bootstrap of the observed count (patients resampled
    with replacement from the whole cohort; exact via a multinomial draw),
  - the observed count,
and the same for totals over stages, categories and arms. Writes to --outdir:
  ppc-summary.csv  every cell and total
  ppc-counts.png   arm x stage panels, by category
  ppc-totals.png   totals by category, by arm and stage, by arm and category
  ppc-modes.png    stage distribution by mode of detection (screen-detected,
                   interval, outside screening) for each arm

Posterior predictive p-values: P(sim >= obs) and P(sim <= obs); values
near 0 in either indicate misfit in that direction. With 98 cells and two
tails, up to about 10 cells are expected below 0.05 by chance (fewer for
small counts, whose p-values are conservative).

With --psis-weights (the psis-weights.csv of psis_reweight.py for the fit),
the posterior predictive summaries (mean, quantiles, both p-values as
weighted proportions) and interval bars are PSIS-weighted; the bootstrap is
not (it resamples the observed data). Each GQ CSV is matched to the weights
through the fit CSV named as fitted_params in its header, by lp__. See
psis_weights.py for the definitions (quantiles: np.quantile with
method="inverted_cdf" and weights).

Usage:
  python nsclc_ppc.py --observed nsclc-real-fit/ppc-observed.csv -o nsclc-real-fit \\
      nsclc-real-fit/ppc-gq_[1-4].csv
"""

import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_recovery_sclc as prs
import psis_weights as pw
from sclc_ppc import warmup_rows

CATS = ['pre-screening', 'T0 screen', 'T0–T1 interval', 'T1 screen',
        'T1–T2 interval', 'T2 screen', 'post-screening']
SHORT = ['pre', 'T0', 'T0–T1', 'T1', 'T1–T2', 'T2', 'post']
STAGES = ['IA1', 'IA2', 'IA3', 'IB', 'II', 'III', 'IV']
ARMS = ['LDCT', 'CXR']
MODES = [('screen-detected (T0, T1, T2)', [1, 3, 5]),
         ('interval (T0–T1, T1–T2)', [2, 4]),
         ('outside screening (pre, post)', [0, 6])]
PP = prs.POST            # posterior predictive (series slot 1)
PP_LIGHT = prs.POST_LIGHT
BOOT = prs.TRUTH         # bootstrap (series slot 2)
OBS = prs.INK


def read_gq(path):
    """pp_count draws (draws x arm x stage x category), warmup dropped."""
    with open(path) as f:
        lines = (l for l in f if not l.startswith('#') and l.strip())
        h = next(lines).strip().split(',')
        idx = [h.index(f'pp_count.{a + 1}.{s + 1}.{c + 1}')
               for a in range(2) for s in range(7) for c in range(7)]
        rows = []
        for l in lines:
            v = l.rstrip('\n').split(',')
            if len(v) == len(h):
                rows.append([int(float(v[i])) for i in idx])
    d = np.array(rows, int)[warmup_rows(path):]
    return d.reshape(len(d), 2, 7, 7)


def read_observed(path):
    with open(path) as f:
        n_pt = int(re.search(r'N_pt = (\d+)', f.readline()).group(1))
        rows = list(csv.DictReader(f))
    cell = np.array([(int(r['arm']) - 1, int(r['stage']) - 1, int(r['category']) - 1) for r in rows])
    return n_pt, cell


def interval_plot(ax, sims, boots, obs, labels, w=None):
    """Posterior predictive 50%/90% bars with median, bootstrap 90%, observed.
    w: normalised PSIS weights of the predictive draws, or None (unweighted)."""
    x = np.arange(len(sims))
    if w is None:
        q = np.array([np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]) for v in sims]).T
    else:
        q = np.array([pw.wquantile(v, [0.05, 0.25, 0.5, 0.75, 0.95], w) for v in sims]).T
    bq = np.array([np.quantile(v, [0.05, 0.95]) for v in boots]).T
    ax.vlines(x - 0.12, q[0], q[4], color=PP_LIGHT, lw=6, label='posterior predictive 90%')
    ax.vlines(x - 0.12, q[1], q[3], color=PP, lw=6, label='posterior predictive 50%')
    ax.hlines(q[2], x - 0.24, x, color=prs.SURFACE, lw=1.5)
    ax.vlines(x + 0.14, bq[0], bq[1], color=BOOT, lw=2, label='bootstrap 90%')
    ax.scatter(x + 0.14, obs, s=22, color=OBS, zorder=3, label='observed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(axis='y', color=prs.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gq', nargs='+', help='generated-quantities CSVs, one per chain')
    ap.add_argument('--observed', required=True)
    ap.add_argument('-o', '--outdir', default='nsclc-real-fit')
    ap.add_argument('-B', type=int, default=4000, help='bootstrap resamples')
    ap.add_argument('-s', '--seed', type=int, default=1)
    ap.add_argument('--psis-weights', help='psis-weights.csv from psis_reweight.py for the fit: '
                    'PSIS-weighted predictive summaries (see psis_weights.py)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    parts = [read_gq(p) for p in args.gq]
    sim = np.concatenate(parts)
    psis_w = None
    if args.psis_weights:
        chain_w = pw.gq_weights(args.psis_weights, args.gq)
        for p, d, cw in zip(args.gq, parts, chain_w):
            if len(d) != len(cw):
                raise SystemExit(f'{p}: {len(d)} draws but its fit has {len(cw)} weighted draws')
        psis_w = pw.normalise(np.concatenate(chain_w))
    n_pt, cell = read_observed(args.observed)
    obs = np.zeros((2, 7, 7), int)
    np.add.at(obs, tuple(cell.T), 1)

    # Bootstrap: resample all n_pt patients; only the NSCLC patients' counts matter
    rng = np.random.default_rng(args.seed)
    m = len(cell)
    times = rng.multinomial(n_pt, np.r_[np.full(m, 1 / n_pt), 1 - m / n_pt], size=args.B)[:, :m]
    boot = np.zeros((args.B, 2, 7, 7), int)
    for k, (a, s, c) in enumerate(cell):
        boot[:, a, s, c] += times[:, k]

    print(f'{len(sim)} posterior predictive draws; {args.B} bootstrap resamples; '
          f'{n_pt} patients, {m} observed NSCLC diagnoses')
    if psis_w is not None:
        print(f'PSIS-weighted predictive summaries ({args.psis_weights}; Kish ESS of the weights '
              f'{pw.kish_ess(psis_w):.0f} of {len(psis_w)}); the bootstrap is unweighted')

    rows = []
    def add_row(arm, stage, cat, s, b, o):
        if psis_w is None:
            rows.append(dict(arm=arm, stage=stage, category=cat, observed=int(o),
                             pp_mean=np.mean(s), pp_q05=np.quantile(s, 0.05), pp_q25=np.quantile(s, 0.25),
                             pp_q50=np.quantile(s, 0.5), pp_q75=np.quantile(s, 0.75),
                             pp_q95=np.quantile(s, 0.95),
                             p_sim_ge_obs=np.mean(s >= o), p_sim_le_obs=np.mean(s <= o),
                             boot_q05=np.quantile(b, 0.05), boot_q95=np.quantile(b, 0.95)))
            return
        q = pw.wquantile(s, [0.05, 0.25, 0.5, 0.75, 0.95], psis_w)
        rows.append(dict(arm=arm, stage=stage, category=cat, observed=int(o),
                         pp_mean=pw.wmean(s, psis_w), pp_q05=q[0], pp_q25=q[1], pp_q50=q[2], pp_q75=q[3],
                         pp_q95=q[4], p_sim_ge_obs=pw.wprop(s >= o, psis_w), p_sim_le_obs=pw.wprop(s <= o, psis_w),
                         boot_q05=np.quantile(b, 0.05), boot_q95=np.quantile(b, 0.95)))
    for a in range(2):
        for s in range(7):
            for c in range(7):
                add_row(ARMS[a], STAGES[s], CATS[c], sim[:, a, s, c], boot[:, a, s, c], obs[a, s, c])
            add_row(ARMS[a], STAGES[s], 'total', sim[:, a, s].sum(1), boot[:, a, s].sum(1), obs[a, s].sum())
        for c in range(7):
            add_row(ARMS[a], 'all', CATS[c], sim[:, a, :, c].sum(1), boot[:, a, :, c].sum(1),
                    obs[a, :, c].sum())
        add_row(ARMS[a], 'all', 'total', sim[:, a].sum((1, 2)), boot[:, a].sum((1, 2)), obs[a].sum())
    for s in range(7):
        add_row('both', STAGES[s], 'total', sim[:, :, s].sum((1, 2)), boot[:, :, s].sum((1, 2)),
                obs[:, s].sum())
    for c in range(7):
        add_row('both', 'all', CATS[c], sim[:, :, :, c].sum((1, 2)), boot[:, :, :, c].sum((1, 2)),
                obs[:, :, c].sum())
    add_row('both', 'all', 'total', sim.sum((1, 2, 3)), boot.sum((1, 2, 3)), obs.sum())

    path = os.path.join(args.outdir, 'ppc-summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()})

    # Console: every total, plus the individual cells with either p-value below 0.05
    def line(r):
        flag = ' *' if min(r['p_sim_ge_obs'], r['p_sim_le_obs']) < 0.05 else ''
        return (f'{r["arm"]:>4} {r["stage"]:5} {r["category"]:15} {r["observed"]:4d} {r["pp_mean"]:8.1f} '
                f'{r["pp_q05"]:5.0f}–{r["pp_q95"]:<5.0f} {r["boot_q05"]:5.0f}–{r["boot_q95"]:<5.0f} '
                f'{r["p_sim_ge_obs"]:11.3f} {r["p_sim_le_obs"]:11.3f}{flag}')
    head = (f'{"arm":>4} {"stage":5} {"category":15} {"obs":>4} {"pp mean":>8} {"pp 90%":>11} '
            f'{"boot 90%":>11} {"P(sim>=obs)":>11} {"P(sim<=obs)":>11}')
    cells = [r for r in rows if r['arm'] in ARMS and r['stage'] in STAGES and r['category'] != 'total']
    flagged = [r for r in cells if min(r['p_sim_ge_obs'], r['p_sim_le_obs']) < 0.05]
    print(f'\nTotals\n{head}')
    for r in rows:
        if r not in cells:
            print(line(r))
    print(f'\nCells (arm x stage x category) with either p-value below 0.05: {len(flagged)} of '
          f'{len(cells)} (up to about {0.1 * len(cells):.0f} expected by chance)\n{head}')
    for r in flagged:
        print(line(r))
    print('(* either p-value below 0.05)\nwrote', path)

    # Figure 1: rows = arm, columns = stage, x = category
    # y shared within each column (stage) so LDCT and CXR compare directly
    fig, axes = plt.subplots(2, 7, figsize=(20, 6.4), sharey='col')
    for a in range(2):
        for s in range(7):
            ax = axes[a, s]
            interval_plot(ax, [sim[:, a, s, c] for c in range(7)], [boot[:, a, s, c] for c in range(7)],
                          obs[a, s], SHORT, w=psis_w)
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_title(f'{ARMS[a]}, stage {STAGES[s]}', loc='left', fontsize=9)
            if s == 0:
                ax.set_ylabel('NSCLC diagnoses')
    # fit each shared column to both rows (set_ylim(bottom=0) stopped autoscaling after row 1)
    for col in axes.T:
        col[0].set_ylim(0, 1.05 * max(ax.dataLim.y1 for ax in col))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('Real NSCLC data: posterior predictive vs observed counts by screening round '
                 f'({len(sim)} draws{"" if psis_w is None else ", PSIS-weighted"}; bootstrap {args.B})',
                 x=0.01, ha='left', color=prs.INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    p = os.path.join(args.outdir, 'ppc-counts.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('wrote', p)

    # Figure 2: totals
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5))
    interval_plot(axes[0], [sim[:, :, :, c].sum((1, 2)) for c in range(7)],
                  [boot[:, :, :, c].sum((1, 2)) for c in range(7)], obs.sum((0, 1)), SHORT, w=psis_w)
    axes[0].set_title('both arms, all stages, by category', loc='left')
    labs = [f'{ARMS[a]}\n{STAGES[s]}' for a in range(2) for s in range(7)]
    interval_plot(axes[1], [sim[:, a, s].sum(1) for a in range(2) for s in range(7)],
                  [boot[:, a, s].sum(1) for a in range(2) for s in range(7)],
                  [obs[a, s].sum() for a in range(2) for s in range(7)], labs, w=psis_w)
    axes[1].set_title('all categories, by arm and stage', loc='left')
    labs = [f'{ARMS[a]}\n{SHORT[c]}' for a in range(2) for c in range(7)]
    interval_plot(axes[2], [sim[:, a, :, c].sum(1) for a in range(2) for c in range(7)],
                  [boot[:, a, :, c].sum(1) for a in range(2) for c in range(7)],
                  [obs[a, :, c].sum() for a in range(2) for c in range(7)], labs, w=psis_w)
    axes[2].set_title('all stages, by arm and category', loc='left')
    for ax in axes:
        ax.set_ylabel('NSCLC diagnoses')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    p = os.path.join(args.outdir, 'ppc-totals.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('wrote', p)

    # Figure 3: stage distribution by mode of detection, per arm
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
    for a in range(2):
        for k, (name, cs) in enumerate(MODES):
            ax = axes[a, k]
            interval_plot(ax, [sim[:, a, s][:, cs].sum(1) for s in range(7)],
                          [boot[:, a, s][:, cs].sum(1) for s in range(7)],
                          [obs[a, s, cs].sum() for s in range(7)], STAGES, w=psis_w)
            ax.set_title(f'{ARMS[a]}: {name}', loc='left', fontsize=9)
            if k == 0:
                ax.set_ylabel('NSCLC diagnoses')
    fig.suptitle('Stage at diagnosis by mode of detection' + ('' if psis_w is None else ' (PSIS-weighted)'),
                 x=0.01, ha='left', color=prs.INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    p = os.path.join(args.outdir, 'ppc-modes.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Posterior predictive check of SCLC diagnosis counts, with a bootstrap
comparator.

Inputs:
  generated-quantities CSVs from sclc-model.stan run on the fitted draws
  with the cohort from sclc_ppc_data.py (one per chain; saved warmup draws
  are dropped using the fitted_params CSV named in each file's header),
  and the observed-diagnoses CSV from sclc_ppc_data.py --observed.

For each arm (1, 2), stage (limited, extensive) and category (pre-screening,
T0 screen, T0-T1 interval, T1 screen, T1-T2 interval, T2 screen,
post-screening) it compares:
  - the posterior predictive distribution of the count (pp_count),
  - a non-parametric bootstrap of the observed count (patients resampled
    with replacement from the whole cohort; exact via a multinomial draw),
  - the observed count,
and writes ppc-summary.csv and ppc-counts.png (plus totals) to --outdir.

Posterior predictive p-values: P(sim >= obs) and P(sim <= obs); values
near 0 in either indicate misfit in that direction.

With --psis-weights (the psis-weights.csv of psis_reweight.py for the fit),
the posterior predictive summaries (mean, quantiles, both p-values as
weighted proportions) and interval bars are PSIS-weighted; the bootstrap is
not (it resamples the observed data). Each GQ CSV is matched to the weights
through the fit CSV named as fitted_params in its header, by lp__. See
psis_weights.py for the definitions (quantiles: np.quantile with
method="inverted_cdf" and weights).

Usage:
  python sclc_ppc.py --observed sclc-ppc-observed.csv -o sclc-real-fit gq_[1-4].csv
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

CATS = ['pre-screening', 'T0 screen', 'T0–T1 interval', 'T1 screen',
        'T1–T2 interval', 'T2 screen', 'post-screening']
SHORT = ['pre', 'T0', 'T0–T1', 'T1', 'T1–T2', 'T2', 'post']
STAGES = ['limited', 'extensive']
ARM_NAMES = ['LDCT', 'CXR']   # arm 1, arm 2 (rndgroup); figures only, the CSV keeps 1/2
PP = prs.POST            # posterior predictive (series slot 1)
PP_LIGHT = prs.POST_LIGHT
BOOT = prs.TRUTH         # bootstrap (series slot 2)
OBS = prs.INK


def warmup_rows(gq_path):
    """Number of leading warmup rows in a GQ CSV, from its fitted_params CSV."""
    fitted = None
    for line in open(gq_path):
        if not line.startswith('#'):
            break
        m = re.match(r'#\s*fitted_params = (\S+)', line)
        if m:
            fitted = m.group(1)
    if fitted is None:
        return 0
    if not os.path.isabs(fitted) and not os.path.exists(fitted):
        fitted = os.path.join(os.path.dirname(gq_path), fitted)
    num, save = 1000, False
    for line in open(fitted):
        if not line.startswith('#'):
            break
        m = re.match(r'#\s*num_warmup = (\d+)', line)
        if m:
            num = int(m.group(1))
        m = re.match(r'#\s*save_warmup = (\S+)', line)
        if m:
            save = m.group(1) in ('1', 'true')
    return num if save else 0


def read_gq(path):
    rows = [l for l in open(path) if not l.startswith('#') and l.strip()]
    h = rows[0].strip().split(',')
    d = np.array([[float(x) for x in r.strip().split(',')] for r in rows[1:]
                  if len(r.split(',')) == len(h)])
    d = d[warmup_rows(path):]
    out = np.zeros((len(d), 2, 2, 7), int)
    for a in range(2):
        for s in range(2):
            for c in range(7):
                out[:, a, s, c] = d[:, h.index(f'pp_count.{a + 1}.{s + 1}.{c + 1}')]
    return out


def read_observed(path):
    n_pt = None
    with open(path) as f:
        first = f.readline()
        n_pt = int(re.search(r'N_pt = (\d+)', first).group(1))
        rows = list(csv.DictReader(f))
    cell = np.array([(int(r['arm']) - 1, int(r['stage']) - 1, int(r['category']) - 1) for r in rows])
    return n_pt, cell


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gq', nargs='+', help='generated-quantities CSVs, one per chain')
    ap.add_argument('--observed', required=True)
    ap.add_argument('-o', '--outdir', default='sclc-real-fit')
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
    obs = np.zeros((2, 2, 7), int)
    np.add.at(obs, tuple(cell.T), 1)

    # Bootstrap: resample all n_pt patients; only the SCLC patients' counts matter
    rng = np.random.default_rng(args.seed)
    m = len(cell)
    times = rng.multinomial(n_pt, np.r_[np.full(m, 1 / n_pt), 1 - m / n_pt], size=args.B)[:, :m]
    boot = np.zeros((args.B, 2, 2, 7), int)
    for k, (a, s, c) in enumerate(cell):
        boot[:, a, s, c] += times[:, k]

    print(f'{len(sim)} posterior predictive draws; {args.B} bootstrap resamples; '
          f'{n_pt} patients, {m} observed SCLC diagnoses')
    if psis_w is not None:
        print(f'PSIS-weighted predictive summaries ({args.psis_weights}; Kish ESS of the weights '
              f'{pw.kish_ess(psis_w):.0f} of {len(psis_w)}); the bootstrap is unweighted')

    def summ(x):
        if psis_w is None:
            return np.mean(x), *np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        return pw.wmean(x, psis_w), *pw.wquantile(x, [0.05, 0.25, 0.5, 0.75, 0.95], psis_w)

    def pq(x, qs):
        """Predictive quantiles over draws (axis 0) for the figures."""
        return np.quantile(x, qs, axis=0) if psis_w is None else pw.wquantile(x, qs, psis_w, axis=0)

    rows = []
    def add_row(arm, stage, cat, s, b, o):
        mean, q05, q25, q50, q75, q95 = summ(s)
        if psis_w is None:
            p_ge, p_le = np.mean(s >= o), np.mean(s <= o)
        else:
            p_ge, p_le = pw.wprop(s >= o, psis_w), pw.wprop(s <= o, psis_w)
        rows.append(dict(arm=arm, stage=stage, category=cat, observed=int(o),
                         pp_mean=mean, pp_q05=q05, pp_q25=q25, pp_q50=q50, pp_q75=q75, pp_q95=q95,
                         p_sim_ge_obs=p_ge, p_sim_le_obs=p_le,
                         boot_q05=np.quantile(b, 0.05), boot_q95=np.quantile(b, 0.95)))
    for a in range(2):
        for s in range(2):
            for c in range(7):
                add_row(a + 1, STAGES[s], CATS[c], sim[:, a, s, c], boot[:, a, s, c], obs[a, s, c])
            add_row(a + 1, STAGES[s], 'total', sim[:, a, s].sum(1), boot[:, a, s].sum(1), obs[a, s].sum())
    for c in range(7):
        add_row('both', 'both', CATS[c], sim[:, :, :, c].sum((1, 2)), boot[:, :, :, c].sum((1, 2)),
                obs[:, :, c].sum())
    add_row('both', 'both', 'total', sim.sum((1, 2, 3)), boot.sum((1, 2, 3)), obs.sum())

    path = os.path.join(args.outdir, 'ppc-summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()})
    print(f'\n{"arm":>4} {"stage":9} {"category":15} {"obs":>4} {"pp mean":>8} {"pp 90%":>11} '
          f'{"boot 90%":>11} {"P(sim>=obs)":>11} {"P(sim<=obs)":>11}')
    for r in rows:
        flag = ' *' if min(r['p_sim_ge_obs'], r['p_sim_le_obs']) < 0.05 else ''
        print(f'{r["arm"]:>4} {r["stage"]:9} {r["category"]:15} {r["observed"]:4d} {r["pp_mean"]:8.1f} '
              f'{r["pp_q05"]:5.0f}–{r["pp_q95"]:<5.0f} {r["boot_q05"]:5.0f}–{r["boot_q95"]:<5.0f} '
              f'{r["p_sim_ge_obs"]:11.3f} {r["p_sim_le_obs"]:11.3f}{flag}')
    print('(* either p-value below 0.05)\nwrote', path)

    # Figure: rows = arm, columns = stage
    # y shared within each column (stage) so LDCT and CXR compare directly
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey='col')
    x = np.arange(7)
    for a in range(2):
        for s in range(2):
            ax = axes[a, s]
            q = pq(sim[:, a, s], [0.05, 0.25, 0.5, 0.75, 0.95])
            bq = np.quantile(boot[:, a, s], [0.05, 0.95], axis=0)
            ax.vlines(x - 0.12, q[0], q[4], color=PP_LIGHT, lw=7, label='posterior predictive 90%')
            ax.vlines(x - 0.12, q[1], q[3], color=PP, lw=7, label='posterior predictive 50%')
            ax.hlines(q[2], x - 0.25, x + 0.01, color=prs.SURFACE, lw=1.5)
            ax.vlines(x + 0.14, bq[0], bq[1], color=BOOT, lw=2, label='bootstrap 90%')
            ax.scatter(x + 0.14, obs[a, s], s=30, color=OBS, zorder=3, label='observed')
            ax.set_title(f'{ARM_NAMES[a]}, {STAGES[s]}', loc='left')
            ax.set_xticks(x)
            ax.set_xticklabels(SHORT, fontsize=8)
            ax.grid(axis='y', color=prs.GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            if s == 0:
                ax.set_ylabel('SCLC diagnoses')
    # fit each shared column to both rows (set_ylim(bottom=0) stopped autoscaling after row 1)
    for col in axes.T:
        col[0].set_ylim(0, 1.05 * max(ax.dataLim.y1 for ax in col))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('Real SCLC data: posterior predictive vs observed counts by screening round '
                 f'({len(sim)} draws{"" if psis_w is None else ", PSIS-weighted"}; bootstrap {args.B})',
                 x=0.01, ha='left', color=prs.INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    p = os.path.join(args.outdir, 'ppc-counts.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)

    # Totals: by category (both arms and stages) and by arm x stage
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [7, 4]})
    groups = [([sim[:, :, :, c].sum((1, 2)) for c in range(7)],
               [boot[:, :, :, c].sum((1, 2)) for c in range(7)],
               [obs[:, :, c].sum() for c in range(7)], SHORT, 'all arms and stages, by category'),
              ([sim[:, a, s].sum(1) for a in range(2) for s in range(2)],
               [boot[:, a, s].sum(1) for a in range(2) for s in range(2)],
               [obs[a, s].sum() for a in range(2) for s in range(2)],
               [f'{ARM_NAMES[a]}\n{STAGES[s]}' for a in range(2) for s in range(2)],
               'totals by arm and stage')]
    for ax, (ss, bb, oo, labs, title) in zip(axes, groups):
        x = np.arange(len(ss))
        q = np.array([pq(v, [0.05, 0.25, 0.5, 0.75, 0.95]) for v in ss]).T
        bq = np.array([np.quantile(v, [0.05, 0.95]) for v in bb]).T
        ax.vlines(x - 0.12, q[0], q[4], color=PP_LIGHT, lw=7)
        ax.vlines(x - 0.12, q[1], q[3], color=PP, lw=7)
        ax.hlines(q[2], x - 0.25, x + 0.01, color=prs.SURFACE, lw=1.5)
        ax.vlines(x + 0.14, bq[0], bq[1], color=BOOT, lw=2)
        ax.scatter(x + 0.14, oo, s=30, color=OBS, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(title, loc='left')
        ax.grid(axis='y', color=prs.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('SCLC diagnoses')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    p = os.path.join(args.outdir, 'ppc-totals.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()

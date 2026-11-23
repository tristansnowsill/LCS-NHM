#!/usr/bin/env python3
"""Draw parameter sets from the priors of nsclc-model-marginal.stan or
sclc-model.stan and write them as CmdStan sampling CSVs, so that the model's
standalone generated quantities (CmdStan generate_quantities with
fitted_params=<these files>) gives the prior predictive distribution of the
diagnosis counts (pp_count), exactly as for the posterior predictive check.

The column layout (and the header comments) are copied from a CSV of a fit
of the same model to the same data (e.g. a .sampling.csv from
strip_warmup.py), so every column the model expects is present. Parameters
are drawn from their priors; transformed parameters are filled in; sampler
diagnostics and generated quantities are written as 0 (generate_quantities
recomputes them). The NSCLC random effects eta are drawn from N(0, 1) (the
simulation draws its own random effects, so they are not used).

Priors, including the constraints:
  NSCLC  as in analyse_nsclc_fit.prior_draws: sens ordered within each arm
         and xi increasing (by rejection from the independent priors),
         (m, c) bivariate normal with m > 0
  SCLC   logit sens: limited N(0, sqrt 2), extensive N(5.220108, 2.286569),
         limited < extensive within each arm (by rejection);
         log_or_sens_final ~ gamma(4, 4 / exp(-1.5));
         (m, c) ~ MVN([2, -10.5], [[1, -0.8], [-0.8, 25/16]]), m > 0;
         b_plco ~ gamma(10, 10); b_cons ~ N(-7, 2);
         mu, phi ~ gamma(4) with means 0.262766, 0.141599, 1.960784; phi[1] < phi[2]

Usage:
  python prior_predictive_draws.py nsclc output/hpc/empirical-nsclc.thinned_1.sampling.csv \\
      -n 1000 --chains 4 -o output/prior/empirical-nsclc.prior
  -> output/prior/empirical-nsclc.prior_{1..4}.csv
"""

import argparse
import os

import numpy as np

from analyse_nsclc_fit import prior_draws as nsclc_prior_draws


def logit(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p) - np.log1p(-p)


def nsclc_draws(n, rng_seed):
    d = nsclc_prior_draws(n=n, seed=rng_seed)
    out = {k: d[k] for k in d if not k.startswith(('p_clin', 'sojourn'))}
    for g in (1, 2):
        for k in range(1, 8):
            out[f'logit_sens.{g}.{k}'] = logit(d[f'sens.{g}.{k}'])
    return out


def sclc_draws(n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    mc = rng.multivariate_normal([2.0, -10.5], [[1.0, -0.8], [-0.8, 25.0 / 16.0]], size=2 * n)
    mc = mc[mc[:, 0] > 0][:n]
    out['m_SCLC'], out['c_SCLC'] = mc[:, 0], mc[:, 1]
    out['b_SCLC_plco'] = rng.gamma(10.0, 1 / 10.0, n)
    for k in (1, 2):
        out[f'b_SCLC_cons.{k}'] = rng.normal(-7.0, 2.0, n)
    out['mu'] = rng.gamma(4.0, 0.262766 / 4.0, n)
    phi = np.empty((0, 2))                                   # phi[1] < phi[2]
    while len(phi) < n:
        x = np.column_stack([rng.gamma(4.0, 0.141599 / 4.0, n), rng.gamma(4.0, 1.960784 / 4.0, n)])
        phi = np.concatenate([phi, x[x[:, 0] < x[:, 1]]])
    out['phi.1'], out['phi.2'] = phi[:n].T
    for g in (1, 2):
        out[f'log_or_sens_final.{g}'] = rng.gamma(4.0, np.exp(-1.5) / 4.0, n)
        parts = []
        while sum(len(p) for p in parts) < n:
            x = np.column_stack([rng.normal(0.0, np.sqrt(2.0), n), rng.normal(5.220108, 2.286569, n)])
            parts.append(x[x[:, 0] < x[:, 1]])
        x = np.concatenate(parts)[:n]
        for k in (1, 2):
            out[f'logit_sens_SCLC.{g}.{k}'] = x[:, k - 1]
            out[f'sens_SCLC.{g}.{k}'] = 1 / (1 + np.exp(-x[:, k - 1]))
            out[f'sens_SCLC_final.{g}.{k}'] = 1 / (1 + np.exp(-(x[:, k - 1] + out[f'log_or_sens_final.{g}'])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', choices=['nsclc', 'sclc'])
    ap.add_argument('template', help='CmdStan CSV of a fit of the same model to the same data')
    ap.add_argument('-n', type=int, default=1000, help='draws per chain file')
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('-s', '--seed', type=int, default=20261014)
    ap.add_argument('-o', '--prefix', required=True, help='output files are <prefix>_<chain>.csv')
    args = ap.parse_args()

    lines = open(args.template).read().split('\n')
    i_head = next(i for i, l in enumerate(lines) if l and not l.startswith('#'))
    comments = [l.replace('save_warmup = true', 'save_warmup = false').replace('save_warmup = 1', 'save_warmup = 0')
                for l in lines[:i_head]]
    header = lines[i_head].split(',')

    total = args.n * args.chains
    draws = (nsclc_draws if args.model == 'nsclc' else sclc_draws)(total, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    n_eta = sum(1 for h in header if h.startswith('eta.'))
    if n_eta:
        eta = rng.standard_normal((total, n_eta))
        for j in range(n_eta):
            draws[f'eta.{j + 1}'] = eta[:, j]
    missing_params = [h for h in header if h not in draws and not h.endswith('__')
                      and not h.startswith(('pp_count', 'eig_gap'))]
    if missing_params:
        raise SystemExit(f'no prior draws for columns {missing_params[:10]}')
    # Check strict ordering (generate_quantities rejects ties in ordered parameters)
    if args.model == 'nsclc':
        for g in (1, 2):
            ls = np.column_stack([draws[f'logit_sens.{g}.{k}'] for k in range(1, 8)])
            assert (np.diff(ls, axis=1) > 1e-9).all()
        assert (np.diff(np.column_stack([draws[f'xi.{k}'] for k in range(1, 8)]), axis=1) > 0).all()

    cols = [draws[h] if h in draws else np.zeros(total) for h in header]
    data = np.column_stack(cols)
    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    for c in range(args.chains):
        path = f'{args.prefix}_{c + 1}.csv'
        with open(path, 'w') as f:
            f.write('\n'.join(comments) + '\n')
            f.write(','.join(header) + '\n')
            for row in data[c * args.n:(c + 1) * args.n]:
                f.write(','.join(f'{v:.12g}' for v in row) + '\n')
        print(f'wrote {path} ({args.n} prior draws)')


if __name__ == '__main__':
    main()

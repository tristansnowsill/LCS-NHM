#!/usr/bin/env python3
"""Tables for the NSCLC and SCLC natural history models fitted to the trial
data (the successors of the tables in tables-to-regenerate.docx).

  Table 1A   prevalence of lung cancer at randomisation, by PLCOm2012 centile
  Table 1B   probability of incident preclinical lung cancer within 5 years,
             by PLCOm2012 centile and smoking status, and of any lung cancer
             (prevalent or incident within 5 years)
  Table 2    stage distribution of NSCLC in the absence of screening, by
             centile of the random effect and averaged over it
  Supp. 1    posterior summary of the model parameters
  Supp. 2    prior and posterior predictive distributions of the diagnosis
             counts by screening round, arm, histotype and stage, against
             the trial data

Inputs (defaults in brackets):
  --nsclc-fit, --sclc-fit       CmdStan CSVs of the fits (saved warmup is
                                dropped) [output/hpc/empirical-{nsclc,sclc}.thinned_[1-4].csv]
  --nsclc-post, --sclc-post     posterior predictive generated quantities
                                [nsclc-real-fit-corrected/ppc-gq_*.csv,
                                 sclc-real-fit-corrected/ppc-gq_*.csv]
  --nsclc-prior, --sclc-prior   prior predictive generated quantities, from
                                prior_predictive_draws.py and generate_quantities
                                [prior-predictive/{nsclc,sclc}-prior-gq_*.csv]
  --nsclc-observed, --sclc-observed
                                observed diagnoses from {nsclc,sclc}_ppc_data.py
                                [data/empirical-{nsclc,sclc}.ppc-observed.csv]
  --data                        full trial data, for the PLCO centiles
                                [data/empirical.json]

  --nsclc-psis-weights, --sclc-psis-weights
                                optional psis-weights.csv from psis_reweight.py
                                for each fit: PSIS-weighted posterior summaries
                                (importance-weighted to the full data). Fits
                                and posterior predictive draws are matched to
                                the weights by lp__ (GQ files through their
                                fitted_params fits). Prior predictive columns
                                stay unweighted.

Outputs in --outdir [tables/]: tables.md, one CSV per table, and tables.docx
(if python-docx is installed).

PSIS weighting (see psis_weights.py): means, quantiles (np.quantile with
method="inverted_cdf" and weights) and SDs use the weights. A quantity of
one model uses that model's weights; quantities pairing the NSCLC and SCLC
draws by index ("any disease", "any lung cancer") use the product of the two
draws' weights, renormalised (valid because the two posteriors are
independent); if only one model is weighted, its weights alone.

Definitions:
  * PLCO centiles are of raw_plco over all trial participants; columns are
    labelled with the PLCOm2012 6-year risk inv_logit(raw_plco) and the
    centile. Covariates are centred with the data's mean_plco, as in the
    models.
  * Prevalence (Table 1A): the baseline state probabilities of each model,
    softmax(0, b_plco * plco + b_cons). The NSCLC and SCLC models are fitted
    separately; "any disease" is 1 - P(no NSCLC) P(no SCLC), pairing the two
    models' draws by index (their posteriors are independent).
  * Incidence (Table 1B): 1 - exp(-H(5)), with the Gompertz cumulative hazard
    H(t) = rate / shape * (exp(shape t) - 1), rate = exp(c + m plco),
    shape = m * beta(smoking), ignoring competing risks. "Any lung cancer" is
    1 - P(no prevalent NSCLC) P(no prevalent SCLC) exp(-H_NSCLC(5) - H_SCLC(5)).
  * Stage distribution (Table 2): a cancer starts in IA1 and presents at
    stage k with probability prod_{j<k} z lambda_j / (z lambda_j + xi_j) *
    xi_k / (z lambda_k + xi_k) (lambda_7 = 0); z = exp(sigma_re e -
    sigma_re^2 / 2) at the e = Phi^-1(centile) of the random effect, or
    averaged over e ~ N(0, 1) by 45-node Gauss-Hermite quadrature.
  * Priors (Supp. 1): mean and 90% interval of each parameter's marginal
    prior, as specified in nsclc-model-marginal.stan and sclc-model.stan.
    Exact for parameters with independent priors: gamma and normal priors
    directly; m > 0 is a truncation of the bivariate normal prior of (m, c),
    so m is a truncated normal and c's marginal is found by numerical
    integration. Monte Carlo for the sensitivities and the NSCLC and
    SCLC presentation rates, whose priors are constrained to increase with
    stage: independent draws from the stated priors, rejecting draws that
    are not increasing (identical components are sorted first, which
    samples the same constrained distribution with fewer rejections), with
    draws added until every Monte Carlo standard error is at most
    --prior-mcse (mean: sd / sqrt(N); quantiles: half the distance between
    the order statistics at Nq -/+ sqrt(Nq(1 - q))). LDCT and CXR have the
    same priors.
  * Predictive counts (Supp. 2): categories as in the posterior predictive
    checks (official screen timepoints; interval by the last screen
    attended; post-screening after T2 or more than 1.5 years after the last
    screen; "never screened" = no screen before the diagnosis).
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

import plot_recovery as pr
import plot_recovery_sclc as prs
import psis_weights as pw
from analyse_nsclc_fit import LAMBDA_MEAN, XI_MEAN, SENS_BETA
from analyse_sclc_fit import LOGIT_SENS_PRIOR
from nsclc_ppc import read_gq as read_gq_nsclc, read_observed as read_obs_nsclc
from sclc_ppc import read_gq as read_gq_sclc, read_observed as read_obs_sclc

STAGES = ['IA1', 'IA2', 'IA3', 'IB', 'II', 'III', 'IV']
SCLC_STAGES = ['Limited', 'Extensive']
CENTILES = [5, 25, 50, 75, 95]
ARMS = {1: 'LDCT', 2: 'CXR'}
# Supp. 2 row order: (category index in pp_count, label)
CATEGORIES = [(1, 'T0 screen'), (3, 'T1 screen'), (5, 'T2 screen'), (2, 'T0–T1 interval'),
              (4, 'T1–T2 interval'), (6, 'Post-screening'), (0, 'Never screened')]


# ---------------------------------------------------------------- formatting

def num(x):
    """At most three significant figures and at most one decimal place."""
    if not np.isfinite(x):
        return 'NA'
    if x == 0:
        return '0'
    if abs(x) >= 100:
        digits = int(math.floor(math.log10(abs(x)))) - 2
        s = f'{round(x, -digits):.0f}'
    else:
        s = f'{x:.1f}'
    return s.replace('-', '−')


def dec3(x):
    return f'{x:.3f}'.replace('-', '−')


def mean_ci(x, f=num, w=None):
    """w: normalised PSIS weights of the draws, or None (unweighted)."""
    if w is None:
        q05, q95 = np.quantile(x, [0.05, 0.95])
        return f'{f(np.mean(x))} ({f(q05)}, {f(q95)})'
    q05, q95 = pw.wquantile(x, [0.05, 0.95], w)
    return f'{f(pw.wmean(x, w))} ({f(q05)}, {f(q95)})'


def pred_summary(x, w=None):
    """mean (SD), median (10th–90th centile) of a count. w: normalised PSIS
    weights of the draws, or None (unweighted)."""
    if w is None:
        q10, q50, q90 = np.quantile(x, [0.1, 0.5, 0.9], method='inverted_cdf')
        return f'{num(np.mean(x))} ({num(np.std(x, ddof=1))}), {int(q50)} ({int(q10)}–{int(q90)})'
    q10, q50, q90 = pw.wquantile(x, [0.1, 0.5, 0.9], w)
    return f'{num(pw.wmean(x, w))} ({num(pw.wsd(x, w))}), {int(q50)} ({int(q10)}–{int(q90)})'


# ---------------------------------------------------------------- inputs

def load_fit(paths, reader):
    chains = [reader(p) for p in paths]
    h = chains[0][0]
    d = np.vstack([c[1] for c in chains])
    return {k: d[:, i] for i, k in enumerate(h)}


def softmax_states(b_plco, b_cons, x):
    """Baseline state probabilities: columns none, stage 1..K."""
    eta = np.column_stack([np.zeros_like(b_plco)] + [b_plco * x + b for b in b_cons])
    eta -= eta.max(1, keepdims=True)
    p = np.exp(eta)
    return p / p.sum(1, keepdims=True)


def stage_probs(lam, xi, z):
    """P(present at stage k), k = 1..7, for random effect z (arrays over draws)."""
    reach = np.ones_like(z)
    out = []
    for k in range(7):
        rate_out = xi[k] + (z * lam[k] if k < 6 else 0.0)
        out.append(reach * xi[k] / rate_out)
        if k < 6:
            reach = reach * z * lam[k] / rate_out
    return np.array(out)


def first_n(w, n):
    """PSIS weights of the first n draws, renormalised (None if unweighted)."""
    return None if w is None else pw.normalise(w[:n])


def paired(wN, wS, n):
    """Weights for the first n NSCLC and SCLC draws paired by index: the
    product of the two models' weights, renormalised (None if neither is
    weighted)."""
    if wN is None and wS is None:
        return None
    a = np.ones(n) if wN is None else wN[:n]
    b = np.ones(n) if wS is None else wS[:n]
    return pw.normalise(a * b)


# ---------------------------------------------------------------- priors (Supp. 1)
# Summaries are (mean, 5th centile, 95th centile).

NSCLC_MC = ([2.0, -7.5], [[1.0, -0.8], [-0.8, 25.0 / 16.0]])     # (m, c), m > 0
SCLC_MC = ([2.0, -10.5], [[1.0, -0.8], [-0.8, 25.0 / 16.0]])


def exact(dist):
    return (float(dist.mean()), *map(float, dist.ppf([0.05, 0.95])))


def gamma_prior(shape, mean):
    return stats.gamma(shape, scale=mean / shape)


def truncated_mc(mean, cov):
    """Exact summaries of m and c under a bivariate normal prior on (m, c)
    truncated to m > 0."""
    mu_m, mu_c = mean
    s_m = math.sqrt(cov[0][0])
    a = -mu_m / s_m
    p_keep = norm.sf(a)
    m = exact(stats.truncnorm(a, np.inf, loc=mu_m, scale=s_m))
    # c given m is normal; integrate over m > 0 for the CDF of c
    slope = cov[0][1] / cov[0][0]
    s_cm = math.sqrt(cov[1][1] - cov[0][1] ** 2 / cov[0][0])

    def cdf(x):
        f = lambda u: norm.pdf(u, mu_m, s_m) * norm.cdf((x - mu_c - slope * (u - mu_m)) / s_cm)
        return quad(f, 0.0, np.inf, epsabs=1e-13, epsrel=1e-12)[0] / p_keep

    s_c = math.sqrt(cov[1][1])
    q = [brentq(lambda x: cdf(x) - p, mu_c - 12 * s_c, mu_c + 12 * s_c, xtol=1e-10) for p in (0.05, 0.95)]
    c = (mu_c + cov[0][1] / s_m * norm.pdf(a) / p_keep, *q)
    return m, c


def hist_quantile(cum, lo, w, p):
    """Quantile p from cumulative histogram counts, interpolating linearly
    within the bin."""
    t = p * cum[-1]
    i = int(np.searchsorted(cum, t))
    prev = cum[i - 1] if i else 0
    return lo + (i + (t - prev) / max(cum[i] - prev, 1)) * w


def ordered_mc(sample, iid, transform, rng, target, n_start=1_000_000, chunk=4_000_000, bins=2 ** 21):
    """Monte Carlo summaries of a prior constrained to increasing values.
    sample(rng, n): n x K independent draws from the stated priors; iid:
    slices of identical components, sorted before the check; transform:
    applied to the kept draws before summarising. Draws are added until the
    largest Monte Carlo standard error is at most target.

    Draws are not stored: means come from running sums, and centiles from
    histograms with `bins` bins per parameter spanning three times the range
    of a pilot sample (draws outside go to the end bins, far beyond the 5th
    and 95th centiles), so memory does not grow with the number of draws.
    Returns ([(mean, q05, q95) per component], accepted draws, largest MCSE)."""
    def draw():
        x = sample(rng, chunk)
        for s in iid:
            x[:, s].sort(axis=1)
        return transform(x[(np.diff(x, axis=1) > 0).all(axis=1)])

    pilot = [draw()]
    while sum(len(y) for y in pilot) < n_start:
        pilot.append(draw())
    y = np.concatenate(pilot)
    K = y.shape[1]
    span = y.max(axis=0) - y.min(axis=0)
    lo = y.min(axis=0) - span
    w = 3 * span / bins
    centre = y.mean(axis=0)
    counts = np.zeros((K, bins), np.int64)
    s1, s2, n = np.zeros(K), np.zeros(K), 0
    need = 0
    while True:
        while n < need or pilot:
            y = pilot.pop() if pilot else draw()
            d = y - centre
            s1 += d.sum(axis=0)
            s2 += (d * d).sum(axis=0)
            n += len(y)
            idx = np.clip(((y - lo) / w).astype(np.int64), 0, bins - 1)
            for j in range(K):
                counts[j] += np.bincount(idx[:, j], minlength=bins)
        out, worst = [], 0.0
        for j in range(K):
            m1 = s1[j] / n
            worst = max(worst, math.sqrt(max(s2[j] / n - m1 * m1, 0.0) / n))
            cum = np.cumsum(counts[j])
            qs = []
            for q in (0.05, 0.95):
                qs.append(hist_quantile(cum, lo[j], w[j], q))
                dq = math.sqrt(q * (1 - q) / n)
                worst = max(worst, (hist_quantile(cum, lo[j], w[j], q + dq)
                                    - hist_quantile(cum, lo[j], w[j], q - dq)) / 2)
            out.append((float(centre[j] + m1), *map(float, qs)))
        if worst <= target:
            return out, n, worst
        need = int(n * (worst / target) ** 2 * 1.1) + 1


def logit(p):
    return np.log(p) - np.log1p(-p)


def prior_summaries(seed, target):
    """{'NSCLC': {...}, 'SCLC': {...}}: prior (mean, q05, q95) by parameter
    name, plus the Monte Carlo sample sizes and largest standard error."""
    rng = np.random.default_rng(seed)
    pN, pS, mc_info = {}, {}, []

    sens, n, w = ordered_mc(lambda r, n: np.column_stack([r.beta(a, b, n) for a, b in SENS_BETA]),
                            [slice(0, 4)], logit, rng, target)
    mc_info.append(('NSCLC sensitivity', n, w))
    for g in (1, 2):
        for k in range(7):
            pN[f'logit_sens.{g}.{k + 1}'] = sens[k]
    xi, n, w = ordered_mc(lambda r, n: r.gamma(4.0, XI_MEAN / 4.0, size=(n, 7)),
                          [slice(0, 3)], lambda x: x, rng, target)
    mc_info.append(('NSCLC presentation rates', n, w))
    for k in range(7):
        pN[f'xi.{k + 1}'] = xi[k]
    (mu1, sd1), (mu2, sd2) = LOGIT_SENS_PRIOR[1], LOGIT_SENS_PRIOR[2]
    sens, n, w = ordered_mc(lambda r, n: np.column_stack([r.normal(mu1, sd1, n), r.normal(mu2, sd2, n)]),
                            [], lambda x: x, rng, target)
    mc_info.append(('SCLC sensitivity', n, w))
    for g in (1, 2):
        for k in range(2):
            pS[f'logit_sens_SCLC.{g}.{k + 1}'] = sens[k]
    phi, n, w = ordered_mc(lambda r, n: np.column_stack([r.gamma(4.0, 0.141599 / 4.0, n),
                                                         r.gamma(4.0, 1.960784 / 4.0, n)]),
                           [], lambda x: x, rng, target)
    mc_info.append(('SCLC presentation rates', n, w))
    pS['phi.1'], pS['phi.2'] = phi

    for g in (1, 2):
        pN[f'log_or_sens_final.{g}'] = exact(gamma_prior(4.0, 0.082))
        pS[f'log_or_sens_final.{g}'] = exact(gamma_prior(4.0, math.exp(-1.5)))
    pN['b_NSCLC_plco'] = exact(gamma_prior(10.0, 1.0))
    pS['b_SCLC_plco'] = exact(gamma_prior(10.0, 1.0))
    for k in range(7):
        pN[f'b_NSCLC_cons.{k + 1}'] = exact(stats.norm(-8.0, 2.0))
    for k in range(2):
        pS[f'b_SCLC_cons.{k + 1}'] = exact(stats.norm(-7.0, 2.0))
    pN['m_NSCLC'], pN['c_NSCLC'] = truncated_mc(*NSCLC_MC)
    pS['m_SCLC'], pS['c_SCLC'] = truncated_mc(*SCLC_MC)
    for k in range(6):
        pN[f'lambda.{k + 1}'] = exact(gamma_prior(4.0, LAMBDA_MEAN[k]))
    pN['sigma_re'] = exact(gamma_prior(16.0, 0.607))
    pS['mu'] = exact(gamma_prior(4.0, 0.262766))
    return {'NSCLC': pN, 'SCLC': pS, 'mc': mc_info}


def prior_cell(s):
    return f'{dec3(s[0])} ({dec3(s[1])}, {dec3(s[2])})'


# ---------------------------------------------------------------- tables
# Each table is a dict: title, header (list of rows of cells; a header cell
# may be (text, span) to span several columns), rows (list of rows; a row of
# one cell is a section heading spanning the table), notes.

def table_1a(N, S, x_cent, col_labels, wN=None, wS=None):
    n = min(len(N['lp__']), len(S['lp__']))
    wn, ws, wp = first_n(wN, n), first_n(wS, n), paired(wN, wS, n)
    rows, data = [], {}
    pN = [softmax_states(N['b_NSCLC_plco'][:n], [N[f'b_NSCLC_cons.{k}'][:n] for k in range(1, 8)], x)
          for x in x_cent]
    pS = [softmax_states(S['b_SCLC_plco'][:n], [S[f'b_SCLC_cons.{k}'][:n] for k in (1, 2)], x)
          for x in x_cent]
    rows.append(['Probability of prevalent disease [per 100,000 (mean, 90% credible interval)]'])
    rows.append(['Any disease'] + [mean_ci(1e5 * (1 - a[:, 0] * b[:, 0]), w=wp) for a, b in zip(pN, pS)])
    rows.append(['NSCLC'])
    for k in range(7):
        rows.append([f'Stage {STAGES[k]}'] + [mean_ci(1e5 * a[:, k + 1], w=wn) for a in pN])
    rows.append(['SCLC'])
    for k in range(2):
        rows.append([SCLC_STAGES[k]] + [mean_ci(1e5 * b[:, k + 1], w=ws) for b in pS])
    return dict(title='Table 1A: Prevalence of lung cancer based on the natural history models',
                header=[['Outcome', ('PLCOm2012 [predicted 6-year risk, centile in the trial]', 5)],
                        [''] + col_labels],
                rows=rows,
                notes=['All numbers are given to a maximum of three significant figures and one decimal place.',
                       'NSCLC and SCLC are modelled separately; "any disease" combines them as independent.'])


def table_1b(N, S, x_cent, col_labels, beta_former, beta_current, years=5.0, wN=None, wS=None):
    n = min(len(N['lp__']), len(S['lp__']))
    wp = paired(wN, wS, n)
    smk = [(1, 'Current smoker'), (0, 'Ex-smoker')]

    def H(c, m, x, cig):
        shape = m * (beta_former + (beta_current - beta_former) * cig)
        return np.exp(c + m * x) / shape * np.expm1(shape * years)

    rows = [['Probability of incident preclinical disease within 5 years '
             '[per 100,000 (mean, 90% credible interval)]']]
    for name, P, c, m, wm in (('NSCLC', N, 'c_NSCLC', 'm_NSCLC', first_n(wN, n)),
                              ('SCLC', S, 'c_SCLC', 'm_SCLC', first_n(wS, n))):
        rows.append([name])
        for cig, lab in smk:
            rows.append([lab] + [mean_ci(1e5 * -np.expm1(-H(P[c][:n], P[m][:n], x, cig)), w=wm)
                                 for x in x_cent])
    rows.append(['Probability of any lung cancer (prevalent cancer or incident preclinical disease '
                 'within 5 years) [% (mean, 90% credible interval)]'])
    for cig, lab in smk:
        cells = []
        for x in x_cent:
            nonN = softmax_states(N['b_NSCLC_plco'][:n], [N[f'b_NSCLC_cons.{k}'][:n] for k in range(1, 8)], x)[:, 0]
            nonS = softmax_states(S['b_SCLC_plco'][:n], [S[f'b_SCLC_cons.{k}'][:n] for k in (1, 2)], x)[:, 0]
            h = H(N['c_NSCLC'][:n], N['m_NSCLC'][:n], x, cig) + H(S['c_SCLC'][:n], S['m_SCLC'][:n], x, cig)
            cells.append(mean_ci(100 * (1 - nonN * nonS * np.exp(-h)), w=wp))
        rows.append([lab] + cells)
    return dict(title='Table 1B: Incidence of lung cancer based on the natural history models',
                header=[['Outcome', ('PLCOm2012 [predicted 6-year risk, centile in the trial]', 5)],
                        [''] + col_labels],
                rows=rows,
                notes=['All numbers are given to a maximum of three significant figures and one decimal place; '
                       'probabilities of incident preclinical disease ignore competing risks.'])


def table_2(N, n_gh=45, wN=None):
    lam = np.array([N[f'lambda.{k}'] for k in range(1, 7)])
    xi = np.array([N[f'xi.{k}'] for k in range(1, 8)])
    sig = N['sigma_re']
    cols = []
    for q in CENTILES:
        z = np.exp(sig * norm.ppf(q / 100) - 0.5 * sig ** 2)
        cols.append(stage_probs(lam, xi, z))
    e, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    avg = sum(wj * stage_probs(lam, xi, np.exp(sig * ej - 0.5 * sig ** 2)) for ej, wj in zip(e, w))
    cols.append(avg)
    rows = [[STAGES[k]] + [mean_ci(100 * c[k], w=wN) for c in cols] for k in range(7)]
    return dict(title='Table 2: Stage distribution of NSCLC in the absence of screening',
                header=[['Stage', ('Centile of random-effects distribution for heterogeneity in progression rates', 5),
                         'Population average'],
                        ['[% (mean, 90% credible interval)]'] + [f'{q}th' for q in CENTILES] + ['']],
                rows=rows,
                notes=['Without competing risks. Population average by Gauss–Hermite quadrature '
                       f'({n_gh} nodes) over the random effect.'])


def supp_1(N, S, priors, wN=None, wS=None):
    r = []
    sec = lambda t: r.append([t])
    fits = {'NSCLC': (N, wN), 'SCLC': (S, wS)}

    def row(lab, model, name):
        P, w = fits[model]
        r.append([lab, prior_cell(priors[model][name]), mean_ci(P[name], dec3, w)])

    sec('Sensitivity of screening for NSCLC (log-odds scale)')
    for g in (1, 2):
        sec(ARMS[g])
        for k in range(1, 8):
            row(f'Stage {STAGES[k - 1]}', 'NSCLC', f'logit_sens.{g}.{k}')
    sec('Sensitivity of screening for SCLC (log-odds scale)')
    for g in (1, 2):
        sec(ARMS[g])
        for k in (1, 2):
            row(SCLC_STAGES[k - 1], 'SCLC', f'logit_sens_SCLC.{g}.{k}')
    sec('Final screen effect, NSCLC (log-odds scale)')
    for g in (1, 2):
        row(ARMS[g], 'NSCLC', f'log_or_sens_final.{g}')
    sec('Final screen effect, SCLC (log-odds scale)')
    for g in (1, 2):
        row(ARMS[g], 'SCLC', f'log_or_sens_final.{g}')
    sec('NSCLC prevalence')
    row('PLCO coefficient', 'NSCLC', 'b_NSCLC_plco')
    sec('Stage-specific intercept')
    for k in range(1, 8):
        row(f'Stage {STAGES[k - 1]}', 'NSCLC', f'b_NSCLC_cons.{k}')
    sec('SCLC prevalence')
    row('PLCO coefficient', 'SCLC', 'b_SCLC_plco')
    sec('Stage-specific intercept')
    for k in (1, 2):
        row(SCLC_STAGES[k - 1], 'SCLC', f'b_SCLC_cons.{k}')
    sec('NSCLC incidence')
    row('Parameter ‘c’', 'NSCLC', 'c_NSCLC')
    row('Parameter ‘m’', 'NSCLC', 'm_NSCLC')
    sec('SCLC incidence')
    row('Parameter ‘c’', 'SCLC', 'c_SCLC')
    row('Parameter ‘m’', 'SCLC', 'm_SCLC')
    sec('NSCLC progression rates (per year)')
    for k in range(1, 7):
        row(f'Stage {STAGES[k - 1]}', 'NSCLC', f'lambda.{k}')
    row('Random effects parameter σ', 'NSCLC', 'sigma_re')
    sec('NSCLC presentation rates (per year)')
    for k in range(1, 8):
        row(f'Stage {STAGES[k - 1]}', 'NSCLC', f'xi.{k}')
    sec('SCLC progression rate (per year)')
    row('Limited', 'SCLC', 'mu')
    sec('SCLC presentation rates (per year)')
    row('Limited', 'SCLC', 'phi.1')
    row('Extensive', 'SCLC', 'phi.2')
    n_mc = min(n for _, n, _ in priors['mc'])
    worst = max(w for _, _, w in priors['mc'])
    return dict(title='Supplementary Table 1: Prior and posterior distribution summary',
                header=[['Parameter', 'Prior mean (90% credible interval)',
                         'Posterior mean (90% credible interval)']], rows=r,
                notes=['NSCLC progression rates are from the stage shown to the next stage, for a random '
                       'effect of 1 (each patient\'s rates are multiplied by their random effect).',
                       'Prior summaries are exact where the prior has no ordering constraint (the constraint '
                       'm > 0 is a truncation of the bivariate normal prior of c and m, handled exactly). '
                       'Sensitivities and presentation rates (NSCLC and SCLC) are constrained to increase with stage; '
                       'their prior summaries are by Monte Carlo, rejecting draws that violate the ordering '
                       f'(at least {n_mc:,} accepted draws; Monte Carlo standard error at most {worst:.5f}). '
                       'LDCT and CXR have the same priors.'])


def supp_2(pp, obs, w_post=(None, None)):
    """pp: {'prior'/'post': (nsclc draws x 2 x 7 x 7, sclc draws x 2 x 2 x 7)}; obs: same without draws.
    w_post: normalised PSIS weights of the (NSCLC, SCLC) posterior predictive draws, or None."""
    rows = []
    for c, clab in CATEGORIES:
        first_c = True
        for g in (2, 1):
            first_g = True
            for hist, K, labs in (('NSCLC', 7, STAGES), ('SCLC', 2, ['Lim.', 'Ext.'])):
                h = 0 if hist == 'NSCLC' else 1
                for k in range(K):
                    rows.append([clab if first_c else '', ARMS[g] if first_g else '',
                                 hist if k == 0 else '', labs[k],
                                 pred_summary(pp['prior'][h][:, g - 1, k, c]),
                                 pred_summary(pp['post'][h][:, g - 1, k, c], w_post[h]),
                                 str(int(obs[h][g - 1, k, c]))])
                    first_c = first_g = False
    return dict(title='Supplementary Table 2: Comparison of prior and posterior predictive distributions '
                      'with the trial data',
                header=[['Timepoint', 'Imaging technology', 'Histotype', 'Stage', 'Prior predictive distribution',
                         'Posterior predictive distribution', 'Trial actual']],
                rows=rows,
                notes=['Predictive distributions are given as mean (standard deviation), median '
                       '(10th centile–90th centile).',
                       'Never screened: diagnosed with no screen before the diagnosis. Post-screening: diagnosed '
                       'after the T2 screen, or more than 1.5 years after the last screen. Intervals are assigned '
                       'by the last screen attended.',
                       'Key: CXR, chest X-ray; Ext., extensive; LDCT, low-dose computed tomography; Lim., limited; '
                       'NSCLC, non-small-cell lung cancer; SCLC, small-cell lung cancer; T0, baseline '
                       '(randomisation); T1, 12 months post-baseline; T2, 24 months post-baseline.'])


# ---------------------------------------------------------------- output

def expand(row):
    """Header row as plain cells: a (text, span) cell becomes text then blanks."""
    out = []
    for c in row:
        text, span = c if isinstance(c, tuple) else (c, 1)
        out += [text] + [''] * (span - 1)
    return out


def width(t):
    return max(len(r) for r in [expand(h) for h in t['header']] + [r for r in t['rows'] if len(r) > 1])


def to_markdown(t):
    w = width(t)
    out = [f'### {t["title"]}', '']
    head = [expand(h) for h in t['header']]
    first = head[0] + [''] * (w - len(head[0]))
    out.append('| ' + ' | '.join(first) + ' |')
    out.append('|' + '---|' * w)
    for h in head[1:]:
        out.append('| ' + ' | '.join(h + [''] * (w - len(h))) + ' |')
    for r in t['rows']:
        if len(r) == 1:
            out.append(f'| **{r[0]}** |' + ' |' * (w - 1))
        else:
            out.append('| ' + ' | '.join(r) + ' |')
    out.append('')
    out += [f'*{n}*' for n in t['notes']]
    return '\n'.join(out) + '\n'


def to_csv(t, path):
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        for h in t['header']:
            wr.writerow(expand(h))
        for r in t['rows']:
            wr.writerow(r)


def to_docx(tables, path):
    import docx
    from docx.shared import Pt
    from docx.enum.section import WD_ORIENT
    doc = docx.Document()
    style = doc.styles['Normal']
    style.font.size = Pt(9)
    for i, t in enumerate(tables):
        if i:
            doc.add_page_break()
        doc.add_paragraph().add_run(t['title']).bold = True
        w = width(t)
        tab = doc.add_table(rows=0, cols=w)
        tab.style = 'Table Grid'
        for h in t['header']:
            cells = tab.add_row().cells
            j = 0
            for c in h:
                text, span = c if isinstance(c, tuple) else (c, 1)
                cell = cells[j].merge(cells[j + span - 1]) if span > 1 else cells[j]
                cell.text = text
                j += span
            for c in cells:
                for p in c.paragraphs:
                    for run in p.runs:
                        run.bold = True
        for r in t['rows']:
            cells = tab.add_row().cells
            if len(r) == 1:
                m = cells[0].merge(cells[w - 1])
                m.text = r[0]
                m.paragraphs[0].runs[0].italic = True
            else:
                for j, v in enumerate(r):
                    cells[j].text = v
        for n in t['notes']:
            doc.add_paragraph(n).runs[0].font.size = Pt(8)
    doc.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nsclc-fit', nargs='+', default=sorted(glob.glob('output/hpc/empirical-nsclc.thinned_[1-4].csv')))
    ap.add_argument('--sclc-fit', nargs='+', default=sorted(glob.glob('output/hpc/empirical-sclc.thinned_[1-4].csv')))
    ap.add_argument('--nsclc-post', nargs='+', default=sorted(glob.glob('nsclc-real-fit-corrected/ppc-gq_*.csv')))
    ap.add_argument('--sclc-post', nargs='+', default=sorted(glob.glob('sclc-real-fit-corrected/ppc-gq_*.csv')))
    ap.add_argument('--nsclc-prior', nargs='+', default=sorted(glob.glob('prior-predictive/nsclc-prior-gq_*.csv')))
    ap.add_argument('--sclc-prior', nargs='+', default=sorted(glob.glob('prior-predictive/sclc-prior-gq_*.csv')))
    ap.add_argument('--nsclc-observed', default='data/empirical-nsclc.ppc-observed.csv')
    ap.add_argument('--sclc-observed', default='data/empirical-sclc.ppc-observed.csv')
    ap.add_argument('--data', default='data/empirical.json')
    ap.add_argument('--nsclc-psis-weights', help='psis-weights.csv from psis_reweight.py for the NSCLC fit')
    ap.add_argument('--sclc-psis-weights', help='psis-weights.csv from psis_reweight.py for the SCLC fit')
    ap.add_argument('--prior-seed', type=int, default=20261123, help='seed for the Monte Carlo prior summaries')
    ap.add_argument('--prior-mcse', type=float, default=0.00025,
                    help='largest Monte Carlo standard error allowed in the prior summaries')
    ap.add_argument('-o', '--outdir', default='tables')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    N = load_fit(args.nsclc_fit, pr.read_cmdstan_csv)
    S = load_fit(args.sclc_fit, prs.read_cmdstan_csv)
    with open(args.data) as f:
        d = json.load(f)
    raw = np.array(d['raw_plco'])
    cent = np.percentile(raw, CENTILES)
    x_cent = cent - d['mean_plco']
    col_labels = [f'{100 / (1 + np.exp(-c)):.1f}%, {q}th' for c, q in zip(cent, CENTILES)]
    print(f'{len(N["lp__"])} NSCLC and {len(S["lp__"])} SCLC posterior draws; PLCO centiles (raw): '
          + ', '.join(f'{q}th {c:.3f}' for q, c in zip(CENTILES, cent)))

    pp = {'prior': (np.concatenate([read_gq_nsclc(p) for p in args.nsclc_prior]),
                    np.concatenate([read_gq_sclc(p) for p in args.sclc_prior])),
          'post': (np.concatenate([read_gq_nsclc(p) for p in args.nsclc_post]),
                   np.concatenate([read_gq_sclc(p) for p in args.sclc_post]))}
    obs = []
    for reader, path, K in ((read_obs_nsclc, args.nsclc_observed, 7), (read_obs_sclc, args.sclc_observed, 2)):
        _, cell = reader(path)
        o = np.zeros((2, K, 7), int)
        np.add.at(o, tuple(cell.T), 1)
        obs.append(o)
    print(f'predictive draws: prior {len(pp["prior"][0])} / {len(pp["prior"][1])}, '
          f'posterior {len(pp["post"][0])} / {len(pp["post"][1])} (NSCLC / SCLC); '
          f'observed {obs[0].sum()} NSCLC, {obs[1].sum()} SCLC')

    # PSIS weights (fits matched by lp__; posterior predictive GQ through their fitted_params fits)
    wN = wS = None
    w_post = [None, None]
    used = []
    for h, (path, fits, post, P) in enumerate(((args.nsclc_psis_weights, args.nsclc_fit, args.nsclc_post, N),
                                               (args.sclc_psis_weights, args.sclc_fit, args.sclc_post, S))):
        if not path:
            continue
        w = pw.normalise(np.concatenate(pw.align(path, [pw.read_lp(p) for p in fits], fits)))
        if len(w) != len(P['lp__']):
            raise SystemExit(f'{path}: {len(w)} weights for {len(P["lp__"])} posterior draws')
        wp = pw.normalise(np.concatenate(pw.gq_weights(path, post)))
        if len(wp) != len(pp['post'][h]):
            raise SystemExit(f'{path}: {len(wp)} weights for {len(pp["post"][h])} posterior predictive draws')
        if h == 0:
            wN = w
        else:
            wS = w
        w_post[h] = wp
        used.append(f'{"NSCLC SCLC".split()[h]} {path}')
        print(f'{"NSCLC SCLC".split()[h]}: PSIS-weighted (Kish ESS of the weights {pw.kish_ess(w):.0f} of {len(w)})')

    priors = prior_summaries(args.prior_seed, args.prior_mcse)
    for name, n, w in priors['mc']:
        print(f'prior {name}: {n:,} accepted Monte Carlo draws, largest MCSE {w:.6f}')

    tables = [('table-1a', table_1a(N, S, x_cent, col_labels, wN, wS)),
              ('table-1b', table_1b(N, S, x_cent, col_labels, d['beta_former'], d['beta_current'],
                                    wN=wN, wS=wS)),
              ('table-2', table_2(N, wN=wN)),
              ('supp-table-1', supp_1(N, S, priors, wN, wS)),
              ('supp-table-2', supp_2(pp, obs, tuple(w_post)))]
    if used:
        note = ('Posterior summaries are PSIS-weighted (importance-weighted from the fit to thinned data to '
                'the full data; weights: ' + '; '.join(used) + ').')
        for name, t in tables:
            if name == 'table-2' and wN is None:
                continue
            t['notes'].append(note + (' Prior predictive distributions are unweighted.'
                                      if name == 'supp-table-2' else ''))
    with open(os.path.join(args.outdir, 'tables.md'), 'w') as f:
        f.write('\n'.join(to_markdown(t) for _, t in tables))
    for name, t in tables:
        to_csv(t, os.path.join(args.outdir, f'{name}.csv'))
    print('wrote', os.path.join(args.outdir, 'tables.md'), 'and CSVs')
    try:
        to_docx([t for _, t in tables], os.path.join(args.outdir, 'tables.docx'))
        print('wrote', os.path.join(args.outdir, 'tables.docx'))
    except ImportError:
        print('python-docx not installed: tables.docx not written')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Simulate synthetic trial data from the NSCLC screening model
(nsclc-model-marginal.stan; the priors are the same as in the archived
nsclc-model-updated-priors.stan).

Data-generating process:

  * Baseline state at the prevalence screen (no lung cancer, or undiagnosed
    NSCLC at stage 1..7) from a multinomial logit in the centred PLCOm2012
    predictor: softmax(0, b_NSCLC_plco * plco + b_NSCLC_cons).
  * Gompertz time from randomisation to stage 1, with
        shape = m_NSCLC * (beta_former + (beta_current - beta_former) * cigsmok)
        rate  = exp(c_NSCLC + m_NSCLC * plco).
  * In undiagnosed stage k, competing exponential times to progression
    (rate z * lambda[k], none from stage 7) and clinical presentation
    (rate xi[k]). z = exp(sigma_re * eta - sigma_re^2 / 2), eta ~ N(0, 1), is
    a mean-one log-normal random effect.
  * Screens at exactly 0, 1 and 2 years (the last is the final screen), which
    detect undiagnosed NSCLC at stage k with probability
    inv_logit(logit_sens[arm, k]), or inv_logit(logit_sens[arm, k] +
    log_or_sens_final[arm]) at the final screen. There are no false positives.
  * Right censoring at exactly 8 years (6 years after the final screen).

Each patient has up to 6 records:
    screen(0), follow-up(0, 1), screen(1), follow-up(1, 2), screen(2), follow-up(2, 8)
stopping at the first diagnosis.

True values come from one of two modes:

  --mode prior (default): drawn from tightened versions of the model's
      priors, for recovery tests:
        * gamma priors (lambda, xi, log_or_sens_final, b_NSCLC_plco) use
          shape 25 with the same means;
        * beta priors on sensitivity keep their means with effective sample
          size 50 instead of 20;
        * normal priors use half the standard deviation, i.e. the
          (m_NSCLC, c_NSCLC) covariance is divided by 4;
        * sigma_re comes from its gamma(16, 16 / 0.607) prior unchanged;
        * the ordering constraints on logit_sens and xi are respected by
          rejection sampling, so the draws follow the model's implied priors.
  --mode targets: the earlier fixed values (natural-history targets,
      c_NSCLC calibrated to ~5% diagnosed); see fixed_truth().

Usage:
    python simulate_nsclc.py --n-sets 4 --seed 101
writes data/simulated-nsclc-recovery-<i>.json and data/nsclc-recovery-truth-<i>.json.
"""

import argparse
import json

import numpy as np

# Inputs taken from the simulated SCLC data
SCLC_DATA = "data/simulated-sclc.json"

N_PATIENTS = 50_000
PLCO_SD = 0.7525879
P_CURRENT_SMOKER = 0.4819
SCREEN_TIMES = [0.0, 1.0, 2.0]
CENSOR_TIME = 8.0

# Prior settings, as in nsclc-model-marginal.stan
LAMBDA_PRIOR_MEAN = np.array([1.56143, 1.56143, 1.56143, 1.375, 2.02174, 0.80634])
XI_PRIOR_MEAN = np.array([0.08692, 0.08692, 0.08692, 0.1875, 0.15217, 0.41317, 1.35135])
SENS_PRIOR_MEAN = np.array([0.7, 0.7, 0.7, 0.7, 0.8, 0.9, 0.95])
LOG_OR_FINAL_PRIOR_MEAN = 0.082
B_PLCO_PRIOR_MEAN = 1.0
B_CONS_PRIOR = (-8.0, 2.0)
MC_PRIOR_MEAN = np.array([2.0, -7.5])
MC_PRIOR_COV = np.array([[1.0, -0.8], [-0.8, 25.0 / 16.0]])
SIGMA_RE_PRIOR_SHAPE = 16.0
SIGMA_RE_PRIOR_MEAN = 0.607

# Tightening for drawing true values
GAMMA_SHAPE = 25.0
BETA_ESS = 50.0
NORMAL_SD_FACTOR = 0.5


def ordered_draw(sampler, rng, batch=20_000):
    """One draw from sampler(rng, n) (an n x k array), conditional on its
    elements being strictly increasing, by rejection."""
    while True:
        x = sampler(rng, batch)
        ok = np.all(np.diff(x, axis=1) > 0, axis=1)
        if ok.any():
            return x[np.argmax(ok)]


def draw_truth(rng):
    """True values drawn from the tightened priors."""
    logit_sens = []
    for _ in range(2):
        a, b = SENS_PRIOR_MEAN * BETA_ESS, (1 - SENS_PRIOR_MEAN) * BETA_ESS
        s = ordered_draw(lambda r, n: r.beta(a, b, size=(n, 7)), rng)
        logit_sens.append(np.log(s / (1 - s)))
    lam = rng.gamma(GAMMA_SHAPE, LAMBDA_PRIOR_MEAN / GAMMA_SHAPE)
    xi = ordered_draw(lambda r, n: r.gamma(GAMMA_SHAPE, XI_PRIOR_MEAN / GAMMA_SHAPE, size=(n, 7)), rng)
    while True:
        m, c = rng.multivariate_normal(MC_PRIOR_MEAN, MC_PRIOR_COV * NORMAL_SD_FACTOR**2)
        if m > 0:
            break
    return {
        "logit_sens": [ls.tolist() for ls in logit_sens],
        "log_or_sens_final": rng.gamma(GAMMA_SHAPE, LOG_OR_FINAL_PRIOR_MEAN / GAMMA_SHAPE, 2).tolist(),
        "b_NSCLC_plco": float(rng.gamma(GAMMA_SHAPE, B_PLCO_PRIOR_MEAN / GAMMA_SHAPE)),
        "b_NSCLC_cons": rng.normal(B_CONS_PRIOR[0], B_CONS_PRIOR[1] * NORMAL_SD_FACTOR, 7).tolist(),
        "c_NSCLC": float(c),
        "m_NSCLC": float(m),
        "lambda": lam.tolist(),
        "xi": xi.tolist(),
        "sigma_re": float(rng.gamma(SIGMA_RE_PRIOR_SHAPE, SIGMA_RE_PRIOR_MEAN / SIGMA_RE_PRIOR_SHAPE)),
    }


def fixed_truth():
    """The earlier fixed true values: for a typical patient (z = 1) without
    screening, stages 1..7 present with probabilities 0.01, 0.01, 0.03,
    0.05, 0.1, 0.4 and 0.4, the expected preclinical duration for stage-7
    presenters is 4 years (equal mean sojourn of 4/7 years per stage),
    baseline prevalence is roughly steady-state, and c_NSCLC gives about 5%
    diagnosed by 8 years."""
    p_present = np.array([0.01, 0.01, 0.03, 0.05, 0.1, 0.4, 0.4])
    reach = 1 - np.concatenate(([0.0], np.cumsum(p_present)[:-1]))
    psi = np.full(7, 7 / 4.0)
    xi = psi * p_present / reach
    c = -5.362
    sens_base, step = [0.0, -0.8], [np.exp(-0.3), np.exp(-0.5)]
    return {
        "logit_sens": [(sens_base[g] + step[g] * np.arange(7)).tolist() for g in range(2)],
        "log_or_sens_final": [0.3, 0.2],
        "b_NSCLC_plco": 1.0,
        "b_NSCLC_cons": (c + np.log(reach / psi)).tolist(),
        "c_NSCLC": c,
        "m_NSCLC": 1.0,
        "lambda": (psi - xi)[:-1].tolist(),
        "xi": xi.tolist(),
        "sigma_re": 0.5,
    }


def simulate(truth, beta_former, beta_current, mean_plco, rng, n=N_PATIENTS):
    lam, xi = np.array(truth["lambda"]), np.array(truth["xi"])
    logit_sens = np.array(truth["logit_sens"])
    sens = 1 / (1 + np.exp(-logit_sens))
    sens_final = 1 / (1 + np.exp(-(logit_sens + np.array(truth["log_or_sens_final"])[:, None])))
    m, c = truth["m_NSCLC"], truth["c_NSCLC"]

    raw_plco = rng.normal(mean_plco, PLCO_SD, n)
    plco = raw_plco - mean_plco
    cigsmok = (rng.random(n) < P_CURRENT_SMOKER).astype(int)
    rndgroup = rng.integers(1, 3, n)
    eta = rng.standard_normal(n)
    sigma = truth["sigma_re"]
    z = np.exp(sigma * eta - 0.5 * sigma**2)

    # Baseline state: 0 = no lung cancer, k = undiagnosed stage k
    lin = np.column_stack([np.zeros(n),
                           truth["b_NSCLC_plco"] * plco[:, None] + np.array(truth["b_NSCLC_cons"])[None, :]])
    probs = np.exp(lin - lin.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    baseline = (rng.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)

    # Gompertz incidence time, by inverting rate/shape * (exp(shape t) - 1) = E
    shape = m * (beta_former + (beta_current - beta_former) * cigsmok)
    rate = np.exp(c + m * plco)
    t_inc = np.log1p(shape * rng.exponential(size=n) / rate) / shape

    # Walk each patient through the stages. entry[:, k] is the time of
    # entering stage k (1..7), inf if never entered.
    lam_ext = np.append(lam, 0.0)
    entry = np.full((n, 8), np.inf)
    stage = np.where(baseline > 0, baseline, 1)
    t = np.where(baseline > 0, 0.0, t_inc)
    active = t < CENSOR_TIME
    entry[active, stage[active]] = t[active]
    t_clin = np.full(n, np.inf)
    clin_stage = np.zeros(n, dtype=int)
    for _ in range(7):
        if not active.any():
            break
        k = stage - 1
        psi = z * lam_ext[k] + xi[k]
        t_exit = t + rng.exponential(size=n) / psi
        presents = rng.random(n) < xi[k] / psi
        done = active & presents
        t_clin[done], clin_stage[done] = t_exit[done], stage[done]
        moves = active & ~presents
        stage = np.where(moves, stage + 1, stage)
        t = np.where(moves, t_exit, t)
        active = moves & (t < CENSOR_TIME)
        entry[active, stage[active]] = t[active]

    u_screen = rng.random((n, len(SCREEN_TIMES)))
    stage_idx = np.arange(8)

    recordtype, event, final_screen, screen_timepoint, t_l, t_r = [], [], [], [], [], []
    pt_first, pt_records = [], []

    # tp: screening round of a screen record (0, 1, 2 for T0, T1, T2), -1 for
    # follow-up records (as screen_timepoint from dta_to_json.py)
    def add(rt, ev, fs, tl, tr, tp=-1):
        recordtype.append(rt)
        event.append(int(ev))
        final_screen.append(fs)
        screen_timepoint.append(tp)
        t_l.append(float(tl))
        t_r.append(float(tr))

    for i in range(n):
        first = len(recordtype)
        prev = 0.0
        diagnosed = False
        for k, s in enumerate(SCREEN_TIMES):
            if k > 0:
                if t_clin[i] <= s:
                    add(2, clin_stage[i], 0, prev, t_clin[i])
                    diagnosed = True
                    break
                add(2, 0, 0, prev, s)
            st = int(np.where(entry[i] <= s, stage_idx, 0).max())
            is_final = int(k == len(SCREEN_TIMES) - 1)
            if st > 0:
                se = (sens_final if is_final else sens)[rndgroup[i] - 1][st - 1]
                if u_screen[i, k] < se:
                    add(1, st, is_final, s, s, tp=k)
                    diagnosed = True
                    break
            add(1, 0, is_final, s, s, tp=k)
            prev = s
        if not diagnosed:
            if t_clin[i] <= CENSOR_TIME:
                add(2, clin_stage[i], 0, prev, t_clin[i])
            else:
                add(2, 0, 0, prev, CENSOR_TIME)
        pt_first.append(first + 1)  # Stan indices are 1-based
        pt_records.append(len(recordtype) - first)

    data = {
        "mean_plco": mean_plco,
        "beta_current": beta_current,
        "beta_former": beta_former,
        "N": len(recordtype),
        "N_pt": n,
        "wt": [1.0] * n,
        "raw_plco": raw_plco.tolist(),
        "cigsmok": cigsmok.tolist(),
        "pt_records": pt_records,
        "pt_first": pt_first,
        "rndgroup": rndgroup.tolist(),
        "recordtype": recordtype,
        "event": event,
        "final_screen": final_screen,
        "screen_timepoint": screen_timepoint,
        "t_l": t_l,
        "t_r": t_r,
    }
    full_truth = dict(truth, eta=eta.tolist(), sens=sens.tolist(), sens_final=sens_final.tolist())
    return data, full_truth


def summarise(data, truth):
    rt, ev = np.array(data["recordtype"]), np.array(data["event"])
    n_diag = np.count_nonzero(ev)
    print(f"  patients: {data['N_pt']}, records: {data['N']}, "
          f"diagnosed: {n_diag} ({n_diag / data['N_pt']:.2%})")
    print("  stage:           " + " ".join(f"{k:5d}" for k in range(1, 8)))
    print("  screen-detected  " + " ".join(f"{((rt == 1) & (ev == k)).sum():5d}" for k in range(1, 8)))
    print("  clinical         " + " ".join(f"{((rt == 2) & (ev == k)).sum():5d}" for k in range(1, 8)))
    print("  m_NSCLC %.3f, c_NSCLC %.3f, sigma_re %.3f" % (truth["m_NSCLC"], truth["c_NSCLC"], truth["sigma_re"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["prior", "targets"], default="prior")
    ap.add_argument("--n-sets", type=int, default=4)
    ap.add_argument("--seed", type=int, default=101,
                    help="seed for the first set; set i uses seed + i - 1")
    ap.add_argument("--grainsize", type=int, default=1, help="reduce_sum grain size")
    ap.add_argument("--sclc-data", default=SCLC_DATA,
                    help="SCLC data file supplying beta_current, beta_former and mean_plco")
    ap.add_argument("--prefix", default="data/simulated-nsclc-recovery",
                    help="data files are <prefix>-<i>.json")
    ap.add_argument("--truth-prefix", default="data/nsclc-recovery-truth",
                    help="true values are written to <truth-prefix>-<i>.json")
    args = ap.parse_args()

    with open(args.sclc_data) as f:
        sclc = json.load(f)
    consts = (sclc["beta_former"], sclc["beta_current"], sclc["mean_plco"])

    for i in range(1, args.n_sets + 1):
        seed = args.seed + i - 1
        rng = np.random.default_rng(seed)
        truth = draw_truth(rng) if args.mode == "prior" else fixed_truth()
        data, full_truth = simulate(truth, *consts, rng)
        data["grainsize"] = args.grainsize
        full_truth.update(seed=seed)

        data_file = f"{args.prefix}-{i}.json"
        truth_file = f"{args.truth_prefix}-{i}.json"
        with open(data_file, "w") as f:
            json.dump(data, f)
        with open(truth_file, "w") as f:
            json.dump(full_truth, f, indent=1)
        print(f"Set {i} (seed {seed}): {data_file}, {truth_file}")
        summarise(data, full_truth)


if __name__ == "__main__":
    main()

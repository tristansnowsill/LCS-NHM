#!/usr/bin/env python3
"""Simulate synthetic trial data from the SCLC screening model (sclc-model.stan).

Data-generating process:

  * Baseline state at the prevalence screen (no lung cancer, undiagnosed
    limited SCLC, undiagnosed extensive SCLC) from a multinomial logit in the
    centred PLCOm2012 predictor: softmax(0, b_SCLC_plco * plco + b_SCLC_cons).
  * Gompertz time from randomisation to limited SCLC, with
        shape = m_SCLC * (beta_former + (beta_current - beta_former) * cigsmok)
        rate  = exp(c_SCLC + m_SCLC * plco).
  * From limited SCLC, competing exponential times to progression (rate mu)
    and clinical presentation (rate phi[1]); from extensive SCLC, an
    exponential time to clinical presentation (rate phi[2]).
  * Screens at exactly 0, 1 and 2 years (the last is the final screen), which
    detect undiagnosed SCLC at stage k with probability
    inv_logit(logit_sens_SCLC[arm, k]), or inv_logit(logit_sens_SCLC[arm, k] +
    log_or_sens_final[arm]) at the final screen. There are no false positives.
  * 125,000 patients by default (see N_PATIENTS).
  * Right censoring at exactly 8 years (6 years after the final screen), as
    in simulate_nsclc.py.

Each patient has up to 6 records:
    screen(0), follow-up(0, 1), screen(1), follow-up(1, 2), screen(2), follow-up(2, 8)
stopping at the first diagnosis.

True values come from one of two modes:

  --mode prior (default): drawn from tightened versions of the model's
      priors, for recovery tests, in the same way as simulate_nsclc.py:
        * gamma priors (mu, phi, log_or_sens_final, b_SCLC_plco) use shape
          25 with the same means;
        * normal priors use half the standard deviation: logit_sens_SCLC,
          b_SCLC_cons, and (m_SCLC, c_SCLC), whose covariance is divided by 4;
        * the ordering of logit_sens_SCLC within each arm and m_SCLC > 0 are
          respected by rejection sampling, so the draws follow the model's
          implied priors.
  --mode centres: fixed values at the prior centres (CENTRE_TRUTH); override
      any of them with --truth, a JSON file containing a subset of its keys.

True values use the Stan parameter names, so they can be compared directly
with the fitted posterior (e.g. with plot_recovery.py).

Usage:
    python simulate_sclc.py --n-sets 4 --seed 201
writes data/simulated-sclc-recovery-<i>.json and data/sclc-recovery-truth-<i>.json.
"""

import argparse
import json

import numpy as np

# Inputs taken from the user's simulated SCLC data, as in simulate_nsclc.py
SCLC_DATA = "data/simulated-sclc.json"

# Larger than the NSCLC sets (50,000): SCLC is rarer, and at 125,000 every
# recovery set drawn with seeds 201-204 has over 300 SCLC cases (the real
# data have 287 in 53,416). Thinning keeps only the cases plus a sample of
# non-cancer patients, so the cohort size hardly affects fitting cost.
N_PATIENTS = 125_000
PLCO_SD = 0.7525879
P_CURRENT_SMOKER = 0.4819
SCREEN_TIMES = [0.0, 1.0, 2.0]
CENSOR_TIME = 8.0

# Prior settings, as in sclc-model.stan
SENS_PRIOR = [(0.0, np.sqrt(2.0)), (5.220108, 2.286569)]  # normal on logit, by stage
LOG_OR_FINAL_PRIOR_MEAN = np.exp(-1.5)
B_PLCO_PRIOR_MEAN = 1.0
B_CONS_PRIOR = (-7.0, 2.0)
MC_PRIOR_MEAN = np.array([2.0, -10.5])
MC_PRIOR_COV = np.array([[1.0, -0.8], [-0.8, 25.0 / 16.0]])
MU_PRIOR_MEAN = 0.277591
PHI_PRIOR_MEAN = np.array([0.149588, 1.960784])

# Tightening for drawing true values
GAMMA_SHAPE = 25.0
NORMAL_SD_FACTOR = 0.5

CENTRE_TRUTH = {
    "logit_sens_SCLC": [[SENS_PRIOR[0][0], SENS_PRIOR[1][0]]] * 2,
    "log_or_sens_final": [LOG_OR_FINAL_PRIOR_MEAN] * 2,
    "b_SCLC_plco": B_PLCO_PRIOR_MEAN,
    "b_SCLC_cons": [B_CONS_PRIOR[0]] * 2,
    "c_SCLC": MC_PRIOR_MEAN[1],
    "m_SCLC": MC_PRIOR_MEAN[0],
    "mu": MU_PRIOR_MEAN,
    "phi": PHI_PRIOR_MEAN.tolist(),
}


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
    sens_mean = np.array([p[0] for p in SENS_PRIOR])
    sens_sd = np.array([p[1] for p in SENS_PRIOR]) * NORMAL_SD_FACTOR
    logit_sens = [ordered_draw(lambda r, n: r.normal(sens_mean, sens_sd, size=(n, 2)), rng)
                  for _ in range(2)]
    while True:
        m, c = rng.multivariate_normal(MC_PRIOR_MEAN, MC_PRIOR_COV * NORMAL_SD_FACTOR**2)
        if m > 0:
            break
    return {
        "logit_sens_SCLC": [ls.tolist() for ls in logit_sens],
        "log_or_sens_final": rng.gamma(GAMMA_SHAPE, LOG_OR_FINAL_PRIOR_MEAN / GAMMA_SHAPE, 2).tolist(),
        "b_SCLC_plco": float(rng.gamma(GAMMA_SHAPE, B_PLCO_PRIOR_MEAN / GAMMA_SHAPE)),
        "b_SCLC_cons": rng.normal(B_CONS_PRIOR[0], B_CONS_PRIOR[1] * NORMAL_SD_FACTOR, 2).tolist(),
        "c_SCLC": float(c),
        "m_SCLC": float(m),
        "mu": float(rng.gamma(GAMMA_SHAPE, MU_PRIOR_MEAN / GAMMA_SHAPE)),
        "phi": rng.gamma(GAMMA_SHAPE, PHI_PRIOR_MEAN / GAMMA_SHAPE).tolist(),
    }


def simulate(truth, beta_former, beta_current, mean_plco, rng, n=N_PATIENTS):
    logit_sens = np.array(truth["logit_sens_SCLC"])
    sens = 1 / (1 + np.exp(-logit_sens))
    sens_final = 1 / (1 + np.exp(-(logit_sens + np.array(truth["log_or_sens_final"])[:, None])))
    mu = truth["mu"]
    phi_L, phi_E = truth["phi"]
    m, c = truth["m_SCLC"], truth["c_SCLC"]

    raw_plco = rng.normal(mean_plco, PLCO_SD, n)
    plco = raw_plco - mean_plco
    cigsmok = (rng.random(n) < P_CURRENT_SMOKER).astype(int)
    rndgroup = rng.integers(1, 3, n)

    # Baseline state: 0 = none, 1 = limited, 2 = extensive
    lin = np.column_stack([np.zeros(n),
                           truth["b_SCLC_plco"] * plco[:, None] + np.array(truth["b_SCLC_cons"])[None, :]])
    probs = np.exp(lin - lin.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    baseline = (rng.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)

    # Gompertz incidence time, by inverting rate/shape * (exp(shape t) - 1) = E
    shape = m * (beta_former + (beta_current - beta_former) * cigsmok)
    rate = np.exp(c + m * plco)
    t_inc = np.log1p(shape * rng.exponential(size=n) / rate) / shape

    # Times of entering limited (T_L) and extensive (T_E) SCLC and of
    # clinical presentation (T_clin); inf if it never happens
    T_L = np.where(baseline == 0, t_inc, 0.0)
    t_prog = rng.exponential(1.0 / mu, n)
    t_pres_L = rng.exponential(1.0 / phi_L, n)
    t_pres_E = rng.exponential(1.0 / phi_E, n)
    progresses = t_prog < t_pres_L
    T_E = np.where(baseline == 2, 0.0, np.where(progresses, T_L + t_prog, np.inf))
    T_clin = np.where(np.isfinite(T_E), T_E + t_pres_E, T_L + t_pres_L)
    clin_stage = np.where(np.isfinite(T_E), 2, 1)

    u_screen = rng.random((n, len(SCREEN_TIMES)))

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
                if T_clin[i] <= s:
                    add(2, clin_stage[i], 0, prev, T_clin[i])
                    diagnosed = True
                    break
                add(2, 0, 0, prev, s)
            st = 0 if s < T_L[i] else (1 if s < T_E[i] else 2)
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
            if T_clin[i] <= CENSOR_TIME:
                add(2, clin_stage[i], 0, prev, T_clin[i])
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
    full_truth = dict(truth, sens_SCLC=sens.tolist(), sens_SCLC_final=sens_final.tolist())
    return data, full_truth


def summarise(data, truth):
    rt, ev = np.array(data["recordtype"]), np.array(data["event"])
    n_diag = np.count_nonzero(ev)
    print(f"  patients: {data['N_pt']}, records: {data['N']}, "
          f"diagnosed: {n_diag} ({n_diag / data['N_pt']:.2%})")
    for stage, name in [(1, "limited"), (2, "extensive")]:
        print(f"  {name:9s}: {((rt == 1) & (ev == stage)).sum():5d} screen-detected, "
              f"{((rt == 2) & (ev == stage)).sum():5d} clinical")
    print("  m_SCLC %.3f, c_SCLC %.3f, mu %.3f, phi %.3f %.3f" % (
        truth["m_SCLC"], truth["c_SCLC"], truth["mu"], *truth["phi"]))
    print("  sens (arm 1) %.3f %.3f, (arm 2) %.3f %.3f" % tuple(np.ravel(truth["sens_SCLC"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["prior", "centres"], default="prior")
    ap.add_argument("--truth", help="--mode centres: JSON file overriding some of the true values")
    ap.add_argument("--n-sets", type=int, default=4)
    ap.add_argument("--n-patients", type=int, default=N_PATIENTS)
    ap.add_argument("--seed", type=int, default=201,
                    help="seed for the first set; set i uses seed + i - 1")
    ap.add_argument("--grainsize", type=int, default=1, help="reduce_sum grain size")
    ap.add_argument("--sclc-data", default=SCLC_DATA,
                    help="SCLC data file supplying beta_current, beta_former and mean_plco")
    ap.add_argument("--prefix", default="data/simulated-sclc-recovery",
                    help="data files are <prefix>-<i>.json")
    ap.add_argument("--truth-prefix", default="data/sclc-recovery-truth",
                    help="true values are written to <truth-prefix>-<i>.json")
    args = ap.parse_args()

    centre = dict(CENTRE_TRUTH)
    if args.truth:
        if args.mode != "centres":
            ap.error("--truth needs --mode centres")
        with open(args.truth) as f:
            overrides = json.load(f)
        unknown = set(overrides) - set(CENTRE_TRUTH)
        if unknown:
            ap.error(f"unknown parameters in --truth: {sorted(unknown)}")
        centre.update(overrides)

    with open(args.sclc_data) as f:
        sclc = json.load(f)
    consts = (sclc["beta_former"], sclc["beta_current"], sclc["mean_plco"])

    for i in range(1, args.n_sets + 1):
        seed = args.seed + i - 1
        rng = np.random.default_rng(seed)
        truth = draw_truth(rng) if args.mode == "prior" else centre
        data, full_truth = simulate(truth, *consts, rng, n=args.n_patients)
        data["grainsize"] = args.grainsize
        data["N_g"] = 1  # no longer used by the model, but still declared
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

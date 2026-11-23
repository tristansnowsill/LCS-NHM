"""Build the posterior predictive cohort for sclc-model.stan.

Reads the full SCLC data (e.g. empirical-sclc.json, from censor_sclc.py) and
the data file used for the fit (e.g. empirical-sclc.thinned.json), and writes:

  -o FILE         the fit's data plus the posterior predictive cohort (N_pp,
                  pp_*), for running the model's generated quantities on the
                  fitted draws (CmdStan generate_quantities / cmdstanr
                  $generate_quantities())
  --observed FILE CSV of the observed SCLC diagnoses: arm, stage, category,
                  type, time, plus the cohort size (for the bootstrap in
                  sclc_ppc.py)

With --empty, only N_pp = 0 and empty pp_* arrays are added (for ordinary
fits with the posterior predictive simulation switched off).

Screens are labelled T0, T1, T2 by the data's screen_timepoint (the
official screening round, from dta_to_json.py).

Categories (as clinical_category in sclc-model.stan):
  1 pre-screening, 2 at T0 screen, 3 T0-T1 interval, 4 at T1 screen,
  5 T1-T2 interval, 6 at T2 screen, 7 post-screening.
A screen-detected diagnosis is "at T<label> screen". A clinical diagnosis at
time t is pre-screening if no screen came before t; post-screening if the last
screen before t was T2 or more than 1.5 years earlier; otherwise T0-T1 if the
last screen was T0 and T1-T2 if it was T1.

Every patient keeps their own covariates, arm and screens. For SCLC patients,
follow-up after diagnosis is unobserved, so their screens after diagnosis
and their end of follow-up come from a proxy: a patient without SCLC in the
same arm with the same smoking status, still under follow-up at the
diagnosis time, with the same screening history up to the diagnosis (the
same T0/T1/T2 labels attended, and no later screen before the diagnosis
time), chosen at random from the --k candidates nearest in PLCO.
"""

import argparse
import json

import numpy as np

CATEGORIES = ['pre-screening', 'T0 screen', 'T0-T1 interval', 'T1 screen',
              'T1-T2 interval', 'T2 screen', 'post-screening']


def clinical_category(t, n_before, s_last, l_last):
    """As in sclc-model.stan (1-based)."""
    if n_before == 0:
        return 1
    if t - s_last > 1.5 or l_last == 2:
        return 7
    if l_last == 0:
        return 3
    return 5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('full', help='full SCLC data (all participants)')
    ap.add_argument('fit_data', help='data file used for the fit (thinned or not)')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--observed', help='CSV of observed SCLC diagnoses')
    ap.add_argument('--empty', action='store_true', help='add N_pp = 0 only')
    ap.add_argument('-k', type=int, default=5, help='proxy chosen from the k nearest in PLCO')
    ap.add_argument('-s', '--seed', type=int, default=20261010)
    args = ap.parse_args()

    with open(args.fit_data) as f:
        out = json.load(f)
    if args.empty:
        out.update(N_pp=0, pp_raw_plco=[], pp_cigsmok=[], pp_rndgroup=[], pp_n_screens=[],
                   pp_screen_time=[], pp_screen_label=[], pp_screen_final=[], pp_censor=[])
        with open(args.output, 'w') as f:
            json.dump(out, f)
        print(f'Wrote {args.output} with N_pp = 0')
        return

    with open(args.full) as f:
        d = json.load(f)
    rt, ev = np.array(d['recordtype']), np.array(d['event'])
    tl, tr, fs = np.array(d['t_l']), np.array(d['t_r']), np.array(d['final_screen'])
    pf, pr = np.array(d['pt_first']) - 1, np.array(d['pt_records'])
    if 'screen_timepoint' not in d:
        raise SystemExit(f'{args.full} has no screen_timepoint (rebuild it with dta_to_json.py)')
    tp = np.array(d['screen_timepoint'])
    n = d['N_pt']
    plco = np.array(d['raw_plco'])
    cig, arm = np.array(d['cigsmok']), np.array(d['rndgroup'])

    scr_t, scr_l, scr_f = [], [], []
    end = np.zeros(n)
    dx_type = np.zeros(n, int)          # 0 none, 1 screen-detected, 2 clinical
    dx_stage = np.zeros(n, int)
    dx_time = np.full(n, np.nan)
    for i in range(n):
        js = range(pf[i], pf[i] + pr[i])
        t = [tl[j] for j in js if rt[j] == 1]
        scr_t.append(t)
        scr_l.append([int(tp[j]) for j in js if rt[j] == 1])
        scr_f.append([int(fs[j]) for j in js if rt[j] == 1])
        last = pf[i] + pr[i] - 1
        end[i] = max(tl[last], tr[last])
        if ev[last] > 0:
            dx_type[i] = rt[last]
            dx_stage[i] = ev[last]
            dx_time[i] = tl[last] if rt[last] == 1 else tr[last]
    sclc = dx_type > 0

    # Observed categories
    obs_cat = np.zeros(n, int)
    for i in np.where(sclc)[0]:
        if dx_type[i] == 1:
            obs_cat[i] = 2 + 2 * scr_l[i][-1]
        else:
            k = sum(1 for s in scr_t[i] if s < dx_time[i])
            obs_cat[i] = clinical_category(dx_time[i], k, scr_t[i][k - 1] if k else 0.0,
                                           scr_l[i][k - 1] if k else 0)

    # Proxies for SCLC patients
    rng = np.random.default_rng(args.seed)
    pool = np.where(~sclc)[0]
    pp_t = [list(t) for t in scr_t]
    pp_l = [list(l) for l in scr_l]
    pp_f = [list(f) for f in scr_f]
    pp_end = end.copy()
    n_relaxed = 0
    for i in np.where(sclc)[0]:
        t = dx_time[i]
        if dx_type[i] == 1:                  # keep screens up to and including the detecting one
            keep = [k for k, s in enumerate(scr_t[i]) if s <= t]
        else:
            keep = [k for k, s in enumerate(scr_t[i]) if s < t]
        labs = {scr_l[i][k] for k in keep}
        last_lab = max(labs) if labs else -1
        base = pool[(arm[pool] == arm[i]) & (cig[pool] == cig[i]) & (end[pool] > t)]

        def same_history(j):
            lj = {l for l, s in zip(scr_l[j], scr_t[j]) if l <= last_lab}
            later_before = any(l > last_lab and s <= t for l, s in zip(scr_l[j], scr_t[j]))
            return lj == labs and not later_before
        cand = np.array([j for j in base if same_history(j)])
        if len(cand) == 0:
            cand = base
            n_relaxed += 1
        near = cand[np.argsort(np.abs(plco[cand] - plco[i]))[:args.k]]
        j = rng.choice(near)
        add = [k for k, l in enumerate(scr_l[j]) if l > last_lab and scr_t[j][k] > t]
        pp_t[i] = [scr_t[i][k] for k in keep] + [scr_t[j][k] for k in add]
        pp_l[i] = [scr_l[i][k] for k in keep] + [scr_l[j][k] for k in add]
        pp_f[i] = [scr_f[i][k] for k in keep] + [scr_f[j][k] for k in add]
        pp_end[i] = end[j]

    pad = lambda x: [list(v) + [0] * (3 - len(v)) for v in x]
    out.update(
        N_pp=n,
        pp_raw_plco=plco.tolist(),
        pp_cigsmok=cig.tolist(),
        pp_rndgroup=arm.tolist(),
        pp_n_screens=[len(v) for v in pp_t],
        pp_screen_time=pad(pp_t),
        pp_screen_label=pad(pp_l),
        pp_screen_final=pad(pp_f),
        pp_censor=pp_end.tolist(),
    )
    with open(args.output, 'w') as f:
        json.dump(out, f)

    print(f'Cohort: {n} patients, {sclc.sum()} SCLC diagnoses; proxies for {sclc.sum()} SCLC '
          f'patients ({n_relaxed} without a history-matched candidate: matched on arm, '
          f'smoking and follow-up only)')
    print('Observed counts (arm, stage: ' + ', '.join(CATEGORIES) + '):')
    for a in (1, 2):
        for st in (1, 2):
            c = [int(((arm == a) & (dx_stage == st) & (obs_cat == k)).sum()) for k in range(1, 8)]
            print(f'  arm {a}, {"limited" if st == 1 else "extensive"}: {c}')
    print(f'Wrote {args.output}')

    if args.observed:
        with open(args.observed, 'w') as f:
            f.write(f'# cohort size N_pt = {n}\n')
            f.write('patient,arm,stage,category,category_name,type,time\n')
            for i in np.where(sclc)[0]:
                f.write(f'{i + 1},{arm[i]},{dx_stage[i]},{obs_cat[i]},{CATEGORIES[obs_cat[i] - 1]},'
                        f'{"screen" if dx_type[i] == 1 else "clinical"},{dx_time[i]:.6g}\n')
        print(f'Wrote {args.observed}')


if __name__ == '__main__':
    main()

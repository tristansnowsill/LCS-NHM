#!/usr/bin/env python3
"""Build the CmdStan JSON data for the lung cancer models from the trial
Stata file (replaces the Stata script).

Reads the .dta with polars_readstat.scan_readstat and writes one JSON with
every participant's screen and follow-up records, in the format of the
earlier data/empirical.json (then use censor_nsclc.py / censor_sclc.py to
make the model-specific files).

Input columns used (one row per participant):
  pid               participant id (output is sorted by pid)
  rndgroup          randomisation group (1 = LDCT, 2 = CXR)
  cigsmok           current smoker (1) or not (0)
  plco              PLCOm2012 linear predictor (written as raw_plco)
  scr_days0/1/2     days from randomisation to the T0/T1/T2 screen
                    (missing if that screen did not happen)
  candx_days        days from randomisation to diagnosis (used for
                    clinically presenting cancers)
  fup_days          days from randomisation to the end of follow-up (alive)
                    or to death
  deathstat         died (true/1) or alive
  sclc              1 if an SCLC was diagnosed, else 0
  stag_sclc         SCLC stage (1 limited, 2 extensive)
  stag_custom8      imputed 8th-edition stage (1 IA1, 2 IA2, 3 IA3, 4 IB,
                    5 II, 6 III, 7 IV); used for NSCLC only
  can_scr           screen result associated with the lung cancer diagnosis
                    (0 no lung cancer, 1 positive screen, 2 negative screen,
                    3 missed screen, 4 post-screening): 0 means no cancer,
                    1 a screen-detected cancer, and 2, 3 or 4 a clinically
                    presenting cancer

A participant with a screen before randomisation (negative scr_days) has
all their days shifted, as if randomisation happened x days earlier, where
-x is their earliest screen day: x is added to scr_days0/1/2, fup_days and
candx_days. Each such participant is reported.

Records, in time order for each participant (times are days / 365):
  * a screen record (recordtype 1, t_l = t_r = screen time) for each
    attended screen;
  * a follow-up record (recordtype 2) for each interval between
    randomisation, screens and the end of the participant's history.
  No zero-length follow-up record is written between randomisation and a
  screen on day 0, or after a last screen on the last day of follow-up
  (unless it carries a death).

Each participant's history ends with:
  * screen-detected cancer: at the latest attended screen (event = stage);
    nothing after it. (The detecting screen is the latest attended screen;
    candx_days is not used for these.)
  * clinically presenting cancer: a follow-up record ending at candx_days
    (event = stage); screens on or after candx_days are not written (a
    screen on the diagnosis day triggers a warning: it should not happen);
  * no cancer: a follow-up record ending at fup_days, with event -1 if the
    participant died and 0 otherwise.

Option --recode-delayed [DAYS] (off by default; DAYS defaults to 270): a
sensitivity check for diagnosis delays. A screen-detected NSCLC (SCLC is not
affected) whose candx_days is more than DAYS after its detecting screen is
recoded as clinically presenting at candx_days: the detecting screen becomes
a negative screen, as do any earlier screens, and the history ends with a
follow-up record at candx_days. This deliberately misclassifies cancers that
were found through screening follow-up, so screen sensitivity will be
underestimated. The output then also has the scalar recoded_delay_days, and
the recoded cancers are counted by arm and screening round.

Output (the format of the earlier data/empirical.json, without the unused
fields ocm_*, gamma_ocm, mean_age, age, m, n and N_NLCA_*):
  N, N_pt                     numbers of records and participants
  pt_first, pt_records        1-based index of each participant's first
                              record, and their number of records
  recordtype                  1 screen, 2 follow-up
  t_l, t_r                    start and end times of each record (years)
  event                       0 nothing, -1 death, k > 0 diagnosis at stage k
                              (NSCLC: stag_custom8; SCLC: stag_sclc)
  sclc                        at the diagnosis record 1 (SCLC) or 0 (NSCLC);
                              -1 on every other record
  final_screen                1 on the T2 screen record (from scr_days2)
  screen_timepoint            0, 1, 2 for the T0, T1, T2 screen records;
                              -1 for follow-up records
  raw_plco, cigsmok, rndgroup per participant
  mean_plco                   mean of plco over all participants
  beta_current, beta_former   Gompertz shape multipliers (fixed below)
  grainsize, N_g              1 and N_pt

Usage:
  python dta_to_json.py trial.dta -o data/empirical.json
  python dta_to_json.py trial.dta -o data/empirical-recode270.json --recode-delayed
"""

import argparse
import json
import sys

import polars as pl

BETA_CURRENT = 0.1096189
BETA_FORMER = 0.0470296
DAYS_PER_YEAR = 365.0
COLUMNS = ['pid', 'rndgroup', 'cigsmok', 'plco', 'scr_days0', 'scr_days1', 'scr_days2',
           'candx_days', 'fup_days', 'deathstat', 'sclc', 'stag_sclc',
           'stag_custom8', 'can_scr']
STAGES = ['IA1', 'IA2', 'IA3', 'IB', 'II', 'III', 'IV']


def load_dta(path):
    import polars_readstat
    return polars_readstat.scan_readstat(path).select(COLUMNS).collect()


class Warnings:
    """Collects warnings by kind, with a few example pids each."""

    def __init__(self):
        self.kinds = {}

    def add(self, kind, pid):
        self.kinds.setdefault(kind, []).append(pid)

    def report(self):
        for kind, pids in self.kinds.items():
            ex = ', '.join(str(p) for p in pids[:10])
            print(f'WARNING: {kind}: {len(pids)} participant(s), e.g. pid {ex}', file=sys.stderr)


def shift_days(r, w):
    """Shift a participant with a screen before randomisation (see above)."""
    days = [r[f'scr_days{k}'] for k in range(3) if r[f'scr_days{k}'] is not None]
    if days and min(days) < 0:
        x = -min(days)
        for c in ('scr_days0', 'scr_days1', 'scr_days2', 'fup_days', 'candx_days'):
            if r[c] is not None:
                r[c] += x
        w.add(f'screen before randomisation: all days shifted by +x so the earliest screen '
              f'is on day 0 (x = {x:g})', r['pid'])
    return r


def truthy(x):
    return x is not None and bool(x)


def build(df, recode_delay=None):
    """Records for every participant in df (a polars DataFrame with COLUMNS).
    recode_delay: if not None, screen-detected NSCLC diagnosed more than this
    many days after the detecting screen are recoded as clinically presenting
    (see --recode-delayed)."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f'missing columns: {missing}')
    df = df.sort('pid')
    if df['pid'].n_unique() != df.height:
        raise SystemExit('pid is not unique')
    for c in ('rndgroup', 'cigsmok', 'plco', 'fup_days'):
        if df[c].null_count():
            raise SystemExit(f'{c} has {df[c].null_count()} missing values')
    bad = df.filter(~pl.col('rndgroup').is_in([1, 2]) | ~pl.col('cigsmok').is_in([0, 1]))
    if bad.height:
        raise SystemExit(f'rndgroup not 1/2 or cigsmok not 0/1 for pid {bad["pid"].to_list()[:10]}')

    w = Warnings()
    rec = {k: [] for k in ('recordtype', 't_l', 't_r', 'event', 'sclc', 'final_screen',
                           'screen_timepoint')}
    pt_first, pt_records = [], []
    recoded = []                        # (arm, screening round, stage) of recoded cancers

    def add(rt, a, b, ev=0, sc=-1, fin=0, tp=-1):
        rec['recordtype'].append(rt)
        rec['t_l'].append(a / DAYS_PER_YEAR)
        rec['t_r'].append(b / DAYS_PER_YEAR)
        rec['event'].append(ev)
        rec['sclc'].append(sc)
        rec['final_screen'].append(fin)
        rec['screen_timepoint'].append(tp)

    for r in df.iter_rows(named=True):
        pid = r['pid']
        start = len(rec['recordtype'])
        r = shift_days(r, w)

        # Attended screens, in time order
        scr = [(r[f'scr_days{k}'], k) for k in range(3) if r[f'scr_days{k}'] is not None]
        if [d for d, _ in scr] != sorted(d for d, _ in scr) or len({d for d, _ in scr}) < len(scr):
            w.add('screens not in T0 < T1 < T2 order (sorted by day)', pid)
        scr.sort()

        # Cancer status
        can_scr = r['can_scr']
        if can_scr not in (0, 1, 2, 3, 4):
            w.add('can_scr missing or not 0-4 (treated as no cancer)', pid)
        cancer = can_scr in (1, 2, 3, 4)
        detected = can_scr == 1
        if cancer:
            is_sclc = r['sclc'] == 1
            stage = r['stag_sclc'] if is_sclc else r['stag_custom8']
            if stage is None or not (1 <= stage <= (2 if is_sclc else 7)):
                w.add(f'{"SCLC" if is_sclc else "NSCLC"} with missing or invalid stage '
                      '(treated as no cancer)', pid)
                cancer = False
        elif r['sclc'] == 1:
            w.add('sclc == 1 but can_scr is 0 or missing (treated as no cancer)', pid)

        if cancer and detected and not scr:
            w.add('screen-detected cancer with no attended screen (treated as clinical)', pid)
            detected = False
        if cancer and detected and not is_sclc and recode_delay is not None:
            if r['candx_days'] is None:
                w.add('screen-detected NSCLC with missing candx_days (not recoded)', pid)
            elif r['candx_days'] - scr[-1][0] > recode_delay:
                detected = False
                recoded.append((int(r['rndgroup']), scr[-1][1], int(stage)))
        if cancer and not detected and r['candx_days'] is None:
            w.add('clinically presenting cancer with missing candx_days (treated as no cancer)', pid)
            cancer = False

        # End of the history
        if cancer and detected:
            end, end_ev = scr[-1][0], int(stage)
        elif cancer:
            end, end_ev = r['candx_days'], int(stage)
            if any(d == end for d, _ in scr):
                w.add('clinically presenting cancer diagnosed on a screen day (screen dropped)', pid)
            if end > r['fup_days']:
                w.add('candx_days after fup_days', pid)
            scr = [(d, k) for d, k in scr if d < end]
        else:
            end, end_ev = r['fup_days'], -1 if truthy(r['deathstat']) else 0
            if any(d > end for d, _ in scr):
                w.add('screen after fup_days (dropped)', pid)
                scr = [(d, k) for d, k in scr if d <= end]

        # Records
        t = 0
        for i, (d, k) in enumerate(scr):
            if d > t:
                add(2, t, d)
            last = cancer and detected and i == len(scr) - 1
            add(1, d, d, ev=end_ev if last else 0, sc=(1 if is_sclc else 0) if last else -1,
                fin=int(k == 2), tp=k)
            t = d
        if not (cancer and detected):
            if end > t or end_ev != 0 or not scr:
                if end < t:
                    w.add('end of history before the last screen', pid)
                add(2, t, end, ev=end_ev, sc=(1 if is_sclc else 0) if cancer else -1)
                if not scr and end == 0 and end_ev == 0:
                    w.add('no screens and no follow-up (one zero-length record written)', pid)

        pt_first.append(start + 1)
        pt_records.append(len(rec['recordtype']) - start)

    w.report()
    if recode_delay is not None:
        print(f'Recoded {len(recoded)} screen-detected NSCLC with delay > {recode_delay:g} days '
              'as clinically presenting (arm, round: count):')
        for a, an in ((1, 'LDCT'), (2, 'CXR')):
            print(f'  {an:4s} ' + ', '.join(
                f'T{k} {sum(1 for x in recoded if x[0] == a and x[1] == k)}' for k in range(3)))
    out = dict(
        N=len(rec['recordtype']),
        N_pt=df.height,
        mean_plco=float(df['plco'].mean()),
        beta_current=BETA_CURRENT,
        beta_former=BETA_FORMER,
        grainsize=1,
        N_g=df.height,
        pt_first=pt_first,
        pt_records=pt_records,
        **rec,
        raw_plco=df['plco'].to_list(),
        cigsmok=[int(x) for x in df['cigsmok']],
        rndgroup=[int(x) for x in df['rndgroup']],
    )
    if recode_delay is not None:
        out['recoded_delay_days'] = recode_delay
    return out


def summarise(d):
    """Counts of diagnoses by arm, cancer type, mode of detection and stage."""
    import numpy as np
    ev, rt, sc = (np.array(d[k]) for k in ('event', 'recordtype', 'sclc'))
    arm = np.repeat(np.array(d['rndgroup']), d['pt_records'])
    print(f'{d["N_pt"]} participants, {d["N"]} records; mean_plco = {d["mean_plco"]:.6g}')
    print(f'deaths without lung cancer: {(ev == -1).sum()}')
    tot = 0
    for a, an in ((1, 'LDCT'), (2, 'CXR')):
        for s, sn, labs in ((0, 'NSCLC', STAGES), (1, 'SCLC', ['limited', 'extensive'])):
            for t, tn in ((2, 'clinical'), (1, 'screen-detected')):
                m = (ev > 0) & (sc == s) & (rt == t) & (arm == a)
                c = np.bincount(ev[m], minlength=len(labs) + 1)[1:]
                tot += m.sum()
                print(f'  {an:4s} {sn:5s} {tn:15s} {m.sum():5d}  ' +
                      ', '.join(f'{l} {n}' for l, n in zip(labs, c)))
    print(f'total cancers: {tot}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dta', help='trial Stata file')
    ap.add_argument('-o', '--output', required=True, help='JSON file to write')
    ap.add_argument('--recode-delayed', type=float, nargs='?', const=270.0, default=None,
                    metavar='DAYS',
                    help='recode screen-detected NSCLC diagnosed more than DAYS (default 270) '
                         'after the screen as clinically presenting (sensitivity check; off by default)')
    args = ap.parse_args()
    d = build(load_dta(args.dta), recode_delay=args.recode_delayed)
    with open(args.output, 'w') as f:
        json.dump(d, f)
    summarise(d)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()

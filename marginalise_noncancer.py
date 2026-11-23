import argparse
import numpy as np
import json

parser = argparse.ArgumentParser(
    prog='marginalise_noncancer',
    description='''
    Prepares full (unthinned) NSCLC data for evaluating the log density of
    nsclc-model-marginal.stan with every non-cancer patient's random effect
    integrated out, e.g. with BridgeStan for PSIS reweighting from thinned
    data to the full data. Non-cancer patients (no record with event > 0) get
    wt = 1 + EPS, so the model marginalises them (it does so for any
    wt != 1) instead of giving them an eta; cancer patients keep wt = 1 and
    their eta, in data order (the same order as in thinned data). If the data
    have no N_pp, N_pp = 0 and empty pp_* arrays are added (as
    nsclc_ppc_data.py --empty does).
    '''
)
parser.add_argument('filename', help='Full NSCLC data, e.g. data/empirical-nsclc.json')
parser.add_argument('-o', '--output', required=True, help='Where to write the modified data')
parser.add_argument('--eps', type=float, default=1e-9,
                    help='Non-cancer patients get wt = 1 + EPS (default 1e-9)')

args = parser.parse_args()

with open(args.filename, 'r') as f:
    data = json.load(f)

first = np.array(data['pt_first']) - 1
n_rec = np.array(data['pt_records'])
event = np.array(data['event'])
cancer = np.array([(event[s:s + n] > 0).any() for s, n in zip(first, n_rec)])

wt = np.array(data.get('wt', [1.0] * data['N_pt']), dtype=float)
if (wt != 1).any():
    raise ValueError('Expected unthinned data with wt = 1 for every patient')
if 1.0 + args.eps == 1.0:
    raise ValueError('EPS is too small: 1 + EPS rounds to 1')
wt[~cancer] = 1.0 + args.eps
data['wt'] = wt.tolist()

if 'N_pp' not in data:
    data['N_pp'] = 0
    for k in ('pp_raw_plco', 'pp_cigsmok', 'pp_rndgroup', 'pp_n_screens', 'pp_screen_time',
              'pp_screen_label', 'pp_screen_final', 'pp_censor'):
        data[k] = []

with open(args.output, 'w') as f:
    json.dump(data, f)

print(f'{args.output}: {cancer.sum()} cancer patients (wt 1, eta), '
      f'{(~cancer).sum()} non-cancer patients (wt 1 + {args.eps:g}, marginalised)')

import argparse
import numpy as np
import json

parser = argparse.ArgumentParser(
    prog='censor_nsclc',
    description='''
    Prepares CmdStan JSON for the NSCLC model from data that also records
    deaths and SCLC diagnoses. Both are treated as censoring: records with
    event -1 (death without a lung cancer diagnosis) and records with
    event > 0 and sclc == 1 (SCLC diagnosis) get event 0. The sclc variable is
    dropped. If the data have no `wt`, every patient gets weight 1.
    '''
)
parser.add_argument('filename')
parser.add_argument('-o', '--output', required=True, help='Where to write the NSCLC data')

args = parser.parse_args()

with open(args.filename, 'r') as f:
    data = json.load(f)

event = np.array(data['event'])
sclc = np.array(data['sclc'])

if ((event > 0) & (sclc == -1)).any():
    raise ValueError('Some diagnoses have no cancer type (event > 0 but sclc == -1)')

death = event == -1
sclc_dx = (event > 0) & (sclc == 1)
event[death | sclc_dx] = 0

data['event'] = event.tolist()
del data['sclc']
if 'wt' not in data:
    data['wt'] = [1.0] * data['N_pt']

with open(args.output, 'w') as f:
    json.dump(data, f)

print(f'Censored {death.sum()} deaths and {sclc_dx.sum()} SCLC diagnoses; '
      f'{(event > 0).sum()} NSCLC diagnoses remain')

import argparse
import numpy as np
import json
from warnings import warn

parser = argparse.ArgumentParser(
    prog='thin_noncancer_records',
    description='''
    Takes CmdStan JSON for a lung cancer screening model and thins non-cancer
    records, keeping every cancer patient. Non-cancer patients are sampled
    systematically with inclusion probability proportional to
    exp(a * (raw_plco - mean_plco)), capped at 1, and get weight `wt` equal to
    the inverse of their inclusion probability. A non-cancer patient's
    contribution to the likelihood grows roughly like exp(m_NSCLC * plco), so
    sampling in proportion to it represents the high-PLCO patients, who
    dominate the likelihood, far better than equal-probability sampling. Given
    several exponents a (the default is 0.5 and 2.5), the size measure is the
    equal mixture of exp(a * plco) / mean(exp(a * plco)), which keeps the
    sample accurate whatever the true PLCO effect.
    Patients are ordered by smoking status, trial arm and PLCO before
    sampling, so the sample is also implicitly stratified by these baseline
    variables. Each patient's inclusion probability depends only on PLCO.
    Non-cancer patients with no screens and no follow-up time contribute
    nothing to the likelihood and are never kept.
    '''
)
parser.add_argument('filename')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-n', type=int, help='How many non-cancer records to leave')
group.add_argument('-p', type=float, help='The proportion of non-cancer records to keep')
parser.add_argument('-a', '--exponent', type=float, nargs='+', default=[0.5, 2.5],
                    help='Inclusion probability is proportional to exp(a * plco), or to an equal mixture of these over several values of a; 0 gives equal probabilities (default 0.5 2.5)')
parser.add_argument('-s', '--seed', type=int, default=1234, help='Ensures reproducibility')
parser.add_argument('--skip-kolmogorov-smirnov', action='store_true', help='Do not check the weighted PLCO distribution after thinning')

args = parser.parse_args()

# Read the JSON file
with open(args.filename, 'r') as f:
    full_data = json.load(f)

# Check how many cancer and non-cancer records there are
N = full_data['N']
N_pt = full_data['N_pt']
(recordtype, event, t_l, t_r, final_screen, rndgroup, cigsmok,
        raw_plco, pt_first, pt_records, wt) = map(
    lambda k: np.array(full_data[k]), [
        'recordtype', 'event', 't_l', 't_r', 'final_screen',
        'rndgroup', 'cigsmok', 'raw_plco', 'pt_first', 'pt_records',
        'wt'
    ])

if (wt != 1.0).any():
    warn('Some records in the provided data have non-unity weights but these will be ignored')

pt_events = np.split(event, pt_first[1:]-1)
cancer_dx = np.array(list(map(lambda x: (x != 0).any(), pt_events)))
cancer_pts = np.arange(N_pt)[cancer_dx]
noncancer_pts = np.arange(N_pt)[np.logical_not(cancer_dx)]
N_cancer = len(cancer_pts)

# Non-cancer patients with no screens and no follow-up time (every record of
# zero length, e.g. lost to follow-up on the day of randomisation) contribute
# exactly nothing to the likelihood, so they get zero probability of being
# kept rather than taking up places in the sample
no_info = np.array([
    not (recordtype[pt_first[pt] - 1:pt_first[pt] - 1 + pt_records[pt]] == 1).any()
    and (t_r[pt_first[pt] - 1:pt_first[pt] - 1 + pt_records[pt]]
         == t_l[pt_first[pt] - 1:pt_first[pt] - 1 + pt_records[pt]]).all()
    for pt in noncancer_pts], dtype=bool)
if no_info.any():
    warn(f'{no_info.sum()} non-cancer patients have no screens and no follow-up time; '
         'they contribute nothing to the likelihood and are never kept')
noncancer_pts = noncancer_pts[~no_info]
N_noncancer = len(noncancer_pts)

N_noncancer_thinned = args.n
if args.n is None:
    N_noncancer_thinned = int(N_noncancer * args.p)
if not 0 < N_noncancer_thinned <= N_noncancer:
    raise ValueError(f'Cannot keep {N_noncancer_thinned} of {N_noncancer} non-cancer patients')

def single_record_indices(pt):
    return np.arange(pt_first[pt]-1, pt_first[pt]+pt_records[pt]-1)

def record_indices(participants):
    return np.concat([single_record_indices(pt) for pt in participants])

# Order non-cancer patients by smoking status, trial arm and PLCO, so that
# systematic sampling spreads the sample evenly over all of them. Only
# baseline variables are used: sorting on post-randomisation information such
# as the record pattern would not change any inclusion probability, but on
# real data it splits the PLCO ordering into short runs and makes the sample
# less accurate.
noncancer_pts = noncancer_pts[np.lexsort((
    raw_plco[noncancer_pts], rndgroup[noncancer_pts], cigsmok[noncancer_pts]
))]

# Inclusion probabilities proportional to exp(a * plco), summing to the
# sample size; any probability above 1 is capped at 1 and the rest rescaled
centred_plco = raw_plco[noncancer_pts] - full_data['mean_plco']
size = sum(np.exp(a * centred_plco) / np.exp(a * centred_plco).mean() for a in args.exponent)
capped = np.zeros(N_noncancer, dtype=bool)
while True:
    incl_prob = np.ones(N_noncancer)
    incl_prob[~capped] = (N_noncancer_thinned - capped.sum()) * size[~capped] / size[~capped].sum()
    if not (incl_prob[~capped] > 1).any():
        break
    capped |= incl_prob > 1

# Systematic sampling: one random start, then unit steps along the
# cumulative inclusion probabilities
rng = np.random.default_rng(args.seed)
cum_prob = np.concat(([0], np.cumsum(incl_prob)))
points = rng.uniform() + np.arange(N_noncancer_thinned)
chosen = np.minimum(np.searchsorted(cum_prob, points, side='right') - 1, N_noncancer - 1)
retained_noncancer = noncancer_pts[chosen]
wt_noncancer = 1 / incl_prob[chosen]

if (wt_noncancer == 1.0).any():
    warn(f'{(wt_noncancer == 1.0).sum()} non-cancer patients were kept with certainty (wt = 1), so the model will give them a random effect as for unweighted patients')

retained = np.concat((retained_noncancer, cancer_pts))

# Check the weighted PLCO distribution of the retained non-cancer patients
# against all non-cancer patients (unless told explicitly to skip). The
# critical value uses Kish's effective sample size of the weights.
if not args.skip_kolmogorov_smirnov:
    grid = np.sort(raw_plco[noncancer_pts])
    ecdf_full = np.arange(1, N_noncancer + 1) / N_noncancer
    order = np.argsort(raw_plco[retained_noncancer])
    cum_wt = np.cumsum(wt_noncancer[order]) / wt_noncancer.sum()
    ecdf_thinned = np.concat(([0], cum_wt))[np.searchsorted(raw_plco[retained_noncancer][order], grid, side='right')]
    ks_stat = np.abs(ecdf_full - ecdf_thinned).max()
    n_eff = wt_noncancer.sum()**2 / (wt_noncancer**2).sum()
    if ks_stat > 1.36 / np.sqrt(n_eff):
        warn(f'The weighted Kolmogorov-Smirnov distance for PLCO is {ks_stat:0.3f}, above the 5% critical value {1.36 / np.sqrt(n_eff):0.3f} for effective sample size {n_eff:0.0f}, suggesting poor thinning')

# Assemble the thinned data
thinned_data = full_data

wt_unsorted = np.concat((wt_noncancer, np.ones(N_cancer)))

retained_order = np.argsort(retained) # We will get the records back in the order they originally appeared
retained = retained[retained_order]
retained_records = record_indices(retained)

thinned_data['N']            = len(retained_records)
thinned_data['N_pt']         = len(retained)
thinned_data['recordtype']   = recordtype[retained_records].tolist()
thinned_data['event']        = event[retained_records].tolist()
thinned_data['t_l']          = t_l[retained_records].tolist()
thinned_data['t_r']          = t_r[retained_records].tolist()
thinned_data['final_screen'] = final_screen[retained_records].tolist()
thinned_data['rndgroup']     = rndgroup[retained].tolist()
thinned_data['cigsmok']      = cigsmok[retained].tolist()
thinned_data['raw_plco']     = raw_plco[retained].tolist()
thinned_data['pt_records']   = pt_records[retained].tolist()
thinned_data['pt_first']     = (np.concat(([0], pt_records[retained][:-1])).cumsum(axis=0)+1).tolist()
thinned_data['wt']           = wt_unsorted[retained_order].tolist()

# Subset any other per-patient or per-record arrays (e.g. age) in the same way
model_keys = {'recordtype', 'event', 't_l', 't_r', 'final_screen', 'rndgroup',
              'cigsmok', 'raw_plco', 'pt_records', 'pt_first', 'wt'}
for k, v in full_data.items():
    if k in model_keys or not isinstance(v, list):
        continue
    if len(v) == N_pt and N_pt != N:
        thinned_data[k] = np.array(v)[retained].tolist()
    elif len(v) == N and N_pt != N:
        thinned_data[k] = np.array(v)[retained_records].tolist()
    elif len(v) in (N, N_pt):
        warn(f'Cannot tell whether {k} is per patient or per record, so it was left unchanged')

# Print the thinned data
print(json.dumps(thinned_data))

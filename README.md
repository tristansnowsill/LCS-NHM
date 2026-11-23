# LCS-NHM: a natural history model for lung cancer screening

Bayesian natural history and screening models for non-small-cell (NSCLC) and small-cell (SCLC) lung cancer, calibrated to participant-level data from the National Lung Screening Trial (NLST). They estimate prevalence, Gompertz incidence, stage progression, clinical presentation and the stage-specific sensitivity of low-dose CT (LDCT) and chest X-ray (CXR). The likelihood is computed exactly for each participant: there is no simulation, and screening history is handled by conditioning.

This repository holds the code for:

> Snowsill TM, Peters JL, Griffin EA, Whyte S, Mandrik O, Hyde CJ. A natural history model for lung cancer screening estimated through Bayesian calibration. Manuscript in revision.

## The models

| | NSCLC (`nsclc-model-marginal.stan`) | SCLC (`sclc-model.stan`) |
|---|---|---|
| Undiagnosed stages | IA1, IA2, IA3, IB, II, III, IV | limited, extensive |
| Prevalence at randomisation | multinomial logistic in the PLCOm2012 linear predictor | same |
| Incidence | Gompertz hazard driven by the PLCOm2012 linear predictor | same |
| Progression | stage-specific rates times a mean-one log-normal random effect per patient | one rate, limited → extensive |
| Clinical presentation | stage-specific rates, increasing with stage | stage-specific rates, increasing with stage |
| Screening | sensitivity by test and stage, increasing with stage, plus a final-screen effect | same |
| State evolution | product integration with uniformization | Gauss–Legendre quadrature |

Each sub-model censors participants at a diagnosis of the other histology and at death without lung cancer.

## Data

The NLST data are **not** included. Researchers can request them from the National Cancer Institute's Cancer Data Access System (https://cdas.cancer.gov/nlst/). This work used data release NLST-519. The scripts assume the data files are in `data/`. The SLURM scripts write the fit outputs to `output/` on the cluster; the analysis scripts' defaults expect them copied to `output/hpc/`.

## Software

- Stata 16.1, for the first data-preparation step
- CmdStan 2.39 or later (models compiled with `STAN_THREADS=true`)
- BridgeStan 2.9.0 and R with `argparse`, `bridgestan`, `cmdstanr`, `posterior` and `data.table` (full-data log densities for PSIS)
- Python 3.10 or later with NumPy (2.0 or later, for weighted quantiles), SciPy, pandas, matplotlib and polars with `polars_readstat`; python-docx is optional (Word tables)

The fits were run on a SLURM cluster: 16 threads per chain, about 12–18 hours per NSCLC chain and 10 minutes per SCLC chain.

## Pipeline

### 1. Data preparation

```bash
# NLST SAS files -> analysis dataset (PLCOm2012, 8th edition stage, SCLC stage): analyse-nlst-2026.do in Stata, saved as trial.dta
python dta_to_json.py trial.dta -o data/empirical.json
python censor_nsclc.py data/empirical.json -o data/empirical-nsclc.json
python censor_sclc.py  data/empirical.json -o data/empirical-sclc.json

# Thin the non-cancer participants (all cancer patients kept; PPS sampling on PLCOm2012 with inverse-probability weights)
python thin_noncancer_records.py data/empirical-nsclc.json -n 100  > data/empirical-nsclc.thinned.tmp.json
python thin_noncancer_records.py data/empirical-sclc.json  -n 1000 > data/empirical-sclc.thinned.tmp.json

# Add the posterior predictive cohort (or N_pp = 0 for fitting)
python nsclc_ppc_data.py data/empirical-nsclc.json data/empirical-nsclc.thinned.tmp.json --empty -o data/empirical-nsclc.thinned.json
python sclc_ppc_data.py  data/empirical-sclc.json  data/empirical-sclc.thinned.tmp.json  --empty -o data/empirical-sclc.thinned.json
python nsclc_ppc_data.py data/empirical-nsclc.json data/empirical-nsclc.thinned.json -o data/empirical-nsclc.thinned.ppc.json --observed data/empirical-nsclc.ppc-observed.csv
python sclc_ppc_data.py  data/empirical-sclc.json  data/empirical-sclc.thinned.json  -o data/empirical-sclc.thinned.ppc.json  --observed data/empirical-sclc.ppc-observed.csv

# Full-data files for the PSIS log densities
python marginalise_noncancer.py data/empirical-nsclc.json -o data/empirical-nsclc.marginal.json
python sclc_ppc_data.py data/empirical-sclc.json data/empirical-sclc.json --empty -o data/empirical-sclc.pp0.json
```

### 2. Sampling from the thinned-data posterior

Compile both models with CmdStan (`STAN_THREADS=true`), then submit `run_nsclc.slurm` and `run_sclc.slurm`. Each runs 4 chains of 1,000 warmup and 2,000 sampling iterations.

### 3. Pareto smoothed importance sampling (PSIS) to the full data

Build the BridgeStan libraries (`<model>_model.so`, with `STAN_THREADS=true`). Then submit `run_full_log_density.slurm` and `run_full_sclc_log_density.slurm`, which call `full-log-density.R` in chunks of 100 draws. Copy the chains and the `psis-input` files to `output/hpc/`, then combine:

```bash
python psis_reweight.py -o psis-nsclc output/hpc/empirical-nsclc.psis-input_*.csv --fit output/hpc/empirical-nsclc.thinned_[1-4].csv
python psis_reweight.py -o psis-sclc  output/hpc/empirical-sclc.psis-input_*.csv  --fit output/hpc/empirical-sclc.thinned_[1-4].csv
```

`psis-weights.csv` holds one weight per draw. The scripts below take it through `--psis-weights` and match it to the draws by `lp__`. `psis_resample.py` produces equal-weight resampled draws instead.

### 4. Analyses, posterior predictive checks, tables and figures

```bash
python strip_warmup.py output/hpc/empirical-nsclc.thinned_1.csv output/hpc/empirical-nsclc.thinned_1.sampling.csv   # each chain, both models

python analyse_nsclc_fit.py -o nsclc-fit --psis-weights psis-nsclc/psis-weights.csv output/hpc/empirical-nsclc.thinned_[1-4].csv
python analyse_sclc_fit.py  -o sclc-fit  --psis-weights psis-sclc/psis-weights.csv  output/hpc/empirical-sclc.thinned_[1-4].csv

# Posterior predictive check: CmdStan generate_quantities on each chain, data = *.thinned.ppc.json, then
python nsclc_ppc.py --observed data/empirical-nsclc.ppc-observed.csv -o nsclc-fit --psis-weights psis-nsclc/psis-weights.csv nsclc-fit/ppc-gq_[1-4].csv
python sclc_ppc.py  --observed data/empirical-sclc.ppc-observed.csv  -o sclc-fit  --psis-weights psis-sclc/psis-weights.csv  sclc-fit/ppc-gq_[1-4].csv

# Prior predictive draws (then generate_quantities with them as fitted_params)
python prior_predictive_draws.py nsclc output/hpc/empirical-nsclc.thinned_1.sampling.csv -o output/prior/empirical-nsclc.prior
python prior_predictive_draws.py sclc  output/hpc/empirical-sclc.thinned_1.sampling.csv  -o output/prior/empirical-sclc.prior

python make_tables.py -o tables --nsclc-psis-weights psis-nsclc/psis-weights.csv --sclc-psis-weights psis-sclc/psis-weights.csv   # see --help for input paths
python plot_psis_sensitivity.py              # sensitivity by stage, LDCT vs CXR
python psis_lp_scatter.py                    # full- vs thinned-data log density
python psis_report_figs.py                   # PSIS diagnostics and parameter shifts
python prevalent_sensitivity.py              # sensitivity of one screen for prevalent NSCLC
```

### 5. Parameter recovery study

```bash
python simulate_nsclc.py --n-sets 4 --seed 101
python simulate_sclc.py  --n-sets 4 --seed 201
# thin each set (-n 100), fit with CmdStan, then
python plot_recovery.py      -o recovery-nsclc --set data/nsclc-recovery-truth-1.json output/simulated-nsclc-recovery-1.thinned.csv ...
python plot_recovery_sclc.py -o recovery-sclc  --set data/sclc-recovery-truth-1.json  output/simulated-sclc-recovery-1.thinned.csv ...
```

Each script's docstring and `--help` give its options and the definitions it uses.

## Licence

MIT; see `LICENSE`.

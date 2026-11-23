#!/bin/bash
#SBATCH --job-name=full_sclc_log_density
#SBATCH --export=ALL,CMDSTAN=/lustre/home/tms211/cmdstan-2.39.0/,BRIDGESTAN=/lustre/home/tms211/bridgestan-2.9.0/,cmdstanr_no_ver_check=TRUE,STAN_NUM_THREADS=4
#SBATCH --array=1-80
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_sclc_log_density-2026-11_%a.out
#SBATCH --error=logs/full_sclc_log_density-2026-11_%a.err

set -euo pipefail

# 2026-11 refit: needs sclc-model_model.so rebuilt from the current
# sclc-model.stan (corrected mu/phi priors, phi ordered).

# Load R
module load R

# Array task 1-80 -> chain 1-4, chunk 1-20 of that chain's draws
# (tasks 1-20 are chain 1, 21-40 chain 2, and so on)
N_DRAWS=2000    # post-warmup draws per chain
N_CHUNKS=20     # chunks per chain
if (( N_DRAWS % N_CHUNKS != 0 )); then
  echo "N_DRAWS ($N_DRAWS) is not a multiple of N_CHUNKS ($N_CHUNKS)" >&2
  exit 1
fi
CHUNK_SIZE=$(( N_DRAWS / N_CHUNKS ))
TASK=$SLURM_ARRAY_TASK_ID
CHAIN=$(( (TASK - 1) / N_CHUNKS + 1 ))
CHUNK=$(( (TASK - 1) % N_CHUNKS + 1 ))
START=$(( (CHUNK - 1) * CHUNK_SIZE + 1 ))
END=$(( CHUNK * CHUNK_SIZE ))

echo "Task $TASK: chain $CHAIN, chunk $CHUNK (draws $START-$END) on $(hostname), started $(date -u)"

# Obtain full log-densities
Rscript full-log-density.R sclc-model_model.so data/empirical-sclc.pp0.json \
  output/empirical-sclc-2026-11.thinned_${CHAIN}.csv \
  output/empirical-sclc-2026-11.psis-input_${CHAIN}.${CHUNK}.csv \
  --range $START $END

echo "Chain $CHAIN chunk $CHUNK finished $(date -u)"

#!/bin/bash
#SBATCH --job-name=nsclc
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive                 # one chain per node: gradients are memory-bandwidth bound
#SBATCH --mem=16G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/empirical-nsclc-2026-11.thinned_%a.out
#SBATCH --error=logs/empirical-nsclc-2026-11.thinned_%a.err

# Fit nsclc-model-marginal.stan to the real NSCLC data thinned to 100
# non-cancer patients: 4 chains as a SLURM array, one chain per node,
# 16 threads each, 1,000 warmup + 2,000 sampling iterations, CmdStan
# random initial values and identity starting metric.
#
# 2026-11 refit: prior means of lambda and xi corrected (IA and III);
# recompile nsclc-model-marginal first. Outputs carry the -2026-11 tag so
# the previous fit (empirical-nsclc.thinned_*) is not overwritten.

set -euo pipefail

# Uncomment if the executable cannot find libtbb.so at run time
# export LD_LIBRARY_PATH=$CMDSTAN/stan/lib/stan_math/lib/tbb${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Load GCC
module load GCC/14.3.0

CHAIN=$SLURM_ARRAY_TASK_ID

echo "Chain $CHAIN on $(hostname), $SLURM_CPUS_PER_TASK CPUs, started $(date -u)"
lscpu | grep -E '^(Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\))' || true

# All chains share the seed; id gives each chain its own random number stream
./nsclc-model-marginal \
  sample num_warmup=1000 num_samples=2000 save_warmup=true \
  id=$CHAIN \
  data file=data/empirical-nsclc.thinned.json \
  random seed=20261012 \
  output file=output/empirical-nsclc-2026-11.thinned_$CHAIN.csv refresh=1 \
  num_threads=$SLURM_CPUS_PER_TASK

echo "Chain $CHAIN finished $(date -u)"


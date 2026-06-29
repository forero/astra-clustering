#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH --mem=0
#SBATCH -t 1:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --open-mode=append
#
# Reanalysis covariance for the weighted-2PCF vectors from the fiducial run.
# No new simulation — partitions the cached c000 box into 64 subboxes.
# Usage: sbatch queue/run_weighted_cov.sh [cosmo] [hod]

cosmo=${1:-c000}
hod=${2:-484}

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering

srun -n 1 -c 256 python scripts/weighted_subbox_cov.py "$cosmo" "$hod"

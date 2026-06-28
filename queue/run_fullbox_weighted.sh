#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH --mem=0
#SBATCH -t 2:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --open-mode=append
#
# ASTRA full-box WEIGHTED-2PCF experiment. Like run_fullbox_cosmo.sh but runs
# 7 weight schemes x 3 statistics (data/astra_random/cross) of weighted xi per
# iteration, so -t is bumped to 2 h (still backfill-friendly).
# Usage: sbatch queue/run_fullbox_weighted.sh <cosmo> <hod> [iterations] [outroot]

cosmo=$1
hod=$2
iters=${3:-3}
outroot=${4:-fullbox_weighted}
if [[ -z "$cosmo" || -z "$hod" ]]; then
    echo "Usage: sbatch queue/run_fullbox_weighted.sh <cosmo> <hod> [iterations] [outroot]" >&2
    exit 1
fi

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering

srun -n 1 -c 256 python scripts/pipeline_fullbox_weighted.py "$cosmo" "$hod" \
    --iterations "$iters" --outroot "$outroot"

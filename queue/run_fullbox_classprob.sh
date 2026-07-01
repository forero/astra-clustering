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
# ASTRA full-box CLASS-PROBABILITY weighted-2PCF experiment. Builds P_class from
# n-prob-iters (>=10) ASTRA realisations (reusing any fbw_rvalues_iter*.npz already
# cached by pipeline_fullbox_weighted.py in the same run directory), then measures
# 9 weighted statistics ONCE. -t matches run_fullbox_weighted.sh (still
# backfill-friendly for a one-off run).
# Usage: sbatch queue/run_fullbox_classprob.sh <cosmo> <hod> [n_prob_iters] [outroot]

cosmo=$1
hod=$2
niters=${3:-10}
outroot=${4:-fullbox_weighted}
if [[ -z "$cosmo" || -z "$hod" ]]; then
    echo "Usage: sbatch queue/run_fullbox_classprob.sh <cosmo> <hod> [n_prob_iters] [outroot]" >&2
    exit 1
fi

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering

srun -n 1 -c 256 python scripts/pipeline_fullbox_classprob.py "$cosmo" "$hod" \
    --n-prob-iters "$niters" --outroot "$outroot"

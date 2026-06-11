#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --open-mode=append

# Usage: sbatch -J astra_<cosmo>_hod<NNN> queue/run_subboxes_cosmo.sh <cosmo> <hod>
# e.g.:  sbatch -J astra_c100_hod179 queue/run_subboxes_cosmo.sh c100 179

cosmo=$1
hod=$2
if [[ -z "$cosmo" || -z "$hod" ]]; then
    echo "Usage: sbatch queue/run_subboxes_cosmo.sh <cosmo> <hod>" >&2
    exit 1
fi

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering

srun -n 1 -c 8 python scripts/pipeline_subboxes_cosmo.py "$cosmo" "$hod"

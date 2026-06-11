#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --open-mode=append

# Full-box pipeline needs a whole CPU node: ~8M-point Delaunay (several GB)
# and corrfunc pair counting on 128 physical cores.
# Usage: sbatch -J astra_fb_<cosmo>_hod<NNN> queue/run_fullbox_cosmo.sh <cosmo> <hod> [iterations]

cosmo=$1
hod=$2
iters=${3:-3}
if [[ -z "$cosmo" || -z "$hod" ]]; then
    echo "Usage: sbatch queue/run_fullbox_cosmo.sh <cosmo> <hod> [iterations]" >&2
    exit 1
fi

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering

srun -n 1 -c 256 python scripts/pipeline_fullbox_cosmo.py "$cosmo" "$hod" --iterations "$iters"

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

# Full-box pipeline needs a whole CPU node: ~8M-point Delaunay (several GB)
# and corrfunc pair counting on 128 physical cores.
# Runtime is ~25 min (24-27 min measured); -t is kept under 1 h so the jobs are
# eligible for short Slurm backfill windows. Do NOT bump back to 8 h: the
# over-reservation chokes scheduling and a multi-hundred-job campaign trickles
# in at ~1 job/h instead of backfilling.
# Usage: sbatch -J astra_fb_<cosmo>_hod<NNN> queue/run_fullbox_cosmo.sh <cosmo> <hod> [iterations] [outroot]

cosmo=$1
hod=$2
iters=${3:-3}
outroot=${4:-fullbox}
if [[ -z "$cosmo" || -z "$hod" ]]; then
    echo "Usage: sbatch queue/run_fullbox_cosmo.sh <cosmo> <hod> [iterations] [outroot]" >&2
    exit 1
fi

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering

srun -n 1 -c 256 python scripts/pipeline_fullbox_cosmo.py "$cosmo" "$hod" \
    --iterations "$iters" --outroot "$outroot"

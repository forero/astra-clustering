#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH --mem=0
#SBATCH -t 1:00:00
#SBATCH -o logs/slurm-%A_%a.out
#SBATCH -e logs/slurm-%A_%a.err
#SBATCH --open-mode=append
#
# Job-array runner for the Tier-3 full campaign.  Each array task reads its line
# (1-indexed by SLURM_ARRAY_TASK_ID) from a manifest of "cosmo hod" pairs and runs
# the full-box pipeline for that pair.  Same compute footprint as
# run_fullbox_cosmo.sh (one whole CPU node, ~25 min at 3 iters).
#
# Usage (via launch_tier3_full.sh):
#   sbatch --array=1-M%THROTTLE queue/run_fullbox_array.sh <manifest> <iters> <outroot>

manifest=$1
iters=${2:-3}
outroot=${3:-fullbox_tier3}

cd /pscratch/sd/f/forero/astra-clustering || exit 1
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$manifest")
cosmo=$(echo "$line" | awk '{print $1}')
hod=$(echo "$line" | awk '{print $2}')
if [[ -z "$cosmo" || -z "$hod" ]]; then
    echo "Empty manifest line ${SLURM_ARRAY_TASK_ID} in $manifest" >&2
    exit 1
fi

# skip if completed since the manifest was built (resume safety)
printf -v hod3 '%03d' "$hod"
if [[ -f "data/${outroot}/${cosmo}_hod${hod3}/fullbox_info.npz" ]]; then
    echo "Already complete: ${cosmo}_hod${hod3} -- skipping."
    exit 0
fi

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

srun -n 1 -c 256 python scripts/pipeline_fullbox_cosmo.py "$cosmo" "$hod" \
    --iterations "$iters" --outroot "$outroot"

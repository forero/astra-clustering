#!/bin/bash
# Submit the nine Fisher (cosmology, HOD) full-box runs in parallel.
# Same (cosmo, hod) pairs as the subbox runs — see CLAUDE.md,
# "Fisher-matrix design".  Each job takes a full CPU node.
#
# Usage:  bash queue/launch_fisher_fullbox.sh [iterations]   (default 3)

cd /pscratch/sd/f/forero/astra-clustering || exit 1

iters=${1:-3}

runs=(
    "c000 484"
    "c100 179"
    "c101 152"
    "c102 556"
    "c103 861"
    "c104 498"
    "c105 589"
    "c112 507"
    "c113 483"
)

for run in "${runs[@]}"; do
    read -r cosmo hod <<< "$run"
    printf -v hod3 '%03d' "$hod"
    sbatch -J "astra_fb_${cosmo}_hod${hod3}" queue/run_fullbox_cosmo.sh "$cosmo" "$hod" "$iters"
done

squeue --me -o "%.10i %.24j %.9P %.2t %.10M %.6D %R"

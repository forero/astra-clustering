#!/bin/bash
# Submit the nine Fisher (cosmology, HOD) subbox runs in parallel.
# Cosmology/HOD pairs from CLAUDE.md, "Fisher-matrix design" (decided 2026-06-10):
#   fiducial c000 hod484; derivative pairs c100/c101 (omega_b), c102/c103 (omega_c),
#   c104/c105 (n_s), c112/c113 (sigma8).
#
# Usage:  bash queue/launch_fisher_subboxes.sh

cd /pscratch/sd/f/forero/astra-clustering || exit 1

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
    sbatch -J "astra_${cosmo}_hod${hod3}" queue/run_subboxes_cosmo.sh "$cosmo" "$hod"
done

squeue --me -o "%.10i %.22j %.9P %.2t %.10M %.6D %R"

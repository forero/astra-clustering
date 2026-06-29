#!/bin/bash
# Minimal weighted-2PCF multi-cosmology campaign.
#
# Mirror of launch_fisher_fullbox.sh, but runs the WEIGHTED-2PCF pipeline
# (pipeline_fullbox_weighted.py via run_fullbox_weighted.sh) on the same nine
# Fisher (cosmology, HOD) pairs.  This gives weighted xi for every scheme on the
# matched +/- grid, so central-difference derivatives d xi_w / d theta can be
# formed for {w_b, w_c, n_s, sigma8} and compared head-to-head against the
# quantile legs.  c000/hod484 is the fiducial (already done); skip-aware on
# fbw_info.npz so reruns only fill gaps.
#
# Covariance for the weighted vectors comes from reanalysis of the c000 run
# (scripts/weighted_subbox_cov.py) — no new sims for C.
#
# Usage:  bash queue/launch_fisher_fullbox_weighted.sh [iterations]   (default 3)

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
    tag="${cosmo}_hod${hod3}"
    if [[ -f "data/fullbox_weighted/${tag}/fbw_info.npz" ]]; then
        echo "skip ${tag} (fbw_info.npz exists)"
        continue
    fi
    sbatch -J "fbw_${cosmo}_hod${hod3}" queue/run_fullbox_weighted.sh "$cosmo" "$hod" "$iters"
done

squeue --me -o "%.10i %.24j %.9P %.2t %.10M %.6D %R"

#!/bin/bash
# Weighted-2PCF c000 HOD ensemble, for the emulability head-to-head.
#
# The quantile pipeline already has the 50 maximin c000 HOD draws on disk
# (data/fullbox/c000_hodNNN/, used by emulator_hod_c000.py); the weighted pipeline
# has only the fiducial hod484.  This submits the weighted pipeline on the SAME 50
# maximin draws so that emulator error can be measured leg-for-leg on identical
# HODs, weighted vs quantile (scripts/emulability_weighted_vs_quantile.py).
#
# Skip-aware on fbw_info.npz (hod484 already done -> 49 new runs).  regular qos,
# matching the c000 +50 quantile array; the runs are ~30-40 min each and backfill.
#
# Usage:  bash queue/launch_weighted_c000_ensemble.sh [iterations]   (default 3)

cd /pscratch/sd/f/forero/astra-clustering || exit 1

iters=${1:-3}
sel=data/hod_calibration/hod_selection_c000.txt

n_sub=0
while read -r hod; do
    [[ -z "$hod" || "$hod" == \#* ]] && continue
    printf -v hod3 '%03d' "$hod"
    tag="c000_hod${hod3}"
    if [[ -f "data/fullbox_weighted/${tag}/fbw_info.npz" ]]; then
        echo "skip ${tag} (fbw_info.npz exists)"
        continue
    fi
    sbatch -J "fbwens_c000_hod${hod3}" queue/run_fullbox_weighted.sh c000 "$hod" "$iters"
    n_sub=$((n_sub + 1))
done < "$sel"

echo "submitted ${n_sub} weighted c000 ensemble runs (iters=${iters})"
squeue --me -o "%.10i %.24j %.9P %.2t %.10M %.6D %R" | head -20
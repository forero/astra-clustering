#!/bin/bash
# Iteration experiment: does raising the ASTRA-random iteration count make the
# noise-limited random-quadrupole vectors learnable?
#
# Picks 10 already-run c000 HOD draws (evenly spaced across the ensemble so they
# span HOD space) and reruns them at a higher iteration count into an ISOLATED
# output root (data/fullbox_iter10/) so the main 3-iteration set is untouched.
# Compare the new per-bin noise (xi_std/sqrt(N)) against the existing 3-iteration
# noise and the cross-draw HOD spread.
#
# Usage: bash queue/launch_iter_experiment.sh [niter] [ndraws]
set -euo pipefail
cd /pscratch/sd/f/forero/astra-clustering

NITER=${1:-10}
NDRAWS=${2:-10}
OUTROOT="fullbox_iter${NITER}"

# evenly-spaced subset of the completed c000 draws
mapfile -t HODS < <(
  ls -d data/fullbox/c000_hod*/ 2>/dev/null \
    | sed -E 's#.*/c000_hod([0-9]+)/#\1#' | sort -n \
    | awk -v n="$NDRAWS" '{a[NR]=$0} END{for(i=0;i<n;i++) print a[1+int(i*(NR-1)/(n-1))]}'
)

echo "Experiment: ${#HODS[@]} c000 draws at ${NITER} iterations -> data/${OUTROOT}/"
echo "Draws: ${HODS[*]}"

for hod in "${HODS[@]}"; do
    hod=$((10#$hod))                       # strip leading zeros
    if [[ -f "data/${OUTROOT}/c000_hod$(printf %03d "$hod")/fullbox_info.npz" ]]; then
        echo "  hod${hod}: already done in ${OUTROOT}, skip"
        continue
    fi
    job="astra_it_c000_hod$(printf %03d "$hod")"
    sbatch -J "$job" -t 02:00:00 \
        queue/run_fullbox_cosmo.sh c000 "$hod" "$NITER" "$OUTROOT"
done

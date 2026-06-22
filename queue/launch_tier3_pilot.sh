#!/bin/bash
# Tier-3 emulator pilot: run the full-box pipeline on the c130-c181 pilot HOD
# selections (scripts/select_tier3_pilot.py), into the ISOLATED root
# data/fullbox_tier3/ so the Fisher dataset (data/fullbox/) is untouched.
#
# 3 ASTRA iterations (cheap, ~25 min, fits the 1-h queue limit); the small scales
# that carry the signal are already signal-rich. Skips runs already complete.
#
# Usage: bash queue/launch_tier3_pilot.sh [cosmo|all] [iterations]
cd /pscratch/sd/f/forero/astra-clustering || exit 1

target=${1:-all}
iters=${2:-3}
OUTROOT=fullbox_tier3
PILOT="c130 c135 c140 c145 c150 c155 c160 c165 c170 c180"
cosmos=$([[ "$target" == "all" ]] && echo "$PILOT" || echo "$target")

queued=$(squeue --me -h -o "%j")
submitted=0; skipped=0
for cosmo in $cosmos; do
    sel="data/tier3_pilot/hod_selection_${cosmo}.txt"
    if [[ ! -f "$sel" ]]; then
        echo "Selection not found: $sel  (run scripts/select_tier3_pilot.py)" >&2
        exit 1
    fi
    while read -r hod; do
        [[ -z "$hod" ]] && continue
        printf -v hod3 '%03d' "$hod"
        job="astra_t3_${cosmo}_hod${hod3}"
        if [[ -f "data/${OUTROOT}/${cosmo}_hod${hod3}/fullbox_info.npz" ]] \
           || grep -qx "$job" <<< "$queued"; then
            skipped=$((skipped + 1)); continue
        fi
        sbatch -J "$job" queue/run_fullbox_cosmo.sh "$cosmo" "$hod" "$iters" "$OUTROOT"
        submitted=$((submitted + 1))
    done < "$sel"
done
echo "Submitted $submitted job(s); skipped $skipped already-complete run(s)."

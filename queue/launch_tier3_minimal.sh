#!/bin/bash
# Minimal-coverage run: fill the cosmology axis to ALL 52 (c130-c181) at N HODs each,
# reusing the 10 pilot cosmologies (50 HODs).  3 iters, monopole-focused; runs on
# REGULAR qos within the user balance (-t 0:35:00 so the cost estimate fits).
# Goal: drive the broad-prior monopole LOCO from ~18x CV toward the ~1x CV
# interpolation floor by densifying the cosmology axis.
#
# Usage: bash queue/launch_tier3_minimal.sh [N=20] [iters=3] [tlimit=0:35:00]
cd /pscratch/sd/f/forero/astra-clustering || exit 1

N=${1:-20}
iters=${2:-3}
tlimit=${3:-0:35:00}
OUTROOT=fullbox_tier3
COSMOS=$(seq 130 181 | sed 's/^/c/')
mkdir -p logs

stamp=$(date +%Y%m%d_%H%M%S)
manifest="data/tier3_pilot/manifest_min${N}_${stamp}.txt"
: > "$manifest"
total=0; todo=0
for cosmo in $COSMOS; do
    sel="data/tier3_pilot/hod_selection_${cosmo}_min${N}.txt"
    if [[ ! -f "$sel" ]]; then
        echo "Selection not found: $sel  (run scripts/select_tier3_min.py $N)" >&2
        exit 1
    fi
    while read -r hod; do
        [[ -z "$hod" ]] && continue
        total=$((total + 1))
        printf -v hod3 '%03d' "$hod"
        [[ -f "data/${OUTROOT}/${cosmo}_hod${hod3}/fullbox_info.npz" ]] && continue
        echo "$cosmo $hod" >> "$manifest"
        todo=$((todo + 1))
    done < "$sel"
done

echo "Manifest: $manifest"
echo "Total selected: $total   already complete: $((total - todo))   to submit: $todo"
[[ "$todo" -eq 0 ]] && { echo "Nothing to do."; exit 0; }

jid=$(sbatch --parsable -A desi -q regular --time="$tlimit" -J astra_t3min \
        --array="1-${todo}%200" \
        queue/run_fullbox_array.sh "$manifest" "$iters" "$OUTROOT")
echo "Submitted array job $jid  (1-${todo}%200, -t $tlimit, iters=$iters)"

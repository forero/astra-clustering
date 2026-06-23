#!/bin/bash
# Tier-3 FULL campaign launcher: full-box pipeline over the whole c130-c181 block
# (52 cosmologies) x the maximin HOD selections (scripts/select_tier3_full.py),
# into the SAME isolated root data/fullbox_tier3/ as the pilot (pilot runs are a
# maximin prefix -> reused).  3 ASTRA iterations (matches the pilot).
#
# Submits ONE throttled job array (scheduler-friendly for thousands of runs)
# instead of thousands of individual sbatch calls.  Re-running rebuilds the
# manifest over only the not-yet-complete runs, so it resumes cleanly.
#
# Usage: bash queue/launch_tier3_full.sh [iters] [throttle]
cd /pscratch/sd/f/forero/astra-clustering || exit 1

iters=${1:-3}
throttle=${2:-500}
OUTROOT=fullbox_tier3
COSMOS=$(seq 130 181 | sed 's/^/c/')
mkdir -p logs data/tier3_pilot

# ---- build the manifest of not-yet-complete (cosmo, hod) pairs ----
stamp=$(date +%Y%m%d_%H%M%S)
manifest="data/tier3_pilot/manifest_full_${stamp}.txt"
: > "$manifest"
total=0; todo=0
for cosmo in $COSMOS; do
    sel="data/tier3_pilot/hod_selection_${cosmo}.txt"
    if [[ ! -f "$sel" ]]; then
        echo "Selection not found: $sel  (run scripts/select_tier3_full.py)" >&2
        exit 1
    fi
    while read -r hod; do
        [[ -z "$hod" ]] && continue
        total=$((total + 1))
        printf -v hod3 '%03d' "$hod"
        if [[ -f "data/${OUTROOT}/${cosmo}_hod${hod3}/fullbox_info.npz" ]]; then
            continue
        fi
        echo "$cosmo $hod" >> "$manifest"
        todo=$((todo + 1))
    done < "$sel"
done

echo "Manifest: $manifest"
echo "Total selected runs: $total   already complete: $((total - todo))   to submit: $todo"
if [[ "$todo" -eq 0 ]]; then
    echo "Nothing to do -- campaign already complete."
    exit 0
fi

# ---- submit the throttled array ----
# forero's per-user desi balance (~535 node-hr) is far below the ~4700 node-hr
# campaign cost, so we run on the OVERRUN qos: free, but low-priority and
# preemptible.  --requeue + the per-task skip-logic make preemption safe (jobs
# resume; re-running this launcher rebuilds the manifest over what's left).
jid=$(sbatch --parsable -A desi -q overrun --requeue -J astra_t3 \
        --array="1-${todo}%${throttle}" \
        queue/run_fullbox_array.sh "$manifest" "$iters" "$OUTROOT")
echo "Submitted overrun array job $jid  (1-${todo}%${throttle}, iters=$iters)"

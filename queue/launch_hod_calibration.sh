#!/bin/bash
# Tier-1 HOD calibration: run the full-box pipeline on the selected c000 HOD
# draws, to measure dxi/dtheta_HOD and subtract the HOD contamination from the
# Fisher cosmology derivatives (see CLAUDE.md, "Fisher-matrix design").
#
# Reads the draw list written by scripts/select_hod_calibration.py and submits
# one full-node job per draw, skipping any c000 draw already completed
# (fullbox_info.npz present).  Each job is ~25 min on a full CPU node.
#
# Usage:  bash queue/launch_hod_calibration.sh [iterations]   (default 3)

cd /pscratch/sd/f/forero/astra-clustering || exit 1

iters=${1:-3}
sel=data/hod_calibration/hod_selection_c000.txt
if [[ ! -f "$sel" ]]; then
    echo "Selection list not found: $sel" >&2
    echo "Run: python scripts/select_hod_calibration.py" >&2
    exit 1
fi

submitted=0
skipped=0
while read -r hod; do
    [[ -z "$hod" ]] && continue
    printf -v hod3 '%03d' "$hod"
    if [[ -f "data/fullbox/c000_hod${hod3}/fullbox_info.npz" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    sbatch -J "astra_fb_c000_hod${hod3}" queue/run_fullbox_cosmo.sh c000 "$hod" "$iters"
    submitted=$((submitted + 1))
done < "$sel"

echo "Submitted $submitted job(s); skipped $skipped already-complete draw(s)."
squeue --me -o "%.10i %.24j %.9P %.2t %.10M %.6D %R"

#!/bin/bash
# Global response model: run the full-box pipeline on the per-cosmology HOD
# ensembles selected by scripts/select_hod_ensemble.py, to fit one global
# xi(theta_cosmo, theta_HOD) regression (compute_response_global.py) whose
# cosmology derivatives are HOD-clean by construction.
#
# Reads data/hod_ensemble/hod_selection_{cosmo}.txt and submits one full-node
# job per (cosmo, hod), skipping any run already complete (fullbox_info.npz).
# Each job is ~25 min on a full CPU node.
#
# Usage:
#   bash queue/launch_hod_ensemble.sh [cosmo|all] [iterations]
#     cosmo       one of c000 c100 c101 c102 c103 c104 c105 c112 c113, or 'all'
#                 (default: all).  Launch one cosmology first to validate.
#     iterations  ASTRA-random iterations per run (default 3)

cd /pscratch/sd/f/forero/astra-clustering || exit 1

target=${1:-all}
iters=${2:-3}
ALL_COSMO="c000 c100 c101 c102 c103 c104 c105 c112 c113"

if [[ "$target" == "all" ]]; then
    cosmos="$ALL_COSMO"
else
    cosmos="$target"
fi

# names of jobs already queued/running, so a re-run does not duplicate work
queued=$(squeue --me -h -o "%j")

submitted=0
skipped=0
for cosmo in $cosmos; do
    sel="data/hod_ensemble/hod_selection_${cosmo}.txt"
    if [[ ! -f "$sel" ]]; then
        echo "Selection list not found: $sel" >&2
        echo "Run: python scripts/select_hod_ensemble.py" >&2
        exit 1
    fi
    while read -r hod; do
        [[ -z "$hod" ]] && continue
        printf -v hod3 '%03d' "$hod"
        job="astra_fb_${cosmo}_hod${hod3}"
        if [[ -f "data/fullbox/${cosmo}_hod${hod3}/fullbox_info.npz" ]] \
           || grep -qx "$job" <<< "$queued"; then
            skipped=$((skipped + 1))
            continue
        fi
        sbatch -J "$job" queue/run_fullbox_cosmo.sh "$cosmo" "$hod" "$iters"
        submitted=$((submitted + 1))
    done < "$sel"
done

echo "Submitted $submitted job(s); skipped $skipped already-complete run(s)."
squeue --me -o "%.10i %.24j %.9P %.2t %.10M %.6D %R"

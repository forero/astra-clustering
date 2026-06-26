#!/bin/bash
# Mirror this repo (code + data + .git history) from $PSCRATCH to the persistent
# CFS project area, so the purge-eligible scratch copy is backed up.
#
# Run this AFTER git commits are ready (so the mirrored .git is up to date).
# Usage:  bash queue/sync_to_cfs.sh
set -euo pipefail

SRC=/pscratch/sd/f/forero/astra-clustering/
DEST=/global/cfs/cdirs/desi/users/forero/astra-clustering/

mkdir -p "$DEST"
rsync -ah --info=progress2 --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.ipynb_checkpoints/' \
  "$SRC" "$DEST"

echo "Synced $SRC -> $DEST at $(date)"

#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:30:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gpus 1
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --open-mode=append
#
# Tier-3 MLP emulator: 10-fold leave-one-cosmology-out + external anchor.
# Usage: sbatch queue/run_emulator_tier3.sh [ensemble] [epochs]

ENS=${1:-5}
EPOCHS=${2:-2000}

cd /pscratch/sd/f/forero/astra-clustering
unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

srun -n 1 python scripts/emulator_tier3_mlp.py --ensemble "$ENS" --epochs "$EPOCHS"

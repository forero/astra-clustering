#!/bin/bash
#SBATCH -J astra_subboxes
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --open-mode=append

unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering

srun -n 1 -c 8 python scripts/pipeline_subboxes.py

#!/usr/bin/env python3
"""
ASTRA box clustering pipeline.

Steps
-----
1. Load HOD mock (los=z, RSD via X_PERP / Y_PERP / Z_RSD columns)
2. Cut a 500 Mpc/h cube centred at the origin
3. Generate ASTRA randoms (1× data) for the Delaunay classification
4. Run ASTRA: Delaunay triangulation → local density r → quantile labels
5. Split data AND randoms into quantiles at the same r bin edges; save
6. Generate geometry randoms (5× data) for the Landy-Szalay 2PCF estimator
   (subbox has open boundaries — periodic BC does not apply)
7. Compute 2PCF monopole (ℓ=0) + quadrupole (ℓ=2) per quantile with pycorr

Usage
-----
  salloc -N 1 -C cpu -q interactive -t 30:00 -A desi -c 8 --mem=32G
  unset PYTHONPATH
  source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
  cd /pscratch/sd/f/forero/astra-clustering
  srun -n 1 -c 8 python scripts/pipeline_single_box.py
"""

import sys
import numpy as np
import pandas as pd
import fitsio
from pathlib import Path
from pycorr import TwoPointCorrelationFunction

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from astra import AstraSplit

# ── configuration ──────────────────────────────────────────────────────────────
HOD_FILE = Path(
    '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
    '/c000_ph000/seed0/hod000.fits'
)
OUT_DIR       = REPO_ROOT / 'data'
BOXSIZE       = 500.0   # Mpc/h — subbox side length
LOS           = 'z'
N_Q           = 4       # number of ASTRA quantiles
N_RAND        = 1       # ASTRA randoms: N_RAND × n_data
N_RAND_GEOM   = 5       # geometry randoms: N_RAND_GEOM × n_data (for LS estimator)
SEED          = 42
NTHREADS      = 8

S_EDGES  = np.linspace(0, 150, 31)   # 30 bins, 0–150 Mpc/h
MU_EDGES = np.linspace(-1, 1, 241)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load HOD and apply RSD ──────────────────────────────────────────────────
print('Loading HOD catalog ...')
data, hdr = fitsio.read(str(HOD_FILE), header=True)

q_par  = hdr['Q_PAR']    # Alcock-Paczynski dilation along LOS
q_perp = hdr['Q_PERP']   # Alcock-Paczynski dilation transverse

# los=z: transverse positions + RSD shift along z
pos_full = np.c_[
    data['X_PERP'] / q_perp,
    data['Y_PERP'] / q_perp,
    data['Z_RSD']  / q_par,
]
print(f'  Full box: {len(pos_full):,} galaxies')

# ── 2. Cut 500 Mpc/h subbox centred at the origin ─────────────────────────────
half = BOXSIZE / 2
mask = (
    (pos_full[:, 0] >= -half) & (pos_full[:, 0] < half) &
    (pos_full[:, 1] >= -half) & (pos_full[:, 1] < half) &
    (pos_full[:, 2] >= -half) & (pos_full[:, 2] < half)
)
positions = pos_full[mask].astype(np.float64)
boxsize   = np.array([BOXSIZE, BOXSIZE, BOXSIZE])
print(f'  Subbox ({BOXSIZE:.0f} Mpc/h): {len(positions):,} galaxies')

# ── 3. Generate ASTRA randoms ─────────────────────────────────────────────────
print('\nGenerating ASTRA randoms ...')
astra = AstraSplit()
rand_positions = astra.generate_uniform_randoms(
    positions, boxsize, n_factor=N_RAND, seed=SEED,
)
print(f'  {len(rand_positions):,} randoms  (factor={N_RAND}×)')

# ── 4. Run ASTRA ───────────────────────────────────────────────────────────────
print('\nBuilding ASTRA dataframe ...')
df_full = astra.build_dataframe(positions, rand_positions)

print('Running ASTRA classification ...')
class_rows = astra.classify(df_full)

print(f'\nAssigning {N_Q} quantile labels ...')
df_class = astra.assign_quantiles(class_rows, n_quantiles=N_Q)

# ── 5. Save per-quantile catalogs ─────────────────────────────────────────────
# TARGETID 0…n_data−1       → index into positions
# TARGETID n_data…n_data+n_rand−1 → index into rand_positions (subtract n_data)
n_data = len(positions)

print('\nSaving per-quantile catalogs ...')
for q in range(1, N_Q + 1):
    df_q_data = df_class[df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
    pos_q = positions[df_q_data['TARGETID'].values]
    np.save(OUT_DIR / f'data_quantile_q{q}.npy', pos_q)
    print(f'  Q{q} data:    {len(pos_q):,} galaxies')

    df_q_rand = df_class[~df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
    rand_q = rand_positions[df_q_rand['TARGETID'].values - n_data]
    np.save(OUT_DIR / f'rand_quantile_q{q}.npy', rand_q)
    print(f'  Q{q} randoms: {len(rand_q):,} randoms')

# ── 6. Geometry randoms for Landy-Szalay estimator ────────────────────────────
# The subbox has open boundaries — the boxsize= periodic-BC flag in pycorr
# must NOT be used.  Instead we pass explicit uniform randoms so pycorr
# computes (DD − 2DR + RR) / RR.
print('\nGenerating geometry randoms for 2PCF ...')
rng_geom    = np.random.default_rng(SEED + 1)
geom_randoms = rng_geom.uniform(
    low=-boxsize / 2, high=boxsize / 2,
    size=(N_RAND_GEOM * len(positions), 3),
)
print(f'  {len(geom_randoms):,} geometry randoms  (factor={N_RAND_GEOM}×)')

# ── 7. Compute 2PCF per quantile ──────────────────────────────────────────────
edges = (S_EDGES, MU_EDGES)

def compute_and_save_tpcf(pos_in, label, out_stem, pos2=None):
    """Auto- or cross-correlation using the Landy-Szalay estimator.

    If pos2 is None: auto-correlation of pos_in.
    If pos2 is given: cross-correlation between pos_in and pos2.
    The same geometry randoms are used for both samples.
    """
    desc = f'{label} ({len(pos_in):,}'
    desc += f' × {len(pos2):,})' if pos2 is not None else ')'
    print(f'  {desc} ...')
    kwargs = dict(
        data_positions1=pos_in,
        randoms_positions1=geom_randoms,
        engine='corrfunc',
        nthreads=NTHREADS,
        compute_sepsavg=True,
        position_type='pos',
        los=LOS,
    )
    if pos2 is not None:
        kwargs['data_positions2'] = pos2
        kwargs['randoms_positions2'] = geom_randoms
    xi = TwoPointCorrelationFunction('smu', edges=edges, **kwargs)
    xi.save(str(OUT_DIR / f'{out_stem}.npy'))
    s, multipoles = xi(ells=(0, 2), return_sep=True)
    xi0, xi2 = multipoles[0], multipoles[1]
    np.savez(OUT_DIR / f'multipoles_{out_stem}.npz', s=s, xi0=xi0, xi2=xi2)
    print(f'    xi0[15]={xi0[15]:.4f}  xi2[15]={xi2[15]:.4f}')

# per-quantile auto-correlations
print('\nComputing 2PCF for data quantiles ...')
for q in range(1, N_Q + 1):
    pos_q = np.load(OUT_DIR / f'data_quantile_q{q}.npy')
    compute_and_save_tpcf(pos_q, f'data Q{q}', f'tpcf_data_q{q}')

print('\nComputing 2PCF for random quantiles ...')
for q in range(1, N_Q + 1):
    rand_q = np.load(OUT_DIR / f'rand_quantile_q{q}.npy')
    compute_and_save_tpcf(rand_q, f'randoms Q{q}', f'tpcf_rand_q{q}')

# full-sample auto-correlation
print('\nComputing 2PCF for full data sample ...')
np.save(OUT_DIR / 'full_data.npy', positions)
compute_and_save_tpcf(positions, 'full data', 'tpcf_full_data')

# cross-correlations: full data × each data quantile
print('\nComputing cross-correlation: full data × data quantiles ...')
for q in range(1, N_Q + 1):
    pos_q = np.load(OUT_DIR / f'data_quantile_q{q}.npy')
    compute_and_save_tpcf(positions, f'full data × data Q{q}',
                          f'tpcf_cross_full_data_q{q}', pos2=pos_q)

# cross-correlations: full data × each random quantile
print('\nComputing cross-correlation: full data × random quantiles ...')
for q in range(1, N_Q + 1):
    rand_q = np.load(OUT_DIR / f'rand_quantile_q{q}.npy')
    compute_and_save_tpcf(positions, f'full data × randoms Q{q}',
                          f'tpcf_cross_full_rand_q{q}', pos2=rand_q)

print('\n=== Done ===')
print(f'Output files in {OUT_DIR}:')
for fn in sorted(OUT_DIR.iterdir()):
    print(f'  {fn.name}')

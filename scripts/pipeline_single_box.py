#!/usr/bin/env python3
"""
ASTRA box clustering pipeline — with multiple ASTRA random realisations.

Steps
-----
1. Load HOD mock (los=z, RSD via X_PERP / Y_PERP / Z_RSD columns)
2. Cut a 500 Mpc/h cube centred at the origin
3. Generate geometry randoms (5× data, fixed seed) for the Landy-Szalay estimator
4. Compute full-data auto-correlation once (deterministic, iteration-independent)
5. Repeat N_ASTRA_ITERATIONS times:
   a. Draw new ASTRA randoms (seed = SEED + iteration)
   b. Run ASTRA: Delaunay triangulation → local density r → quantile labels
   c. Split data AND randoms into N_Q quantiles independently (equal counts each)
   d. Compute 2PCF monopole (ℓ=0) + quadrupole (ℓ=2) per quantile with pycorr
6. Save mean and standard deviation of all multipoles across iterations

Usage
-----
  salloc -N 1 -C cpu -q interactive -t 60:00 -A desi -c 8 --mem=32G
  unset PYTHONPATH
  source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
  cd /pscratch/sd/f/forero/astra-clustering
  srun -n 1 -c 8 python scripts/pipeline_single_box.py
"""

import sys
import time
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import fitsio
from pycorr import TwoPointCorrelationFunction

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from astra import AstraSplit

# ── configuration ──────────────────────────────────────────────────────────────
HOD_FILE = Path(
    '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
    '/c000_ph000/seed0/hod000.fits'
)
OUT_DIR            = REPO_ROOT / 'data'
LOG_DIR            = REPO_ROOT / 'logs'
BOXSIZE            = 500.0   # Mpc/h — subbox side length
LOS                = 'z'
N_Q                = 4       # number of ASTRA quantiles
N_RAND             = 1       # ASTRA randoms: N_RAND × n_data
N_RAND_GEOM        = 5       # geometry randoms factor (for LS estimator)
N_ASTRA_ITERATIONS = 30      # number of independent ASTRA random realisations
SEED               = 42      # ASTRA iteration i uses seed SEED+i
SEED_GEOM          = SEED + 1000   # well-separated from iteration seeds
NTHREADS           = 8

S_EDGES  = np.linspace(0, 150, 16)   # 15 bins, 0–150 Mpc/h
MU_EDGES = np.linspace(-1, 1, 241)

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── logging setup ─────────────────────────────────────────────────────────────
run_tag  = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOG_DIR / f'pipeline_{run_tag}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info(f'Log file: {log_file}')

# ── timing helper ─────────────────────────────────────────────────────────────
@contextmanager
def timer(label):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info(f'  timing [{label}]: {elapsed:.1f}s  ({elapsed/60:.2f} min)')

pipeline_start = time.perf_counter()

# ── 1. Load HOD and apply RSD ──────────────────────────────────────────────────
log.info('Loading HOD catalog ...')
with timer('load HOD'):
    data, hdr = fitsio.read(str(HOD_FILE), header=True)
    q_par  = hdr['Q_PAR']
    q_perp = hdr['Q_PERP']
    pos_full = np.c_[
        data['X_PERP'] / q_perp,
        data['Y_PERP'] / q_perp,
        data['Z_RSD']  / q_par,
    ]
log.info(f'  Full box: {len(pos_full):,} galaxies')

# ── 2. Cut 500 Mpc/h subbox centred at the origin ─────────────────────────────
half = BOXSIZE / 2
mask = (
    (pos_full[:, 0] >= -half) & (pos_full[:, 0] < half) &
    (pos_full[:, 1] >= -half) & (pos_full[:, 1] < half) &
    (pos_full[:, 2] >= -half) & (pos_full[:, 2] < half)
)
positions = pos_full[mask].astype(np.float64)
log.info(f'  Subbox ({BOXSIZE:.0f} Mpc/h): {len(positions):,} galaxies')

# ── 3. Generate geometry randoms (fixed across all iterations) ─────────────────
log.info('Generating geometry randoms ...')
with timer('geometry randoms'):
    rng_geom     = np.random.default_rng(SEED_GEOM)
    geom_randoms = rng_geom.uniform(
        low=positions.min(axis=0), high=positions.max(axis=0),
        size=(N_RAND_GEOM * len(positions), 3),
    )
log.info(f'  {len(geom_randoms):,} geometry randoms  (factor={N_RAND_GEOM}×, seed={SEED_GEOM})')

# ── helper: compute 2PCF ───────────────────────────────────────────────────────
edges = (S_EDGES, MU_EDGES)
astra = AstraSplit()


def compute_tpcf(pos_in, pos2=None):
    """Return (xi_object, s, xi0, xi2) using the fixed geometry randoms."""
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
    s, multipoles = xi(ells=(0, 2), return_sep=True)
    return xi, s, multipoles[0], multipoles[1]


# ── 4. Full-data auto-correlation (constant across iterations) ─────────────────
log.info('Computing 2PCF for full data sample (once) ...')
with timer('full data 2PCF'):
    xi_full, s_ref, xi0_full, xi2_full = compute_tpcf(positions)
xi_full.save(str(OUT_DIR / 'tpcf_full_data.npy'))
np.save(OUT_DIR / 'full_data.npy', positions)
log.info(f'  xi0[7]={xi0_full[7]:.4f}  xi2[7]={xi2_full[7]:.4f}')

# ── 5. Loop over ASTRA realisations ───────────────────────────────────────────
all_stems = (
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)
accum = {stem: {'xi0': [], 'xi2': []} for stem in all_stems}

iter_times = []

for it in range(N_ASTRA_ITERATIONS):
    iter_start = time.perf_counter()
    log.info(f'')
    log.info(f'=== ASTRA iteration {it + 1}/{N_ASTRA_ITERATIONS}  (seed={SEED + it}) ===')
    is_last = (it == N_ASTRA_ITERATIONS - 1)

    # 5a. ASTRA randoms
    with timer('ASTRA randoms'):
        rand_positions = astra.generate_uniform_randoms(
            positions, n_factor=N_RAND, seed=SEED + it,
        )
    log.info(f'  {len(rand_positions):,} ASTRA randoms')

    # 5b. Build dataframe, run ASTRA
    with timer('classify'):
        df_full    = astra.build_dataframe(positions, rand_positions)
        class_rows = astra.classify(df_full)

    with timer('assign quantiles'):
        df_class = astra.assign_quantiles(class_rows, n_quantiles=N_Q)

    # 5c. Split into quantiles
    n_data = len(positions)
    data_q = {}
    rand_q = {}
    for q in range(1, N_Q + 1):
        df_qd     = df_class[df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
        data_q[q] = positions[df_qd['TARGETID'].values]
        df_qr     = df_class[~df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
        rand_q[q] = rand_positions[df_qr['TARGETID'].values - n_data]
        log.info(f'  Q{q}: {len(data_q[q]):,} data  {len(rand_q[q]):,} randoms')

    if is_last:
        log.info('  Saving per-quantile catalogs ...')
        for q in range(1, N_Q + 1):
            np.save(OUT_DIR / f'data_quantile_q{q}.npy', data_q[q])
            np.save(OUT_DIR / f'rand_quantile_q{q}.npy', rand_q[q])

    # 5d. Compute 2PCFs and accumulate
    log.info('  2PCF: data quantiles ...')
    with timer('2PCF data quantiles'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(data_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'tpcf_data_q{q}.npy'))
            accum[f'tpcf_data_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_data_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    log.info('  2PCF: random quantiles ...')
    with timer('2PCF random quantiles'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(rand_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'tpcf_rand_q{q}.npy'))
            accum[f'tpcf_rand_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_rand_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    log.info('  2PCF: cross full data × data quantiles ...')
    with timer('2PCF cross full×data'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(positions, pos2=data_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'tpcf_cross_full_data_q{q}.npy'))
            accum[f'tpcf_cross_full_data_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_cross_full_data_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    log.info('  2PCF: cross full data × random quantiles ...')
    with timer('2PCF cross full×rand'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(positions, pos2=rand_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'tpcf_cross_full_rand_q{q}.npy'))
            accum[f'tpcf_cross_full_rand_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_cross_full_rand_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    iter_elapsed = time.perf_counter() - iter_start
    iter_times.append(iter_elapsed)
    elapsed_total = time.perf_counter() - pipeline_start
    remaining     = np.mean(iter_times) * (N_ASTRA_ITERATIONS - it - 1)
    log.info(
        f'  timing [iteration {it + 1}]: {iter_elapsed:.1f}s  '
        f'| elapsed: {elapsed_total/60:.1f} min  '
        f'| ETA: {remaining/60:.1f} min'
    )

# ── 6. Save averaged multipoles ────────────────────────────────────────────────
log.info('')
log.info('Saving averaged multipoles ...')
with timer('save results'):
    np.savez(OUT_DIR / 'multipoles_tpcf_full_data.npz',
             s=s_ref,
             xi0=xi0_full,     xi0_std=np.zeros_like(xi0_full),
             xi2=xi2_full,     xi2_std=np.zeros_like(xi2_full))

    for stem in all_stems:
        xi0_arr = np.array(accum[stem]['xi0'])   # (N_ASTRA_ITERATIONS, n_bins)
        xi2_arr = np.array(accum[stem]['xi2'])
        np.savez(OUT_DIR / f'multipoles_{stem}.npz',
                 s=s_ref,
                 xi0=xi0_arr.mean(axis=0),
                 xi0_std=xi0_arr.std(axis=0, ddof=1),
                 xi2=xi2_arr.mean(axis=0),
                 xi2_std=xi2_arr.std(axis=0, ddof=1),
                 xi0_all=xi0_arr,
                 xi2_all=xi2_arr)

# ── timing summary ─────────────────────────────────────────────────────────────
total_elapsed = time.perf_counter() - pipeline_start
log.info('')
log.info('=== Timing summary ===')
log.info(f'  N_ASTRA_ITERATIONS : {N_ASTRA_ITERATIONS}')
log.info(f'  N_Q                : {N_Q}')
log.info(f'  n_s_bins           : {len(S_EDGES) - 1}')
log.info(f'  n_data             : {len(positions):,}')
log.info(f'  Iteration times    : min={min(iter_times)/60:.2f}  '
         f'mean={np.mean(iter_times)/60:.2f}  '
         f'max={max(iter_times)/60:.2f}  min')
log.info(f'  Total wall time    : {total_elapsed/60:.2f} min  '
         f'({total_elapsed/3600:.2f} h)')
log.info(f'  Log written to     : {log_file}')

log.info('')
log.info('=== Done ===')
log.info(f'Output files in {OUT_DIR}:')
for fn in sorted(OUT_DIR.iterdir()):
    log.info(f'  {fn.name}')

#!/usr/bin/env python3
"""
ASTRA full-box pipeline for a given (cosmology, HOD) pair.

Computes the same statistics as the subbox pipelines, but on the whole
2000 Mpc/h AbacusSummit box (~4M galaxies) instead of 500 Mpc/h subboxes:

  - full-sample auto-correlation (once; deterministic)
  - per ASTRA iteration (seed = SEED + iteration):
      data quantile autos, random quantile autos,
      full x data-quantile crosses, full x random-quantile crosses

Output goes to  data/fullbox/{cosmo}_hod{NNN}/  with a 'fullbox_' prefix.
Multipole .npz files store per-iteration arrays (xi0_all, xi2_all) plus
mean and std over iterations (ASTRA-randoms noise; cosmic variance is the
subbox pipeline's job).  Statistics use the same Landy-Szalay estimator,
binning and seeds as the subbox runs, so results are directly comparable;
the much larger volume makes the integral-constraint offset ~64x smaller.

Scaling choices for the big box:
  - astra.classify_fast (vectorised edge counting) instead of classify
  - the RR pair count over the 5x geometry randoms is computed once and
    reused by every Landy-Szalay estimator (pycorr R1R2= kwarg).  The
    reused auto-RR differs from a recomputed cross-RR only by self-pairs
    and a 1/N normalisation (verified ~1/N_rand, i.e. < 1e-4 in the first
    s bin at N_rand = 20M, < 1e-7 elsewhere); the convention is identical
    for all cosmologies, so it cancels exactly in finite differences
  - per-quantile catalogs are not written to disk (~200 MB/run); pycorr
    objects from the last iteration are saved instead

Usage (full CPU node; ~3-5 h for 3 iterations):
  sbatch -J astra_fb_c100_hod179 queue/run_fullbox_cosmo.sh c100 179
  (all nine: bash queue/launch_fisher_fullbox.sh)
"""

import sys
import time
import logging
import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import fitsio
from pycorr import TwoPointCorrelationFunction

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from astra import AstraSplit

# ── command line ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='ASTRA full-box pipeline for one (cosmology, HOD) pair')
parser.add_argument('cosmo', help='AbacusSummit cosmology, e.g. c100')
parser.add_argument('hod', type=int, help='HOD index, e.g. 179')
parser.add_argument('--iterations', type=int, default=3,
                    help='number of ASTRA random realisations (default 3)')
parser.add_argument('--outroot', default='fullbox',
                    help="output subdir under data/ (default 'fullbox'); use a "
                         "separate root for experiments so the main set is untouched")
args = parser.parse_args()

COSMO = args.cosmo
HOD   = args.hod

# ── configuration ──────────────────────────────────────────────────────────────
HOD_FILE = Path(
    '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
    f'/{COSMO}_ph000/seed0/hod{HOD:03d}.fits'
)
RUN_TAG            = f'{COSMO}_hod{HOD:03d}'
OUT_DIR            = REPO_ROOT / 'data' / args.outroot / RUN_TAG
LOG_DIR            = REPO_ROOT / 'logs'
PREFIX             = 'fullbox'
FULL_SIZE          = 2000.0   # Mpc/h — box side before AP rescaling (-1000 to +1000)
LOS                = 'z'
N_Q                = 4
N_RAND             = 1        # ASTRA randoms factor
N_RAND_GEOM        = 5        # geometry randoms factor
N_ASTRA_ITERATIONS = args.iterations
SEED               = 42       # iteration i uses seed SEED + i
SEED_GEOM          = SEED + 1000
NTHREADS           = 128      # physical cores on a Perlmutter CPU node

S_EDGES  = np.linspace(0, 150, 16)   # 15 bins, 0–150 Mpc/h
MU_EDGES = np.linspace(-1, 1, 241)

if not HOD_FILE.is_file():
    sys.exit(f'HOD catalog not found: {HOD_FILE}')

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOG_DIR / f'pipeline_{PREFIX}_{RUN_TAG}_{stamp}.log'
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
log.info(f'Cosmology: {COSMO}  HOD: hod{HOD:03d}  iterations: {N_ASTRA_ITERATIONS}')
log.info(f'Input: {HOD_FILE}')
log.info(f'Output directory: {OUT_DIR}')

# ── timing helper ─────────────────────────────────────────────────────────────
@contextmanager
def timer(label):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info(f'  timing [{label}]: {elapsed:.1f}s  ({elapsed/60:.2f} min)')

pipeline_start = time.perf_counter()

# ── 1. Load HOD catalog and apply RSD + AP rescaling ──────────────────────────
log.info('Loading HOD catalog ...')
with timer('load HOD'):
    data, hdr = fitsio.read(str(HOD_FILE), header=True)
    q_par  = hdr['Q_PAR']
    q_perp = hdr['Q_PERP']
    positions = np.c_[
        data['X_PERP'] / q_perp,
        data['Y_PERP'] / q_perp,
        data['Z_RSD']  / q_par,
    ].astype(np.float64)
    del data
log.info(f'  Full box: {len(positions):,} galaxies  (Q_PAR={q_par:.6f}, Q_PERP={q_perp:.6f})')

# theoretical box bounds after AP rescaling
half = FULL_SIZE / 2
lo = np.array([-half / q_perp, -half / q_perp, -half / q_par])
hi = -lo

# ── 2. Geometry randoms (fixed across iterations) ─────────────────────────────
log.info('Generating geometry randoms ...')
with timer('geometry randoms'):
    rng_geom     = np.random.default_rng(SEED_GEOM)
    geom_randoms = rng_geom.uniform(low=lo, high=hi,
                                    size=(N_RAND_GEOM * len(positions), 3))
log.info(f'  {len(geom_randoms):,} geometry randoms  (factor={N_RAND_GEOM}x, seed={SEED_GEOM})')

# ── 2PCF helper: RR computed once, reused by every estimator ──────────────────
edges = (S_EDGES, MU_EDGES)
astra = AstraSplit()
R1R2  = None   # set by the first call


def compute_tpcf(pos_in, pos2=None):
    """Return (xi_object, s, xi0, xi2); reuses the global RR pair count."""
    global R1R2
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
    if R1R2 is not None:
        kwargs['R1R2'] = R1R2
    xi = TwoPointCorrelationFunction('smu', edges=edges, **kwargs)
    if R1R2 is None:
        R1R2 = xi.R1R2
    s, multipoles = xi(ells=(0, 2), return_sep=True)
    return xi, s, multipoles[0], multipoles[1]


# ── 3. Full-sample auto-correlation (once; also computes RR) ──────────────────
log.info('Computing 2PCF for full sample (computes shared RR) ...')
with timer('full data 2PCF + RR'):
    xi_full, s_ref, xi0_full, xi2_full = compute_tpcf(positions)
xi_full.save(str(OUT_DIR / f'{PREFIX}_tpcf_full_data.npy'))
log.info(f'  xi0[7]={xi0_full[7]:.4f}  xi2[7]={xi2_full[7]:.4f}')

# ── 4. Loop over ASTRA realisations ───────────────────────────────────────────
all_stems = (
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)
accum      = {stem: {'xi0': [], 'xi2': []} for stem in all_stems}
iter_times = []
q_counts   = []   # (iteration, quantile) data / random counts

for it in range(N_ASTRA_ITERATIONS):
    iter_start = time.perf_counter()
    log.info('')
    log.info(f'=== ASTRA iteration {it + 1}/{N_ASTRA_ITERATIONS}  (seed={SEED + it}) ===')
    is_last = (it == N_ASTRA_ITERATIONS - 1)

    with timer('ASTRA randoms'):
        rand_positions = astra.generate_uniform_randoms(
            positions, n_factor=N_RAND, seed=SEED + it,
        )
    log.info(f'  {len(rand_positions):,} ASTRA randoms')

    with timer('classify (fast)'):
        df_full  = astra.build_dataframe(positions, rand_positions)
        df_class = astra.classify_fast(df_full)
        del df_full

    with timer('assign quantiles'):
        df_class = astra.assign_quantiles(df_class, n_quantiles=N_Q)

    n_data = len(positions)
    data_q = {}
    rand_q = {}
    counts = {}
    for q in range(1, N_Q + 1):
        df_qd     = df_class[df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
        data_q[q] = positions[df_qd['TARGETID'].values]
        df_qr     = df_class[~df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
        rand_q[q] = rand_positions[df_qr['TARGETID'].values - n_data]
        counts[q] = (len(data_q[q]), len(rand_q[q]))
        log.info(f'  Q{q}: {counts[q][0]:,} data  {counts[q][1]:,} randoms')
    q_counts.append([counts[q] for q in range(1, N_Q + 1)])
    del df_class

    log.info('  2PCF: data quantiles ...')
    with timer('2PCF data quantiles'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(data_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'{PREFIX}_tpcf_data_q{q}.npy'))
            accum[f'tpcf_data_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_data_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    log.info('  2PCF: random quantiles ...')
    with timer('2PCF random quantiles'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(rand_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'{PREFIX}_tpcf_rand_q{q}.npy'))
            accum[f'tpcf_rand_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_rand_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    log.info('  2PCF: cross full data x data quantiles ...')
    with timer('2PCF cross fullxdata'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(positions, pos2=data_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'{PREFIX}_tpcf_cross_full_data_q{q}.npy'))
            accum[f'tpcf_cross_full_data_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_cross_full_data_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    log.info('  2PCF: cross full data x random quantiles ...')
    with timer('2PCF cross fullxrand'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(positions, pos2=rand_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'{PREFIX}_tpcf_cross_full_rand_q{q}.npy'))
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

# ── 5. Save multipoles + run metadata ─────────────────────────────────────────
log.info('')
log.info('Saving multipoles ...')
with timer('save results'):
    xi0_full_arr = xi0_full[None, :]
    xi2_full_arr = xi2_full[None, :]
    np.savez(OUT_DIR / f'{PREFIX}_multipoles_tpcf_full_data.npz',
             s=s_ref,
             xi0=xi0_full, xi0_std=np.zeros_like(xi0_full),
             xi2=xi2_full, xi2_std=np.zeros_like(xi2_full),
             xi0_all=xi0_full_arr, xi2_all=xi2_full_arr)

    for stem in all_stems:
        xi0_arr = np.array(accum[stem]['xi0'])   # (N_ASTRA_ITERATIONS, n_bins)
        xi2_arr = np.array(accum[stem]['xi2'])
        ddof = 1 if N_ASTRA_ITERATIONS > 1 else 0
        np.savez(OUT_DIR / f'{PREFIX}_multipoles_{stem}.npz',
                 s=s_ref,
                 xi0=xi0_arr.mean(axis=0),
                 xi0_std=xi0_arr.std(axis=0, ddof=ddof),
                 xi2=xi2_arr.mean(axis=0),
                 xi2_std=xi2_arr.std(axis=0, ddof=ddof),
                 xi0_all=xi0_arr,
                 xi2_all=xi2_arr)
        log.info(f'  {PREFIX}_multipoles_{stem}.npz saved (N={N_ASTRA_ITERATIONS})')

    np.savez(OUT_DIR / f'{PREFIX}_info.npz',
             cosmo=COSMO,
             hod=HOD,
             hod_file=str(HOD_FILE),
             q_par=q_par,
             q_perp=q_perp,
             seed=SEED,
             seed_geom=SEED_GEOM,
             n_iterations=N_ASTRA_ITERATIONS,
             n_data=len(positions),
             box_lo=lo,
             box_hi=hi,
             quantile_counts=np.array(q_counts),   # (iter, quantile, data/rand)
             s_edges=S_EDGES,
             mu_edges=MU_EDGES)
    log.info(f'  {PREFIX}_info.npz saved')

# ── timing summary ─────────────────────────────────────────────────────────────
total_elapsed = time.perf_counter() - pipeline_start
log.info('')
log.info('=== Timing summary ===')
log.info(f'  Cosmology      : {COSMO}  hod{HOD:03d}')
log.info(f'  Iterations     : {N_ASTRA_ITERATIONS}')
log.info(f'  n_data         : {len(positions):,}')
if iter_times:
    log.info(f'  Iteration times: min={min(iter_times)/60:.2f}  '
             f'mean={np.mean(iter_times)/60:.2f}  '
             f'max={max(iter_times)/60:.2f}  min')
log.info(f'  Total wall time: {total_elapsed/60:.2f} min  ({total_elapsed/3600:.2f} h)')
log.info(f'  Log written to : {log_file}')
log.info('')
log.info('=== Done ===')
for fn in sorted(OUT_DIR.glob(f'{PREFIX}_multipoles_*.npz')):
    log.info(f'  {fn.name}')

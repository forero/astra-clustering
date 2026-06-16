#!/usr/bin/env python3
"""
ASTRA subbox pipeline for a given (cosmology, HOD) pair — Fisher derivative runs.

Same computation as pipeline_subboxes.py: divides the 2000 Mpc/h box into a
4×4×4 grid of 500 Mpc/h subboxes (64 total), one ASTRA random realisation per
subbox.  The cosmology and HOD index are taken from the command line and select
the input catalog; outputs go to a per-run subdirectory

    data/{cosmo}_hod{NNN}/

so the nine Fisher cosmologies (see CLAUDE.md, "Fisher-matrix design") can run
in parallel without clobbering each other.

All seeds are identical across cosmologies (subbox sb_idx uses SEED + sb_idx
for ASTRA randoms, SEED_GEOM + sb_idx for geometry randoms), so the random
noise is matched between cosmologies and cancels in the finite differences.

Every multipole .npz stores the per-subbox arrays (xi0_all, xi2_all) in
addition to the mean and std — these are what the derivative, average and
covariance computations need later.

Usage:
  python scripts/pipeline_subboxes_cosmo.py c100 179
  (submit via queue/run_subboxes_cosmo.sh; launch all nine with
   queue/launch_fisher_subboxes.sh)
"""

import sys
import time
import logging
import argparse
import itertools
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
parser = argparse.ArgumentParser(description='ASTRA subbox pipeline for one (cosmology, HOD) pair')
parser.add_argument('cosmo', help='AbacusSummit cosmology, e.g. c100')
parser.add_argument('hod', type=int, help='HOD index, e.g. 179')
args = parser.parse_args()

COSMO = args.cosmo
HOD   = args.hod

# ── configuration ──────────────────────────────────────────────────────────────
HOD_FILE = Path(
    '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
    f'/{COSMO}_ph000/seed0/hod{HOD:03d}.fits'
)
RUN_TAG     = f'{COSMO}_hod{HOD:03d}'
OUT_DIR     = REPO_ROOT / 'data' / RUN_TAG
LOG_DIR     = REPO_ROOT / 'logs'
PREFIX      = 'subbox'       # same file names as pipeline_subboxes.py outputs
FULL_SIZE   = 2000.0         # Mpc/h — full box side (-1000 to +1000)
SUBBOX_SIZE = 500.0          # Mpc/h — subbox side (4×4×4 = 64 subboxes)
LOS         = 'z'
N_Q         = 4
N_RAND      = 1              # ASTRA randoms factor
N_RAND_GEOM = 5              # geometry randoms factor
SEED        = 42             # subbox sb_idx uses seed SEED + sb_idx
SEED_GEOM   = SEED + 1000   # well-separated from ASTRA seeds
NTHREADS    = 8

S_EDGES  = np.linspace(0, 150, 16)   # 15 bins, 0–150 Mpc/h
MU_EDGES = np.linspace(-1, 1, 241)

if not HOD_FILE.is_file():
    sys.exit(f'HOD catalog not found: {HOD_FILE}')

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── subbox grid ────────────────────────────────────────────────────────────────
# Built after the catalog is loaded: the AP rescaling divides positions by
# Q_PERP/Q_PAR, so the box edge sits at ±(FULL_SIZE/2)/q per axis.  The grid
# must tile that volume exactly — a fixed ±1000 grid leaves empty slabs in
# the edge subboxes whenever q > 1 (e.g. c103: ±985.6 in z), and the LS
# geometry randoms would then cover volume that contains no galaxies.
N_SIDE     = int(FULL_SIZE / SUBBOX_SIZE)   # 4
N_SUBBOXES = N_SIDE ** 3                    # 64

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
log.info(f'Cosmology: {COSMO}  HOD: hod{HOD:03d}')
log.info(f'Input: {HOD_FILE}')
log.info(f'Output directory: {OUT_DIR}')
log.info(f'Subbox grid: {N_SIDE}×{N_SIDE}×{N_SIDE} = {N_SUBBOXES} subboxes '
         f'of nominally {SUBBOX_SIZE:.0f} Mpc/h (exact size set by AP rescaling)')

# ── timing helper ─────────────────────────────────────────────────────────────
@contextmanager
def timer(label):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info(f'  timing [{label}]: {elapsed:.1f}s  ({elapsed/60:.2f} min)')

pipeline_start = time.perf_counter()

# ── 1. Load full HOD catalog (once) ───────────────────────────────────────────
log.info('Loading HOD catalog ...')
with timer('load HOD'):
    data, hdr = fitsio.read(str(HOD_FILE), header=True)
    q_par  = hdr['Q_PAR']
    q_perp = hdr['Q_PERP']
    pos_full = np.c_[
        data['X_PERP'] / q_perp,
        data['Y_PERP'] / q_perp,
        data['Z_RSD']  / q_par,
    ].astype(np.float64)
log.info(f'  Full box: {len(pos_full):,} galaxies  (Q_PAR={q_par:.6f}, Q_PERP={q_perp:.6f})')

# AP-rescaled box bounds and subbox grid
half     = FULL_SIZE / 2
box_lo   = np.array([-half / q_perp, -half / q_perp, -half / q_par])
box_hi   = -box_lo
step     = (box_hi - box_lo) / N_SIDE
subboxes = []
for ix, iy, iz in itertools.product(range(N_SIDE), repeat=3):
    lo = box_lo + np.array([ix, iy, iz]) * step
    subboxes.append((ix, iy, iz, lo, lo + step))
log.info(f'Subbox grid tiles the AP-rescaled box: '
         f'x,y in ±{box_hi[0]:.2f}, z in ±{box_hi[2]:.2f} Mpc/h; '
         f'subbox size {step[0]:.2f} x {step[1]:.2f} x {step[2]:.2f} Mpc/h')

# ── accumulators ──────────────────────────────────────────────────────────────
all_stems = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)
accum    = {stem: {'xi0': [], 'xi2': []} for stem in all_stems}
s_ref    = None
astra    = AstraSplit()
edges    = (S_EDGES, MU_EDGES)
sb_times = []
n_data_sb = []   # galaxies per subbox — saved for diagnostics / weighting

# ── 2. Loop over subboxes ──────────────────────────────────────────────────────
for sb_idx, (ix, iy, iz, lo, hi) in enumerate(subboxes):
    sb_start = time.perf_counter()
    log.info('')
    log.info(
        f'=== Subbox {sb_idx + 1}/{N_SUBBOXES}  '
        f'ix={ix} iy={iy} iz={iz}  '
        f'corner ({lo[0]:.0f},{lo[1]:.0f},{lo[2]:.0f}) ==='
    )
    is_last = (sb_idx == N_SUBBOXES - 1)

    # Cut subbox
    mask = (
        (pos_full[:, 0] >= lo[0]) & (pos_full[:, 0] < hi[0]) &
        (pos_full[:, 1] >= lo[1]) & (pos_full[:, 1] < hi[1]) &
        (pos_full[:, 2] >= lo[2]) & (pos_full[:, 2] < hi[2])
    )
    positions = pos_full[mask]
    n_data_sb.append(len(positions))
    log.info(f'  {len(positions):,} galaxies')

    # Geometry randoms spanning the theoretical subbox volume
    rng_geom     = np.random.default_rng(SEED_GEOM + sb_idx)
    geom_randoms = rng_geom.uniform(low=lo, high=hi,
                                    size=(N_RAND_GEOM * len(positions), 3))

    # 2PCF helper — captures geom_randoms for this subbox
    def compute_tpcf(pos_in, pos2=None):
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

    # Full-data auto-correlation
    log.info('  2PCF: full data ...')
    with timer('full data 2PCF'):
        xi_obj, s, xi0, xi2 = compute_tpcf(positions)
    if s_ref is None:
        s_ref = s
    if is_last:
        xi_obj.save(str(OUT_DIR / f'{PREFIX}_tpcf_full_data.npy'))
        np.save(OUT_DIR / f'{PREFIX}_full_data.npy', positions)
    accum['tpcf_full_data']['xi0'].append(xi0)
    accum['tpcf_full_data']['xi2'].append(xi2)
    log.info(f'    xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    # ASTRA classification
    with timer('ASTRA randoms'):
        rand_positions = astra.generate_uniform_randoms(
            positions, n_factor=N_RAND, seed=SEED + sb_idx,
        )
    with timer('classify'):
        df_full    = astra.build_dataframe(positions, rand_positions)
        class_rows = astra.classify(df_full)
    with timer('assign quantiles'):
        df_class = astra.assign_quantiles(class_rows, n_quantiles=N_Q)

    # Split into quantiles
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
            np.save(OUT_DIR / f'{PREFIX}_data_quantile_q{q}.npy', data_q[q])
            np.save(OUT_DIR / f'{PREFIX}_rand_quantile_q{q}.npy', rand_q[q])

    # 2PCFs
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

    log.info('  2PCF: cross full data × data quantiles ...')
    with timer('2PCF cross full×data'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(positions, pos2=data_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'{PREFIX}_tpcf_cross_full_data_q{q}.npy'))
            accum[f'tpcf_cross_full_data_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_cross_full_data_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    log.info('  2PCF: cross full data × random quantiles ...')
    with timer('2PCF cross full×rand'):
        for q in range(1, N_Q + 1):
            xi_obj, _, xi0, xi2 = compute_tpcf(positions, pos2=rand_q[q])
            if is_last:
                xi_obj.save(str(OUT_DIR / f'{PREFIX}_tpcf_cross_full_rand_q{q}.npy'))
            accum[f'tpcf_cross_full_rand_q{q}']['xi0'].append(xi0)
            accum[f'tpcf_cross_full_rand_q{q}']['xi2'].append(xi2)
            log.info(f'    Q{q}: xi0[7]={xi0[7]:.4f}  xi2[7]={xi2[7]:.4f}')

    sb_elapsed = time.perf_counter() - sb_start
    sb_times.append(sb_elapsed)
    elapsed_total = time.perf_counter() - pipeline_start
    remaining     = np.mean(sb_times) * (N_SUBBOXES - sb_idx - 1)
    log.info(
        f'  timing [subbox {sb_idx + 1}]: {sb_elapsed:.1f}s  '
        f'| elapsed: {elapsed_total/60:.1f} min  '
        f'| ETA: {remaining/60:.1f} min'
    )

# ── 3. Save averaged multipoles + per-subbox arrays ───────────────────────────
log.info('')
log.info('Saving averaged multipoles ...')
with timer('save results'):
    for stem in all_stems:
        xi0_arr = np.array(accum[stem]['xi0'])   # (N_SUBBOXES, n_bins)
        xi2_arr = np.array(accum[stem]['xi2'])
        np.savez(OUT_DIR / f'{PREFIX}_multipoles_{stem}.npz',
                 s=s_ref,
                 xi0=xi0_arr.mean(axis=0),
                 xi0_std=xi0_arr.std(axis=0, ddof=1),
                 xi2=xi2_arr.mean(axis=0),
                 xi2_std=xi2_arr.std(axis=0, ddof=1),
                 xi0_all=xi0_arr,
                 xi2_all=xi2_arr)
        log.info(f'  {PREFIX}_multipoles_{stem}.npz saved (N={N_SUBBOXES})')

    # run metadata: subbox grid + galaxy counts, for weighting/diagnostics
    np.savez(OUT_DIR / f'{PREFIX}_info.npz',
             cosmo=COSMO,
             hod=HOD,
             hod_file=str(HOD_FILE),
             q_par=q_par,
             q_perp=q_perp,
             seed=SEED,
             seed_geom=SEED_GEOM,
             n_data=np.array(n_data_sb),
             box_lo=box_lo,
             box_hi=box_hi,
             ixyz=np.array([(ix, iy, iz) for ix, iy, iz, _, _ in subboxes]),
             s_edges=S_EDGES,
             mu_edges=MU_EDGES)
    log.info(f'  {PREFIX}_info.npz saved')

# ── timing summary ─────────────────────────────────────────────────────────────
total_elapsed = time.perf_counter() - pipeline_start
log.info('')
log.info('=== Timing summary ===')
log.info(f'  Cosmology   : {COSMO}  hod{HOD:03d}')
log.info(f'  N_SUBBOXES  : {N_SUBBOXES}')
log.info(f'  N_Q         : {N_Q}')
log.info(f'  n_s_bins    : {len(S_EDGES) - 1}')
log.info(f'  Subbox times: min={min(sb_times)/60:.2f}  '
         f'mean={np.mean(sb_times)/60:.2f}  '
         f'max={max(sb_times)/60:.2f}  min')
log.info(f'  Total wall time: {total_elapsed/60:.2f} min  ({total_elapsed/3600:.2f} h)')
log.info(f'  Log written to : {log_file}')
log.info('')
log.info('=== Done ===')
log.info(f'Output files in {OUT_DIR}:')
for fn in sorted(OUT_DIR.glob(f'{PREFIX}_multipoles_*.npz')):
    log.info(f'  {fn.name}')

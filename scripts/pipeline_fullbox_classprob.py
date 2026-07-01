#!/usr/bin/env python3
"""
ASTRA full-box CLASS-PROBABILITY weighted-2PCF pipeline (experimental).

A single ASTRA iteration's r value is noisy for galaxies near a class boundary:
a slightly different random draw can flip a galaxy sheet<->filament. Instead of
weighting by r itself (pipeline_fullbox_weighted.py) or splitting into quantiles
of a single iteration, this estimates a per-galaxy PROBABILITY of belonging to
each of the 4 discrete ASTRA classes (void/sheet/filament/knot, fixed r-
thresholds, not quantiles) by averaging the classification over
N_PROB_ITERS (>=10) independent ASTRA random realisations:

    P_class(i) = (# iterations where galaxy i's r falls in that class)
                 / (# iterations where galaxy i has a finite r)

This is a soft, noise-reduced membership instead of one noisy hard split.
ONLY data (galaxies) are weighted -- astra_randoms are excluded from this
scheme entirely (they carry no class-probability information we use here).
geometry_randoms remain the unweighted Landy-Szalay reference, as always.

Because the classification randomness is marginalised into P_class itself, the
weighted clustering only needs to be measured ONCE (not iteration-averaged like
the other pipelines): the per-galaxy weight is already a stable, iteration-
independent number.

Statistics (mean-normalised weights w_i = P_class(i)/<P_class> over kept
galaxies; geometry randoms unweighted; ell=0,2 for each):
  - data-auto weighted by P_class, one per class                     (4)
  - full (unweighted) data x class-weighted data, one per class      (4)
    (parallels the established tpcf_cross_full_data_qN quantile statistic;
    same catalog passed as both legs with different weights -- the resulting
    self-pair spike at s=0 is the same accepted, negligible effect already
    used throughout pipeline_fullbox_cosmo.py's full x quantile crosses)
  - void-weighted x knot-weighted data                                (1)
    (mirrors the void/knot extreme-environment cross found to be the
    highest-S/N leg in the earlier vector-search work)

Per-iteration r values are cached in the SAME format/location used by
pipeline_fullbox_weighted.py (fbw_rvalues_iter{it}.npz), so the two pipelines
share iterations -- no duplicated Delaunay work. Extra iterations needed here
skip the weighted-TPCF loop entirely: only the Delaunay + fast classification
is needed to build r, so reaching N_PROB_ITERS=10 is much cheaper than running
the full weighted pipeline for 10 iterations.

Output: data/{outroot}/{cosmo}_hod{NNN}/  with prefix 'fbwp_' (+ shared 'fbw_'
r-value caches in the same directory).

Usage (full CPU node):
  sbatch queue/run_fullbox_classprob.sh c000 484 10
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
parser = argparse.ArgumentParser(description='ASTRA full-box class-probability weighted-2PCF pipeline')
parser.add_argument('cosmo', help='AbacusSummit cosmology, e.g. c000')
parser.add_argument('hod', type=int, help='HOD index, e.g. 484')
parser.add_argument('--n-prob-iters', type=int, default=10,
                    help='number of ASTRA realisations used to build P_class (default 10)')
parser.add_argument('--outroot', default='fullbox_weighted',
                    help="output subdir under data/ (default 'fullbox_weighted', shared with "
                         "pipeline_fullbox_weighted.py so r-value caches are reused)")
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
RPREFIX            = 'fbw'    # shared r-value cache prefix (pipeline_fullbox_weighted.py)
PREFIX             = 'fbwp'   # this pipeline's own output prefix
FULL_SIZE          = 2000.0
LOS                = 'z'
N_RAND             = 1        # ASTRA randoms factor
N_RAND_GEOM        = 5        # geometry randoms factor
N_PROB_ITERS       = args.n_prob_iters
MIN_FINITE         = max(3, N_PROB_ITERS // 2)   # min finite-r iterations kept per galaxy
SEED               = 42
SEED_GEOM          = SEED + 1000
NTHREADS           = 128

S_EDGES  = np.linspace(0, 150, 16)   # 15 bins, 0–150 Mpc/h
MU_EDGES = np.linspace(-1, 1, 241)

# fixed ASTRA class thresholds (CLAUDE.md): void/sheet/filament/knot
CLASSES = ['void', 'sheet', 'filament', 'knot']
CLASS_EDGES = [(-1.0, -0.9), (-0.9, 0.0), (0.0, 0.9), (0.9, 1.0)]  # (lo, hi], void lo inclusive

if not HOD_FILE.is_file():
    sys.exit(f'HOD catalog not found: {HOD_FILE}')

if N_PROB_ITERS < 10:
    sys.exit(f'N_PROB_ITERS={N_PROB_ITERS} < 10: need >=10 iterations for a stable P_class estimate')

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOG_DIR / f'pipeline_{PREFIX}_{RUN_TAG}_{stamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info(f'Log file: {log_file}')
log.info(f'Class-probability weighted-2PCF run | Cosmology: {COSMO}  HOD: hod{HOD:03d}  '
         f'N_PROB_ITERS: {N_PROB_ITERS}')
log.info(f'Output directory: {OUT_DIR}')


@contextmanager
def timer(label):
    t0 = time.perf_counter()
    yield
    log.info(f'  timing [{label}]: {time.perf_counter() - t0:.1f}s')


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
n_data = len(positions)
log.info(f'  Full box: {n_data:,} galaxies  (Q_PAR={q_par:.6f}, Q_PERP={q_perp:.6f})')

half = FULL_SIZE / 2
lo = np.array([-half / q_perp, -half / q_perp, -half / q_par])
hi = -lo

# ── 2. Geometry randoms (fixed; unweighted Landy-Szalay reference) ────────────
log.info('Generating geometry randoms ...')
with timer('geometry randoms'):
    rng_geom     = np.random.default_rng(SEED_GEOM)
    geom_randoms = rng_geom.uniform(low=lo, high=hi,
                                    size=(N_RAND_GEOM * n_data, 3))
log.info(f'  {len(geom_randoms):,} geometry randoms  (factor={N_RAND_GEOM}x, seed={SEED_GEOM})')

edges = (S_EDGES, MU_EDGES)
astra = AstraSplit()
R1R2  = None


def weighted_tpcf(pos1, w1, pos2=None, w2=None):
    """Weighted Landy-Szalay multipoles; geometry randoms unweighted, RR reused."""
    global R1R2
    kwargs = dict(
        data_positions1=pos1, data_weights1=w1,
        randoms_positions1=geom_randoms,
        engine='corrfunc', nthreads=NTHREADS,
        compute_sepsavg=True, position_type='pos', los=LOS,
    )
    if pos2 is not None:
        kwargs['data_positions2'] = pos2
        kwargs['data_weights2']   = w2
        kwargs['randoms_positions2'] = geom_randoms
    if R1R2 is not None:
        kwargs['R1R2'] = R1R2
    xi = TwoPointCorrelationFunction('smu', edges=edges, **kwargs)
    if R1R2 is None:
        R1R2 = xi.R1R2
    s, mp = xi(ells=(0, 2), return_sep=True)
    return s, mp[0], mp[1]


def norm_weights(raw, mask):
    """Mean-normalise weights (over `mask`) to <w>=1; 0 outside mask."""
    m = raw[mask].mean()
    if not np.isfinite(m) or abs(m) < 1e-12:
        raise ValueError(f'weight mean {m} too small to normalise')
    w = np.zeros_like(raw)
    w[mask] = raw[mask] / m
    return w.astype(np.float64)


# ── 3. Build / load per-iteration r values (shared cache with the weighted pipeline) ──
R = np.full((n_data, N_PROB_ITERS), np.nan, dtype=np.float64)

for it in range(N_PROB_ITERS):
    rfile = OUT_DIR / f'{RPREFIX}_rvalues_iter{it}.npz'
    if rfile.is_file():
        log.info(f'Iteration {it}: reusing cached {rfile.name}')
        a = np.load(rfile)
        R[:, it] = a['r_data'].astype(np.float64)
        continue

    iter_start = time.perf_counter()
    log.info(f'Iteration {it}: computing fresh (seed={SEED + it}) ...')
    with timer('ASTRA randoms'):
        rand_positions = astra.generate_uniform_randoms(
            positions, n_factor=N_RAND, seed=SEED + it,
        )
    n_rand = len(rand_positions)

    with timer('classify (fast)'):
        df_full  = astra.build_dataframe(positions, rand_positions)
        df_class = astra.classify_fast(df_full)
        del df_full

    nd  = df_class['NDATA'].values.astype(np.float64)
    nr  = df_class['NRAND'].values.astype(np.float64)
    tot = nd + nr
    r_all = np.where(tot > 0, (nd - nr) / tot, np.nan)
    isdata = df_class['ISDATA'].values.astype(bool)
    tid    = df_class['TARGETID'].values
    del df_class

    r_data  = np.full(n_data, np.nan, dtype=np.float64)
    r_arand = np.full(n_rand, np.nan, dtype=np.float64)
    r_data[tid[isdata]]            = r_all[isdata]
    r_arand[tid[~isdata] - n_data] = r_all[~isdata]
    log.info(f'  finite-r: {np.isfinite(r_data).sum():,}/{n_data:,} data  '
             f'{np.isfinite(r_arand).sum():,}/{n_rand:,} astra_randoms')

    np.savez(rfile,
             r_data=r_data.astype(np.float32),
             r_arand=r_arand.astype(np.float32),
             seed=SEED + it, n_data=n_data, n_rand=n_rand)
    R[:, it] = r_data
    log.info(f'  timing [iteration {it}]: {(time.perf_counter() - iter_start)/60:.2f} min')

# ── 4. Build per-galaxy class probabilities ───────────────────────────────────
log.info('')
log.info('Building per-galaxy class probabilities ...')
class_idx = np.full(R.shape, -1, dtype=np.int8)
for c, (clo, chi) in enumerate(CLASS_EDGES):
    if c == 0:
        m = (R >= clo) & (R <= chi)
    else:
        m = (R > clo) & (R <= chi)
    class_idx[m] = c

denom = (class_idx >= 0).sum(axis=1)
keep  = denom >= MIN_FINITE
log.info(f'  kept {keep.sum():,}/{n_data:,} galaxies with >= {MIN_FINITE}/{N_PROB_ITERS} finite-r iterations')

P = np.zeros((n_data, 4), dtype=np.float64)
for c in range(4):
    P[:, c] = np.where(denom > 0, (class_idx == c).sum(axis=1) / np.maximum(denom, 1), 0.0)

# sanity: probabilities sum to 1 for kept galaxies
psum = P[keep].sum(axis=1)
log.info(f'  sum_c P_class check (kept galaxies): mean={psum.mean():.6f}  '
         f'min={psum.min():.6f}  max={psum.max():.6f}')

# aggregate mean P per class vs single-iteration (iter 0) class fractions
log.info('  aggregate <P_class> (kept)  vs  iteration-0 single-draw class fraction:')
r0 = R[:, 0]
m0 = np.isfinite(r0)
for c, name in enumerate(CLASSES):
    clo, chi = CLASS_EDGES[c]
    frac0 = (((r0 >= clo) if c == 0 else (r0 > clo)) & (r0 <= chi) & m0).sum() / m0.sum()
    log.info(f'    {name:9s}: <P>={P[keep, c].mean():.4f}   iter0 frac={frac0:.4f}')

# boundary-vs-stable diagnostic: distribution of max_c P_class per galaxy
maxp = P[keep].max(axis=1)
qs = np.percentile(maxp, [5, 25, 50, 75, 95])
log.info(f'  max_c P_class percentiles (5/25/50/75/95): '
         + '  '.join(f'{q:.3f}' for q in qs))
log.info(f'  fraction with max_c P_class >= 0.9 (stable classification): '
         f'{(maxp >= 0.9).mean():.3f}')
log.info(f'  fraction with max_c P_class < 0.5 (no majority class, all 4 possible): '
         f'{(maxp < 0.5).mean():.3f}')

np.savez(OUT_DIR / f'{PREFIX}_pclass.npz',
         P=P.astype(np.float32), keep=keep, denom=denom,
         classes=np.array(CLASSES), n_prob_iters=N_PROB_ITERS, min_finite=MIN_FINITE)
log.info(f'  saved {PREFIX}_pclass.npz')

# ── 5. Mean-normalised class weights (data only) ──────────────────────────────
weights = {c: norm_weights(P[:, i], keep) for i, c in enumerate(CLASSES)}
w_full  = norm_weights(np.ones(n_data), keep)   # trivially = keep mask (mean of ones = 1)

pos_kept = positions[keep]


def w_kept(name):
    return weights[name][keep]


# ── 6. Weighted 2PCF statistics ────────────────────────────────────────────────
log.info('')
log.info('Computing weighted 2PCF ...')
results = {}

for c in CLASSES:
    with timer(f'{c}_auto'):
        s, x0, x2 = weighted_tpcf(pos_kept, w_kept(c))
    results[f'{c}_auto'] = (s, x0, x2)
    log.info(f'  {c}_auto            xi0[7]={x0[7]:+.4f}')

for c in CLASSES:
    with timer(f'full_x_{c}'):
        s, x0, x2 = weighted_tpcf(pos_kept, w_full[keep],
                                   pos2=pos_kept, w2=w_kept(c))
    results[f'full_x_{c}'] = (s, x0, x2)
    log.info(f'  full_x_{c}         xi0[7]={x0[7]:+.4f}')

with timer('void_x_knot'):
    s, x0, x2 = weighted_tpcf(pos_kept, w_kept('void'), pos2=pos_kept, w2=w_kept('knot'))
results['void_x_knot'] = (s, x0, x2)
log.info(f'  void_x_knot        xi0[7]={x0[7]:+.4f}')

# ── 7. Save ────────────────────────────────────────────────────────────────────
log.info('')
log.info('Saving multipoles (single measurement -- classification noise already '
         'marginalised into P_class, so no per-iteration std is stored) ...')
for stem, (s, x0, x2) in results.items():
    np.savez(OUT_DIR / f'{PREFIX}_multipoles_{stem}.npz', s=s, xi0=x0, xi2=x2)

np.savez(OUT_DIR / f'{PREFIX}_info.npz',
         cosmo=COSMO, hod=HOD, hod_file=str(HOD_FILE),
         q_par=q_par, q_perp=q_perp, seed=SEED, seed_geom=SEED_GEOM,
         n_prob_iters=N_PROB_ITERS, min_finite=MIN_FINITE,
         n_data=n_data, n_kept=int(keep.sum()),
         classes=np.array(CLASSES), stems=np.array(list(results)),
         box_lo=lo, box_hi=hi, s_edges=S_EDGES, mu_edges=MU_EDGES)

log.info(f'  saved {len(results)} multipole files + {PREFIX}_info.npz + {PREFIX}_pclass.npz')
log.info(f'Total wall time: {(time.perf_counter() - pipeline_start)/60:.2f} min')
log.info('=== Done ===')

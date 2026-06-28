#!/usr/bin/env python3
"""
ASTRA full-box WEIGHTED-2PCF pipeline (experimental).

Instead of splitting galaxies into ASTRA quantiles of the environment parameter
r = (n_data - n_rand)/(n_data + n_rand) and correlating each quantile, this uses
r itself as a per-object WEIGHT and measures weighted correlation functions.

Three populations are kept strictly separate (as in the standard pipeline):
  - data            : galaxies (carry RSD via Z_RSD), each with an r value
  - astra_randoms   : the 1x uniform points that enter the Delaunay, each with r
  - geometry randoms: 5x uniform, the unweighted Landy-Szalay reference (window)

Per ASTRA iteration we compute, for every weight scheme f(r), the weighted ell=0,2
multipoles of three statistics, with geometry randoms as the (unweighted) reference:
  - data-auto                    : data weighted by f(r_data)
  - astra_random-auto            : astra_randoms weighted by f(r_arand)
  - data x astra_random cross    : data*f(r_data) x astra_randoms*f(r_arand) (same iteration)

Estimator (unified for all schemes): weighted Landy-Szalay with weights
MEAN-NORMALISED per population per iteration, w_i = f(r_i) / <f(r)>.  With this
normalisation the 'uniform' scheme (f=1) reproduces the standard xi exactly (a
built-in unit test), and for the signed-r scheme the weighted xi IS the marked
correlation numerator W; the marked-correlation monopole M0 = (1+W0)/(1+xi0) is
formed in the plot script.

Weight schemes:  uniform, knot (1+r)/2, void (1-r)/2, rank (CDF), power [(1+r)/2]^2,
                 exp exp(2r), signed r.

r values for data and astra_randoms are stored per iteration so the weighting can
be recomputed/extended later without re-running the (expensive) Delaunay.

Output: data/{outroot}/{cosmo}_hod{NNN}/  with prefix 'fbw_'.

Usage (full CPU node):
  sbatch queue/run_fullbox_weighted.sh c000 484 3 fullbox_weighted
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
from scipy.stats import rankdata
from pycorr import TwoPointCorrelationFunction

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from astra import AstraSplit

# ── command line ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='ASTRA full-box weighted-2PCF pipeline')
parser.add_argument('cosmo', help='AbacusSummit cosmology, e.g. c000')
parser.add_argument('hod', type=int, help='HOD index, e.g. 484')
parser.add_argument('--iterations', type=int, default=3,
                    help='number of ASTRA random realisations (default 3)')
parser.add_argument('--outroot', default='fullbox_weighted',
                    help="output subdir under data/ (default 'fullbox_weighted')")
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
PREFIX             = 'fbw'
FULL_SIZE          = 2000.0
LOS                = 'z'
N_RAND             = 1        # ASTRA randoms factor
N_RAND_GEOM        = 5        # geometry randoms factor
N_ASTRA_ITERATIONS = args.iterations
SEED               = 42
SEED_GEOM          = SEED + 1000
NTHREADS           = 128

S_EDGES  = np.linspace(0, 150, 16)   # 15 bins, 0–150 Mpc/h
MU_EDGES = np.linspace(-1, 1, 241)

# ── weight schemes f(r); mean-normalisation is applied separately ─────────────
WEIGHT_SCHEMES = {
    'uniform': lambda r: np.ones_like(r),
    'knot':    lambda r: (1.0 + r) / 2.0,
    'void':    lambda r: (1.0 - r) / 2.0,
    'rank':    lambda r: rankdata(r) / len(r),
    'power':   lambda r: ((1.0 + r) / 2.0) ** 2,
    'exp':     lambda r: np.exp(2.0 * r),
    'signed':  lambda r: r.copy(),
}
STATS = ('data', 'arand', 'cross')

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
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info(f'Log file: {log_file}')
log.info(f'Weighted-2PCF run | Cosmology: {COSMO}  HOD: hod{HOD:03d}  iterations: {N_ASTRA_ITERATIONS}')
log.info(f'Weight schemes: {", ".join(WEIGHT_SCHEMES)}')
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

# ── 2. Geometry randoms (fixed across iterations; unweighted LS reference) ─────
log.info('Generating geometry randoms ...')
with timer('geometry randoms'):
    rng_geom     = np.random.default_rng(SEED_GEOM)
    geom_randoms = rng_geom.uniform(low=lo, high=hi,
                                    size=(N_RAND_GEOM * n_data, 3))
log.info(f'  {len(geom_randoms):,} geometry randoms  (factor={N_RAND_GEOM}x, seed={SEED_GEOM})')

# ── weighted 2PCF helper: geometry-random RR computed once, reused ────────────
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


def norm_weights(raw):
    """Mean-normalise weights to <w>=1 so 'uniform' reproduces standard xi."""
    m = np.mean(raw)
    if not np.isfinite(m) or abs(m) < 1e-12:
        raise ValueError(f'weight mean {m} too small to normalise')
    return (raw / m).astype(np.float64)


# ── 3. Loop over ASTRA realisations ───────────────────────────────────────────
stems = [f'{sch}_{st}' for sch in WEIGHT_SCHEMES for st in STATS]
accum = {stem: {'xi0': [], 'xi2': []} for stem in stems}
mean_r = []   # (iteration) -> dict of population mean r
s_ref = None

for it in range(N_ASTRA_ITERATIONS):
    iter_start = time.perf_counter()
    log.info('')
    log.info(f'=== ASTRA iteration {it + 1}/{N_ASTRA_ITERATIONS}  (seed={SEED + it}) ===')

    with timer('ASTRA randoms'):
        rand_positions = astra.generate_uniform_randoms(
            positions, n_factor=N_RAND, seed=SEED + it,
        )
    n_rand = len(rand_positions)
    log.info(f'  {n_rand:,} ASTRA randoms')

    with timer('classify (fast)'):
        df_full  = astra.build_dataframe(positions, rand_positions)
        df_class = astra.classify_fast(df_full)
        del df_full

    # r = (NDATA - NRAND)/(NDATA + NRAND) per object; align to position arrays
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

    # drop the (rare) objects with no Delaunay neighbours so every scheme uses
    # the same sample; uniform on this sample still matches standard xi to ~1e-6
    md = np.isfinite(r_data)
    ma = np.isfinite(r_arand)
    pos_d, rd = positions[md],      r_data[md]
    pos_a, ra = rand_positions[ma], r_arand[ma]
    log.info(f'  finite-r: {md.sum():,}/{n_data:,} data  {ma.sum():,}/{n_rand:,} astra_randoms')
    log.info(f'  <r> data={np.mean(rd):+.4f}  astra_randoms={np.mean(ra):+.4f}')
    mean_r.append({'data': float(np.mean(rd)), 'arand': float(np.mean(ra))})

    # store r for this iteration (float32; positions are reproducible from seed)
    np.savez(OUT_DIR / f'{PREFIX}_rvalues_iter{it}.npz',
             r_data=r_data.astype(np.float32),
             r_arand=r_arand.astype(np.float32),
             seed=SEED + it, n_data=n_data, n_rand=n_rand)

    # per scheme: build mean-normalised weights and compute the three statistics
    for sch, f in WEIGHT_SCHEMES.items():
        wd = norm_weights(f(rd))
        wa = norm_weights(f(ra))
        with timer(f'{sch}'):
            s_ref, x0, x2 = weighted_tpcf(pos_d, wd)
            accum[f'{sch}_data']['xi0'].append(x0); accum[f'{sch}_data']['xi2'].append(x2)

            _, x0, x2 = weighted_tpcf(pos_a, wa)
            accum[f'{sch}_arand']['xi0'].append(x0); accum[f'{sch}_arand']['xi2'].append(x2)

            _, x0, x2 = weighted_tpcf(pos_d, wd, pos2=pos_a, w2=wa)
            accum[f'{sch}_cross']['xi0'].append(x0); accum[f'{sch}_cross']['xi2'].append(x2)
        log.info(f'    {sch:8s}: data xi0[7]={accum[f"{sch}_data"]["xi0"][-1][7]:+.4f}  '
                 f'arand xi0[7]={accum[f"{sch}_arand"]["xi0"][-1][7]:+.4f}  '
                 f'cross xi0[7]={accum[f"{sch}_cross"]["xi0"][-1][7]:+.4f}')

    log.info(f'  timing [iteration {it + 1}]: {(time.perf_counter() - iter_start)/60:.2f} min')

# ── 4. Save multipoles + run metadata ─────────────────────────────────────────
log.info('')
log.info('Saving multipoles ...')
ddof = 1 if N_ASTRA_ITERATIONS > 1 else 0
for stem in stems:
    xi0 = np.array(accum[stem]['xi0'])
    xi2 = np.array(accum[stem]['xi2'])
    np.savez(OUT_DIR / f'{PREFIX}_multipoles_{stem}.npz',
             s=s_ref,
             xi0=xi0.mean(0), xi0_std=xi0.std(0, ddof=ddof),
             xi2=xi2.mean(0), xi2_std=xi2.std(0, ddof=ddof),
             xi0_all=xi0, xi2_all=xi2)

np.savez(OUT_DIR / f'{PREFIX}_info.npz',
         cosmo=COSMO, hod=HOD, hod_file=str(HOD_FILE),
         q_par=q_par, q_perp=q_perp, seed=SEED, seed_geom=SEED_GEOM,
         n_iterations=N_ASTRA_ITERATIONS, n_data=n_data,
         schemes=list(WEIGHT_SCHEMES), stats=list(STATS),
         mean_r_data=np.array([m['data'] for m in mean_r]),
         mean_r_arand=np.array([m['arand'] for m in mean_r]),
         box_lo=lo, box_hi=hi, s_edges=S_EDGES, mu_edges=MU_EDGES)

log.info(f'  saved {len(stems)} multipole files + {PREFIX}_info.npz + r-value caches')
log.info(f'Total wall time: {(time.perf_counter() - pipeline_start)/60:.2f} min')
log.info('=== Done ===')

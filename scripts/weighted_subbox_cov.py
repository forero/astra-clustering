#!/usr/bin/env python3
"""
Cosmic-variance covariance for the WEIGHTED-2PCF vectors, by reanalysis of the
fiducial full-box weighted run — NO new simulation.

The full-box weighted pipeline caches, per ASTRA iteration, the environment
parameter r of every data galaxy and every ASTRA random (fbw_rvalues_iter*.npz),
together with the seed that generated the ASTRA randoms.  The galaxy positions
are reproducible from the HOD FITS file and the ASTRA randoms from their seed,
so the whole (positions, r) field of the fiducial box can be reconstructed here
without re-running the (expensive) Delaunay.

We then tile the periodic box into NDIV^3 = 64 subboxes (side FULL/NDIV = 500
Mpc/h, matching the quantile subbox covariance) and, in each subbox, measure the
weighted Landy-Szalay multipoles of the three statistics (data-auto, arand-auto,
data x arand cross) for every weight scheme, with the weights MEAN-NORMALISED
WITHIN THE SUBBOX (so 'uniform' reproduces the standard subbox xi, exactly as in
the full-box estimator).  Each subbox uses its own uniform geometry randoms
(open boundary), with RR computed once per subbox and reused across schemes and
statistics.

The 64 subboxes x N_ITER iterations give the covariance samples.  The full-box
(2000 Mpc/h) covariance is the subbox covariance scaled by the volume ratio
V_sub / V_full = 1 / NDIV^3 = 1/64 (the same scaling the quantile Fisher uses,
C_subbox / 64).

CAVEATS
  * The r field is the GLOBAL Delaunay classification restricted to each subbox,
    not an independent per-subbox classification.  This captures the dominant
    cosmic variance of the (weighted) density field but omits the extra scatter a
    truly independent subbox would get from re-estimating r locally (ASTRA-random
    noise + local edge effects).  It therefore slightly UNDER-states the weighted
    covariance; treat the resulting sigmas as mildly optimistic.  The faithful
    upgrade is a weighted analogue of pipeline_subboxes_cosmo.py (local Delaunay
    per subbox) — more compute, deferred.
  * The N_ITER iterations of one box are correlated samples, not independent
    realisations; pooling them (64*N_ITER) buys covariance DOF (so the per-scheme
    90-bin vector is invertible) but the effective independent count is nearer 64.

Output: data/fullbox_weighted/cov/weighted_subbox_cov.npz
  s, ndiv, n_sub, n_iter, vol_scale, schemes, stats, ells,
  and per stem '{scheme}_{stat}':
     {stem}_xi0_sub  (n_sub*n_iter, nbins)   subbox monopole samples
     {stem}_xi2_sub  (n_sub*n_iter, nbins)   subbox quadrupole samples
  plus a stacked covariance for the per-scheme data vector
  [data,arand,cross] x [ell0,ell2] (90 bins): '{scheme}_cov' and '{scheme}_mean'.

Run on a compute node (uses corrfunc, 128 threads), e.g.
  salloc -N 1 -C cpu -q interactive -t 60:00 -A desi -c 128 --mem=0
  srun -n 1 -c 128 python scripts/weighted_subbox_cov.py [cosmo hod]
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import fitsio
from scipy.stats import rankdata
from pycorr import TwoPointCorrelationFunction

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from astra import AstraSplit

# ── command line ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Weighted-2PCF subbox covariance (reanalysis)')
parser.add_argument('cosmo', nargs='?', default='c000')
parser.add_argument('hod', nargs='?', type=int, default=484)
parser.add_argument('--ndiv', type=int, default=4, help='subboxes per axis (default 4 -> 64)')
parser.add_argument('--outroot', default='fullbox_weighted')
args = parser.parse_args()

COSMO, HOD, NDIV = args.cosmo, args.hod, args.ndiv

RUN_TAG  = f'{COSMO}_hod{HOD:03d}'
RUN_DIR  = REPO_ROOT / 'data' / args.outroot / RUN_TAG
COV_DIR  = REPO_ROOT / 'data' / args.outroot / 'cov'
HOD_FILE = Path('/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
                f'/{COSMO}_ph000/seed0/hod{HOD:03d}.fits')

FULL_SIZE   = 2000.0
LOS         = 'z'
N_RAND      = 1
N_RAND_GEOM = 5
NTHREADS    = 128
S_EDGES     = np.linspace(0, 150, 16)
MU_EDGES    = np.linspace(-1, 1, 241)
EDGES       = (S_EDGES, MU_EDGES)

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

info = np.load(RUN_DIR / 'fbw_info.npz', allow_pickle=True)
SEED      = int(info['seed'])
SEED_GEOM = int(info['seed_geom'])
N_ITER    = int(info['n_iterations'])
q_par     = float(info['q_par'])
q_perp    = float(info['q_perp'])
box_lo    = np.asarray(info['box_lo'], float)
box_hi    = np.asarray(info['box_hi'], float)


def norm_weights(raw):
    m = np.mean(raw)
    if not np.isfinite(m) or abs(m) < 1e-12:
        raise ValueError(f'weight mean {m} too small to normalise')
    return (raw / m).astype(np.float64)


def load_positions():
    """Reproduce the data positions exactly as the pipeline did (RSD + AP)."""
    data, _ = fitsio.read(str(HOD_FILE), header=True)
    pos = np.c_[data['X_PERP'] / q_perp,
                data['Y_PERP'] / q_perp,
                data['Z_RSD']  / q_par].astype(np.float64)
    return pos


def main():
    if not (RUN_DIR / 'fbw_info.npz').is_file():
        sys.exit(f'No weighted run at {RUN_DIR}')
    COV_DIR.mkdir(parents=True, exist_ok=True)
    astra = AstraSplit()
    pos_data_full = load_positions()
    n_data = len(pos_data_full)
    print(f'Loaded {n_data:,} data galaxies for {RUN_TAG}; box {box_lo} .. {box_hi}')

    # subbox edges per axis
    edges_ax = [np.linspace(box_lo[a], box_hi[a], NDIV + 1) for a in range(3)]
    n_sub = NDIV ** 3
    stems = [f'{sch}_{st}' for sch in WEIGHT_SCHEMES for st in STATS]
    samples = {f'{stem}_xi{ell}': [] for stem in stems for ell in (0, 2)}
    s_ref = None
    t0 = time.perf_counter()

    for it in range(N_ITER):
        seed_it = SEED + it
        rv = np.load(RUN_DIR / f'fbw_rvalues_iter{it}.npz')
        r_data  = rv['r_data'].astype(np.float64)     # (n_data,) incl NaN
        r_arand = rv['r_arand'].astype(np.float64)    # (n_rand,) incl NaN
        assert int(rv['seed']) == seed_it, 'seed mismatch vs cached r'
        # reproduce ASTRA randoms from the same seed used by the pipeline
        pos_arand_full = astra.generate_uniform_randoms(
            pos_data_full, n_factor=N_RAND, seed=seed_it)
        assert len(pos_arand_full) == len(r_arand), 'arand count mismatch'

        # keep only finite-r objects (same sample as the full-box run)
        md = np.isfinite(r_data)
        ma = np.isfinite(r_arand)
        pd_, rd = pos_data_full[md],  r_data[md]
        pa_, ra = pos_arand_full[ma], r_arand[ma]

        rng_geom = np.random.default_rng(SEED_GEOM + it)  # per-iteration geom seed

        # bin every object into its subbox index
        def sub_index(pos):
            ix = np.clip(np.digitize(pos[:, 0], edges_ax[0]) - 1, 0, NDIV - 1)
            iy = np.clip(np.digitize(pos[:, 1], edges_ax[1]) - 1, 0, NDIV - 1)
            iz = np.clip(np.digitize(pos[:, 2], edges_ax[2]) - 1, 0, NDIV - 1)
            return (ix * NDIV + iy) * NDIV + iz
        sid_d = sub_index(pd_)
        sid_a = sub_index(pa_)

        for sb in range(n_sub):
            iz =  sb % NDIV
            iy = (sb // NDIV) % NDIV
            ix =  sb // (NDIV * NDIV)
            lo = np.array([edges_ax[0][ix], edges_ax[1][iy], edges_ax[2][iz]])
            hi = np.array([edges_ax[0][ix + 1], edges_ax[1][iy + 1], edges_ax[2][iz + 1]])

            md_s = sid_d == sb
            ma_s = sid_a == sb
            sub_pd, sub_rd = pd_[md_s], rd[md_s]
            sub_pa, sub_ra = pa_[ma_s], ra[ma_s]
            n_geom = N_RAND_GEOM * len(sub_pd)
            geom = rng_geom.uniform(low=lo, high=hi, size=(max(n_geom, 1), 3))

            R1R2 = [None]

            def wtpcf(p1, w1, p2=None, w2=None):
                kw = dict(data_positions1=p1, data_weights1=w1,
                          randoms_positions1=geom,
                          engine='corrfunc', nthreads=NTHREADS,
                          compute_sepsavg=True, position_type='pos', los=LOS)
                if p2 is not None:
                    kw['data_positions2'] = p2
                    kw['data_weights2'] = w2
                    kw['randoms_positions2'] = geom
                if R1R2[0] is not None:
                    kw['R1R2'] = R1R2[0]
                xi = TwoPointCorrelationFunction('smu', edges=EDGES, **kw)
                if R1R2[0] is None:
                    R1R2[0] = xi.R1R2
                s, mp = xi(ells=(0, 2), return_sep=True)
                return s, mp[0], mp[1]

            for sch, f in WEIGHT_SCHEMES.items():
                wd = norm_weights(f(sub_rd))
                wa = norm_weights(f(sub_ra))
                s, x0, x2 = wtpcf(sub_pd, wd)
                samples[f'{sch}_data_xi0'].append(x0); samples[f'{sch}_data_xi2'].append(x2)
                _, x0, x2 = wtpcf(sub_pa, wa)
                samples[f'{sch}_arand_xi0'].append(x0); samples[f'{sch}_arand_xi2'].append(x2)
                _, x0, x2 = wtpcf(sub_pd, wd, p2=sub_pa, w2=wa)
                samples[f'{sch}_cross_xi0'].append(x0); samples[f'{sch}_cross_xi2'].append(x2)
                if s_ref is None:
                    s_ref = s
        print(f'  iteration {it + 1}/{N_ITER} done  ({(time.perf_counter() - t0)/60:.1f} min)')

    # stack into (n_sub*n_iter, nbins) and form per-scheme covariances
    nbin = len(s_ref)
    vol_scale = 1.0 / (NDIV ** 3)
    out = dict(s=s_ref, ndiv=NDIV, n_sub=n_sub, n_iter=N_ITER,
               vol_scale=vol_scale, schemes=list(WEIGHT_SCHEMES),
               stats=list(STATS), ells=np.array([0, 2]))
    for stem in stems:
        out[f'{stem}_xi0_sub'] = np.array(samples[f'{stem}_xi0'])
        out[f'{stem}_xi2_sub'] = np.array(samples[f'{stem}_xi2'])
    for sch in WEIGHT_SCHEMES:
        blocks = []
        for st in STATS:
            blocks.append(np.array(samples[f'{sch}_{st}_xi0']))
            blocks.append(np.array(samples[f'{sch}_{st}_xi2']))
        V = np.hstack(blocks)                        # (n_sub*n_iter, 6*nbin)
        out[f'{sch}_mean'] = V.mean(0)
        out[f'{sch}_cov']  = np.cov(V, rowvar=False) * vol_scale   # full-box volume
    np.savez(COV_DIR / 'weighted_subbox_cov.npz', **out)
    print(f'Saved {COV_DIR / "weighted_subbox_cov.npz"}  '
          f'({n_sub * N_ITER} samples, {6 * nbin}-bin per-scheme vector)')


if __name__ == '__main__':
    main()

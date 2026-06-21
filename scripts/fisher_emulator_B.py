#!/usr/bin/env python3
"""
Approach B: emulator-based, HOD-matched cosmology derivatives -> joint Fisher.

The catalogs were never HOD-matched across cosmologies, which is what forces the
global-response model to *absorb* the HOD mismatch with a (linear) HOD term.  Here
we instead remove the mismatch directly: train a NONLINEAR HOD emulator separately
in each of the 9 cosmologies (50 maximin draws each), evaluate every cosmology's
emulator at a SINGLE common fiducial HOD point theta*, and central-difference the
+/- pairs.  Because both legs of each difference are predicted at the identical
theta*, the cosmology derivative is HOD-matched by construction and clean to
nonlinear order in the HOD.  The cosmology-independence test
(test_hod_response_cosmo_independence.py) is the validation that evaluating each
emulator at theta* is interpolation, not extrapolation.

Cosmology axis stays a derivative star (9 points) -> only first derivatives at the
fiducial, which is exactly (and only) what Fisher needs.  This is NOT an MCMC-ready
cosmology emulator; that needs the space-filling c130-c181 grid.

Pipeline per cosmology: standardise theta_HOD (pooled prior) -> standardise the
510-D data vector -> PCA -> GP (Matern-5/2 ARD + white) per component -> predict at
theta*.  Then:
  * cosmology derivatives  d xi/d theta_p = [xi_c+(theta*) - xi_c-(theta*)] / d theta_p
  * HOD gradient at theta*  d xi/d theta_HOD,q  (finite-diff the c000 emulator)
written as derivative_emuB_{p}.npz / hod_gradient_emuB.npz (drop-in for the global
files), then fisher_joint's machinery is reused with those derivatives.

Outputs (all under plots/emulator_fisher_B/):
  derivative_emuB_vs_global.png         sanity: emuB vs linear-global derivatives
  fisher_joint_*.png                    the Fisher figures, recomputed with emuB
Also writes data/derivatives/derivative_emuB_{p}.npz + hod_gradient_emuB.npz.

Usage (any node):  python scripts/fisher_emulator_B.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
FB   = DATA / 'fullbox'
ENS  = DATA / 'hod_ensemble'
DER  = DATA / 'derivatives'
PLOT = REPO / 'plots' / 'emulator_fisher_B'

FID_COSMO = 'c000'
FID_HOD   = 484                       # the Fisher fiducial HOD -> theta*
N_Q = 4
STEMS = (['tpcf_full_data'] +
         [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
         [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
         [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
         [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)])

# central-difference pairs: param -> (plus cosmology, minus cosmology)
PAIRS = {'lnwb': ('c100', 'c101'), 'lnwc': ('c102', 'c103'),
         'ns':   ('c104', 'c105'), 'lns8': ('c112', 'c113')}
COSMOS = [FID_COSMO] + [c for pr in PAIRS.values() for c in pr]
# cosmology regressors from the abacus table (matches compute_response_global)
COSMO_COL = [('lnwb', 'omega_b', True), ('lnwc', 'omega_cdm', True),
             ('ns', 'n_s', False), ('lns8', 'sigma8_m', True)]
PIDX = {'lnwb': 0, 'lnwc': 1, 'ns': 2, 'lns8': 3}


def cosmo_regressors():
    df = pd.read_csv(DATA / 'abacus_cosmologies_params.csv', index_col=0)
    return {tag: np.array([np.log(row[c]) if lg else row[c] for _, c, lg in COSMO_COL])
            for tag, row in df.iterrows()}


def hod_table(cosmo):
    df = pd.read_csv(ENS / f'hod_params_{cosmo}.csv').set_index('hod')
    return list(df.columns), df


def load_cosmo(cosmo, hod_df):
    """X (n,12) HOD params, Y (n,nout) concatenated data vector, blocks, s."""
    X, Y, s, blocks = [], [], None, None
    for d in sorted(FB.glob(f'{cosmo}_hod*')):
        if not (d / 'fullbox_info.npz').is_file():
            continue
        hod = int(d.name.split('_hod')[1])
        if hod not in hod_df.index:
            continue
        vec, blk, i = [], [], 0
        for stem in STEMS:
            a = np.load(d / f'fullbox_multipoles_{stem}.npz')
            if s is None:
                s = a['s']
            for ell in (0, 2):
                vec.append(a[f'xi{ell}'])
                blk.append((stem, ell, slice(i, i + len(s)))); i += len(s)
        X.append(hod_df.loc[hod].values.astype(float)); Y.append(np.concatenate(vec))
        blocks = blk
    return np.array(X), np.array(Y), s, blocks


def gp():
    k = (ConstantKernel(1.0) * Matern(length_scale=np.ones(12), nu=2.5)
         + WhiteKernel(1e-2))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=2, alpha=1e-6)


def train_emulator(Xz, Y):
    """Return predict(Zq)->data vector, fit on standardised HOD inputs Xz."""
    ymu, ysd = Y.mean(0), Y.std(0)
    Yz = (Y - ymu) / ysd
    ncomp = min(len(Xz) - 1, Yz.shape[1])
    pca = PCA(n_components=ncomp).fit(Yz)
    keep = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.999) + 1)
    Tp = pca.transform(Yz)[:, :keep]
    models = [gp().fit(Xz, Tp[:, k]) for k in range(keep)]

    def predict(Zq):                                   # Zq (m,12) standardised
        comp = np.zeros((len(Zq), ncomp))
        for k, m in enumerate(models):
            comp[:, k] = m.predict(Zq)
        return pca.inverse_transform(comp) * ysd + ymu
    return predict


def main():
    PLOT.mkdir(parents=True, exist_ok=True)
    creg = cosmo_regressors()
    names, fid_df = hod_table(FID_COSMO)
    theta_star = fid_df.loc[FID_HOD].values.astype(float)         # (12,)

    # pooled standardisation across all 9x50 draws
    allp = np.vstack([hod_table(c)[1].loc[
        [int(d.name.split('_hod')[1]) for d in sorted(FB.glob(f'{c}_hod*'))
         if (d / 'fullbox_info.npz').is_file()]].values for c in COSMOS])
    mu, sd = allp.mean(0), allp.std(0)
    zstar = ((theta_star - mu) / sd)[None, :]

    # train each cosmology's emulator, predict the data vector at theta*
    xi_at_star, predictors, s, blocks = {}, {}, None, None
    print('Training per-cosmology HOD emulators (50 draws each), predicting at theta*:')
    for c in COSMOS:
        _, df = hod_table(c)
        X, Y, s, blocks = load_cosmo(c, df)
        Xz = (X - mu) / sd
        dist = np.linalg.norm(zstar - Xz.mean(0))                 # theta* vs cloud centre
        pred = train_emulator(Xz, Y)
        predictors[c] = pred
        xi_at_star[c] = pred(zstar)[0]
        print(f'  {c}: {len(X)} draws, theta* at {dist:.1f} std from cloud centre '
              f'(cloud spans ~{np.median(np.linalg.norm(Xz - Xz.mean(0), axis=1)):.1f})')

    # ---- cosmology derivatives: matched-HOD central differences ----
    nbins = len(s)
    der = {p: {} for p in PAIRS}
    for p, (cp, cm) in PAIRS.items():
        dtheta = creg[cp][PIDX[p]] - creg[cm][PIDX[p]]
        dxi = (xi_at_star[cp] - xi_at_star[cm]) / dtheta           # (nout,)
        for stem, ell, sl in blocks:
            der[p][f'{stem}_dxi{ell}'] = dxi[sl]
        np.savez(DER / f'derivative_emuB_{p}.npz', s=s, **der[p])
    print(f'Saved derivative_emuB_{{{",".join(PAIRS)}}}.npz')

    # ---- HOD gradient at theta* from the c000 emulator (finite diff) ----
    eps = 0.05                                                     # std-space step
    grad = {}
    for q in range(12):
        zp = zstar.copy(); zp[0, q] += eps
        zm = zstar.copy(); zm[0, q] -= eps
        dY = (predictors[FID_COSMO](zp)[0] - predictors[FID_COSMO](zm)[0]) / (2 * eps * sd[q])
        for stem, ell, sl in blocks:
            grad.setdefault(f'{stem}_g{ell}', np.zeros((12, nbins)))[q] = dY[sl]
    np.savez(DER / 'hod_gradient_emuB.npz', s=s, names=np.array(names),
             param_mean=mu, param_std_prior=sd, n_runs=len(COSMOS) * 50, **grad)
    print('Saved hod_gradient_emuB.npz')

    # ---- sanity: full-auto dxi/dlnsigma8 vs 2 xi at theta* ----
    xi_fid = xi_at_star[FID_COSMO][blocks[0][2]]                   # full_data ell0
    dlns8  = der['lns8']['tpcf_full_data_dxi0']
    small  = s < 40
    ratio  = np.mean(dlns8[small]) / np.mean(2 * xi_fid[small])
    print(f'\nSanity dxi/dlnsigma8 vs 2xi (emuB, full auto l0, s<40): ratio = {ratio:.2f} '
          f'(global-linear gave 0.43; expect ~1)')

    # ---- comparison figure: emuB vs linear-global derivative (full auto) ----
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, p in zip(axes, PAIRS):
        g = np.load(DER / f'derivative_global_{p}.npz')['tpcf_full_data_dxi0']
        b = der[p]['tpcf_full_data_dxi0']
        ax.plot(s, s**2 * g, 'k--', lw=1.6, label='global (linear HOD)')
        ax.plot(s, s**2 * b, 'C3',  lw=2.0, label='emulator-B (matched HOD)')
        ax.axhline(0, color='grey', lw=0.5); ax.set_title(p, fontsize=10)
        ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    axes[0].set_ylabel(r'$s^2\,\partial\xi_0/\partial\theta$'); axes[0].legend(fontsize=8)
    fig.suptitle('Cosmology derivatives at theta* (full auto): '
                 'emulator-B matched-HOD vs linear-global', y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT / 'derivative_emuB_vs_global.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {PLOT / "derivative_emuB_vs_global.png"}')

    # ---- reuse fisher_joint with the emuB derivatives, into the new folder ----
    import fisher_joint as fj
    fj.PLOT_DIR = PLOT
    fj.deriv_source = lambda: ('emulator-B (matched-HOD)',
                               'derivative_emuB_{p}.npz', 'hod_gradient_emuB.npz')
    print('\n===== joint Fisher with emulator-B derivatives =====')
    fj.main()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Step-1 emulator bake-off: can a better model/representation break the ~8x-CV
within-cosmology floor?  Pits the current MLP against a PCA+GP that combines the
three top suspects at once:
  (1) model class : GP (right for <= few hundred points) vs the data-hungry MLP
  (2) loss/metric : PCA done in the COVARIANCE-WHITENED metric (compresses in the
                    chi^2 directions that matter for inference), vs per-bin MSE
  (3) target      : RATIO  r = xi / xi_ref  (xi_ref = training HOD-mean), removing
                    the dominant amplitude/shape so the regressor learns the
                    residual HOD response, vs raw per-bin xi.
A PCA+Ridge (same representation, linear) is included as a free reference to show
how much of any gain is the representation vs the GP nonlinearity.

Protocol: identical to emulator_tier3_within_cosmo.py -- 5-fold WITHIN each pilot
cosmology (zero cosmology extrapolation), aggregate the held-out RMS/CV over folds
and cosmologies.  So results are directly comparable to the ~8x MLP floor.

Outputs
  data/emulator_tier3/bakeoff.npz
  plots/emulator_tier3/emulator_bakeoff.png

Usage (GPU node for the MLP):  python scripts/emulator_bakeoff.py [--folds 5] [--ncomp 8]
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import emulator_tier3_mlp as emu
from emulator_tier3_learning_curve import cv_box_per_column   # per-bin CV *std* (box)

REPO = Path(__file__).resolve().parents[1]
PRIORITY = {'tpcf_rand_q1', 'tpcf_rand_q4',
            'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4'}


def gp_factory():
    def make():
        k = (ConstantKernel(1.0, (1e-3, 1e3))
             * Matern(length_scale=np.ones(12), length_scale_bounds=(1e-1, 1e3), nu=2.5)
             + WhiteKernel(1e-2, (1e-6, 1e1)))
        return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                        n_restarts_optimizer=2, alpha=1e-8)
    return make


def pca_reg_predict(Xh_tr, Y_tr, Xh_te, cv, ncomp, regressor):
    """Amplitude-factored residual + DIAGONAL cov-weighted PCA + per-component
    regressor.  We predict d = xi - xi_ref (xi_ref = training HOD-mean: removes the
    bulk amplitude/shape, the robust stand-in for a ratio that is safe at the xi
    zero-crossing), scaled per bin by the cosmic-variance std so PCA compresses in
    the chi^2-relevant directions (diagonal cov-weighting; full 240-D whitening is
    unstable -- the pooled C is not positive-definite)."""
    ref = Y_tr.mean(0)
    scale = cv + 1e-30                                     # per-bin CV std (diag weight)
    Z = (Y_tr - ref) / scale                              # cov-weighted residual
    k = min(ncomp, Z.shape[0] - 1)
    pca = PCA(n_components=k).fit(Z)
    coeff = pca.transform(Z)
    hmu, hsd = Xh_tr.mean(0), Xh_tr.std(0) + 1e-12
    Ztr, Zte = (Xh_tr - hmu) / hsd, (Xh_te - hmu) / hsd
    pred_c = np.zeros((len(Zte), k))
    for j in range(k):
        pred_c[:, j] = regressor().fit(Ztr, coeff[:, j]).predict(Zte)
    Zpred = pca.inverse_transform(pred_c)
    return ref + Zpred * scale                            # back to xi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--ncomp', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=1500)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  folds={args.folds}  ncomp={args.ncomp}')

    ds = emu.load('dataset.npz')
    mask, blocks = emu.select_targets(ds, emu.PRIMARY_STEMS)
    X, cosmo = ds['X'], ds['cosmo']
    Xh = X[:, 8:]                                          # 12 HOD params
    Y, Ynoise = ds['Y'][:, mask], ds['Ynoise'][:, mask]
    cv = cv_box_per_column(blocks)                         # per-bin CV std (box volume)
    pri = np.concatenate([np.r_[sl] for st, el, sl in blocks if st in PRIORITY])
    cosmos = sorted(set(cosmo))

    methods = ['MLP', 'PCA+Ridge', 'PCA+GP']
    res = {m: {'pri': [], 'all': []} for m in methods}
    rng = np.random.default_rng(0)
    for c in cosmos:
        idx = np.where(cosmo == c)[0]
        perm = rng.permutation(len(idx))
        pred = {m: np.zeros((len(idx), Y.shape[1])) for m in methods}
        for f in range(args.folds):
            te_l = perm[f::args.folds]; tr_l = np.setdiff1d(np.arange(len(idx)), te_l)
            tr, te = idx[tr_l], idx[te_l]
            pred['MLP'][te_l], _, _ = emu.fit_ensemble(X[tr], Y[tr], Ynoise[tr], X[te],
                                                       n_ens=1, epochs=args.epochs, seed0=f)
            pred['PCA+Ridge'][te_l] = pca_reg_predict(Xh[tr], Y[tr], Xh[te], cv,
                                                      args.ncomp, lambda: Ridge(alpha=1.0))
            pred['PCA+GP'][te_l] = pca_reg_predict(Xh[tr], Y[tr], Xh[te], cv,
                                                   args.ncomp, gp_factory())
        Yc = Y[idx]
        for m in methods:
            rms = np.sqrt(np.mean((pred[m] - Yc) ** 2, 0))
            res[m]['pri'].append(np.median((rms / cv)[pri]))
            res[m]['all'].append(np.median(rms / cv))
        print(f'{c}:  ' + '   '.join(f'{m} pri={res[m]["pri"][-1]:5.1f}' for m in methods))

    print('\n=== aggregate within-cosmology floor (median over cosmologies, xCV) ===')
    for m in methods:
        print(f'  {m:10s} priority={np.median(res[m]["pri"]):5.2f}   '
              f'all={np.median(res[m]["all"]):5.2f}')
    np.savez(REPO / 'data/emulator_tier3/bakeoff.npz', cosmos=np.array(cosmos),
             **{f'{m}_pri': np.array(res[m]['pri']) for m in methods},
             **{f'{m}_all': np.array(res[m]['all']) for m in methods})

    # ---- figure: per-cosmology bars per method (priority legs) ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(cosmos)); w = 0.27
    for i, (m, col) in enumerate(zip(methods, ['C7', 'C0', 'C3'])):
        ax.bar(x + (i - 1) * w, res[m]['pri'], w, color=col,
               label=f'{m} (med {np.median(res[m]["pri"]):.1f}×)')
    ax.axhline(8, color='C1', ls=':', lw=1, label='current MLP ~8×CV floor')
    ax.axhline(1, color='k', ls='--', lw=1, label='cosmic variance (target)')
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(cosmos, rotation=45, fontsize=8)
    ax.set_ylabel('within-cosmology RMS / CV  (priority legs)')
    ax.set_title('Emulator bake-off: PCA+GP (ratio, cov-whitened) vs MLP\n'
                 'lower = better; tests model-class + metric + representation at once')
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/emulator_bakeoff.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

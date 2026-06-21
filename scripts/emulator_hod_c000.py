#!/usr/bin/env python3
"""
Tier-2 prototype: a pure-HOD emulator at the c000 cosmology.

Trains  f: theta_HOD (12-D) -> data vector  on the 50 c000 full-box runs (the
maximin HOD ensemble).  The cosmology is fixed, so this is the genuinely
space-filling axis of the campaign; it is a *prototype* (50 points in 12-D is
thin) whose job is to (a) validate the emulator machinery and (b) measure how
much a nonlinear GP buys over the linear response model already in
compute_response_global.py, on the same 50 points.

Pipeline: standardise HOD inputs (prior spread) -> standardise outputs per bin ->
PCA compress -> fit per-component {Ridge-linear, GP Matern-5/2 + white noise} ->
leave-one-out CV.  Acceptance is judged per s-bin against two yardsticks:
  * ASTRA noise floor  = mean per-iteration scatter xi_std (no emulator beats it)
  * HOD signal spread  = std of the 50 training vectors (what there is to learn)
Good emulator: LOO RMS ~ noise floor, and << signal spread.

Outputs
  data/emulator/emulator_hod_c000_loo.npz   (per-bin LOO RMS, noise, spread, both models)
  plots/emulator/emulator_hod_c000_loo.png  (LOO RMS vs noise/spread, linear vs GP)
  plots/emulator/emulator_hod_c000_pred.png (predicted vs true, a few held-out runs)

Usage (any node):  python scripts/emulator_hod_c000.py [--all-stems]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
FB_DIR    = DATA_DIR / 'fullbox'
ENS_DIR   = DATA_DIR / 'hod_ensemble'
OUT_DIR   = DATA_DIR / 'emulator'
PLOT_DIR  = REPO_ROOT / 'plots' / 'emulator'

COSMO = 'c000'
N_Q = 4
ALL_STEMS = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)


def load_training(stems):
    """Return (X (n,12), Y (n,nout), Ynoise (nout,), names, s, blocks).

    X = HOD params; Y = concatenated [xi0,xi2] over stems; Ynoise = per-bin ASTRA
    scatter; blocks = list of (stem, ell, slice) for plotting/labelling."""
    df = pd.read_csv(ENS_DIR / f'hod_params_{COSMO}.csv').set_index('hod')
    names = list(df.columns)
    X, rows, Y, Ynoise = [], [], [], []
    s = None
    for d in sorted(FB_DIR.glob(f'{COSMO}_hod*')):
        if not (d / 'fullbox_info.npz').is_file():
            continue
        hod = int(d.name.split('_hod')[1])
        if hod not in df.index:
            continue
        vec, noise = [], []
        ok = True
        for stem in stems:
            f = d / f'fullbox_multipoles_{stem}.npz'
            if not f.is_file():
                ok = False; break
            a = np.load(f)
            if s is None:
                s = a['s']
            for ell in (0, 2):
                vec.append(a[f'xi{ell}']); noise.append(a[f'xi{ell}_std'])
        if not ok:
            continue
        X.append(df.loc[hod].values.astype(float))
        Y.append(np.concatenate(vec)); Ynoise.append(np.concatenate(noise))
        rows.append(hod)
    X, Y, Ynoise = np.array(X), np.array(Y), np.array(Ynoise)
    blocks, i = [], 0
    for stem in stems:
        for ell in (0, 2):
            blocks.append((stem, ell, slice(i, i + len(s)))); i += len(s)
    print(f'Loaded {len(X)} {COSMO} runs; output vector {Y.shape[1]}-D '
          f'({len(stems)} stems x 2 ell x {len(s)} bins)')
    return X, Y, Ynoise.mean(0), names, s, blocks


def fit_predict(model_fn, Xtr, Ytr_pca, Xte):
    """Fit one model per PCA component on standardised inputs; predict Xte."""
    preds = []
    for k in range(Ytr_pca.shape[1]):
        m = model_fn().fit(Xtr, Ytr_pca[:, k])
        preds.append(m.predict(Xte))
    return np.array(preds).T                                  # (nte, ncomp)


def make_models():
    """Return {name: callable producing a fresh estimator}."""
    def gp():
        k = (ConstantKernel(1.0) * Matern(length_scale=np.ones(12), nu=2.5)
             + WhiteKernel(1e-2))
        return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                        n_restarts_optimizer=2, alpha=1e-6)
    return {'linear': lambda: Ridge(alpha=1.0), 'GP': gp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all-stems', action='store_true',
                    help='emulate all 21 stems (630-D) instead of full-auto only')
    args = ap.parse_args()
    stems = ALL_STEMS if args.all_stems else ['tpcf_full_data']
    OUT_DIR.mkdir(parents=True, exist_ok=True); PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X, Y, noise, names, s, blocks = load_training(stems)
    n = len(X)
    # standardise inputs by prior spread (all 500 draws -> stable scale)
    prior = pd.read_csv(ENS_DIR / f'hod_params_{COSMO}.csv')[names].values
    xmu, xsd = prior.mean(0), prior.std(0)
    Xz = (X - xmu) / xsd
    # output standardisation + PCA basis (fit on full set; LOO refits the model
    # in PCA space, an acceptable approximation at this sample size)
    ymu, ysd = Y.mean(0), Y.std(0)
    Yz = (Y - ymu) / ysd
    ncomp = min(n - 1, Yz.shape[1])
    pca = PCA(n_components=ncomp).fit(Yz)
    var = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.searchsorted(var, 0.999) + 1)
    print(f'PCA: {keep} comps for 99.9% variance (of {ncomp})')

    models = make_models()
    loo = {}                                                  # name -> (n, nout)
    for name, fn in models.items():
        pred = np.zeros_like(Y)
        for i in range(n):                                    # leave-one-out
            tr = np.arange(n) != i
            Ytr_pca = pca.transform(Yz[tr])[:, :keep]
            p_pca = fit_predict(fn, Xz[tr], Ytr_pca, Xz[i:i + 1])
            full = np.zeros((1, ncomp)); full[0, :keep] = p_pca[0]
            pred[i] = pca.inverse_transform(full)[0] * ysd + ymu
        loo[name] = pred
        rms = np.sqrt(np.mean((pred - Y) ** 2, axis=0))
        print(f'{name:7s} LOO: median(RMS/spread)={np.median(rms / ysd):.3f}')

    # per-environment breakdown: how well do we predict data/random mono+quad
    # in each quantile?  median LOO RMS / signal spread per (stem, ell).
    print(f'\nPer-block LOO median(RMS/spread)   [{" vs ".join(loo)}]:')
    for stem, ell, sl in blocks:
        vals = [np.median(np.sqrt(np.mean((loo[k][:, sl] - Y[:, sl]) ** 2, 0))
                          / ysd[sl]) for k in loo]
        print(f'  {stem:24s} ℓ{ell}: '
              + '  '.join(f'{k}={v:.3f}' for k, v in zip(loo, vals)))

    spread = ysd                                              # std of training vectors
    np.savez(OUT_DIR / f'emulator_hod_{COSMO}_loo.npz',
             s=s, noise=noise, spread=spread, Y=Y,
             block_labels=np.array([f'{stem}|{ell}' for stem, ell, _ in blocks]),
             **{f'rms_{k}': np.sqrt(np.mean((v - Y) ** 2, 0)) for k, v in loo.items()},
             **{f'pred_{k}': v for k, v in loo.items()})
    print(f'Saved {OUT_DIR / f"emulator_hod_{COSMO}_loo.npz"}')

    # ---- figure 1: per-bin LOO RMS vs noise floor & signal spread ----
    nb = len(blocks)
    ncol = min(4, nb); nrow = int(np.ceil(nb / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (stem, ell, sl) in zip(axs.flat, blocks):
        ax.plot(s, spread[sl], 'k:',  lw=1.4, label='HOD signal spread')
        ax.plot(s, noise[sl],  'C7--', lw=1.4, label='ASTRA noise floor')
        for c, name in zip(('C0', 'C3'), loo):
            rms = np.sqrt(np.mean((loo[name][:, sl] - Y[:, sl]) ** 2, 0))
            ax.plot(s, rms, c, lw=1.8, label=f'{name} LOO RMS')
        ax.set_yscale('log'); ax.set_title(f'{stem} ℓ{ell}', fontsize=8)
        ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f'{COSMO} HOD emulator — leave-one-out error vs yardsticks', y=1.0)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f'emulator_hod_{COSMO}_loo.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {PLOT_DIR / f"emulator_hod_{COSMO}_loo.png"}')

    # ---- figure 2: per-environment held-out prediction for one run ----
    # rows = data ℓ0, data ℓ2, rand ℓ0, rand ℓ2 ; cols = Q1..Q4
    blk = {(stem, ell): sl for stem, ell, sl in blocks}
    cols = [q for q in range(1, N_Q + 1) if ('tpcf_data_q%d' % q, 0) in blk]
    if cols:
        i = n // 2                                            # a held-out run
        rowdefs = [('data', 'tpcf_data_q%d', 0), ('data', 'tpcf_data_q%d', 2),
                   ('rand', 'tpcf_rand_q%d', 0), ('rand', 'tpcf_rand_q%d', 2)]
        fig, axs = plt.subplots(len(rowdefs), len(cols),
                                figsize=(3.2 * len(cols), 2.6 * len(rowdefs)),
                                sharex=True)
        for r, (kind, tmpl, ell) in enumerate(rowdefs):
            for c, q in enumerate(cols):
                ax = axs[r, c]; sl = blk[(tmpl % q, ell)]
                ax.plot(s, s ** 2 * Y[i, sl], 'k-o', ms=3, label='truth')
                for col, name in zip(('C0', 'C3'), loo):
                    ax.plot(s, s ** 2 * loo[name][i, sl], col, lw=1.5, label=name)
                ax.axhline(0, color='grey', lw=0.5)
                if r == 0:
                    ax.set_title(f'Q{q}', fontsize=10)
                if c == 0:
                    ax.set_ylabel(f'{kind} '
                                  + rf'$s^2\xi_{{{ell}}}$', fontsize=9)
                if r == len(rowdefs) - 1:
                    ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f'{COSMO} held-out run #{i} (LOO): data & random '
                     f'monopole+quadrupole per environment', y=1.0)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f'emulator_hod_{COSMO}_pred.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {PLOT_DIR / f"emulator_hod_{COSMO}_pred.png"}')


if __name__ == '__main__':
    main()

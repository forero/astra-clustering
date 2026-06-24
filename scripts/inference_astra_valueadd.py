#!/usr/bin/env python3
"""
THE ASTRA value-add test, in the inference framework.

Does adding the ASTRA environment-quantile legs tighten cosmology constraints over
the plain full-sample 2PCF -- in an actual MCMC recovery, not just Fisher?  We run
the SAME synthetic recovery (same mock, emulator, covariance, parameters, MCMC
settings) for two data vectors and compare the marginalised sigma:

  BASELINE : tpcf_full_data          (the standard 2PCF monopole)
  ASTRA    : tpcf_full_data + 6 clean environment-quantile monopoles
             (data/rand void & knot autos + full-crosses)

The same synthetic data draw is used for both (the baseline is the full_data
sub-block of the ASTRA vector), so the comparison is apples-to-apples; the value-add
is sigma_BASELINE / sigma_ASTRA per parameter.  Covariance = C_CV (subbox, full
matrix) + diagonal C_emu (emulator error, from c000 k-fold residuals).  Fit the 4
LCDM params, HOD fixed (prototype, consistent with inference_monopole.py).

Outputs
  data/emulator_tier3/astra_valueadd.npz
  plots/emulator_tier3/astra_valueadd_corner.png

Usage (GPU node):  python scripts/inference_astra_valueadd.py [--steps 4000] [--ensemble 3]
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import emcee

import emulator_tier3_mlp as emu
from inference_monopole import train_emulator

REPO = Path(__file__).resolve().parents[1]
FULL = ['tpcf_full_data']
QUANT = ['tpcf_data_q4', 'tpcf_cross_full_data_q4', 'tpcf_rand_q1', 'tpcf_rand_q4',
         'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4']
ASTRA = FULL + QUANT
COSMO_NAMES = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
FIT = [0, 1, 2, 3]
FIT_LABELS = [r'\omega_b', r'\omega_{cdm}', 'h', 'n_s']


def load(name, legs):
    d = emu.load(name)
    cols = [np.where((d['stem'] == st) & (d['ell'] == 0))[0] for st in legs]
    cols = np.concatenate(cols)
    return d['X'], d['Y'][:, cols], d['Ynoise'][:, cols], d['cosmo'], d['s']


def subbox_cov(legs, nb):
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    per = []
    for t in tags:
        cols, ok = [], True
        for st in legs:
            f = REPO / 'data' / t / f'subbox_multipoles_{st}.npz'
            if not f.is_file():
                ok = False; break
            cols.append(np.load(f)['xi0_all'])
        if ok:
            V = np.hstack(cols); per.append(V - V.mean(0))
    X = np.vstack(per)
    return np.cov(X, rowvar=False) / 64.0, X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}')

    Xa, Ya, Na, ca, s = load('dataset.npz', ASTRA)
    Xb, Yb, Nb, cb, _ = load('dataset_anchor.npz', ASTRA)
    X = np.vstack([Xa, Xb]); Y = np.vstack([Ya, Yb]); Yn = np.vstack([Na, Nb])
    cosmo = np.concatenate([ca, cb]); nb = len(s)
    nfull = len(FULL) * nb
    print(f'{len(Y)} runs; ASTRA vector {Y.shape[1]}-D (full {nfull} + quant {Y.shape[1]-nfull})')

    # covariance: C_CV (full matrix) + diagonal C_emu from c000 5-fold residuals.
    # The emulator is trained CV-WEIGHTED (cv = sqrt(diag C_CV)) -- the inference metric.
    C_CV, nsamp = subbox_cov(ASTRA, nb)
    cv = np.sqrt(np.diag(C_CV))
    c0 = np.where(cosmo == 'c000')[0]
    rng = np.random.default_rng(0); perm = rng.permutation(len(c0)); resid = np.zeros((len(c0), Y.shape[1]))
    for f in range(5):
        te = c0[perm[f::5]]; tr = c0[np.setdiff1d(np.arange(len(c0)), perm[f::5])]
        pr = train_emulator(X, Y, Yn, exclude=list(np.setdiff1d(np.arange(len(X)), tr)),
                            n_ens=1, epochs=args.epochs, cv=cv)
        resid[perm[f::5]] = Y[te] - np.array([pr(X[i])[0] for i in te])
    C_emu = np.diag(resid.var(0))
    hartlap = (nsamp - Y.shape[1] - 2) / (nsamp - 1)
    print(f'C_CV {nsamp} samples; Hartlap(full vec)={hartlap:.2f}; '
          f'C_emu/C_CV diag med={np.median(np.sqrt(np.diag(C_emu)/np.diag(C_CV))):.2f}')

    # one emulator on all data; one shared synthetic draw
    predict = train_emulator(X, Y, Yn, exclude=[], n_ens=args.ensemble, epochs=args.epochs, cv=cv)
    theta0 = X[c0[len(c0) // 2], :8].copy(); theta_hod = X[c0[len(c0) // 2], 8:].copy()
    lo, hi = X[:, :8].min(0), X[:, :8].max(0)
    theta_inj = theta0.copy()
    for k in FIT:
        theta_inj[k] = theta0[k] + 0.3 * (hi[k] - theta0[k] if theta0[k] < (lo[k]+hi[k])/2 else lo[k] - theta0[k])
    C_tot = C_CV + C_emu
    d_full = predict(np.concatenate([theta_inj, theta_hod]))[0]
    d_synth = d_full + rng.multivariate_normal(np.zeros(len(C_tot)), C_tot)

    def recover(cols):
        C = C_tot[np.ix_(cols, cols)]; C += 1e-3 * np.median(np.diag(C)) * np.eye(len(C))
        h = (nsamp - len(C) - 2) / (nsamp - 1)
        Cinv = h * np.linalg.inv(C)
        dv = d_synth[cols]

        def logprob(p):
            if np.any(p < lo[FIT]) or np.any(p > hi[FIT]):
                return -np.inf
            th = theta_inj.copy(); th[FIT] = p
            r = dv - predict(np.concatenate([th, theta_hod]))[0][cols]
            return -0.5 * r @ Cinv @ r
        nd, nw = len(FIT), 32
        p0 = theta_inj[FIT] + (hi[FIT] - lo[FIT]) * 1e-3 * np.random.randn(nw, nd)
        sm = emcee.EnsembleSampler(nw, nd, logprob); sm.run_mcmc(p0, args.steps, progress=False)
        return sm.get_chain(discard=args.steps // 2, flat=True)

    ch_base = recover(np.arange(nfull))                  # full_data only
    ch_astra = recover(np.arange(Y.shape[1]))            # full + quantiles
    print('\n=== ASTRA value-add (synthetic recovery; sigma per param) ===')
    print(f'{"param":10s} {"BASELINE 2PCF":>14s} {"FULL+ASTRA":>14s} {"tighter by":>11s}')
    for i, k in enumerate(FIT):
        sb, sa = ch_base[:, i].std(), ch_astra[:, i].std()
        print(f'{COSMO_NAMES[k]:10s} {sb:14.3g} {sa:14.3g} {sb/sa:10.2f}x')

    np.savez(REPO / 'data/emulator_tier3/astra_valueadd.npz',
             chain_base=ch_base, chain_astra=ch_astra, truth=theta_inj[FIT],
             names=np.array([COSMO_NAMES[k] for k in FIT]))
    try:
        from getdist import MCSamples, plots
        nd = len(FIT)
        ss = [MCSamples(samples=ch_base, names=[f'p{i}' for i in range(nd)], labels=FIT_LABELS,
                        settings={'smooth_scale_2D': 0.6}, label='full 2PCF only'),
              MCSamples(samples=ch_astra, names=[f'p{i}' for i in range(nd)], labels=FIT_LABELS,
                        settings={'smooth_scale_2D': 0.6}, label='full + ASTRA quantiles')]
        g = plots.get_subplot_plotter(); g.settings.num_plot_contours = 2
        g.triangle_plot(ss, filled=True, legend_labels=['full 2PCF only', 'full + ASTRA quantiles'])
        for i in range(nd):
            for j in range(i + 1):
                ax = g.subplots[i, j]
                if ax is None:
                    continue
                ax.axvline(theta_inj[FIT[j]], color='k', lw=1, ls='--')
                if i != j:
                    ax.axhline(theta_inj[FIT[i]], color='k', lw=1, ls='--')
        out = REPO / 'plots/emulator_tier3/astra_valueadd_corner.png'
        g.export(str(out)); print(f'Saved {out}')
    except Exception as ex:
        print(f'corner skipped: {ex}')


if __name__ == '__main__':
    main()

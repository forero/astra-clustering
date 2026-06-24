#!/usr/bin/env python3
"""
HOD marginalisation: the honest cosmology errors.

The inference prototype fixes the 12 HOD parameters at the mock's truth -- optimistic.
This runs the SAME c000 recovery (6 clean monopole legs, 4 LCDM params) twice:
  FIXED : vary the 4 cosmology params only (HOD held at truth)
  MARG  : vary cosmology + all 12 HOD params, then marginalise (the HOD are nuisance)
and reports how much the cosmology errors inflate -- the realistic number.

Covariance C = C_CV (subbox) + C_emu (Fisher-set LOCO residuals) + C_label.
The emulator already takes the 12 HOD as inputs, so marginalisation just opens those
directions in the MCMC with uniform priors over the training HOD range.

Outputs
  data/emulator_tier3/inference_hodmarg.npz
  plots/emulator_tier3/inference_hodmarg_corner.png

Usage (GPU node):  python scripts/inference_hodmarg.py [--steps 8000] [--ensemble 3]
"""
import argparse
from pathlib import Path
import numpy as np
import emcee

import emulator_tier3_mlp as emu
from inference_monopole import train_emulator, subbox_cov, load_monopole, LEGS

REPO = Path(__file__).resolve().parents[1]
COSMO_NAMES = ['omega_b', 'omega_cdm', 'h', 'n_s']
FIT = [0, 1, 2, 3]                                    # cosmology indices (in the 20-vec)
HOD = list(range(8, 20))                              # 12 HOD indices (in the 20-vec)
LABELS = [r'\omega_b', r'\omega_{cdm}', 'h', 'n_s']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}')

    Xa, Ya, Na, ca, stems, ells, cols, s = load_monopole('dataset.npz')
    Xb, Yb, Nb, cb, *_ = load_monopole('dataset_anchor.npz')
    X = np.vstack([Xa, Xb]); Y = np.vstack([Ya, Yb]); Yn = np.vstack([Na, Nb])
    cosmo = np.concatenate([ca, cb]); nb = len(s)

    c0 = np.where(cosmo == 'c000')[0]; mock = int(c0[len(c0) // 2])
    d = Y[mock].copy(); base = X[mock].copy()            # full 20-vector (cosmo+HOD truth)
    print(f'mock = c000 #{mock}; {Y.shape[1]}-D vector ({len(LEGS)} legs x {nb})')

    C_CV, nsamp = subbox_cov(stems, nb)
    lc = np.load(REPO / 'data/emulator_tier3/monopole_loco.npz', allow_pickle=True)
    lc_stems = lc['stems'].astype(str); lc_cos = lc['cosmo'].astype(str)
    leg_cols = np.concatenate([np.where(lc_stems == st)[0] * nb + np.arange(nb) for st in LEGS])
    fisher = np.array([int(c[1:]) < 130 for c in lc_cos])
    C_emu = np.cov(lc['resid'][np.ix_(fisher, leg_cols)], rowvar=False)
    C_label = np.diag((Yn[mock] ** 2) / 3.0)
    C = C_CV + C_emu + C_label
    C += 1e-3 * np.median(np.diag(C)) * np.eye(len(C))
    Cinv = (nsamp - len(C) - 2) / (nsamp - 1) * np.linalg.inv(C)

    predict = train_emulator(X, Y, Yn, exclude=[mock], n_ens=args.ensemble, epochs=args.epochs,
                             cv=np.sqrt(np.diag(C_CV)))   # CV-weighted loss
    lo, hi = X.min(0), X.max(0)                           # training-domain priors (all 20)

    def recover(vfit, steps):
        vfit = np.array(vfit)

        def logprob(p):
            if np.any(p < lo[vfit]) or np.any(p > hi[vfit]):
                return -np.inf
            th = base.copy(); th[vfit] = p
            r = d - predict(th)[0]
            return -0.5 * r @ Cinv @ r
        nd = len(vfit); nw = max(32, 2 * nd + 2)
        p0 = base[vfit] + (hi[vfit] - lo[vfit]) * 1e-3 * np.random.randn(nw, nd)
        sm = emcee.EnsembleSampler(nw, nd, logprob)
        sm.run_mcmc(p0, steps, progress=False)
        return sm.get_chain(discard=steps // 2, flat=True)

    ch_fix = recover(FIT, args.steps)                              # HOD fixed (4-D)
    ch_mrg = recover(FIT + HOD, args.steps)                        # cosmo+HOD (16-D)
    ch_mrg_cos = ch_mrg[:, :len(FIT)]                              # cosmology block (marginalised)

    print('\n=== HOD marginalisation (cosmology sigma) ===')
    print(f'{"param":10s} {"HOD fixed":>12s} {"HOD marg":>12s} {"inflated":>10s}')
    for i, k in enumerate(FIT):
        sf, sm_ = ch_fix[:, i].std(), ch_mrg_cos[:, i].std()
        print(f'{COSMO_NAMES[i]:10s} {sf:12.3g} {sm_:12.3g} {sm_/sf:9.1f}x')

    np.savez(REPO / 'data/emulator_tier3/inference_hodmarg.npz',
             chain_fixed=ch_fix, chain_marg_cosmo=ch_mrg_cos, truth=base[FIT],
             names=np.array(COSMO_NAMES))
    try:
        from getdist import MCSamples, plots
        nd = len(FIT)
        ss = [MCSamples(samples=ch_fix, names=[f'p{i}' for i in range(nd)], labels=LABELS,
                        settings={'smooth_scale_2D': 0.6}, label='HOD fixed'),
              MCSamples(samples=ch_mrg_cos, names=[f'p{i}' for i in range(nd)], labels=LABELS,
                        settings={'smooth_scale_2D': 0.6}, label='HOD marginalised')]
        g = plots.get_subplot_plotter(); g.settings.num_plot_contours = 2
        g.triangle_plot(ss, filled=True, legend_labels=['HOD fixed', 'HOD marginalised'])
        for i in range(nd):
            for j in range(i + 1):
                ax = g.subplots[i, j]
                if ax is None:
                    continue
                ax.axvline(base[FIT[j]], color='k', lw=1, ls='--')
                if i != j:
                    ax.axhline(base[FIT[i]], color='k', lw=1, ls='--')
        out = REPO / 'plots/emulator_tier3/inference_hodmarg_corner.png'
        g.export(str(out)); print(f'Saved {out}')
    except Exception as ex:
        print(f'corner skipped: {ex}')


if __name__ == '__main__':
    main()

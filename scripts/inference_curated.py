#!/usr/bin/env python3
"""
Inference with the GREEDY-CURATED best vector:
  full 2PCF monopole (tpcf_full_data l0)  +  void-random-cross monopole
  (tpcf_cross_full_rand_q1 l0)
which the emulator-aware greedy (emulator_greedy.py) found to be the strongest small
vector -- led by an ASTRA leg, not the plain 2PCF.

Three deliverables, all CV-weighted emulator, C = C_CV + C_emu (C_emu from an on-the-fly
c000 5-fold, so it covers full_data which the LOCO-residual C_emu did not):
  (1) VALUE-ADD : synthetic recovery, baseline full 2PCF vs curated (full+void-cross)
  (2) RECOVERY  : curated vector, synthetic + real c000 (HOD fixed)
  (3) HOD-MARG  : curated vector, real c000, HOD fixed vs marginalised (12 HOD nuisance)

Outputs (plots/emulator_tier3/):
  inference_curated_valueadd.png, inference_curated_recovery.png, inference_curated_hodmarg.png
  data/emulator_tier3/inference_curated.npz

Usage (GPU node):  python scripts/inference_curated.py [--steps 4000] [--ensemble 3]
"""
import argparse
from pathlib import Path
import numpy as np
import emcee

import emulator_tier3_mlp as emu
from inference_monopole import train_emulator
from inference_astra_valueadd import load, subbox_cov

REPO = Path(__file__).resolve().parents[1]
FULL = ['tpcf_full_data']
CURATED = ['tpcf_full_data', 'tpcf_cross_full_rand_q1']
COSMO_NAMES = ['omega_b', 'omega_cdm', 'h', 'n_s']
LABELS = [r'\omega_b', r'\omega_{cdm}', 'h', 'n_s']
FIT = [0, 1, 2, 3]
HOD = list(range(8, 20))


def cemu_cfold(X, Y, Yn, cosmo, cv, epochs, fid='c000'):
    """Full emulator-error covariance from a CV-weighted c000 5-fold."""
    c0 = np.where(cosmo == fid)[0]
    rng = np.random.default_rng(0); perm = rng.permutation(len(c0))
    resid = np.zeros((len(c0), Y.shape[1]))
    for f in range(5):
        te = c0[perm[f::5]]; tr = c0[np.setdiff1d(np.arange(len(c0)), perm[f::5])]
        pr = train_emulator(X, Y, Yn, exclude=list(np.setdiff1d(np.arange(len(X)), tr)),
                            n_ens=1, epochs=epochs, cv=cv)
        resid[perm[f::5]] = Y[te] - np.array([pr(X[i])[0] for i in te])
    return np.cov(resid, rowvar=False)


def corner(chains, truths, labels, colors, legend, truth_lines, out):
    from getdist import MCSamples, plots
    nd = len(FIT)
    ss = [MCSamples(samples=ch, names=[f'p{i}' for i in range(nd)], labels=labels,
                    settings={'smooth_scale_2D': 0.6}, label=lab)
          for ch, lab in zip(chains, legend)]
    g = plots.get_subplot_plotter(); g.settings.num_plot_contours = 2
    g.triangle_plot(ss, filled=True, contour_colors=colors, legend_labels=legend)
    for i in range(nd):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if ax is None:
                continue
            ax.axvline(truth_lines[j], color='k', lw=1, ls='--')
            if i != j:
                ax.axhline(truth_lines[i], color='k', lw=1, ls='--')
    g.export(str(out)); print(f'Saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  curated = full 2PCF + void-random-cross (monopole)')

    Xa, Ya, Na, ca, s = load('dataset.npz', CURATED)
    Xb, Yb, Nb, cb, _ = load('dataset_anchor.npz', CURATED)
    X = np.vstack([Xa, Xb]); Y = np.vstack([Ya, Yb]); Yn = np.vstack([Na, Nb])
    cosmo = np.concatenate([ca, cb]); nb = len(s)
    nfull = len(FULL) * nb                                 # full_data columns (baseline)
    print(f'{len(Y)} runs; curated {Y.shape[1]}-D (full {nfull} + void-cross {Y.shape[1]-nfull})')

    C_CV, nsamp = subbox_cov(CURATED, nb); cv = np.sqrt(np.diag(C_CV))
    C_emu = cemu_cfold(X, Y, Yn, cosmo, cv, args.epochs)
    print(f'C_emu/C_CV diag med = {np.median(np.sqrt(np.diag(C_emu)/np.diag(C_CV))):.2f}')
    C_tot = C_CV + C_emu

    c0 = np.where(cosmo == 'c000')[0]; mock = int(c0[len(c0) // 2])
    base = X[mock].copy(); d_real = Y[mock].copy()
    C_label = np.diag((Yn[mock] ** 2) / 3.0)
    predict = train_emulator(X, Y, Yn, exclude=[mock], n_ens=args.ensemble, epochs=args.epochs, cv=cv)
    lo, hi = X.min(0), X.max(0)
    hart = (nsamp - Y.shape[1] - 2) / (nsamp - 1)

    def recover(cols, vfit, d_vec, Cmat, base20, steps):
        idx = np.array(cols); vfit = np.array(vfit)
        C = Cmat[np.ix_(idx, idx)]; C += 1e-3 * np.median(np.diag(C)) * np.eye(len(C))
        h = (nsamp - len(C) - 2) / (nsamp - 1); Cinv = h * np.linalg.inv(C)

        def logprob(p):
            if np.any(p < lo[vfit]) or np.any(p > hi[vfit]):
                return -np.inf
            th = base20.copy(); th[vfit] = p
            r = d_vec - predict(th)[0][idx]
            return -0.5 * r @ Cinv @ r
        ndim = len(vfit); nw = max(32, 2 * ndim + 2)
        p0 = base20[vfit] + (hi[vfit] - lo[vfit]) * 1e-3 * np.random.randn(nw, ndim)
        sm = emcee.EnsembleSampler(nw, ndim, logprob); sm.run_mcmc(p0, steps, progress=False)
        return sm.get_chain(discard=steps // 2, flat=True)

    cols_full = np.arange(nfull); cols_cur = np.arange(Y.shape[1])

    # (1) VALUE-ADD: synthetic, baseline vs curated
    inj = base.copy()
    for k in FIT:
        inj[k] = base[k] + 0.3 * (hi[k] - base[k] if base[k] < (lo[k]+hi[k])/2 else lo[k] - base[k])
    rng = np.random.default_rng(1)
    d_syn = predict(inj)[0] + rng.multivariate_normal(np.zeros(len(C_tot)), C_tot)
    ch_base = recover(cols_full, FIT, d_syn[cols_full], C_tot, inj, args.steps)
    ch_cur = recover(cols_cur, FIT, d_syn[cols_cur], C_tot, inj, args.steps)
    print('\n=== (1) value-add (synthetic) sigma ===')
    for i, k in enumerate(FIT):
        sb, sc = ch_base[:, i].std(), ch_cur[:, i].std()
        print(f'  {COSMO_NAMES[i]:10s} full2PCF={sb:.3g}  curated={sc:.3g}  tighter {sb/sc:.2f}x')
    corner([ch_base, ch_cur], None, LABELS, ['#d62728', '#1f77b4'],
           ['full 2PCF only', 'full + void-random-cross'], inj[FIT],
           REPO / 'plots/emulator_tier3/inference_curated_valueadd.png')

    # (2) RECOVERY: curated vector, synthetic vs real c000 (HOD fixed)
    ch_syn = recover(cols_cur, FIT, d_syn[cols_cur], C_tot, inj, args.steps)   # truth=inj
    ch_realfix = recover(cols_cur, FIT, d_real, C_tot + C_label, base, args.steps)
    print('\n=== (2) recovery (curated): real c000 HOD-fixed sigma & pulls ===')
    for i, k in enumerate(FIT):
        m = ch_realfix[:, i]
        print(f'  {COSMO_NAMES[i]:10s} truth={base[k]:.4g} post={m.mean():.4g}±{m.std():.2g} '
              f'({(m.mean()-base[k])/m.std():+.1f}σ)')
    corner([ch_realfix], None, LABELS, ['#1f77b4'], ['real c000 (curated)'], base[FIT],
           REPO / 'plots/emulator_tier3/inference_curated_recovery.png')

    # (3) HOD-MARG: curated vector, real c000, fixed vs marginalised
    ch_marg = recover(cols_cur, FIT + HOD, d_real, C_tot + C_label, base, max(8000, args.steps))
    ch_marg_c = ch_marg[:, :len(FIT)]
    print('\n=== (3) HOD marginalisation (curated): sigma fixed vs marg ===')
    for i, k in enumerate(FIT):
        sf, sm_ = ch_realfix[:, i].std(), ch_marg_c[:, i].std()
        print(f'  {COSMO_NAMES[i]:10s} fixed={sf:.3g}  marg={sm_:.3g}  inflated {sm_/sf:.1f}x')
    corner([ch_realfix, ch_marg_c], None, LABELS, ['#d62728', '#1f77b4'],
           ['HOD fixed', 'HOD marginalised'], base[FIT],
           REPO / 'plots/emulator_tier3/inference_curated_hodmarg.png')

    np.savez(REPO / 'data/emulator_tier3/inference_curated.npz',
             ch_base=ch_base, ch_cur=ch_cur, inj=inj[FIT],
             ch_realfix=ch_realfix, ch_marg_c=ch_marg_c, truth=base[FIT],
             names=np.array(COSMO_NAMES))


if __name__ == '__main__':
    main()

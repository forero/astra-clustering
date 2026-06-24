#!/usr/bin/env python3
"""
Recovery tests INCLUDING the dark-energy EoS (w0, wa), on real cosmologies with
non-trivial w0/wa.

Vector: full 2PCF monopole + quadrupole (the quadrupole carries the AP/RSD signal
that constrains w0/wa) + the top environment leg (void-random-cross monopole):
  [(full_data,0), (full_data,2), (cross_full_rand_q1,0)].
For each mock cosmology we train the CV-weighted emulator with that cosmology HELD
OUT (genuine test), build C = C_CV (subbox) + C_emu (the mock's own held-out residual
variance) + C_label, and recover the 6 broad params {omega_b,omega_cdm,h,n_s,w0,wa}
with emcee. We check the truth is recovered (pulls), especially w0, wa.

Mocks (non-trivial w0/wa, decent LOCO): c178, c147, c157 (spread over the w0-wa plane).

Outputs
  data/emulator_tier3/inference_w0wa.npz
  plots/emulator_tier3/inference_w0wa_<cosmo>.png   (one corner per mock)

Usage (GPU node):  python scripts/inference_w0wa.py [--steps 6000] [--ensemble 3]
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import emcee

import emulator_tier3_mlp as emu
from inference_monopole import train_emulator

REPO = Path(__file__).resolve().parents[1]
LEGS = [('tpcf_full_data', 0), ('tpcf_full_data', 2), ('tpcf_cross_full_rand_q1', 0)]
MOCKS = ['c178', 'c147', 'c157']
NAMES = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
LAB = [r'\omega_b', r'\omega_{cdm}', 'h', 'n_s', 'w_0', 'w_a']
FIT = [0, 1, 2, 3, 6, 7]                              # incl w0 (6), wa (7)


def load(name):
    d = emu.load(name)
    cols = np.concatenate([np.where((d['stem'] == st) & (d['ell'] == el))[0] for st, el in LEGS])
    return d['X'], d['Y'][:, cols], d['Ynoise'][:, cols], d['cosmo'], d['s']


def subbox_cov(nb):
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    per = []
    for t in tags:
        cols, ok = [], True
        for st, el in LEGS:
            f = REPO / 'data' / t / f'subbox_multipoles_{st}.npz'
            if not f.is_file():
                ok = False; break
            cols.append(np.load(f)[f'xi{el}_all'])
        if ok:
            V = np.hstack(cols); per.append(V - V.mean(0))
    Xs = np.vstack(per)
    return np.cov(Xs, rowvar=False) / 64.0, Xs.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  vector={[f"{s}|{e}" for s,e in LEGS]}')

    Xa, Ya, Na, ca, s = load('dataset.npz'); Xb, Yb, Nb, cb, _ = load('dataset_anchor.npz')
    X = np.vstack([Xa, Xb]); Y = np.vstack([Ya, Yb]); Yn = np.vstack([Na, Nb])
    cosmo = np.concatenate([ca, cb]); nb = len(s)
    C_CV, nsamp = subbox_cov(nb); cv = np.sqrt(np.diag(C_CV))
    lo, hi = X.min(0), X.max(0)
    print(f'{len(Y)} runs; vector {Y.shape[1]}-D')

    out = {}
    for mc in MOCKS:
        te = cosmo == mc; midx = np.where(te)[0]
        # emulator with this cosmology held out (genuine test) + its residual C_emu
        predict = train_emulator(X, Y, Yn, exclude=list(midx),
                                 n_ens=args.ensemble, epochs=args.epochs, cv=cv)
        resid = Y[te] - np.array([predict(X[i])[0] for i in midx])
        C_emu = np.diag(resid.var(0))
        mock = int(midx[len(midx) // 2]); base = X[mock].copy(); d = Y[mock].copy()
        C = C_CV + C_emu + np.diag((Yn[mock] ** 2) / 3.0)
        C += 1e-3 * np.median(np.diag(C)) * np.eye(len(C))
        Cinv = (nsamp - len(C) - 2) / (nsamp - 1) * np.linalg.inv(C)

        def logprob(p):
            if np.any(p < lo[FIT]) or np.any(p > hi[FIT]):
                return -np.inf
            th = base.copy(); th[FIT] = p
            r = d - predict(th)[0]
            return -0.5 * r @ Cinv @ r
        nd, nw = len(FIT), 40
        p0 = base[FIT] + (hi[FIT] - lo[FIT]) * 1e-3 * np.random.randn(nw, nd)
        sm = emcee.EnsembleSampler(nw, nd, logprob); sm.run_mcmc(p0, args.steps, progress=False)
        ch = sm.get_chain(discard=args.steps // 2, flat=True)
        out[mc] = (ch, base[FIT].copy())
        r0 = d - predict(base)[0]
        print(f'\n=== {mc} (w0={base[6]:.3f}, wa={base[7]:.3f}); chi2/dof@truth={r0@Cinv@r0/len(d):.2f} ===')
        for i, k in enumerate(FIT):
            m = ch[:, i]
            print(f'  {NAMES[k]:10s} truth={base[k]:.4g} post={m.mean():.4g}±{m.std():.2g} '
                  f'({(m.mean()-base[k])/m.std():+.1f}σ)')

        try:
            from getdist import MCSamples, plots
            S = MCSamples(samples=ch, names=[f'p{i}' for i in range(nd)], labels=LAB,
                          settings={'smooth_scale_2D': 0.6})
            g = plots.get_subplot_plotter(); g.settings.num_plot_contours = 2
            g.triangle_plot([S], filled=True, contour_colors=['#1f77b4'])
            for i in range(nd):
                for j in range(i + 1):
                    ax = g.subplots[i, j]
                    if ax is None:
                        continue
                    ax.axvline(base[FIT[j]], color='k', lw=1, ls='--')
                    if i != j:
                        ax.axhline(base[FIT[i]], color='k', lw=1, ls='--')
            p = REPO / f'plots/emulator_tier3/inference_w0wa_{mc}.png'
            g.export(str(p)); print(f'Saved {p}')
        except Exception as ex:
            print(f'corner skipped: {ex}')

    np.savez(REPO / 'data/emulator_tier3/inference_w0wa.npz',
             **{f'chain_{m}': out[m][0] for m in out},
             **{f'truth_{m}': out[m][1] for m in out},
             names=np.array([NAMES[k] for k in FIT]))


if __name__ == '__main__':
    main()

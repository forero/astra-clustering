#!/usr/bin/env python3
"""
Emulator-AWARE greedy leg selection.

The old Fisher greedy ranked legs with C = C_CV only (a perfect emulator) and crowned
the environment crosses. The value-add test showed that ranking does not survive real
inference, where C = C_CV + C_emu down-weights the (emulator-error-dominated) quantile
legs. This redoes the greedy with the REALISTIC budget and emulator-derived
derivatives, and sweeps a C_emu scaling alpha to show how the optimal vector shifts as
the campaign drives C_emu -> 0.

  derivatives  D = d xi / d theta  (finite-difference the CV-weighted emulator at c000)
  covariance   C(alpha) = C_CV (subbox) + alpha * C_emu (CV-weighted LOCO residuals)
  FoM          det( D_cosmo^T C^-1 D_cosmo )^(1/2)  over {omega_b, omega_cdm, h, n_s}
  greedy       forward-add the (stem, ell) leg-block that maximises the FoM, per alpha

Candidates: full_data + the 8 environment stems (void/knot data&random autos & full-
crosses), monopole AND quadrupole = 18 leg-blocks.

Outputs
  data/emulator_tier3/greedy.npz
  plots/emulator_tier3/emulator_greedy.png

Usage (GPU node):  python scripts/emulator_greedy.py [--ensemble 1] [--epochs 1500]
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu
from inference_monopole import train_emulator

REPO = Path(__file__).resolve().parents[1]
CANDID = (['tpcf_full_data'] + ['tpcf_data_q1', 'tpcf_data_q4',
          'tpcf_cross_full_data_q1', 'tpcf_cross_full_data_q4',
          'tpcf_rand_q1', 'tpcf_rand_q4',
          'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4'])
ELLS = [0, 2]
FIT = [0, 1, 2, 3]                                   # FoM over omega_b, omega_cdm, h, n_s
ALPHAS = [1.0, 0.3, 0.1, 0.0]                        # C_emu scaling (0 = perfect emulator)


def load_legs():
    a = emu.load('dataset.npz'); b = emu.load('dataset_anchor.npz')
    legs, cols = [], []
    for st in CANDID:
        for el in ELLS:
            c = np.where((a['stem'] == st) & (a['ell'] == el))[0]
            legs.append((st, el)); cols.append(c)
    nb = len(a['s']); col_idx = np.concatenate(cols)
    X = np.vstack([a['X'], b['X']])
    Y = np.vstack([a['Y'][:, col_idx], b['Y'][:, col_idx]])
    Yn = np.vstack([a['Ynoise'][:, col_idx], b['Ynoise'][:, col_idx]])
    cosmo = np.concatenate([a['cosmo'], b['cosmo']])
    blocks = [(st, el, slice(i * nb, (i + 1) * nb)) for i, (st, el) in enumerate(legs)]
    return X, Y, Yn, cosmo, a['s'], blocks


def subbox_cov(blocks):
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    per = []
    for t in tags:
        cols, ok = [], True
        for st, el, _ in blocks:
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
    ap.add_argument('--ensemble', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=1500)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}')

    X, Y, Yn, cosmo, s, blocks = load_legs()
    nb = len(s); ncol = Y.shape[1]
    C_CV, nsamp = subbox_cov(blocks); cv = np.sqrt(np.diag(C_CV))
    print(f'{len(Y)} runs; {len(blocks)} leg-blocks x {nb} bins = {ncol}-D')

    # C_emu: CV-weighted LOCO residuals over all cosmologies
    cosmos = sorted(set(cosmo)); resid = np.zeros_like(Y)
    for c in cosmos:
        te = cosmo == c
        pr = train_emulator(X, Y, Yn, exclude=list(np.where(te)[0]),
                            n_ens=args.ensemble, epochs=args.epochs, cv=cv)
        resid[te] = Y[te] - np.array([pr(X[i])[0] for i in np.where(te)[0]])
    C_emu = np.cov(resid, rowvar=False)
    print(f'C_emu/C_CV diag med = {np.median(np.sqrt(np.diag(C_emu)/np.diag(C_CV))):.2f}')

    # derivatives: finite-difference the all-data CV-weighted emulator at the c000 fiducial
    predict = train_emulator(X, Y, Yn, exclude=[], n_ens=max(3, args.ensemble), epochs=args.epochs, cv=cv)
    c0 = np.where(cosmo == 'c000')[0]; th0 = X[c0[len(c0) // 2]].copy()
    lo, hi = X[:, :8].min(0), X[:, :8].max(0)
    D = np.zeros((ncol, 8))
    for k in range(8):
        dth = 0.05 * (hi[k] - lo[k])
        tp = th0.copy(); tp[k] += dth; tm = th0.copy(); tm[k] -= dth
        D[:, k] = (predict(tp)[0] - predict(tm)[0]) / (2 * dth)

    def fom(sel_cols, alpha):
        idx = np.array(sel_cols)
        C = C_CV[np.ix_(idx, idx)] + alpha * C_emu[np.ix_(idx, idx)]
        C += 1e-3 * np.median(np.diag(C)) * np.eye(len(C))
        Cinv = np.linalg.inv(C)
        F = D[idx][:, FIT].T @ Cinv @ D[idx][:, FIT]
        sign, logdet = np.linalg.slogdet(F)
        return np.exp(0.5 * logdet) if sign > 0 else 0.0

    results = {}
    for alpha in ALPHAS:
        chosen, sel_cols, curve = [], [], []
        remaining = list(range(len(blocks)))
        while remaining:
            gains = [(fom(sel_cols + list(np.r_[blocks[b][2]]), alpha), b) for b in remaining]
            f_best, b_best = max(gains)
            chosen.append(b_best); sel_cols += list(np.r_[blocks[b_best][2]])
            remaining.remove(b_best); curve.append(f_best)
        results[alpha] = (chosen, curve)
        order = ' > '.join(f'{blocks[b][0].replace("tpcf_","")}l{blocks[b][1]}' for b in chosen[:6])
        print(f'\nalpha={alpha}: FoM saturates {curve[-1]/curve[len(chosen)//2]:.2f}x past mid; top: {order}')

    np.savez(REPO / 'data/emulator_tier3/greedy.npz',
             legs=np.array([f'{b[0]}|{b[1]}' for b in blocks]), alphas=np.array(ALPHAS),
             **{f'order_{a}': np.array(results[a][0]) for a in ALPHAS},
             **{f'curve_{a}': np.array(results[a][1]) for a in ALPHAS})

    # figure: cumulative FoM vs number of legs, per alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha, col in zip(ALPHAS, ['C3', 'C1', 'C0', 'C2']):
        ch, cu = results[alpha]
        ax.plot(np.arange(1, len(cu) + 1), np.array(cu) / cu[0], 'o-', color=col,
                label=f'$\\alpha$(C_emu)={alpha}' + (' (perfect emul.)' if alpha == 0 else ''))
    ax.set_xlabel('number of legs added (greedy order)'); ax.set_ylabel('cumulative FoM / first-leg FoM')
    ax.set_yscale('log')
    ax.set_title('Emulator-aware greedy: FoM vs data vector, swept over emulator error\n'
                 'alpha=1 current emulator, alpha=0 perfect (old Fisher regime)')
    ax.legend(fontsize=8); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/emulator_greedy.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')

    # print the greedy ORDER table (rank of each leg per alpha)
    print(f'\n{"leg":24s} ' + ' '.join(f'a={a:>4}' for a in ALPHAS))
    for bi, (st, el, _) in enumerate(blocks):
        ranks = [list(results[a][0]).index(bi) + 1 for a in ALPHAS]
        print(f'{st.replace("tpcf_","")+" l"+str(el):24s} ' + ' '.join(f'{r:5d}' for r in ranks))


if __name__ == '__main__':
    main()

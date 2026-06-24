#!/usr/bin/env python3
"""
Per-leg emulator-quality diagnostic (default: the top greedy leg,
void-random-cross monopole tpcf_cross_full_rand_q1 l0), for the CV-WEIGHTED emulator.

Trains the curated 2-leg emulator (full_data l0 + the target leg) CV-weighted, leave-
one-cosmology-out, and shows how well it predicts the target leg via four panels:
  (a) per-bin RMS/CV vs s, with the CV target (=1) and the HOD signal spread;
  (b) predicted vs true s^2 xi for a few held-out cosmologies (Fisher / tier3 / edge);
  (c) predicted vs true scatter at a small-scale bin, Fisher vs tier3;
  (d) per-cosmology median RMS/CV (interpolation vs extrapolation).

Outputs  plots/emulator_tier3/leg_diagnostic_<leg>.png

Usage (GPU node):  python scripts/emulator_leg_diagnostic.py [--leg tpcf_cross_full_rand_q1] [--ell 0]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu
from inference_monopole import train_emulator
from inference_astra_valueadd import subbox_cov

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', default='tpcf_cross_full_rand_q1')
    ap.add_argument('--ell', type=int, default=0)
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    LEGS = ['tpcf_full_data', args.leg]                   # curated 2-leg training context

    def load(name):
        d = emu.load(name)
        cols = np.concatenate([np.where((d['stem'] == st) & (d['ell'] == args.ell))[0] for st in LEGS])
        return d['X'], d['Y'][:, cols], d['Ynoise'][:, cols], d['cosmo'], d['s']
    Xa, Ya, Na, ca, s = load('dataset.npz'); Xb, Yb, Nb, cb, _ = load('dataset_anchor.npz')
    X = np.vstack([Xa, Xb]); Y = np.vstack([Ya, Yb]); Yn = np.vstack([Na, Nb])
    cosmo = np.concatenate([ca, cb]); nb = len(s)
    C_CV, _ = subbox_cov(LEGS, nb); cv = np.sqrt(np.diag(C_CV))
    tgt = slice(nb, 2 * nb)                               # target leg = 2nd block
    print(f'{len(Y)} runs; diagnosing {args.leg} l{args.ell}')

    # CV-weighted LOCO
    Yp = np.zeros_like(Y)
    for c in sorted(set(cosmo)):
        te = cosmo == c
        pr = train_emulator(X, Y, Yn, exclude=list(np.where(te)[0]),
                            n_ens=args.ensemble, epochs=args.epochs, cv=cv)
        Yp[te] = np.array([pr(X[i])[0] for i in np.where(te)[0]])

    Yt, Ypt, cvt = Y[:, tgt], Yp[:, tgt], cv[tgt]
    rms = np.sqrt(np.mean((Ypt - Yt) ** 2, 0))
    spread = Yt.std(0)
    fisher = np.array([int(x[1:]) < 130 for x in cosmo])
    percos = {c: np.median(np.sqrt(np.mean((Ypt[cosmo == c] - Yt[cosmo == c]) ** 2, 0)) / cvt)
              for c in sorted(set(cosmo))}
    print(f'overall median RMS/CV = {np.median(rms / cvt):.2f}; '
          f'Fisher {np.median([percos[c] for c in percos if int(c[1:])<130]):.2f}, '
          f'tier3 {np.median([percos[c] for c in percos if int(c[1:])>=130]):.2f}')

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # (a) per-bin RMS/CV vs s
    ax[0, 0].plot(s, rms / cvt, 'C0-o', lw=2, label='LOCO RMS / CV')
    ax[0, 0].plot(s, spread / cvt, 'k:', lw=1.4, label='HOD signal spread / CV')
    ax[0, 0].axhline(1, color='C3', ls='--', lw=1, label='cosmic variance (target)')
    ax[0, 0].set_yscale('log'); ax[0, 0].set_xlabel(r'$s\,[h^{-1}$Mpc]')
    ax[0, 0].set_ylabel('/ CV'); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title('(a) emulator error vs scale')
    # (b) pred vs true s^2 xi for example cosmologies
    examples = [c for c in ['c000', 'c130', 'c160'] if c in set(cosmo)]
    for c, col in zip(examples, ['C2', 'C0', 'C3']):
        idx = np.where(cosmo == c)[0]; i = idx[len(idx) // 2]
        ax[0, 1].plot(s, s**2 * Yt[i], col + '-o', ms=3, label=f'{c} truth')
        ax[0, 1].plot(s, s**2 * Ypt[i], col + '--', lw=2, label=f'{c} emul')
    ax[0, 1].axhline(0, color='grey', lw=0.5); ax[0, 1].set_xlabel(r'$s\,[h^{-1}$Mpc]')
    ax[0, 1].set_ylabel(r'$s^2\xi_0$'); ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title('(b) predicted vs true (held-out)')
    # (c) pred vs true scatter at a small-s bin
    b = 2                                                 # ~25 Mpc/h bin
    ax[1, 0].scatter(Yt[fisher, b], Ypt[fisher, b], s=8, c='C2', label='Fisher (LCDM-nbhd)')
    ax[1, 0].scatter(Yt[~fisher, b], Ypt[~fisher, b], s=8, c='C0', alpha=0.5, label='tier3 (broad)')
    lim = [min(Yt[:, b].min(), Ypt[:, b].min()), max(Yt[:, b].max(), Ypt[:, b].max())]
    ax[1, 0].plot(lim, lim, 'k--', lw=1); ax[1, 0].set_xlabel(f'true xi0 (s={s[b]:.0f})')
    ax[1, 0].set_ylabel('emulated xi0'); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title(f'(c) pred vs true at s={s[b]:.0f} Mpc/h')
    # (d) per-cosmology median RMS/CV
    cc = sorted(percos, key=lambda c: int(c[1:])); x = np.arange(len(cc))
    ax[1, 1].bar(x, [percos[c] for c in cc],
                 color=['C2' if int(c[1:]) < 130 else 'C0' for c in cc])
    ax[1, 1].axhline(1, color='C3', ls='--', lw=1)
    ax[1, 1].set_yscale('log'); ax[1, 1].set_xticks([]); ax[1, 1].set_xlabel('cosmology (green=Fisher, blue=tier3)')
    ax[1, 1].set_ylabel('median RMS/CV'); ax[1, 1].set_title('(d) per-cosmology floor')

    fig.suptitle(f'Emulator quality on {args.leg.replace("tpcf_","")} '
                 f'l{args.ell} (CV-weighted, LOCO)', y=1.0)
    fig.tight_layout()
    p = REPO / f'plots/emulator_tier3/leg_diagnostic_{args.leg.replace("tpcf_","")}_l{args.ell}.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

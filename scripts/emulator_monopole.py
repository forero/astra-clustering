#!/usr/bin/env python3
"""
Monopole-only emulator (forward model for inference) + cross-cosmology validation.

The data-vs-random split showed the MONOPOLES (ell=0, data and random) are the
inference-ready legs (~1.5x CV within-cosmology), while quadrupoles are
iteration-noise-limited.  This builds the monopole forward model xi0(theta) over
ALL available cosmologies (10 tier3 + 9 Fisher = 19) and validates it
leave-one-cosmology-out -- the regime that matters for inference (predicting at a
NEW cosmology), which is the honest gate on whether inference-now is meaningful
given the sparse cosmology axis.

Target: the 8 PRIMARY_STEMS, MONOPOLE only (8 legs x 15 bins = 120-D).
Model:  the MLP (emu.fit_ensemble) -- scales to ~1900 points better than a GP.

Outputs
  data/emulator_tier3/monopole_loco.npz   LOCO truth/pred/pred_std + residuals (for C_emu)
  plots/emulator_tier3/monopole_loco.png  per-cosmology LOCO floor (RMS/CV)

Usage (GPU node):  python scripts/emulator_monopole.py [--ensemble 5] [--epochs 2000]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu
from emulator_tier3_learning_curve import cv_box_per_column

REPO = Path(__file__).resolve().parents[1]


def load_legs(ells):
    """Stack tier3 + anchor; keep the requested multipole(s) of the 8 primary legs.
    Returns X, Y, Ynoise, cosmo, cv, s, and leg_blocks=[(stem,ell,slice),...]."""
    a = emu.load('dataset.npz'); b = emu.load('dataset_anchor.npz')
    mask, blocks = emu.select_targets(a, emu.PRIMARY_STEMS)
    sel_ell, sel_stem = a['ell'][mask], a['stem'][mask]
    keep = np.isin(sel_ell, ells)
    X = np.vstack([a['X'], b['X']])
    Y = np.vstack([a['Y'][:, mask], b['Y'][:, mask]])[:, keep]
    Yn = np.vstack([a['Ynoise'][:, mask], b['Ynoise'][:, mask]])[:, keep]
    cosmo = np.concatenate([a['cosmo'], b['cosmo']])
    cv = cv_box_per_column(blocks)[keep]
    nb = len(a['s'])
    leg_blocks = [(str(sel_stem[keep][k * nb]), int(sel_ell[keep][k * nb]),
                   slice(k * nb, (k + 1) * nb)) for k in range(Y.shape[1] // nb)]
    return X, Y, Yn, cosmo, cv, a['s'], leg_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--ells', default='0', help='"0" (monopole) or "0,2" (mono+quad)')
    args = ap.parse_args()
    ells = [int(x) for x in args.ells.split(',')]
    tag = 'monopole' if ells == [0] else 'monoquad'
    print(f'device={emu.DEVICE}  ensemble={args.ensemble}  ells={ells}  tag={tag}')

    X, Y, Yn, cosmo, cv, s, legs = load_legs(ells)
    nb = len(s)
    print(f'{len(Y)} runs, {len(set(cosmo))} cosmologies; {tag} target {Y.shape[1]}-D '
          f'({len(legs)} stem×ell blocks x {nb} bins)')

    cosmos = sorted(set(cosmo))
    Y_pred = np.zeros_like(Y); Y_pred_std = np.zeros_like(Y)
    per_cos = {}
    for c in cosmos:
        te = cosmo == c
        pm, ps, _ = emu.fit_ensemble(X[~te], Y[~te], Yn[~te], X[te],
                                     n_ens=args.ensemble, epochs=args.epochs,
                                     seed0=hash(c) % 9999)
        Y_pred[te] = pm; Y_pred_std[te] = ps
        per_cos[c] = np.median(np.sqrt(np.mean((pm - Y[te]) ** 2, 0)) / cv)

    def grp_median(keys):
        return np.median([per_cos[c] for c in cosmos if c in keys])
    fisher = {c for c in cosmos if int(c[1:]) < 130}
    tier3 = {c for c in cosmos if int(c[1:]) >= 130}
    print(f'\n=== {tag} LOCO (per-cosmology median RMS/CV) ===')
    print(f'  Fisher (LCDM-nbhd): {grp_median(fisher):.2f}xCV   '
          f'tier3 (broad): {grp_median(tier3):.2f}xCV')
    # per (stem, ell)
    rms_all = np.sqrt(np.mean((Y_pred - Y) ** 2, 0))
    for st, el, sl in legs:
        print(f'  {st:24s} l{el}: {np.median((rms_all/cv)[sl]):.2f}')

    out = REPO / f'data/emulator_tier3/{tag}_loco.npz'
    np.savez(out, s=s, Y=Y, Y_pred=Y_pred, Y_pred_std=Y_pred_std, cv=cv, cosmo=cosmo,
             stems=np.array([b[0] for b in legs]), ells_blk=np.array([b[1] for b in legs]),
             resid=Y_pred - Y)
    print(f'Saved {out}')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cc = sorted(per_cos, key=lambda c: int(c[1:])); x = np.arange(len(cc))
    colors = ['C0' if int(c[1:]) >= 130 else 'C2' for c in cc]
    ax.bar(x, [per_cos[c] for c in cc], color=colors)
    ax.axhline(1, color='k', ls='--', lw=1, label='cosmic variance')
    ax.axhline(grp_median(tier3), color='C0', ls=':', lw=1, label=f'tier3 med {grp_median(tier3):.1f}×')
    ax.axhline(grp_median(fisher), color='C2', ls=':', lw=1, label=f'Fisher med {grp_median(fisher):.1f}×')
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(cc, rotation=90, fontsize=6)
    ax.set_ylabel(f'{tag} LOCO RMS / CV')
    ax.set_title(f'{tag} emulator LOCO over 52 tier3 + 9 Fisher cosmologies\n'
                 '(blue = tier3 c130–c181, green = Fisher LCDM-neighbourhood)')
    ax.legend(fontsize=8); fig.tight_layout()
    p = REPO / f'plots/emulator_tier3/{tag}_loco.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

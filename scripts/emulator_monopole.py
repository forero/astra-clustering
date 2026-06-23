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


def load_all_monopole():
    """Stack tier3 + anchor; keep monopole columns of the 8 primary legs."""
    a = emu.load('dataset.npz'); b = emu.load('dataset_anchor.npz')
    mask, blocks = emu.select_targets(a, emu.PRIMARY_STEMS)
    ell = a['ell'][mask]
    mono = ell == 0                                        # monopole columns within the selection
    X = np.vstack([a['X'], b['X']])
    Y = np.vstack([a['Y'][:, mask], b['Y'][:, mask]])[:, mono]
    Yn = np.vstack([a['Ynoise'][:, mask], b['Ynoise'][:, mask]])[:, mono]
    cosmo = np.concatenate([a['cosmo'], b['cosmo']])
    # monopole blocks + CV
    cv_full = cv_box_per_column(blocks)
    cv = cv_full[mono]
    stems = [st for st, el, _ in blocks if el == 0]
    return X, Y, Yn, cosmo, cv, a['s'], stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  ensemble={args.ensemble}')

    X, Y, Yn, cosmo, cv, s, stems = load_all_monopole()
    nb = len(s); nleg = len(stems)
    print(f'{len(Y)} runs, {len(set(cosmo))} cosmologies; monopole target {Y.shape[1]}-D '
          f'({nleg} legs x {nb} bins)')

    cosmos = sorted(set(cosmo))
    Y_pred = np.zeros_like(Y); Y_pred_std = np.zeros_like(Y)
    per_cos = {}
    for c in cosmos:
        te = cosmo == c
        pm, ps, _ = emu.fit_ensemble(X[~te], Y[~te], Yn[~te], X[te],
                                     n_ens=args.ensemble, epochs=args.epochs,
                                     seed0=hash(c) % 9999)
        Y_pred[te] = pm; Y_pred_std[te] = ps
        rms = np.sqrt(np.mean((pm - Y[te]) ** 2, 0))
        per_cos[c] = np.median(rms / cv)
        print(f'  LOCO {c}: monopole median RMS/CV = {per_cos[c]:.2f}')

    rms_all = np.sqrt(np.mean((Y_pred - Y) ** 2, 0))
    print(f'\nOverall monopole LOCO floor: median RMS/CV = {np.median(rms_all/cv):.2f}')
    # per-leg
    for i, st in enumerate(stems):
        sl = slice(i * nb, (i + 1) * nb)
        print(f'  {st:26s}: {np.median((rms_all/cv)[sl]):.2f}')

    np.savez(REPO / 'data/emulator_tier3/monopole_loco.npz',
             s=s, Y=Y, Y_pred=Y_pred, Y_pred_std=Y_pred_std, cv=cv,
             cosmo=cosmo, stems=np.array(stems),
             resid=Y_pred - Y)                             # residuals for C_emu
    print(f'Saved {REPO / "data/emulator_tier3/monopole_loco.npz"}')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cc = sorted(per_cos, key=lambda c: int(c[1:]))
    x = np.arange(len(cc))
    colors = ['C0' if int(c[1:]) >= 130 else 'C2' for c in cc]   # tier3 vs Fisher
    ax.bar(x, [per_cos[c] for c in cc], color=colors)
    ax.axhline(1, color='k', ls='--', lw=1, label='cosmic variance (target)')
    ax.axhline(np.median(list(per_cos.values())), color='C3', ls=':', lw=1,
               label=f'median {np.median(list(per_cos.values())):.1f}×')
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(cc, rotation=45, fontsize=8)
    ax.set_ylabel('monopole LOCO RMS / CV')
    ax.set_title('Monopole emulator — leave-one-cosmology-out floor\n'
                 '(blue = tier3 c130–c181, green = Fisher LCDM-neighbourhood)')
    ax.legend(); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/monopole_loco.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

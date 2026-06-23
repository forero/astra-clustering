#!/usr/bin/env python3
"""
Within-cosmology HOD-interpolation floor test.

Disambiguates the flat learning curve: is the ~40x-cosmic-variance floor set by
sparse COSMOLOGY sampling (curable by the full campaign) or by something else
(coarse 15-bin data, 3-iteration target noise, model capacity, genuine
emulability -- NOT curable by more cosmologies)?

This trains and tests entirely WITHIN each pilot cosmology (zero cosmology
extrapolation): 5-fold over that cosmology's 50 HODs, predict the held-out HODs.
The cosmology inputs are constant within a cosmology, so the MLP learns only the
12-D HOD response -- the genuinely space-filling axis.

Read:
  * within-cosmology floor ~ few x CV  -> cosmology coverage IS the lever; the
    full campaign's density gain should pay off.
  * within-cosmology floor ~ tens x CV -> the limit is NOT cosmology sampling;
    more cosmologies won't help (need finer bins / more iterations).
Compare against the cross-cosmology LOCO floor (~40-70x CV from the learning curve).

Outputs
  data/emulator_tier3/within_cosmo_floor.npz
  plots/emulator_tier3/mlp_within_cosmo_floor.png

Usage (GPU node):  python scripts/emulator_tier3_within_cosmo.py [--folds 5] [--epochs 1500]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu
from emulator_tier3_learning_curve import cv_box_per_column, PRIORITY

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=1500)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  folds={args.folds}  epochs={args.epochs}')

    ds = emu.load('dataset.npz')
    mask, blocks = emu.select_targets(ds, emu.PRIMARY_STEMS)
    X, cosmo = ds['X'], ds['cosmo']
    Y, Ynoise = ds['Y'][:, mask], ds['Ynoise'][:, mask]
    cv = cv_box_per_column(blocks)
    pri = np.concatenate([np.r_[sl] for st, el, sl in blocks if st in PRIORITY])

    cosmos = sorted(set(cosmo))
    rng = np.random.default_rng(0)
    pri_floor, all_floor = {}, {}
    for c in cosmos:
        idx = np.where(cosmo == c)[0]
        perm = rng.permutation(len(idx))
        pred = np.zeros((len(idx), Y.shape[1]))
        for f in range(args.folds):                       # k-fold within this cosmology
            te_local = perm[f::args.folds]
            tr_local = np.setdiff1d(np.arange(len(idx)), te_local)
            tr, te = idx[tr_local], idx[te_local]
            pm, _, _ = emu.fit_ensemble(X[tr], Y[tr], Ynoise[tr], X[te],
                                        n_ens=1, epochs=args.epochs, seed0=hash(c) % 9999 + f)
            pred[te_local] = pm
        Yc = Y[idx]
        rms = np.sqrt(np.mean((pred - Yc) ** 2, 0))       # per-bin RMS over the 50 held-out HODs
        pri_floor[c] = np.median((rms / cv)[pri])
        all_floor[c] = np.median(rms / cv)
        print(f'{c}: within-cosmo floor  priority={pri_floor[c]:6.1f}xCV   all={all_floor[c]:6.1f}xCV')

    pv = np.array([pri_floor[c] for c in cosmos])
    av = np.array([all_floor[c] for c in cosmos])
    print(f'\nMEDIAN within-cosmology floor: priority={np.median(pv):.1f}xCV  all={np.median(av):.1f}xCV')
    print(f'(compare cross-cosmology LOCO floor ~40-70xCV; cosmic variance = 1)')

    np.savez(REPO / 'data/emulator_tier3/within_cosmo_floor.npz',
             cosmos=np.array(cosmos), pri=pv, all=av)

    # ---- figure: per-cosmology within floor vs the LOCO floor band & CV ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cosmos))
    ax.bar(x - 0.2, pv, 0.4, color='C0', label='priority random void/knot')
    ax.bar(x + 0.2, av, 0.4, color='C3', alpha=0.7, label='all target legs')
    ax.axhspan(40, 70, color='grey', alpha=0.2, label='cross-cosmology LOCO floor (~40-70×CV)')
    ax.axhline(1, color='k', lw=1, ls='--', label='cosmic variance (inference target)')
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(cosmos, rotation=45, fontsize=8)
    ax.set_ylabel('within-cosmology median RMS / CV(box)')
    ax.set_title('Within-cosmology HOD-interpolation floor\n'
                 '(below the grey band → cosmology coverage is the lever; '
                 'inside it → a non-cosmology floor)')
    ax.legend(fontsize=8); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/mlp_within_cosmo_floor.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

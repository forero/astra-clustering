#!/usr/bin/env python3
"""
HOD learning curve at fixed cosmology (c000) — does more HODs lower the
within-cosmology emulator floor?

c000 now has 100 full-box runs (50 original + 50 extra, select_extra_hods.py).
With a FIXED held-out test set, we train on K HODs for K = 10..75 and measure the
held-out error vs cosmic variance.  Reading:
  * still dropping at K=75  -> HODs are a real lever; the campaign's 100/cosmology
    (and maybe more) lower the ~8x within-cosmology floor.
  * plateaued by K~40-50    -> HODs are NOT the limiter; the floor is the coarse
    15-bin data or the MLP architecture (test finer bins / better model instead).

Reads dataset_anchor.npz (rebuild it first so c000 has 100 rows:
  python scripts/build_emulator_dataset.py).

Outputs
  data/emulator_tier3/c000_hod_curve.npz
  plots/emulator_tier3/mlp_c000_hod_curve.png

Usage (GPU node):  python scripts/emulator_c000_hod_curve.py [--reps 4] [--epochs 1500]
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
COSMO = 'c000'
N_TEST = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=1500)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  reps={args.reps}  epochs={args.epochs}')

    an = emu.load('dataset_anchor.npz')
    mask, blocks = emu.select_targets(an, emu.PRIMARY_STEMS)
    cmask = an['cosmo'] == COSMO
    X = an['X'][cmask]
    Y, Ynoise = an['Y'][cmask][:, mask], an['Ynoise'][cmask][:, mask]
    hod = an['hod'][cmask]
    cv = cv_box_per_column(blocks)
    pri = np.concatenate([np.r_[sl] for st, el, sl in blocks if st in PRIORITY])

    # fixed held-out test set; training pool = the rest.  With all 100 c000 runs
    # this lets K go past 50 -> directly tests whether the floor keeps dropping.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(hod))
    test = perm[:N_TEST]
    pool = perm[N_TEST:]
    print(f'{COSMO}: {len(hod)} runs; train pool = {len(pool)}, fixed test = {len(test)}')
    Ks = [k for k in (10, 20, 30, 40, 50, 60, 70) if k <= len(pool)]

    pri_m, pri_s, all_m = [], [], []
    for K in Ks:
        pv, av = [], []
        for r in range(args.reps):
            tr = rng.choice(pool, size=K, replace=False)
            pm, _, _ = emu.fit_ensemble(X[tr], Y[tr], Ynoise[tr], X[test],
                                        n_ens=1, epochs=args.epochs, seed0=10 * K + r)
            rms = np.sqrt(np.mean((pm - Y[test]) ** 2, 0))
            pv.append(np.median((rms / cv)[pri])); av.append(np.median(rms / cv))
        pri_m.append(np.median(pv)); pri_s.append(np.std(pv)); all_m.append(np.median(av))
        print(f'K={K:2d} train HODs: priority med(RMS/CV)={pri_m[-1]:5.1f}  all={all_m[-1]:5.1f}')

    Ks = np.array(Ks); pri_m = np.array(pri_m); pri_s = np.array(pri_s); all_m = np.array(all_m)
    np.savez(REPO / 'data/emulator_tier3/c000_hod_curve.npz',
             K=Ks, pri_m=pri_m, pri_s=pri_s, all_m=all_m)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(Ks, pri_m - pri_s, pri_m + pri_s, color='C0', alpha=0.2)
    ax.plot(Ks, pri_m, 'o-', color='C0', lw=2, label='priority random void/knot')
    ax.plot(Ks, all_m, 's-', color='C3', lw=1.5, alpha=0.7, label='all target legs')
    ax.axhline(1, color='grey', ls='--', lw=1, label='cosmic variance (inference target)')
    ax.axhline(8, color='C1', ls=':', lw=1, label='earlier ~8×CV floor (50 HODs)')
    ax.set_yscale('log')
    ax.set_xlabel('number of training HODs (c000; independent test = new runs)')
    ax.set_ylabel('held-out median RMS / CV(box)')
    ax.set_title(f'c000 within-cosmology HOD learning curve (test = {len(test)} new runs)\n'
                 'still falling at the largest K → more HODs help; flat → floor is bins/architecture')
    ax.legend(fontsize=8); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/mlp_c000_hod_curve.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

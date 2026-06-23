#!/usr/bin/env python3
"""
Learning curve vs NUMBER OF TRAINING COSMOLOGIES (pilot data).

Tests the hypothesis the whole full campaign rests on: that the pilot's LOCO
failure is sparse-cosmology EXTRAPOLATION, curable by adding cosmologies.  Using
only the 10 pilot cosmologies on disk, for each N_train in 1..9 we repeatedly
(a) pick a held-out test cosmology, (b) train on N_train of the other 9, (c)
measure held-out error.  If the curve is still falling steeply at N=9, the full
52-cosmology campaign should keep improving; if it has plateaued, that is a
warning to catch before spending weeks of overrun compute.

Metric = median over s-bins of  RMS / cosmic-variance(2 Gpc/h box)  -- the
inference-relevant yardstick (emulator error must drop below cosmic variance),
reported for the priority random void/knot legs and for all target legs.  At
N_train=9 this reduces to the standard LOCO measurement.

Reuses emulator_tier3_mlp (load, select_targets, fit_ensemble).  Single model per
fit (n_ens=1) -- we want the trend, not calibrated error bars.

Outputs
  data/emulator_tier3/learning_curve.npz
  plots/emulator_tier3/mlp_learning_curve_vs_ncosmo.png

Usage (GPU node):  python scripts/emulator_tier3_learning_curve.py [--reps 12] [--epochs 1500]
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu

REPO = Path(__file__).resolve().parents[1]
PRIORITY = ['tpcf_rand_q1', 'tpcf_rand_q4',
            'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4']


def cv_box_per_column(blocks):
    """Cosmic-variance std at the 2 Gpc/h box for each selected column, aligned to
    `blocks` (pooled mean-subtracted subboxes across the 9 Fisher cosmologies, /√64)."""
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    ncol = blocks[-1][2].stop
    cv = np.zeros(ncol)
    for st, el, sl in blocks:
        cols = []
        for t in tags:
            f = REPO / 'data' / t / f'subbox_multipoles_{st}.npz'
            if f.is_file():
                x = np.load(f)[f'xi{el}_all']             # (n_subbox, nbins) per-subbox
                cols.append(x - x.mean(0))                # remove each cosmology's per-bin mean
        X = np.vstack(cols)
        cv[sl] = X.std(0) / np.sqrt(64)
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=1500)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  reps={args.reps}  epochs={args.epochs}')

    ds = emu.load('dataset.npz')
    mask, blocks = emu.select_targets(ds, emu.PRIMARY_STEMS)
    X, cosmo = ds['X'], ds['cosmo']
    Y, Ynoise = ds['Y'][:, mask], ds['Ynoise'][:, mask]
    cv = cv_box_per_column(blocks)
    pri = np.concatenate([np.r_[sl] for st, el, sl in blocks if st in PRIORITY])
    s = ds['s']

    cosmos = sorted(set(cosmo))
    Ns = list(range(1, len(cosmos)))                      # 1..9
    rng = np.random.default_rng(0)
    res_pri = {N: [] for N in Ns}
    res_all = {N: [] for N in Ns}

    for N in Ns:
        for r in range(args.reps):
            tc = cosmos[rng.integers(len(cosmos))]        # held-out test cosmology
            pool = [c for c in cosmos if c != tc]
            train_cos = list(rng.choice(pool, size=N, replace=False))
            trm = np.isin(cosmo, train_cos)
            te = cosmo == tc
            pm, _, _ = emu.fit_ensemble(X[trm], Y[trm], Ynoise[trm], X[te],
                                        n_ens=1, epochs=args.epochs,
                                        seed0=1000 * N + r)
            rms = np.sqrt(np.mean((pm - Y[te]) ** 2, 0))
            res_pri[N].append(np.median((rms / cv)[pri]))
            res_all[N].append(np.median(rms / cv))
        print(f'N={N}: priority med(RMS/CV)={np.mean(res_pri[N]):.1f} '
              f'all={np.mean(res_all[N]):.1f}')

    pri_m = np.array([np.mean(res_pri[N]) for N in Ns])
    pri_s = np.array([np.std(res_pri[N]) for N in Ns])
    all_m = np.array([np.mean(res_all[N]) for N in Ns])
    all_s = np.array([np.std(res_all[N]) for N in Ns])

    out = REPO / 'data/emulator_tier3/learning_curve.npz'
    np.savez(out, N=np.array(Ns), pri_mean=pri_m, pri_std=pri_s,
             all_mean=all_m, all_std=all_s,
             pri_raw=np.array([res_pri[N] for N in Ns]),
             all_raw=np.array([res_all[N] for N in Ns]))
    print(f'Saved {out}')

    fig, ax = plt.subplots(figsize=(7, 5))
    for m, sd, lab, c in [(pri_m, pri_s, 'priority random void/knot', 'C0'),
                          (all_m, all_s, 'all target legs', 'C3')]:
        ax.plot(Ns, m, 'o-', color=c, lw=2, label=lab)
        ax.fill_between(Ns, m - sd, m + sd, color=c, alpha=0.2)
    ax.axhline(1, color='grey', lw=1, ls='--', label='cosmic variance (inference target)')
    ax.set_yscale('log'); ax.set_xlabel('number of training cosmologies')
    ax.set_ylabel('held-out median  RMS / cosmic-variance(box)')
    ax.set_title('Emulator learning curve vs cosmology sampling (pilot)\n'
                 'N=9 = LOCO; full campaign reaches N=51')
    ax.legend(); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/mlp_learning_curve_vs_ncosmo.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

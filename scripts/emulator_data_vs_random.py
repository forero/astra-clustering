#!/usr/bin/env python3
"""
Data-vs-random emulability split.

The binning proxy showed the floor is limited by intrinsic scatter, not resolution,
and pointed at the RANDOM void/knot legs (RMS/spread ~0.7).  This breaks the
within-cosmology emulator floor down PER LEG and PER MULTIPOLE so we can see
whether the DATA legs emulate much better than the RANDOM ones (and whether it is
specifically the random QUADRUPOLES, per the old "random quads are noise-dominated"
finding).  If so, the inference vector should lean on data quantiles.

Same protocol/model as emulator_bakeoff: PCA+GP (cov-weighted, amplitude-factored),
5-fold WITHIN each of the 10 tier3 cosmologies, aggregate.  Reports both RMS/CV
(inference metric) and RMS/spread (CV-noise-free modelling quality).

Outputs
  data/emulator_tier3/data_vs_random.npz
  plots/emulator_tier3/data_vs_random.png

Usage (login node OK):  python scripts/emulator_data_vs_random.py [--ncomp 8]
"""
import argparse, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import emulator_tier3_mlp as emu
from emulator_tier3_learning_curve import cv_box_per_column
from emulator_bakeoff import pca_reg_predict, gp_factory

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ncomp', type=int, default=8)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    ds = emu.load('dataset.npz')
    mask, blocks = emu.select_targets(ds, emu.PRIMARY_STEMS)
    Xh = ds['X'][:, 8:]; Y = ds['Y'][:, mask]; cosmo = ds['cosmo']
    cv = cv_box_per_column(blocks)
    cosmos = sorted(set(cosmo))

    # accumulate per-(stem,ell) block: lists of per-cosmology median RMS/CV & RMS/spread
    agg = {(st, el): {'cv': [], 'sp': []} for st, el, _ in blocks}
    rng = np.random.default_rng(0)
    for c in cosmos:
        idx = np.where(cosmo == c)[0]; perm = rng.permutation(len(idx))
        pred = np.zeros((len(idx), Y.shape[1]))
        for f in range(args.folds):                       # joint PCA+GP over all legs
            te = perm[f::args.folds]; tr = np.setdiff1d(np.arange(len(idx)), te)
            pred[te] = pca_reg_predict(Xh[idx[tr]], Y[idx[tr]], Xh[idx[te]],
                                       cv, args.ncomp, gp_factory())
        Yc = Y[idx]
        rms = np.sqrt(np.mean((pred - Yc) ** 2, 0))
        spread = Yc.std(0) + 1e-30
        for st, el, sl in blocks:
            agg[(st, el)]['cv'].append(np.median((rms / cv)[sl]))
            agg[(st, el)]['sp'].append(np.median((rms / spread)[sl]))

    labels, fam, cvv, spv = [], [], [], []
    for st, el, _ in blocks:
        labels.append(f'{st.replace("tpcf_","")} ℓ{el}')
        fam.append('random' if 'rand' in st else 'data')
        cvv.append(np.median(agg[(st, el)]['cv']))
        spv.append(np.median(agg[(st, el)]['sp']))
    cvv, spv = np.array(cvv), np.array(spv)
    fam = np.array(fam)

    print('leg                          RMS/CV   RMS/spread')
    for lab, a, b in zip(labels, cvv, spv):
        print(f'  {lab:26s} {a:6.1f}     {b:5.3f}')
    for f in ('data', 'random'):
        m = fam == f
        print(f'  -- {f:6s} median:        {np.median(cvv[m]):6.1f}     {np.median(spv[m]):5.3f}')
        for ell, tag in [(0, 'ℓ0'), (2, 'ℓ2')]:
            me = m & np.array([f'ℓ{ell}' in l for l in labels])
            print(f'       {f} {tag}:            {np.median(cvv[me]):6.1f}     {np.median(spv[me]):5.3f}')

    np.savez(REPO / 'data/emulator_tier3/data_vs_random.npz',
             labels=np.array(labels), family=fam, rms_cv=cvv, rms_spread=spv)

    # ---- figure: per-leg bars, data (blue) vs random (red), two metrics ----
    x = np.arange(len(labels)); colors = ['C0' if f == 'data' else 'C3' for f in fam]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5))
    a0.bar(x, cvv, color=colors); a0.axhline(1, color='k', ls='--', lw=1, label='cosmic variance')
    a0.set_yscale('log'); a0.set_ylabel('within-cosmology RMS / CV'); a0.set_title('inference metric (RMS/CV)')
    a1.bar(x, spv, color=colors); a1.axhline(1, color='grey', ls=':', lw=1)
    a1.set_ylabel('RMS / signal spread'); a1.set_title('modelling quality (RMS/spread; lower=better, <1 beats the mean)')
    for ax in (a0, a1):
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    from matplotlib.patches import Patch
    a0.legend(handles=[Patch(color='C0', label='data legs'), Patch(color='C3', label='random legs')],
              fontsize=9)
    fig.suptitle('Data vs random emulability (within-cosmology, PCA+GP)', y=1.02)
    fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/data_vs_random.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Zero-compute proxy for "does finer binning help the emulator?"

We can't make the existing data FINER (binning is lossy), but we can make it
COARSER and read the TREND.  Merge adjacent s-bins of the cached 15-bin data into
n_bins in {3,5,8,10,12,15}, rebin the subbox CV the same way, and measure the
within-cosmology emulator floor (PCA+GP, the bake-off challenger) at each.

Read:
  * floor (RMS/CV) DROPS as n_bins 5->15  -> the emulator IS resolution-sensitive;
    finer (a pipeline recompute) is very likely to keep helping -> worth the cost.
  * floor FLAT across 5->15               -> resolution isn't the lever in this
    range; finer probably won't help (caveat: can't probe <10 Mpc/h this way).

Protocol matches emulator_tier3_within_cosmo / emulator_bakeoff: 5-fold WITHIN each
of the 10 tier3 cosmologies, aggregate.  CPU-only, ~minutes.

Outputs
  data/emulator_tier3/binning_proxy.npz
  plots/emulator_tier3/binning_proxy.png

Usage (login node OK):  python scripts/binning_proxy.py [--ncomp 8]
"""
import argparse, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import emulator_tier3_mlp as emu
from scale_information import subbox_block
from emulator_bakeoff import pca_reg_predict, gp_factory, PRIORITY

REPO = Path(__file__).resolve().parents[1]
S_MAX = 150.0


def rebin(Y, blocks, n_new):
    """Rebin each (stem,ell) 15-bin block to n_new contiguous-averaged bins.
    Returns rebinned Y, rebinned CV std, priority column mask, new bin centres."""
    Yr_cols, cv_cols, pri, col = [], [], [], 0
    for st, el, sl in blocks:
        groups = np.array_split(np.arange(sl.stop - sl.start), n_new)
        Yblk = Y[:, sl]
        Yr_cols.append(np.column_stack([Yblk[:, g].mean(1) for g in groups]))
        Xsb = subbox_block(st, el)                          # (npooled, 15)
        Xsb_r = np.column_stack([Xsb[:, g].mean(1) for g in groups])
        cv_cols.append(Xsb_r.std(0) / np.sqrt(64.0))
        if st in PRIORITY:
            pri += list(range(col, col + n_new))
        col += n_new
    return np.hstack(Yr_cols), np.concatenate(cv_cols), np.array(pri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ncomp', type=int, default=8)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    ds = emu.load('dataset.npz')
    mask, blocks = emu.select_targets(ds, emu.PRIMARY_STEMS)
    Xh = ds['X'][:, 8:]; Y = ds['Y'][:, mask]; cosmo = ds['cosmo']
    cosmos = sorted(set(cosmo))

    nbins_list = [3, 5, 8, 10, 12, 15]
    pri_floor, all_floor, pri_spread = [], [], []
    for n_new in nbins_list:
        Yr, cv, pri = rebin(Y, blocks, n_new)
        rng = np.random.default_rng(0)
        pv, av, pvs = [], [], []
        for c in cosmos:
            idx = np.where(cosmo == c)[0]; perm = rng.permutation(len(idx))
            pred = np.zeros((len(idx), Yr.shape[1]))
            for f in range(args.folds):
                te = perm[f::args.folds]; tr = np.setdiff1d(np.arange(len(idx)), te)
                pred[te] = pca_reg_predict(Xh[idx[tr]], Yr[idx[tr]], Xh[idx[te]],
                                           cv, args.ncomp, gp_factory())
            rms = np.sqrt(np.mean((pred - Yr[idx]) ** 2, 0))
            spread = Yr[idx].std(0) + 1e-30               # CV-noise-free denominator
            pv.append(np.median((rms / cv)[pri])); av.append(np.median(rms / cv))
            pvs.append(np.median((rms / spread)[pri]))
        pri_floor.append(np.median(pv)); all_floor.append(np.median(av))
        pri_spread.append(np.median(pvs))
        print(f'n_bins={n_new:2d} (~{S_MAX/n_new:4.0f} Mpc/h):  '
              f'priority {pri_floor[-1]:5.2f}xCV   all {all_floor[-1]:5.2f}xCV   '
              f'priority {pri_spread[-1]:5.3f}xSPREAD')

    nb = np.array(nbins_list)
    pri_floor = np.array(pri_floor); all_floor = np.array(all_floor); pri_spread = np.array(pri_spread)
    np.savez(REPO / 'data/emulator_tier3/binning_proxy.npz',
             nbins=nb, pri=pri_floor, all=all_floor, pri_spread=pri_spread)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))
    a0.plot(nb, pri_floor, 'o-', color='C0', lw=2, label='priority random void/knot')
    a0.plot(nb, all_floor, 's-', color='C3', lw=1.5, label='all target legs')
    a0.axhline(1, color='grey', ls='--', lw=1, label='cosmic variance')
    a0.set_ylabel('RMS / CV'); a0.set_title('RMS / CV  (inference metric; CV from 64 subboxes)')
    a1.plot(nb, pri_spread, 'o-', color='C0', lw=2, label='priority random void/knot')
    a1.set_ylabel('RMS / signal spread')
    a1.set_title('RMS / SPREAD  (CV-estimate-noise-free)\nupturn at 15 here = real; gone = CV artifact')
    for ax in (a0, a1):
        ax.set_xlabel('number of s-bins over 0–150 Mpc/h  (right = finer)')
        sec = ax.secondary_xaxis('top', functions=(lambda x: S_MAX / np.where(x == 0, 1, x),
                                                   lambda w: S_MAX / np.where(w == 0, 1, w)))
        sec.set_xlabel('approx bin width [Mpc/h]'); ax.legend(fontsize=8)
    fig.suptitle('Coarsening-trend proxy: does the emulator floor improve with resolution?', y=1.02)
    fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/binning_proxy.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

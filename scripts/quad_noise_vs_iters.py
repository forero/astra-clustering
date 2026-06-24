#!/usr/bin/env python3
"""
ZERO-COMPUTE test: how does the ASTRA quadrupole noise scale with iterations?

Every run already stores xi2_std = the iteration-to-iteration scatter of the
quadrupole (the ASTRA random-realisation noise, per iteration). The noise on an
N-iteration MEAN is sigma_1/sqrt(N) with sigma_1 = xi2_std. So from the existing
3-iteration runs we can predict the quad noise at any N for free, and compare it to
cosmic variance (the threshold below which iterations stop limiting the emulator).

We also cross-check the sigma_1 estimate against the 10-iteration c000 runs
(data/fullbox_iter10) -- if the per-iteration scatter matches, the 1/sqrt(N) law
holds (as the earlier iteration experiment found).

Outputs  plots/emulator_tier3/quad_noise_vs_iters.png

Usage (login node OK):  python scripts/quad_noise_vs_iters.py
"""
import glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
LEGS = ['tpcf_full_data', 'tpcf_cross_full_rand_q1', 'tpcf_rand_q1', 'tpcf_rand_q4',
        'tpcf_cross_full_rand_q4', 'tpcf_data_q4']
NITER = np.arange(1, 26)


def mean_iter_scatter(root, leg, ell=2, maxruns=300):
    """Average per-iteration scatter (xi{ell}_std) over runs in `root`."""
    files = sorted(glob.glob(str(REPO / f'data/{root}/c*_hod*/fullbox_multipoles_{leg}.npz')))[:maxruns]
    sig = [np.load(f)[f'xi{ell}_std'] for f in files]
    return np.mean(sig, 0), len(sig)


def cv_box(leg, ell=2):
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    cols = []
    for t in tags:
        f = REPO / 'data' / t / f'subbox_multipoles_{leg}.npz'
        if f.is_file():
            x = np.load(f)[f'xi{ell}_all']; cols.append(x - x.mean(0))
    return np.vstack(cols).std(0) / np.sqrt(64.0)


def main():
    s = np.load(glob.glob(str(REPO / 'data/fullbox_tier3/c*_hod*/fullbox_multipoles_tpcf_full_data.npz'))[0])['s']
    small = s < 60                                        # the information-bearing scales

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    print(f'{"leg":26s} quad iter-noise/CV @N=3  @N=10  @N=25   N for <1')
    for leg in LEGS:
        sig1, n = mean_iter_scatter('fullbox_tier3', leg, ell=2)   # per-iteration scatter
        cv = cv_box(leg, ell=2)
        # median over small scales of (sigma_1/sqrt(N))/CV
        ratio = np.array([np.median((sig1 / np.sqrt(N) / cv)[small]) for N in NITER])
        ncross = NITER[np.argmax(ratio < 1)] if np.any(ratio < 1) else np.inf
        ax[0].plot(NITER, ratio, 'o-', ms=3, label=leg.replace('tpcf_', ''))
        i3, i10, i25 = ratio[2], ratio[9], ratio[24]
        print(f'{leg:26s} {i3:18.1f} {i10:6.1f} {i25:6.1f}   {ncross}')
    ax[0].axhline(1, color='k', ls='--', lw=1, label='= cosmic variance')
    ax[0].axvline(3, color='grey', ls=':', lw=1); ax[0].axvline(10, color='grey', ls=':', lw=1)
    ax[0].set_xlabel('ASTRA iterations N'); ax[0].set_ylabel('quad iter-noise / CV (median, s<60)')
    ax[0].set_yscale('log'); ax[0].legend(fontsize=7)
    ax[0].set_title('(a) quadrupole iteration-noise vs N (1/sqrt(N))\nbelow 1 = no longer limiting')

    # (b) empirical 1/sqrt(N) cross-check: sigma_1 from 3-iter vs 10-iter c000
    have10 = glob.glob(str(REPO / 'data/fullbox_iter10/c000_hod*/fullbox_multipoles_tpcf_rand_q4.npz'))
    if have10:
        for leg, col in zip(['tpcf_rand_q4', 'tpcf_cross_full_rand_q1'], ['C0', 'C1']):
            s3, _ = mean_iter_scatter('fullbox_tier3', leg, ell=2)
            s10 = np.mean([np.load(f)[f'xi2_std']
                           for f in glob.glob(str(REPO / f'data/fullbox_iter10/c000_hod*/fullbox_multipoles_{leg}.npz'))], 0)
            ax[1].plot(s, s3, col + '-', label=f'{leg.replace("tpcf_","")} sigma_1 (3-iter runs)')
            ax[1].plot(s, s10, col + '--', label=f'{leg.replace("tpcf_","")} sigma_1 (10-iter runs)')
        ax[1].set_xlabel(r'$s\,[h^{-1}$Mpc]'); ax[1].set_ylabel('per-iteration scatter xi2_std')
        ax[1].set_yscale('log'); ax[1].legend(fontsize=7)
        ax[1].set_title('(b) per-iteration scatter: 3-iter vs 10-iter c000\n(overlap => i.i.d., 1/sqrt(N) holds)')
    fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/quad_noise_vs_iters.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

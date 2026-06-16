#!/usr/bin/env python3
"""
Tier-0 comparison: subbox-paired vs full-box derivatives, at full-box volume.

Both derivative sources are evaluated against the *same* covariance — the 64
c000 fiducial subboxes, scaled to the full 2000 Mpc/h box volume (C_subbox/64,
Hartlap-corrected on the sample inverse).  Only the derivative numerator and
its noise model differ:

  subbox   : derivative_{param}.npz       (mean of 64 paired subbox diffs);
             noise bias = trace(Cinv @ cov(per-subbox diffs)/N_SB)  — full
             bin-bin noise covariance.
  full-box : derivative_fullbox_{param}.npz (one phase-matched 2000-box diff);
             noise bias = trace(Cinv @ diag(noisevar))  — diagonal noise from
             the N_ITER ASTRA-random iterations (cosmic variance cancels in the
             phase-matched difference).

For each parameter and data vector it prints sigma(theta) at full-box volume
under both derivatives and the derivative-noise fraction (bias/F), then draws,
per parameter, the implied Gaussians for the full-auto vector under each method.

Output: plots/derivatives/fisher_fullbox_compare_{param}.png
        plots/derivatives/fisher_fullbox_compare_summary.png

Usage (any node):
  python scripts/plot_fisher_fullbox_compare.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
DER_DIR   = DATA_DIR / 'derivatives'
PLOT_DIR  = REPO_ROOT / 'plots' / 'derivatives'

N_SB     = 64
FID_TAG  = 'c000_hod484'
VOL_FAC  = N_SB          # full box = 64 subbox volumes -> covariance / 64

PARAMS = {
    'lnwb': dict(fid=0.02237, label=r'\omega_b', log=True),
    'lnwc': dict(fid=0.1200,  label=r'\omega_{cdm}', log=True),
    'ns':   dict(fid=0.9649,  label=r'n_s', log=False),
    'lns8': dict(fid=0.807952, label=r'\sigma_8', log=True),
}

COMBO_REBIN = 2
VECTORS = [
    ('full auto',      [('tpcf_full_data', (0,), 1)]),
    ('full x data Q4', [('tpcf_cross_full_data_q4', (0,), 1)]),
    ('full x rand Q1', [('tpcf_cross_full_rand_q1', (0,), 1)]),
    ('concat x2',      [('tpcf_full_data',          (0,), COMBO_REBIN),
                        ('tpcf_cross_full_data_q4', (0,), COMBO_REBIN),
                        ('tpcf_cross_full_rand_q1', (0,), COMBO_REBIN)]),
]


def rebin(arr, k):
    if k == 1:
        return arr
    arr = np.atleast_2d(arr)
    n   = arr.shape[1]
    out = np.column_stack([arr[:, i:i + k].mean(axis=1) for i in range(0, n, k)])
    return out[0] if out.shape[0] == 1 else out


def rebin_var(v, k):
    """Diagonal variance of a k-bin average: (1/k^2) sum of the k variances."""
    if k == 1:
        return v
    n = v.shape[0]
    return np.array([v[i:i + k].sum() / k ** 2 for i in range(0, n, k)])


def cov_fullbox(pieces):
    """Hartlap-corrected inverse covariance at full-box volume, + nbins."""
    X_parts = []
    for stem, ells, k in pieces:
        c0 = np.load(DATA_DIR / FID_TAG / f'subbox_multipoles_{stem}.npz')
        for ell in ells:
            X_parts.append(rebin(c0[f'xi{ell}_all'], k))
    X    = np.hstack(X_parts)
    nb   = X.shape[1]
    hart = (N_SB - nb - 2) / (N_SB - 1)
    # covariance of the full-box-volume vector = C_subbox / VOL_FAC
    Cinv = hart * VOL_FAC * np.linalg.inv(np.cov(X, rowvar=False))
    return Cinv, nb


def sigma_subbox(der, pieces, Cinv):
    d_parts, Ds_parts = [], []
    for stem, ells, k in pieces:
        for ell in ells:
            d_parts.append(rebin(der[f'{stem}_dxi{ell}'], k))
            Ds_parts.append(rebin(der[f'{stem}_dxi{ell}_all'], k))
    d  = np.concatenate(d_parts)
    Ds = np.hstack(Ds_parts)
    F    = d @ Cinv @ d
    bias = np.trace(Cinv @ (np.cov(Ds, rowvar=False) / N_SB))
    if F <= 0 or F - bias <= 0:
        return None
    return 1.0 / np.sqrt(F - bias), bias / F


def sigma_fullbox(der, pieces, Cinv):
    d_parts, v_parts = [], []
    for stem, ells, k in pieces:
        for ell in ells:
            d_parts.append(rebin(der[f'{stem}_dxi{ell}'], k))
            v_parts.append(rebin_var(der[f'{stem}_dxi{ell}_noisevar'], k))
    d = np.concatenate(d_parts)
    v = np.concatenate(v_parts)
    F    = d @ Cinv @ d
    bias = np.trace(Cinv @ np.diag(v))
    if F <= 0 or F - bias <= 0:
        return None
    return 1.0 / np.sqrt(F - bias), bias / F


def to_phys(sig_ln, cfg):
    return cfg['fid'] * sig_ln if cfg['log'] else sig_ln


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}   # param -> (sig_sb_fullauto, sig_fb_fullauto)

    for param, cfg in PARAMS.items():
        f_sb = DER_DIR / f'derivative_{param}.npz'
        f_fb = DER_DIR / f'derivative_fullbox_{param}.npz'
        if not (f_sb.is_file() and f_fb.is_file()):
            print(f'Skipping {param}: need both {f_sb.name} and {f_fb.name}')
            continue
        der_sb = np.load(f_sb)
        der_fb = np.load(f_fb)

        print(f'\n=== {param}  (sigma at full 2000 Mpc/h box volume) ===')
        print(f'  {"vector":14s} {"sigma_subbox":>14s} {"sigma_fullbox":>14s} '
              f'{"noise_sb":>9s} {"noise_fb":>9s}')
        rows = []
        for name, pieces in VECTORS:
            Cinv, _ = cov_fullbox(pieces)
            r_sb = sigma_subbox(der_sb, pieces, Cinv)
            r_fb = sigma_fullbox(der_fb, pieces, Cinv)
            s_sb = to_phys(r_sb[0], cfg) if r_sb else np.nan
            s_fb = to_phys(r_fb[0], cfg) if r_fb else np.nan
            n_sb = r_sb[1] if r_sb else np.nan
            n_fb = r_fb[1] if r_fb else np.nan
            rows.append((name, s_sb, s_fb, n_sb, n_fb))
            print(f'  {name:14s} {s_sb:14.5g} {s_fb:14.5g} '
                  f'{n_sb:9.0%} {n_fb:9.0%}')
            if name == 'full auto':
                summary[param] = (s_sb, s_fb)

        # per-parameter Gaussian figure for the full-auto vector
        Cinv, _ = cov_fullbox(VECTORS[0][1])
        r_sb = sigma_subbox(der_sb, VECTORS[0][1], Cinv)
        r_fb = sigma_fullbox(der_fb, VECTORS[0][1], Cinv)
        fid  = cfg['fid']
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        width = max(to_phys(r_sb[0], cfg), to_phys(r_fb[0], cfg)) * 3.5
        x = np.linspace(fid - width, fid + width, 800)
        for r, color, lab in ((r_sb, '#377eb8', 'subbox-paired deriv'),
                              (r_fb, '#e41a1c', 'full-box deriv')):
            if r is None:
                continue
            s = to_phys(r[0], cfg)
            ax.plot(x, np.exp(-0.5 * ((x - fid) / s) ** 2), color=color, lw=2,
                    label=rf'{lab}  ($\sigma$={s:.2g}, noise {r[1]:.0%})')
        ax.axvline(fid, color='k', lw=0.8, ls='--')
        ax.set_xlabel(rf'${cfg["label"]}$')
        ax.set_ylabel('likelihood (peak-normalised)')
        ax.set_title(rf'Full-auto Fisher for ${cfg["label"]}$ at full-box volume',
                     fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        fig.tight_layout()
        path = PLOT_DIR / f'fisher_fullbox_compare_{param}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {path}')

    # summary bar chart of the improvement factor (full auto)
    if summary:
        params = list(summary)
        ratio  = [summary[p][0] / summary[p][1] for p in params]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(range(len(params)),
               ratio, color='#4daf4a')
        ax.axhline(1, color='k', lw=0.8, ls='--')
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([f'${PARAMS[p]["label"]}$' for p in params])
        ax.set_ylabel(r'$\sigma_{\rm subbox}/\sigma_{\rm full\text{-}box}$ '
                      '(full auto)')
        ax.set_title('Tier-0 improvement: full-box vs subbox-paired derivative',
                     fontsize=11)
        for i, r in enumerate(ratio):
            ax.text(i, r + 0.02, f'{r:.2f}x', ha='center', fontsize=9)
        fig.tight_layout()
        path = PLOT_DIR / 'fisher_fullbox_compare_summary.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'\nSaved {path}')


if __name__ == '__main__':
    main()

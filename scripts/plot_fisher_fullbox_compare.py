#!/usr/bin/env python3
"""
Tier-0 comparison: subbox-paired vs full-box derivatives, at full-box volume.

Both derivative sources are evaluated against the *same* covariance — the 64
c000 fiducial subboxes, scaled to the full 2000 Mpc/h box volume (C_subbox/64,
Hartlap-corrected on the sample inverse).  Only the derivative numerator and
its noise model differ:

  subbox          : derivative_{param}.npz        (mean of 64 paired subbox
                    diffs); noise bias = trace(Cinv @ cov(per-subbox diffs)/N_SB).
  full-box raw    : derivative_fullbox_{param}.npz (one phase-matched 2000-box
                    diff); noise bias = trace(Cinv @ diag(noisevar)) — diagonal
                    noise from the N_ITER iterations (cosmic variance cancels).
  full-box HOD-corr: derivative_hodcorr_{param}.npz (full-box diff with the
                    Tier-1 HOD contamination subtracted); same diagonal noise.
                    This is the current best estimate; absent until
                    compute_hod_derivatives.py has run.

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


METHODS = [   # (label, derivative-file template, sigma fn, colour)
    ('subbox-paired',     'derivative_{}.npz',          sigma_subbox, '#377eb8'),
    ('full-box raw',      'derivative_fullbox_{}.npz',   sigma_fullbox, '#e41a1c'),
    ('full-box HOD-corr', 'derivative_hodcorr_{}.npz',   sigma_fullbox, '#4daf4a'),
]


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}   # param -> {method label: sigma (full auto)}

    for param, cfg in PARAMS.items():
        ders = {}
        for lab, tmpl, _, _ in METHODS:
            f = DER_DIR / tmpl.format(param)
            if f.is_file():
                ders[lab] = np.load(f)
        if 'full-box raw' not in ders:
            print(f'Skipping {param}: no full-box derivative')
            continue

        print(f'\n=== {param}  (sigma at full 2000 Mpc/h box volume) ===')
        hdr = f'  {"vector":14s}' + ''.join(f'{lab:>20s}' for lab, *_ in METHODS)
        print(hdr)
        summary[param] = {}
        for name, pieces in VECTORS:
            Cinv, _ = cov_fullbox(pieces)
            cells = f'  {name:14s}'
            for lab, _, sig_fn, _ in METHODS:
                if lab not in ders:
                    cells += f'{"-":>20s}'; continue
                r = sig_fn(ders[lab], pieces, Cinv)
                if r is None:
                    cells += f'{"nan":>20s}'
                else:
                    s = to_phys(r[0], cfg)
                    cells += f'{f"{s:.4g} ({r[1]:.0%})":>20s}'
                    if name == 'full auto':
                        summary[param][lab] = s
            print(cells)

        # per-parameter Gaussian figure for the full-auto vector, all methods
        Cinv, _ = cov_fullbox(VECTORS[0][1])
        fid = cfg['fid']
        results = []
        for lab, _, sig_fn, color in METHODS:
            if lab not in ders:
                continue
            r = sig_fn(ders[lab], VECTORS[0][1], Cinv)
            if r is not None:
                results.append((lab, to_phys(r[0], cfg), r[1], color))
        if not results:
            continue
        width = max(s for _, s, _, _ in results) * 3.5
        x = np.linspace(fid - width, fid + width, 800)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for lab, s, noise, color in results:
            ax.plot(x, np.exp(-0.5 * ((x - fid) / s) ** 2), color=color, lw=2,
                    label=rf'{lab}  ($\sigma$={s:.2g}, noise {noise:.0%})')
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

    # summary: grouped bar chart of sigma per method, per parameter (full auto)
    if summary:
        params = list(summary)
        labels = [lab for lab, *_ in METHODS]
        colors = {lab: c for lab, _, _, c in METHODS}
        fig, ax = plt.subplots(figsize=(8, 4.5))
        w = 0.27
        for j, lab in enumerate(labels):
            vals = [summary[p].get(lab, np.nan) for p in params]
            # normalise each parameter to its full-box-raw sigma for comparability
            norm = [summary[p].get('full-box raw', np.nan) for p in params]
            rel  = [v / n if (v == v and n == n) else np.nan
                    for v, n in zip(vals, norm)]
            ax.bar([i + (j - 1) * w for i in range(len(params))], rel, w,
                   label=lab, color=colors[lab])
        ax.axhline(1, color='k', lw=0.8, ls='--')
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([f'${PARAMS[p]["label"]}$' for p in params])
        ax.set_ylabel(r'$\sigma\,/\,\sigma_{\rm full\text{-}box\ raw}$ (full auto)')
        ax.set_title('Fisher σ per derivative method (full-auto vector)',
                     fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = PLOT_DIR / 'fisher_fullbox_compare_summary.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'\nSaved {path}')


if __name__ == '__main__':
    main()

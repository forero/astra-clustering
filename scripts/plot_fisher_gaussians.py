#!/usr/bin/env python3
"""
Gaussian visualisation of per-data-vector Fisher uncertainties.

Uses the **HOD-corrected full-box derivatives** (derivative_hodcorr_{param}.npz,
from compute_hod_derivatives.py) — the current best estimate, with the Tier-1
HOD contamination subtracted.  For each parameter and data vector it computes a
one-parameter Fisher at the full 2000 Mpc/h box volume:

  * covariance from the 64 c000 fiducial subboxes, scaled to full-box volume
    (C_subbox / 64), Hartlap-corrected;
  * derivative noise diagonal, from the N_ITER full-box iterations (cosmic
    variance cancels in the phase-matched ± difference).

Only data vectors whose derivative-noise bias is below NOISE_MAX of the Fisher
information are shown.  Gaussians are drawn centred on the fiducial parameter.

Output: plots/derivatives/fisher_gaussians_{param}.png
        plots/derivatives/fisher_covariances.png

Usage (any node):
  python scripts/plot_fisher_gaussians.py
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

N_SB      = 64
VOL_FAC   = N_SB     # full 2000 Mpc/h box = 64 subbox volumes -> covariance / 64
N_Q       = 4
NOISE_MAX = 0.10     # max derivative-noise fraction of F to include a vector
FID_TAG   = 'c000_hod484'

PARAMS = {
    'lnwb': dict(fid=0.02237, plus=0.02282, minus=0.02193,
                 label=r'\omega_b', log=True),
    'lnwc': dict(fid=0.1200,  plus=0.1240,  minus=0.1161,
                 label=r'\omega_{cdm}', log=True),
    'ns':   dict(fid=0.9649,  plus=0.9749,  minus=0.9549,
                 label=r'n_s', log=False),
    'lns8': dict(fid=0.807952, plus=0.824120, minus=0.792107,
                 label=r'\sigma_8', log=True),
}


def rebin(arr, k):
    if k == 1:
        return arr
    arr  = np.atleast_2d(arr)
    n    = arr.shape[1]
    out  = np.column_stack([arr[:, i:i + k].mean(axis=1) for i in range(0, n, k)])
    return out[0] if out.shape[0] == 1 else out


def rebin_var(v, k):
    """Diagonal variance of a k-bin average: (1/k^2) sum of the k variances."""
    if k == 1:
        return v
    n = v.shape[0]
    return np.array([v[i:i + k].sum() / k ** 2 for i in range(0, n, k)])


COMBO_REBIN = 2   # 15 bins -> 8 per piece; combo = 24 bins, Hartlap 0.60

VECTORS = [
    ('full auto ($\\ell$=0)',
     [('tpcf_full_data', (0,), 1)]),
    ('full x data Q4 ($\\ell$=0)',
     [('tpcf_cross_full_data_q4', (0,), 1)]),
    ('full x rand Q1 ($\\ell$=0)',
     [('tpcf_cross_full_rand_q1', (0,), 1)]),
    (f'concat, rebinned x{COMBO_REBIN} ($\\ell$=0)',
     [('tpcf_full_data',          (0,), COMBO_REBIN),
      ('tpcf_cross_full_data_q4', (0,), COMBO_REBIN),
      ('tpcf_cross_full_rand_q1', (0,), COMBO_REBIN)]),
]


def fisher_sigma(der, pieces):
    """(sigma, noise_fraction) at full-box volume; None if not measurable."""
    X_parts, d_parts, v_parts = [], [], []
    for stem, ells, k in pieces:
        c0 = np.load(DATA_DIR / FID_TAG / f'subbox_multipoles_{stem}.npz')
        for ell in ells:
            X_parts.append(rebin(c0[f'xi{ell}_all'], k))
            d_parts.append(rebin(der[f'{stem}_dxi{ell}'], k))
            v_parts.append(rebin_var(der[f'{stem}_dxi{ell}_noisevar'], k))
    X  = np.hstack(X_parts)
    d  = np.concatenate(d_parts)
    v  = np.concatenate(v_parts)
    nb   = X.shape[1]
    hart = (N_SB - nb - 2) / (N_SB - 1)
    Cinv = hart * VOL_FAC * np.linalg.inv(np.cov(X, rowvar=False))
    F    = d @ Cinv @ d
    bias = np.trace(Cinv @ np.diag(v))
    if F <= 0 or F - bias <= 0:
        return None
    return 1.0 / np.sqrt(F - bias), bias / F


def plot_param(param, der_file):
    cfg = PARAMS[param]
    der = np.load(der_file)
    fid = cfg['fid']

    results = []
    for name, pieces in VECTORS:
        r = fisher_sigma(der, pieces)
        if r is not None:
            sig_ln, noise = r
            sig = fid * sig_ln if cfg['log'] else sig_ln
            results.append((name, sig, noise))
    if not results:
        print(f'{param}: no data vector passes the noise cut, skipping')
        return
    results.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('viridis')
    width = results[-1][1] * 3.5
    x = np.linspace(fid - width, fid + width, 800)
    for k, (name, sig, noise) in enumerate(results):
        g = np.exp(-0.5 * ((x - fid) / sig) ** 2)
        color = cmap(k / max(len(results) - 1, 1))
        ax.plot(x, g, color=color, lw=2,
                label=rf'{name}  ($\sigma$={sig:.2g}, noise {noise:.0%})')
    ax.axvline(fid, color='k', lw=0.8, ls='--')
    for v, lab in ((cfg['plus'], '+ step'), (cfg['minus'], '$-$ step')):
        if fid - width < v < fid + width:
            ax.axvline(v, color='grey', lw=0.8, ls=':')
            ax.text(v, 1.02, lab, ha='center', fontsize=7, color='grey')
    ax.set_xlabel(rf'${cfg["label"]}$')
    ax.set_ylabel('likelihood (peak-normalised)')
    ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(
        rf'Fisher forecasts for ${cfg["label"]}$ per data vector '
        r'(HOD-corrected full-box derivative, full 2000 Mpc/h volume, '
        'Hartlap-corrected)',
        y=1.00, fontsize=10)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f'fisher_gaussians_{param}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    for name, sig, noise in results:
        print(f'  {name:30s} sigma({param}) = {sig:.5g}  (noise {noise:.0%})')


def vector_samples(pieces):
    """(64, nb) subbox samples of a data vector, plus block sizes."""
    parts, blocks = [], []
    for stem, ells, k in pieces:
        c0 = np.load(DATA_DIR / FID_TAG / f'subbox_multipoles_{stem}.npz')
        for ell in ells:
            arr = rebin(c0[f'xi{ell}_all'], k)
            parts.append(arr)
            blocks.append(arr.shape[1])
    return np.hstack(parts), blocks


def plot_covariances():
    """Correlation matrices of every data vector in the comparison."""
    fig, axes = plt.subplots(1, len(VECTORS), figsize=(4.2 * len(VECTORS), 4.4))
    im = None
    for ax, (name, pieces) in zip(np.atleast_1d(axes), VECTORS):
        X, blocks = vector_samples(pieces)
        C   = np.cov(X, rowvar=False)
        std = np.sqrt(np.diag(C))
        R   = C / np.outer(std, std)
        im  = ax.imshow(R, origin='lower', vmin=-1, vmax=1, cmap='RdBu_r')
        edge = 0
        for b in blocks[:-1]:
            edge += b
            ax.axhline(edge - 0.5, color='k', lw=0.8)
            ax.axvline(edge - 0.5, color='k', lw=0.8)
        ax.set_title(f'{name}\n({X.shape[1]} bins, Hartlap '
                     f'{(N_SB - X.shape[1] - 2) / (N_SB - 1):.2f})', fontsize=9)
        ax.set_xlabel('bin index')
        ax.set_ylabel('bin index')
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.8, pad=0.02,
                 label='correlation coefficient')
    fig.suptitle('Correlation matrices of the Fisher data vectors '
                 f'({N_SB} fiducial subboxes)', y=1.02)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / 'fisher_covariances.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def main():
    plot_covariances()
    found = False
    for param in PARAMS:
        der_file = DER_DIR / f'derivative_hodcorr_{param}.npz'
        if der_file.is_file():
            plot_param(param, der_file)
            found = True
        else:
            print(f'Skipping {param}: no {der_file.name} '
                  '(run compute_hod_derivatives.py)')
    if not found:
        raise SystemExit('No HOD-corrected derivative files found — run '
                         'compute_hod_derivatives.py first.')


if __name__ == '__main__':
    main()

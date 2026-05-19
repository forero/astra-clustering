#!/usr/bin/env python3
"""
Plot 2PCF results from pipeline_single_box.py.

Lines show the mean over N_ASTRA_ITERATIONS; shaded bands show ±1σ.
Covariance figures show the normalized correlation matrix (r_ij) derived
from the per-iteration multipole arrays saved in each .npz file.

Run from any node (no srun needed):
  python scripts/plot.py

Output: plots/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
PLOT_DIR  = REPO_ROOT / 'plots'
PLOT_DIR.mkdir(exist_ok=True)

N_Q    = 4
COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']  # Q1=voids → Q4=knots
LABELS = [f'Q{q}' for q in range(1, N_Q + 1)]

# ── load multipoles ────────────────────────────────────────────────────────────
def load(stem):
    d = np.load(DATA_DIR / f'multipoles_{stem}.npz')
    out = {
        's':       d['s'],
        'xi0':     d['xi0'],     'xi0_std': d['xi0_std'],
        'xi2':     d['xi2'],     'xi2_std': d['xi2_std'],
    }
    # per-iteration arrays present for all stems except full_data
    if 'xi0_all' in d:
        out['xi0_all'] = d['xi0_all']
        out['xi2_all'] = d['xi2_all']
    return out

data_q       = [load(f'tpcf_data_q{q}')             for q in range(1, N_Q + 1)]
rand_q       = [load(f'tpcf_rand_q{q}')             for q in range(1, N_Q + 1)]
cross_data_q = [load(f'tpcf_cross_full_data_q{q}')  for q in range(1, N_Q + 1)]
cross_rand_q = [load(f'tpcf_cross_full_rand_q{q}')  for q in range(1, N_Q + 1)]
full         =  load('tpcf_full_data')


# ── plotting helpers ───────────────────────────────────────────────────────────

def band(ax, d, key, color, label=None, lw=2, ls='-', alpha=0.2):
    """Plot mean line with ±1σ shaded band."""
    s  = d['s']
    y  = d[key]
    ye = d[key + '_std']
    ax.plot(s, s**2 * y, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(s, s**2 * (y - ye), s**2 * (y + ye),
                    color=color, alpha=alpha)


def corr_matrix(arr):
    """Normalized correlation matrix from an (N_iter, n_bins) array."""
    C   = np.cov(arr, rowvar=False)          # (n_bins, n_bins)
    std = np.sqrt(np.diag(C))
    with np.errstate(invalid='ignore'):
        R = C / np.outer(std, std)
    np.fill_diagonal(R, 1.0)
    R = np.nan_to_num(R, nan=0.0)
    return R


def savefig(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close(fig)


# ── Figure 1: data monopole ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for d, label, color in zip(data_q, LABELS, COLORS):
    band(ax, d, 'xi0', color=color, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_0(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Data monopole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
savefig(fig, 'data_monopole_per_quantile.png')

# ── Figure 2: data quadrupole ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for d, label, color in zip(data_q, LABELS, COLORS):
    band(ax, d, 'xi2', color=color, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_2(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Data quadrupole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
savefig(fig, 'data_quadrupole_per_quantile.png')

# ── Figure 3: data monopole + quadrupole side by side ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for d, label, color in zip(data_q, LABELS, COLORS):
    band(axes[0], d, 'xi0', color=color, label=label)
    band(axes[1], d, 'xi2', color=color, label=label)
for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
    ax.set_title(f'Data {title}')
    ax.set_xlim(0, 150)
    ax.legend(fontsize=8)
fig.suptitle('ASTRA data quantile 2PCF  —  500 Mpc/h subbox, los=z', y=1.01)
fig.tight_layout()
savefig(fig, 'data_multipoles_all_quantiles.png')

# ── Figure 4: random monopole per quantile ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for r, label, color in zip(rand_q, LABELS, COLORS):
    band(ax, r, 'xi0', color=color, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_0(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Random monopole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
savefig(fig, 'rand_monopole_per_quantile.png')

# ── Figure 5: random quadrupole per quantile ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for r, label, color in zip(rand_q, LABELS, COLORS):
    band(ax, r, 'xi2', color=color, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_2(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Random quadrupole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
savefig(fig, 'rand_quadrupole_per_quantile.png')

# ── Figure 6: data vs randoms monopole, one panel per quantile ────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for idx, ax in enumerate(axes.flat):
    d, r, color = data_q[idx], rand_q[idx], COLORS[idx]
    band(ax, d, 'xi0', color=color, label='data')
    band(ax, r, 'xi0', color=color, label='randoms', lw=1.5, ls='--', alpha=0.15)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_title(f'Q{idx + 1}')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^2\,\xi_0$')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 150)
fig.suptitle('Data vs ASTRA-random monopole per quantile', y=1.01)
fig.tight_layout()
savefig(fig, 'data_vs_rand_monopole.png')

# ── Figure 7: full data auto-correlation ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
band(axes[0], full, 'xi0', color='k')
band(axes[1], full, 'xi2', color='k')
for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
    ax.set_title(f'Full data {title}')
    ax.set_xlim(0, 150)
fig.suptitle('Full data auto-correlation  —  500 Mpc/h subbox, los=z', y=1.01)
fig.tight_layout()
savefig(fig, 'full_data_autocorr.png')

# ── Figure 8: cross-correlation full data × data quantiles ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for d, label, color in zip(cross_data_q, LABELS, COLORS):
    band(axes[0], d, 'xi0', color=color, label=label)
    band(axes[1], d, 'xi2', color=color, label=label)
for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
    ax.set_title(f'Full data × data quantile {title}')
    ax.set_xlim(0, 150)
    ax.legend(title='Q1=underdense → Q4=overdense', fontsize=8)
fig.suptitle('Cross-correlation: full data × ASTRA data quantiles', y=1.01)
fig.tight_layout()
savefig(fig, 'cross_full_data_quantiles.png')

# ── Figure 9: cross-correlation full data × random quantiles ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for r, label, color in zip(cross_rand_q, LABELS, COLORS):
    band(axes[0], r, 'xi0', color=color, label=label)
    band(axes[1], r, 'xi2', color=color, label=label)
for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
    ax.set_title(f'Full data × random quantile {title}')
    ax.set_xlim(0, 150)
    ax.legend(title='Q1=underdense → Q4=overdense', fontsize=8)
fig.suptitle('Cross-correlation: full data × ASTRA random quantiles', y=1.01)
fig.tight_layout()
savefig(fig, 'cross_full_rand_quantiles.png')


# ── Covariance figures (Figures 10–13) ────────────────────────────────────────
# Each shows a 2-row × N_Q-column grid of normalised correlation matrices.
# Rows: monopole (ℓ=0) and quadrupole (ℓ=2).
# Columns: quantiles Q1–Q4.
# Colour scale: RdBu_r, symmetric around 0, range [-1, 1].

def cov_figure(data_list, suptitle, filename):
    s = data_list[0]['s']
    extent = [s[0], s[-1], s[0], s[-1]]

    fig, axes = plt.subplots(2, N_Q, figsize=(4 * N_Q, 7), squeeze=False)
    im = None
    for q_idx, d in enumerate(data_list):
        for row, (key, ell) in enumerate([('xi0', 0), ('xi2', 2)]):
            ax  = axes[row, q_idx]
            R   = corr_matrix(d[key + '_all'])
            im  = ax.imshow(R, origin='lower', vmin=-1, vmax=1,
                            cmap='RdBu_r', extent=extent, aspect='auto')
            ax.set_title(f'{LABELS[q_idx]}   $\\ell={ell}$', fontsize=10)
            ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$', fontsize=8)
            if q_idx == 0:
                ax.set_ylabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$', fontsize=8)

    fig.colorbar(im, ax=axes, label='Correlation coefficient',
                 shrink=0.6, pad=0.02)
    fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    savefig(fig, filename)


# Figure 10: data quantile correlation matrices
cov_figure(
    data_q,
    'Data quantile correlation matrices  (ASTRA variance)',
    'cov_data_quantiles.png',
)

# Figure 11: random quantile correlation matrices
cov_figure(
    rand_q,
    'Random quantile correlation matrices  (ASTRA variance)',
    'cov_rand_quantiles.png',
)

# Figure 12: cross full data × data quantile correlation matrices
cov_figure(
    cross_data_q,
    'Cross-corr (full data × data quantile) correlation matrices  (ASTRA variance)',
    'cov_cross_full_data_quantiles.png',
)

# Figure 13: cross full data × random quantile correlation matrices
cov_figure(
    cross_rand_q,
    'Cross-corr (full data × random quantile) correlation matrices  (ASTRA variance)',
    'cov_cross_full_rand_quantiles.png',
)

print('Done.')

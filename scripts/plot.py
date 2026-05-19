#!/usr/bin/env python3
"""
Plot 2PCF results from pipeline.py.

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
    return {'s': d['s'], 'xi0': d['xi0'], 'xi2': d['xi2']}

data_q       = [load(f'tpcf_data_q{q}')             for q in range(1, N_Q + 1)]
rand_q       = [load(f'tpcf_rand_q{q}')             for q in range(1, N_Q + 1)]
cross_data_q = [load(f'tpcf_cross_full_data_q{q}')  for q in range(1, N_Q + 1)]
cross_rand_q = [load(f'tpcf_cross_full_rand_q{q}')  for q in range(1, N_Q + 1)]
full         =  load('tpcf_full_data')


def savefig(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close(fig)


# ── Figure 1: data monopole ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for d, label, color in zip(data_q, LABELS, COLORS):
    ax.plot(d['s'], d['s']**2 * d['xi0'], color=color, lw=2, label=label)
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
    ax.plot(d['s'], d['s']**2 * d['xi2'], color=color, lw=2, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_2(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Data quadrupole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
savefig(fig, 'data_quadrupole_per_quantile.png')

# ── Figure 3: data monopole + quadrupole side by side ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for d, label, color in zip(data_q, LABELS, COLORS):
    axes[0].plot(d['s'], d['s']**2 * d['xi0'], color=color, lw=2, label=label)
    axes[1].plot(d['s'], d['s']**2 * d['xi2'], color=color, lw=2, label=label)
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
    ax.plot(r['s'], r['s']**2 * r['xi0'], color=color, lw=2, label=label)
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
    ax.plot(r['s'], r['s']**2 * r['xi2'], color=color, lw=2, label=label)
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
    s = d['s']
    ax.plot(s, s**2 * d['xi0'], color=color, lw=2,   label='data')
    ax.plot(s, s**2 * r['xi0'], color=color, lw=1.5, ls='--', label='randoms')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_title(f'Q{idx + 1}')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^2\,\xi_0$')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 150)
fig.suptitle('Data vs ASTRA-random monopole per quantile', y=1.01)
fig.tight_layout()
savefig(fig, 'data_vs_rand_monopole.png')

# ── Figure 7: full data auto-correlation ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
s = full['s']
axes[0].plot(s, s**2 * full['xi0'], color='k', lw=2)
axes[1].plot(s, s**2 * full['xi2'], color='k', lw=2)
for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
    ax.set_title(f'Full data {title}')
    ax.set_xlim(0, 150)
fig.suptitle('Full data auto-correlation  —  500 Mpc/h subbox, los=z', y=1.01)
fig.tight_layout()
savefig(fig, 'full_data_autocorr.png')

# ── Figure 8: cross-correlation full data × data quantiles ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for d, label, color in zip(cross_data_q, LABELS, COLORS):
    axes[0].plot(d['s'], d['s']**2 * d['xi0'], color=color, lw=2, label=label)
    axes[1].plot(d['s'], d['s']**2 * d['xi2'], color=color, lw=2, label=label)
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

# ── Figure 9: cross-correlation full data × random quantiles ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for r, label, color in zip(cross_rand_q, LABELS, COLORS):
    axes[0].plot(r['s'], r['s']**2 * r['xi0'], color=color, lw=2, label=label)
    axes[1].plot(r['s'], r['s']**2 * r['xi2'], color=color, lw=2, label=label)
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

print('Done.')

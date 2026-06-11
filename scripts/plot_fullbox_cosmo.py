#!/usr/bin/env python3
"""
Plot 2PCF results from pipeline_fullbox_cosmo.py runs.

Each (cosmology, HOD) full-box run lives in data/fullbox/{cosmo}_hod{NNN}/
and gets its own figure directory plots/fullbox/{cosmo}_hod{NNN}/ with
3 figures (no covariance figures: with only a few ASTRA iterations the
correlation matrices are not meaningful):

  fullbox_autocorr_quantiles.png   — quantile autos, data (solid) and
                                     randoms (dashed), ℓ=0 and ℓ=2 panels
  fullbox_cross_full_quantiles.png — full × data-quantile (solid) and
                                     × random-quantile (dashed) crosses
  fullbox_full_data_autocorr.png   — full-sample auto-correlation

Bands show ±1σ over the ASTRA random realisations (no band for the
full-sample auto-correlation, which is deterministic).

Run from any node (no srun needed):
  python scripts/plot_fullbox_cosmo.py c100 179    # one run
  python scripts/plot_fullbox_cosmo.py             # all completed runs
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data' / 'fullbox'
PLOT_ROOT = REPO_ROOT / 'plots' / 'fullbox'

PREFIX = 'fullbox'
N_Q    = 4
COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
LABELS = [f'Q{q}' for q in range(1, N_Q + 1)]

# the last file the pipeline writes — its presence marks a completed run
DONE_MARKER = f'{PREFIX}_info.npz'


def find_runs():
    """All completed data/fullbox/{cosmo}_hod{NNN}/ run directories."""
    runs = []
    for d in sorted(DATA_DIR.glob('c*_hod*')):
        if d.is_dir() and re.fullmatch(r'c\d{3}_hod\d{3,}', d.name):
            if (d / DONE_MARKER).is_file():
                runs.append(d)
            else:
                print(f'Skipping {d.name} (no {DONE_MARKER} — run not finished)')
    return runs


def plot_run(run_dir):
    tag      = run_dir.name                      # e.g. c100_hod179
    plot_dir = PLOT_ROOT / tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    info     = np.load(run_dir / DONE_MARKER)
    n_iter   = int(info['n_iterations'])
    subtitle = f'{tag}  —  full 2000 Mpc/h box  —  {n_iter} ASTRA realisations'

    def load(stem):
        d = np.load(run_dir / f'{PREFIX}_multipoles_{stem}.npz')
        return {
            's':       d['s'],
            'xi0':     d['xi0'],     'xi0_std': d['xi0_std'],
            'xi2':     d['xi2'],     'xi2_std': d['xi2_std'],
        }

    data_q       = [load(f'tpcf_data_q{q}')            for q in range(1, N_Q + 1)]
    rand_q       = [load(f'tpcf_rand_q{q}')            for q in range(1, N_Q + 1)]
    cross_data_q = [load(f'tpcf_cross_full_data_q{q}') for q in range(1, N_Q + 1)]
    cross_rand_q = [load(f'tpcf_cross_full_rand_q{q}') for q in range(1, N_Q + 1)]
    full         =  load('tpcf_full_data')

    def band(ax, d, key, color, label=None, lw=2, ls='-', alpha=0.2):
        s  = d['s']
        y  = d[key]
        ye = d[key + '_std']
        ax.plot(s, s**2 * y, color=color, lw=lw, ls=ls, label=label)
        if np.any(ye > 0):
            ax.fill_between(s, s**2 * (y - ye), s**2 * (y + ye),
                            color=color, alpha=alpha)

    def savefig(fig, name):
        path = plot_dir / f'{PREFIX}_{name}'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
        plt.close(fig)

    # ── Figure 1: quantile auto-correlations (data + randoms) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for d, r, label, color in zip(data_q, rand_q, LABELS, COLORS):
        band(axes[0], d, 'xi0', color=color, label=label)
        band(axes[0], r, 'xi0', color=color, lw=1.5, ls='--', alpha=0.1)
        band(axes[1], d, 'xi2', color=color, label=label)
        band(axes[1], r, 'xi2', color=color, lw=1.5, ls='--', alpha=0.1)
    for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
        ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
        ax.set_title(title)
        ax.set_xlim(0, 150)
        ax.legend(title='solid = data, dashed = randoms\nQ1=underdense → Q4=overdense',
                  fontsize=8, title_fontsize=8)
    fig.suptitle(f'Quantile auto-correlations  —  {subtitle}', y=1.01)
    fig.tight_layout()
    savefig(fig, 'autocorr_quantiles.png')

    # ── Figure 2: full data auto-correlation ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    band(axes[0], full, 'xi0', color='k')
    band(axes[1], full, 'xi2', color='k')
    for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
        ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
        ax.set_title(f'Full data {title}')
        ax.set_xlim(0, 150)
    fig.suptitle(f'Full data auto-correlation  —  {subtitle}', y=1.01)
    fig.tight_layout()
    savefig(fig, 'full_data_autocorr.png')

    # ── Figure 3: full data × quantile cross-correlations (data + randoms) ────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for d, r, label, color in zip(cross_data_q, cross_rand_q, LABELS, COLORS):
        band(axes[0], d, 'xi0', color=color, label=label)
        band(axes[0], r, 'xi0', color=color, lw=1.5, ls='--', alpha=0.1)
        band(axes[1], d, 'xi2', color=color, label=label)
        band(axes[1], r, 'xi2', color=color, lw=1.5, ls='--', alpha=0.1)
    for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
        ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
        ax.set_title(title)
        ax.set_xlim(0, 150)
        ax.legend(title='solid = full×data Q, dashed = full×random Q\nQ1=underdense → Q4=overdense',
                  fontsize=8, title_fontsize=8)
    fig.suptitle(f'Cross-correlations: full data × ASTRA quantiles  —  {subtitle}', y=1.01)
    fig.tight_layout()
    savefig(fig, 'cross_full_quantiles.png')


def main():
    parser = argparse.ArgumentParser(
        description='Plot full-box 2PCF results for (cosmology, HOD) runs')
    parser.add_argument('cosmo', nargs='?', help='cosmology, e.g. c100 (omit to plot all runs)')
    parser.add_argument('hod', nargs='?', type=int, help='HOD index, e.g. 179')
    args = parser.parse_args()

    if (args.cosmo is None) != (args.hod is None):
        parser.error('give both cosmo and hod, or neither')

    if args.cosmo is not None:
        run_dir = DATA_DIR / f'{args.cosmo}_hod{args.hod:03d}'
        if not (run_dir / DONE_MARKER).is_file():
            sys.exit(f'No completed run in {run_dir} (missing {DONE_MARKER})')
        runs = [run_dir]
    else:
        runs = find_runs()
        if not runs:
            sys.exit(f'No completed runs found under {DATA_DIR}/c*_hod*/')

    for run_dir in runs:
        print(f'=== Plotting {run_dir.name} ===')
        plot_run(run_dir)
    print('Done.')


if __name__ == '__main__':
    main()

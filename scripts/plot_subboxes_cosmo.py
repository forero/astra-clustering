#!/usr/bin/env python3
"""
Plot 2PCF results from pipeline_subboxes_cosmo.py runs.

Each (cosmology, HOD) run lives in data/{cosmo}_hod{NNN}/ and gets its own
figure directory plots/{cosmo}_hod{NNN}/ with 7 figures:

  autocorr_quantiles    — quantile auto-correlations, data (solid) and
                          randoms (dashed), monopole + quadrupole panels
  cross_full_quantiles  — full data × quantile cross-correlations, data
                          (solid) and randoms (dashed), same layout
  full_data_autocorr    — full-sample auto-correlation
  cov_*                 — 4 normalised correlation-matrix figures

Lines show the mean over 64 subboxes; shaded bands show ±1σ (cosmic
variance); covariance figures use the per-subbox multipole arrays.

Run from any node (no srun needed):
  python scripts/plot_subboxes_cosmo.py c100 179    # one run
  python scripts/plot_subboxes_cosmo.py             # all completed runs
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
DATA_DIR  = REPO_ROOT / 'data'
PLOT_ROOT = REPO_ROOT / 'plots'

PREFIX = 'subbox'
N_Q    = 4
COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
LABELS = [f'Q{q}' for q in range(1, N_Q + 1)]

# the last file the pipeline writes — its presence marks a completed run
DONE_MARKER = f'{PREFIX}_info.npz'


def find_runs():
    """All completed data/{cosmo}_hod{NNN}/ run directories."""
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
    subtitle = f'{tag}  —  64 subboxes of 500 Mpc/h  —  cosmic variance'

    def load(stem):
        d = np.load(run_dir / f'{PREFIX}_multipoles_{stem}.npz')
        return {
            's':       d['s'],
            'xi0':     d['xi0'],     'xi0_std': d['xi0_std'],
            'xi2':     d['xi2'],     'xi2_std': d['xi2_std'],
            'xi0_all': d['xi0_all'],
            'xi2_all': d['xi2_all'],
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
        ax.fill_between(s, s**2 * (y - ye), s**2 * (y + ye),
                        color=color, alpha=alpha)

    def corr_matrix(arr):
        C   = np.cov(arr, rowvar=False)
        std = np.sqrt(np.diag(C))
        with np.errstate(invalid='ignore'):
            R = C / np.outer(std, std)
        np.fill_diagonal(R, 1.0)
        return np.nan_to_num(R, nan=0.0)

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

    # ── Covariance figures (Figures 10–13) ────────────────────────────────────
    def cov_figure(data_list, suptitle, filename):
        s      = data_list[0]['s']
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

    cov_figure(data_q,       f'Data quantile correlation matrices  —  {subtitle}',
               'cov_data_quantiles.png')
    cov_figure(rand_q,       f'Random quantile correlation matrices  —  {subtitle}',
               'cov_rand_quantiles.png')
    cov_figure(cross_data_q, f'Cross-corr (full×data quantile) correlation matrices  —  {subtitle}',
               'cov_cross_full_data_quantiles.png')
    cov_figure(cross_rand_q, f'Cross-corr (full×random quantile) correlation matrices  —  {subtitle}',
               'cov_cross_full_rand_quantiles.png')


def main():
    parser = argparse.ArgumentParser(
        description='Plot subbox 2PCF results for (cosmology, HOD) runs')
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

#!/usr/bin/env python3
"""
Figures for the ASTRA full-box weighted-2PCF experiment (pipeline_fullbox_weighted).

  weighted_xi_multipoles.png   s^2 xi_ell(s) for the positive weight schemes
                               (uniform/knot/void/rank/power/exp), 3 rows
                               (data-auto / astra_random-auto / data x astra_random
                               cross) x 2 cols (ell=0, ell=2), iteration mean +/- std.
  marked_correlation_signed.png  marked-correlation monopole M0(s)=(1+W0)/(1+xi0)
                               for the signed-r mark, per statistic (M=1 = null).

Usage:  python scripts/plot_fullbox_weighted.py [cosmo hod]  (default c000 484)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
COSMO = sys.argv[1] if len(sys.argv) > 1 else 'c000'
HOD   = int(sys.argv[2]) if len(sys.argv) > 2 else 484
RUN   = f'{COSMO}_hod{HOD:03d}'
DDIR  = REPO / 'data' / 'fullbox_weighted' / RUN
PDIR  = REPO / 'plots' / 'fullbox_weighted'
PDIR.mkdir(parents=True, exist_ok=True)
PREFIX = 'fbw'

POS_SCHEMES = ['uniform', 'knot', 'void', 'rank', 'power', 'exp']
STATS = [('data', 'data-auto'), ('arand', 'astra_random-auto'), ('cross', 'data x astra_random')]
COLORS = {'uniform': 'k', 'knot': '#d62728', 'void': '#1f77b4', 'rank': '#2ca02c',
          'power': '#ff7f0e', 'exp': '#9467bd'}


def load(stem):
    f = DDIR / f'{PREFIX}_multipoles_{stem}.npz'
    return np.load(f) if f.is_file() else None


# ── Figure 1: weighted xi multipoles ─────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True)
for row, (st, title) in enumerate(STATS):
    for col, ell in enumerate([0, 2]):
        ax = axes[row, col]
        for sch in POS_SCHEMES:
            d = load(f'{sch}_{st}')
            if d is None:
                continue
            s = d['s']
            xi = d[f'xi{ell}']; err = d[f'xi{ell}_std']
            lw = 2.2 if sch == 'uniform' else 1.4
            ax.plot(s, s**2 * xi, color=COLORS[sch], lw=lw,
                    label=sch + (' (=standard xi)' if sch == 'uniform' else ''))
            ax.fill_between(s, s**2 * (xi - err), s**2 * (xi + err),
                            color=COLORS[sch], alpha=0.15)
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_title(f'{title}   ell={ell}')
        if col == 0:
            ax.set_ylabel(r'$s^2\,\xi_\ell(s)$')
        if row == 2:
            ax.set_xlabel('s  [Mpc/h]')
        if row == 0 and col == 0:
            ax.legend(fontsize=8, ncol=2)
fig.suptitle(f'ASTRA weighted-2PCF  ({RUN}, full box, mean$\\pm$std over iterations)', y=0.995)
fig.tight_layout()
out1 = PDIR / 'weighted_xi_multipoles.png'
fig.savefig(out1, dpi=120)
print('Saved', out1)

# ── Figure 2: marked correlation monopole for the signed-r mark ──────────────
# M0(s) = (1 + W0_signed) / (1 + xi0_uniform), per iteration then mean +/- std
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
ok = True
for ax, (st, title) in zip(axes, STATS):
    w = load(f'signed_{st}'); u = load(f'uniform_{st}')
    if w is None or u is None:
        ok = False
        ax.text(0.5, 0.5, 'signed/uniform missing', ha='center', transform=ax.transAxes)
        continue
    s = w['s']
    M = (1.0 + w['xi0_all']) / (1.0 + u['xi0_all'])   # (n_iter, n_bins)
    ax.plot(s, M.mean(0), color='#8c564b', lw=2)
    if M.shape[0] > 1:
        e = M.std(0, ddof=1)
        ax.fill_between(s, M.mean(0) - e, M.mean(0) + e, color='#8c564b', alpha=0.2)
    ax.axhline(1.0, color='grey', ls='--', lw=1)
    ax.set_title(f'marked corr M0   {title}')
    ax.set_xlabel('s  [Mpc/h]')
axes[0].set_ylabel(r'$M_0(s)=(1+W_0)/(1+\xi_0)$')
fig.suptitle(f'Signed-r marked correlation monopole  ({RUN})  -  M0=1 is the null', y=1.02)
fig.tight_layout()
out2 = PDIR / 'marked_correlation_signed.png'
fig.savefig(out2, dpi=120)
print('Saved', out2)

# ── quick text sanity: uniform should equal the standard full-data xi ────────
u = load('uniform_data')
if u is not None:
    print(f"uniform data-auto xi0[7]={u['xi0'][7]:+.4f} (should match standard full-data xi0)")

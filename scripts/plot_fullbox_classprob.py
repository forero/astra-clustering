#!/usr/bin/env python3
"""
Figures for the ASTRA full-box CLASS-PROBABILITY weighted-2PCF experiment
(pipeline_fullbox_classprob.py).

  classprob_stability.png       histogram of max_c P_class per galaxy (stable
                                 classification vs boundary/ambiguous galaxies)
                                 + mean P_class per class as a bar chart.
  classprob_auto_multipoles.png s^2 xi_ell(s) for the 4 class-weighted data-autos
                                 (void/sheet/filament/knot), ell=0 and ell=2, with
                                 the equal-count quantile data-Q autos (Q1..Q4,
                                 from the existing quantile pipeline) overlaid as
                                 a qualitative sanity check.
  classprob_cross_multipoles.png s^2 xi_ell(s) for full x class-weighted (4) and
                                 void x knot (1) crosses.

Usage:  python scripts/plot_fullbox_classprob.py [cosmo hod]  (default c000 484)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO  = Path(__file__).resolve().parents[1]
COSMO = sys.argv[1] if len(sys.argv) > 1 else 'c000'
HOD   = int(sys.argv[2]) if len(sys.argv) > 2 else 484
RUN   = f'{COSMO}_hod{HOD:03d}'
DDIR  = REPO / 'data' / 'fullbox_weighted' / RUN
QDIR  = REPO / 'data' / 'fullbox' / RUN
PDIR  = REPO / 'plots' / 'fullbox_weighted'
PDIR.mkdir(parents=True, exist_ok=True)
PREFIX = 'fbwp'

CLASSES = ['void', 'sheet', 'filament', 'knot']
COLORS  = {'void': '#1f77b4', 'sheet': '#2ca02c', 'filament': '#ff7f0e', 'knot': '#d62728'}


def load(stem):
    f = DDIR / f'{PREFIX}_multipoles_{stem}.npz'
    return np.load(f) if f.is_file() else None


def load_quantile(q):
    f = QDIR / f'fullbox_multipoles_tpcf_data_q{q}.npz'
    return np.load(f) if f.is_file() else None


pc_file = DDIR / f'{PREFIX}_pclass.npz'
if not pc_file.is_file():
    sys.exit(f'{pc_file} not found -- run pipeline_fullbox_classprob.py first')
pc = np.load(pc_file)
P = pc['P']; keep = pc['keep']

# ── Figure 1: classification stability diagnostics ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
maxp = P[keep].max(axis=1)
axes[0].hist(maxp, bins=40, color='#555', edgecolor='none')
axes[0].axvline(0.9, color='crimson', ls='--', lw=1.2, label='stable (>=0.9)')
axes[0].axvline(0.5, color='orange', ls='--', lw=1.2, label='no majority (<0.5)')
axes[0].set_xlabel(r'$\max_c P_{\rm class}(i)$'); axes[0].set_ylabel('N galaxies')
axes[0].set_title('Classification stability'); axes[0].legend(fontsize=8)

means = [P[keep, i].mean() for i in range(4)]
axes[1].bar(CLASSES, means, color=[COLORS[c] for c in CLASSES])
for i, m in enumerate(means):
    axes[1].text(i, m, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
axes[1].set_ylabel(r'$\langle P_{\rm class}\rangle$ (kept galaxies)')
axes[1].set_title(f'Mean class occupancy  (N_PROB_ITERS={int(pc["n_prob_iters"])}, '
                   f'kept={keep.sum():,}/{len(keep):,})')
fig.suptitle(f'{RUN}: per-galaxy class-probability diagnostics', y=1.02)
fig.tight_layout()
out1 = PDIR / 'classprob_stability.png'
fig.savefig(out1, dpi=120, bbox_inches='tight')
print('Saved', out1)

# ── Figure 2: class-weighted data-auto multipoles vs quantile data-Q autos ────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
for col, ell in enumerate([0, 2]):
    ax = axes[col]
    for c in CLASSES:
        d = load(f'{c}_auto')
        if d is None:
            continue
        s = d['s']; xi = d[f'xi{ell}']
        ax.plot(s, s**2 * xi, color=COLORS[c], lw=2.0, label=f'{c} (P-weighted)')
    qcolors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))
    for q in range(1, 5):
        d = load_quantile(q)
        if d is None:
            continue
        s = d['s']; xi = d[f'xi{ell}']
        ax.plot(s, s**2 * xi, color=qcolors[q - 1], lw=1.2, ls='--', alpha=0.7,
                label=f'quantile Q{q}' if col == 0 else None)
    ax.axhline(0, color='grey', lw=0.6)
    ax.set_title(f'ell={ell}'); ax.set_xlabel('s  [Mpc/h]')
    if col == 0:
        ax.set_ylabel(r'$s^2\,\xi_\ell(s)$')
        ax.legend(fontsize=7, ncol=2)
fig.suptitle(f'{RUN}: class-probability-weighted data-auto vs quantile data-Q autos '
             '(qualitative check)', y=1.02)
fig.tight_layout()
out2 = PDIR / 'classprob_auto_multipoles.png'
fig.savefig(out2, dpi=120, bbox_inches='tight')
print('Saved', out2)

# ── Figure 3: crosses ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
for col, ell in enumerate([0, 2]):
    ax = axes[col]
    for c in CLASSES:
        d = load(f'full_x_{c}')
        if d is None:
            continue
        s = d['s']; xi = d[f'xi{ell}']
        ax.plot(s, s**2 * xi, color=COLORS[c], lw=2.0, label=f'full x {c}')
    d = load('void_x_knot')
    if d is not None:
        s = d['s']; xi = d[f'xi{ell}']
        ax.plot(s, s**2 * xi, color='k', lw=2.2, ls=':', label='void x knot')
    ax.axhline(0, color='grey', lw=0.6)
    ax.set_title(f'ell={ell}'); ax.set_xlabel('s  [Mpc/h]')
    if col == 0:
        ax.set_ylabel(r'$s^2\,\xi_\ell(s)$')
        ax.legend(fontsize=8)
fig.suptitle(f'{RUN}: class-probability crosses (full x class, void x knot)', y=1.02)
fig.tight_layout()
out3 = PDIR / 'classprob_cross_multipoles.png'
fig.savefig(out3, dpi=120, bbox_inches='tight')
print('Saved', out3)

# ── text sanity checks ─────────────────────────────────────────────────────────
print()
print('Sanity checks:')
psum = P[keep].sum(axis=1)
print(f'  sum_c P_class over kept: min={psum.min():.6f} max={psum.max():.6f} (expect ~1)')
print(f'  fraction stable (max P>=0.9): {(maxp >= 0.9).mean():.3f}')
print(f'  fraction ambiguous (max P<0.5): {(maxp < 0.5).mean():.3f}')
d_void = load('void_auto'); d_knot = load('knot_auto')
if d_void is not None and d_knot is not None:
    print(f'  void_auto xi0[7]={d_void["xi0"][7]:+.4f}  knot_auto xi0[7]={d_knot["xi0"][7]:+.4f} '
          '(expect void<0<knot, mirroring the quantile Q1/Q4 sign pattern)')

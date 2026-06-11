#!/usr/bin/env python3
"""
Central-difference derivatives of the clustering statistics with respect
to the cosmological parameters, from the Fisher subbox runs.

For each parameter with both ± runs complete (see CLAUDE.md,
"Fisher-matrix design"), the derivative is estimated subbox-by-subbox:

    d xi_i / d theta  =  [ xi_i(c+) - xi_i(c-) ] / (2 dtheta)

for each subbox i.  Because the ± runs share initial-condition phases and
random seeds, the per-subbox differences are strongly correlated and the
paired estimate is much more precise than differencing the means of
independent volumes.  The mean over the 64 subboxes gives the derivative;
the scatter gives its uncertainty.

Caveat (CLAUDE.md): the ± catalogs carry slightly different HOD draws, so
a smooth amplitude-like contamination is expected on top of the cosmology
signal.

Outputs
  data/derivatives/derivative_{param}.npz   — per-stem mean and error,
                                              plus per-subbox arrays
  plots/derivatives/derivative_{param}.png  — 2x5 panel figure
                                              (rows: ell=0,2; columns:
                                              full auto, data Q autos,
                                              random Q autos, full x data Q,
                                              full x random Q)

Usage (any node):
  python scripts/compute_derivatives.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
OUT_DIR   = DATA_DIR / 'derivatives'
PLOT_DIR  = REPO_ROOT / 'plots' / 'derivatives'

N_Q    = 4
COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
LABELS = [f'Q{q}' for q in range(1, N_Q + 1)]

# parameter: (plus run, minus run, half-step dtheta, latex label)
PAIRS = {
    'lnwb': ('c100_hod179', 'c101_hod152', 0.020, r'\ln\omega_b'),
    'lnwc': ('c102_hod556', 'c103_hod861', 0.033, r'\ln\omega_c'),
    'ns':   ('c104_hod498', 'c105_hod589', 0.010, r'n_s'),
    'lns8': ('c112_hod507', 'c113_hod483', 0.020, r'\ln\sigma_8'),
}

STEMS = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)


def run_complete(tag):
    return (DATA_DIR / tag / 'subbox_info.npz').is_file()


def compute_derivative(param, tag_p, tag_m, dtheta):
    """Per-subbox central differences for every stem; returns dict + s."""
    out = {}
    s = None
    for stem in STEMS:
        dp = np.load(DATA_DIR / tag_p / f'subbox_multipoles_{stem}.npz')
        dm = np.load(DATA_DIR / tag_m / f'subbox_multipoles_{stem}.npz')
        if s is None:
            s = dp['s']
        for ell in (0, 2):
            key  = f'xi{ell}'
            dd   = (dp[key + '_all'] - dm[key + '_all']) / (2 * dtheta)  # (64, nbins)
            out[f'{stem}_dxi{ell}_all'] = dd
            out[f'{stem}_dxi{ell}']     = dd.mean(axis=0)
            out[f'{stem}_dxi{ell}_err'] = dd.std(axis=0, ddof=1) / np.sqrt(dd.shape[0])
    return s, out


def plot_derivative(param, label, s, der, tag_p, tag_m):
    """2 rows (ell=0,2) x 5 columns (statistic family)."""
    families = [
        ('Full auto',        ['tpcf_full_data'],                                None),
        ('Data Q autos',     [f'tpcf_data_q{q}' for q in range(1, N_Q + 1)],    LABELS),
        ('Random Q autos',   [f'tpcf_rand_q{q}' for q in range(1, N_Q + 1)],    LABELS),
        ('Full x data Q',    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)], LABELS),
        ('Full x random Q',  [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)], LABELS),
    ]
    fig, axes = plt.subplots(2, len(families), figsize=(4 * len(families), 7.5),
                             sharex=True)
    for col, (title, stems, labels) in enumerate(families):
        colors = ['k'] if labels is None else COLORS
        labels = [None] if labels is None else labels
        for row, ell in enumerate((0, 2)):
            ax = axes[row, col]
            for stem, color, lab in zip(stems, colors, labels):
                y  = der[f'{stem}_dxi{ell}']
                ye = der[f'{stem}_dxi{ell}_err']
                ax.plot(s, s**2 * y, color=color, lw=1.8, label=lab)
                ax.fill_between(s, s**2 * (y - ye), s**2 * (y + ye),
                                color=color, alpha=0.25)
            ax.axhline(0, color='k', lw=0.8, ls='--')
            ax.set_xlim(0, 150)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if row == 1:
                ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
            if col == 0:
                ax.set_ylabel(rf'$s^2\,\partial\xi_{ell}/\partial {label}'
                              rf'\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
            if labels[0] is not None and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        rf'Central difference $\partial\xi_\ell/\partial {label}$'
        f'  —  ({tag_p} $-$ {tag_m}) / 2$\\Delta$,  64 paired subboxes,'
        f'  mean $\\pm$ error of mean',
        y=1.00, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = PLOT_DIR / f'derivative_{param}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    for param, (tag_p, tag_m, dtheta, label) in PAIRS.items():
        if not (run_complete(tag_p) and run_complete(tag_m)):
            missing = [t for t in (tag_p, tag_m) if not run_complete(t)]
            print(f'Skipping d/d{param}: missing {", ".join(missing)}')
            continue
        print(f'=== d/d{param}:  ({tag_p} - {tag_m}) / {2 * dtheta} ===')
        s, der = compute_derivative(param, tag_p, tag_m, dtheta)
        np.savez(OUT_DIR / f'derivative_{param}.npz',
                 s=s, dtheta=dtheta, tag_plus=tag_p, tag_minus=tag_m, **der)
        print(f'Saved {OUT_DIR / f"derivative_{param}.npz"}')
        plot_derivative(param, label, s, der, tag_p, tag_m)
        done += 1
    if done == 0:
        raise SystemExit('No complete ± pair found.')


if __name__ == '__main__':
    main()

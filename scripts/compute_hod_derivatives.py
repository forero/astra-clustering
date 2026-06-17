#!/usr/bin/env python3
"""
Tier-1: measure ∂ξ/∂θ_HOD from the c000 calibration runs and subtract the
HOD contamination from the full-box cosmology derivatives.

The Fisher ± pairs are not HOD-matched, so each full-box derivative
(derivative_fullbox_{param}.npz) contains a term

    [∂ξ/∂θ_HOD] · Δθ_HOD / (2 dθ_cosmo)

where Δθ_HOD = θ_HOD(c+) − θ_HOD(c−) is the HOD-parameter mismatch between the
two catalogs of the pair.  We estimate ∂ξ/∂θ_HOD by regressing the full-box ξ
of the completed c000 calibration draws on their HOD parameters (linear, with
intercept, parameters standardised for conditioning), then evaluate the
contamination per pair and subtract it.

Assumes the HOD response is cosmology-independent to first order (gradient
measured at c000, applied to the ± cosmologies' HOD mismatch) and locally
linear over the prior neighbourhood — both standard, both noted as caveats.

Inputs
  data/hod_calibration/hod_params_c000.csv   (from select_hod_calibration.py)
  data/fullbox/c000_hod{NNN}/                (calibration runs)
  data/derivatives/derivative_fullbox_{param}.npz

Outputs
  data/derivatives/derivative_hodcorr_{param}.npz
      s, dtheta, and per (stem, ell): _dxi{ell} (corrected), _contam{ell}
  plots/derivatives/hod_contamination_{param}.png   (full-auto, ℓ=0 and ℓ=2)

Usage (any node):
  python scripts/compute_hod_derivatives.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitsio

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
FB_DIR    = DATA_DIR / 'fullbox'
DER_DIR   = DATA_DIR / 'derivatives'
CAL_DIR   = DATA_DIR / 'hod_calibration'
PLOT_DIR  = REPO_ROOT / 'plots' / 'derivatives'
HOD_BASE  = Path('/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior')

N_Q = 4
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


def load_param_table():
    lines  = (CAL_DIR / 'hod_params_c000.csv').read_text().splitlines()
    names  = lines[0].split(',')[1:]
    table  = {}
    for ln in lines[1:]:
        parts = ln.split(',')
        table[int(parts[0])] = np.array([float(x) for x in parts[1:]])
    return names, table


def hod_params_for(tag, names):
    """Read the varying HOD params for a '{cosmo}_hod{NNN}' tag's catalog."""
    cosmo, hod = tag.split('_hod')
    f = HOD_BASE / f'{cosmo}_ph000' / 'seed0' / f'hod{int(hod):03d}.fits'
    h = fitsio.read_header(str(f), ext=1)
    return np.array([float(h[k]) for k in names])


def completed_calibration_runs(table):
    """(hod indices, param matrix) for calibration draws with a finished run."""
    sel = [int(x) for x in
           (CAL_DIR / 'hod_selection_c000.txt').read_text().split()]
    hods, rows = [], []
    for h in sel:
        if (FB_DIR / f'c000_hod{h:03d}' / 'fullbox_info.npz').is_file():
            hods.append(h)
            rows.append(table[h])
    return hods, np.array(rows)


def fit_gradient(hods, P):
    """
    Per stem/ell, regress xi on standardised HOD params (with intercept).
    Returns s and dict {stem}_g{ell}: physical gradient (nparams, nbins).
    """
    mu, sd = P.mean(0), P.std(0)
    Z  = (P - mu) / sd
    A  = np.column_stack([np.ones(len(Z)), Z])          # (n, 1+nparam)
    grads, s = {}, None
    for stem in STEMS:
        xis = {0: [], 2: []}
        for h in hods:
            d = np.load(FB_DIR / f'c000_hod{h:03d}'
                        / f'fullbox_multipoles_{stem}.npz')
            if s is None:
                s = d['s']
            xis[0].append(d['xi0']); xis[2].append(d['xi2'])
        for ell in (0, 2):
            Y    = np.array(xis[ell])                   # (n, nbins)
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            g_std = coef[1:]                            # (nparam, nbins)
            grads[f'{stem}_g{ell}'] = g_std / sd[:, None]   # -> physical
    return s, grads, mu, sd


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    names, table = load_param_table()
    hods, P = completed_calibration_runs(table)
    nparam  = len(names)
    print(f'Calibration draws completed: {len(hods)} / '
          f'{len((CAL_DIR / "hod_selection_c000.txt").read_text().split())}')
    if len(hods) < nparam + 2:
        raise SystemExit(
            f'Need > {nparam + 1} completed c000 calibration runs to fit '
            f'{nparam} HOD parameters; have {len(hods)}. '
            f'Launch with queue/launch_hod_calibration.sh and rerun.')

    s, grads, mu, sd = fit_gradient(hods, P)

    done = 0
    for param, (tag_p, tag_m, dtheta, label) in PAIRS.items():
        f_fb = DER_DIR / f'derivative_fullbox_{param}.npz'
        if not f_fb.is_file():
            print(f'Skipping {param}: no {f_fb.name}')
            continue
        der  = dict(np.load(f_fb))
        dth_hod = hod_params_for(tag_p, names) - hod_params_for(tag_m, names)
        rms = np.sqrt(np.mean(((dth_hod) / sd) ** 2))
        print(f'=== {param}: HOD mismatch |Δθ/σ_prior| rms = {rms:.3f} '
              f'(2Δθ_cosmo={2 * dtheta}) ===')

        out = {'s': s, 'dtheta': dtheta, 'tag_plus': tag_p, 'tag_minus': tag_m}
        for stem in STEMS:
            for ell in (0, 2):
                G      = grads[f'{stem}_g{ell}']            # (nparam, nbins)
                contam = (dth_hod @ G) / (2 * dtheta)        # (nbins,)
                raw    = der[f'{stem}_dxi{ell}']
                out[f'{stem}_dxi{ell}']    = raw - contam
                out[f'{stem}_contam{ell}'] = contam
                # carry the noise variance through unchanged
                if f'{stem}_dxi{ell}_noisevar' in der:
                    out[f'{stem}_dxi{ell}_noisevar'] = der[f'{stem}_dxi{ell}_noisevar']
        np.savez(DER_DIR / f'derivative_hodcorr_{param}.npz', **out)
        print(f'Saved {DER_DIR / f"derivative_hodcorr_{param}.npz"}')

        # figure: full-auto raw / contamination / corrected, ℓ=0 and ℓ=2
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for ax, ell in zip(axes, (0, 2)):
            raw    = der[f'tpcf_full_data_dxi{ell}']
            contam = out[f'tpcf_full_data_contam{ell}']
            ax.plot(s, s**2 * raw,            'k',  lw=2, label='raw full-box')
            ax.plot(s, s**2 * contam,         'C1', lw=1.8, ls='--',
                    label='HOD contamination')
            ax.plot(s, s**2 * (raw - contam), 'C3', lw=2, label='HOD-corrected')
            ax.axhline(0, color='grey', lw=0.6)
            ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
            ax.set_title(rf'$\ell={ell}$', fontsize=10)
        axes[0].set_ylabel(rf'$s^2\,\partial\xi/\partial {label}$')
        axes[0].legend(fontsize=8)
        fig.suptitle(rf'HOD-contamination subtraction for $\partial\xi/'
                     rf'\partial {label}$ (full auto, {len(hods)} c000 draws)',
                     y=1.02)
        fig.tight_layout()
        path = PLOT_DIR / f'hod_contamination_{param}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')
        done += 1

    if done == 0:
        raise SystemExit('No full-box derivative files to correct — run '
                         'compute_derivatives_fullbox.py first.')


if __name__ == '__main__':
    main()

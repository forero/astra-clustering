#!/usr/bin/env python3
"""
Test the cosmology-independence of the HOD response.

The whole global-response / HOD-marginalisation design assumes ∂ξ/∂θ_HOD is the
same in every cosmology (one gradient, measured across the grid, applied to each
pair's HOD mismatch).  This script checks that assumption empirically: it fits
the HOD gradient *separately within each cosmology* that has enough same-phase
draws, then compares the per-cosmology gradients against each other with the
gradients' own fit errors propagated.

Within one cosmology the cosmology parameters are constant, so the regressors are
only the 12 varying yuan23 HOD parameters (+ intercept):

    ξ_h = ξ₀ + Σ_q b_q · z_q,h          z = HOD params, standardised by the
                                         POOLED prior spread (same scaling for
                                         every cosmology so b_q are comparable)

Comparison statistic (screening): for each other cosmology c vs the fiducial
c000, per (HOD param q, s-bin b),

    Δ = g_q^c(b) − g_q^000(b)
    z = Δ / sqrt(var_q^c(b) + var_q^000(b))

aggregated to χ²/dof and the fraction of |z|>2.  Bins are correlated, so χ²/dof
is an approximate screen, not a calibrated p-value; a value ≈1 (and few |z|>2)
means the HOD response is consistent with cosmology-independence along the axes
the ready cosmologies span.

SCOPE depends on which ensembles are filled in.  With only c000/c100/c101 ready
this tests the ω_b axis (c100/c101 = ω_b ±2%); the σ₈/ω_c/n_s axes come online as
c112/c113, c102/c103, c104/c105 fill in.  The script auto-detects ready
cosmologies (>= MIN_DRAWS) and prints the axes covered.

Outputs
  data/derivatives/hod_gradient_percosmo.npz  (per-cosmology gradients + variance)
  plots/derivatives/hod_response_cosmo_independence_ell{0,2}.png
      full-auto 3x4 grid: s^2 ∂ξ/∂θ_q overlaid per cosmology with ±1σ bands

Usage (any node):  python scripts/test_hod_response_cosmo_independence.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
FB_DIR    = DATA_DIR / 'fullbox'
ENS_DIR   = DATA_DIR / 'hod_ensemble'
DER_DIR   = DATA_DIR / 'derivatives'
PLOT_DIR  = REPO_ROOT / 'plots' / 'derivatives'

MIN_DRAWS = 15          # need > 13 = 12 HOD params + intercept to fit a cosmology
FIDUCIAL  = 'c000'      # reference cosmology for the pairwise comparison
N_Q = 4
STEMS = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)

# which abacus parameter each cosmology varies vs the fiducial (for the axis label)
COSMO_AXIS = {
    'c100': 'ω_b +', 'c101': 'ω_b −',
    'c102': 'ω_c +', 'c103': 'ω_c −',
    'c104': 'n_s +',  'c105': 'n_s −',
    'c112': 'σ₈ +',  'c113': 'σ₈ −',
}


def hod_tables():
    """{cosmo: {hod index: param array}}, and the shared param names."""
    tables, names = {}, None
    for f in sorted(ENS_DIR.glob('hod_params_*.csv')):
        cosmo = f.stem.replace('hod_params_', '')
        df = pd.read_csv(f)
        names = list(df.columns[1:])
        tables[cosmo] = {int(r['hod']): r.values[1:].astype(float)
                         for _, r in df.iterrows()}
    return names, tables


def ready_runs(hod_tab):
    """{cosmo: [(hod, dir)]} for cosmologies with >= MIN_DRAWS complete runs."""
    by_cosmo = {}
    for d in sorted(FB_DIR.glob('c*_hod*')):
        if not (d / 'fullbox_info.npz').is_file():
            continue
        cosmo, hod = d.name.split('_hod')
        hod = int(hod)
        if cosmo in hod_tab and hod in hod_tab[cosmo]:
            by_cosmo.setdefault(cosmo, []).append((hod, d))
    return {c: v for c, v in by_cosmo.items() if len(v) >= MIN_DRAWS}


def load_stack(runs, stem):
    """(s, {ell: Y (nrun,nbins)}) for one stem across a cosmology's runs."""
    s = None
    ys = {0: [], 2: []}
    for hod, d in runs:
        arr = np.load(d / f'fullbox_multipoles_{stem}.npz')
        if s is None:
            s = arr['s']
        ys[0].append(arr['xi0']); ys[2].append(arr['xi2'])
    return s, {ell: np.array(ys[ell]) for ell in (0, 2)}


def fit_gradient(Z, Y):
    """Standardised-HOD lstsq. Returns (grad_std (nhod,nbins), var_std (nhod,nbins)).

    grad_std are the standardised slopes; var_std their fit variance (diagonal of
    the coefficient covariance, σ²_bin · (AᵀA)⁻¹_qq)."""
    A = np.column_stack([np.ones(len(Z)), Z])              # (n, 1+nhod)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)           # (1+nhod, nbins)
    resid = Y - A @ coef
    dof = len(Z) - A.shape[1]
    sigma2 = (resid ** 2).sum(0) / dof                     # (nbins,)
    AtA_inv = np.linalg.inv(A.T @ A)
    diag = np.diag(AtA_inv)[1:]                            # drop intercept -> (nhod,)
    var_std = diag[:, None] * sigma2[None, :]              # (nhod, nbins)
    return coef[1:], var_std


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    names, hod_tab = hod_tables()
    if names is None:
        raise SystemExit('No data/hod_ensemble/hod_params_*.csv; run '
                         'select_hod_ensemble.py first.')
    runs = ready_runs(hod_tab)
    cosmos = sorted(runs)
    if FIDUCIAL not in cosmos or len(cosmos) < 2:
        raise SystemExit(f'Need {FIDUCIAL} and >=1 other cosmology with '
                         f'>= {MIN_DRAWS} draws; ready: '
                         f'{ {c: len(runs[c]) for c in cosmos} }')
    nhod = len(names)
    print('Ready cosmologies (draws):',
          {c: len(runs[c]) for c in cosmos})
    axes_cov = sorted({COSMO_AXIS[c].split()[0] for c in cosmos if c in COSMO_AXIS})
    print('Cosmology axes spanned by this test:', ', '.join(axes_cov) or '(none)')

    # pooled standardisation so per-cosmology slopes are in the same units
    P_all = np.array([hod_tab[c][h] for c in cosmos for h, _ in runs[c]])
    mu, sd = P_all.mean(0), P_all.std(0)

    # per-cosmology, per-stem/ell gradients + variance (physical units)
    grad = {}   # (cosmo, stem, ell) -> (nhod, nbins) physical
    var  = {}
    s = None
    for c in cosmos:
        Z = (np.array([hod_tab[c][h] for h, _ in runs[c]]) - mu) / sd
        for stem in STEMS:
            s_, Ys = load_stack(runs[c], stem)
            if s is None:
                s = s_
            for ell in (0, 2):
                g_std, v_std = fit_gradient(Z, Ys[ell])
                grad[(c, stem, ell)] = g_std / sd[:, None]
                var[(c, stem, ell)]  = v_std / (sd[:, None] ** 2)

    # ---- comparison vs fiducial: chi2/dof and |z|>2 fraction ----
    print(f'\nHOD-gradient agreement vs {FIDUCIAL} '
          f'(full-auto, screening χ²/dof; bins correlated so approximate):')
    print(f'  {"cosmo":<6} {"axis":<6} {"ell":<4} {"chi2/dof":>9} {"|z|>2 frac":>11}')
    for c in cosmos:
        if c == FIDUCIAL:
            continue
        for ell in (0, 2):
            g0 = grad[(FIDUCIAL, 'tpcf_full_data', ell)]
            gc = grad[(c, 'tpcf_full_data', ell)]
            v  = var[(FIDUCIAL, 'tpcf_full_data', ell)] + var[(c, 'tpcf_full_data', ell)]
            z  = (gc - g0) / np.sqrt(v)
            chi2dof = np.nansum(z ** 2) / z.size
            frac = np.mean(np.abs(z) > 2)
            axis = COSMO_AXIS.get(c, '?')
            print(f'  {c:<6} {axis:<6} {ell:<4} {chi2dof:9.2f} {frac:11.2f}')

    # ---- save ----
    out = {'s': s, 'names': np.array(names), 'cosmos': np.array(cosmos)}
    for (c, stem, ell), g in grad.items():
        out[f'{c}_{stem}_g{ell}']   = g
        out[f'{c}_{stem}_var{ell}'] = var[(c, stem, ell)]
    np.savez(DER_DIR / 'hod_gradient_percosmo.npz', **out)
    print(f'\nSaved {DER_DIR / "hod_gradient_percosmo.npz"}')

    # ---- figures: full-auto 3x4 grid per HOD param, overlay cosmologies ----
    for ell in (0, 2):
        fig, axs = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
        for q, ax in enumerate(axs.flat):
            if q >= nhod:
                ax.axis('off'); continue
            for i, c in enumerate(cosmos):
                g = grad[(c, 'tpcf_full_data', ell)][q]
                e = np.sqrt(var[(c, 'tpcf_full_data', ell)][q])
                lbl = f'{c} ({COSMO_AXIS[c]})' if c in COSMO_AXIS else f'{c} (fid)'
                ax.plot(s, s ** 2 * g, color=f'C{i}', lw=1.6, label=lbl)
                ax.fill_between(s, s ** 2 * (g - e), s ** 2 * (g + e),
                                color=f'C{i}', alpha=0.18, lw=0)
            ax.axhline(0, color='grey', lw=0.5)
            ax.set_title(names[q], fontsize=9)
        axs[0, 0].legend(fontsize=7)
        for ax in axs[-1]:
            ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
        fig.suptitle(rf'HOD response $s^2\,\partial\xi_{{{ell}}}/\partial\theta_q$ '
                     f'(full auto) per cosmology — cosmology-independence check', y=1.0)
        fig.tight_layout()
        path = PLOT_DIR / f'hod_response_cosmo_independence_ell{ell}.png'
        fig.savefig(path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()

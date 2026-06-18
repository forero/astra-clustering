#!/usr/bin/env python3
"""
Global response model: one linear fit of the full-box ξ on cosmology *and* HOD
parameters across all completed runs, giving HOD-clean cosmology derivatives and
the HOD gradient in a single step.

This supersedes the Tier-0/Tier-1 derivative path (compute_derivatives_fullbox.py
+ compute_hod_derivatives.py).  Those formed phase-matched ± differences from a
single, HOD-*mismatched* catalog per cosmology and then subtracted a modelled
contamination term.  Here we instead fit, for each statistic bin,

    ξ(θ) ≈ ξ₀ + Σ_p a_p θ_cosmo,p + Σ_q b_q θ_HOD,q

over every completed (cosmo, HOD) full-box run.  Because the HOD term absorbs the
HOD variation explicitly, the cosmology coefficients a_p ARE the HOD-marginalised
cosmology derivatives — the cross-cosmology HOD mismatch no longer contaminates
them.  All runs share ph000, so cosmic variance cancels in the cosmology
coefficients just as it did in the phase-matched difference.

Regressors: [ln ω_b, ln ω_c, n_s, ln σ₈] (from abacus_cosmologies_params.csv) and
the 12 varying yuan23 HOD parameters (from data/hod_ensemble/hod_params_{cosmo}.csv).
All columns are standardised for conditioning, then coefficients are converted back
to physical units.

Outputs
  data/derivatives/derivative_global_{lnwb,lnwc,ns,lns8}.npz
      s, and per (stem, ell): {stem}_dxi{ell}   (drop-in for derivative_hodcorr_*)
  data/derivatives/hod_gradient_global.npz
      s, names, param_mean, param_std_prior, n_runs, and {stem}_g{ell} (12, nbins)
  plots/derivatives/response_global_vs_fd_{param}.png   (global vs FD cross-check)

Usage (any node):
  python scripts/compute_response_global.py
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

N_Q = 4
STEMS = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)

# cosmology regressors: key -> (csv column, take ln?)
COSMO_PARAMS = [
    ('lnwb', 'omega_b',   True),
    ('lnwc', 'omega_cdm', True),
    ('ns',   'n_s',       False),
    ('lns8', 'sigma8_m',  True),
]


def cosmo_regressors():
    """{cosmo tag (c000..): np.array of [lnwb, lnwc, ns, lns8]}."""
    df = pd.read_csv(DATA_DIR / 'abacus_cosmologies_params.csv', index_col=0)
    out = {}
    for tag, row in df.iterrows():
        out[tag] = np.array([np.log(row[c]) if log else row[c]
                             for _, c, log in COSMO_PARAMS])
    return out


def hod_tables():
    """{cosmo: (names, {hod index: param array})} from the ensemble CSVs."""
    tables, names = {}, None
    for f in sorted(ENS_DIR.glob('hod_params_*.csv')):
        cosmo = f.stem.replace('hod_params_', '')
        df = pd.read_csv(f)
        names = list(df.columns[1:])
        tables[cosmo] = {int(r['hod']): r.values[1:].astype(float)
                         for _, r in df.iterrows()}
    return names, tables


def completed_runs(hod_tab):
    """List of (cosmo, hod) full-box runs that are complete and have HOD params."""
    runs = []
    for d in sorted(FB_DIR.glob('c*_hod*')):
        if not (d / 'fullbox_info.npz').is_file():
            continue
        cosmo, hod = d.name.split('_hod')
        hod = int(hod)
        if cosmo in hod_tab and hod in hod_tab[cosmo]:
            runs.append((cosmo, hod))
    return runs


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    creg          = cosmo_regressors()
    names, hod_tab = hod_tables()
    if names is None:
        raise SystemExit('No data/hod_ensemble/hod_params_*.csv; run '
                         'select_hod_ensemble.py first.')
    runs = completed_runs(hod_tab)
    ncos, nhod = len(COSMO_PARAMS), len(names)
    print(f'Completed full-box runs usable in the fit: {len(runs)} '
          f'across {len({c for c, _ in runs})} cosmologies')
    if len(runs) < ncos + nhod + 2:
        raise SystemExit(f'Need > {ncos + nhod + 1} runs to fit '
                         f'{ncos + nhod} parameters; have {len(runs)}.')

    # design matrix: cosmo regressors then HOD regressors (raw, then standardise)
    Xraw = np.array([
        np.concatenate([creg[c], hod_tab[c][h]]) for c, h in runs
    ])                                                       # (nrun, ncos+nhod)
    mu, sd = Xraw.mean(0), Xraw.std(0)
    Z = (Xraw - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])                # (nrun, 1+ncos+nhod)
    print(f'Design-matrix condition number: {np.linalg.cond(A):.2f} '
          f'({len(runs)} runs, {ncos + nhod} params)')

    # fit every stem/ell, convert standardised slopes -> physical
    s = None
    cosmo_der = {p: {} for p, _, _ in COSMO_PARAMS}          # param -> {stem_dxiL}
    hod_grad  = {}                                           # stem_gL -> (nhod,nbins)
    for stem in STEMS:
        ys = {0: [], 2: []}
        for c, h in runs:
            d = np.load(FB_DIR / f'{c}_hod{h:03d}' / f'fullbox_multipoles_{stem}.npz')
            if s is None:
                s = d['s']
            ys[0].append(d['xi0']); ys[2].append(d['xi2'])
        for ell in (0, 2):
            Y = np.array(ys[ell])                            # (nrun, nbins)
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            phys = coef[1:] / sd[:, None]                    # (ncos+nhod, nbins)
            for i, (p, _, _) in enumerate(COSMO_PARAMS):
                cosmo_der[p][f'{stem}_dxi{ell}'] = phys[i]
            hod_grad[f'{stem}_g{ell}'] = phys[ncos:]

    # prior widths from the pooled HOD draws (the yuan23 prior, all cosmologies)
    P_all = np.array([v for tab in hod_tab.values() for v in tab.values()])
    np.savez(DER_DIR / 'hod_gradient_global.npz',
             s=s, names=np.array(names),
             param_mean=P_all.mean(0), param_std_prior=P_all.std(0),
             n_runs=len(runs), **hod_grad)
    print(f'Saved {DER_DIR / "hod_gradient_global.npz"}')
    for p, _, _ in COSMO_PARAMS:
        np.savez(DER_DIR / f'derivative_global_{p}.npz', s=s, **cosmo_der[p])
        print(f'Saved {DER_DIR / f"derivative_global_{p}.npz"}')

    # ---- sanity: full-auto dxi/dln(sigma8) should be ~ 2 xi ----
    xi_fid = np.mean([np.load(FB_DIR / f'{c}_hod{h:03d}'
                              / 'fullbox_multipoles_tpcf_full_data.npz')['xi0']
                      for c, h in runs if c == 'c000'], axis=0)
    dlns8  = cosmo_der['lns8']['tpcf_full_data_dxi0']
    small  = s < 40
    ratio  = np.mean(dlns8[small]) / np.mean(2 * xi_fid[small])
    print(f'Sanity dξ/dlnσ₈ vs 2ξ (full auto ℓ=0, s<40): ratio = {ratio:.2f} '
          f'(expect ≈1)')

    # ---- cross-check figure: global vs finite-difference (hodcorr) ----
    for p, _, _ in COSMO_PARAMS:
        fd = DER_DIR / f'derivative_hodcorr_{p}.npz'
        if not fd.is_file():
            continue
        f = np.load(fd)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for ax, ell in zip(axes, (0, 2)):
            g = cosmo_der[p][f'tpcf_full_data_dxi{ell}']
            ax.plot(s, s**2 * g, 'C3', lw=2, label='global regression')
            ax.plot(s, s**2 * f[f'tpcf_full_data_dxi{ell}'], 'k--', lw=1.6,
                    label='finite-difference (hodcorr)')
            ax.axhline(0, color='grey', lw=0.6)
            ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
            ax.set_title(rf'$\ell={ell}$', fontsize=10)
        axes[0].set_ylabel(rf'$s^2\,\partial\xi/\partial\,${p}')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'Cosmology derivative {p}: global vs FD (full auto, '
                     f'{len(runs)} runs)', y=1.02)
        fig.tight_layout()
        path = PLOT_DIR / f'response_global_vs_fd_{p}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()

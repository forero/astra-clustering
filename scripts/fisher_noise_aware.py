#!/usr/bin/env python3
"""
Improvement #1: noise-aware Fisher with the derivative uncertainty propagated.

A Fisher matrix treats the derivative D as exact.  When D is estimated from noisy
simulations, D = D_true + delta, and E[delta C^-1 delta] = Tr(C^-1 Cov(delta)) is
ALWAYS added to the Fisher -- it reads derivative noise as constraining power.
This inflates the velocity-free random-quadrupole legs (vector_search note).

We propagate the uncertainty of the derivatives that the search actually uses: the
GLOBAL linear response model.  For each data-vector bin we refit
    xi = beta . [1, theta_cosmo (4), theta_HOD (12)]
over all 450 runs (exactly compute_response_global) and read off, per bin, the
fitted cosmology slope AND its standard error
    Var(slope_p)_b = sigma_resid,b^2 * (A^T A)^-1_pp / sd_p^2.
The (A^T A)^-1 part is common to all bins; the bin-to-bin difference is the
residual variance sigma_resid,b^2 -- large exactly where the linear model fits
badly (the noisy random quadrupoles).  This is the right Cov(delta) for these
derivatives, and unlike the per-cosmology GP-at-theta* estimate it is not blown up
by the small 2% cosmology spacing.

Noise-aware Fisher: F = D C^-1 D^T, then subtract Tr(C^-1 Cov(delta_p)) from each
cosmology diagonal (different parameters use different +/- cosmologies, so their
derivative noise is independent -> only the diagonal is corrected).  If the
corrected cosmology block stops being positive-definite, the vector's apparent
gain was pure derivative noise.

Outputs:
  data/derivatives/derivative_global_var_{p}.npz   (per-bin slope variance)
  plots/vector_search/noise_aware_ranking.png      (naive vs noise-aware gains)
Run: python scripts/fisher_noise_aware.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj
import compute_response_global as crg

DER  = crg.DER_DIR
PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
MQ = (0, 2); NB = 15
PARAMS = list(fj.COSMO)                                   # lnwb, lnwc, ns, lns8
N_Q = 4
STEMS = (['tpcf_full_data'] +
         [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
         [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
         [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
         [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)])
RANDLEG = {s for s in STEMS if 'rand' in s}
SHORT = {}
for fam, pre in [('data_q', 'dQ'), ('rand_q', 'rQ'),
                 ('cross_full_data_q', 'xdQ'), ('cross_full_rand_q', 'xrQ')]:
    for q in range(1, 5):
        SHORT[f'tpcf_{fam}{q}'] = f'{pre}{q}'


def fit_global_variance():
    """Per (stem, ell, bin): cosmology-slope variance from the *random* (ASTRA-
    iteration) noise only.  Cosmic variance cancels in the phase-matched cosmology
    difference, so the random part of the global slope comes from the per-run
    ASTRA-realisation scatter sigma_astra^2 = mean_run(xi_std^2)/N_iter -- zero for
    the deterministic full auto, large for the noise-limited random quadrupoles.
    The regression *residual* is NOT used (it is dominated by HOD nonlinearity, a
    systematic, which would massively overstate the random derivative noise)."""
    creg = crg.cosmo_regressors()
    names, hod_tab = crg.hod_tables()
    runs = crg.completed_runs(hod_tab)
    ncos = len(crg.COSMO_PARAMS)
    Xraw = np.array([np.concatenate([creg[c], hod_tab[c][h]]) for c, h in runs])
    mu, sd = Xraw.mean(0), Xraw.std(0)
    A = np.column_stack([np.ones(len(Xraw)), (Xraw - mu) / sd])     # (n, 1+16)
    AtA_inv = np.linalg.inv(A.T @ A)
    diag_cos = np.diag(AtA_inv)[1:1 + ncos]                         # cosmology slopes
    sd_cos = sd[:ncos]
    dvar = {p: {} for p, _, _ in crg.COSMO_PARAMS}
    for stem in STEMS:
        for ell in (0, 2):
            s2, niters = [], []
            for c, h in runs:
                a = np.load(crg.FB_DIR / f'{c}_hod{h:03d}'
                            / f'fullbox_multipoles_{stem}.npz')
                s2.append(a[f'xi{ell}_std'] ** 2)
                niters.append(a[f'xi{ell}_all'].shape[0])
            sig2_astra = np.mean(s2, 0) / np.mean(niters)           # (nbins,) random only
            for i, (p, _, _) in enumerate(crg.COSMO_PARAMS):
                dvar[p][f'{stem}_dxi{ell}'] = sig2_astra * diag_cos[i] / sd_cos[i] ** 2
    for p, _, _ in crg.COSMO_PARAMS:
        np.savez(DER / f'derivative_global_var_{p}.npz', **dvar[p])
    print(f'Saved derivative_global_var_{{{",".join(p for p,_,_ in crg.COSMO_PARAMS)}}}.npz')
    return dvar


def main():
    dvar = fit_global_variance()
    ncos = len(PARAMS)

    def fom3(pieces, noise_aware, report=False):
        a = fj.assemble(pieces); Cinv = a['Cinv']; nb = a['nb']
        D = np.vstack([a['D_cos'], a['D_hod']])
        F = D @ Cinv @ D.T
        B = np.zeros(ncos)
        if noise_aware:
            cd = np.diag(Cinv); col = 0
            Vc = np.zeros((ncos, nb))
            for stem, ells, k in pieces:
                for ell in ells:
                    for i, p in enumerate(PARAMS):
                        Vc[i, col:col + NB] = dvar[p][f'{stem}_dxi{ell}']
                    col += NB
            B = np.array([float((cd * Vc[i]).sum()) for i in range(ncos)])
            for i in range(ncos):
                F[i, i] -= B[i]
        pd = np.all(np.linalg.eigvalsh(F[:ncos, :ncos]) > 0)
        Fp = F.copy(); Fp[ncos:, ncos:] += np.diag(1.0 / a['sd_pr'] ** 2)
        cov = fj.to_phys_cov(np.linalg.inv(Fp)[:ncos, :ncos], PARAMS)
        ok = pd and np.all(np.diag(cov[:3, :3]) > 0)
        if report:
            Fdiag = (D @ Cinv @ D.T).diagonal()[:ncos]
            print('   bias/F per cosmo param:',
                  ', '.join(f'{p}={B[i]/Fdiag[i]:.2f}' for i, p in enumerate(PARAMS)))
        return (np.linalg.slogdet(cov[:3, :3])[1] if ok else np.nan), ok

    base = [('tpcf_full_data', MQ, 1)]
    ref_ld, _ = fom3(base, False)
    g = lambda ld: float(np.exp(-0.5 * (ld - ref_ld))) if np.isfinite(ld) else np.nan
    print('\nFull-auto bias check (should be small -- signal-rich):')
    fom3(base, True, report=True)
    print(f'Full auto: naive FoM3 gain {g(fom3(base,False)[0]):.2f}, '
          f'noise-aware {g(fom3(base,True)[0]):.2f}\n')

    print(f'{"+stem":7s} {"naive":>7s} {"noise-aware":>12s}   leg')
    rows = []
    for s in STEMS[1:]:
        pieces = base + [(s, MQ, 1)]
        gn = g(fom3(pieces, False)[0])
        lda, ok = fom3(pieces, True); ga = g(lda)
        rows.append((SHORT[s], gn, ga, s in RANDLEG, ok))
        ga_s = f'{ga:.2f}' if np.isfinite(ga) else 'noise-dom.'
        print(f'{SHORT[s]:7s} {gn:7.2f} {ga_s:>12s}   {"RAND" if s in RANDLEG else "data"}')

    # ---- figure ----
    rows.sort(key=lambda r: -r[1])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label='naive Fisher', color='C7')
    ax.bar(x + 0.2, [r[2] if np.isfinite(r[2]) else 0 for r in rows], 0.4,
           label='noise-aware (derivative variance subtracted)', color='C0')
    for i, r in enumerate(rows):
        if r[3]:
            ax.text(i, 0.15, '*', color='C3', ha='center', fontsize=13)
        if not np.isfinite(r[2]):
            ax.text(i + 0.2, 0.3, 'noise-dom.', color='C3', ha='center', rotation=90, fontsize=6)
    ax.axhline(1, color='k', lw=0.8, ls='--', label='full auto')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=45, ha='right')
    ax.set_ylabel('FoM3 gain over full auto'); ax.legend()
    ax.set_title('Noise-aware re-ranking: global-derivative variance subtracted '
                 '(* = random leg)')
    fig.tight_layout(); fig.savefig(PLOT / 'noise_aware_ranking.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'\nSaved {PLOT / "noise_aware_ranking.png"}')


if __name__ == '__main__':
    main()

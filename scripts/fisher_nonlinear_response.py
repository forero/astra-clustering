#!/usr/bin/env python3
"""
Improvement #3: nonlinear (quadratic-in-HOD) global response model.

#1 showed the random-leg gains are not a derivative-*noise* artefact.  The
remaining worry is a derivative *bias*: the global model is linear in the HOD,
but the true HOD response is mildly nonlinear (GP captured ~75% vs ~18% for the
line).  A mis-modelled HOD term can leak into the jointly-fitted cosmology slope.

We test this by refitting the global response with HOD nonlinearity included:
    xi = c0 + sum_p A_p z_cosmo,p
            + sum_q B_q z_HOD,q + sum_q C_q z_HOD,q^2 + sum_{q<r} D_qr z_HOD,q z_HOD,r
where the regressors are centred at the Fisher fiducial (c000, hod484), so the
first derivative at the fiducial is just the linear coefficient (A_p, B_q); the
quadratic terms only change those coefficients by absorbing HOD curvature.  We
then (a) compare the nonlinear cosmology derivatives to the linear ones per stem,
(b) check the sigma8 sanity (dxi/dlnsigma8 vs 2xi), and (c) recompute the Fisher
FoM3 ranking with the nonlinear derivatives.  If the random-leg gains move a lot,
they were nonlinearity-biased; if they are stable, they are robust.

Outputs:
  data/derivatives/derivative_nl_{p}.npz, hod_gradient_nl.npz
  plots/vector_search/nonlinear_response.png   (derivative change + re-ranking)
Run: python scripts/fisher_nonlinear_response.py
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj
import compute_response_global as crg

DER = crg.DER_DIR
PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
MQ = (0, 2); NB = 15
PARAMS = list(fj.COSMO)
FID_HOD = 484
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


def designs():
    """Linear and quadratic design matrices, both centred at the fiducial."""
    creg = crg.cosmo_regressors()
    names, hod_tab = crg.hod_tables()
    runs = crg.completed_runs(hod_tab)
    cfid = creg['c000']; hfid = hod_tab['c000'][FID_HOD]
    Zc = np.array([creg[c] - cfid for c, h in runs])              # (n,4) centred
    Zh = np.array([hod_tab[c][h] - hfid for c, h in runs])        # (n,12) centred
    sdc, sdh = Zc.std(0), Zh.std(0)
    zc, zh = Zc / sdc, Zh / sdh                                   # standardised
    n = len(runs); ncos = 4
    # linear design
    A_lin = np.column_stack([np.ones(n), zc, zh])
    # quadratic: + squares + pairwise HOD interactions
    sq = zh ** 2
    inter = np.column_stack([zh[:, i] * zh[:, j] for i, j in combinations(range(12), 2)])
    A_quad = np.column_stack([A_lin, sq, inter])
    return runs, A_lin, A_quad, sdc, sdh, ncos


def fit(A, runs, sdc, ncos):
    """Return cosmology derivatives {p:{stem_dxi:..}} and HOD linear coefs."""
    AtA_inv = np.linalg.inv(A.T @ A)
    der = {p: {} for p in PARAMS}; grad = {}
    for stem in STEMS:
        for ell in (0, 2):
            Y = np.array([np.load(crg.FB_DIR / f'{c}_hod{h:03d}'
                          / f'fullbox_multipoles_{stem}.npz')[f'xi{ell}'] for c, h in runs])
            coef = AtA_inv @ (A.T @ Y)                            # (npar, nbins)
            for i, p in enumerate(PARAMS):
                der[p][f'{stem}_dxi{ell}'] = coef[1 + i] / sdc[i]  # cosmo slope at fiducial
            grad[f'{stem}_g{ell}'] = coef[1 + ncos:1 + ncos + 12]  # HOD linear (standardised)
    return der, grad, np.linalg.cond(A)


def main():
    runs, A_lin, A_quad, sdc, sdh, ncos = designs()
    der_lin, _, cond_lin = fit(A_lin, runs, sdc, ncos)
    der_nl, grad_nl_std, cond_nl = fit(A_quad, runs, sdc, ncos)
    print(f'design condition numbers: linear {cond_lin:.1f}, quadratic {cond_nl:.1f} '
          f'({len(runs)} runs, {A_quad.shape[1]} params)')

    # save nonlinear derivatives + HOD gradient (physical, at fiducial)
    g0 = np.load(DER / 'hod_gradient_global.npz', allow_pickle=True)
    s = g0['s']
    for p in PARAMS:
        np.savez(DER / f'derivative_nl_{p}.npz', s=s, **der_nl[p])
    grad_phys = {k: v / sdh[:, None] for k, v in grad_nl_std.items()}
    np.savez(DER / 'hod_gradient_nl.npz', s=s, names=g0['names'],
             param_mean=g0['param_mean'], param_std_prior=g0['param_std_prior'],
             n_runs=len(runs), **grad_phys)
    print('Saved derivative_nl_*.npz, hod_gradient_nl.npz')

    # sigma8 sanity: dxi/dlnsigma8 vs 2 xi (full auto l0, s<40)
    xi_fid = np.mean([np.load(crg.FB_DIR / f'c000_hod{h:03d}'
                      / 'fullbox_multipoles_tpcf_full_data.npz')['xi0']
                      for c, h in runs if c == 'c000'], axis=0)
    sm = s < 40
    for tag, der in [('linear', der_lin), ('nonlinear', der_nl)]:
        r = np.mean(der['lns8']['tpcf_full_data_dxi0'][sm]) / np.mean(2 * xi_fid[sm])
        print(f'  sigma8 sanity ({tag}): dxi/dlnsigma8 / 2xi = {r:.2f}')

    # per-stem derivative change (full auto l0 reference scale)
    print('\nrelative change |D_nl - D_lin| / rms(D_lin) per stem (cosmo-avg):')
    chg = {}
    for stem in STEMS:
        num = den = 0.0
        for p in PARAMS:
            for ell in (0, 2):
                dl = der_lin[p][f'{stem}_dxi{ell}']; dn = der_nl[p][f'{stem}_dxi{ell}']
                num += np.sum((dn - dl) ** 2); den += np.sum(dl ** 2)
        chg[stem] = np.sqrt(num / den)
    for stem in STEMS:
        print(f'  {SHORT.get(stem,"full"):6s} {chg[stem]:.2f}'
              + ('   RAND' if stem in RANDLEG else ''))

    # recompute Fisher FoM3 gains, linear vs nonlinear derivatives
    def fom3_with(der_fmt, grad_file):
        fj.deriv_source = lambda: ('x', der_fmt, grad_file)
        base = [('tpcf_full_data', MQ, 1)]
        def f3(pieces):
            cov = fj.to_phys_cov(fj.fisher(pieces)['cov_marg'], PARAMS)
            return np.linalg.slogdet(cov[:3, :3])[1]
        ref = f3(base)
        out = {}
        for stem in STEMS[1:]:
            out[stem] = float(np.exp(-0.5 * (f3(base + [(stem, MQ, 1)]) - ref)))
        return out
    gain_lin = fom3_with('derivative_global_{p}.npz', 'hod_gradient_global.npz')
    gain_nl = fom3_with('derivative_nl_{p}.npz', 'hod_gradient_nl.npz')

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    order = sorted(STEMS[1:], key=lambda st: -gain_lin[st])
    x = np.arange(len(order))
    a1.bar([SHORT[s] for s in order], [chg[s] for s in order],
           color=['C3' if s in RANDLEG else 'C2' for s in order])
    a1.set_ylabel('relative derivative change (nl vs lin)')
    a1.set_title('(a) how much HOD nonlinearity moves the cosmology derivative')
    a1.tick_params(axis='x', rotation=45)
    a2.bar(x - 0.2, [gain_lin[s] for s in order], 0.4, label='linear', color='C7')
    a2.bar(x + 0.2, [gain_nl[s] for s in order], 0.4, label='nonlinear (HOD quad)', color='C0')
    for i, s in enumerate(order):
        if s in RANDLEG:
            a2.text(i, 0.15, '*', color='C3', ha='center', fontsize=12)
    a2.axhline(1, color='k', lw=0.8, ls='--'); a2.set_xticks(x)
    a2.set_xticklabels([SHORT[s] for s in order], rotation=45, ha='right')
    a2.set_ylabel('FoM3 gain over full auto'); a2.legend()
    a2.set_title('(b) Fisher ranking: linear vs nonlinear derivatives (* = random)')
    fig.suptitle('Nonlinear-response test of the random-leg cosmology derivatives', y=1.02)
    fig.tight_layout(); fig.savefig(PLOT / 'nonlinear_response.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'\nSaved {PLOT / "nonlinear_response.png"}')


if __name__ == '__main__':
    main()

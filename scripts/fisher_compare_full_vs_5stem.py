#!/usr/bin/env python3
"""
Fisher corner comparison (noise-aware, HOD-marginalised) of the standard 2PCF
against the optimal ASTRA vectors -- including the optimal constraints driven by
the void/knot random crosses xrQ4 + xrQ1 (the first two picks of the all-stems
greedy chain, which alone take FoM3 from 1 to ~48x).

Four overlaid constraints:
  * full auto (2PCF)                            baseline
  * full + xrQ4 + xrQ1                          the optimal random-cross pair
  * data-leg optimal (5 stem)                   survey-robust subset
  * all-stems optimal (6 stem)                  the best: full + xrQ4 + xrQ1 +
                                                xdQ3 + xdQ4 + dQ1 + xrQ3
All use the noise-aware Fisher (derivative-noise bias subtracted, matching
greedy_chain_all). Needs data/derivatives/derivative_global_var_*.npz.
Output: plots/vector_search/fisher_compare_full_vs_5stem.png
"""
from pathlib import Path
import numpy as np
import fisher_joint as fj

PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
DER = fj.DER_DIR
MQ = (0, 2); NB = 15
PARAMS = list(fj.COSMO); ncos = len(PARAMS)
DVAR = {p: np.load(DER / f'derivative_global_var_{p}.npz') for p in PARAMS}

VECTORS = {
    'full auto (2PCF)': ['tpcf_full_data'],
    'full + xrQ4 + xrQ1 (optimal pair)': ['tpcf_full_data',
        'tpcf_cross_full_rand_q4', 'tpcf_cross_full_rand_q1'],
    'data-leg optimal (5 stem)': ['tpcf_full_data', 'tpcf_cross_full_data_q1',
        'tpcf_data_q1', 'tpcf_data_q2', 'tpcf_cross_full_data_q3', 'tpcf_data_q4'],
    'all-stems optimal (6 stem)': ['tpcf_full_data', 'tpcf_cross_full_rand_q4',
        'tpcf_cross_full_rand_q1', 'tpcf_cross_full_data_q3',
        'tpcf_cross_full_data_q4', 'tpcf_data_q1', 'tpcf_cross_full_rand_q3'],
}
COLORS = {'full auto (2PCF)': '#888888',
          'full + xrQ4 + xrQ1 (optimal pair)': '#ff7f0e',
          'data-leg optimal (5 stem)': '#2ca02c',
          'all-stems optimal (6 stem)': '#1f77b4'}


def noise_aware_cov(stems):
    """Noise-aware HOD-marginalised 4x4 cosmology covariance (ln units)."""
    pieces = [(s, MQ, 1) for s in stems]
    a = fj.assemble(pieces); Cinv = a['Cinv']; nb = a['nb']
    D = np.vstack([a['D_cos'], a['D_hod']]); F = D @ Cinv @ D.T
    cd = np.diag(Cinv); Vc = np.zeros((ncos, nb)); col = 0
    for stem, ells, k in pieces:
        for ell in ells:
            for i, p in enumerate(PARAMS):
                Vc[i, col:col + NB] = DVAR[p][f'{stem}_dxi{ell}']
            col += NB
    for i in range(ncos):
        F[i, i] -= float((cd * Vc[i]).sum())
    Fp = F.copy(); Fp[ncos:, ncos:] += np.diag(1.0 / a['sd_pr'] ** 2)
    return np.linalg.inv(Fp)[:ncos, :ncos]


def main():
    covs = {n: noise_aware_cov(s) for n, s in VECTORS.items()}
    phys = {n: fj.to_phys_cov(c, PARAMS) for n, c in covs.items()}
    base = phys['full auto (2PCF)']; bsig = np.sqrt(np.diag(base))
    print('Noise-aware HOD-marginalised forecast (sigma; factor vs 2PCF; FoM3):')
    print(f'  {"vector":34s} ' + '  '.join(f'{p:>9s}' for p in PARAMS) + '   FoM3')
    for n in VECTORS:
        sig = np.sqrt(np.diag(phys[n]))
        g3 = np.exp(-0.5 * (np.linalg.slogdet(phys[n][:3, :3])[1]
                            - np.linalg.slogdet(base[:3, :3])[1]))
        fac = '  '.join(f'{bsig[i]/sig[i]:7.1f}x' if n != 'full auto (2PCF)'
                        else f'{sig[i]:9.2e}' for i in range(ncos))
        print(f'  {n:34s} {fac}   {g3:5.0f}x')

    fj.corner([(n, covs[n], COLORS[n]) for n in VECTORS], PARAMS,
              PLOT / 'fisher_compare_full_vs_5stem.png',
              'Noise-aware HOD-marginalised forecast: 2PCF vs ASTRA optimal vectors '
              '(the optimal constraints come from the xrQ4+xrQ1 random crosses)',
              ranges_from=base)


if __name__ == '__main__':
    main()

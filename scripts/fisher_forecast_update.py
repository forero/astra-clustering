#!/usr/bin/env python3
"""
Updated Fisher forecast with the new optimal ASTRA stems.

Compares three HOD-marginalised, NOISE-AWARE forecasts (derivative-noise bias of
fisher_noise_aware.py subtracted):
  * full auto (the standard 2PCF baseline)
  * data-leg optimal      full + xdQ1 + dQ1 + dQ2 + xdQ3 + dQ4   (survey-robust)
  * all-stems optimal     full + xrQ4 + xrQ1 + xdQ3 + xdQ4 + dQ1 + xrQ3  (box ceiling)

Produces the corner plot (68/95% ellipses) and the sigma table.  Needs
data/derivatives/derivative_global_var_*.npz from fisher_noise_aware.py.
Output: plots/vector_search/fisher_forecast_compare.png
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
    'data-leg optimal (5 stem)': ['tpcf_full_data', 'tpcf_cross_full_data_q1',
        'tpcf_data_q1', 'tpcf_data_q2', 'tpcf_cross_full_data_q3', 'tpcf_data_q4'],
    'all-stems optimal (6 stem)': ['tpcf_full_data', 'tpcf_cross_full_rand_q4',
        'tpcf_cross_full_rand_q1', 'tpcf_cross_full_data_q3',
        'tpcf_cross_full_data_q4', 'tpcf_data_q1', 'tpcf_cross_full_rand_q3'],
}
COLORS = {'full auto (2PCF)': '#888888', 'data-leg optimal (5 stem)': '#2ca02c',
          'all-stems optimal (6 stem)': '#1f77b4'}


def noise_aware_cov(stems):
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
    return np.linalg.inv(Fp)[:ncos, :ncos]                       # ln units, 4x4


def main():
    covs = {name: noise_aware_cov(stems) for name, stems in VECTORS.items()}
    phys = {name: fj.to_phys_cov(c, PARAMS) for name, c in covs.items()}
    base = phys['full auto (2PCF)']
    bsig = np.sqrt(np.diag(base))
    print('Noise-aware HOD-marginalised forecast (sigma; factor vs 2PCF):')
    print(f'  {"vector":28s} ' + '  '.join(f'{p:>9s}' for p in PARAMS))
    for name in VECTORS:
        sig = np.sqrt(np.diag(phys[name]))
        print(f'  {name:28s} ' + '  '.join(f'{sig[i]:.2e}' for i in range(ncos)))
        if name != 'full auto (2PCF)':
            print(f'  {"  -> x tighter":28s} '
                  + '  '.join(f'{bsig[i]/sig[i]:8.1f}x' for i in range(ncos)))
            g3 = np.exp(-0.5 * (np.linalg.slogdet(phys[name][:3, :3])[1]
                                - np.linalg.slogdet(base[:3, :3])[1]))
            print(f'  {"  -> FoM3 gain":28s} {g3:.0f}x')

    cov_sets = [(name, covs[name], COLORS[name]) for name in VECTORS]
    fj.corner(cov_sets, PARAMS, PLOT / 'fisher_forecast_compare.png',
              'Updated Fisher forecast (noise-aware, HOD-marginalised): '
              '2PCF vs ASTRA optimal vectors', ranges_from=base)


if __name__ == '__main__':
    main()

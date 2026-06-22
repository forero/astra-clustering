#!/usr/bin/env python3
"""
Fisher corner comparison (noise-aware, HOD-marginalised): the standard 2PCF
(full-sample auto) vs the two-stem optimal ASTRA vector full + xrQ4 + xrQ1 (the
void/knot full-cross random crosses, the first two picks of the greedy chain).
The FoM3 gain is printed and shown on the figure.

Needs data/derivatives/derivative_global_var_*.npz (fisher_noise_aware.py).
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
    'full + xrQ4 + xrQ1': ['tpcf_full_data', 'tpcf_cross_full_rand_q4',
                           'tpcf_cross_full_rand_q1'],
}
COLORS = {'full auto (2PCF)': '#888888', 'full + xrQ4 + xrQ1': '#1f77b4'}


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
    opt = phys['full + xrQ4 + xrQ1']; osig = np.sqrt(np.diag(opt))
    g3 = float(np.exp(-0.5 * (np.linalg.slogdet(opt[:3, :3])[1]
                              - np.linalg.slogdet(base[:3, :3])[1])))
    g4 = float(np.exp(-0.5 * (np.linalg.slogdet(opt)[1]
                              - np.linalg.slogdet(base)[1])))
    fac = {p: bsig[i] / osig[i] for i, p in enumerate(PARAMS)}
    print('Noise-aware HOD-marginalised forecast:')
    print(f'  {"param":6s} {"2PCF":>10s} {"full+xrQ4+xrQ1":>16s} {"x tighter":>10s}')
    for i, p in enumerate(PARAMS):
        print(f'  {p:6s} {bsig[i]:10.2e} {osig[i]:16.2e} {fac[p]:9.1f}x')
    fom_txt = (f'FoM3 = {g3:.0f}x  (FoM4 = {g4:.0f}x);  '
               f'wb {fac["lnwb"]:.1f}x  wc {fac["lnwc"]:.1f}x  '
               f'ns {fac["ns"]:.1f}x  s8 {fac["lns8"]:.1f}x')
    print('\n' + fom_txt)

    fj.corner([(n, covs[n], COLORS[n]) for n in VECTORS], PARAMS,
              PLOT / 'fisher_compare_full_vs_5stem.png',
              '2PCF vs ASTRA full + xrQ4 + xrQ1 (noise-aware, HOD-marginalised)\n'
              + fom_txt, ranges_from=base)


if __name__ == '__main__':
    main()

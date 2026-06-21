#!/usr/bin/env python3
"""
Fisher corner comparison: the full-sample auto-correlation (standard 2PCF) vs the
5-stem greedy ASTRA vector (full + xdQ1 + dQ1 + dQ2 + xdQ3 + dQ4), HOD-marginalised.
Output: plots/vector_search/fisher_compare_full_vs_5stem.png.
Run: python scripts/fisher_compare_full_vs_5stem.py
"""
from pathlib import Path
import numpy as np
import fisher_joint as fj

PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
MQ = (0, 2)
PARAMS = list(fj.COSMO)
FIVE = ['tpcf_cross_full_data_q1', 'tpcf_data_q1', 'tpcf_data_q2',
        'tpcf_cross_full_data_q3', 'tpcf_data_q4']                 # xdQ1 dQ1 dQ2 xdQ3 dQ4


def main():
    full = [('tpcf_full_data', MQ, 1)]
    five = full + [(s, MQ, 1) for s in FIVE]
    cov_full = fj.fisher(full)['cov_marg']
    cov_five = fj.fisher(five)['cov_marg']

    pf = fj.to_phys_cov(cov_full, PARAMS); pv = fj.to_phys_cov(cov_five, PARAMS)
    sf, sv = np.sqrt(np.diag(pf)), np.sqrt(np.diag(pv))
    print(f'Derivatives: {fj.deriv_source()[0]};  covariance: pooled 576 subboxes\n')
    print(f'{"param":8s} {"full auto":>11s} {"5-stem":>11s} {"factor":>7s}')
    for i, p in enumerate(PARAMS):
        print(f'{p:8s} {sf[i]:11.3e} {sv[i]:11.3e} {sf[i]/sv[i]:6.2f}x')
    g3 = np.exp(-0.5 * (np.linalg.slogdet(pv[:3, :3])[1] - np.linalg.slogdet(pf[:3, :3])[1]))
    print(f'\nFoM3 (wb,wc,ns) gain of 5-stem over full auto: {g3:.1f}x')

    fj.corner([('full auto (2PCF)', cov_full, '#888888'),
               ('5-stem ASTRA', cov_five, '#1f77b4')],
              PARAMS, PLOT / 'fisher_compare_full_vs_5stem.png',
              'Fisher 68/95%: full-auto 2PCF vs 5-stem ASTRA '
              '(full + xdQ1 + dQ1 + dQ2 + xdQ3 + dQ4), HOD-marginalised',
              ranges_from=pf)


if __name__ == '__main__':
    main()

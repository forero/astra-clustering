#!/usr/bin/env python3
"""
Greedy next-addition search: fix a base data vector (default the best clean pair,
full auto + dQ1 + xdQ1) and rank which remaining ASTRA stem most improves the
HOD-marginalised Fisher FoM, both vs the simple full-auto baseline and vs the
current base. Random-leg additions are flagged (derivative-noise artefact; see the
vector_search note). Run: python scripts/fisher_vector_addto.py
"""
from pathlib import Path
import numpy as np
import fisher_joint as fj

MQ = (0, 2)
PARAMS = list(fj.COSMO)
FAMILY = {'dQ':  [f'tpcf_data_q{q}' for q in range(1, 5)],
          'rQ':  [f'tpcf_rand_q{q}' for q in range(1, 5)],
          'xdQ': [f'tpcf_cross_full_data_q{q}' for q in range(1, 5)],
          'xrQ': [f'tpcf_cross_full_rand_q{q}' for q in range(1, 5)]}
STEMS = [s for fam in FAMILY.values() for s in fam]
LABEL = {s: f'{fam}{q}' for fam, lst in FAMILY.items()
         for q, s in zip(range(1, 5), lst)}
RANDLEG = set(FAMILY['rQ'] + FAMILY['xrQ'])
BASE_STEMS = ['tpcf_data_q1', 'tpcf_cross_full_data_q1']      # dQ1 + xdQ1


def logdets(pieces):
    r = fj.fisher(pieces)
    cov = fj.to_phys_cov(r['cov_marg'], PARAMS)
    return np.linalg.slogdet(cov)[1], np.linalg.slogdet(cov[:3, :3])[1], np.sqrt(np.diag(cov))


def main():
    full = [('tpcf_full_data', MQ, 1)]
    base = full + [(s, MQ, 1) for s in BASE_STEMS]
    bl4, bl3, bsig = logdets(full)                            # simple full auto
    cl4, cl3, csig = logdets(base)                            # full + dQ1 + xdQ1
    g = lambda ld, ref: float(np.exp(-0.5 * (ld - ref)))
    print(f'Derivatives: {fj.deriv_source()[0]};  covariance: pooled 576 subboxes')
    print(f'baseline   = full auto                : FoM3 gain 1.00')
    print(f'current    = full + dQ1 + xdQ1        : FoM3 gain {g(cl3, bl3):.2f} '
          f'(vs full auto)\n')
    print(f'{"+ add stem":10s} {"FoM3/full":>9s} {"FoM4/full":>9s} {"FoM3/current":>12s} '
          f'  sig_wb    sig_wc    sig_ns')
    rows = []
    for s in STEMS:
        if s in BASE_STEMS:
            continue
        ld4, ld3, sig = logdets(base + [(s, MQ, 1)])
        rows.append((g(ld3, bl3), g(ld4, bl4), g(ld3, cl3), sig, s))
    for g3f, g4f, g3c, sig, s in sorted(rows, reverse=True):
        flag = '*' if s in RANDLEG else ' '
        print(f'{LABEL[s]:9s}{flag}{g3f:9.2f} {g4f:9.2f} {g3c:12.2f}   '
              + '  '.join(f'{sig[i]:7.1e}' for i in range(3)))
    clean = [r for r in rows if r[4] not in RANDLEG]
    best = max(clean)
    print(f'\nBest CLEAN (data-leg) addition: +{LABEL[best[4]]}  -> '
          f'FoM3 gain {best[0]:.1f} vs full auto, {best[2]:.2f}x on top of current.')
    print('  per-param sigma: ' + ', '.join(
        f'{p}={best[3][i]:.2e}' for i, p in enumerate(PARAMS)))


if __name__ == '__main__':
    main()

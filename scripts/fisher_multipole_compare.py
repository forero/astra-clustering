#!/usr/bin/env python3
"""
Is the monopole, the quadrupole, or both that drives the cosmology gain?

Two tests, HOD-marginalised, global derivatives, pooled 576-subbox covariance:

  A. config comparison -- for the full auto and the 5-stem ASTRA vector, score
     mono-only (l0), quad-only (l2), and mono+quad. FoM3 gain vs the standard
     2PCF (full auto, mono+quad).
  B. greedy 'poll' -- treat each (piece, multipole) as one of 12 selectable units
     of the 5-stem vector and greedily add the unit that most increases FoM3,
     recording whether each pick is a monopole or a quadrupole.

Output: plots/vector_search/multipole_compare.png   +   printed tables.
Run: python scripts/fisher_multipole_compare.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj

PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
PARAMS = list(fj.COSMO)
STEMS5 = ['tpcf_full_data', 'tpcf_cross_full_data_q1', 'tpcf_data_q1',
          'tpcf_data_q2', 'tpcf_cross_full_data_q3', 'tpcf_data_q4']
SHORT = ['full', 'xdQ1', 'dQ1', 'dQ2', 'xdQ3', 'dQ4']
CONFIGS = {'mono (l0)': (0,), 'quad (l2)': (2,), 'mono+quad': (0, 2)}


def stats(pieces):
    r = fj.fisher(pieces)
    cov = fj.to_phys_cov(r['cov_marg'], PARAMS)
    ld3 = np.linalg.slogdet(cov[:3, :3])[1]
    return ld3, np.sqrt(np.diag(cov)), r['nb'], r['hart']


def main():
    print(f'Derivatives: {fj.deriv_source()[0]};  pooled 576-subbox covariance\n')
    # reference: standard 2PCF = full auto mono+quad
    ref3, refsig, _, _ = stats([('tpcf_full_data', (0, 2), 1)])
    gain = lambda ld: float(np.exp(-0.5 * (ld - ref3)))

    # ---- A. config comparison ----
    print('A. config comparison (FoM3 gain vs full-auto mono+quad):')
    print(f'  {"vector":9s} {"config":11s} {"nb":>3s} {"hart":>5s} {"FoM3":>6s}   '
          + '  '.join(f'sig_{p}' for p in PARAMS))
    barvals = {}
    for vname, stems in [('full auto', ['tpcf_full_data']), ('5-stem', STEMS5)]:
        for cname, ells in CONFIGS.items():
            ld3, sig, nb, hart = stats([(s, ells, 1) for s in stems])
            barvals[(vname, cname)] = gain(ld3)
            print(f'  {vname:9s} {cname:11s} {nb:3d} {hart:5.2f} {gain(ld3):6.2f}   '
                  + '  '.join(f'{v:7.1e}' for v in sig))

    # ---- B. greedy poll over the 12 (piece, multipole) units ----
    units = [(s, (ell,), lab + ('-l0' if ell == 0 else '-l2'))
             for s, lab in zip(STEMS5, SHORT) for ell in (0, 2)]
    print('\nB. greedy poll over the 12 (piece x multipole) units of the 5-stem vector:')
    chosen, picks, hist = [], [], []
    f0 = None
    for step in range(1, len(units) + 1):
        best = None
        for u in units:
            if u in chosen:
                continue
            ld3, sig, nb, hart = stats([(s, e, 1) for s, e, _ in chosen + [u]])
            if best is None or ld3 < best[0]:
                best = (ld3, u, sig)
        chosen.append(best[1]); picks.append(best[1][2])
        g = gain(best[0]); hist.append(g)
        kind = 'l2' if best[1][2].endswith('l2') else 'l0'
        print(f'  step {step:2d}: +{best[1][2]:9s} ({kind})  FoM3={g:6.2f}')
    n_l0 = sum(p.endswith('l0') for p in picks[:6])
    print(f'  -> of the first 6 picks: {n_l0} monopoles, {6-n_l0} quadrupoles')

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(3); w = 0.38
    for k, v in enumerate(['full auto', '5-stem']):
        a1.bar(x + (k - 0.5) * w, [barvals[(v, c)] for c in CONFIGS], w, label=v)
    a1.set_xticks(x); a1.set_xticklabels(list(CONFIGS)); a1.set_yscale('log')
    a1.axhline(1, color='k', lw=0.8, ls='--'); a1.set_ylabel('FoM3 gain vs full-auto mono+quad')
    a1.set_title('A. monopole vs quadrupole vs both'); a1.legend()
    cols = ['C0' if p.endswith('l0') else 'C3' for p in picks]
    a2.bar(range(1, 13), hist, color=cols)
    for i, p in enumerate(picks):
        a2.text(i + 1, hist[i], p[:-3], rotation=90, va='bottom', ha='center', fontsize=7)
    a2.set_xlabel('greedy step (blue = monopole picked, red = quadrupole)')
    a2.set_ylabel('FoM3 gain'); a2.set_title('B. greedy poll over (piece x multipole) units')
    fig.suptitle('Monopole / quadrupole information in the 5-stem ASTRA vector', y=1.02)
    fig.tight_layout(); fig.savefig(PLOT / 'multipole_compare.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'\nSaved {PLOT / "multipole_compare.png"}')


if __name__ == '__main__':
    main()

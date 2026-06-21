#!/usr/bin/env python3
"""
Greedy forward selection over the trustworthy DATA-leg ASTRA stems.

Start from the full-sample auto-correlation and, at each step, append the single
data-quantile auto or full x data-quantile cross that most increases the
HOD-marginalised Fisher FoM3 (omega_b, omega_c, n_s; sigma8 dropped -- unreliable
derivative). Random legs are excluded: the vector_search note shows their gains are
a derivative-noise artefact. Goes up to 5 added stems and reports the chain plus a
diminishing-returns figure.

Output: plots/vector_search/greedy_chain.png   +   printed chain table.
Run: python scripts/fisher_greedy_chain.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj

PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
MQ = (0, 2)
PARAMS = list(fj.COSMO)
NSTEPS = 5

# data-leg candidate pool (no random legs) with short labels
POOL = {f'tpcf_data_q{q}': f'dQ{q}' for q in range(1, 5)}
POOL.update({f'tpcf_cross_full_data_q{q}': f'xdQ{q}' for q in range(1, 5)})


def cov_of(pieces):
    return fj.to_phys_cov(fj.fisher(pieces)['cov_marg'], PARAMS)


def fom(cov, k):                      # 1/sqrt(det) of the leading k x k block
    return np.exp(-0.5 * np.linalg.slogdet(cov[:k, :k])[1])


def main():
    full = [('tpcf_full_data', MQ, 1)]
    base_cov = cov_of(full)
    f3_0, f4_0 = fom(base_cov, 3), fom(base_cov, 4)
    bsig = np.sqrt(np.diag(base_cov))
    print(f'Derivatives: {fj.deriv_source()[0]};  covariance: pooled 576 subboxes')
    print(f'\nGreedy chain (data legs only), FoM gain vs the full-auto 2PCF:\n')
    print(f'{"step":4s} {"+add":6s} {"FoM3":>6s} {"FoM4":>6s} {"marg.":>6s}   '
          + '  '.join(f'sig_{p:<4s}' for p in PARAMS))
    print(f'{"0":4s} {"full":6s} {1.0:6.2f} {1.0:6.2f} {"--":>6s}   '
          + '  '.join(f'{v:8.1e}' for v in bsig))

    chosen, pieces, prev3 = [], list(full), f3_0
    hist = [(0, 'full', 1.0, 1.0, bsig)]
    for step in range(1, NSTEPS + 1):
        best = None
        for stem, lab in POOL.items():
            if stem in chosen:
                continue
            cov = cov_of(pieces + [(stem, MQ, 1)])
            g3 = fom(cov, 3) / f3_0
            if best is None or g3 > best[0]:
                best = (g3, fom(cov, 4) / f4_0, stem, lab, np.sqrt(np.diag(cov)))
        g3, g4, stem, lab, sig = best
        chosen.append(stem); pieces.append((stem, MQ, 1))
        print(f'{step:<4d} {lab:6s} {g3:6.2f} {g4:6.2f} {g3/prev3:6.2f}   '
              + '  '.join(f'{v:8.1e}' for v in sig))
        hist.append((step, lab, g3, g4, sig)); prev3 = g3

    chain = ' + '.join(['full'] + [h[1] for h in hist[1:]])
    print(f'\nGreedy chain: {chain}')
    print(f'Final ({NSTEPS} stems): FoM3 gain {hist[-1][2]:.1f}x; per-param vs full auto: '
          + ', '.join(f'{p} {bsig[i]/hist[-1][4][i]:.1f}x' for i, p in enumerate(PARAMS)))

    # ---- figure: diminishing returns ----
    steps = [h[0] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(steps, [h[2] for h in hist], 'o-', color='C0', label='FoM3 (wb,wc,ns)')
    a1.plot(steps, [h[3] for h in hist], 's--', color='C7', label='FoM4 (+ s8)')
    for h in hist:
        a1.annotate(h[1], (h[0], h[2]), textcoords='offset points', xytext=(4, 6), fontsize=8)
    a1.set_xlabel('number of ASTRA data-leg stems added'); a1.set_ylabel('FoM gain over full auto')
    a1.set_title('Greedy chain: cumulative Fisher gain'); a1.legend(); a1.grid(alpha=0.3)
    for i, p in enumerate(PARAMS):
        a2.plot(steps, [bsig[i] / h[4][i] for h in hist], 'o-', label=fj.COSMO[p][1].replace('\\', ''))
    a2.set_xlabel('number of ASTRA data-leg stems added'); a2.set_ylabel('sigma improvement vs full auto')
    a2.set_title('Per-parameter improvement'); a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle(f'Greedy forward selection (data legs): {chain}', y=1.02, fontsize=11)
    fig.tight_layout(); fig.savefig(PLOT / 'greedy_chain.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {PLOT / "greedy_chain.png"}')


if __name__ == '__main__':
    main()

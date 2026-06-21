#!/usr/bin/env python3
"""
Data-vector search: which ASTRA quantile statistic (or pair) drives the most
Fisher gain over the plain full-sample auto-correlation?

Candidates are the 16 ASTRA stems -- data-quantile autos, random-quantile autos,
full x data-Q crosses, full x rand-Q crosses (Q1..Q4) -- each mono+quad at native
15-bin resolution (the 576-sample pooled covariance keeps Hartlap benign, so no
rebinning).  We sweep all 16 singles and all C(16,2)=120 pairs, in two framings:

  standalone : the candidate alone        vs  full auto          ("better than 2PCF?")
  additive   : full auto (+) candidate    vs  full auto          ("what ASTRA adds")

For every candidate we compute the HOD-marginalised 4x4 cosmology covariance
(fisher_joint route a, global derivatives) and reduce it to:
  FoM4 = 1/sqrt(det Cov4)                  -- all four params
  FoM3 = 1/sqrt(det Cov3), dropping sigma8 -- robust to the unreliable sigma8 deriv
plus the per-parameter sigma.  Gain = FoM_candidate / FoM_baseline.

We report FoM3 as the headline (sigma8 derivatives bracket the truth 0.43/3.67, so
a 4-param FoM is sigma8-sensitive), and flag candidates whose gain leans on the
noise-limited random-quadrupole legs.

Outputs (plots/vector_search/):
  single_vector_gains.png   16 singles ranked (standalone + additive, FoM3)
  pair_gain_heatmap.png     16x16 additive-FoM3 gain matrix (diag = single)
  per_parameter_best.png    best single stem per cosmology parameter
  vector_search_results.csv full ranked table
"""

from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fisher_joint as fj

PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
PLOT.mkdir(parents=True, exist_ok=True)
MQ = (0, 2)
PARAMS = list(fj.COSMO)                                  # lnwb, lnwc, ns, lns8

# 16 ASTRA candidate stems, grouped, with short labels
FAMILY = {
    'dQ':  [f'tpcf_data_q{q}'            for q in range(1, 5)],   # data autos
    'rQ':  [f'tpcf_rand_q{q}'            for q in range(1, 5)],   # random autos
    'xdQ': [f'tpcf_cross_full_data_q{q}' for q in range(1, 5)],   # full x data crosses
    'xrQ': [f'tpcf_cross_full_rand_q{q}' for q in range(1, 5)],   # full x rand crosses
}
STEMS = [s for fam in FAMILY.values() for s in fam]
LABEL = {s: f'{fam}{q}' for fam, lst in FAMILY.items()
         for q, s in zip(range(1, 5), lst)}
RANDLEG = set(FAMILY['rQ'] + FAMILY['xrQ'])             # noise-limited rand quad legs
BASE = [('tpcf_full_data', MQ, 1)]


def piece(stem):
    return [(stem, MQ, 1)]


def metrics(pieces):
    """HOD-marginalised (route a) logdet of the 4- and 3-param cov + per-param sigma."""
    r = fj.fisher(pieces)
    cov = fj.to_phys_cov(r['cov_marg'], PARAMS)
    s4, ld4 = np.linalg.slogdet(cov)
    s3, ld3 = np.linalg.slogdet(cov[:3, :3])             # drop sigma8 -> robust FoM
    sig = np.sqrt(np.diag(cov))
    return ld4, ld3, sig, r['nb'], r['hart']


def main():
    src, _, _ = fj.deriv_source()
    print(f'Derivatives: {src};  covariance: '
          f'{"pooled 576" if fj.POOL_COV else "c000 64"} subbox samples\n')

    bld4, bld3, bsig, _, _ = metrics(BASE)
    gain = lambda ld, bld: float(np.exp(-0.5 * (ld - bld)))   # FoM ratio = sqrt(det0/det)

    # ---- singles ----
    rows = []
    for s in STEMS:
        a_ld4, a_ld3, a_sig, a_nb, a_h = metrics(piece(s))          # standalone
        f_ld4, f_ld3, f_sig, f_nb, f_h = metrics(BASE + piece(s))   # additive
        rows.append(dict(kind='single', label=LABEL[s], stems=(s,),
                         g3_alone=gain(a_ld3, bld3), g4_alone=gain(a_ld4, bld4),
                         g3_add=gain(f_ld3, bld3), g4_add=gain(f_ld4, bld4),
                         sig_add=f_sig, nb=f_nb, randleg=s in RANDLEG))

    # ---- pairs ----
    G = np.full((16, 16), np.nan)                                   # additive FoM3 gain
    for s in STEMS:                                                 # diagonal = single
        G[STEMS.index(s), STEMS.index(s)] = next(
            r['g3_add'] for r in rows if r['stems'] == (s,))
    for si, sj in combinations(STEMS, 2):
        a_ld4, a_ld3, *_ = metrics(piece(si) + piece(sj))
        f_ld4, f_ld3, f_sig, f_nb, _ = metrics(BASE + piece(si) + piece(sj))
        rows.append(dict(kind='pair', label=f'{LABEL[si]}+{LABEL[sj]}', stems=(si, sj),
                         g3_alone=gain(a_ld3, bld3), g4_alone=gain(a_ld4, bld4),
                         g3_add=gain(f_ld3, bld3), g4_add=gain(f_ld4, bld4),
                         sig_add=f_sig, nb=f_nb,
                         randleg=(si in RANDLEG or sj in RANDLEG)))
        i, j = STEMS.index(si), STEMS.index(sj)
        G[i, j] = G[j, i] = gain(f_ld3, bld3)

    # ---- console: rankings ----
    def show(title, items, key):
        print(f'\n{title}  (gain = FoM / full-auto FoM; flag * = rand-leg)')
        print(f'  {"vector":22s} {"FoM3":>6s} {"FoM4":>6s}  '
              + '  '.join(f'sig_{p}' for p in PARAMS))
        for r in items:
            flag = '*' if r['randleg'] else ' '
            print(f'  {r["label"]:22s}{flag}{r[key]:6.2f} {r[key.replace("3","4")]:6.2f}  '
                  + '  '.join(f'{v:7.1e}' for v in r['sig_add']))

    singles = [r for r in rows if r['kind'] == 'single']
    pairs   = [r for r in rows if r['kind'] == 'pair']
    print(f'Baseline full auto: sig = ' + ', '.join(
        f'{p}={v:.2e}' for p, v in zip(PARAMS, bsig)))
    show('TOP 8 SINGLES, additive (full + stem)',
         sorted(singles, key=lambda r: -r['g3_add'])[:8], 'g3_add')
    show('TOP 8 SINGLES, standalone (stem alone)',
         sorted(singles, key=lambda r: -r['g3_alone'])[:8], 'g3_alone')
    show('TOP 12 PAIRS, additive (full + stem_i + stem_j)',
         sorted(pairs, key=lambda r: -r['g3_add'])[:12], 'g3_add')

    # ---- CSV ----
    import csv
    with open(PLOT / 'vector_search_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'label', 'g3_alone', 'g4_alone', 'g3_add', 'g4_add',
                    'nb', 'randleg'] + [f'sig_{p}_add' for p in PARAMS])
        for r in sorted(rows, key=lambda r: -r['g3_add']):
            w.writerow([r['kind'], r['label'], f'{r["g3_alone"]:.4f}',
                        f'{r["g4_alone"]:.4f}', f'{r["g3_add"]:.4f}',
                        f'{r["g4_add"]:.4f}', r['nb'], int(r['randleg'])]
                       + [f'{v:.4e}' for v in r['sig_add']])
    print(f'\nSaved {PLOT / "vector_search_results.csv"}')

    # ---- figure 1: single-vector gains ----
    order = sorted(range(16), key=lambda i: -singles[i]['g3_add'])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(16)
    ax.bar(x - 0.2, [singles[i]['g3_add'] for i in order], 0.4, label='additive (full+stem)', color='C0')
    ax.bar(x + 0.2, [singles[i]['g3_alone'] for i in order], 0.4, label='standalone (stem alone)', color='C7')
    ax.axhline(1, color='k', lw=0.8, ls='--', label='full auto')
    for k, i in enumerate(order):
        if singles[i]['randleg']:
            ax.text(k, 0.02, '*', ha='center', va='bottom', color='C3', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([singles[i]['label'] for i in order], rotation=45, ha='right')
    ax.set_ylabel('FoM3 gain over full auto'); ax.legend()
    ax.set_title('Single ASTRA vectors: Fisher gain (FoM3 = ωb,ωc,ns; * = rand leg)')
    fig.tight_layout(); fig.savefig(PLOT / 'single_vector_gains.png', dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {PLOT / "single_vector_gains.png"}')

    # ---- figure 2: pair-gain heatmap ----
    fig, ax = plt.subplots(figsize=(10, 8.4))
    im = ax.imshow(G, cmap='viridis', origin='upper')
    ax.set_xticks(range(16)); ax.set_yticks(range(16))
    ax.set_xticklabels([LABEL[s] for s in STEMS], rotation=90, fontsize=8)
    ax.set_yticklabels([LABEL[s] for s in STEMS], fontsize=8)
    for b in (3.5, 7.5, 11.5):                                    # family separators
        ax.axhline(b, color='w', lw=0.6); ax.axvline(b, color='w', lw=0.6)
    fig.colorbar(im, label='FoM3 gain (full + stem_i + stem_j)')
    ax.set_title('Pair gains over full auto (diag = single; FoM3, additive)')
    fig.tight_layout(); fig.savefig(PLOT / 'pair_gain_heatmap.png', dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {PLOT / "pair_gain_heatmap.png"}')

    # ---- figure 3: best single stem per parameter ----
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (k, p) in zip(axs, enumerate(PARAMS)):
        improv = [(bsig[k] / r['sig_add'][k], r['label'], r['randleg']) for r in singles]
        improv.sort(reverse=True)
        top = improv[:6]
        ax.barh(range(len(top)), [t[0] for t in top],
                color=['C3' if t[2] else 'C2' for t in top])
        ax.set_yticks(range(len(top))); ax.set_yticklabels([t[1] for t in top], fontsize=9)
        ax.invert_yaxis(); ax.axvline(1, color='k', lw=0.8, ls='--')
        ax.set_title(fj.COSMO[p][1].replace('\\', ''), fontsize=11)
        ax.set_xlabel('sigma improvement (full+stem vs full)')
    fig.suptitle('Which single ASTRA vector helps each parameter most (additive)', y=1.03)
    fig.tight_layout(); fig.savefig(PLOT / 'per_parameter_best.png', dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {PLOT / "per_parameter_best.png"}')


if __name__ == '__main__':
    main()

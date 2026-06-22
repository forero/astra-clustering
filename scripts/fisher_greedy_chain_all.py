#!/usr/bin/env python3
"""
Greedy forward selection over ALL 16 ASTRA stems (data + random legs), using the
NOISE-AWARE Fisher (improvement #1).  The random legs were vindicated by the
noise and nonlinearity tests (Sec. caveat of the note), so we now let greedy pick
from data autos, random autos, and both kinds of full-cross crosses, scoring each
candidate by the HOD-marginalised FoM3 with the derivative-noise bias subtracted.

For a fair comparison we run two chains with the SAME noise-aware metric:
  (A) data legs only          (reproduces the conservative chain)
  (B) all 16 stems            (random legs allowed)
so the difference isolates what folding in the random legs buys.

Needs data/derivatives/derivative_global_var_*.npz (from fisher_noise_aware.py).
Output: plots/vector_search/greedy_chain_all.png  +  printed chains.
Run: python scripts/fisher_greedy_chain_all.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj

DER = fj.DER_DIR
PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
MQ = (0, 2); NB = 15; NSTEPS = 6
PARAMS = list(fj.COSMO)
DVAR = {p: np.load(DER / f'derivative_global_var_{p}.npz') for p in PARAMS}

POOL = {}
for fam, pre in [('data_q', 'dQ'), ('rand_q', 'rQ'),
                 ('cross_full_data_q', 'xdQ'), ('cross_full_rand_q', 'xrQ')]:
    for q in range(1, 5):
        POOL[f'tpcf_{fam}{q}'] = f'{pre}{q}'
DATA_LEGS = [s for s in POOL if 'rand' not in s]
ncos = len(PARAMS)


def fom3_na(pieces):
    """Noise-aware FoM3: subtract Tr(C^-1 Cov(delta)) from the cosmology diagonal."""
    a = fj.assemble(pieces); Cinv = a['Cinv']; nb = a['nb']
    D = np.vstack([a['D_cos'], a['D_hod']])
    F = D @ Cinv @ D.T
    cd = np.diag(Cinv); Vc = np.zeros((ncos, nb)); col = 0
    for stem, ells, k in pieces:
        for ell in ells:
            for i, p in enumerate(PARAMS):
                Vc[i, col:col + NB] = DVAR[p][f'{stem}_dxi{ell}']
            col += NB
    for i in range(ncos):
        F[i, i] -= float((cd * Vc[i]).sum())
    if not np.all(np.linalg.eigvalsh(F[:ncos, :ncos]) > 0):
        return np.nan, None
    Fp = F.copy(); Fp[ncos:, ncos:] += np.diag(1.0 / a['sd_pr'] ** 2)
    cov = fj.to_phys_cov(np.linalg.inv(Fp)[:ncos, :ncos], PARAMS)
    if not np.all(np.diag(cov[:3, :3]) > 0):
        return np.nan, None
    return np.linalg.slogdet(cov[:3, :3])[1], np.sqrt(np.diag(cov))


def greedy(pool):
    full = [('tpcf_full_data', MQ, 1)]
    ref_ld, ref_sig = fom3_na(full)
    chosen, pieces, hist = [], list(full), [(0, 'full', 1.0, ref_sig)]
    for step in range(1, NSTEPS + 1):
        best = None
        for stem in pool:
            if stem in chosen:
                continue
            ld, sig = fom3_na(pieces + [(stem, MQ, 1)])
            if np.isfinite(ld) and (best is None or ld < best[0]):
                best = (ld, stem, sig)
        if best is None:
            break
        ld, stem, sig = best
        chosen.append(stem); pieces.append((stem, MQ, 1))
        hist.append((step, POOL[stem], float(np.exp(-0.5 * (ld - ref_ld))), sig))
    return hist, ref_sig


def main():
    print('Noise-aware greedy chains (FoM3 gain vs full auto):\n')
    chains = {}
    for name, pool in [('data legs only', DATA_LEGS), ('all 16 stems', list(POOL))]:
        hist, refsig = greedy(pool)
        chains[name] = (hist, refsig)
        chain = 'full ' + ' '.join('+' + h[1] for h in hist[1:])
        print(f'[{name}]  {chain}')
        print(f'  {"step":4s} {"add":6s} {"FoM3":>6s}   sig(wb,wc,ns,s8)')
        for step, lab, gain, sig in hist:
            print(f'  {step:<4d} {lab:6s} {gain:6.2f}   '
                  + ', '.join(f'{v:.2e}' for v in sig))
        f = hist[-1]
        print(f'  final FoM3={f[2]:.1f}x; per-param vs full auto: '
              + ', '.join(f'{p} {refsig[i]/f[3][i]:.1f}x' for i, p in enumerate(PARAMS)) + '\n')

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, mark in [('data legs only', 's--'), ('all 16 stems', 'o-')]:
        hist, _ = chains[name]
        a1.plot([h[0] for h in hist], [h[2] for h in hist], mark, label=name)
        for h in hist[1:]:
            col = 'C3' if h[1].startswith('rQ') or h[1].startswith('xrQ') else 'C0'
            a1.annotate(h[1], (h[0], h[2]), textcoords='offset points', xytext=(4, 5),
                        fontsize=8, color=col)
    a1.set_xlabel('number of ASTRA stems added'); a1.set_ylabel('noise-aware FoM3 gain')
    a1.set_title('Greedy chain: data legs vs all stems'); a1.legend(); a1.grid(alpha=0.3)
    hist, refsig = chains['all 16 stems']
    for i, p in enumerate(PARAMS):
        a2.plot([h[0] for h in hist], [refsig[i] / h[3][i] for h in hist], 'o-',
                label=fj.COSMO[p][1].replace('\\', ''))
    a2.set_xlabel('number of ASTRA stems added (all-stems chain)')
    a2.set_ylabel('sigma improvement vs full auto'); a2.set_title('Per-parameter (all stems)')
    a2.legend(); a2.grid(alpha=0.3)
    chain = 'full ' + ' '.join('+' + h[1] for h in chains['all 16 stems'][0][1:])
    fig.suptitle(f'Noise-aware greedy chain with random legs: {chain}', y=1.02, fontsize=11)
    fig.tight_layout(); fig.savefig(PLOT / 'greedy_chain_all.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {PLOT / "greedy_chain_all.png"}')


if __name__ == '__main__':
    main()

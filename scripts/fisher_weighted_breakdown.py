#!/usr/bin/env python3
"""
Follow-up #2 to the weighted-2PCF Fisher: a FAIRER baseline + per-leg breakdown.

The headline table (fisher_weighted_compare.py) pitted the 90-bin weighted vector
[data-auto, arand-auto, cross] x [ell0, ell2] against the 30-bin quantile full-auto
and the weighted side won on every parameter.  Two questions that table can't
answer, both addressed here entirely from CACHED data (no new sims):

  (1) PER-LEG BREAKDOWN.  Is the weighted gain the re-weighting itself, or just
      that the weighted vector bundles three legs?  We compute the HOD-fixed
      marginalised sigma for every sub-vector of the weighted vector
      (data only / arand only / cross only / pairs / full) under the SAME weighted
      covariance.  The cleanest controlled experiment lives here: the weighted
      'uniform' data-auto IS the standard full-data xi (by the mean-normalised
      estimator's unit test), so 'scheme data-auto vs uniform data-auto' isolates
      the pure effect of re-weighting one fixed population, with identical bins and
      covariance.

  (2) FAIRER QUANTILE BASELINE.  Instead of only the single quantile full-auto,
      compare the weighted vector against the project's best quantile MULTI-leg
      vectors (data-Q autos; full + data-Q + rand-Q autos), each with its proper
      pooled 576-subbox covariance (9 cosmologies x 64 subboxes, mean-subtracted
      per cosmology) so the high-dimensional vectors stay Hartlap-positive.

Caveats (read ratios, not absolute sigma): all derivatives are HOD-fixed over the
matched single-HOD +/- pairs (same pre-Tier-1 HOD contamination as the early
quantile Fisher; sigma8 worst).  The weighted side uses the 192-sample c000
reanalysis covariance while the quantile multi-leg side uses the 576-sample pooled
covariance -- the weighted side is therefore at a Hartlap DISADVANTAGE, so if it
still wins the conclusion is conservative.  Pooling the weighted covariance across
the 8 +/- runs is the deferred honest upgrade.

Outputs
  plots/fullbox_weighted/weighted_perleg_breakdown.png   (sigma per sub-vector)
  plots/fullbox_weighted/weighted_legs_measured.png      (the measured legs)
  data/fullbox_weighted/cov/fisher_weighted_breakdown.npz

Run (any node; needs the 8 +/- weighted runs + weighted covariance built):
  python scripts/fisher_weighted_breakdown.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
WDIR = REPO / 'data' / 'fullbox_weighted'
COVF = WDIR / 'cov' / 'weighted_subbox_cov.npz'
DERIV = REPO / 'data' / 'derivatives'
PLOTS = REPO / 'plots' / 'fullbox_weighted'

PAIRS = {
    'lnwb': ('c100_hod179', 'c101_hod152', 0.020, r'$\ln\omega_b$'),
    'lnwc': ('c102_hod556', 'c103_hod861', 0.033, r'$\ln\omega_c$'),
    'ns':   ('c104_hod498', 'c105_hod589', 0.010, r'$n_s$'),
    'lns8': ('c112_hod507', 'c113_hod483', 0.020, r'$\ln\sigma_8$'),
}
PARAMS = list(PAIRS)
PLABEL = [PAIRS[p][3] for p in PARAMS]
STATS = ('data', 'arand', 'cross')          # block order in the 90-bin vector
ELLS = (0, 2)
NBIN = 15                                    # native s-bins per (stat, ell)

# 9 matched-HOD Fisher cosmologies (for the pooled quantile covariance)
TAGS9 = ['c000_hod484', 'c100_hod179', 'c101_hod152', 'c102_hod556',
         'c103_hod861', 'c104_hod498', 'c105_hod589', 'c112_hod507', 'c113_hod483']


def hartlap(n_samp, n_bin):
    return (n_samp - n_bin - 2) / (n_samp - 1)


def fisher_sigma(D, cov, n_samp):
    """D:(nparam,nbin) cov:(nbin,nbin) -> marginalised sigma, hartlap."""
    n_bin = cov.shape[0]
    h = hartlap(n_samp, n_bin)
    if h <= 0:
        return np.full(D.shape[0], np.nan), h
    try:
        Cinv = np.linalg.inv(cov) * h
        F = D @ Cinv @ D.T
        sig = np.sqrt(np.diag(np.linalg.inv(F)))
    except np.linalg.LinAlgError:
        return np.full(D.shape[0], np.nan), h
    return sig, h


# ── weighted side ────────────────────────────────────────────────────────────
def w_multipole(tag, scheme, stat, ell):
    return np.load(WDIR / tag / f'fbw_multipoles_{scheme}_{stat}.npz')[f'xi{ell}']


def block_cols(stat, ell):
    """column slice of a (stat,ell) block inside the 90-bin weighted vector."""
    i = STATS.index(stat) * len(ELLS) + ELLS.index(ell)
    return slice(i * NBIN, (i + 1) * NBIN)


def weighted_derivative(scheme):
    """(nparam, 90) central-difference derivative for one scheme; None if incomplete."""
    D = []
    for p in PARAMS:
        tp, tm, dth, _ = PAIRS[p]
        if not ((WDIR / tp / 'fbw_info.npz').is_file() and (WDIR / tm / 'fbw_info.npz').is_file()):
            return None
        row = []
        for stat in STATS:
            for ell in ELLS:
                row.append((w_multipole(tp, scheme, stat, ell) -
                            w_multipole(tm, scheme, stat, ell)) / (2 * dth))
        D.append(np.hstack(row))
    return np.array(D)


SUBSETS = {                                  # which (stat,ell) blocks each sub-vector uses
    'data':        [('data', 0), ('data', 2)],
    'arand':       [('arand', 0), ('arand', 2)],
    'cross':       [('cross', 0), ('cross', 2)],
    'data+arand':  [('data', 0), ('data', 2), ('arand', 0), ('arand', 2)],
    'data+cross':  [('data', 0), ('data', 2), ('cross', 0), ('cross', 2)],
    'full':        [(s, e) for s in STATS for e in ELLS],
}


def subset_index(blocks):
    idx = []
    for stat, ell in blocks:
        sl = block_cols(stat, ell)
        idx.extend(range(sl.start, sl.stop))
    return np.array(idx)


# ── pooled quantile covariance (9 cosmologies x 64 subboxes) ─────────────────
def pooled_quantile_cov(stems):
    """Pooled, mean-subtracted-per-cosmology covariance for a list of stems
    (each contributes xi0_all and xi2_all).  Returns cov (nbin,nbin) at full-box
    volume and the per-cosmology sample count 64."""
    per_cosmo = []
    for tag in TAGS9:
        legs = []
        for stem in stems:
            z = np.load(REPO / 'data' / tag / f'subbox_multipoles_{stem}.npz')
            legs.append(z['xi0_all']); legs.append(z['xi2_all'])
        per_cosmo.append(np.hstack(legs))            # (64, nbin)
    per_cosmo = [V - V.mean(0, keepdims=True) for V in per_cosmo]
    X = np.vstack(per_cosmo)                          # (576, nbin)
    n_samp = X.shape[0]
    cov = (X.T @ X) / (n_samp - 1) / 64.0             # /64 -> full-box volume
    return cov, n_samp


def quantile_derivative(stems):
    D = []
    for p in PARAMS:
        d = np.load(DERIV / f'derivative_fullbox_{p}.npz')
        row = []
        for stem in stems:
            row.append(d[f'{stem}_dxi0']); row.append(d[f'{stem}_dxi2'])
        D.append(np.hstack(row))
    return np.array(D)


def quantile_sigma(stems, label):
    cov, n = pooled_quantile_cov(stems)
    D = quantile_derivative(stems)
    sig, h = fisher_sigma(D, cov, n)
    return label, sig, h, cov.shape[0], n


def main():
    if not COVF.is_file():
        sys.exit(f'Missing {COVF}; run scripts/weighted_subbox_cov.py first.')
    covz = np.load(COVF, allow_pickle=True)
    s = covz['s']
    schemes = [str(x) for x in covz['schemes']]
    n_w = int(covz['n_sub']) * int(covz['n_iter'])           # 192

    # ===== Part 1: per-leg breakdown (weighted cov, native 15 bins) ==========
    print(f'\n=== PART 1 — per-leg breakdown (weighted cov, {n_w} samples, 15 bins/leg) ===')
    print('HOD-fixed marginalised sigma; rows = weighted sub-vector, per scheme.\n')

    breakdown = {}            # breakdown[scheme][subset] = (sigma(4,), hartlap)
    uniform_data_sigma = None
    for scheme in schemes:
        Dfull = weighted_derivative(scheme)
        if Dfull is None:
            print(f'  {scheme}: +/- runs incomplete, skipped'); continue
        cov_full = covz[f'{scheme}_cov']                      # (90,90)
        breakdown[scheme] = {}
        for name, blocks in SUBSETS.items():
            idx = subset_index(blocks)
            sig, h = fisher_sigma(Dfull[:, idx], cov_full[np.ix_(idx, idx)], n_w)
            breakdown[scheme][name] = (sig, h)
        if scheme == 'uniform':
            uniform_data_sigma = breakdown['uniform']['data'][0]

    # controlled experiment: scheme data-auto vs uniform data-auto (== standard xi)
    print('  CONTROLLED — data-auto only (one fixed population, pure re-weighting effect):')
    print(f'    {"scheme":10s} ' + ' '.join(f'{l:>11s}' for l in PLABEL) + '   gain vs uniform-data')
    for scheme in schemes:
        if scheme not in breakdown:
            continue
        sig = breakdown[scheme]['data'][0]
        gain = uniform_data_sigma / sig
        tag = '  (== standard xi)' if scheme == 'uniform' else ''
        print(f'    {scheme:10s} ' + ' '.join(f'{v:11.3e}' for v in sig) +
              '   ' + ' '.join(f'{g:4.2f}x' for g in gain) + tag)

    # full breakdown table for the two representative schemes
    for scheme in ('rank', 'signed'):
        if scheme not in breakdown:
            continue
        print(f'\n  {scheme.upper()} — sub-vector sigma (gain vs this scheme\'s data-auto):')
        base = breakdown[scheme]['data'][0]
        print(f'    {"sub-vector":12s} {"nb":>3s} {"hart":>5s} ' +
              ' '.join(f'{l:>11s}' for l in PLABEL))
        for name in SUBSETS:
            sig, h = breakdown[scheme][name]
            nb = len(subset_index(SUBSETS[name]))
            print(f'    {name:12s} {nb:3d} {h:5.2f} ' +
                  ' '.join(f'{v:11.3e}' for v in sig))

    # ===== Part 2: fairer quantile multi-leg baseline (pooled 576 cov) ========
    print(f'\n=== PART 2 — fairer quantile MULTI-leg baseline (pooled 576-subbox cov) ===')
    data_q = [f'tpcf_data_q{i}' for i in (1, 2, 3, 4)]
    rand_q = [f'tpcf_rand_q{i}' for i in (1, 2, 3, 4)]
    q_vectors = [
        (['tpcf_full_data'], 'quantile full-auto'),
        (data_q, 'quantile data-Q autos'),
        (['tpcf_full_data'] + data_q + rand_q, 'quantile full+data+rand Q autos'),
    ]
    qres = [quantile_sigma(stems, lab) for stems, lab in q_vectors]

    print(f'    {"vector":34s} {"nb":>4s} {"hart":>5s} ' +
          ' '.join(f'{l:>11s}' for l in PLABEL))
    for lab, sig, h, nb, n in qres:
        print(f'    {lab:34s} {nb:4d} {h:5.2f} ' +
              ' '.join(f'{v:11.3e}' for v in sig))
    # weighted full vectors (192-sample cov) for side-by-side
    for scheme in ('rank', 'signed'):
        if scheme in breakdown:
            sig, h = breakdown[scheme]['full']
            print(f'    {("weighted "+scheme+" full"):34s} {90:4d} {h:5.2f} ' +
                  ' '.join(f'{v:11.3e}' for v in sig) + '   (192-sample cov)')

    # best quantile per param (over the multi-leg vectors) for the gain summary
    q_best = np.min(np.vstack([sig for _, sig, _, _, _ in qres]), axis=0)
    print('\n  weighted-full gain vs BEST quantile multi-leg (per param):')
    for scheme in ('rank', 'signed'):
        if scheme in breakdown:
            sig = breakdown[scheme]['full'][0]
            print(f'    {scheme:8s} ' +
                  ' '.join(f'{g:4.2f}x' for g in q_best / sig))

    # ===== figure 1: per-leg breakdown bars =================================
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_schemes = [s for s in ('uniform', 'rank', 'signed') if s in breakdown]
    order = list(SUBSETS)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True)
    x = np.arange(len(order))
    w = 0.8 / len(plot_schemes)
    colors = {'uniform': 'tab:gray', 'rank': 'tab:blue', 'signed': 'tab:red'}
    for j, p in enumerate(PARAMS):
        ax = axes[j]
        for k, scheme in enumerate(plot_schemes):
            vals = [breakdown[scheme][n][0][j] for n in order]
            ax.bar(x + (k - (len(plot_schemes) - 1) / 2) * w, vals, w,
                   label=scheme, color=colors.get(scheme))
        # reference: standard full-data xi (== weighted uniform data-auto)
        ax.axhline(uniform_data_sigma[j], color='k', ls='--', lw=1,
                   label='standard $\\xi$ (uniform data)')
        ax.set_yscale('log'); ax.set_title(PLABEL[j])
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel(r'$\sigma$ (HOD-fixed)') if j == 0 else None
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Weighted-2PCF per-leg breakdown — which legs carry the constraint '
                 '(lower $\\sigma$ = better)')
    fig.tight_layout()
    f1 = PLOTS / 'weighted_perleg_breakdown.png'
    fig.savefig(f1, dpi=130); plt.close(fig)
    print(f'\nSaved {f1}')

    # ===== figure 2: the measured legs =====================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    leg_schemes = [s for s in ('uniform', 'void', 'knot', 'rank', 'signed') if s in schemes]
    s2 = s ** 2
    for r, ell in enumerate(ELLS):
        for c, stat in enumerate(STATS):
            ax = axes[r, c]
            for scheme in leg_schemes:
                xi = w_multipole('c000_hod484', scheme, stat, ell)
                ax.plot(s, s2 * xi, marker='.', ms=4, label=scheme)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_title(f'{stat}-auto  $\\ell={ell}$' if stat != 'cross'
                         else f'data$\\times$arand  $\\ell={ell}$', fontsize=10)
            if c == 0:
                ax.set_ylabel(r'$s^2\,\xi_{w}(s)$')
            if r == 1:
                ax.set_xlabel(r'$s\ [h^{-1}\mathrm{Mpc}]$')
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Weighted-2PCF measured legs (c000/hod484) — the three populations $\\times$ multipole')
    fig.tight_layout()
    f2 = PLOTS / 'weighted_legs_measured.png'
    fig.savefig(f2, dpi=130); plt.close(fig)
    print(f'Saved {f2}')

    # ===== save =============================================================
    out = WDIR / 'cov' / 'fisher_weighted_breakdown.npz'
    save = {'params': PARAMS, 'subsets': list(SUBSETS), 'schemes': schemes,
            'uniform_data_sigma': uniform_data_sigma,
            'quantile_labels': [lab for _, lab, _, _, _ in zip(qres, [v[1] for v in q_vectors], qres, qres, qres)]}
    for scheme in breakdown:
        for name in SUBSETS:
            save[f'{scheme}__{name}'] = breakdown[scheme][name][0]
    for lab, sig, h, nb, n in qres:
        save['Q__' + lab.replace(' ', '_')] = sig
    np.savez(out, **save)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

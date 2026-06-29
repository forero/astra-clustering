#!/usr/bin/env python3
"""
Head-to-head Fisher comparison: WEIGHTED-2PCF schemes vs the quantile full-auto.

For each weight scheme, build the central-difference cosmology derivatives of the
weighted data vector [data-auto, arand-auto, cross] x [ell0, ell2] over the matched
+/- Fisher pairs, and form an HOD-FIXED Fisher with the reanalysis weighted subbox
covariance (scripts/weighted_subbox_cov.py).  Report the marginalised 1-sigma on
{w_b, w_c, n_s, ln sigma8} at full-box (2000 Mpc/h) volume.

The reference is the standard quantile full-sample auto-correlation (mono+quad),
built the same way: full-box +/- derivatives (data/derivatives/derivative_fullbox_*)
and the 64 c000 subboxes scaled to full-box volume (C_subbox/64).  Both sides are
HOD-FIXED and single-box, so the comparison is apples-to-apples.

IMPORTANT — this is a RELATIVE comparison.  The +/- runs use matched-but-mismatched
single HODs, so every derivative here carries the same HOD contamination the
quantile Fisher had before the Tier-1 / global-response correction (sigma8 worst).
The absolute sigmas are optimistic; read the weighted-vs-quantile RATIOS, and which
scheme/parameter the weighting helps.

Outputs
  data/fullbox_weighted/cov/fisher_weighted_compare.npz   (sigma table)
  plots/fullbox_weighted/fisher_weighted_compare.png      (sigma per param, schemes vs quantile)

Run (any node; needs the 8 +/- weighted runs done and the covariance built):
  python scripts/fisher_weighted_compare.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
WDIR = REPO_ROOT / 'data' / 'fullbox_weighted'
COVF = WDIR / 'cov' / 'weighted_subbox_cov.npz'
DERIV_DIR = REPO_ROOT / 'data' / 'derivatives'
PLOT_DIR  = REPO_ROOT / 'plots' / 'fullbox_weighted'

# same +/- pairs / half-steps as compute_derivatives_fullbox.py
PAIRS = {
    'lnwb': ('c100_hod179', 'c101_hod152', 0.020, r'$\ln\omega_b$'),
    'lnwc': ('c102_hod556', 'c103_hod861', 0.033, r'$\ln\omega_c$'),
    'ns':   ('c104_hod498', 'c105_hod589', 0.010, r'$n_s$'),
    'lns8': ('c112_hod507', 'c113_hod483', 0.020, r'$\ln\sigma_8$'),
}
PARAMS = list(PAIRS)
STATS  = ('data', 'arand', 'cross')


def hartlap(n_samp, n_bin):
    return (n_samp - n_bin - 2) / (n_samp - 1)


def fisher_sigma(D, cov, n_samp):
    """D: (n_param, n_bin) derivatives; cov: (n_bin, n_bin). Marginalised sigma."""
    n_bin = cov.shape[0]
    h = hartlap(n_samp, n_bin)
    if h <= 0:
        return np.full(D.shape[0], np.nan), h
    Cinv = np.linalg.inv(cov) * h
    F = D @ Cinv @ D.T
    Finv = np.linalg.inv(F)
    return np.sqrt(np.diag(Finv)), h


# ── weighted side ────────────────────────────────────────────────────────────
def weighted_multipole(tag, scheme, stat, ell):
    f = WDIR / tag / f'fbw_multipoles_{scheme}_{stat}.npz'
    return np.load(f)[f'xi{ell}']


def run_complete(tag):
    return (WDIR / tag / 'fbw_info.npz').is_file()


def weighted_fisher(cov_npz, schemes):
    n_samp = int(cov_npz['n_sub']) * int(cov_npz['n_iter'])
    results = {}
    for sch in schemes:
        cov = cov_npz[f'{sch}_cov']            # (6*nbin, 6*nbin): data,arand,cross x ell0,ell2
        # build derivative matrix (n_param, 6*nbin) in the SAME block order as the cov
        D = []
        ok = True
        for param in PARAMS:
            tag_p, tag_m, dth, _ = PAIRS[param]
            if not (run_complete(tag_p) and run_complete(tag_m)):
                ok = False
                break
            blocks = []
            for stat in STATS:
                for ell in (0, 2):
                    dp = weighted_multipole(tag_p, sch, stat, ell)
                    dm = weighted_multipole(tag_m, sch, stat, ell)
                    blocks.append((dp - dm) / (2 * dth))
            D.append(np.hstack(blocks))
        if not ok:
            results[sch] = (None, None)
            continue
        D = np.array(D)
        sig, h = fisher_sigma(D, cov, n_samp)
        results[sch] = (sig, h)
    return results, n_samp


# ── quantile reference (full-auto mono+quad) ─────────────────────────────────
def quantile_reference():
    stem = 'tpcf_full_data'
    sub = np.load(REPO_ROOT / 'data' / 'c000_hod484' / f'subbox_multipoles_{stem}.npz')
    V = np.hstack([sub['xi0_all'], sub['xi2_all']])      # (64, 2*nbin)
    n_samp = V.shape[0]
    cov = np.cov(V, rowvar=False) / 64.0                  # full-box volume
    D = []
    for param in PARAMS:
        d = np.load(DERIV_DIR / f'derivative_fullbox_{param}.npz')
        D.append(np.hstack([d[f'{stem}_dxi0'], d[f'{stem}_dxi2']]))
    D = np.array(D)
    sig, h = fisher_sigma(D, cov, n_samp)
    return sig, h, n_samp, cov.shape[0]


def main():
    if not COVF.is_file():
        sys.exit(f'Missing weighted covariance {COVF}; run scripts/weighted_subbox_cov.py first.')
    cov_npz = np.load(COVF, allow_pickle=True)
    schemes = list(cov_npz['schemes'])
    nbin_w = 6 * len(cov_npz['s'])

    wres, n_samp_w = weighted_fisher(cov_npz, schemes)
    qsig, qh, n_samp_q, nbin_q = quantile_reference()

    # ── print table ──
    print(f'\nHOD-fixed marginalised sigma at full-box (2000 Mpc/h) volume')
    print(f'  weighted vector per scheme: {nbin_w} bins, {n_samp_w} cov samples (Hartlap below)')
    print(f'  quantile full-auto ref:     {nbin_q} bins, {n_samp_q} cov samples (Hartlap {qh:.2f})\n')
    hdr = f'{"vector":24s} ' + ' '.join(f'{p:>12s}' for p in PARAMS) + '   hart'
    print(hdr); print('-' * len(hdr))
    print(f'{"quantile full-auto":24s} ' +
          ' '.join(f'{qsig[i]:12.3e}' for i in range(len(PARAMS))) + f'   {qh:.2f}')
    print('-' * len(hdr))
    table = {'quantile_full_auto': qsig}
    for sch in schemes:
        sig, h = wres[sch]
        if sig is None:
            print(f'{sch:24s} {"(+/- runs incomplete)":>40s}')
            continue
        ratio = qsig / sig                       # >1 means weighted scheme is TIGHTER
        print(f'{sch:24s} ' + ' '.join(f'{sig[i]:12.3e}' for i in range(len(PARAMS))) +
              f'   {h:.2f}')
        print(f'{"  (gain vs quantile)":24s} ' +
              ' '.join(f'{ratio[i]:11.2f}x' for i in range(len(PARAMS))))
        table[sch] = sig

    # ── figure: sigma per parameter, schemes vs quantile baseline ──
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 4))
    valid = [s for s in schemes if wres[s][0] is not None]
    x = np.arange(len(valid))
    for j, param in enumerate(PARAMS):
        ax = axes[j]
        vals = [wres[s][0][j] for s in valid]
        ax.bar(x, vals, color='tab:blue', alpha=0.8)
        ax.axhline(qsig[j], color='k', ls='--', label='quantile full-auto')
        ax.set_xticks(x); ax.set_xticklabels(valid, rotation=45, ha='right')
        ax.set_title(PAIRS[param][3]); ax.set_ylabel(r'$\sigma$ (HOD-fixed)')
        ax.set_yscale('log')
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle('Weighted-2PCF schemes vs quantile full-auto (HOD-fixed, full-box)')
    fig.tight_layout()
    out = PLOT_DIR / 'fisher_weighted_compare.png'
    fig.savefig(out, dpi=130)
    print(f'\nSaved {out}')

    np.savez(WDIR / 'cov' / 'fisher_weighted_compare.npz',
             params=PARAMS, schemes=schemes,
             quantile_sigma=qsig, n_samp_weighted=n_samp_w,
             **{f'{s}_sigma': (wres[s][0] if wres[s][0] is not None else np.full(len(PARAMS), np.nan))
                for s in schemes})
    print(f'Saved {WDIR / "cov" / "fisher_weighted_compare.npz"}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Anatomy of the two cross-correlation legs in the optimal vector full + xrQ4 + xrQ1
(full sample x void/knot random quantiles).  For each cross leg shows the three
Fisher ingredients explicitly:

  (1) the measured monopole and quadrupole  xi_0, xi_2  (c000 fiducial mean),
  (2) the cosmology derivatives  d xi_{0,2} / d theta  (global response model),
  (3) the covariance / correlation matrix of the full + xrQ4 + xrQ1 vector
      (including the cross-covariance of the two legs with the full auto).

Outputs:
  plots/vector_search/crosslegs_data_derivatives.png
  plots/vector_search/crosslegs_covariance.png
  data/derivatives/crosslegs_xrQ4_xrQ1.npz  (s, xi0/xi2, derivs, covariance)
Run: python scripts/fisher_crosslegs_details.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj

REPO = Path(__file__).resolve().parents[1]
FB = REPO / 'data' / 'fullbox'; DER = fj.DER_DIR
PLOT = REPO / 'plots' / 'vector_search'
MQ = (0, 2); NB = 15
PARAMS = list(fj.COSMO)
PLABEL = {p: f'${fj.COSMO[p][1]}$' for p in PARAMS}
CROSS = {'tpcf_cross_full_rand_q4': 'xrQ4 (full x rand-knot)',
         'tpcf_cross_full_rand_q1': 'xrQ1 (full x rand-void)'}
VEC = ['tpcf_full_data'] + list(CROSS)            # full + xrQ4 + xrQ1
DERV = {p: np.load(DER / f'derivative_global_{p}.npz') for p in PARAMS}


def c000_mean(stem, ell):
    ds = [d for d in sorted(FB.glob('c000_hod*')) if (d / 'fullbox_info.npz').is_file()]
    arr = [np.load(d / f'fullbox_multipoles_{stem}.npz') for d in ds]
    s = arr[0]['s']
    return s, np.mean([a[f'xi{ell}'] for a in arr], 0)


def main():
    s = c000_mean('tpcf_full_data', 0)[0]

    # ---- figure 1: measured mono+quad and cosmology derivatives, per cross leg ----
    fig, axs = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for col, (stem, lab) in enumerate(CROSS.items()):
        x0 = c000_mean(stem, 0)[1]; x2 = c000_mean(stem, 2)[1]
        ax = axs[0, col]
        ax.plot(s, s**2 * x0, 'C0-o', ms=3, label=r'$\xi_0$ (monopole)')
        ax.plot(s, s**2 * x2, 'C3-s', ms=3, label=r'$\xi_2$ (quadrupole)')
        ax.axhline(0, color='grey', lw=0.5); ax.set_title(lab, fontsize=11)
        ax.set_ylabel(r'(1) measured  $s^2\xi$')
        if col == 0:
            ax.legend(fontsize=9)
        for row, ell in [(1, 0), (2, 2)]:
            ax = axs[row, col]
            for i, p in enumerate(PARAMS):
                ax.plot(s, s**2 * DERV[p][f'{stem}_dxi{ell}'], color=f'C{i}',
                        lw=1.6, label=PLABEL[p])
            ax.axhline(0, color='grey', lw=0.5)
            ax.set_ylabel(rf'(2) $s^2\,\partial\xi_{{{ell}}}/\partial\theta$')
            if row == 1 and col == 0:
                ax.legend(fontsize=8, ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel(r'$s\,[h^{-1}\,\mathrm{Mpc}]$')
    fig.suptitle('Cross-correlation legs xrQ4, xrQ1: measured monopole/quadrupole '
                 'and cosmology derivatives', y=0.995)
    fig.tight_layout()
    fig.savefig(PLOT / 'crosslegs_data_derivatives.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {PLOT / "crosslegs_data_derivatives.png"}')

    # ---- figure 2: covariance / correlation matrix of full + xrQ4 + xrQ1 ----
    a = fj.assemble([(st, MQ, 1) for st in VEC])
    C = a['C_cv_full']; nb = a['nb']
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    bnd = np.arange(0, nb + 1, 2 * NB)
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, origin='upper')
    for b in bnd:
        ax.axhline(b - 0.5, color='k', lw=0.7); ax.axvline(b - 0.5, color='k', lw=0.7)
    ticks = bnd[:-1] + NB
    short = ['full', 'xrQ4', 'xrQ1']
    ax.set_xticks(ticks); ax.set_xticklabels(short); ax.set_yticks(ticks); ax.set_yticklabels(short)
    fig.colorbar(im, label='correlation')
    ax.set_title(f'(3) correlation matrix of full + xrQ4 + xrQ1 '
                 f'({nb}$\\times${nb}; pooled 576-subbox $C$)')
    fig.tight_layout()
    fig.savefig(PLOT / 'crosslegs_covariance.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {PLOT / "crosslegs_covariance.png"}')

    # ---- save the actual arrays ----
    out = {'s': s, 'cov_full_xrQ4_xrQ1': C,
           'corr_full_xrQ4_xrQ1': corr, 'vec_order': np.array(VEC)}
    for stem, lab in CROSS.items():
        sh = lab.split()[0]
        for ell in (0, 2):
            out[f'{sh}_xi{ell}'] = c000_mean(stem, ell)[1]
            for p in PARAMS:
                out[f'{sh}_dxi{ell}_{p}'] = DERV[p][f'{stem}_dxi{ell}']
    np.savez(DER / 'crosslegs_xrQ4_xrQ1.npz', **out)
    print(f'Saved {DER / "crosslegs_xrQ4_xrQ1.npz"}')

    # numeric summary
    print('\nCross-leg measured amplitude and omega_c derivative (s~15 Mpc/h):')
    for stem, lab in CROSS.items():
        x0 = c000_mean(stem, 0)[1][1]; d = DERV['lnwc'][f'{stem}_dxi0'][1]
        print(f'  {lab:26s} xi0={x0:+.3f}  dxi0/dlnwc={d:+.3f}')


if __name__ == '__main__':
    main()

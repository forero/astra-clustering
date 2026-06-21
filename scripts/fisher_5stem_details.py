#!/usr/bin/env python3
"""
Step-by-step diagnostics of the Fisher computation for the 5-stem ASTRA vector
(full + xdQ1 + dQ1 + dQ2 + xdQ3 + dQ4), HOD-marginalised.  Produces, in
plots/vector_search/fisher_5stem/:

  step1_data_vector.png    fiducial s^2 xi per piece (mono + quad)
  step2_derivatives.png    cosmology derivatives dxi/dtheta across the 180-bin vector
  step3_correlation.png    correlation matrix of the data vector (pooled 576 subbox C)
  step4_fisher.png         conditional & marginalised cosmology parameter correlations

Reuses fisher_joint.assemble / fisher for D, C, Cinv and the parameter covariances.
Run: python scripts/fisher_5stem_details.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj

REPO = Path(__file__).resolve().parents[1]
FB   = REPO / 'data' / 'fullbox'
OUT  = REPO / 'plots' / 'vector_search' / 'fisher_5stem'; OUT.mkdir(parents=True, exist_ok=True)
MQ = (0, 2)
PARAMS = list(fj.COSMO)
PLABEL = {p: f'${fj.COSMO[p][1]}$' for p in PARAMS}

STEMS = ['tpcf_full_data', 'tpcf_cross_full_data_q1', 'tpcf_data_q1',
         'tpcf_data_q2', 'tpcf_cross_full_data_q3', 'tpcf_data_q4']
SHORT = ['full', 'xdQ1', 'dQ1', 'dQ2', 'xdQ3', 'dQ4']
PIECES = [(s, MQ, 1) for s in STEMS]
NB_PER = 15                                              # bins per multipole


def c000_mean(stem, ell):
    ds = [d for d in sorted(FB.glob('c000_hod*')) if (d / 'fullbox_info.npz').is_file()]
    arr = [np.load(d / f'fullbox_multipoles_{stem}.npz') for d in ds]
    return np.mean([a[f'xi{ell}'] for a in arr], 0), arr[0]['s']


def main():
    a = fj.assemble(PIECES)
    r = fj.fisher(PIECES)
    D_cos, C = a['D_cos'], a['C_cv_full']               # (4, nb), (nb, nb)
    nb, hart = a['nb'], a['hart']
    s = c000_mean('tpcf_full_data', 0)[1]
    print(f'Derivatives: {fj.deriv_source()[0]};  nb={nb}, Hartlap={hart:.3f}, '
          f'pooled cov samples={a["nsamp"]}')

    # ---------- step 1: the data vector ----------
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, stem, lab in zip(axs.flat, STEMS, SHORT):
        x0, _ = c000_mean(stem, 0); x2, _ = c000_mean(stem, 2)
        ax.plot(s, s**2 * x0, 'C0-o', ms=3, label=r'$\ell=0$')
        ax.plot(s, s**2 * x2, 'C3-s', ms=3, label=r'$\ell=2$')
        ax.axhline(0, color='grey', lw=0.5); ax.set_title(lab, fontsize=11)
        ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    axs[0, 0].set_ylabel(r'$s^2\xi$'); axs[1, 0].set_ylabel(r'$s^2\xi$')
    axs[0, 0].legend(fontsize=9)
    fig.suptitle('Step 1 -- the 5-stem data vector (c000 fiducial, '
                 f'{len(STEMS)} pieces $\\times\\,\\{{\\ell{{=}}0,2\\}}\\times 15$ bins '
                 f'$= {nb}$)', y=1.0)
    fig.tight_layout(); fig.savefig(OUT / 'step1_data_vector.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {OUT/"step1_data_vector.png"}')

    # ---------- step 2: cosmology derivatives across the vector ----------
    bnd = np.arange(0, nb + 1, 2 * NB_PER)              # piece boundaries
    fig, axs = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax, p, row in zip(axs, PARAMS, D_cos):
        ax.plot(np.arange(nb), row, 'C2', lw=1.3)
        ax.axhline(0, color='grey', lw=0.5)
        for b in bnd:
            ax.axvline(b, color='k', lw=0.7)
        for b in bnd[:-1]:                              # ell0|ell2 split inside each piece
            ax.axvline(b + NB_PER, color='grey', lw=0.4, ls=':')
        ax.set_ylabel(rf'$\partial\xi/\partial${PLABEL[p]}', fontsize=10)
    for b, lab in zip(bnd[:-1], SHORT):
        axs[0].text(b + NB_PER, axs[0].get_ylim()[1], lab, ha='center', va='bottom', fontsize=9)
    axs[-1].set_xlabel('data-vector bin index  (piece blocks; dotted = $\\ell0|\\ell2$ split)')
    fig.suptitle('Step 2 -- cosmology derivatives (rows of $D$), global response model', y=0.995)
    fig.tight_layout(); fig.savefig(OUT / 'step2_derivatives.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {OUT/"step2_derivatives.png"}')

    # ---------- step 3: correlation matrix ----------
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    fig, ax = plt.subplots(figsize=(9.2, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, origin='upper')
    for b in bnd:
        ax.axhline(b - 0.5, color='k', lw=0.7); ax.axvline(b - 0.5, color='k', lw=0.7)
    ticks = bnd[:-1] + NB_PER
    ax.set_xticks(ticks); ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels(SHORT, fontsize=9)
    fig.colorbar(im, label='correlation'); ax.set_title(
        'Step 3 -- data-vector correlation matrix (pooled 576-subbox $C$, '
        f'{nb}$\\times${nb})')
    fig.tight_layout(); fig.savefig(OUT / 'step3_correlation.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {OUT/"step3_correlation.png"}')

    # ---------- step 4: Fisher -> parameter correlations ----------
    def pcorr(cov):
        d = np.sqrt(np.diag(cov)); return cov / np.outer(d, d), d
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (cov, name) in zip(axes, [(r['cov_cond'], 'conditional (HOD fixed)'),
                                      (r['cov_marg'], 'marginalised (HOD nuisances)')]):
        physcov = fj.to_phys_cov(cov, PARAMS)
        pc, _ = pcorr(physcov); sig = np.sqrt(np.diag(physcov))
        im = ax.imshow(pc, cmap='RdBu_r', vmin=-1, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{pc[i,j]:+.2f}', ha='center', va='center', fontsize=9)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([PLABEL[p] for p in PARAMS]); ax.set_yticklabels([PLABEL[p] for p in PARAMS])
        ax.set_title(name + '\n$\\sigma$ = ' + ', '.join(f'{v:.1e}' for v in sig), fontsize=10)
    fig.colorbar(im, ax=axes, label='parameter correlation', shrink=0.8)
    fig.suptitle('Step 4 -- cosmology parameter correlations from '
                 f'$F=D\\,C^{{-1}}D^{{\\sf T}}$ (Hartlap {hart:.2f})', y=1.02)
    fig.savefig(OUT / 'step4_fisher.png', dpi=140, bbox_inches='tight')
    plt.close(fig); print(f'Saved {OUT/"step4_fisher.png"}')

    # numbers for the note
    pm = fj.to_phys_cov(r['cov_marg'], PARAMS); pc = fj.to_phys_cov(r['cov_cond'], PARAMS)
    print('\nmarginalised sigma:', ', '.join(
        f'{p}={np.sqrt(pm[i,i]):.3e}' for i, p in enumerate(PARAMS)))
    print('conditional  sigma:', ', '.join(
        f'{p}={np.sqrt(pc[i,i]):.3e}' for i, p in enumerate(PARAMS)))


if __name__ == '__main__':
    main()

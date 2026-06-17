#!/usr/bin/env python3
"""
Joint cosmology + HOD Fisher with proper HOD marginalisation.

Builds one Fisher matrix over {ω_b, ω_c, n_s, σ₈} *and* the 12 yuan23 HOD
parameters, then marginalises over the HOD nuisances to get realistic
cosmology errors.

  F = D Cᵀ⁻¹ D  +  F_prior,
  D = [ ∂ξ/∂θ_cosmo  (4 rows, from derivative_hodcorr_* = fixed-HOD cosmology
                      derivative) ;
        ∂ξ/∂θ_HOD    (12 rows, from hod_gradient.npz) ],
  F_prior = diag(0₄ , 1/σ_prior²)   — Gaussian yuan23 prior on the HOD rows,
                                      σ_prior = std of the c000 prior draws.

Reported per cosmology parameter:
  * conditional sigma — invert the 4×4 cosmology sub-block of F_data
    (HOD known exactly; matches the earlier "HOD-corrected" full-box Fisher);
  * marginalised sigma — invert the full (4+12) F and read the 4×4 cosmology
    block of the inverse (HOD marginalised under its prior);
  * degradation = marginalised / conditional.

The HOD prior — not PCA truncation — regularises the poorly-constrained HOD
directions (truncation would be an implicit infinite prior and reintroduce the
fixed-HOD optimism).  PCA enters only as a *robustness check*: marginalising
over the top-k HOD response directions (prior-whitened, SVD-ordered) and showing
the cosmology errors plateau well before k=12.

Data vector: full-auto monopole + quadrupole (30 bins > 16 params; Hartlap
(64-30-2)/63 ≈ 0.51), covariance from the 64 c000 subboxes at full-box volume.

Output: plots/derivatives/fisher_joint_convergence.png
        plots/derivatives/fisher_joint_marginalisation.png

Usage (any node):
  python scripts/fisher_joint.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
DER_DIR   = DATA_DIR / 'derivatives'
PLOT_DIR  = REPO_ROOT / 'plots' / 'derivatives'

N_SB    = 64
VOL_FAC = N_SB              # full 2000 Mpc/h box = 64 subbox volumes
FID_TAG = 'c000_hod484'

# data vector: (stem, multipoles).  Full-auto mono+quad.
VECTOR = [('tpcf_full_data', (0, 2))]

COSMO = {                  # file tag -> (fiducial physical value, label, is_log)
    'lnwb': (0.02237,  r'\omega_b',     True),
    'lnwc': (0.1200,   r'\omega_{cdm}', True),
    'ns':   (0.9649,   r'n_s',          False),
    'lns8': (0.807952, r'\sigma_8',     True),
}


def data_vector_pieces(loader):
    """Concatenate the VECTOR pieces using loader(stem, ell) -> (nbins,) or (.,nbins)."""
    parts = [loader(stem, ell) for stem, ells in VECTOR for ell in ells]
    return np.hstack(parts) if parts[0].ndim > 1 else np.concatenate(parts)


def covariance_inv():
    """Hartlap-corrected inverse covariance at full-box volume, + nbins."""
    def load(stem, ell):
        return np.load(DATA_DIR / FID_TAG / f'subbox_multipoles_{stem}.npz')[f'xi{ell}_all']
    X    = data_vector_pieces(load)            # (64, nb)
    nb   = X.shape[1]
    hart = (N_SB - nb - 2) / (N_SB - 1)
    Cinv = hart * VOL_FAC * np.linalg.inv(np.cov(X, rowvar=False))
    return Cinv, nb


def cosmo_derivatives():
    """(4, nb) fixed-HOD cosmology derivatives from derivative_hodcorr_*."""
    rows, params = [], []
    for param in COSMO:
        f = DER_DIR / f'derivative_hodcorr_{param}.npz'
        if not f.is_file():
            raise SystemExit(f'missing {f.name}; run compute_hod_derivatives.py')
        der = np.load(f)
        rows.append(data_vector_pieces(lambda stem, ell: der[f'{stem}_dxi{ell}']))
        params.append(param)
    return np.array(rows), params


def hod_derivatives():
    """(12, nb) HOD gradient, names, prior std."""
    f = DER_DIR / 'hod_gradient.npz'
    if not f.is_file():
        raise SystemExit(f'missing {f.name}; run compute_hod_derivatives.py')
    g = np.load(f, allow_pickle=True)
    # g[f'{stem}_g{ell}'] is (nparam, nbins); concatenate pieces along bins
    blocks = [g[f'{stem}_g{ell}'] for stem, ells in VECTOR for ell in ells]
    D_hod  = np.hstack(blocks)                 # (nparam, nb)
    return D_hod, list(g['names']), g['param_std_prior']


def cosmo_block_sigma(F):
    """sqrt(diag) of the 4x4 cosmology block of F^{-1}."""
    cov = np.linalg.inv(F)
    return np.sqrt(np.diag(cov)[:4]), cov[:4, :4]


def to_phys(sig, param):
    fid, _, log = COSMO[param]
    return fid * sig if log else sig


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    Cinv, nb            = covariance_inv()
    D_cos, params       = cosmo_derivatives()
    D_hod, names, sd_pr = hod_derivatives()
    ncos, nhod          = len(params), len(names)
    print(f'Data vector: {nb} bins (full-auto mono+quad), Hartlap '
          f'{(N_SB - nb - 2) / (N_SB - 1):.2f}; {ncos} cosmo + {nhod} HOD params')

    D = np.vstack([D_cos, D_hod])               # (16, nb)
    F_data = D @ Cinv @ D.T

    # conditional (HOD fixed): invert the cosmology sub-block only
    sig_cond, _ = cosmo_block_sigma(F_data[:ncos, :ncos] + 0.0)
    sig_cond = np.array([sig_cond[i] for i in range(ncos)])

    # marginalised: full Fisher + Gaussian HOD prior block
    F_prior = np.zeros_like(F_data)
    F_prior[ncos:, ncos:] = np.diag(1.0 / sd_pr ** 2)
    F_tot = F_data + F_prior
    sig_marg, cov_cos = cosmo_block_sigma(F_tot)

    print(f'\n{"param":6s} {"sigma_cond":>12s} {"sigma_marg":>12s} {"degrade":>9s}')
    rows = []
    for i, p in enumerate(params):
        sc, sm = to_phys(sig_cond[i], p), to_phys(sig_marg[i], p)
        rows.append((p, sc, sm, sm / sc))
        print(f'{p:6s} {sc:12.5g} {sm:12.5g} {sm / sc:8.1f}x')

    # ---- PCA robustness check: marginalise over top-k HOD directions ----
    # whiten HOD params by the prior (prior -> identity), SVD-order the response
    Dw = D_hod * sd_pr[:, None]                  # response to 1-sigma_prior steps
    U, S, _ = np.linalg.svd(Dw, full_matrices=False)
    D_psi = (U.T @ Dw)                           # rotated HOD responses, S-ordered
    ks = range(0, nhod + 1)
    conv = np.zeros((len(list(ks)), ncos))
    for k in range(nhod + 1):
        Dk = np.vstack([D_cos, D_psi[:k]]) if k > 0 else D_cos
        Fk = Dk @ Cinv @ Dk.T
        Pk = np.zeros_like(Fk)
        if k > 0:
            Pk[ncos:, ncos:] = np.eye(k)         # identity prior in whitened coords
        s_k, _ = cosmo_block_sigma(Fk + Pk)
        conv[k] = [to_phys(s_k[i], params[i]) / rows[i][1] for i in range(ncos)]

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
    for i, p in enumerate(params):
        ax.plot(range(nhod + 1), conv[:, i], 'o-', color=colors[i],
                label=rf'${COSMO[p][1]}$')
    ax.axhline(1, color='grey', lw=0.8, ls=':')
    ax.set_xlabel('number of HOD response directions marginalised (prior-whitened, SVD-ordered)')
    ax.set_ylabel(r'$\sigma_{\rm marg}/\sigma_{\rm cond}$')
    ax.set_title('Joint Fisher: cosmology error vs HOD directions marginalised',
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p1 = PLOT_DIR / 'fisher_joint_convergence.png'
    fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'\nSaved {p1}')

    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(ncos)
    ax.bar(x - 0.2, [r[1] / r[1] for r in rows], 0.4, color='#999999',
           label='conditional (HOD fixed)')
    ax.bar(x + 0.2, [r[2] / r[1] for r in rows], 0.4, color='#377eb8',
           label='marginalised (HOD prior)')
    for i, r in enumerate(rows):
        ax.text(i + 0.2, r[2] / r[1] + 0.05, f'{r[3]:.1f}x', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'${COSMO[p][1]}$' for p in params])
    ax.set_ylabel(r'$\sigma\,/\,\sigma_{\rm cond}$')
    ax.set_title('Cosmology errors: HOD-fixed vs HOD-marginalised', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = PLOT_DIR / 'fisher_joint_marginalisation.png'
    fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p2}')


if __name__ == '__main__':
    main()

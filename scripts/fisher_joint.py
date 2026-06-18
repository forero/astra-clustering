#!/usr/bin/env python3
"""
Joint cosmology + HOD Fisher with proper HOD marginalisation, for several
data vectors.

Builds one Fisher over {ω_b, ω_c, n_s, σ₈} *and* the 12 yuan23 HOD parameters,
then marginalises over the HOD nuisances, and repeats for a set of data vectors
so the HOD-marginalised cosmology constraints can be compared:

  F = D Cᵀ⁻¹ D  +  F_prior,
  D = [ ∂ξ/∂θ_cosmo (4 rows, derivative_hodcorr_* = fixed-HOD cosmology deriv) ;
        ∂ξ/∂θ_HOD   (12 rows, hod_gradient.npz) ],
  F_prior = diag(0₄ , 1/σ_prior²)   — Gaussian yuan23 prior on the HOD rows.

Per data vector and cosmology parameter:
  * conditional sigma — invert the 4×4 cosmology sub-block of F_data (HOD fixed);
  * marginalised sigma — invert the full (4+12) F+prior, read the cosmology block.

The HOD prior — not PCA truncation — regularises the poorly-constrained HOD
directions.  The data vectors include the ASTRA environment splits (quantile
autos and crosses), the physical motivation being that under/overdense quantiles
respond differently to σ₈ vs the HOD and so may break degeneracies the full-auto
2PCF cannot.  Covariance from the 64 c000 subboxes at full-box volume; vectors
are rebinned where needed to keep nbins well below 64 (Hartlap sane).

Output: plots/derivatives/fisher_joint_convergence.png       (baseline vector)
        plots/derivatives/fisher_joint_marginalisation.png    (baseline vector)
        plots/derivatives/fisher_joint_ellipses.png           (baseline: cond vs marg)
        plots/derivatives/fisher_joint_ellipses_vectors.png   (marg, all vectors)

Usage (any node):
  python scripts/fisher_joint.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
DER_DIR   = DATA_DIR / 'derivatives'
PLOT_DIR  = REPO_ROOT / 'plots' / 'derivatives'

N_SB    = 64
VOL_FAC = N_SB              # full 2000 Mpc/h box = 64 subbox volumes
FID_TAG = 'c000_hod484'

COSMO = {                  # file tag -> (fiducial physical value, label, is_log)
    'lnwb': (0.02237,  r'\omega_b',     True),
    'lnwc': (0.1200,   r'\omega_{cdm}', True),
    'ns':   (0.9649,   r'n_s',          False),
    'lns8': (0.807952, r'\sigma_8',     True),
}

# named data vectors: list of (stem, multipoles, rebin_k).  rebin keeps nbins
# well under 64 (Hartlap (64-nb-2)/63).  Q = quantile (Q1 underdense .. Q4 over).
QS = range(1, 5)
VECTORS = {
    'full auto (mono+quad)':
        [('tpcf_full_data', (0, 2), 1)],                                  # 30 bins
    'data Q autos (mono, x2)':
        [(f'tpcf_data_q{q}', (0,), 2) for q in QS],                       # 32 bins
    'full x data Q (mono, x2)':
        [(f'tpcf_cross_full_data_q{q}', (0,), 2) for q in QS],            # 32 bins
    'full + data Q autos (mono, x2)':
        [('tpcf_full_data', (0,), 2)] +
        [(f'tpcf_data_q{q}', (0,), 2) for q in QS],                       # 40 bins
}
BASELINE = 'full auto (mono+quad)'


def rebin(arr, k):
    if k == 1:
        return arr
    a = np.atleast_2d(arr)
    n = a.shape[1]
    out = np.column_stack([a[:, i:i + k].mean(axis=1) for i in range(0, n, k)])
    return out[0] if out.shape[0] == 1 else out


def assemble(pieces):
    """Return Cinv, nb, hartlap, D_cosmo (4,nb), D_hod (12,nb), prior_std (12,)."""
    g    = np.load(DER_DIR / 'hod_gradient.npz', allow_pickle=True)
    ders = {p: np.load(DER_DIR / f'derivative_hodcorr_{p}.npz') for p in COSMO}
    Xp, Dh = [], []
    Dc = {p: [] for p in COSMO}
    for stem, ells, k in pieces:
        c0 = np.load(DATA_DIR / FID_TAG / f'subbox_multipoles_{stem}.npz')
        for ell in ells:
            Xp.append(rebin(c0[f'xi{ell}_all'], k))
            Dh.append(rebin(g[f'{stem}_g{ell}'], k))
            for p in COSMO:
                Dc[p].append(rebin(ders[p][f'{stem}_dxi{ell}'], k))
    X    = np.hstack(Xp)
    nb   = X.shape[1]
    hart = (N_SB - nb - 2) / (N_SB - 1)
    Cinv = hart * VOL_FAC * np.linalg.inv(np.cov(X, rowvar=False))
    D_cos = np.array([np.concatenate(Dc[p]) for p in COSMO])
    D_hod = np.hstack(Dh)
    return Cinv, nb, hart, D_cos, D_hod, g['param_std_prior']


def fisher(pieces):
    """Conditional and marginalised 4x4 cosmology covariances + the SVD-ordered
    whitened HOD response (for the convergence check)."""
    Cinv, nb, hart, D_cos, D_hod, sd_pr = assemble(pieces)
    ncos = len(COSMO)
    D = np.vstack([D_cos, D_hod])
    F_data = D @ Cinv @ D.T
    cov_cond = np.linalg.inv(F_data[:ncos, :ncos])
    F_prior = np.zeros_like(F_data)
    F_prior[ncos:, ncos:] = np.diag(1.0 / sd_pr ** 2)
    cov_marg = np.linalg.inv(F_data + F_prior)[:ncos, :ncos]
    # whitened, SVD-ordered HOD responses for the k-direction convergence test
    Dw = D_hod * sd_pr[:, None]
    U, _, _ = np.linalg.svd(Dw, full_matrices=False)
    D_psi = U.T @ Dw
    return dict(nb=nb, hart=hart, Cinv=Cinv, D_cos=D_cos, D_psi=D_psi,
                cov_cond=cov_cond, cov_marg=cov_marg)


def to_phys(sig, param):
    fid, _, log = COSMO[param]
    return fid * sig if log else sig


def to_phys_cov(cov, params):
    J = np.diag([COSMO[p][0] if COSMO[p][2] else 1.0 for p in params])
    return J @ cov @ J


def _ellipse(ax, mean, cov2, scale, **kw):
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * scale * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, **kw))


def corner(cov_sets, params, path, title, ranges_from=None):
    """cov_sets: list of (label, cov4x4_param_units, color). Draws 68/95% ellipses."""
    n   = len(params)
    fid = [COSMO[p][0] for p in params]
    phys = [(lab, to_phys_cov(C, params), c) for lab, C, c in cov_sets]
    rng = ranges_from if ranges_from is not None else phys[-1][1]
    sig = np.sqrt(np.diag(rng))
    s68, s95 = np.sqrt(2.30), np.sqrt(6.18)

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off'); continue
            if i == j:
                x = np.linspace(fid[i] - 4 * sig[i], fid[i] + 4 * sig[i], 400)
                for lab, C, c in phys:
                    s = np.sqrt(C[i, i])
                    ax.plot(x, np.exp(-0.5 * ((x - fid[i]) / s) ** 2),
                            color=c, lw=1.8, label=lab)
                ax.set_yticks([])
                if i == 0:
                    ax.legend(fontsize=6.5, loc='upper right')
            else:
                idx = [j, i]
                for lab, C, c in phys:
                    sub = C[np.ix_(idx, idx)]
                    _ellipse(ax, (fid[j], fid[i]), sub, s95, edgecolor=c, lw=0.9, ls='--')
                    _ellipse(ax, (fid[j], fid[i]), sub, s68, edgecolor=c, lw=1.7)
                ax.plot(fid[j], fid[i], 'k+', ms=5)
                ax.set_xlim(fid[j] - 4 * sig[j], fid[j] + 4 * sig[j])
                ax.set_ylim(fid[i] - 4 * sig[i], fid[i] + 4 * sig[i])
            if i == n - 1:
                ax.set_xlabel(rf'${COSMO[params[j]][1]}$')
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(rf'${COSMO[params[i]][1]}$')
            elif j == 0:
                ax.set_ylabel('like.')
            ax.tick_params(labelsize=7)
    fig.suptitle(title, y=0.98, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    params = list(COSMO)
    res = {name: fisher(pieces) for name, pieces in VECTORS.items()}

    # ---- table: HOD-marginalised sigma per parameter, per data vector ----
    print('\nHOD-marginalised sigma (physical units) per data vector:')
    print(f'{"data vector":32s} {"nb":>3s} {"hart":>5s} '
          + ' '.join(f'{COSMO[p][1]:>10s}' for p in params))
    for name in VECTORS:
        r = res[name]
        cells = ' '.join(f'{to_phys(np.sqrt(r["cov_marg"][i, i]), p):>10.3g}'
                         for i, p in enumerate(params))
        print(f'{name:32s} {r["nb"]:3d} {r["hart"]:5.2f} {cells}')
    print('\nfor reference, conditional (HOD-fixed) sigma, baseline vector:')
    rb = res[BASELINE]
    print(f'{BASELINE:32s} {"":3s} {"":5s} '
          + ' '.join(f'{to_phys(np.sqrt(rb["cov_cond"][i, i]), p):>10.3g}'
                     for i, p in enumerate(params)))

    # ---- baseline: convergence + conditional-vs-marginalised ellipses ----
    ncos = len(params)
    nhod = rb['D_psi'].shape[0]
    conv = np.zeros((nhod + 1, ncos))
    for k in range(nhod + 1):
        Dk = np.vstack([rb['D_cos'], rb['D_psi'][:k]]) if k > 0 else rb['D_cos']
        Fk = Dk @ rb['Cinv'] @ Dk.T
        Pk = np.zeros_like(Fk)
        if k > 0:
            Pk[ncos:, ncos:] = np.eye(k)
        ck = np.linalg.inv(Fk + Pk)[:ncos, :ncos]
        conv[k] = [to_phys(np.sqrt(ck[i, i]), params[i])
                   / to_phys(np.sqrt(rb['cov_cond'][i, i]), params[i])
                   for i in range(ncos)]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    cols = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']
    for i, p in enumerate(params):
        ax.plot(range(nhod + 1), conv[:, i], 'o-', color=cols[i], label=rf'${COSMO[p][1]}$')
    ax.axhline(1, color='grey', lw=0.8, ls=':')
    ax.set_xlabel('number of HOD response directions marginalised (prior-whitened, SVD-ordered)')
    ax.set_ylabel(r'$\sigma_{\rm marg}/\sigma_{\rm cond}$')
    ax.set_title(f'Joint Fisher convergence — {BASELINE}', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'fisher_joint_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'\nSaved {PLOT_DIR / "fisher_joint_convergence.png"}')

    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(ncos)
    cond = [np.sqrt(rb['cov_cond'][i, i]) for i in range(ncos)]
    marg = [np.sqrt(rb['cov_marg'][i, i]) for i in range(ncos)]
    ax.bar(x - 0.2, [1] * ncos, 0.4, color='#999999', label='conditional (HOD fixed)')
    ax.bar(x + 0.2, [m / c for m, c in zip(marg, cond)], 0.4, color='#377eb8',
           label='marginalised (HOD prior)')
    for i in range(ncos):
        ax.text(i + 0.2, marg[i] / cond[i] + 0.1, f'{marg[i] / cond[i]:.1f}x',
                ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'${COSMO[p][1]}$' for p in params])
    ax.set_ylabel(r'$\sigma\,/\,\sigma_{\rm cond}$')
    ax.set_title(f'HOD-fixed vs HOD-marginalised — {BASELINE}', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'fisher_joint_marginalisation.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print(f'Saved {PLOT_DIR / "fisher_joint_marginalisation.png"}')

    corner([('HOD fixed', rb['cov_cond'], '#999999'),
            ('HOD marginalised', rb['cov_marg'], '#377eb8')],
           params, PLOT_DIR / 'fisher_joint_ellipses.png',
           'Joint Fisher (68% solid / 95% dashed): HOD-fixed (grey) vs '
           f'HOD-marginalised (blue) — {BASELINE}')

    # ---- NEW: marginalised ellipses for all data vectors ----
    palette = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
    cov_sets = [(name, res[name]['cov_marg'], palette[i % len(palette)])
                for i, name in enumerate(VECTORS)]
    # widest vector sets the axis ranges
    widest = max(cov_sets, key=lambda t: np.sqrt(np.diag(to_phys_cov(t[1], params))).sum())
    corner(cov_sets, params, PLOT_DIR / 'fisher_joint_ellipses_vectors.png',
           'HOD-marginalised cosmology constraints by data vector '
           '(68% solid / 95% dashed)',
           ranges_from=to_phys_cov(widest[1], params))


if __name__ == '__main__':
    main()

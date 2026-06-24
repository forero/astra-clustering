#!/usr/bin/env python3
"""
Shared library for the EMULATOR-BASED FISHER.

The original Fisher had two structural weaknesses: (i) derivatives from the +/- HOD-
mismatched cosmology pairs (HOD contamination -- the Tier-0/1 saga), and (ii) C = C_CV
only (a perfect emulator -> optimistic value-add). This library fixes both using the
trained emulator: HOD-CLEAN derivatives d xi/d theta at FIXED HOD (any of the 8
w0waCDM+ params, including w0, wa), and a REALISTIC covariance C = C_CV + C_emu.

Provides: data/leg loading, subbox C_CV, CV-weighted emulator derivatives, a joint
cosmology+HOD Fisher with HOD marginalisation, leg sub-selection, and a corner-ellipse
plotter. The heavy step (train + LOCO C_emu + derivatives) is done once by
emufisher_build.py and cached; the forecast/campaign/validate scripts are fast.
"""
import glob, os
from pathlib import Path
import numpy as np
import emulator_tier3_mlp as emu

REPO = Path(__file__).resolve().parents[1]
COSMO = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
NHOD = 12                                              # 12 yuan23 HOD params (indices 8..19)


def load_vector(legs):
    """legs = list of (stem, ell). Returns X(20-D inputs), Y, Ynoise, cosmo, s, blocks."""
    a = emu.load('dataset.npz'); b = emu.load('dataset_anchor.npz')
    cols = np.concatenate([np.where((a['stem'] == st) & (a['ell'] == el))[0] for st, el in legs])
    X = np.vstack([a['X'], b['X']])
    Y = np.vstack([a['Y'][:, cols], b['Y'][:, cols]])
    Yn = np.vstack([a['Ynoise'][:, cols], b['Ynoise'][:, cols]])
    cosmo = np.concatenate([a['cosmo'], b['cosmo']]); s = a['s']; nb = len(s)
    blocks = [(st, el, slice(i * nb, (i + 1) * nb)) for i, (st, el) in enumerate(legs)]
    return X, Y, Yn, cosmo, s, blocks


def subbox_cov(legs, nb):
    """Joint box-volume C_CV for the legs (per-subbox xi{ell}_all, pooled, /64)."""
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    per = []
    for t in tags:
        cols, ok = [], True
        for st, el in legs:
            f = REPO / 'data' / t / f'subbox_multipoles_{st}.npz'
            if not f.is_file():
                ok = False; break
            cols.append(np.load(f)[f'xi{el}_all'])
        if ok:
            V = np.hstack(cols); per.append(V - V.mean(0))
    X = np.vstack(per)
    return np.cov(X, rowvar=False) / 64.0, X.shape[0]


def derivatives(predict, theta0, lo, hi, frac=0.05):
    """Central finite-difference dY/dtheta at theta0 for all 20 params (HOD-clean:
    HOD held at theta0[8:] when varying cosmology, and vice-versa)."""
    npar = len(theta0); ncol = len(predict(theta0)[0])
    D = np.zeros((ncol, npar))
    for k in range(npar):
        d = frac * (hi[k] - lo[k])
        if d <= 0:
            continue
        tp = theta0.copy(); tp[k] += d
        tm = theta0.copy(); tm[k] -= d
        D[:, k] = (predict(tp)[0] - predict(tm)[0]) / (2 * d)
    return D


def select_cols(blocks, want):
    """Column indices for a sub-vector: want = list of (stem, ell)."""
    idx = []
    for st, el, sl in blocks:
        if (st, el) in want:
            idx += list(range(sl.start, sl.stop))
    return np.array(idx)


def fisher(D, C_CV, C_emu, cosmo_idx, hod_prior, nsamp, alpha=1.0):
    """Joint cosmo+HOD Fisher with C = C_CV + alpha*C_emu.
    Returns (cond_cov, marg_cov) for the cosmo params (cosmo_idx into the 20-vector).
    cond = HOD fixed; marg = HOD marginalised (Gaussian yuan23 prior on the 12 HOD)."""
    C = C_CV + alpha * C_emu
    C = C + 1e-3 * np.median(np.diag(C)) * np.eye(len(C))
    hart = (nsamp - len(C) - 2) / (nsamp - 1)
    Cinv = hart * np.linalg.inv(C)
    F = D.T @ Cinv @ D                                 # 20 x 20
    hod_idx = list(range(8, 8 + NHOD))
    idx = list(cosmo_idx) + hod_idx
    Fs = F[np.ix_(idx, idx)]
    nc = len(cosmo_idx)
    cond = np.linalg.inv(Fs[:nc, :nc])                 # HOD fixed
    Fp = Fs.copy()
    Fp[nc:, nc:] += np.diag(1.0 / np.asarray(hod_prior) ** 2)
    marg = np.linalg.inv(Fp)[:nc, :nc]                 # HOD marginalised
    return cond, marg


def corner_ellipses(cov_list, mean, labels, colors, legend, out, title=''):
    """68/95% Fisher ellipses for the cosmo params (triangle layout)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    n = len(labels)
    fig, axs = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            if j > i:
                ax.axis('off'); continue
            if i == j:
                for C, col in zip(cov_list, colors):
                    sig = np.sqrt(C[i, i])
                    xx = np.linspace(mean[i] - 4 * sig, mean[i] + 4 * sig, 100)
                    ax.plot(xx, np.exp(-0.5 * ((xx - mean[i]) / sig) ** 2), color=col)
                ax.set_yticks([])
            else:
                for C, col in zip(cov_list, colors):
                    sub = C[np.ix_([j, i], [j, i])]
                    val, vec = np.linalg.eigh(sub)
                    ang = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
                    for k, nsig in enumerate([2.48, 1.52]):     # 95, 68%
                        e = Ellipse((mean[j], mean[i]), 2 * nsig * np.sqrt(val[1]),
                                    2 * nsig * np.sqrt(val[0]), angle=ang,
                                    facecolor=col, alpha=0.25 if k == 0 else 0.45, edgecolor=col)
                        ax.add_patch(e)
                ax.plot(mean[j], mean[i], 'k+', ms=6)
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=9)
    from matplotlib.lines import Line2D
    axs[0, n - 1].axis('off')
    axs[0, n - 1].legend([Line2D([0], [0], color=c, lw=4) for c in colors], legend,
                         fontsize=9, loc='center')
    fig.suptitle(title, y=1.0); fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)

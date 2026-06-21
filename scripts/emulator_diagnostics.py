#!/usr/bin/env python3
"""
GP emulator diagnostics at c000 — understand *how well* and *why* the emulator works.

Four panels (full-auto mono+quad vector, 50 c000 draws, leave-one-out unless noted):

  (a) Predicted vs true, with GP +/-1 sigma error bars. Points on the 1:1 line and
      error bars that reach it = good. Tells you accuracy and whether the GP knows
      when it is wrong.
  (b) Pull histogram  z = (pred - true) / sigma_GP  vs unit Gaussian. THE calibration
      test: std(z) ~ 1 means the GP predictive variance is trustworthy (so it can be
      propagated into the Fisher, as approach B needs); >1 = overconfident error bars,
      <1 = underconfident.
  (c) Learning curve: LOO RMS / signal spread vs training-set size. Shows the
      data-hunger directly — does the c000 emulator want more than 50 draws?
  (d) ARD relevance: inverse GP length-scale per HOD parameter (dominant PCA
      component). Which HOD parameters actually drive the data vector.

GP predictive variance is propagated PCA-component -> data-vector space:
  var(y_b) = ysd_b^2 * sum_k components[k,b]^2 * sigma_k^2.

Output: plots/emulator/gp_diagnostics_c000.png  (+ printed pull/accuracy summary)

Usage (any node):  python scripts/emulator_diagnostics.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

REPO = Path(__file__).resolve().parents[1]
FB   = REPO / 'data' / 'fullbox'
ENS  = REPO / 'data' / 'hod_ensemble'
PLOT = REPO / 'plots' / 'emulator'; PLOT.mkdir(parents=True, exist_ok=True)
COSMO, STEM = 'c000', 'tpcf_full_data'


def gp():
    k = (ConstantKernel(1.0) * Matern(length_scale=np.ones(12), nu=2.5)
         + WhiteKernel(1e-2))
    return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=2, alpha=1e-6)


def load():
    df = pd.read_csv(ENS / f'hod_params_{COSMO}.csv').set_index('hod')
    names = list(df.columns)
    X, Y, s = [], [], None
    for d in sorted(FB.glob(f'{COSMO}_hod*')):
        if not (d / 'fullbox_info.npz').is_file():
            continue
        hod = int(d.name.split('_hod')[1])
        if hod not in df.index:
            continue
        a = np.load(d / f'fullbox_multipoles_{STEM}.npz')
        s = a['s'] if s is None else s
        X.append(df.loc[hod].values.astype(float))
        Y.append(np.concatenate([a['xi0'], a['xi2']]))
    return np.array(X), np.array(Y), s, names


def pca_basis(Yz):
    ncomp = min(len(Yz) - 1, Yz.shape[1])
    p = PCA(n_components=ncomp).fit(Yz)
    keep = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.999) + 1)
    return p, keep


def predict_with_std(models, pca, keep, ysd, ymu, Zq):
    """GP mean + propagated 1-sigma in data-vector space for inputs Zq (m,12)."""
    m = len(Zq); ncomp = pca.n_components_
    comp_m = np.zeros((m, ncomp)); comp_s = np.zeros((m, keep))
    for k, mod in enumerate(models):
        mean, std = mod.predict(Zq, return_std=True)
        comp_m[:, k] = mean; comp_s[:, k] = std
    yz = pca.inverse_transform(comp_m)                       # (m, nfeat)
    V = pca.components_[:keep]                                # (keep, nfeat)
    var_z = (comp_s ** 2) @ (V ** 2)                          # (m, nfeat)
    return yz * ysd + ymu, np.sqrt(var_z) * ysd


def loo(X, Y, Xz):
    """Leave-one-out mean, std, truth in data space."""
    n = len(X); pred = np.zeros_like(Y); sig = np.zeros_like(Y)
    for i in range(n):
        tr = np.arange(n) != i
        ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
        Yz = (Y[tr] - ymu) / ysd
        p, keep = pca_basis(Yz)
        Tp = p.transform(Yz)[:, :keep]
        models = [gp().fit(Xz[tr], Tp[:, k]) for k in range(keep)]
        mean, std = predict_with_std(models, p, keep, ysd, ymu, Xz[i:i + 1])
        pred[i] = mean[0]; sig[i] = std[0]
    return pred, sig


def main():
    X, Y, s, names = load()
    n = len(X)
    prior = pd.read_csv(ENS / f'hod_params_{COSMO}.csv')[names].values
    xmu, xsd = prior.mean(0), prior.std(0)
    Xz = (X - xmu) / xsd
    spread = Y.std(0)

    pred, sig = loo(X, Y, Xz)
    rms = np.sqrt(np.mean((pred - Y) ** 2, 0))
    pulls = ((pred - Y) / sig).ravel()
    print(f'LOO median RMS/spread = {np.median(rms / spread):.3f}')
    print(f'Pull distribution: mean={pulls.mean():+.2f}  std={pulls.std():.2f} '
          f'(std~1 => GP error bars calibrated)')

    # learning curve: subsample training sizes, evaluate on the held-out remainder
    rng = np.random.default_rng(0)
    sizes = [10, 15, 20, 30, 40, n - 1]
    lc_mean, lc_err = [], []
    for m in sizes:
        vals = []
        for _ in range(6):
            idx = rng.permutation(n); tr, te = idx[:m], idx[m:]
            ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
            Yz = (Y[tr] - ymu) / ysd
            p, keep = pca_basis(Yz)
            Tp = p.transform(Yz)[:, :keep]
            mods = [gp().fit(Xz[tr], Tp[:, k]) for k in range(keep)]
            pr, _ = predict_with_std(mods, p, keep, ysd, ymu, Xz[te])
            vals.append(np.median(np.sqrt(np.mean((pr - Y[te]) ** 2, 0)) / spread))
        lc_mean.append(np.mean(vals)); lc_err.append(np.std(vals))

    # ARD relevance from the dominant PCA component fit on all draws
    ymu, ysd = Y.mean(0), Y.std(0); Yz = (Y - ymu) / ysd
    p, keep = pca_basis(Yz)
    g0 = gp().fit(Xz, p.transform(Yz)[:, 0])
    ls = g0.kernel_.k1.k2.length_scale                       # Matern ARD length scales
    relevance = 1.0 / np.asarray(ls)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    # (a) predicted vs true with error bars (s^2 xi, all bins/runs)
    ax = axs[0, 0]
    sb = np.concatenate([s, s]) ** 2
    T, P, S = (Y * sb).ravel(), (pred * sb).ravel(), (sig * sb).ravel()
    ax.errorbar(T, P, yerr=S, fmt='o', ms=2.5, lw=0.5, alpha=0.35, color='C0')
    lo, hi = T.min(), T.max(); ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel(r'truth  $s^2\xi$'); ax.set_ylabel(r'GP prediction  $s^2\xi$')
    ax.set_title('(a) predicted vs true (LOO, $\\pm1\\sigma_{GP}$)')

    # (b) pull histogram vs unit Gaussian
    ax = axs[0, 1]
    ax.hist(pulls, bins=40, density=True, color='C0', alpha=0.6,
            range=(-5, 5), label=f'pulls (std={pulls.std():.2f})')
    xx = np.linspace(-5, 5, 200)
    ax.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'k-', lw=1.5, label='N(0,1)')
    ax.set_xlabel(r'pull $(pred-true)/\sigma_{GP}$'); ax.set_ylabel('density')
    ax.set_title('(b) calibration of GP error bars'); ax.legend(fontsize=9)

    # (c) learning curve
    ax = axs[1, 0]
    ax.errorbar(sizes, lc_mean, yerr=lc_err, fmt='o-', color='C3', capsize=3)
    ax.set_xlabel('training draws'); ax.set_ylabel('held-out median RMS / signal spread')
    ax.set_title('(c) learning curve — data-hunger'); ax.grid(alpha=0.3)
    ax.axhline(0.3, color='grey', ls=':', lw=1)

    # (d) ARD relevance per HOD parameter
    ax = axs[1, 1]
    order = np.argsort(relevance)
    ax.barh(np.arange(12), relevance[order], color='C2')
    ax.set_yticks(np.arange(12)); ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel('relevance  1 / ARD length-scale  (dominant PC)')
    ax.set_title('(d) which HOD params drive the response')

    fig.suptitle(f'{COSMO} GP emulator diagnostics (full auto, mono+quad, {n} draws)',
                 y=1.0, fontsize=13)
    fig.tight_layout()
    out = PLOT / 'gp_diagnostics_c000.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

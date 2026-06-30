#!/usr/bin/env python3
"""
Emulability head-to-head: WEIGHTED-2PCF schemes vs the QUANTILE multi-leg vectors.

The fairer Fisher baseline (fisher_weighted_breakdown.py) showed the quantile
multi-leg vectors carry MORE raw cosmological information than the weighted mark --
the mark is a compression.  Its only remaining case is the secondary property:
the continuous weight is smooth/differentiable in the parameters and might be
*easier to emulate*, so that after accounting for emulator realism the usable
constraint could favour it.  This script tests exactly that, over the genuinely
space-filling axis (the c000 HOD ensemble), entirely from the cached runs once the
weighted c000 ensemble has been produced (queue/launch_weighted_c000_ensemble.sh).

Two measurements, identical emulator on both sides (standardise -> PCA 99.9% ->
per-component GP Matern-5/2 + white, leave-one-out CV over the 50 maximin HODs):

  (1) EMULABILITY.  median(LOO RMS / signal spread) per leg and overall.  Lower =
      more emulable (the emulator captures more of the HOD-induced variation).

  (2) EMULATOR-AWARE FISHER (the decisive one).  For each vector form the
      HOD-fixed marginalised sigma with C = C_CV (cosmic variance) and with
      C = C_CV + C_emu, where C_emu is the LOO-residual covariance of the emulator
      (SUNBIRD-style).  The "emulator tax" sigma_tot/sigma_CV says how much the
      emulator error degrades each vector; the head-to-head is whether the more
      emulable weighted vector closes the raw-information gap on sigma_tot.

Covariance: quantile vectors use the pooled 576-subbox cov; weighted vectors the
192-sample c000 reanalysis cov (weighted side at a Hartlap disadvantage, so a
weighted win would be conservative).  Derivatives are HOD-fixed over the matched
+/- pairs (same pre-Tier-1 contamination on both sides -> relative comparison).

Outputs
  plots/fullbox_weighted/emulability_rms_per_leg.png     (median RMS/spread per leg)
  plots/fullbox_weighted/emulability_fisher_tax.png      (sigma_CV vs sigma_tot per param)
  data/fullbox_weighted/cov/emulability_compare.npz

Run (any node; needs the weighted c000 ensemble + the 8 +/- weighted runs + covs):
  python scripts/emulability_weighted_vs_quantile.py
"""

import sys
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
FB = REPO / 'data' / 'fullbox'
WDIR = REPO / 'data' / 'fullbox_weighted'
ENS = REPO / 'data' / 'hod_ensemble'
DERIV = REPO / 'data' / 'derivatives'
WCOVF = WDIR / 'cov' / 'weighted_subbox_cov.npz'
PLOTS = REPO / 'plots' / 'fullbox_weighted'
SEL = REPO / 'data' / 'hod_calibration' / 'hod_selection_c000.txt'

COSMO = 'c000'
PAIRS = {
    'lnwb': ('c100_hod179', 'c101_hod152', 0.020, r'$\ln\omega_b$'),
    'lnwc': ('c102_hod556', 'c103_hod861', 0.033, r'$\ln\omega_c$'),
    'ns':   ('c104_hod498', 'c105_hod589', 0.010, r'$n_s$'),
    'lns8': ('c112_hod507', 'c113_hod483', 0.020, r'$\ln\sigma_8$'),
}
PARAMS = list(PAIRS)
PLABEL = [PAIRS[p][3] for p in PARAMS]

W_STATS = ('data', 'arand', 'cross')
W_ELLS = (0, 2)
TAGS9 = ['c000_hod484', 'c100_hod179', 'c101_hod152', 'c102_hod556', 'c103_hod861',
         'c104_hod498', 'c105_hod589', 'c112_hod507', 'c113_hod483']

# the two quantile multi-leg vectors to compare against
Q_DATAQ = [f'tpcf_data_q{i}' for i in (1, 2, 3, 4)]
Q_RANDQ = [f'tpcf_rand_q{i}' for i in (1, 2, 3, 4)]
Q_FULL_PLUS = ['tpcf_full_data'] + Q_DATAQ + Q_RANDQ          # Fisher winner


def hartlap(n, nb):
    return (n - nb - 2) / (n - 1)


def fisher_sigma(D, cov, n_samp):
    nb = cov.shape[0]
    h = hartlap(n_samp, nb)
    if h <= 0:
        return np.full(D.shape[0], np.nan), h
    try:
        Cinv = np.linalg.inv(cov) * h
        F = D @ Cinv @ D.T
        return np.sqrt(np.diag(np.linalg.inv(F))), h
    except np.linalg.LinAlgError:
        return np.full(D.shape[0], np.nan), h


# ── leg labels + ensemble loaders ────────────────────────────────────────────
def quantile_leglabels(stems):
    return [f'{s.replace("tpcf_", "")} l{e}' for s in stems for e in (0, 2)]


def w_leglabels():
    return [f'{st} l{e}' for st in W_STATS for e in W_ELLS]


def load_quantile_ensemble(stems, hods):
    """Y (n, nbin), Ynoise (nbin,) over the c000 HOD ensemble for the given stems."""
    Y, Ynoise, used = [], [], []
    s = None
    for h in hods:
        d = FB / f'{COSMO}_hod{h:03d}'
        if not (d / 'fullbox_info.npz').is_file():
            continue
        vec, nz, ok = [], [], True
        for stem in stems:
            f = d / f'fullbox_multipoles_{stem}.npz'
            if not f.is_file():
                ok = False; break
            a = np.load(f); s = a['s'] if s is None else s
            for e in (0, 2):
                vec.append(a[f'xi{e}']); nz.append(a[f'xi{e}_std'])
        if ok:
            Y.append(np.concatenate(vec)); Ynoise.append(np.concatenate(nz)); used.append(h)
    return np.array(Y), np.array(Ynoise), used, s


def load_weighted_ensemble(scheme, hods):
    """Y (n, 90), Ynoise (90,) for a weighted scheme over the ensemble.
    Block order: data l0, data l2, arand l0, arand l2, cross l0, cross l2."""
    Y, Ynoise, used = [], [], []
    s = None
    for h in hods:
        d = WDIR / f'{COSMO}_hod{h:03d}'
        if not (d / 'fbw_info.npz').is_file():
            continue
        vec, nz, ok = [], [], True
        for st in W_STATS:
            f = d / f'fbw_multipoles_{scheme}_{st}.npz'
            if not f.is_file():
                ok = False; break
            a = np.load(f); s = a['s'] if s is None else s
            for e in W_ELLS:
                vec.append(a[f'xi{e}']); nz.append(a[f'xi{e}_std'])
        if ok:
            Y.append(np.concatenate(vec)); Ynoise.append(np.concatenate(nz)); used.append(h)
    return np.array(Y), np.array(Ynoise), used, s


# ── the shared emulator (LOO, GP) ────────────────────────────────────────────
def gp_factory(ndim):
    def gp():
        k = (ConstantKernel(1.0) * Matern(length_scale=np.ones(ndim), nu=2.5)
             + WhiteKernel(1e-2))
        return GaussianProcessRegressor(kernel=k, normalize_y=True,
                                        n_restarts_optimizer=2, alpha=1e-6)
    return gp


def loo_emulate(Xz, Y):
    """Leave-one-out GP prediction of Y from standardised inputs Xz.
    Returns pred (n, nbin)."""
    n = len(Y)
    ymu, ysd = Y.mean(0), Y.std(0)
    ysd = np.where(ysd == 0, 1.0, ysd)
    Yz = (Y - ymu) / ysd
    ncomp = min(n - 1, Yz.shape[1])
    pca = PCA(n_components=ncomp).fit(Yz)
    keep = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.999) + 1)
    gp = gp_factory(Xz.shape[1])
    pred = np.zeros_like(Y)
    for i in range(n):
        tr = np.arange(n) != i
        Ytr = pca.transform(Yz[tr])[:, :keep]
        p = np.array([gp().fit(Xz[tr], Ytr[:, k]).predict(Xz[i:i + 1])[0]
                      for k in range(keep)])
        full = np.zeros((1, ncomp)); full[0, :keep] = p
        pred[i] = pca.inverse_transform(full)[0] * ysd + ymu
    return pred


# ── covariances and derivatives ──────────────────────────────────────────────
def pooled_quantile_cov(stems):
    per = []
    for tag in TAGS9:
        legs = []
        for stem in stems:
            z = np.load(REPO / 'data' / tag / f'subbox_multipoles_{stem}.npz')
            legs += [z['xi0_all'], z['xi2_all']]
        per.append(np.hstack(legs))
    per = [V - V.mean(0, keepdims=True) for V in per]
    X = np.vstack(per)
    return (X.T @ X) / (X.shape[0] - 1) / 64.0, X.shape[0]


def quantile_derivative(stems):
    D = []
    for p in PARAMS:
        d = np.load(DERIV / f'derivative_fullbox_{p}.npz')
        row = []
        for stem in stems:
            row += [d[f'{stem}_dxi0'], d[f'{stem}_dxi2']]
        D.append(np.hstack(row))
    return np.array(D)


def weighted_derivative(scheme):
    def wm(tag, st, e):
        return np.load(WDIR / tag / f'fbw_multipoles_{scheme}_{st}.npz')[f'xi{e}']
    D = []
    for p in PARAMS:
        tp, tm, dth, _ = PAIRS[p]
        if not ((WDIR / tp / 'fbw_info.npz').is_file() and (WDIR / tm / 'fbw_info.npz').is_file()):
            return None
        row = []
        for st in W_STATS:
            for e in W_ELLS:
                row.append((wm(tp, st, e) - wm(tm, st, e)) / (2 * dth))
        D.append(np.hstack(row))
    return np.array(D)


def emulability_and_fisher(Y, Ynoise, Xz, D, C_CV, n_cv):
    """median RMS/spread, RMS/noise, and (sigma_CV, sigma_tot) for one vector."""
    pred = loo_emulate(Xz, Y)
    resid = pred - Y
    rms = np.sqrt(np.mean(resid ** 2, 0))                    # (nbin,)
    spread = Y.std(0); spread = np.where(spread == 0, np.nan, spread)
    noise = Ynoise.mean(0) if Ynoise.ndim > 1 else Ynoise
    C_emu = np.cov(resid, rowvar=False)                      # (nbin, nbin)
    sig_cv, h = fisher_sigma(D, C_CV, n_cv)
    sig_tot, _ = fisher_sigma(D, C_CV + C_emu, n_cv)
    return {
        'rms_over_spread': np.nanmedian(rms / spread),
        'rms_over_noise': np.nanmedian(rms / np.where(noise == 0, np.nan, noise)),
        'rms': rms, 'spread': spread, 'noise': noise,
        'sigma_cv': sig_cv, 'sigma_tot': sig_tot, 'hart': h,
        'leg_rms_over_spread': None,                          # filled by caller
    }


def main():
    hods = [int(x) for x in SEL.read_text().split() if x.strip()]
    # standardised HOD inputs (prior spread over all 500 draws)
    df = pd.read_csv(ENS / f'hod_params_{COSMO}.csv').set_index('hod')
    names = list(df.columns)
    xmu, xsd = df[names].values.mean(0), df[names].values.std(0)

    if not WCOVF.is_file():
        sys.exit(f'Missing {WCOVF}; run scripts/weighted_subbox_cov.py first.')
    wcov = np.load(WCOVF, allow_pickle=True)
    schemes = [str(x) for x in wcov['schemes']]
    n_w = int(wcov['n_sub']) * int(wcov['n_iter'])

    # how many weighted ensemble runs are ready?
    n_w_ready = sum((WDIR / f'{COSMO}_hod{h:03d}' / 'fbw_info.npz').is_file() for h in hods)
    print(f'Weighted c000 ensemble runs ready: {n_w_ready}/{len(hods)}')
    if n_w_ready < 15:
        print('  (<15 weighted runs: the weighted emulator is not meaningful yet.\n'
              '   Run again once queue/launch_weighted_c000_ensemble.sh completes.)')

    results = {}        # label -> result dict
    leg_rms = {}        # label -> (leglabels, per-leg median rms/spread)

    # ===== quantile vectors =====
    for stems, label in [(['tpcf_full_data'], 'Q full-auto'),
                         (Q_FULL_PLUS, 'Q full+dataQ+randQ')]:
        Y, Yn, used, s = load_quantile_ensemble(stems, hods)
        if len(Y) < 15:
            print(f'  skip {label}: only {len(Y)} runs'); continue
        Xz = (df.loc[used, names].values - xmu) / xsd
        C_CV, n_cv = pooled_quantile_cov(stems)
        D = quantile_derivative(stems)
        r = emulability_and_fisher(Y, Yn, Xz, D, C_CV, n_cv)
        # per-leg median rms/spread
        labs = quantile_leglabels(stems); nb = len(s)
        pl = [np.median((r['rms'] / r['spread'])[i * nb:(i + 1) * nb]) for i in range(len(labs))]
        leg_rms[label] = (labs, pl)
        results[label] = r
        print(f'  {label:22s} n={len(Y)} med RMS/spread={r["rms_over_spread"]:.3f} '
              f'RMS/noise={r["rms_over_noise"]:.2f}')

    # ===== weighted schemes (full 3-leg vectors) =====
    if n_w_ready >= 15:
        for scheme in schemes:
            Y, Yn, used, s = load_weighted_ensemble(scheme, hods)
            if len(Y) < 15:
                continue
            Xz = (df.loc[used, names].values - xmu) / xsd
            C_CV = wcov[f'{scheme}_cov']
            D = weighted_derivative(scheme)
            if D is None:
                print(f'  skip weighted {scheme}: +/- runs incomplete'); continue
            r = emulability_and_fisher(Y, Yn, Xz, D, C_CV, n_w)
            labs = w_leglabels(); nb = len(s)
            pl = [np.median((r['rms'] / r['spread'])[i * nb:(i + 1) * nb]) for i in range(len(labs))]
            leg_rms[f'W:{scheme}'] = (labs, pl)
            results[f'W:{scheme}'] = r
            print(f'  W:{scheme:10s} n={len(Y)} med RMS/spread={r["rms_over_spread"]:.3f} '
                  f'RMS/noise={r["rms_over_noise"]:.2f}')

    if not results:
        sys.exit('No vectors had enough runs; nothing to report yet.')

    # ===== summary tables =====
    print('\n=== EMULABILITY (median LOO RMS / signal spread; lower = more emulable) ===')
    for lab, r in results.items():
        print(f'  {lab:24s} RMS/spread={r["rms_over_spread"]:.3f}   RMS/noise={r["rms_over_noise"]:.2f}')

    print('\n=== EMULATOR-AWARE FISHER (HOD-fixed marg sigma) ===')
    print(f'  {"vector":24s} {"hart":>5s} ' + ' '.join(f'{l:>22s}' for l in PLABEL))
    print(f'  {"":24s} {"":>5s} ' + ' '.join(f'{"CV -> CV+emu":>22s}' for _ in PLABEL))
    for lab, r in results.items():
        cells = []
        for j in range(len(PARAMS)):
            cells.append(f'{r["sigma_cv"][j]:.2e}->{r["sigma_tot"][j]:.2e}')
        print(f'  {lab:24s} {r["hart"]:5.2f} ' + ' '.join(f'{c:>22s}' for c in cells))
    print('\n  emulator tax (sigma_tot / sigma_CV; 1.0 = emulator-free):')
    for lab, r in results.items():
        tax = r['sigma_tot'] / r['sigma_cv']
        print(f'    {lab:24s} ' + ' '.join(f'{t:5.2f}x' for t in tax))

    # head-to-head: best weighted sigma_tot vs best quantile sigma_tot
    w_keys = [k for k in results if k.startswith('W:')]
    q_keys = [k for k in results if not k.startswith('W:')]
    if w_keys and q_keys:
        w_best = np.nanmin(np.vstack([results[k]['sigma_tot'] for k in w_keys]), 0)
        q_best = np.nanmin(np.vstack([results[k]['sigma_tot'] for k in q_keys]), 0)
        print('\n  HEAD-TO-HEAD on emulator-aware sigma_tot (quantile/weighted; >1 = weighted tighter):')
        print('    ' + ' '.join(f'{PARAMS[j]}={q_best[j]/w_best[j]:.2f}x' for j in range(len(PARAMS))))

    # ===== figures =====
    PLOTS.mkdir(parents=True, exist_ok=True)
    # fig 1: per-leg median RMS/spread
    fig, ax = plt.subplots(figsize=(13, 5))
    ypos, ylab = [], []
    y = 0
    for lab, (labs, pl) in leg_rms.items():
        for L, v in zip(labs, pl):
            ax.barh(y, v, color='tab:red' if lab.startswith('W:') else 'tab:blue')
            ylab.append(f'{lab} | {L}'); ypos.append(y); y += 1
        y += 0.6
    ax.axvline(1.0, color='k', ls='--', lw=1, label='RMS = signal spread')
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=6)
    ax.set_xlabel('median LOO RMS / signal spread (lower = more emulable)')
    ax.set_title('Emulability per leg — weighted (red) vs quantile (blue)')
    ax.legend(fontsize=8); ax.invert_yaxis()
    fig.tight_layout(); f1 = PLOTS / 'emulability_rms_per_leg.png'
    fig.savefig(f1, dpi=130); plt.close(fig); print(f'\nSaved {f1}')

    # fig 2: sigma_CV vs sigma_tot per param
    labels = list(results)
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4.5 * len(PARAMS), 4.5))
    x = np.arange(len(labels))
    for j, p in enumerate(PARAMS):
        ax = axes[j]
        cv = [results[k]['sigma_cv'][j] for k in labels]
        tot = [results[k]['sigma_tot'][j] for k in labels]
        cols = ['tab:red' if k.startswith('W:') else 'tab:blue' for k in labels]
        ax.bar(x, tot, color=cols, alpha=0.9, label=r'$\sigma_{\rm tot}$ (CV+emu)')
        ax.bar(x, cv, color='none', edgecolor='k', lw=1.2, label=r'$\sigma_{\rm CV}$')
        ax.set_yscale('log'); ax.set_title(PLABEL[j])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=6)
        if j == 0:
            ax.set_ylabel(r'$\sigma$ (HOD-fixed)'); ax.legend(fontsize=7)
    fig.suptitle('Emulator-aware Fisher — open bar = cosmic variance only, '
                 'filled = +emulator error (weighted red, quantile blue)')
    fig.tight_layout(); f2 = PLOTS / 'emulability_fisher_tax.png'
    fig.savefig(f2, dpi=130); plt.close(fig); print(f'Saved {f2}')

    np.savez(WDIR / 'cov' / 'emulability_compare.npz',
             params=PARAMS, labels=labels,
             **{f'{k}__rms_over_spread': np.array(results[k]['rms_over_spread']) for k in results},
             **{f'{k}__sigma_cv': results[k]['sigma_cv'] for k in results},
             **{f'{k}__sigma_tot': results[k]['sigma_tot'] for k in results})
    print(f'Saved {WDIR / "cov" / "emulability_compare.npz"}')


if __name__ == '__main__':
    main()

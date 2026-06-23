#!/usr/bin/env python3
"""
Visual gut-check: how do cosmological parameters move the inference legs around,
at ~fixed HOD?

For each cosmology we average the K HOD draws nearest the prior centre (standardised
HOD space), so HOD is held roughly fixed across cosmologies and the cosmology trend
isn't buried in HOD scatter.  Then plot s^2 xi(s) per leg, one line per cosmology,
coloured by a chosen cosmological parameter.  Uses all cosmologies on disk
(10 tier3 + 9 Fisher = 19), so the colour axis spans a broad range.

Outputs (plots/emulator_tier3/):
  cosmo_dependence_rand_sigma8.png / _omega_cdm.png / _n_s.png
  cosmo_dependence_data_sigma8.png

Usage (login node OK):  python scripts/plot_cosmology_dependence.py [--knn 5]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import emulator_tier3_mlp as emu

REPO = Path(__file__).resolve().parents[1]
# cosmology-parameter columns inside X_cosmo (build order), plus derived sigma8
COSMO_PARAMS = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
GROUPS = {
    'rand': (['tpcf_rand_q1', 'tpcf_rand_q4',
              'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4'],
             'random void/knot legs'),
    'data': (['tpcf_data_q1', 'tpcf_data_q4',
              'tpcf_cross_full_data_q1', 'tpcf_cross_full_data_q4'],
             'data void/knot legs'),
}
PRETTY = {'sigma8': r'$\sigma_8$', 'omega_cdm': r'$\omega_{cdm}$',
          'n_s': r'$n_s$', 'omega_b': r'$\omega_b$'}


def matched_curves(knn):
    """Return per-cosmology HOD-matched mean vector + cosmology-param dict."""
    a = emu.load('dataset.npz'); b = emu.load('dataset_anchor.npz')
    Y = np.vstack([a['Y'], b['Y']])
    X = np.vstack([a['X'], b['X']])                  # [8 cosmo | 12 HOD]
    Xc, Xh = X[:, :len(COSMO_PARAMS)], X[:, len(COSMO_PARAMS):]
    cosmo = np.concatenate([a['cosmo'], b['cosmo']])
    sig8 = np.concatenate([a['sigma8'], b['sigma8']])
    s, stem, ell = a['s'], a['stem'], a['ell']

    # standardise HOD over all runs; reference = prior centre (mean)
    Zh = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-12)
    d = np.linalg.norm(Zh, axis=1)

    cosmos = sorted(set(cosmo))
    curves, params, mism = {}, {c: {} for c in cosmos}, []
    for c in cosmos:
        m = np.where(cosmo == c)[0]
        if knn <= 0:                                # HOD-marginalised: average all draws
            sel = m
        else:
            sel = m[np.argsort(d[m])[:knn]]         # knn HODs closest to the centre
        curves[c] = Y[sel].mean(0)
        mism.append(d[sel].mean())
        for k, name in enumerate(COSMO_PARAMS):
            params[c][name] = Xc[m[0], k]
        params[c]['sigma8'] = sig8[m[0]]
    if knn <= 0:
        print(f'{len(cosmos)} cosmologies; HOD-MARGINALISED (mean over all '
              f'{int(np.median([np.sum(cosmo==c) for c in cosmos]))} draws/cosmology)')
    else:
        print(f'{len(cosmos)} cosmologies; matched-HOD set = {knn} draws nearest the '
              f'prior centre (mean |z| over chosen = {np.mean(mism):.2f})')
    return curves, params, s, stem, ell, cosmos


def figure(group_key, color_param, curves, params, s, stem, ell, cosmos, knn):
    stems, gtitle = GROUPS[group_key]
    vals = np.array([params[c][color_param] for c in cosmos])
    norm = Normalize(vals.min(), vals.max()); cmap = cm.viridis
    nb = len(s)

    fig, axs = plt.subplots(len(stems), 2, figsize=(10, 2.6 * len(stems)),
                            sharex=True, squeeze=False)
    for r, st in enumerate(stems):
        for cc, el in enumerate((0, 2)):
            ax = axs[r, cc]
            sl = np.where((stem == st) & (ell == el))[0]
            for c in cosmos:
                ax.plot(s, s**2 * curves[c][sl], lw=1.3,
                        color=cmap(norm(params[c][color_param])), alpha=0.85)
            ax.axhline(0, color='grey', lw=0.5)
            if r == 0:
                ax.set_title(f'$\\ell={el}$', fontsize=11)
            if cc == 0:
                ax.set_ylabel(f'{st.replace("tpcf_","")}\n' r'$s^2\xi$', fontsize=9)
            if r == len(stems) - 1:
                ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, shrink=0.6, pad=0.02)
    cbar.set_label(PRETTY.get(color_param, color_param), fontsize=12)
    if knn <= 0:
        desc, suff = 'HOD-marginalised (mean over all draws)', 'hodmarg'
    else:
        desc, suff = f'~fixed HOD ({knn} central draws)', f'knn{knn}'
    fig.suptitle(f'{gtitle}: cosmology dependence, {desc}, '
                 f'coloured by {PRETTY.get(color_param, color_param)}',
                 y=0.995, fontsize=12)
    out = REPO / f'plots/emulator_tier3/cosmo_dependence_{group_key}_{color_param}_{suff}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--knn', type=int, default=5)
    args = ap.parse_args()
    curves, params, s, stem, ell, cosmos = matched_curves(args.knn)
    for group, param in [('rand', 'sigma8'), ('rand', 'omega_cdm'),
                         ('rand', 'n_s'), ('data', 'sigma8')]:
        figure(group, param, curves, params, s, stem, ell, cosmos, args.knn)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Clean single-parameter response of the inference legs, SEPARATELY for void (Q1)
and peak/knot (Q4), using the controlled Fisher +/- pairs.

Each Fisher pair varies ONE cosmological parameter against the c000 fiducial:
  omega_b : c100(+) / c101(-)
  omega_cdm: c102(+) / c103(-)
  n_s     : c104(+) / c105(-)
  sigma8  : c112(+) / c113(-)
HOD is controlled by marginalising (mean over all 50 draws/cosmology), so each
panel shows what a single parameter does, at fixed everything else.

We plot the SHIFT  s^2 * (xi_var - xi_fid)  for + (red) and - (blue) -- robust to
the xi zero-crossing and showing the sign/shape of each parameter's effect.  The
fiducial s^2 xi is drawn faint (right axis) for context.

Outputs (plots/emulator_tier3/):
  param_response_rand_void.png / _rand_peak.png   (random Q1 / Q4 legs)
  param_response_data_void.png / _data_peak.png   (data   Q1 / Q4 legs)

Usage (login node OK):  python scripts/plot_param_response_pairs.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu

REPO = Path(__file__).resolve().parents[1]
FID = 'c000'
PAIRS = [(r'$\omega_b$', 'c100', 'c101'),
         (r'$\omega_{cdm}$', 'c102', 'c103'),
         (r'$n_s$', 'c104', 'c105'),
         (r'$\sigma_8$', 'c112', 'c113')]
ENV = {'void': 1, 'peak': 4}
FAMILY = {'rand': ('tpcf_rand_q%d', 'tpcf_cross_full_rand_q%d', 'random'),
          'data': ('tpcf_data_q%d', 'tpcf_cross_full_data_q%d', 'data')}


def cosmo_means():
    d = emu.load('dataset_anchor.npz')
    Y, cosmo = d['Y'], d['cosmo']
    means = {c: Y[cosmo == c].mean(0) for c in set(cosmo)}
    return means, d['s'], d['stem'], d['ell']


def figure(family, envname, means, s, stem, ell):
    auto_t, cross_t, fam = FAMILY[family]
    q = ENV[envname]
    legs = [(auto_t % q, 'auto'), (cross_t % q, 'cross')]
    fid = means[FID]
    fig, axs = plt.subplots(len(PAIRS), 4, figsize=(15, 2.5 * len(PAIRS)),
                            sharex=True, squeeze=False)
    for r, (pname, cp, cm) in enumerate(PAIRS):
        col = 0
        for st, legname in legs:
            for el in (0, 2):
                ax = axs[r, col]; col += 1
                sl = np.where((stem == st) & (ell == el))[0]
                dplus = s**2 * (means[cp][sl] - fid[sl])
                dminus = s**2 * (means[cm][sl] - fid[sl])
                ax.plot(s, dplus, 'C3-o', ms=2.5, lw=1.4, label=f'+ ({cp})')
                ax.plot(s, dminus, 'C0-o', ms=2.5, lw=1.4, label=f'− ({cm})')
                ax.axhline(0, color='grey', lw=0.6)
                # faint fiducial for context (right axis)
                axr = ax.twinx()
                axr.plot(s, s**2 * fid[sl], color='0.6', lw=0.8, ls=':')
                axr.set_yticks([])
                if r == 0:
                    ax.set_title(f'{legname} $\\ell={el}$', fontsize=10)
                if col == 1:
                    ax.set_ylabel(f'{pname}\n' r'$s^2\Delta\xi$', fontsize=10)
                if r == len(PAIRS) - 1:
                    ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    axs[0, 0].legend(fontsize=7, loc='best')
    fig.suptitle(f'{fam} {envname.upper()} (Q{q}) legs: single-parameter response '
                 f'(HOD-marginalised; dotted grey = fiducial $s^2\\xi$, right axis)',
                 y=0.998, fontsize=12)
    fig.tight_layout()
    out = REPO / f'plots/emulator_tier3/param_response_{family}_{envname}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {out}')


def main():
    means, s, stem, ell = cosmo_means()
    have = set(means)
    need = {FID} | {c for _, a, b in PAIRS for c in (a, b)}
    missing = need - have
    if missing:
        print(f'Missing cosmologies (need their anchor runs): {sorted(missing)}')
    for family in ('rand', 'data'):
        for envname in ('void', 'peak'):
            figure(family, envname, means, s, stem, ell)


if __name__ == '__main__':
    main()

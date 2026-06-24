#!/usr/bin/env python3
"""
emulator-based Fisher -- FORECAST (fast; loads emufisher_build.npz).

Forecasts marginalised cosmology errors for several data vectors with the realistic
budget C = C_CV + C_emu and HOD-clean emulator derivatives, marginalising the 12 HOD
nuisances. Compares the plain 2PCF, the greedy-curated vector, and the full ASTRA
vector; reports sigma (HOD-fixed and HOD-marginalised) per parameter + the FoM, and
draws the marginalised corner ellipses.

Outputs  plots/emulator_tier3/emufisher_forecast.png

Usage:  python scripts/emufisher_forecast.py
"""
from pathlib import Path
import numpy as np
from emufisher_lib import fisher, corner_ellipses, COSMO

REPO = Path(__file__).resolve().parents[1]
FITC = [0, 1, 2, 3, 6, 7]                              # omega_b, omega_cdm, h, n_s, w0, wa
FLAB = [r'$\omega_b$', r'$\omega_{cdm}$', '$h$', '$n_s$', '$w_0$', '$w_a$']
ENV = ['tpcf_data_q1', 'tpcf_data_q4', 'tpcf_cross_full_data_q1', 'tpcf_cross_full_data_q4',
       'tpcf_rand_q1', 'tpcf_rand_q4', 'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4']
VECTORS = {
    'full 2PCF':      [('tpcf_full_data', 0), ('tpcf_full_data', 2)],
    'curated':        [('tpcf_full_data', 0), ('tpcf_full_data', 2), ('tpcf_cross_full_rand_q1', 0)],
    'full + all ASTRA': [('tpcf_full_data', el) for el in (0, 2)] + [(st, el) for st in ENV for el in (0, 2)],
}


def main():
    d = np.load(REPO / 'data/emulator_tier3/emufisher_build.npz', allow_pickle=True)
    D, C_CV, C_emu = d['D'], d['C_CV'], d['C_emu']
    nsamp = int(d['nsamp']); theta0 = d['theta0']; hod_prior = d['hod_prior']
    stem, ell = d['leg_stem'].astype(str), d['leg_ell'].astype(int)
    nb = len(d['s'])                                    # bins per leg-block

    def cols_for(legs):
        want = set(legs)                               # expand each selected block to its nb columns
        return np.concatenate([np.arange(i * nb, (i + 1) * nb)
                               for i in range(len(stem)) if (stem[i], ell[i]) in want])

    margs, names, foms = [], [], []
    print(f'{"vector":18s} {"nb":>4s} ' + ' '.join(f'{COSMO[k][:6]:>9s}' for k in FITC) + '   FoM')
    base_fom = None
    for name, legs in VECTORS.items():
        idx = cols_for(legs)
        cond, marg = fisher(D[idx], C_CV[np.ix_(idx, idx)], C_emu[np.ix_(idx, idx)],
                            FITC, hod_prior, nsamp)
        sig = np.sqrt(np.diag(marg))
        fom = 1.0 / np.sqrt(np.linalg.det(marg))
        base_fom = base_fom or fom
        margs.append(marg); names.append(name); foms.append(fom / base_fom)
        print(f'{name:18s} {len(idx):4d} ' + ' '.join(f'{sig[i]:9.2g}' for i in range(len(FITC)))
              + f'   {fom/base_fom:5.2f}x')

    # ratios vs full 2PCF (per param), marginalised
    s0 = np.sqrt(np.diag(margs[0]))
    print('\nmarginalised sigma ratio vs full 2PCF (>1 = tighter):')
    for n, m in zip(names[1:], margs[1:]):
        s = np.sqrt(np.diag(m))
        print(f'  {n:18s} ' + ' '.join(f'{COSMO[k][:5]}={s0[i]/s[i]:.2f}' for i, k in enumerate(FITC)))

    # legend labels carry the FoM (relative to the full-2PCF baseline)
    labels = [f'{n} (FoM {f:.0f}$\\times$)' for n, f in zip(names, foms)]
    corner_ellipses(margs, theta0[FITC], FLAB, ['#d62728', '#1f77b4', '#2ca02c'], labels,
                    REPO / 'plots/emulator_tier3/emufisher_forecast.png',
                    title='Emulator-based Fisher (HOD-marginalised, C=C_CV+C_emu)')
    print(f'\nSaved {REPO / "plots/emulator_tier3/emufisher_forecast.png"}')


if __name__ == '__main__':
    main()

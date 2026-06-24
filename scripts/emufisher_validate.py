#!/usr/bin/env python3
"""
emulator-based Fisher -- VALIDATION against the MCMC (fast).

The Gaussian-linear Fisher is only trustworthy where the posterior is Gaussian. We
check it against the curated-vector MCMC (inference_curated.npz): same vector (full
2PCF monopole + void-random-cross monopole), same 4 LCDM params, HOD fixed and HOD
marginalised. If the Fisher sigma match the MCMC sigma, the Fisher forecast is
reliable for that regime.

Outputs: prints the Fisher-vs-MCMC sigma comparison.

Usage:  python scripts/emufisher_validate.py
"""
from pathlib import Path
import numpy as np
from emufisher_lib import fisher, COSMO

REPO = Path(__file__).resolve().parents[1]
FITC = [0, 1, 2, 3]                                    # MCMC curated fit the 4 LCDM params
VEC = [('tpcf_full_data', 0), ('tpcf_cross_full_rand_q1', 0)]   # monopole curated (matches MCMC)


def main():
    d = np.load(REPO / 'data/emulator_tier3/emufisher_build.npz', allow_pickle=True)
    D, C_CV, C_emu = d['D'], d['C_CV'], d['C_emu']
    nsamp = int(d['nsamp']); hod_prior = d['hod_prior']
    stem, ell = d['leg_stem'].astype(str), d['leg_ell'].astype(int)
    nb = len(d['s'])
    idx = np.concatenate([np.arange(i * nb, (i + 1) * nb)
                          for i in range(len(stem)) if (stem[i], ell[i]) in set(VEC)])

    cond, marg = fisher(D[idx], C_CV[np.ix_(idx, idx)], C_emu[np.ix_(idx, idx)],
                        FITC, hod_prior, nsamp)
    sig_f_cond, sig_f_marg = np.sqrt(np.diag(cond)), np.sqrt(np.diag(marg))

    m = np.load(REPO / 'data/emulator_tier3/inference_curated.npz', allow_pickle=True)
    sig_m_fix = m['ch_realfix'].std(0)                 # MCMC HOD-fixed
    sig_m_marg = m['ch_marg_c'].std(0)                 # MCMC HOD-marginalised

    print('Fisher vs MCMC sigma (curated monopole vector, 4 LCDM):')
    print(f'{"param":10s} {"Fish(fix)":>10s} {"MCMC(fix)":>10s}   {"Fish(marg)":>11s} {"MCMC(marg)":>11s}')
    for i, k in enumerate(FITC):
        print(f'{COSMO[k]:10s} {sig_f_cond[i]:10.2g} {sig_m_fix[i]:10.2g}   '
              f'{sig_f_marg[i]:11.2g} {sig_m_marg[i]:11.2g}   '
              f'(ratio fix {sig_f_cond[i]/sig_m_fix[i]:.2f}, marg {sig_f_marg[i]/sig_m_marg[i]:.2f})')
    print('\nratios near 1 => Gaussian Fisher reliable for this vector/regime.')


if __name__ == '__main__':
    main()

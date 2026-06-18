#!/usr/bin/env python3
"""
Empirical HOD covariance from the same-phase c000 HOD ensemble.

Every c000 full-box run shares the ph000 initial conditions, so the run-to-run
scatter of the data vector is *pure HOD response* across the yuan23 prior —
cosmic variance cancels.  Its sample covariance C_HOD is therefore the
covariance contribution of marginalising the HOD, measured directly (fully
nonlinear, no linear-gradient or Gaussian-prior assumption).

fisher_joint.py uses it for the alternative marginalisation route
C_total = C_CV + C_HOD (4 cosmology parameters only), as a cross-check on the
16-parameter joint Fisher with a yuan23 prior.

We save the per-draw full-box multipoles for every statistic so the Fisher can
rebin and concatenate them into whatever data vector it builds, then take the
covariance — exactly how it assembles the cosmic-variance covariance.  These are
already at full 2000 Mpc/h volume (no subbox volume scaling).

Outputs
  data/derivatives/hod_covariance.npz
      s, n_draws, and per (stem, ell): {stem}_xi{ell}  (n_draws, nbins)

Usage (any node):
  python scripts/compute_hod_covariance.py
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FB_DIR    = REPO_ROOT / 'data' / 'fullbox'
DER_DIR   = REPO_ROOT / 'data' / 'derivatives'

N_Q   = 4
COSMO = 'c000'
STEMS = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)


def main():
    runs = sorted(d for d in FB_DIR.glob(f'{COSMO}_hod*')
                  if (d / 'fullbox_info.npz').is_file())
    if len(runs) < 3:
        raise SystemExit(f'Need several completed {COSMO} full-box runs; '
                         f'have {len(runs)}.')
    print(f'{COSMO} HOD draws for C_HOD: {len(runs)}')

    out = {'n_draws': len(runs)}
    s = None
    for stem in STEMS:
        mats = {0: [], 2: []}
        for d in runs:
            z = np.load(d / f'fullbox_multipoles_{stem}.npz')
            if s is None:
                s = z['s']
            mats[0].append(z['xi0']); mats[2].append(z['xi2'])
        for ell in (0, 2):
            out[f'{stem}_xi{ell}'] = np.array(mats[ell])     # (n_draws, nbins)
    out['s'] = s

    np.savez(DER_DIR / 'hod_covariance.npz', **out)
    print(f'Saved {DER_DIR / "hod_covariance.npz"} '
          f'({len(runs)} draws, {len(STEMS)} stems x 2 multipoles)')


if __name__ == '__main__':
    main()

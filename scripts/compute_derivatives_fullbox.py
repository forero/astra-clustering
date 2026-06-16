#!/usr/bin/env python3
"""
Central-difference derivatives from the *full-box* (2000 Mpc/h) Fisher runs.

This is the Tier-0 alternative to scripts/compute_derivatives.py (which builds
the derivative from paired 500 Mpc/h subboxes).  The full box gives a cleaner
derivative numerator for two reasons:

  * the ± runs share initial-condition phases (ph000), so cosmic variance
    cancels in the difference xi(c+) - xi(c-);
  * the box uses periodic boundary conditions and ~4M galaxies, so there is
    no open-boundary integral-constraint offset and far lower shot noise than
    a 500 Mpc/h subbox.

Per stem and multipole,

    d xi_i / d theta = [ xi_i(c+) - xi_i(c-) ] / (2 dtheta)

with xi the iteration-averaged full-box measurement (xi0/xi2 in the file).

Derivative noise model.  Each full-box xi is a mean over N_ITER ASTRA-random
iterations of a *fixed* box, so xi{ell}_std (the scatter across iterations)
captures the ASTRA-random + estimator noise only — exactly the part that does
NOT cancel in the phase-matched difference.  The diagonal noise variance of
the derivative is therefore

    Var(d xi/d theta)_bin = [ std_+^2 / n_+  +  std_-^2 / n_- ] / (2 dtheta)^2

(std^2/n is the variance of the iteration mean).  This is a diagonal estimate
from only N_ITER iterations — it ignores bin-to-bin noise correlations and so
tends to *under*-state the noise bias; treat the resulting Fisher sigmas as
optimistic.  The covariance itself still comes from the 64 c000 subboxes
(see scripts/plot_fisher_fullbox_compare.py).

Outputs
  data/derivatives/derivative_fullbox_{param}.npz
      s, dtheta, tag_plus, tag_minus, n_plus, n_minus, and per (stem, ell):
        {stem}_dxi{ell}          mean derivative              (nbins,)
        {stem}_dxi{ell}_noisevar diagonal noise variance      (nbins,)

Usage (any node):
  python scripts/compute_derivatives_fullbox.py
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
FB_DIR    = DATA_DIR / 'fullbox'
OUT_DIR   = DATA_DIR / 'derivatives'

N_Q = 4

# Same ± pairs and half-steps as the subbox derivatives.
PAIRS = {
    'lnwb': ('c100_hod179', 'c101_hod152', 0.020, r'\ln\omega_b'),
    'lnwc': ('c102_hod556', 'c103_hod861', 0.033, r'\ln\omega_c'),
    'ns':   ('c104_hod498', 'c105_hod589', 0.010, r'n_s'),
    'lns8': ('c112_hod507', 'c113_hod483', 0.020, r'\ln\sigma_8'),
}

STEMS = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)


def run_complete(tag):
    return (FB_DIR / tag / 'fullbox_info.npz').is_file()


def n_iter(tag):
    info = np.load(FB_DIR / tag / 'fullbox_info.npz')
    return int(info['n_iterations'])


def compute_derivative(tag_p, tag_m, dtheta, n_p, n_m):
    out = {}
    s = None
    for stem in STEMS:
        dp = np.load(FB_DIR / tag_p / f'fullbox_multipoles_{stem}.npz')
        dm = np.load(FB_DIR / tag_m / f'fullbox_multipoles_{stem}.npz')
        if s is None:
            s = dp['s']
        for ell in (0, 2):
            key = f'xi{ell}'
            d   = (dp[key] - dm[key]) / (2 * dtheta)
            # variance of each iteration-mean = std^2 / n_iter
            var = (dp[key + '_std'] ** 2 / n_p
                   + dm[key + '_std'] ** 2 / n_m) / (2 * dtheta) ** 2
            out[f'{stem}_dxi{ell}']          = d
            out[f'{stem}_dxi{ell}_noisevar'] = var
    return s, out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for param, (tag_p, tag_m, dtheta, label) in PAIRS.items():
        if not (run_complete(tag_p) and run_complete(tag_m)):
            missing = [t for t in (tag_p, tag_m) if not run_complete(t)]
            print(f'Skipping d/d{param}: missing full-box {", ".join(missing)}')
            continue
        n_p, n_m = n_iter(tag_p), n_iter(tag_m)
        print(f'=== d/d{param}:  full-box ({tag_p} - {tag_m}) / {2 * dtheta}  '
              f'(n_iter {n_p}/{n_m}) ===')
        s, der = compute_derivative(tag_p, tag_m, dtheta, n_p, n_m)
        np.savez(OUT_DIR / f'derivative_fullbox_{param}.npz',
                 s=s, dtheta=dtheta, tag_plus=tag_p, tag_minus=tag_m,
                 n_plus=n_p, n_minus=n_m, **der)
        print(f'Saved {OUT_DIR / f"derivative_fullbox_{param}.npz"}')
        done += 1
    if done == 0:
        raise SystemExit('No complete full-box ± pair found.')


if __name__ == '__main__':
    main()

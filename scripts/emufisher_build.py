#!/usr/bin/env python3
"""
emulator-based Fisher -- BUILD step (the one GPU run; everything else is fast).

For the full candidate vector (full 2PCF + 8 environment stems, monopole AND
quadrupole = 18 legs), with the CV-weighted emulator, compute and cache:
  D      : HOD-clean derivatives d xi/d theta at the c000 fiducial, for ALL 20 params
           (8 cosmology incl w0,wa + 12 HOD), by central finite-difference;
  C_CV   : subbox cosmic-variance covariance (box volume);
  C_emu  : emulator-error covariance from a CV-weighted leave-one-cosmology-out;
  hod_prior, theta0, leg labels, s.
The forecast / campaign / validate scripts load this -- no re-training.

Outputs  data/emulator_tier3/emufisher_build.npz

Usage (GPU node):  python scripts/emufisher_build.py [--ensemble 3] [--epochs 2000]
"""
import argparse
from pathlib import Path
import numpy as np

import emulator_tier3_mlp as emu
from inference_monopole import train_emulator
from emufisher_lib import load_vector, subbox_cov, derivatives, COSMO, NHOD

REPO = Path(__file__).resolve().parents[1]
LEGS = [(st, el) for st in
        (['tpcf_full_data'] + ['tpcf_data_q1', 'tpcf_data_q4',
         'tpcf_cross_full_data_q1', 'tpcf_cross_full_data_q4', 'tpcf_rand_q1',
         'tpcf_rand_q4', 'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4'])
        for el in (0, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}  {len(LEGS)} legs')

    X, Y, Yn, cosmo, s, blocks = load_vector(LEGS)
    nb = len(s); ncol = Y.shape[1]
    C_CV, nsamp = subbox_cov(LEGS, nb); cv = np.sqrt(np.diag(C_CV))
    print(f'{len(Y)} runs; vector {ncol}-D; C_CV {nsamp} samples')

    # C_emu: CV-weighted leave-one-cosmology-out residual covariance
    resid = np.zeros_like(Y)
    for c in sorted(set(cosmo)):
        te = cosmo == c
        pr = train_emulator(X, Y, Yn, exclude=list(np.where(te)[0]),
                            n_ens=args.ensemble, epochs=args.epochs, cv=cv)
        resid[te] = Y[te] - np.array([pr(X[i])[0] for i in np.where(te)[0]])
    C_emu = np.cov(resid, rowvar=False)
    print(f'C_emu/C_CV diag med = {np.median(np.sqrt(np.diag(C_emu)/np.diag(C_CV))):.2f}')

    # HOD-clean derivatives: emulator trained on ALL, finite-diff at c000 fiducial
    predict = train_emulator(X, Y, Yn, exclude=[], n_ens=max(3, args.ensemble),
                             epochs=args.epochs, cv=cv)
    c0 = np.where(cosmo == 'c000')[0]; theta0 = X[c0[len(c0) // 2]].copy()
    lo, hi = X.min(0), X.max(0)
    D = derivatives(predict, theta0, lo, hi)           # ncol x 20
    hod_prior = X[:, 8:].std(0)                        # yuan23 prior spread (12)

    # sanity: d xi / d ln sigma8 ~ 2 xi  (use omega_cdm as a sigma8 proxy is not exact;
    # instead report the full_data l0 derivative magnitudes for inspection downstream)
    np.savez(REPO / 'data/emulator_tier3/emufisher_build.npz',
             D=D, C_CV=C_CV, C_emu=C_emu, nsamp=nsamp, s=s,
             leg_stem=np.array([b[0] for b in blocks]),
             leg_ell=np.array([b[1] for b in blocks]),
             theta0=theta0, hod_prior=hod_prior,
             cosmo_names=np.array(COSMO))
    print(f'Saved {REPO / "data/emulator_tier3/emufisher_build.npz"}')


if __name__ == '__main__':
    main()

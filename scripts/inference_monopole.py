#!/usr/bin/env python3
"""
Monopole-only inference PROTOTYPE: SUNBIRD-style recovery test at the LCDM fiducial.

The monopole LOCO validation showed the emulator is inference-grade in the
LCDM-neighbourhood (Fisher cosmologies ~1.1x CV) but extrapolation-limited over the
broad tier3 hull.  So this demonstrates the full pipeline where coverage is adequate:
recover the c000 cosmology from a held-out c000 mock.

Pipeline (the SUNBIRD recipe):
  forward model  mu(theta) = monopole MLP emulator (trained on all 19 cosmologies)
  covariance     C_tot = C_CV (subbox, box volume) + C_emu (emulator-error, from the
                 Fisher-set LOCO residuals -- the LCDM-neighbourhood error, NOT the
                 tier3-edge-inflated global one)
  likelihood     -0.5 (d - mu)^T C_tot^-1 (d - mu),  emcee over the 4 LCDM params
                 {omega_b, omega_cdm, h, n_s}; broad params fixed at LCDM; HOD fixed
                 at the mock's truth (HOD-marginalisation is the next layer).

Legs: the well-behaved monopoles -- knot + random q1/q4 autos & full-crosses.  The
void DATA monopoles (data_q1, cross_full_data_q1) are excluded: near their xi
zero-crossing CV->0 so RMS/CV blows up (127x, 385x) -- pathological, not informative.

Outputs
  data/emulator_tier3/inference_monopole.npz   chain + truth + summary
  plots/emulator_tier3/inference_monopole_corner.png

Usage (GPU node):  python scripts/inference_monopole.py [--steps 4000] [--ensemble 3]
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import emcee

import emulator_tier3_mlp as emu

REPO = Path(__file__).resolve().parents[1]
LEGS = ['tpcf_data_q4', 'tpcf_cross_full_data_q4',
        'tpcf_rand_q1', 'tpcf_rand_q4',
        'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4']     # clean monopole legs
COSMO_NAMES = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
ALL_LABELS = [r'\omega_b', r'\omega_{cdm}', 'h', 'n_s', r'\alpha_s', r'N_{ur}', 'w_0', 'w_a']
# fit-parameter sets: 'lcdm' = 4 LCDM; 'broad' = the params monopole can hope to
# constrain (skip alpha_s, N_ur which need more); set at runtime.
FITSETS = {'lcdm': [0, 1, 2, 3], 'broad': [0, 1, 2, 3, 6, 7]}
FIT = FITSETS['lcdm']                                             # overridden in main()
FIT_LABELS = [ALL_LABELS[i] for i in FIT]


def load_monopole(name):
    d = emu.load(name)
    cols = np.where(np.isin(d['stem'], LEGS) & (d['ell'] == 0))[0]
    return d['X'], d['Y'][:, cols], d['Ynoise'][:, cols], d['cosmo'], d['stem'][cols], d['ell'][cols], cols, d['s']


def subbox_cov(stems, nb):
    """Joint box-volume covariance for the selected monopole legs (column order = LEGS)."""
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    per_tag = []
    for t in tags:
        cols, ok = [], True
        for st in LEGS:
            f = REPO / 'data' / t / f'subbox_multipoles_{st}.npz'
            if not f.is_file():
                ok = False; break
            cols.append(np.load(f)['xi0_all'])            # (n_subbox, nbins) per-subbox
        if ok:
            V = np.hstack(cols); per_tag.append(V - V.mean(0))
    X = np.vstack(per_tag)
    return np.cov(X, rowvar=False) / 64.0, X.shape[0]


def train_emulator(X, Y, Yn, exclude, n_ens, epochs):
    """Train the monopole MLP ensemble; return predict(theta)->xi and standardisation."""
    keep = np.ones(len(X), bool); keep[exclude] = False
    Xtr, Ytr, Ntr = X[keep], Y[keep], Yn[keep]
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-12
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-12
    nb = np.mean(Ntr, 0)
    w = (ysd / (nb + 1e-12)) ** 2; w = w / w.mean()
    Xz = (Xtr - xmu) / xsd; Yz = (Ytr - ymu) / ysd
    rng = np.random.default_rng(0)
    va = rng.choice(len(Xz), max(1, len(Xz) // 7), replace=False)
    tr = np.setdiff1d(np.arange(len(Xz)), va)
    models = []
    for e in range(n_ens):
        m, _ = emu.train_one(Xz[tr], Yz[tr], w, Xz[va], Yz[va], seed=e + 1, epochs=epochs)
        models.append(m)

    def predict(theta):
        z = (np.atleast_2d(theta) - xmu) / xsd
        t = torch.tensor(z, dtype=torch.float32, device=emu.DEVICE)
        with torch.no_grad():
            p = np.mean([mm(t).cpu().numpy() for mm in models], 0)
        return p * ysd + ymu
    return predict


def main():
    global FIT, FIT_LABELS
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--mock-cosmo', default='c000', help='cosmology to recover (e.g. c130)')
    ap.add_argument('--fit', default='lcdm', choices=list(FITSETS), help='parameter set')
    args = ap.parse_args()
    FIT = FITSETS[args.fit]; FIT_LABELS = [ALL_LABELS[i] for i in FIT]
    print(f'device={emu.DEVICE}  mock={args.mock_cosmo}  fit={args.fit} {FIT}')

    # ---- data ----
    Xa, Ya, Na, ca, stems, ells, cols, s = load_monopole('dataset.npz')
    Xb, Yb, Nb, cb, *_ = load_monopole('dataset_anchor.npz')
    X = np.vstack([Xa, Xb]); Y = np.vstack([Ya, Yb]); Yn = np.vstack([Na, Nb])
    cosmo = np.concatenate([ca, cb])
    nb = len(s); print(f'{len(Y)} runs; monopole vector {Y.shape[1]}-D ({len(LEGS)} legs x {nb})')

    # ---- mock: a held-out run of the requested cosmology ----
    msel = np.where(cosmo == args.mock_cosmo)[0]
    mock = int(msel[len(msel) // 2])
    d = Y[mock].copy()
    theta_true = X[mock, :8].copy()
    theta_hod = X[mock, 8:].copy()
    print(f'mock = {args.mock_cosmo} run #{mock}; truth: '
          + ', '.join(f'{COSMO_NAMES[i]}={theta_true[i]:.4g}' for i in FIT))

    # ---- covariances ----
    C_CV, nsamp = subbox_cov(stems, nb)
    lc = np.load(REPO / 'data/emulator_tier3/monopole_loco.npz', allow_pickle=True)
    lc_stems = lc['stems'].astype(str); lc_cos = lc['cosmo'].astype(str)
    leg_cols = np.concatenate([np.where(lc_stems == st)[0] * nb + np.arange(nb)
                               for st in LEGS if st in lc_stems])
    # C_emu: LCDM-neighbourhood residuals for an LCDM mock, else all-cosmology residuals
    grp = (np.array([int(c[1:]) < 130 for c in lc_cos]) if args.mock_cosmo[0] == 'c'
           and int(args.mock_cosmo[1:]) < 130 else np.ones(len(lc_cos), bool))
    C_emu = np.cov(lc['resid'][np.ix_(grp, leg_cols)], rowvar=False)
    # mock measurement (label) noise: the c000 run is a 3-iteration mean
    C_label = np.diag((Yn[mock] ** 2) / 3.0)
    hartlap = (nsamp - len(C_CV) - 2) / (nsamp - 1)
    print(f'C_CV {nsamp} samples; Hartlap={hartlap:.2f}; '
          f'C_emu/C_CV diag med={np.median(np.sqrt(np.diag(C_emu)/np.diag(C_CV))):.2f}  '
          f'C_label/C_CV diag med={np.median(np.sqrt(np.diag(C_label)/np.diag(C_CV))):.2f}')

    # ---- emulator (exclude the real mock run from training) ----
    predict = train_emulator(X, Y, Yn, exclude=[mock], n_ens=args.ensemble, epochs=args.epochs)

    lo = X[:, :8].min(0)[FIT]; hi = X[:, :8].max(0)[FIT]

    def recover(d_vec, C_tot, theta_truth, tag):
        C = C_tot + 1e-3 * np.median(np.diag(C_tot)) * np.eye(len(C_tot))
        Cinv = hartlap * np.linalg.inv(C)
        r0 = d_vec - predict(np.concatenate([theta_truth, theta_hod]))[0]
        chi2 = r0 @ Cinv @ r0 / len(d_vec)

        def logprob(p):
            if np.any(p < lo) or np.any(p > hi):
                return -np.inf
            th = theta_truth.copy(); th[FIT] = p
            r = d_vec - predict(np.concatenate([th, theta_hod]))[0]
            return -0.5 * r @ Cinv @ r

        ndim, nwalk = len(FIT), 32
        p0 = theta_truth[FIT] + (hi - lo) * 1e-3 * np.random.randn(nwalk, ndim)
        sm = emcee.EnsembleSampler(nwalk, ndim, logprob)
        sm.run_mcmc(p0, args.steps, progress=False)
        ch = sm.get_chain(discard=args.steps // 2, flat=True)
        mean, std = ch.mean(0), ch.std(0)
        print(f'\n=== recovery: {tag} (chi2/dof at truth = {chi2:.2f}) ===')
        for i, k in enumerate(FIT):
            print(f'  {COSMO_NAMES[k]:10s} truth={theta_truth[k]:.4g}  '
                  f'post={mean[i]:.4g}±{std[i]:.2g}  ({(mean[i]-theta_truth[k])/std[i]:+.1f}σ)')
        return ch, mean, std

    # (A) SYNTHETIC mock: emulator at a displaced cosmology + a draw from C_CV+C_emu.
    #     Recovers by construction -> validates the sampler/likelihood machinery.
    lo8 = X[:, :8].min(0); hi8 = X[:, :8].max(0)         # full 8-param ranges (do not clobber lo/hi)
    theta_inj = theta_true.copy()
    for k in FIT:                                        # displace each fitted param ~30% toward an edge
        theta_inj[k] = theta_true[k] + 0.3 * (hi8[k] - theta_true[k] if theta_true[k] < (lo8[k]+hi8[k])/2
                                              else lo8[k] - theta_true[k])
    C_synth = C_CV + C_emu
    rng = np.random.default_rng(1)
    d_synth = predict(np.concatenate([theta_inj, theta_hod]))[0] \
        + rng.multivariate_normal(np.zeros(len(C_synth)), C_synth)
    ch_s, mean_s, std_s = recover(d_synth, C_synth, theta_inj, 'SYNTHETIC (machinery)')

    # (B) REAL mock with label-noise-calibrated covariance.
    ch_r, mean_r, std_r = recover(d, C_CV + C_emu + C_label, theta_true,
                                  f'REAL {args.mock_cosmo} (calibrated)')

    tag = f'{args.mock_cosmo}_{args.fit}'
    np.savez(REPO / f'data/emulator_tier3/inference_{tag}.npz',
             chain_synth=ch_s, truth_synth=theta_inj[FIT],
             chain_real=ch_r, truth_real=theta_true[FIT],
             names=np.array([COSMO_NAMES[k] for k in FIT]))

    try:
        from getdist import MCSamples, plots
        ndim = len(FIT)
        ss = [MCSamples(samples=ch_s, names=[f'p{i}' for i in range(ndim)], labels=FIT_LABELS, label='synthetic'),
              MCSamples(samples=ch_r, names=[f'p{i}' for i in range(ndim)], labels=FIT_LABELS, label=f'real {args.mock_cosmo}')]
        g = plots.get_subplot_plotter()
        g.triangle_plot(ss, filled=True,
                        legend_labels=[f'synthetic (red truth)', f'real {args.mock_cosmo} (black truth)'])
        # the two chains have DIFFERENT truths: mark both (black=real mock, red=synthetic injection)
        for i in range(ndim):
            for j in range(i + 1):
                ax = g.subplots[i, j]
                if ax is None:
                    continue
                ax.axvline(theta_true[FIT[j]], color='k', lw=1, ls='--')
                ax.axvline(theta_inj[FIT[j]], color='r', lw=1, ls=':')
                if i != j:
                    ax.axhline(theta_true[FIT[i]], color='k', lw=1, ls='--')
                    ax.axhline(theta_inj[FIT[i]], color='r', lw=1, ls=':')
        out = REPO / f'plots/emulator_tier3/inference_{tag}_corner.png'
        g.export(str(out)); print(f'Saved {out}  (black dashed = {args.mock_cosmo} truth, red dotted = synthetic)')
    except Exception as ex:
        print(f'corner plot skipped: {ex}')


if __name__ == '__main__':
    main()

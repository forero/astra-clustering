#!/usr/bin/env python3
"""
Cheap architecture/loss sweep aimed at the legs that gate the ASTRA value-add.

The value-add is killed by the emulator error on the environment-quantile legs
(C_emu ~ 4.6x CV). This asks whether two cheap algorithmic changes lower that error,
in the cross-cosmology regime that matters (a few leave-one-cosmology-out folds):

  (1) JOINT  + iteration-noise-weighted MSE   (current baseline)
  (2) JOINT  + CV-weighted MSE                (loss in the inference metric: 1/CV^2)
  (3) PER-LEG + CV-weighted MSE              (one MLP per leg; loud full leg can't
                                              dominate the shared trunk)

Metric: per-leg median held-out RMS/CV (the quantity that sets C_emu). Lower=better.
If nothing moves the quantile legs, it confirms the limiter is data (label noise +
coverage), not architecture -- i.e. the campaign, not algorithm.

Outputs
  data/emulator_tier3/arch_sweep.npz
  plots/emulator_tier3/arch_sweep.png

Usage (GPU node):  python scripts/emulator_arch_sweep.py [--epochs 1500]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu
from inference_astra_valueadd import load, subbox_cov, ASTRA, FULL, QUANT

REPO = Path(__file__).resolve().parents[1]
TEST_COSMOS = ['c130', 'c145', 'c155', 'c165', 'c180']   # mix of good/bad tier3 folds


def fit(Xtr, Ytr, w, Xte, n_ens, epochs, seed0=0):
    """Train n_ens MLPs with explicit per-column weight w; return mean pred (physical)."""
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-12
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-12
    Xz, Yz = (Xtr - xmu) / xsd, (Ytr - ymu) / ysd
    Xtez = (Xte - xmu) / xsd
    rng = np.random.default_rng(seed0)
    va = rng.choice(len(Xz), max(1, len(Xz) // 7), replace=False)
    tr = np.setdiff1d(np.arange(len(Xz)), va)
    preds = []
    for e in range(n_ens):
        m, _ = emu.train_one(Xz[tr], Yz[tr], w, Xz[va], Yz[va], seed=seed0 + e + 1, epochs=epochs)
        preds.append(emu.predict(m, Xtez) * ysd + ymu)
    return np.mean(preds, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--ensemble', type=int, default=1)
    args = ap.parse_args()
    print(f'device={emu.DEVICE}')

    Xa, Ya, Na, ca, s = load('dataset.npz', ASTRA)
    Xb, Yb, Nb, cb, _ = load('dataset_anchor.npz', ASTRA)
    X = np.vstack([Xa, Xb]); Y = np.vstack([Ya, Yb]); Yn = np.vstack([Na, Nb])
    cosmo = np.concatenate([ca, cb]); nb = len(s)
    C_CV, _ = subbox_cov(ASTRA, nb); cv = np.sqrt(np.diag(C_CV))
    legs = FULL + QUANT
    blocks = [(st, slice(i * nb, (i + 1) * nb)) for i, st in enumerate(legs)]
    print(f'{len(Y)} runs; {Y.shape[1]}-D ({len(legs)} legs)')

    configs = ['joint+noise', 'joint+CVw', 'perleg+CVw']
    # accumulate per-leg RMS/CV across folds for each config
    acc = {cfg: {st: [] for st, _ in blocks} for cfg in configs}
    for c in TEST_COSMOS:
        te = cosmo == c; trm = ~te
        ymu_dummy = Y[trm].std(0) + 1e-12                # ysd for weight scaling
        w_noise = (ymu_dummy / (Yn[trm].mean(0) + 1e-12)) ** 2; w_noise /= w_noise.mean()
        w_cv = (ymu_dummy / (cv + 1e-12)) ** 2; w_cv /= w_cv.mean()

        p_jn = fit(X[trm], Y[trm], w_noise, X[te], args.ensemble, args.epochs, seed0=1)
        p_jc = fit(X[trm], Y[trm], w_cv,    X[te], args.ensemble, args.epochs, seed0=2)
        # per-leg: train one MLP per leg block (CV-weighted within the block)
        p_pl = np.zeros_like(p_jn)
        for st, sl in blocks:
            wl = (Y[trm][:, sl].std(0) / (cv[sl] + 1e-12)) ** 2; wl /= wl.mean()
            p_pl[:, sl] = fit(X[trm], Y[trm][:, sl], wl, X[te], args.ensemble, args.epochs, seed0=3)

        for cfg, pred in zip(configs, [p_jn, p_jc, p_pl]):
            for st, sl in blocks:
                rms = np.sqrt(np.mean((pred[:, sl] - Y[te][:, sl]) ** 2, 0))
                acc[cfg][st].append(np.median(rms / cv[sl]))
        print(f'  fold {c} done')

    print(f'\n{"leg":26s} ' + ' '.join(f'{c:>12s}' for c in configs))
    summary = {cfg: {} for cfg in configs}
    for st, _ in blocks:
        row = [np.median(acc[cfg][st]) for cfg in configs]
        for cfg, v in zip(configs, row):
            summary[cfg][st] = v
        print(f'{st:26s} ' + ' '.join(f'{v:12.1f}' for v in row))
    quant_med = {cfg: np.median([summary[cfg][st] for st in QUANT]) for cfg in configs}
    print('\nquantile-leg median RMS/CV: ' + '  '.join(f'{c}={quant_med[c]:.1f}' for c in configs))

    np.savez(REPO / 'data/emulator_tier3/arch_sweep.npz',
             legs=np.array(legs), configs=np.array(configs),
             table=np.array([[summary[cfg][st] for st in legs] for cfg in configs]))

    # bar chart: per-leg RMS/CV per config
    fig, ax = plt.subplots(figsize=(11, 4.5)); x = np.arange(len(legs)); wbar = 0.27
    for i, (cfg, col) in enumerate(zip(configs, ['C7', 'C0', 'C3'])):
        ax.bar(x + (i - 1) * wbar, [summary[cfg][st] for st in legs], wbar, color=col, label=cfg)
    ax.axhline(1, color='k', ls='--', lw=1, label='cosmic variance')
    ax.set_yscale('log'); ax.set_xticks(x)
    ax.set_xticklabels([l.replace('tpcf_', '') for l in legs], rotation=45, fontsize=8)
    ax.set_ylabel('held-out RMS / CV'); ax.legend(fontsize=8)
    ax.set_title('Architecture/loss sweep on the value-add legs (lower=better)\n'
                 'does CV-weighted loss or per-leg modelling lower the quantile-leg C_emu?')
    fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/arch_sweep.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

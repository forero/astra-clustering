#!/usr/bin/env python3
"""
Tier-3 MLP emulator (pilot) on the cached dataset, validated leave-one-cosmology-out.

Reads data/emulator_tier3/{dataset.npz, dataset_anchor.npz} built by
build_emulator_dataset.py.  Trains a SUNBIRD-style feed-forward MLP
  f: theta (20-D = 8 cosmo + 12 HOD) -> selected xi(s) vector
focused by default on the extreme-environment legs (void Q1 / knot Q4, randoms
first then data, monopole + quadrupole).

Split (see notes/tier3_emulator/mlp_emulator_note):
  * 10-fold LEAVE-ONE-COSMOLOGY-OUT: each fold holds out one of the 10 tier3
    cosmologies (50 runs); the other 9 (450) train, with ~15% of their HODs held
    out as an inner validation set for early stopping.
  * The 9 Fisher cosmologies (dataset_anchor.npz), never trained on, are an
    external generalisation anchor: predicted by a final model trained on ALL 10
    tier3 cosmologies.

Loss: NOISE-WEIGHTED MSE in standardised-bin space -- residuals z-scored per bin
are reweighted by (ysd_b / noise_b)^2 so high-S/N bins drive the fit (the planned
upgrade is a full covariance-weighted loss).  An ENSEMBLE of independently-seeded
MLPs gives the emulator uncertainty used in the pull diagnostic.

Outputs
  data/emulator_tier3/loco_results.npz   LOCO + anchor truth/pred/pred_std, labels
  plots/emulator_tier3/*.png             the 7 diagnostics

Usage (GPU compute node, env loaded):
  python scripts/emulator_tier3_mlp.py [--ensemble 5] [--epochs 2000]
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data' / 'emulator_tier3'
PLOT_DIR  = REPO_ROOT / 'plots' / 'emulator_tier3'

# default target legs: void/knot, randoms first then data, autos + full-crosses
PRIMARY_STEMS = [
    'tpcf_rand_q1', 'tpcf_rand_q4',
    'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4',
    'tpcf_data_q1', 'tpcf_data_q4',
    'tpcf_cross_full_data_q1', 'tpcf_cross_full_data_q4',
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(name):
    d = np.load(DATA_DIR / name, allow_pickle=True)
    X = np.hstack([d['X_cosmo'], d['X_hod']]).astype(np.float64)
    return dict(X=X, Y=d['Y'].astype(np.float64), Ynoise=d['Ynoise'].astype(np.float64),
                s=d['s'], stem=d['stem_labels'].astype(str), ell=d['ell_labels'].astype(int),
                cosmo=d['cosmo_id'].astype(str), hod=d['hod_id'].astype(int),
                sigma8=d['sigma8'])


def select_targets(ds, stems):
    """Boolean column mask + ordered list of (stem, ell, slice) blocks."""
    mask = np.isin(ds['stem'], stems)
    stem_sel, ell_sel = ds['stem'][mask], ds['ell'][mask]
    blocks, i = [], 0
    for st in stems:                                    # preserve requested order
        for el in (0, 2):
            n = int(((stem_sel == st) & (ell_sel == el)).sum())
            if n:
                blocks.append((st, el, slice(i, i + n))); i += n
    return mask, blocks


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(512, 512, 256), p=0.05):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one(Xtr, Ytr_z, w, Xva, Yva_z, seed, epochs, lr=1e-3, patience=150):
    """One MLP, noise-weighted MSE in standardised space; early stop on val."""
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], Ytr_z.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=40)
    wt = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Ytr_t = torch.tensor(Ytr_z, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    Yva_t = torch.tensor(Yva_z, dtype=torch.float32, device=DEVICE)

    def wmse(p, t):
        return (wt * (p - t) ** 2).mean()

    best, best_state, wait = np.inf, None, 0
    hist = {'tr': [], 'va': []}
    n, bs = len(Xtr_t), 128
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]
            opt.zero_grad()
            wmse(model(Xtr_t[idx]), Ytr_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = wmse(model(Xtr_t), Ytr_t).item()
            va = wmse(model(Xva_t), Yva_t).item()
        hist['tr'].append(tr); hist['va'].append(va); sched.step(va)
        if va < best - 1e-6:
            best, wait = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


def fit_ensemble(Xtr_raw, Ytr, noise_tr, Xte_raw, n_ens, epochs, val_frac=0.15, seed0=0):
    """Standardise, split inner val, train n_ens MLPs; return (pred_mean, pred_std,
    hists) for Xte in PHYSICAL units, plus the fitted standardisation."""
    rng = np.random.default_rng(seed0)
    # input standardisation (train stats)
    xmu, xsd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-12
    Xtr = (Xtr_raw - xmu) / xsd
    Xte = (Xte_raw - xmu) / xsd
    # output standardisation (train stats) + noise weights
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-12
    Ytr_z = (Ytr - ymu) / ysd
    noise_bar = noise_tr.mean(0)
    w = (ysd / (noise_bar + 1e-12)) ** 2
    w = w / w.mean()                                    # mean-1 weights
    # inner validation split (by row, across the 9 training cosmologies)
    n = len(Xtr)
    va = rng.choice(n, size=max(1, int(round(val_frac * n))), replace=False)
    tr = np.setdiff1d(np.arange(n), va)

    preds, hists = [], []
    for e in range(n_ens):
        model, hist = train_one(Xtr[tr], Ytr_z[tr], w, Xtr[va], Ytr_z[va],
                                seed=seed0 + 100 * e + 1, epochs=epochs)
        preds.append(predict(model, Xte) * ysd + ymu)   # back to physical units
        hists.append(hist)
    preds = np.array(preds)                              # (n_ens, nte, nout)
    return preds.mean(0), preds.std(0), hists


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=2000)
    args = ap.parse_args()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'device={DEVICE}  ensemble={args.ensemble}  epochs={args.epochs}')

    ds = load('dataset.npz')
    mask, blocks = select_targets(ds, PRIMARY_STEMS)
    s = ds['s']; nb = len(s)
    Y = ds['Y'][:, mask]; Ynoise = ds['Ynoise'][:, mask]
    X = ds['X']; cosmo = ds['cosmo']
    noise_bar_all = Ynoise.mean(0)                      # per-bin ASTRA noise floor
    spread_all = Y.std(0)                               # signal spread (all runs)
    print(f'target vector {Y.shape[1]}-D ({len(blocks)} stem×ell blocks × {nb} bins)')

    # ---- leave-one-cosmology-out ----
    cosmos = sorted(set(cosmo))
    Y_pred = np.zeros_like(Y); Y_pred_std = np.zeros_like(Y)
    hist_keep = None
    for k, c in enumerate(cosmos):
        te = cosmo == c
        trm = ~te
        pm, ps, hists = fit_ensemble(X[trm], Y[trm], Ynoise[trm], X[te],
                                     n_ens=args.ensemble, epochs=args.epochs, seed0=k)
        Y_pred[te] = pm; Y_pred_std[te] = ps
        if hist_keep is None:
            hist_keep = hists                          # keep fold-0 curves for plot 1
        rms = np.sqrt(np.mean((pm - Y[te]) ** 2, 0))
        print(f'  LOCO {c}: median RMS/noise={np.median(rms/noise_bar_all):.2f}  '
              f'RMS/spread={np.median(rms/spread_all):.3f}')

    # ---- external anchor: train on all 10 tier3 cosmologies, predict Fisher grid ----
    an = load('dataset_anchor.npz')
    amask, _ = select_targets(an, PRIMARY_STEMS)
    Ya = an['Y'][:, amask]
    ap_mean, ap_std, _ = fit_ensemble(X, Y, Ynoise, an['X'],
                                      n_ens=args.ensemble, epochs=args.epochs, seed0=999)
    arms = np.sqrt(np.mean((ap_mean - Ya) ** 2, 0))
    print(f'ANCHOR (Fisher grid): median RMS/noise={np.median(arms/noise_bar_all):.2f}  '
          f'RMS/spread={np.median(arms/Ya.std(0)):.3f}')

    np.savez(DATA_DIR / 'loco_results.npz',
             s=s, Y=Y, Y_pred=Y_pred, Y_pred_std=Y_pred_std,
             noise=noise_bar_all, spread=spread_all, cosmo=cosmo,
             stem=ds['stem'][mask], ell=ds['ell'][mask],
             Ya=Ya, Ya_pred=ap_mean, Ya_pred_std=ap_std, cosmo_a=an['cosmo'],
             block_stem=np.array([b[0] for b in blocks]),
             block_ell=np.array([b[1] for b in blocks]),
             block_start=np.array([b[2].start for b in blocks]))
    print(f'Saved {DATA_DIR / "loco_results.npz"}')

    make_plots(s, Y, Y_pred, Y_pred_std, noise_bar_all, spread_all, cosmo,
               blocks, hist_keep, Ya, ap_mean, an['cosmo'], cosmos)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def make_plots(s, Y, Yp, Yps, noise, spread, cosmo, blocks, hists,
               Ya, Yap, cosmo_a, cosmos):
    nb = len(s)
    blk = {(st, el): sl for st, el, sl in blocks}

    # 1) learning curves (fold-0 ensemble)
    fig, ax = plt.subplots(figsize=(6, 4))
    for e, h in enumerate(hists):
        ax.plot(h['tr'], 'C0', alpha=0.5, lw=1, label='train' if e == 0 else None)
        ax.plot(h['va'], 'C3', alpha=0.5, lw=1, label='val' if e == 0 else None)
    ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('noise-weighted MSE')
    ax.set_title(f'Learning curves (fold {cosmos[0]} held out, ensemble)')
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOT_DIR / 'mlp_learning_curves.png', dpi=140); plt.close(fig)

    # 2) LOCO prediction vs truth, void/knot, data & random, mono+quad
    _pred_vs_truth(s, Y, Yp, Yps, cosmo, blk, cosmos[0],
                   PLOT_DIR / 'mlp_loco_pred_vs_truth.png',
                   title=f'LOCO held-out {cosmos[0]}: pred vs truth')

    # 2b) PER-COSMOLOGY LOCO quality: how much does accuracy vary by held-out cosmo?
    cnames = cosmos
    med_noise, med_spread = [], []
    for c in cnames:
        te = cosmo == c
        rms = np.sqrt(np.mean((Yp[te] - Y[te]) ** 2, 0))   # per-bin RMS over that cosmo's runs
        med_noise.append(np.median(rms / noise))
        med_spread.append(np.median(rms / spread))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    xpos = np.arange(len(cnames))
    a0.bar(xpos, med_noise, color='C0'); a0.axhline(1, color='grey', lw=0.8)
    a0.set_xticks(xpos); a0.set_xticklabels(cnames, rotation=45, fontsize=8)
    a0.set_ylabel('median RMS / noise floor'); a0.set_title('per held-out cosmology (1 = at noise)')
    a1.bar(xpos, med_spread, color='C3'); a1.axhline(1, color='grey', lw=0.8)
    a1.set_xticks(xpos); a1.set_xticklabels(cnames, rotation=45, fontsize=8)
    a1.set_ylabel('median RMS / signal spread'); a1.set_title('per held-out cosmology (1 = no better than mean)')
    fig.suptitle('LOCO accuracy by held-out cosmology', y=1.02); fig.tight_layout()
    fig.savefig(PLOT_DIR / 'mlp_loco_per_cosmology.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # 3) per-bin error vs yardsticks (noise floor, signal spread)
    legs = [b for b in blocks]
    ncol = 4; nrow = int(np.ceil(len(legs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (st, el, sl) in zip(axs.flat, legs):
        rms = np.sqrt(np.mean((Yp[:, sl] - Y[:, sl]) ** 2, 0))
        ax.plot(s, spread[sl], 'k:', lw=1.4, label='signal spread')
        ax.plot(s, noise[sl], 'C7--', lw=1.4, label='ASTRA noise floor')
        ax.plot(s, rms, 'C0', lw=2, label='LOCO RMS')
        ax.axvline(40, color='C3', lw=0.8, ls=':')
        ax.set_yscale('log'); ax.set_title(f'{st} ℓ{el}', fontsize=8)
        ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    for ax in axs.flat[len(legs):]:
        ax.axis('off')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle('LOCO error vs yardsticks (red line: s=40 small-scale cut)', y=1.0)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'mlp_error_vs_yardstick.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # 4) pull histogram (pred-truth)/ensemble-std
    pull = ((Yp - Y) / (Yps + 1e-12)).ravel()
    pull = pull[np.isfinite(pull)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(pull, bins=80, range=(-6, 6), density=True, alpha=0.7, label='pulls')
    xx = np.linspace(-6, 6, 200)
    ax.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'C3', lw=2, label='N(0,1)')
    ax.set_xlabel(r'$(\hat\xi-\xi)/\sigma_{\rm ens}$')
    ax.set_title(f'Pull (std={pull.std():.2f}); calibrated→1')
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOT_DIR / 'mlp_pull_hist.png', dpi=140); plt.close(fig)

    # 5) accuracy vs scale cut: median(RMS/noise) over bins with s<s_cut
    fig, ax = plt.subplots(figsize=(6, 4))
    rms_all = np.sqrt(np.mean((Yp - Y) ** 2, 0))
    for st, el, sl in blocks:
        if el != 0:                                     # show monopoles for clarity
            continue
        ratio = rms_all[sl] / noise[sl]
        cum = [np.median(ratio[s <= sc]) for sc in s]
        ax.plot(s, cum, marker='o', ms=3, lw=1.3, label=f'{st.replace("tpcf_","")}')
    ax.axhline(1, color='grey', lw=0.8); ax.axvline(40, color='C3', lw=0.8, ls=':')
    ax.set_xlabel(r'$s_{\rm cut}\,[h^{-1}$Mpc]'); ax.set_ylabel('median RMS/noise ($s<s_{\\rm cut}$)')
    ax.set_title('Accuracy vs scale cut (monopoles; <1 = sub-noise)')
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(PLOT_DIR / 'mlp_accuracy_vs_scale.png', dpi=140); plt.close(fig)

    # 6) fractional-error heatmap: (stem×ell) vs s-bin, median |resid|/spread
    rows, labels = [], []
    for st, el, sl in blocks:
        frac = np.median(np.abs(Yp[:, sl] - Y[:, sl]), 0) / (spread[sl] + 1e-12)
        rows.append(frac); labels.append(f'{st.replace("tpcf_","")} ℓ{el}')
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
    im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(s))); ax.set_xticklabels([f'{x:.0f}' for x in s], fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(r'$s\,[h^{-1}$Mpc]'); ax.set_title('median |resid| / signal spread')
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(PLOT_DIR / 'mlp_fracerror_heatmap.png', dpi=140); plt.close(fig)

    # 7) external anchor pred vs truth (one Fisher cosmology, e.g. c000)
    aref = 'c000' if 'c000' in set(cosmo_a) else sorted(set(cosmo_a))[0]
    _pred_vs_truth(s, Ya, Yap, None, cosmo_a, blk, aref,
                   PLOT_DIR / 'mlp_anchor_pred_vs_truth.png',
                   title=f'External anchor {aref} (never trained): pred vs truth')

    print(f'Saved 8 diagnostic figures to {PLOT_DIR}')


def _pred_vs_truth(s, Y, Yp, Yps, cosmo, blk, cref, outpath, title):
    """4×2 grid: rows (data ℓ0, data ℓ2, rand ℓ0, rand ℓ2) × cols (Q1 void, Q4 knot)."""
    rowdefs = [('data', 'tpcf_data_q%d', 0), ('data', 'tpcf_data_q%d', 2),
               ('rand', 'tpcf_rand_q%d', 0), ('rand', 'tpcf_rand_q%d', 2)]
    qs = [(1, 'void'), (4, 'knot')]
    sel = np.where(cosmo == cref)[0]
    i = sel[len(sel) // 2]                               # a representative run
    fig, axs = plt.subplots(len(rowdefs), len(qs),
                            figsize=(4 * len(qs), 2.6 * len(rowdefs)), sharex=True)
    for r, (kind, tmpl, el) in enumerate(rowdefs):
        for cc, (q, qn) in enumerate(qs):
            ax = axs[r, cc]; key = (tmpl % q, el)
            if key not in blk:
                ax.axis('off'); continue
            sl = blk[key]
            ax.plot(s, s**2 * Y[i, sl], 'k-o', ms=3, label='truth')
            ax.plot(s, s**2 * Yp[i, sl], 'C0', lw=2, label='emulator')
            if Yps is not None:
                ax.fill_between(s, s**2 * (Yp[i, sl] - Yps[i, sl]),
                                s**2 * (Yp[i, sl] + Yps[i, sl]), color='C0', alpha=0.25)
            ax.axhline(0, color='grey', lw=0.5)
            if r == 0:
                ax.set_title(f'Q{q} ({qn})')
            if cc == 0:
                ax.set_ylabel(f'{kind} ' + rf'$s^2\xi_{{{el}}}$', fontsize=9)
            if r == len(rowdefs) - 1:
                ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title, y=1.0); fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Most-informative-spatial-scales analysis (no new runs; all cosmologies on disk).

For every leg (stem x multipole) and s-bin, decompose the variation in xi into:
  * COSMOLOGY signal  S_cos(b) = between-cosmology variance of the per-cosmology
    mean vector (how much xi_b responds to cosmology across the 19 cosmologies),
  * CV noise          N_cv(b)  = subbox cosmic-variance variance at the 2 Gpc/h box,
  * HOD nuisance       N_hod(b) = mean within-cosmology variance across HOD draws.
Informative scales = cosmology signal above BOTH the CV noise and the HOD nuisance.

Also the covariance-weighted CUMULATIVE information per leg,
  I(s_cut) = Tr( C(<s_cut)^-1 . Sigma_cos(<s_cut) ),
with C the subbox CV covariance and Sigma_cos the between-cosmology covariance --
a proper (Fisher-discriminant-like) measure of how much cosmology information the
bins up to s_cut carry.  Done per leg (<=30-D blocks) so C is well conditioned
(576 pooled subbox samples).

Uses dataset.npz (10 tier3 cosmologies) + dataset_anchor.npz (9 Fisher) = 19
cosmologies, 950 runs.  Subbox covariance from the 9 Fisher subbox dirs.

Outputs
  data/emulator_tier3/scale_information.npz
  plots/emulator_tier3/scale_information_snr.png        per-(leg,bin) cosmology SNR & HOD-cleanliness
  plots/emulator_tier3/scale_information_cumulative.png cumulative info vs scale cut per leg

Usage (login node OK -- light linear algebra):  python scripts/scale_information.py
"""
import glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emulator_tier3_mlp as emu

REPO = Path(__file__).resolve().parents[1]
PRIORITY = {'tpcf_rand_q1', 'tpcf_rand_q4',
            'tpcf_cross_full_rand_q1', 'tpcf_cross_full_rand_q4'}


def subbox_block(stem, ell):
    """(Nsubbox_pooled, nbins) per-subbox matrix, mean-subtracted per cosmology."""
    tags = [os.path.basename(os.path.dirname(p))
            for p in glob.glob(str(REPO / 'data/*/subbox_multipoles_tpcf_full_data.npz'))]
    cols = []
    for t in tags:
        f = REPO / 'data' / t / f'subbox_multipoles_{stem}.npz'
        if f.is_file():
            x = np.load(f)[f'xi{ell}']
            cols.append(x - x.mean(0))
    return np.vstack(cols)


def main():
    a = emu.load('dataset.npz')
    b = emu.load('dataset_anchor.npz')
    Y = np.vstack([a['Y'], b['Y']])
    cosmo = np.concatenate([a['cosmo'], b['cosmo']])
    stem, ell, s = a['stem'], a['ell'], a['s']
    nb = len(s)
    cosmos = sorted(set(cosmo))
    print(f'{len(Y)} runs over {len(cosmos)} cosmologies; {nb} s-bins')

    # per-cosmology mean vectors -> between-cosmology (cosmology signal) and
    # mean within-cosmology (HOD nuisance) variance, per column
    M = np.array([Y[cosmo == c].mean(0) for c in cosmos])          # (ncos, 510)
    S_cos = M.var(0)                                               # between-cosmology
    N_hod = np.mean([Y[cosmo == c].var(0) for c in cosmos], 0)     # within-cosmology

    # the legs (stem, ell) in column order
    legs = []
    for k in range(Y.shape[1] // nb):
        sl = slice(k * nb, (k + 1) * nb)
        legs.append((str(stem[sl][0]), int(ell[sl][0]), sl))

    # CV variance per column.  Cumulative information per leg uses the DIAGONAL
    # per-bin Fisher  f(b) = S_cos(b)/N_cv(b)  summed over bins <= s_cut: robust and
    # monotonic.  (A full Tr(C^-1 Sigma) is ill-conditioned on subsets of the highly
    # correlated, open-boundary subbox covariance -- it blows up -- so we avoid it.)
    N_cv = np.zeros(Y.shape[1])
    for st, el, sl in legs:
        Xsb = subbox_block(st, el) / np.sqrt(64.0)                # box-volume scatter
        N_cv[sl] = Xsb.var(0)
    order = np.argsort(s)
    cum = {}
    for st, el, sl in legs:
        f = (S_cos[sl] / N_cv[sl])[order]                         # per-bin diagonal Fisher
        cum[(st, el)] = np.cumsum(f)

    snr_cos = np.sqrt(S_cos / N_cv)                               # cosmology signal / CV noise
    clean   = np.sqrt(S_cos / N_hod)                              # cosmology / HOD nuisance

    np.savez(REPO / 'data/emulator_tier3/scale_information.npz',
             s=s, S_cos=S_cos, N_cv=N_cv, N_hod=N_hod,
             leg_stem=np.array([l[0] for l in legs]),
             leg_ell=np.array([l[1] for l in legs]))

    # ---- per-leg summary print ----
    print('\nleg                         most-informative s   50%-info scale   peak SNR')
    for st, el, sl in legs:
        I = cum[(st, el)]
        speak = s[order][np.argmax(np.diff(np.r_[0, I]))]        # bin adding the most info
        s50 = s[order][np.searchsorted(I / I[-1], 0.5)]
        tag = '*' if st in PRIORITY else ' '
        print(f'{tag}{st:26s} l{el}: {speak:6.0f}            {s50:6.0f}        '
              f'{snr_cos[sl].max():6.1f}')

    # ---- figure 1: per-(leg,bin) cosmology SNR and HOD-cleanliness heatmaps ----
    labels = [f'{("* " if st in PRIORITY else "  ")}{st.replace("tpcf_","")} l{el}'
              for st, el, _ in legs]
    for arr, name, ttl in [(snr_cos, 'snr', 'cosmology signal / CV noise  (>1 = above noise)'),
                           (clean, 'clean', 'cosmology signal / HOD nuisance  (>1 = not HOD-degenerate)')]:
        M2 = np.array([arr[sl] for _, _, sl in legs])
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(legs) + 1.5))
        im = ax.imshow(np.log10(M2 + 1e-3), aspect='auto', cmap='viridis')
        ax.set_xticks(range(nb)); ax.set_xticklabels([f'{x:.0f}' for x in s], fontsize=6)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel(r'$s\,[h^{-1}$Mpc]'); ax.set_title(f'log10  {ttl}')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.axvline(np.searchsorted(s, 40) - 0.5, color='C3', lw=1)
        fig.tight_layout()
        p = REPO / f'plots/emulator_tier3/scale_information_{name}.png'
        fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'Saved {p}')

    # ---- figure 2: cumulative information vs scale cut, per leg ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ss = s[order]
    for st, el, sl in legs:
        I = cum[(st, el)]; frac = I / I[-1]
        pri = st in PRIORITY
        ax.plot(ss, frac, lw=2 if pri else 0.8,
                color=('C0' if (pri and el == 0) else 'C2' if pri else '0.8'),
                alpha=1 if pri else 0.5,
                label=(f'{st.replace("tpcf_","")} l{el}' if pri else None))
    ax.axvline(40, color='C3', lw=1, ls=':', label='s=40 (Fisher small-scale cut)')
    ax.axhline(0.5, color='grey', lw=0.6)
    ax.set_xlabel(r'$s_{\rm cut}\,[h^{-1}$Mpc]')
    ax.set_ylabel('cumulative cosmology information fraction  ($s<s_{\\rm cut}$)')
    ax.set_title('Where the cosmology information lives, per leg\n'
                 '(priority void/knot random legs highlighted; grey = other legs)')
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/scale_information_cumulative.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

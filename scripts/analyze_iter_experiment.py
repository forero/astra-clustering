#!/usr/bin/env python3
"""
Analyse the iteration experiment: do 10 ASTRA iterations (vs 3) lower the noise
floor enough to make the noise-limited random-quadrupole vectors learnable?

For the 10 matched c000 draws rerun at 10 iterations (data/fullbox_iter10/), we
compare, per quantile stem and multipole:
  * noise floor = mean over the 10 draws of  xi_std / sqrt(N_iter)  (error on the
    per-draw mean vector) at N_iter = 3 (data/fullbox/) and 10 (data/fullbox_iter10/)
  * signal spread = std across all 50 c000 draws of the mean vector (the HOD-induced
    variation; a stable, iteration-independent reference)
The ratio noise/spread is the learnability proxy: <0.3 signal-rich (emulatable),
0.3-0.7 marginal, >0.7 noise-limited.  We report how the random quadrupoles move.

Outputs
  plots/emulator/iter_experiment_noise_floor.png   (noise/spread iter3 vs iter10)
  plots/emulator/iter_experiment_randq2_ell2.png   (worst stem: iter3 vs iter10 means)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FB3  = REPO / 'data' / 'fullbox'
FB10 = REPO / 'data' / 'fullbox_iter10'
PLOT = REPO / 'plots' / 'emulator'; PLOT.mkdir(parents=True, exist_ok=True)

N_Q = 4
STEMS = [f'tpcf_data_q{q}' for q in range(1, N_Q + 1)] + \
        [f'tpcf_rand_q{q}' for q in range(1, N_Q + 1)]


def noise_on_mean(root, hods, stem, ell):
    """mean over draws of xi_std/sqrt(N_iter), per s-bin."""
    out = []
    for h in hods:
        a = np.load(root / f'c000_hod{h:03d}' / f'fullbox_multipoles_{stem}.npz')
        niter = a[f'xi{ell}_all'].shape[0]
        out.append(a[f'xi{ell}_std'] / np.sqrt(niter))
    return np.mean(out, 0)


def signal_spread(stem, ell):
    """std across ALL 50 c000 draws of the mean vector (iteration-independent)."""
    vecs = [np.load(d / f'fullbox_multipoles_{stem}.npz')[f'xi{ell}']
            for d in sorted(FB3.glob('c000_hod*')) if (d / 'fullbox_info.npz').is_file()]
    return np.std(vecs, 0), len(vecs)


def verdict(r):
    return 'signal-rich' if r < 0.3 else ('marginal' if r < 0.7 else 'NOISE-LIMITED')


def main():
    hods = sorted(int(d.name.split('_hod')[1])
                  for d in FB10.glob('c000_hod*') if (d / 'fullbox_info.npz').is_file())
    s = np.load(FB10 / f'c000_hod{hods[0]:03d}' / 'fullbox_multipoles_tpcf_data_q1.npz')['s']
    print(f'{len(hods)} matched draws at 10 iterations: {hods}\n')
    print(f"{'stem':14s} {'ell':4s} {'r3':>6s} {'r10':>6s} {'noise drop':>11s}  "
          f"{'iter3':>13s} -> {'iter10':>13s}")

    rows = {}
    for stem in STEMS:
        for ell in (0, 2):
            n3  = noise_on_mean(FB3,  hods, stem, ell)
            n10 = noise_on_mean(FB10, hods, stem, ell)
            spr, ndraw = signal_spread(stem, ell)
            r3, r10 = np.median(n3 / spr), np.median(n10 / spr)
            drop = np.median(n3 / n10)
            rows[(stem, ell)] = (r3, r10, drop)
            print(f'{stem:14s} l{ell:<3d} {r3:6.2f} {r10:6.2f} {drop:10.2f}x  '
                  f'{verdict(r3):>13s} -> {verdict(r10):>13s}')
    print(f'\n(noise drop expected ~sqrt(10/3)=1.83x if per-iteration scatter equal; '
          f'spread from {ndraw} draws)')

    # ---- figure 1: noise/spread iter3 vs iter10, ell=2 (where the action is) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, ell in zip(axes, (0, 2)):
        x = np.arange(len(STEMS))
        r3  = [rows[(st, ell)][0] for st in STEMS]
        r10 = [rows[(st, ell)][1] for st in STEMS]
        ax.bar(x - 0.2, r3,  0.4, label='3 iterations',  color='C7')
        ax.bar(x + 0.2, r10, 0.4, label='10 iterations', color='C0')
        ax.axhspan(0.7, ax.get_ylim()[1] if False else 2.0, color='red',   alpha=0.06)
        ax.axhspan(0.3, 0.7, color='orange', alpha=0.06)
        ax.axhspan(0.0, 0.3, color='green',  alpha=0.06)
        ax.axhline(0.3, color='grey', lw=0.6, ls=':'); ax.axhline(0.7, color='grey', lw=0.6, ls=':')
        ax.set_xticks(x); ax.set_xticklabels([st.replace('tpcf_', '') for st in STEMS],
                                             rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('noise / signal spread'); ax.set_title(f'$\\ell={ell}$')
        ax.set_ylim(0, max(0.9, max(r3) * 1.1))
    axes[0].legend(fontsize=9)
    fig.suptitle('Iteration experiment: noise floor vs signal spread '
                 '(green=signal-rich, orange=marginal, red=noise-limited)', y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT / 'iter_experiment_noise_floor.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {PLOT / "iter_experiment_noise_floor.png"}')

    # ---- figure 2: rand_q2 ell=2 mean vectors, iter3 vs iter10, the 10 draws ----
    stem, ell = 'tpcf_rand_q2', 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, root, tag in zip(axes, (FB3, FB10), ('3 iterations', '10 iterations')):
        for h in hods:
            a = np.load(root / f'c000_hod{h:03d}' / f'fullbox_multipoles_{stem}.npz')
            ax.plot(s, s ** 2 * a[f'xi{ell}'], lw=1.2, alpha=0.8)
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_title(tag); ax.set_xlabel(r'$s\,[h^{-1}$Mpc]')
    axes[0].set_ylabel(rf'$s^2\xi_{{{ell}}}$ (rand Q2)')
    fig.suptitle('rand Q2 quadrupole across the 10 draws: more iterations = '
                 'smoother, signal emerges from noise', y=1.0)
    fig.tight_layout()
    fig.savefig(PLOT / 'iter_experiment_randq2_ell2.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {PLOT / "iter_experiment_randq2_ell2.png"}')


if __name__ == '__main__':
    main()

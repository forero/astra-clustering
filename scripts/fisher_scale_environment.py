#!/usr/bin/env python3
"""
Interpretable, data-oriented decomposition of the ASTRA Fisher information,
keeping BOTH data and random legs (the random legs are survey-robust: in real data
they are built from LSS randoms matched to the radial+angular window).

Three views, all noise-aware (derivative-noise bias subtracted, #1) and
HOD-marginalised:

  #1/#2  Scale-cut survival -- FoM3 and per-parameter sigma of the all-stems
         optimal vector vs a minimum-separation cut s > s_cut.  Shows how much of
         the gain survives dropping the systematics-prone small scales, and (from
         the steps) which scales carry the information.
  #1b    Single scale-band FoM3 -- information from each band in isolation
         (small / intermediate / BAO / large).
  #3     Signature map -- per (population x environment x multipole) piece, the
         additive per-parameter sigma improvement over the full auto.  Answers
         "which data/random x void..knot x mono/quad signatures drive each
         parameter."

Output: plots/vector_search/scale_environment.png  +  printed tables.
Run: python scripts/fisher_scale_environment.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fisher_joint as fj

PLOT = Path(__file__).resolve().parents[1] / 'plots' / 'vector_search'
DER = fj.DER_DIR; DATA = fj.DATA_DIR
PARAMS = list(fj.COSMO); ncos = len(PARAMS)
NB = 15
ENVNAME = {1: 'void', 2: 'sheet', 3: 'filam', 4: 'knot'}
N_Q = 4
ALL_STEMS = (['tpcf_full_data'] +
             [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
             [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
             [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
             [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)])
# the all-stems optimal greedy vector (data + random)
OPT = ['tpcf_full_data', 'tpcf_cross_full_rand_q4', 'tpcf_cross_full_rand_q1',
       'tpcf_cross_full_data_q3', 'tpcf_cross_full_data_q4', 'tpcf_data_q1',
       'tpcf_cross_full_rand_q3']

# ---- preload everything once ----
S = np.load(DER / 'derivative_global_lnwb.npz')['s']
DERV = {p: np.load(DER / f'derivative_global_{p}.npz') for p in PARAMS}
DVAR = {p: np.load(DER / f'derivative_global_var_{p}.npz') for p in PARAMS}
GRAD = np.load(DER / 'hod_gradient_global.npz', allow_pickle=True)
SDPR = GRAD['param_std_prior']
TAGS = fj.cosmo_tags()
SUB = {t: {s: np.load(DATA / t / f'subbox_multipoles_{s}.npz') for s in ALL_STEMS}
       for t in TAGS}


def fisher_cov(stems, ells, mask):
    """Noise-aware HOD-marginalised 4x4 cosmology covariance (phys units)."""
    m = mask
    D_cos = np.array([np.concatenate([DERV[p][f'{s}_dxi{e}'][m] for s in stems for e in ells])
                      for p in PARAMS])
    D_hod = np.hstack([GRAD[f'{s}_g{e}'][:, m] for s in stems for e in ells])
    Vc = np.array([np.concatenate([DVAR[p][f'{s}_dxi{e}'][m] for s in stems for e in ells])
                   for p in PARAMS])
    blocks = []
    for t in TAGS:
        cols = [SUB[t][s][f'xi{e}_all'][:, m] for s in stems for e in ells]
        blocks.append(np.hstack(cols))
    Fl = np.vstack([b - b.mean(0) for b in blocks])
    nsamp = Fl.shape[0]; nb = Fl.shape[1]
    C = Fl.T @ Fl / (nsamp - len(TAGS))
    hart = (nsamp - nb - 2) / (nsamp - 1)
    Cinv = hart * fj.VOL_FAC * np.linalg.inv(C)
    D = np.vstack([D_cos, D_hod]); F = D @ Cinv @ D.T
    cd = np.diag(Cinv)
    for i in range(ncos):
        F[i, i] -= float((cd * Vc[i]).sum())
    Fp = F.copy(); Fp[ncos:, ncos:] += np.diag(1.0 / SDPR ** 2)
    cov = np.linalg.inv(Fp)[:ncos, :ncos]
    return fj.to_phys_cov(cov, PARAMS), hart


def fom3(cov):
    return float(np.exp(-0.5 * np.linalg.slogdet(cov[:3, :3])[1]))


def main():
    base_cov, _ = fisher_cov(['tpcf_full_data'], (0, 2), np.ones(NB, bool))
    bsig = np.sqrt(np.diag(base_cov)); base_f = fom3(base_cov)
    g = lambda cov: fom3(cov) / base_f

    # ---- #1/#2 scale-cut survival (all-stems optimal vector) ----
    print('Scale-cut survival of the all-stems optimal vector (FoM3 vs full auto):')
    cuts = [0, 20, 40, 60, 80]; surv = []
    for sc in cuts:
        cov, hart = fisher_cov(OPT, (0, 2), S >= sc)
        sig = np.sqrt(np.diag(cov)); surv.append((sc, g(cov), bsig / sig, hart))
        print(f'  s>{sc:3d}: FoM3={g(cov):6.1f}x  hart={hart:.2f}  '
              + '  '.join(f'{p} {bsig[i]/sig[i]:.1f}x' for i, p in enumerate(PARAMS)))

    # ---- #1b single scale-band FoM3 ----
    bands = [('small\n<40', 0, 40), ('inter\n40-80', 40, 80),
             ('BAO\n80-130', 80, 130), ('large\n>130', 130, 1e9)]
    print('\nSingle scale-band FoM3 (all-stems optimal, band in isolation):')
    band_f = []
    for name, lo, hi in bands:
        cov, _ = fisher_cov(OPT, (0, 2), (S >= lo) & (S < hi))
        band_f.append(g(cov)); print(f'  {name.replace(chr(10)," ")}: FoM3={g(cov):.1f}x')

    # ---- #3 signature map: per (stem, multipole), additive sigma improvement ----
    sig_map = {0: np.zeros((16, ncos)), 2: np.zeros((16, ncos))}
    stems16 = ALL_STEMS[1:]
    for j, s in enumerate(stems16):
        for ell in (0, 2):
            cov, _ = fisher_cov_two('tpcf_full_data', s, ell)
            sig_map[ell][j] = bsig / np.sqrt(np.diag(cov))

    labels = []
    for fam, pre in [('data_q', 'dat-'), ('rand_q', 'rnd-'),
                     ('cross_full_data_q', 'datX-'), ('cross_full_rand_q', 'rndX-')]:
        for q in range(1, 5):
            labels.append(pre + ENVNAME[q])

    print('\n#3 top-3 signatures per parameter (sigma improvement of full+piece):')
    for i, p in enumerate(PARAMS):
        for ell in (0, 2):
            order = np.argsort(-sig_map[ell][:, i])[:3]
            print(f'  {p:5s} l{ell}: ' + ', '.join(
                f'{labels[j]} {sig_map[ell][j,i]:.2f}x' for j in order))

    # ---- figure ----
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)
    a1 = fig.add_subplot(gs[0, 0])
    a1.plot([c[0] for c in surv], [c[1] for c in surv], 'o-', color='C0')
    a1.set_xlabel(r'$s_{\rm cut}$ (use $s>s_{\rm cut}$)'); a1.set_ylabel('FoM3 gain')
    a1.set_title('#2 scale-cut survival (all-stems opt)'); a1.grid(alpha=0.3)
    a2 = fig.add_subplot(gs[0, 1])
    for i, p in enumerate(PARAMS):
        a2.plot([c[0] for c in surv], [c[2][i] for c in surv], 'o-',
                label=fj.COSMO[p][1].replace('\\', ''))
    a2.set_xlabel(r'$s_{\rm cut}$'); a2.set_ylabel(r'$\sigma$ improvement vs 2PCF')
    a2.set_title('#1 per-parameter vs scale cut'); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    a3 = fig.add_subplot(gs[0, 2])
    a3.bar([b[0] for b in bands], band_f, color='C2')
    a3.set_ylabel('single-band FoM3 gain'); a3.set_title('#1b where the information lives')
    for ax, ell, ttl in [(fig.add_subplot(gs[1, 0]), 0, 'monopole'),
                         (fig.add_subplot(gs[1, 1]), 2, 'quadrupole')]:
        im = ax.imshow(sig_map[ell], cmap='viridis', aspect='auto', vmin=1)
        ax.set_yticks(range(16)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xticks(range(ncos)); ax.set_xticklabels([fj.COSMO[p][1].replace('\\','') for p in PARAMS], fontsize=8)
        for b in (3.5, 7.5, 11.5):
            ax.axhline(b, color='w', lw=0.6)
        ax.set_title(f'#3 signature map: {ttl}\n(sigma improvement, full+piece)')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Scale and environment decomposition of the ASTRA Fisher gain '
                 '(noise-aware; data + random legs)', y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT / 'scale_environment.png', dpi=130, bbox_inches='tight')
    plt.close(fig); print(f'\nSaved {PLOT / "scale_environment.png"}')


def fisher_cov_two(s0, s1, ell):
    """full auto (mono+quad) + one extra stem in one multipole."""
    m = np.ones(NB, bool)
    stems = [s0, s1]; ells_list = [(0, 2), (ell,)]
    D_cos, D_hod, Vc, blocks = [], [], [], {t: [] for t in TAGS}
    Dc = {p: [] for p in PARAMS}; Vcd = {p: [] for p in PARAMS}; Dh = []
    for st, ells in zip(stems, ells_list):
        for e in ells:
            for p in PARAMS:
                Dc[p].append(DERV[p][f'{st}_dxi{e}'][m]); Vcd[p].append(DVAR[p][f'{st}_dxi{e}'][m])
            Dh.append(GRAD[f'{st}_g{e}'][:, m])
            for t in TAGS:
                blocks[t].append(SUB[t][st][f'xi{e}_all'][:, m])
    D_cos = np.array([np.concatenate(Dc[p]) for p in PARAMS])
    Vc = np.array([np.concatenate(Vcd[p]) for p in PARAMS])
    D_hod = np.hstack(Dh)
    Fl = np.vstack([np.hstack(blocks[t]) - np.hstack(blocks[t]).mean(0) for t in TAGS])
    nsamp, nb = Fl.shape
    C = Fl.T @ Fl / (nsamp - len(TAGS)); hart = (nsamp - nb - 2) / (nsamp - 1)
    Cinv = hart * fj.VOL_FAC * np.linalg.inv(C)
    D = np.vstack([D_cos, D_hod]); F = D @ Cinv @ D.T; cd = np.diag(Cinv)
    for i in range(ncos):
        F[i, i] -= float((cd * Vc[i]).sum())
    Fp = F.copy(); Fp[ncos:, ncos:] += np.diag(1.0 / SDPR ** 2)
    return fj.to_phys_cov(np.linalg.inv(Fp)[:ncos, :ncos], PARAMS), hart


if __name__ == '__main__':
    main()

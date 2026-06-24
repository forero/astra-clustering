#!/usr/bin/env python3
"""
emulator-based Fisher -- CAMPAIGN forecast (fast; loads emufisher_build.npz).

Forecasts the cosmology FoM as a function of the emulator-error scaling alpha
(C = C_CV + alpha*C_emu), for the full ASTRA vector. alpha=1 is the current emulator;
alpha->0 is the perfect-emulator limit the campaign approaches by driving C_emu down.
This quantifies, in Fisher terms, what the campaign buys -- and how far below CV the
environment-leg C_emu must fall to capture it.

Outputs  plots/emulator_tier3/emufisher_campaign.png

Usage:  python scripts/emufisher_campaign.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emufisher_lib import fisher, COSMO

REPO = Path(__file__).resolve().parents[1]
FITC = [0, 1, 2, 3, 6, 7]
ALPHAS = np.array([1.0, 0.5, 0.3, 0.1, 0.03, 0.01, 0.0])


def main():
    d = np.load(REPO / 'data/emulator_tier3/emufisher_build.npz', allow_pickle=True)
    D, C_CV, C_emu = d['D'], d['C_CV'], d['C_emu']
    nsamp = int(d['nsamp']); hod_prior = d['hod_prior']
    n = D.shape[0]                                     # full vector (all legs)

    fom, sig = [], []
    for a in ALPHAS:
        cond, marg = fisher(D, C_CV, C_emu, FITC, hod_prior, nsamp, alpha=a)
        fom.append(1.0 / np.sqrt(np.linalg.det(marg)))
        sig.append(np.sqrt(np.diag(marg)))
    fom = np.array(fom); sig = np.array(sig)
    cemu_over_cv = np.median(np.sqrt(np.diag(C_emu) / np.diag(C_CV))) * np.sqrt(ALPHAS)

    print('alpha  C_emu/CV(med)  FoM/FoM(a=1)  ' + ' '.join(f'{COSMO[k][:5]}' for k in FITC))
    for i, a in enumerate(ALPHAS):
        print(f'{a:5.2f} {cemu_over_cv[i]:11.2f}  {fom[i]/fom[0]:11.2f}  '
              + ' '.join(f'{sig[i,j]:.2g}' for j in range(len(FITC))))

    rel = fom / fom[0]
    finite = cemu_over_cv > 0                            # alpha=0 (perfect emulator) is off a log x-axis
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cemu_over_cv[finite], rel[finite], 'o-', color='C0', lw=2)
    for i in np.where(finite)[0]:
        ax.annotate(f'a={ALPHAS[i]:g}', (cemu_over_cv[i], rel[i]), fontsize=7,
                    textcoords='offset points', xytext=(4, 4))
    ax.axvline(1, color='C3', ls='--', lw=1, label='$C_{emu}=C_{CV}$')
    ax.axhline(rel[~finite][0], color='0.5', ls=':', lw=1,
               label=f'perfect emulator ($a\\to0$): {rel[~finite][0]:.0f}$\\times$')
    ax.set_xlabel('median environment-leg $C_{emu}/C_{CV}$ (= sqrt(alpha) x current)')
    ax.set_ylabel('cosmology FoM / current'); ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Campaign forecast: Fisher FoM vs emulator error\n'
                 '(full ASTRA vector, HOD-marginalised; left = better emulator)')
    ax.legend(); fig.tight_layout()
    p = REPO / 'plots/emulator_tier3/emufisher_campaign.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'Saved {p}')


if __name__ == '__main__':
    main()

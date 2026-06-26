import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/pscratch/sd/f/forero/astra-clustering'
d = np.load(f'{REPO}/data/emulator_tier3/emufisher_build.npz', allow_pickle=True)
C_CV, C_emu = d['C_CV'], d['C_emu']
stems, ells, s = d['leg_stem'], d['leg_ell'], d['s']
nb = len(s)                      # 15 bins per leg
nleg = len(stems)                # 18 legs

dcv, demu = np.diag(C_CV), np.diag(C_emu)
ratio_var = demu / dcv           # per-bin C_emu/C_CV (variance units)

# per-leg median over bins (variance ratio); sigma ratio = sqrt
leg_med, leg_sig, labels, ell_arr = [], [], [], []
for i in range(nleg):
    sl = slice(i*nb, (i+1)*nb)
    r = ratio_var[sl]
    leg_med.append(np.median(r))
    leg_sig.append(np.median(np.sqrt(r)))
    labels.append(f"{stems[i]} l{ells[i]}")
    ell_arr.append(int(ells[i]))
leg_med = np.array(leg_med); leg_sig = np.array(leg_sig); ell_arr = np.array(ell_arr)

order = np.argsort(leg_med)
labels_s = [labels[k] for k in order]
colors = ['#1f77b4' if ell_arr[k] == 0 else '#d62728' for k in order]

fig, ax = plt.subplots(2, 1, figsize=(11, 11))

# --- panel 1: per-leg C_emu/C_CV (variance), sorted ---
y = np.arange(nleg)
ax[0].barh(y, leg_med[order], color=colors)
ax[0].set_yticks(y); ax[0].set_yticklabels(labels_s, fontsize=8)
ax[0].axvline(1.0, color='k', lw=1.5, ls='--', label='inference-grade (C_emu = C_CV)')
med_all = np.median(ratio_var)
ax[0].axvline(med_all, color='grey', lw=1.5, ls=':', label=f'all-leg median = {med_all:.1f}')
ax[0].set_xscale('log')
ax[0].set_xlabel('C_emu / C_CV  (variance; <1 = emulator error below cosmic variance)')
ax[0].set_title('Emulator accuracy per leg (52-cosmology cache, 1840 runs)\n'
                'blue = monopole (l=0), red = quadrupole (l=2)')
ax[0].legend(loc='lower right', fontsize=9)
ax[0].grid(axis='x', alpha=0.3)

# --- panel 2: sigma-ratio vs scale s, monopole vs quadrupole bands ---
for i in range(nleg):
    sl = slice(i*nb, (i+1)*nb)
    sig = np.sqrt(ratio_var[sl])
    c = '#1f77b4' if ell_arr[i] == 0 else '#d62728'
    ax[1].plot(s, sig, color=c, alpha=0.5, lw=1)
ax[1].axhline(1.0, color='k', ls='--', lw=1.5, label='inference-grade')
ax[1].set_xlabel('s  [Mpc/h]')
ax[1].set_ylabel('emulator error / cosmic variance  (sigma)')
ax[1].set_title('Per-leg emulator sigma-ratio vs scale  '
                '(blue l=0, red l=2)')
ax[1].set_yscale('log')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

plt.tight_layout()
out = f'{REPO}/plots/emulator_tier3/emufisher_cemu_per_leg.png'
plt.savefig(out, dpi=120)
print('Saved', out)

# text summary
mono = ell_arr == 0; quad = ell_arr == 2
print(f"all-leg median C_emu/C_CV (var) = {med_all:.2f}  (sigma {np.sqrt(med_all):.2f})")
print(f"monopole legs: median var-ratio {np.median(leg_med[mono]):.2f}  (sigma {np.median(leg_sig[mono]):.2f})")
print(f"quadrupole legs: median var-ratio {np.median(leg_med[quad]):.2f}  (sigma {np.median(leg_sig[quad]):.2f})")
print("\nbest (most emulable) legs:")
for k in order[:5]:
    print(f"  {labels[k]:28s} var {leg_med[k]:6.2f}  sigma {leg_sig[k]:.2f}")
print("worst legs:")
for k in order[-5:]:
    print(f"  {labels[k]:28s} var {leg_med[k]:6.2f}  sigma {leg_sig[k]:.2f}")

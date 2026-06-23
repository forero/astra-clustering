#!/usr/bin/env python3
"""
Build the cached MLP-emulator dataset(s) once, so retraining never re-globs the
~1000 run directories.

Two archives are written under data/emulator_tier3/:
  dataset.npz         <- the Tier-3 training block (data/fullbox_tier3/,
                          cosmologies c130..c180, 10 x 50 runs)
  dataset_anchor.npz  <- the Fisher grid (data/fullbox/, c000,c100-c105,c112,c113),
                          NOT in the training block; an external generalisation anchor.

Each archive stores the FULL stem set (all 17 stems x 2 multipoles x 15 bins =
510-D) regardless of which legs a given model trains on -- target selection is a
column mask at train time, no disk re-read.

Inputs per row:
  X_cosmo (N,8)  : omega_b, omega_cdm, h, n_s, alpha_s, N_ur, w0_fld, wa_fld
                   (the parameters that vary across c130..c181; A_s and
                    omega_ncdm are fixed, sigma8 is kept as a label not an input)
  X_hod   (N,12) : the 12 yuan23 HOD parameters

Targets per row:
  Y       (N,510): concatenated [xi0, xi2] over the 17 stems
  Ynoise  (N,510): the matching per-bin ASTRA-iteration scatter (xi*_std)

Bookkeeping:
  s (15,), cosmo_param_names (8,), hod_param_names (12,),
  stem_labels (510,), ell_labels (510,), bin_index (510,),
  cosmo_id (N,), hod_id (N,), sigma8 (N,)

Usage (any node, env loaded):  python scripts/build_emulator_dataset.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / 'data'
OUT_DIR   = DATA_DIR / 'emulator_tier3'
COSMO_CSV = DATA_DIR / 'abacus_cosmologies_params.csv'

N_Q = 4
STEMS = (
    ['tpcf_full_data'] +
    [f'tpcf_data_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_rand_q{q}'            for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_data_q{q}' for q in range(1, N_Q + 1)] +
    [f'tpcf_cross_full_rand_q{q}' for q in range(1, N_Q + 1)]
)
ELLS = (0, 2)
# the 8 cosmology parameters that vary across the c130..c181 block
COSMO_PARAMS = ['omega_b', 'omega_cdm', 'h', 'n_s', 'alpha_s',
                'N_ur', 'w0_fld', 'wa_fld']

# (subdir under data/, hod-param csv directory, glob prefix) per archive
SOURCES = {
    'dataset':        ('fullbox_tier3', DATA_DIR / 'tier3_pilot',   None),
    'dataset_anchor': ('fullbox',       DATA_DIR / 'hod_ensemble',  None),
}


def build_blocks(s):
    """Per-(stem, ell) labelling parallel to the flattened Y/Ynoise columns."""
    stem_labels, ell_labels, bin_index = [], [], []
    for stem in STEMS:
        for ell in ELLS:
            stem_labels += [stem] * len(s)
            ell_labels  += [ell] * len(s)
            bin_index   += list(range(len(s)))
    return (np.array(stem_labels), np.array(ell_labels), np.array(bin_index))


def load_run(rundir):
    """Return (Y, Ynoise, s) for one run, or (None, None, None) if incomplete."""
    if not (rundir / 'fullbox_info.npz').is_file():
        return None, None, None
    vec, noise, s = [], [], None
    for stem in STEMS:
        f = rundir / f'fullbox_multipoles_{stem}.npz'
        if not f.is_file():
            return None, None, None
        a = np.load(f)
        if s is None:
            s = a['s']
        for ell in ELLS:
            vec.append(a[f'xi{ell}'])
            noise.append(a[f'xi{ell}_std'])
    return np.concatenate(vec), np.concatenate(noise), s


def build_archive(name, subdir, hod_csv_dir, cosmo_df):
    fb = DATA_DIR / subdir
    rundirs = sorted(fb.glob('c*_hod*'))
    # cache the per-cosmology HOD param tables on demand
    hod_tables = {}
    Xc, Xh, Y, Yn = [], [], [], []
    cosmo_id, hod_id, sig8 = [], [], []
    s_ref = None
    skipped = 0
    for d in rundirs:
        cosmo, hod = d.name.split('_hod')
        hod = int(hod)
        if cosmo not in cosmo_df.index:
            skipped += 1; continue
        if cosmo not in hod_tables:
            csv = hod_csv_dir / f'hod_params_{cosmo}.csv'
            hod_tables[cosmo] = (pd.read_csv(csv).set_index('hod')
                                 if csv.is_file() else None)
        ht = hod_tables[cosmo]
        if ht is None or hod not in ht.index:
            skipped += 1; continue
        y, yn, s = load_run(d)
        if y is None:
            skipped += 1; continue
        if s_ref is None:
            s_ref = s
        Xc.append(cosmo_df.loc[cosmo, COSMO_PARAMS].values.astype(float))
        Xh.append(ht.loc[hod].values.astype(float))
        Y.append(y); Yn.append(yn)
        cosmo_id.append(cosmo); hod_id.append(hod)
        sig8.append(float(cosmo_df.loc[cosmo, 'sigma8_m']))

    Xc, Xh = np.array(Xc), np.array(Xh)
    Y, Yn = np.array(Y), np.array(Yn)
    hod_names = list(hod_tables[cosmo_id[0]].columns)
    stem_labels, ell_labels, bin_index = build_blocks(s_ref)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{name}.npz'
    np.savez(out,
             X_cosmo=Xc, X_hod=Xh, Y=Y, Ynoise=Yn, s=s_ref,
             cosmo_param_names=np.array(COSMO_PARAMS),
             hod_param_names=np.array(hod_names),
             stem_labels=stem_labels, ell_labels=ell_labels, bin_index=bin_index,
             cosmo_id=np.array(cosmo_id), hod_id=np.array(hod_id),
             sigma8=np.array(sig8))
    ncos = len(set(cosmo_id))
    print(f'[{name}] {len(Y)} runs, {ncos} cosmologies, '
          f'input {Xc.shape[1]}+{Xh.shape[1]}-D, output {Y.shape[1]}-D; '
          f'skipped {skipped}. -> {out}')
    # per-cosmology counts
    for c in sorted(set(cosmo_id)):
        print(f'    {c}: {cosmo_id.count(c)}')


def main():
    cosmo_df = pd.read_csv(COSMO_CSV, index_col=0)
    for name, (subdir, hod_dir, _) in SOURCES.items():
        build_archive(name, subdir, hod_dir, cosmo_df)


if __name__ == '__main__':
    main()

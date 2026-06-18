#!/usr/bin/env python3
"""
Select a spanning set of HOD draws *per cosmology* for the global response model.

Generalises select_hod_calibration.py from c000-only to all nine Fisher
cosmologies.  Each cosmology has its own ~500 yuan23 HOD draws (different
parameter values at each hodNNN index); we run a maximin (farthest-point)
subset of N per cosmology so the global ξ(θ_cosmo, θ_HOD) regression
(compute_response_global.py) sees well-spread HOD coverage in every cosmology.

The 12 varying HOD parameters are determined once from c000 and applied to every
cosmology (same yuan23 prior), so the regression design is consistent across
cosmologies.  Each cosmology's maximin search is seeded on its existing Fisher
pick (hod484 for c000, hod179 for c100, ...) so the runs already on disk are
reused rather than orphaned.

Outputs (data/hod_ensemble/):
  hod_params_{cosmo}.csv     all draws of that cosmology: index + every varying param
  hod_selection_{cosmo}.txt  the chosen HOD indices, one per line (incl. the seed)

Usage (login node OK — header reads only):
  python scripts/select_hod_ensemble.py [N]          # default N = 50
"""

import sys
from pathlib import Path

import numpy as np
import fitsio

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / 'data' / 'hod_ensemble'
HOD_BASE  = Path('/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior')

# existing Fisher pick per cosmology -> seeds the maximin search so the runs
# already on disk are reused (see CLAUDE.md, "Cosmologies and chosen HOD catalogs").
SEED_PICK = {
    'c000': 484,
    'c100': 179, 'c101': 152,
    'c102': 556, 'c103': 861,
    'c104': 498, 'c105': 589,
    'c112': 507, 'c113': 483,
}

# candidate HOD parameters in the table-HDU header (yuan23 prior)
CAND = ['LOGM_CUT', 'LOGM1', 'SIGMA', 'ALPHA', 'KAPPA', 'ALPHA_C', 'ALPHA_S',
        'S', 'S_V', 'S_P', 'S_R', 'ACENT', 'ASAT', 'BCENT', 'BSAT']


def read_params(cosmo, cand):
    """Return (hod indices, param matrix over `cand`) for one cosmology."""
    files = sorted((HOD_BASE / f'{cosmo}_ph000' / 'seed0').glob('hod*.fits'))
    idx   = np.array([int(f.stem[3:]) for f in files])
    rows  = [[float(fitsio.read_header(str(f), ext=1)[k]) for k in cand]
             for f in files]
    return idx, np.array(rows)


def varying_params():
    """The HOD parameters that actually vary across c000 draws (used for all)."""
    idx, P = read_params('c000', CAND)
    keep   = [k for k, c in zip(CAND, range(P.shape[1])) if P[:, c].std() > 0]
    print(f'Varying HOD parameters ({len(keep)}): {", ".join(keep)}')
    return keep


def maximin(Z, start, n):
    """Farthest-point indices into Z (standardised), seeded at `start`."""
    chosen = [start]
    dmin   = np.linalg.norm(Z - Z[start], axis=1)
    while len(chosen) < min(n, len(Z)):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[nxt], axis=1))
    return chosen


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = varying_params()
    for cosmo, seed_hod in SEED_PICK.items():
        idx, P = read_params(cosmo, names)
        print(f'{cosmo}: {len(idx)} HOD catalogs')

        csv = OUT_DIR / f'hod_params_{cosmo}.csv'
        with open(csv, 'w') as fh:
            fh.write('hod,' + ','.join(names) + '\n')
            for i, row in zip(idx, P):
                fh.write(f'{i},' + ','.join(f'{v:.10g}' for v in row) + '\n')

        Z = (P - P.mean(0)) / P.std(0)
        if seed_hod not in idx:
            sys.exit(f'{cosmo}: seed hod{seed_hod} not among available draws')
        start  = int(np.where(idx == seed_hod)[0][0])
        chosen = maximin(Z, start, N)
        sel    = sorted(int(idx[c]) for c in chosen)

        txt = OUT_DIR / f'hod_selection_{cosmo}.txt'
        txt.write_text('\n'.join(str(i) for i in sel) + '\n')
        print(f'  selected {len(sel)} (maximin, seed hod{seed_hod}); '
              f'wrote {csv.name}, {txt.name}')


if __name__ == '__main__':
    main()

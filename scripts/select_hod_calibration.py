#!/usr/bin/env python3
"""
Select a spanning set of c000 HOD draws for the Tier-1 ∂ξ/∂θ_HOD calibration.

The Fisher ± pairs are not HOD-matched, so each cosmology derivative carries a
smooth HOD-contamination term.  To subtract it we measure the HOD response
∂ξ/∂θ_HOD by running the full-box pipeline on extra *existing* c000 HOD draws
(no new HOD generation) and regressing ξ on the HOD parameters.

This script reads the HOD parameters from all available c000 catalog headers,
keeps the parameters that actually vary, standardises them, and picks N draws by
**maximin (farthest-point) sampling** seeded on the Fisher fiducial (hod484) so
the regression design is well spread across the prior with few points.

Outputs (data/hod_calibration/):
  hod_params_c000.csv     all available c000 draws: index + every varying param
  hod_selection_c000.txt  the chosen HOD indices, one per line (incl. 484)

Usage (login node OK — header reads only):
  python scripts/select_hod_calibration.py [N]      # default N = 50
"""

import sys
from pathlib import Path

import numpy as np
import fitsio

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / 'data' / 'hod_calibration'
HOD_DIR   = Path('/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
                 '/c000_ph000/seed0')
FIDUCIAL  = 484

# candidate HOD parameters in the table-HDU header (yuan23 prior)
CAND = ['LOGM_CUT', 'LOGM1', 'SIGMA', 'ALPHA', 'KAPPA', 'ALPHA_C', 'ALPHA_S',
        'S', 'S_V', 'S_P', 'S_R', 'ACENT', 'ASAT', 'BCENT', 'BSAT']


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(HOD_DIR.glob('hod*.fits'))
    idx   = np.array([int(f.stem[3:]) for f in files])
    print(f'{len(files)} c000 HOD catalogs found')

    rows = []
    for f in files:
        h = fitsio.read_header(str(f), ext=1)
        rows.append([float(h[k]) for k in CAND])
    P = np.array(rows)

    # keep only parameters that actually vary across the draws
    varying = [k for k, c in zip(CAND, range(P.shape[1])) if P[:, c].std() > 0]
    keep    = [CAND.index(k) for k in varying]
    P       = P[:, keep]
    print(f'Varying HOD parameters ({len(varying)}): {", ".join(varying)}')

    # write the full parameter table (all draws) for the regression step
    csv = OUT_DIR / 'hod_params_c000.csv'
    with open(csv, 'w') as fh:
        fh.write('hod,' + ','.join(varying) + '\n')
        for i, row in zip(idx, P):
            fh.write(f'{i},' + ','.join(f'{v:.10g}' for v in row) + '\n')
    print(f'Saved {csv}')

    # standardise and maximin-select, seeded on the fiducial
    Z = (P - P.mean(0)) / P.std(0)
    if FIDUCIAL not in idx:
        sys.exit(f'fiducial hod{FIDUCIAL} not among available c000 draws')
    start = int(np.where(idx == FIDUCIAL)[0][0])

    chosen = [start]
    dmin   = np.linalg.norm(Z - Z[start], axis=1)
    while len(chosen) < min(N, len(idx)):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[nxt], axis=1))

    sel = sorted(int(idx[c]) for c in chosen)
    txt = OUT_DIR / 'hod_selection_c000.txt'
    txt.write_text('\n'.join(str(i) for i in sel) + '\n')
    print(f'Selected {len(sel)} draws (maximin, seeded on hod{FIDUCIAL}):')
    print('  ' + ' '.join(str(i) for i in sel))
    print(f'Saved {txt}')


if __name__ == '__main__':
    main()

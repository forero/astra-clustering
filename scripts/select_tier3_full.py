#!/usr/bin/env python3
"""
Tier-3 FULL campaign: maximin HOD selection for the whole AbacusSummit emulator
block c130-c181 (52 cosmologies), N draws each.

Same machinery as select_tier3_pilot.py (reuses select_hod_ensemble), extended to
all 52 cosmologies.  maximin (greedy farthest-point) is PREFIX-STABLE: with the
same centre seed, the first 50 of an N=100 selection equal the pilot's N=50
selection, so the 500 pilot runs already on disk are a subset and get skipped by
the launcher.  Writes to data/tier3_pilot/ (same dir the launcher and dataset
builder read), overwriting the pilot's 10 selection files with their supersets.

Usage (login node OK -- header reads only):  python scripts/select_tier3_full.py [N]
"""
import sys
from pathlib import Path
import numpy as np
import select_hod_ensemble as she        # reuse read_params, varying_params, maximin

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'data' / 'tier3_pilot'
FULL_COSMOS = [f'c{n}' for n in range(130, 182)]        # c130..c181


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    OUT.mkdir(parents=True, exist_ok=True)
    names = she.varying_params()
    for cosmo in FULL_COSMOS:
        idx, P = she.read_params(cosmo, names)
        with open(OUT / f'hod_params_{cosmo}.csv', 'w') as fh:
            fh.write('hod,' + ','.join(names) + '\n')
            for i, row in zip(idx, P):
                fh.write(f'{i},' + ','.join(f'{v:.10g}' for v in row) + '\n')
        Z = (P - P.mean(0)) / P.std(0)
        start = int(np.argmin(np.linalg.norm(Z, axis=1)))        # nearest the centre
        chosen = she.maximin(Z, start, N)
        sel = sorted(int(idx[c]) for c in chosen)
        (OUT / f'hod_selection_{cosmo}.txt').write_text('\n'.join(map(str, sel)) + '\n')
        print(f'{cosmo}: {len(idx)} draws -> selected {len(sel)} (seed hod{idx[start]})')


if __name__ == '__main__':
    main()

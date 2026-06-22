#!/usr/bin/env python3
"""
Tier-3 emulator pilot: select maximin HOD draws for ~10 space-filling cosmologies
from the AbacusSummit emulator block c130-c181 (see notes/tier3_emulator/).

Reuses the maximin machinery of select_hod_ensemble.py, but for cosmologies that
have no existing Fisher pick -- so each cosmology's farthest-point search is seeded
on the draw nearest the prior centre (the most central HOD), which is stable.

Writes, per cosmology, to data/tier3_pilot/:
  hod_params_{cosmo}.csv     all draws: index + every varying yuan23 param
  hod_selection_{cosmo}.txt  the chosen HOD indices (one per line)

Usage (login node OK -- header reads only):  python scripts/select_tier3_pilot.py [N]
"""
import sys
from pathlib import Path
import numpy as np
import select_hod_ensemble as she        # reuse read_params, varying_params, maximin

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'data' / 'tier3_pilot'
PILOT_COSMOS = ['c130', 'c135', 'c140', 'c145', 'c150',
                'c155', 'c160', 'c165', 'c170', 'c180']


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    OUT.mkdir(parents=True, exist_ok=True)
    names = she.varying_params()
    for cosmo in PILOT_COSMOS:
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
        print(f'{cosmo}: {len(idx)} draws -> selected {len(sel)} '
              f'(maximin, seed hod{idx[start]})')


if __name__ == '__main__':
    main()

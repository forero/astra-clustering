#!/usr/bin/env python3
"""
Select N_extra ADDITIONAL maximin HOD draws for one cosmology, extending an
existing selection (farthest-point search seeded with the already-chosen set, so
the existing runs are kept and the new draws fill the remaining gaps).

Used to test whether more HODs lower the within-cosmology emulator floor: e.g.
c000 already has 50 runs in data/fullbox/; this picks 50 more.

Writes data/hod_ensemble/hod_selection_{cosmo}_extra{N}.txt (the NEW indices only).

Usage:  python scripts/select_extra_hods.py <cosmo> <n_extra>
"""
import sys
from pathlib import Path
import numpy as np
import select_hod_ensemble as she

REPO = Path(__file__).resolve().parents[1]
ENS = REPO / 'data' / 'hod_ensemble'


def main():
    cosmo = sys.argv[1]
    n_extra = int(sys.argv[2])
    names = she.varying_params()
    idx, P = she.read_params(cosmo, names)
    Z = (P - P.mean(0)) / P.std(0)
    pos = {int(h): k for k, h in enumerate(idx)}          # hod index -> row in Z

    existing = [int(x) for x in
                (ENS / f'hod_selection_{cosmo}.txt').read_text().split()]
    chosen_rows = [pos[h] for h in existing]

    # farthest-point, seeded with the whole existing set
    dmin = np.min([np.linalg.norm(Z - Z[r], axis=1) for r in chosen_rows], axis=0)
    new_rows = []
    while len(new_rows) < n_extra:
        nxt = int(np.argmax(dmin))
        new_rows.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[nxt], axis=1))

    new = sorted(int(idx[r]) for r in new_rows)
    out = ENS / f'hod_selection_{cosmo}_extra{n_extra}.txt'
    out.write_text('\n'.join(map(str, new)) + '\n')
    overlap = set(new) & set(existing)
    print(f'{cosmo}: {len(idx)} draws; {len(existing)} existing + {len(new)} new '
          f'(overlap with existing: {len(overlap)})')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

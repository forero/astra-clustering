#!/usr/bin/env python3
"""
Minimal-coverage selection: N maximin HOD draws per cosmology for the WHOLE
c130-c181 block, reading the cached param CSVs (data/tier3_pilot/hod_params_*.csv;
fast, no FITS).  Same centre-seed maximin as select_tier3_full, so the N draws are
a prefix of the larger selections -> the pilot cosmologies' runs (maximin-50) are
reused.  Writes hod_selection_{cosmo}_min{N}.txt (does NOT clobber the 100-draw
full-campaign selections).

Usage:  python scripts/select_tier3_min.py [N=20]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import select_hod_ensemble as she

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'data' / 'tier3_pilot'
COSMOS = [f'c{n}' for n in range(130, 182)]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    names = she.varying_params()
    for cosmo in COSMOS:
        df = pd.read_csv(OUT / f'hod_params_{cosmo}.csv').set_index('hod')
        idx = df.index.values
        P = df[names].values.astype(float)
        Z = (P - P.mean(0)) / P.std(0)
        start = int(np.argmin(np.linalg.norm(Z, axis=1)))
        chosen = she.maximin(Z, start, N)
        sel = sorted(int(idx[c]) for c in chosen)
        (OUT / f'hod_selection_{cosmo}_min{N}.txt').write_text('\n'.join(map(str, sel)) + '\n')
    print(f'Wrote {len(COSMOS)} hod_selection_*_min{N}.txt (N={N})')


if __name__ == '__main__':
    main()

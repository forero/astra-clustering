# astra-clustering

Minimal self-contained pipeline to run the ASTRA cosmic-web classification
on a simulation box and measure the two-point correlation function per
environment quantile.

---

## What is ASTRA?

ASTRA classifies each galaxy by its local density environment using a
**Delaunay triangulation** over the combined data + random catalog:

```
r = (ndata_neighbours - nrand_neighbours) / (ndata_neighbours + nrand_neighbours)
```

| Type     | r range         |
|----------|-----------------|
| Void     | −1 ≤ r ≤ −0.9  |
| Sheet    | −0.9 < r ≤ 0   |
| Filament |  0 < r ≤ 0.9   |
| Knot     |  0.9 < r ≤ 1   |

Galaxies are then binned into **quantiles of r** (Q1 = most underdense,
Q4 = most overdense).  The same bin edges are applied to the randoms.

---

## Repo layout

```
astra-clustering/
├── astra.py          ← ASTRA algorithm module
├── scripts/
│   ├── pipeline_single_box.py   ← end-to-end pipeline (load → classify → split → 2PCF)
│   └── plot.py       ← produces 6 figures
├── data/             ← pipeline output (.npy / .npz files)
├── plots/            ← figure output (.png files)
├── CLAUDE.md
└── README.md
```

---

## Environment

Always use the **cosmodesi** environment — it provides pycorr, fitsio,
scipy, and pandas:

```bash
unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```

Available tags: `main` (rolling), `2026_02`, `2025_12`, `2025_05`, `dr1`.

---

## Running

### 1. Get a compute node

```bash
salloc -N 1 -C cpu -q interactive -t 60:00 -A desi -c 8 --mem=32G
```

Increase `-t` if needed — with N_ASTRA_ITERATIONS=10 the full run takes ~60–90 min
(each iteration: ~5–10 min Delaunay + ~10 min 2PCF for 16 correlation functions).

### 2. Load environment and run

```bash
unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /pscratch/sd/f/forero/astra-clustering
srun -n 1 -c 8 python scripts/pipeline_single_box.py
python scripts/plot.py        # no srun needed — lightweight
```

---

## Pipeline steps (`pipeline_single_box.py`)

1. **Load** `c000_ph000/seed0/hod000.fits` from the EMC HOD catalog  
   Path: `/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/`

2. **Apply RSD** (los=z): use `X_PERP`/`Y_PERP`/`Z_RSD`, divided by
   `Q_PERP`/`Q_PAR` from the FITS header (Alcock-Paczynski correction)

3. **Cut** a 500 Mpc/h cube centred at the origin (~62k galaxies)

4. **Geometry randoms** (`N_RAND_GEOM=5×`, `SEED_GEOM=SEED+1000`): generated
   **once before the loop**, used only for the Landy-Szalay 2PCF estimator.  
   The subbox has **open boundaries** — `boxsize=` periodic BC must not be used.

5. **Full-data auto-correlation** (once, outside the loop): deterministic,
   same each iteration.

6. **Repeat N_ASTRA_ITERATIONS times** (seed = SEED + iteration):  
   a. Generate new ASTRA randoms (`N_RAND=1×`, uniform, bounds from data min/max)  
   b. Build combined data+random dataframe  
   c. Run ASTRA: Delaunay → neighbour counts → `r` for every point  
   d. Split into quantiles: data and randoms each split independently with
      `pd.qcut` so both populations have equal counts per quantile  
   e. Compute 2PCF (ℓ=0, ℓ=2) for: data quantile autos, random quantile autos,
      full data × data quantile cross, full data × random quantile cross  
   f. Accumulate multipole arrays across iterations  
   g. On the **last** iteration: save per-quantile catalogs + pycorr objects

7. **Save averaged multipoles**: mean and std (ddof=1) over all iterations.  
   Keys in each `.npz`: `s`, `xi0`, `xi0_std`, `xi2`, `xi2_std`

Output directory: `data/` (inside the repo)

### Two random catalogs — why they are different

| Catalog | Size | Seed | Purpose |
|---------|------|------|---------|
| ASTRA randoms | 1× data | `SEED + iteration` | Enter Delaunay; split by environment; regenerated each iteration |
| Geometry randoms | 5× data | `SEED + 1000` | Uniform; correct for open-boundary geometry in LS estimator; fixed across iterations |

---

## Plots (`plot.py`)

Output: `plots/` (inside the repo)

| File | Content |
|------|---------|
| `data_monopole_per_quantile.png` | s²ξ₀(s) for data Q1–Q4 |
| `data_quadrupole_per_quantile.png` | s²ξ₂(s) for data Q1–Q4 |
| `data_multipoles_all_quantiles.png` | Monopole + quadrupole side by side |
| `rand_monopole_per_quantile.png` | s²ξ₀(s) for random Q1–Q4 |
| `rand_quadrupole_per_quantile.png` | s²ξ₂(s) for random Q1–Q4 |
| `data_vs_rand_monopole.png` | Data vs randoms monopole, 2×2 panel per quantile |
| `full_data_autocorr.png` | Full sample auto-correlation monopole + quadrupole |
| `cross_full_data_quantiles.png` | Cross-correlation full data × each data quantile |
| `cross_full_rand_quantiles.png` | Cross-correlation full data × each random quantile |

The random quantile 2PCF is non-trivial (not ≈ 0) because ASTRA randoms
trace the same cosmic-web environments as the data — overdense random
quantiles cluster just as overdense galaxies do.

---

## Key parameters (top of `pipeline_single_box.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `BOXSIZE` | 500 Mpc/h | Subbox side length |
| `LOS` | `'z'` | Line-of-sight axis |
| `N_Q` | 4 | Number of ASTRA quantiles |
| `N_RAND` | 1 | ASTRA randoms factor |
| `N_RAND_GEOM` | 5 | Geometry randoms factor |
| `N_ASTRA_ITERATIONS` | 10 | Number of independent ASTRA random realisations |
| `SEED` | 42 | Base RNG seed; iteration i uses `SEED+i` for ASTRA randoms |
| `SEED_GEOM` | 1042 | Geometry randoms seed (fixed, `SEED+1000`) |
| `NTHREADS` | 8 | CPU threads for pycorr / corrfunc |
| `S_EDGES` | 0–150 Mpc/h, 30 bins | 2PCF s binning |

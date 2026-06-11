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
│   ├── pipeline_single_box.py    ← end-to-end pipeline (load → classify → split → 2PCF)
│   ├── pipeline_subboxes.py      ← 64-subbox loop on c000 hod000 (cosmic variance)
│   ├── pipeline_subboxes_cosmo.py← same, parameterised by (cosmo, hod) → data/{cosmo}_hod{NNN}/
│   ├── plot.py                   ← figures for pipeline_single_box output
│   ├── plot_subboxes.py          ← figures for pipeline_subboxes output
│   └── plot_subboxes_cosmo.py    ← 7 deduplicated figures per (cosmo, hod) run
├── queue/
│   ├── run_single_box.sh         ← sbatch wrapper, single-box pipeline
│   ├── run_subboxes.sh           ← sbatch wrapper, subbox pipeline
│   ├── run_subboxes_cosmo.sh     ← sbatch wrapper taking <cosmo> <hod> arguments
│   └── launch_fisher_subboxes.sh ← submits all nine Fisher (cosmo, hod) runs
├── data/             ← pipeline output (.npy / .npz); per-run subdirs {cosmo}_hod{NNN}/
├── plots/            ← figure output (.png); per-run subdirs {cosmo}_hod{NNN}/
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
| `data_monopole_per_quantile.png` | s²ξ₀(s) for data Q1–Q4 (mean ± 1σ) |
| `data_quadrupole_per_quantile.png` | s²ξ₂(s) for data Q1–Q4 (mean ± 1σ) |
| `data_multipoles_all_quantiles.png` | Monopole + quadrupole side by side |
| `rand_monopole_per_quantile.png` | s²ξ₀(s) for random Q1–Q4 (mean ± 1σ) |
| `rand_quadrupole_per_quantile.png` | s²ξ₂(s) for random Q1–Q4 (mean ± 1σ) |
| `data_vs_rand_monopole.png` | Data vs randoms monopole, 2×2 panel per quantile |
| `full_data_autocorr.png` | Full sample auto-correlation monopole + quadrupole |
| `cross_full_data_quantiles.png` | Cross-correlation full data × each data quantile |
| `cross_full_rand_quantiles.png` | Cross-correlation full data × each random quantile |
| `cov_data_quantiles.png` | Normalised correlation matrices for data quantiles (2×4 grid: ℓ=0,2 × Q1–Q4) |
| `cov_rand_quantiles.png` | Normalised correlation matrices for random quantiles |
| `cov_cross_full_data_quantiles.png` | Normalised correlation matrices for full×data cross-corrs |
| `cov_cross_full_rand_quantiles.png` | Normalised correlation matrices for full×random cross-corrs |

The random quantile 2PCF is non-trivial (not ≈ 0) because ASTRA randoms
trace the same cosmic-web environments as the data — overdense random
quantiles cluster just as overdense galaxies do.

### Per-cosmology figures (`plot_subboxes_cosmo.py`)

For the Fisher runs, `python scripts/plot_subboxes_cosmo.py [cosmo hod]`
(no arguments = all completed runs) writes a deduplicated 7-figure set to
`plots/{cosmo}_hod{NNN}/`:

| File | Content |
|------|---------|
| `subbox_autocorr_quantiles.png` | Quantile autos, data (solid) + randoms (dashed), ℓ=0 and ℓ=2 panels |
| `subbox_cross_full_quantiles.png` | Full × data-quantile (solid) and × random-quantile (dashed) crosses, same layout |
| `subbox_full_data_autocorr.png` | Full-sample auto-correlation |
| `subbox_cov_*.png` (×4) | Normalised correlation matrices per statistic family |

A run is considered complete when `subbox_info.npz` exists (the last file
the pipeline writes); incomplete runs are skipped.

### Zero crossing of ξ(s) at ~120 Mpc/h

All statistics (full auto, quantile autos, crosses) cross zero at the same
s ≈ 117–123 Mpc/h.  This is expected and physical: linear-theory ξ crosses
zero at ~125–130 Mpc/h because P(k→0) → 0 forces compensation
(∫ξ d³r ≈ 0), at a comoving scale set by the matter-radiation equality
horizon.  Linear bias and Kaiser RSD rescale amplitude only, so all
linearly-biased tracers cross at the same scale — the common crossing is a
consistency check that ASTRA quantiles behave as linearly biased tracers;
a displaced crossing would indicate scale-dependent environmental bias.
The measured crossing sits a few Mpc/h inside the linear prediction because
of the 500 Mpc/h subbox integral constraint (LS references the subbox mean
density, an offset ~10⁻³ comparable to |ξ| at s > 100).  References:
Prada, Klypin, Yepes, Nuza & Gottlöber 2011 (arXiv:1111.2889, zero crossing
as equality-horizon standard ruler); Anselmi, Starkman & Sheth 2016 +
Anselmi et al. 2018 PhRvL 121, 021302 (linear-point standard ruler).

---

## Key parameters (top of `pipeline_single_box.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `BOXSIZE` | 500 Mpc/h | Subbox side length |
| `LOS` | `'z'` | Line-of-sight axis |
| `N_Q` | 4 | Number of ASTRA quantiles |
| `N_RAND` | 1 | ASTRA randoms factor |
| `N_RAND_GEOM` | 5 | Geometry randoms factor |
| `N_ASTRA_ITERATIONS` | 30 | Number of independent ASTRA random realisations |
| `SEED` | 42 | Base RNG seed; iteration i uses `SEED+i` for ASTRA randoms |
| `SEED_GEOM` | 1042 | Geometry randoms seed (fixed, `SEED+1000`) |
| `NTHREADS` | 8 | CPU threads for pycorr / corrfunc |
| `S_EDGES` | 0–150 Mpc/h, 15 bins | 2PCF s binning |

---

## Fisher-matrix design (decided 2026-06-10)

Goal: Fisher forecast on **{ω_b, ω_c, n_s, σ₈}** from the ASTRA quantile clustering
statistics, using central differences over the AbacusSummit linear derivative grid
(all ph000, phase-matched to c000 so cosmic variance cancels in the differences;
θ\* and σ₈ are held fixed via h and A_s retuning — see
`data/abacus_cosmologies_params.csv`).

### Cosmologies and chosen HOD catalogs

Catalogs: `/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/{cXXX}_ph000/seed0/hodNNN.fits`

| Cosmology | Variation | HOD file |
|-----------|-----------|----------|
| c000 | fiducial | `hod484` |
| c100 / c101 | ω_b ±2% | `hod179` / `hod152` |
| c102 / c103 | ω_c ±3.3% | `hod556` / `hod861` |
| c104 / c105 | n_s ±0.01 | `hod498` / `hod589` |
| c112 / c113 | σ₈ ±2% | `hod507` / `hod483` |

### Why these specific HOD files

The emulator catalogs are **not HOD-matched across cosmologies**: each cosmology has
its own random ~500-subset of a 1177-point HOD Latin hypercube with *different*
parameter values at the same `hodNNN` index (verified from all 4500 FITS headers —
zero exact matches). No new HOD catalogs will be generated. The files above were
selected by minimising the HOD-parameter mismatch within each ± pair around a common
mid-prior anchor point (within-pair distance ≈ 0.14–0.16 of the prior range vs ≈ 0.40
for random draws). Galaxy counts are fixed per cosmology by construction, so number
density is matched automatically.

**Important — fiducial change:** the Fisher fiducial is `c000` `hod484`, **not** the
`hod000` used by all pipeline results currently in `data/` and `plots/`. `hod000` is
a prior-edge draw (normalised α≈0.96, logM_cut≈0.95, σ≈0.93) with no nearby draws in
the other cosmologies; the c000 baseline must be rerun on `hod484` before assembling
the Fisher matrix.

Residual within-pair HOD mismatch (up to ~0.3 of the prior range in single parameters,
e.g. B_cen for the ω_b pair) can contaminate derivatives at a level comparable to the
cosmology signal. Mitigations if a derivative looks suspect: average the top-K matched
pairs, or measure ∂ξ/∂θ_HOD from extra c000 draws and subtract.

### Findings from the first completed runs (2026-06-11, c000 + c100)

- **Paired-subbox differencing works.** With matched phases and seeds, forming
  Δξ subbox-by-subbox before averaging shrinks the error ~2× vs treating runs
  as independent; the c100−c000 difference is detected at S/N up to ~78/bin.
- **HOD contamination measured directly.** The c100_hod179 − c000_hod484
  difference is a roughly flat **−14…−16%** amplitude shift in ξ₀ — a ~7% bias
  change from the HOD mismatch, ~15× larger than the ~1% ω_b signal.
  **One-sided derivatives anchored on c000 are therefore unusable.**
- **Use central differences within the matched ± pairs only** (e.g.
  (c100−c101)/0.04 for ln ω_b); the c000 fiducial then drops out of the
  derivative. Expect residual amplitude-like contamination; since the
  contamination is smooth in s while e.g. the ω_b signal is BAO-shaped,
  projecting out a constant-bias (amplitude) mode is the first defence.
- If needed, calibrate ∂ξ/∂θ_HOD by running the pipeline on a few *extra
  existing* c000 HOD catalogs (500 draws on disk; no new HOD generation) and
  subtract the contamination explicitly. The σ₈ derivative needs this most.

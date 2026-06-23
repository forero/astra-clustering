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
├── astra.py          ← ASTRA algorithm module (classify for small catalogs,
│                       classify_fast for full-box scale; identical results)
├── scripts/
│   ├── pipeline_single_box.py    ← end-to-end pipeline (load → classify → split → 2PCF)
│   ├── pipeline_subboxes.py      ← 64-subbox loop on c000 hod000 (cosmic variance)
│   ├── pipeline_subboxes_cosmo.py← same, parameterised by (cosmo, hod) → data/{cosmo}_hod{NNN}/
│   ├── pipeline_fullbox_cosmo.py ← whole 2000 Mpc/h box per (cosmo, hod) → data/fullbox/…/
│   ├── plot.py                   ← figures for pipeline_single_box output
│   ├── plot_subboxes.py          ← figures for pipeline_subboxes output
│   ├── plot_subboxes_cosmo.py    ← 7 deduplicated figures per (cosmo, hod) subbox run
│   ├── plot_fullbox_cosmo.py     ← 3 figures per (cosmo, hod) full-box run
│   ├── compute_derivatives.py    ← central-difference derivatives from completed ± pairs
│   ├── compute_derivatives_fullbox.py ← Tier-0: derivatives from phase-matched full-box ± diffs
│   ├── plot_fisher_gaussians.py  ← per-data-vector Fisher σ as Gaussians around fiducial
│   ├── plot_fisher_fullbox_compare.py ← Tier-0: subbox vs full-box derivative σ comparison
│   ├── select_hod_calibration.py ← Tier-1: maximin-pick c000 HOD draws for ∂ξ/∂θ_HOD
│   ├── compute_hod_derivatives.py← Tier-1: regress ∂ξ/∂θ_HOD, subtract HOD contamination
│   ├── compute_response_global.py← global linear response ξ(θ_cosmo,θ_HOD) → HOD-clean derivs
│   ├── fisher_joint.py           ← joint cosmology+HOD Fisher, HOD-marginalised errors
│   ├── test_hod_response_cosmo_independence.py ← per-cosmology HOD-gradient comparison
│   ├── emulator_hod_c000.py      ← tier-2 GP/linear HOD emulator at c000 (LOO)
│   ├── emulator_diagnostics.py   ← GP calibration / learning curve / ARD relevance
│   ├── fisher_emulator_B.py      ← approach B: per-cosmology GP → matched-HOD derivatives
│   ├── analyze_iter_experiment.py← iter-3 vs iter-10 noise-floor analysis
│   ├── fisher_vector_search.py   ← sweep 16 singles + 120 pairs for the best data vector
│   ├── fisher_vector_addto.py    ← greedy next-stem ranking from a fixed base
│   ├── fisher_greedy_chain.py    ← greedy forward selection (data legs)
│   ├── fisher_greedy_chain_all.py← greedy chain over all 16 stems (noise-aware)
│   ├── fisher_5stem_details.py   ← step-by-step anatomy of a multi-stem Fisher
│   ├── fisher_multipole_compare.py← monopole vs quadrupole vs both
│   ├── fisher_compare_full_vs_5stem.py ← 2PCF vs full+xrQ4+xrQ1 corner (FoM printed)
│   ├── fisher_crosslegs_details.py← xrQ4/xrQ1 measured mono/quad, derivs, covariance
│   ├── fisher_forecast_update.py ← noise-aware 3-way forecast corner
│   ├── fisher_noise_aware.py     ← #1: subtract derivative-noise bias Tr(C⁻¹Covδ)
│   ├── fisher_nonlinear_response.py ← #3: quadratic-in-HOD response, bias check
│   ├── fisher_scale_environment.py← scale-cut survival + signature map (data+random)
│   ├── select_tier3_pilot.py     ← tier-3: maximin HODs for c130–c181 pilot
│   ├── select_tier3_full.py      ← tier-3: maximin HODs for the full c130–c181 campaign (52 cosmo)
│   ├── build_emulator_dataset.py ← cache tier3 + Fisher-anchor runs → data/emulator_tier3/*.npz
│   ├── emulator_tier3_mlp.py     ← tier-3 MLP emulator: LOCO + anchor, 8 diagnostics
│   ├── emulator_tier3_learning_curve.py ← held-out error vs #training cosmologies
│   ├── emulator_tier3_within_cosmo.py   ← within-cosmology HOD-interp floor (no cosmo extrapolation)
│   ├── select_extra_hods.py      ← pick N more maximin HODs extending an existing selection
│   ├── emulator_c000_hod_curve.py← c000 held-out error vs #training HODs (fixed test)
│   └── scale_information.py       ← most-informative spatial scales (cosmology signal vs CV/HOD per s-bin)
├── queue/
│   ├── run_single_box.sh         ← sbatch wrapper, single-box pipeline
│   ├── run_subboxes.sh           ← sbatch wrapper, subbox pipeline
│   ├── run_subboxes_cosmo.sh     ← sbatch wrapper taking <cosmo> <hod> arguments
│   ├── run_fullbox_cosmo.sh      ← same for the full-box pipeline (full CPU node)
│   ├── launch_fisher_subboxes.sh ← submits all nine Fisher (cosmo, hod) subbox runs
│   ├── launch_fisher_fullbox.sh  ← submits all nine full-box runs
│   ├── launch_hod_calibration.sh ← Tier-1: submits the selected c000 HOD-calibration full-box runs
│   ├── launch_hod_ensemble.sh    ← global-response ensemble: 50 HODs × 9 cosmologies
│   ├── launch_iter_experiment.sh ← higher-iteration runs → data/fullbox_iter10/
│   ├── launch_tier3_pilot.sh     ← tier-3 pilot: c130–c181 × 50 HODs → data/fullbox_tier3/
│   ├── run_emulator_tier3.sh     ← GPU sbatch wrapper for emulator_tier3_mlp.py
│   ├── run_fullbox_array.sh      ← job-array runner (manifest-driven full-box pipeline)
│   └── launch_tier3_full.sh      ← full campaign: skip-aware manifest + throttled overrun array
├── data/             ← pipeline output; subdirs {cosmo}_hod{NNN}/, fullbox/, derivatives/
├── plots/            ← figure output; mirrors the data/ layout
├── notes/            ← LaTeX technical notes (zero_crossing/, fisher/, vector_search/, tier3_emulator/)
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

### Batch-queue timings (measured on the 2026-06 Fisher runs)

`launch_fisher_subboxes.sh` / `launch_fisher_fullbox.sh` each `sbatch` the
corresponding run script nine times (one per cosmology), inheriting its limits.

| Run script | `#SBATCH -t` | cores `-c` | actual wall (9 runs) |
|------------|--------------|-----------|----------------------|
| `run_subboxes_cosmo.sh` | 4:00:00 | 8 | **3:14–3:28** (~85% of cap) |
| `run_fullbox_cosmo.sh`  | 1:00:00 | 256 | **0:24–0:27** (3 iterations) |

- The **subbox limit is tight** — ~30–45 min of margin against 4 h. Bump to
  `-t 5:00:00` on reruns, or raise `-c` (it uses only 8 of the node's 256 cores
  while `regular` qos charges the whole node anyway).
- **Always request < 1 h for full-box runs** (`run_fullbox_cosmo.sh` now sets
  `-t 1:00:00`). Runtime is ~25 min; the old `-t 8:00:00` was ~20× the real
  cost and choked Slurm **backfill** — the scheduler can't slot an 8 h job into
  short gaps, so a multi-hundred-job campaign (e.g. the HOD ensemble) trickled
  in at ~1 job/h behind higher-priority work instead of bursting. If a campaign
  is already queued at the old limit, fix the pending jobs in place:
  `squeue -u forero -h -t PENDING -o "%i" | xargs -I{} scontrol update jobid={} TimeLimit=01:00:00`.
  Full-box uses all 256 cores, hence ~8× faster than the 8-core subbox runs
  despite covering the whole box.

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

### Derivatives and first Fisher estimates (2026-06-11, ω_b pair)

Workflow: `python scripts/compute_derivatives.py` (paired per-subbox central
differences for every complete ± pair → `data/derivatives/derivative_{param}.npz`
+ 2×5-panel figure), then `python scripts/plot_fisher_gaussians.py`
(one-parameter Fisher per data vector: c000 64-subbox covariance,
Hartlap-corrected, derivative-noise bias subtracted; Gaussians around the
fiducial in `plots/derivatives/`). Both auto-detect which pairs exist.

Findings from the ω_b pair (c100−c101):

- The pair difference is a **scale-dependent** −6%…+9% tilt in ξ₀ (S/N≈19
  at small s) — still HOD-dominated (expected ω_b signal ~1%), and a tilt,
  not a flat offset, so a constant-amplitude nuisance won't fully absorb it.
  All σ values below are therefore optimistic; read them as *relative*
  comparisons between data vectors.
- Per-vector ranking (ℓ=0+2, per 500 Mpc/h subbox): **full×dataQ3 (σ_lnωb
  ≈0.0092) beats the full auto (0.0102)**; then full×dataQ4, randQ1 auto,
  dataQ3/Q4 autos. Underdense random autos carry no usable signal at current
  derivative precision (noise-bias dominated).
- The quadrupole adds almost nothing for ω_b; doubling the bins costs more
  Hartlap penalty than it gains.
- Concatenating vectors that share the full-sample leg
  (full ⊕ full×dQ4 ⊕ full×rQ1) is ~a tie with the full auto alone, even
  after rebinning 15→8 bins per piece to tame Hartlap ((64−nb−2)/63):
  the pieces are too redundant. More information requires independent
  tracers in the vector or more covariance samples, not more crosses.

### Full 4-parameter grid complete (2026-06-16)

All nine (cosmo, hod) runs are now finished at **both** resolutions (64-subbox
and full box), and all four derivatives exist:
`derivative_{lnwb,lnwc,ns,lns8}.npz`. The c100/c101/c102/c103 subboxes were
re-run under the **AP-rescaled subbox-grid fix** (`pipeline_subboxes_cosmo.py`):
the grid now tiles the box bounds ±(FULL_SIZE/2)/q per axis rather than a fixed
±1000 grid, which previously left empty slabs in edge subboxes whenever q > 1
(e.g. c103: ±985.6 in z) so the LS geometry randoms covered galaxy-free volume.
`subbox_info.npz` now also stores `box_lo`/`box_hi`.

**Scope of the current Fisher comparison.** `plot_fisher_gaussians.py` now
compares a fixed set of **four monopole-only (ℓ=0) vectors** — `full auto`,
`full×data Q4`, `full×rand Q1`, and a rebinned×2 concatenation of the three —
*not* the earlier exhaustive quantile sweep that found full×dataQ3 best for ω_b.
σ values are per 500 Mpc/h subbox, Hartlap-corrected, derivative-noise-bias
subtracted. Read them as relative comparisons (HOD contamination makes the
absolute numbers optimistic).

Per-vector ranking by parameter (σ, with derivative-noise fraction):

| Param | Best vector | σ (noise) | runner-up | weakest |
|-------|-------------|-----------|-----------|---------|
| ω_b (ln) | **full auto** | 0.000213 (1%) | concat 0.000221 | full×randQ1 0.00060 (10%) |
| ω_c (ln) | **full×data Q4** | 0.000333 (0%) | full auto 0.000334 | full×randQ1 0.00054 |
| n_s | **concat** | 0.000727 (0%) | full×dataQ4 0.000733 | full×randQ1 0.00120 |
| σ₈ (ln) | concat | 0.0724 (60%) | full×randQ1 0.0915 | full×dataQ4 0.139 (83%) |

Conclusions:

- **The full-sample auto and the full×data-Q4 cross are the two workhorses**,
  statistically tied at the top for ω_b, ω_c and n_s (sub-percent σ differences).
  Either is a fine single-vector choice for the three clean parameters.
- **full×rand Q1 (underdense random leg) is consistently weakest** (~1.6× worse
  on ω_c/n_s, ~3× on ω_b) and carries the most derivative noise — little
  independent signal at current precision.
- **Concatenating helps only marginally**, and only for n_s and σ₈; for ω_b it
  is a hair *worse* than the full auto alone. The pieces share the full-sample
  leg, so the Hartlap penalty roughly cancels the added information — consistent
  with the ω_b-only finding above.
- **σ₈ is not yet usable.** Every vector is 60–83% derivative noise, σ(ln σ₈) an
  order of magnitude looser than the other parameters. This is the HOD-mismatch
  contamination the design flagged; the σ₈ row needs explicit ∂ξ/∂θ_HOD
  calibration (extra c000 HOD draws) before its constraints mean anything.

### Tier 0 — full-box derivatives fix the σ₈ *noise* (2026-06-16)

`compute_derivatives_fullbox.py` builds the derivative numerator from the
**phase-matched full-box ± difference** (cosmic variance cancels, ~4M galaxies,
periodic BC → no open-boundary integral constraint), with a diagonal noise model
from the N_ITER=3 ASTRA-random iterations. `plot_fisher_fullbox_compare.py` then
evaluates **both** derivative sources against the same covariance — the 64 c000
subboxes scaled to full-box volume (C_subbox/64, Hartlap-corrected) — so only the
derivative numerator/noise differs. Neither needs new compute (full-box data was
already on disk). Result (full-auto vector, σ at full 2000 Mpc/h volume):

| Param | σ subbox-deriv (noise) | σ full-box-deriv (noise) | gain |
|-------|------------------------|--------------------------|------|
| ω_b | 2.67e-5 (1%) | 2.76e-5 (0%) | 0.97× |
| ω_c | 4.18e-5 (0%) | 4.10e-5 (0%) | 1.02× |
| n_s | 9.55e-5 (0%) | 9.60e-5 (0%) | 0.99× |
| **σ₈ (ln)** | **0.0164 (79%)** | **0.00925 (0%)** | **1.78×** |

- **Consistency check passes:** for the three already-clean parameters the two
  derivative methods agree to ≈3%, confirming they estimate the same dξ/dθ.
- **σ₈ is the win:** the full-auto σ(σ₈) goes from 0.0164 at 79% derivative noise
  (unusable) to 0.00925 at ≈0% noise — ~1.8× tighter *and* now trustworthy — from
  reanalysis alone. The cosmic-variance cancellation in the phase-matched full-box
  difference removes the noise that dominated the subbox-paired σ₈ derivative.
- **Caveat 1 — noise model:** the full-box noise is diagonal from only 3
  iterations and so *under*-states the bias; read "0% noise" as "noise no longer
  dominant," not exactly zero. (full×rand Q1 for σ₈ goes to F−bias≤0 → NaN: that
  underdense leg carries no σ₈ signal regardless of method.)
- **Caveat 2 — bias remains:** Tier 0 fixes derivative *noise*, not the HOD
  *contamination bias*. The full-box σ₈ difference still contains the c112/c113
  HOD mismatch, so 0.00925 is noise-clean but possibly biased. The remaining
  defence is still the Tier-1 ∂ξ/∂θ_HOD calibration from extra c000 draws.

### Tier 1 — HOD-contamination calibration (run complete 2026-06-17)

Subtract the HOD contamination from the full-box derivatives by measuring
∂ξ/∂θ_HOD on extra existing c000 draws and projecting it onto each pair's HOD
mismatch. Workflow:

1. `python scripts/select_hod_calibration.py [N]` — reads the 12 varying yuan23
   HOD parameters (LOGM_CUT, LOGM1, SIGMA, ALPHA, KAPPA, ALPHA_C, ALPHA_S, S,
   ACENT, ASAT, BCENT, BSAT) from all 500 c000 catalog headers and picks N=50 by
   maximin sampling seeded on the fiducial hod484 →
   `data/hod_calibration/{hod_params_c000.csv, hod_selection_c000.txt}`.
2. `bash queue/launch_hod_calibration.sh` — submits one full-box job per selected
   draw (skips any already done; 49 new, ~25 min each on a full node).
3. `python scripts/compute_hod_derivatives.py` — regresses full-box ξ on the
   standardised HOD params (linear + intercept), evaluates the contamination
   `[∂ξ/∂θ_HOD]·Δθ_HOD/(2 dθ_cosmo)` per pair, subtracts it →
   `data/derivatives/derivative_hodcorr_{param}.npz` +
   `plots/derivatives/hod_contamination_{param}.png`. Needs >13 completed
   calibration runs (fits 12 params); gates itself otherwise.

Assumptions/caveats: HOD response is taken cosmology-independent to first order
(gradient from c000, applied to the ± cosmologies' mismatch) and locally linear
over the prior. The measured ± HOD mismatches are **large** — ω_b pair 0.84
prior-σ rms (2.3σ in SIGMA), σ₈ pair 0.50σ rms (1.4σ in ALPHA_S) — so even the
low-*noise* ω_b derivative may carry significant HOD *bias*; the noise diagnostic
does not catch it. This is why the subtraction matters for every parameter, not
just σ₈.

**Results — all 50 draws complete (2026-06-17).** The 49 jobs finished
(`COMPLETED`, no failures); the regression now runs on the full maximin set.

- **Fit is well-conditioned:** design-matrix condition number 1.8 (37 dof).
  The ± pair HOD midpoints sit 2.0–2.9 standardised-σ from the calibration-cloud
  center, which in 12-D is *inside* the cloud (draws span 2.3–4.0σ, mean 3.45 ≈
  √12) — i.e. the correction interpolates, it does not extrapolate.
- **The unphysical sign is gone.** With 50 draws the corrected σ₈ monopole
  derivative is **positive and declining at small s** — the expected ≈2ξ shape
  for ∂ξ/∂ln σ₈ — recovering the cosmology signal the HOD contamination had been
  masking. (The preliminary 21-draw fit gave a *negative* small-s derivative; that
  was a coverage/stability artefact of an arbitrary first-to-finish subset, not
  extrapolation, and it resolved once the spanning set completed.)
- **Contamination is a large, smooth tilt** for every parameter (largest at small
  s, sign-changing near the BAO/zero-crossing scale). It is comparable to or
  larger than the raw full-box derivative, confirming the derivatives were heavily
  HOD-contaminated and that the subtraction is essential. (Do not read the
  |contamination|/|raw| ratio literally — it is inflated wherever the raw
  derivative crosses zero; judge from the curve shapes instead.)
- **Still a first-order, single-gradient model.** The cosmology-independence and
  local-linearity assumptions remain; treat the corrected derivatives as the best
  current estimate, not the final word.

### Fisher updated with HOD-corrected derivatives (2026-06-17)

Both Fisher scripts now consume the HOD-corrected full-box derivatives:
`plot_fisher_fullbox_compare.py` adds a third method (subbox → full-box raw →
full-box HOD-corrected); `plot_fisher_gaussians.py` now uses
`derivative_hodcorr_*` at full-box volume (C_subbox/64, diagonal noise).
Full-auto σ at full 2000 Mpc/h volume:

| Param | full-box raw | HOD-corrected | change |
|-------|--------------|---------------|--------|
| ω_b | 2.76e-5 | 2.69e-6 | 10.2× tighter |
| ω_c | 4.10e-5 | 4.94e-5 | 0.83× (looser) |
| n_s | 9.60e-5 | 1.43e-4 | 0.67× (looser) |
| σ₈ (ln) | 9.25e-3 | 5.23e-4 | 17.7× tighter |

- **The correction is physically validated.** For σ₈, theory gives ∂ξ/∂ln σ₈ =
  2ξ. The *raw* full-box derivative is only **11%** of 2ξ (the c112−c113
  difference was gutted — the HOD mismatch cancelled ~89% of the σ₈ signal); the
  *corrected* derivative is **136%** of 2ξ — back in the right ballpark. So the
  dramatic ω_b/σ₈ tightening is the raw σ being artificially **loose** (suppressed
  derivative), not the corrected σ being spuriously tight.
- **ω_b and σ₈ raw differences were strongly HOD-suppressed**, so their correction
  is large; ω_c and n_s were not, so their corrected σ barely moves (slightly
  looser, a sign the correction is a mild perturbation there).
- **Critical caveat — these σ hold the HOD parameters FIXED.** They do *not*
  marginalise over HOD nuisances. The σ₈ derivative is amplitude-like (≈2ξ), hence
  strongly degenerate with HOD bias/amplitude, so the realistic marginalised σ₈
  would be **much weaker** than 5e-4 — that number is an optimistic "HOD known
  exactly" bound. ω_b is BAO-shaped and far less degenerate, so its improvement is
  more robust. The proper next step is a joint cosmology+HOD Fisher (or projecting
  out the amplitude mode, as flagged in the design) to get marginalised errors.

### Joint cosmology+HOD Fisher (2026-06-17)

`scripts/fisher_joint.py` builds one Fisher over {ω_b, ω_c, n_s, σ₈} + the 12
HOD parameters and marginalises over the HOD nuisances. Cosmology rows =
`derivative_hodcorr_*` (fixed-HOD cosmology derivative); HOD rows =
`hod_gradient.npz` (the regression gradient, now saved by
compute_hod_derivatives.py); covariance = 64 c000 subboxes at full-box volume.
The poorly-constrained HOD directions are regularised by a **Gaussian yuan23
prior block** (σ_prior = std of the c000 prior draws) — *not* by PCA truncation,
which would be an implicit infinite prior and reintroduce the fixed-HOD optimism.
Data vector: full-auto mono+quad (30 bins > 16 params, Hartlap 0.51).

Conditional = HOD fixed (invert the 4×4 cosmology block of F_data, so it is
marginalised over the *other* cosmology params); marginalised = invert the full
16×16 F+prior, take the cosmology block:

| Param | σ_cond (HOD fixed) | σ_marg (HOD marg.) | degrade | σ_marg / fiducial |
|-------|--------------------|--------------------|---------|-------------------|
| ω_b | 3.08e-5 | 2.49e-4 | 8.1× | 1.1% |
| ω_c | 4.88e-4 | 1.01e-3 | 2.1× | 0.85% |
| n_s | 5.78e-4 | 5.89e-3 | 10.2× | 0.61% |
| σ₈ (ln) | 3.18e-3 | 9.31e-3 | 2.9× | 1.15% |

- **Corner plot** `fisher_joint_ellipses.png` shows the 4-parameter joint
  constraints (68%/95% ellipses in physical units), overlaying HOD-fixed (grey)
  vs HOD-marginalised (blue); the blue ellipses dwarf the grey and their tilts
  show the cosmology-parameter degeneracies that HOD marginalisation opens up.
- **Robustness check passes:** the convergence curve (`fisher_joint_convergence.png`)
  shows every parameter plateaus by ~4 marginalised HOD directions and is flat
  through k=12 — the HOD response is effectively low-rank, the prior-controlled
  tail is irrelevant, and a k≈5 PCA truncation would give the same answer (PCA as
  check, not method).
- **The degradation pattern is not what the amplitude argument alone predicts.**
  n_s and ω_b degrade most (10×, 8×) — their tight conditional errors overlap the
  broadband HOD tilt/amplitude directions; ω_c and σ₈ degrade least (2.1×, 2.9×).
  But in *fractional* terms σ₈ ends up the **weakest** (1.15%), consistent with
  its amplitude degeneracy — its degradation factor is only moderate because its
  conditional error was already loose. All four land at ~0.6–1.2% from a single
  2 Gpc box full-auto 2PCF, which is healthy.
- **Caveats:** HOD gradient measured at c000 (cosmology-independence assumption —
  relax via the multi-HOD-per-cosmology runs); single box volume; full-auto only
  (adding ASTRA quantile vectors could break degeneracies further — the point of
  ASTRA — at a Hartlap cost); point-estimate derivatives (no derivative-noise or
  gradient-uncertainty propagation); prior width proxied by the c000 draw spread.

### ASTRA quantile vectors break HOD degeneracies (2026-06-18)

`fisher_joint.py` now runs the joint HOD-marginalised Fisher over a *set* of data
vectors (helpers `assemble`/`fisher`/`corner`; quantile pieces rebinned ×2 to keep
Hartlap sane), so the ASTRA environment splits can be compared against the
full-auto baseline. New figure `fisher_joint_ellipses_vectors.png` overlays the
marginalised corner plot for all vectors. HOD-marginalised σ (physical units):

| Data vector | nb | Hartlap | ω_b | ω_c | n_s | σ₈ |
|-------------|----|---------|-----|-----|-----|----|
| full auto (mono+quad) — baseline | 30 | 0.51 | 2.49e-4 | 1.01e-3 | 5.89e-3 | 9.31e-3 |
| data Q autos (mono, ×2) | 32 | 0.48 | 2.13e-4 | 8.32e-4 | 2.78e-3 | 8.35e-3 |
| full × data Q (mono, ×2) | 32 | 0.48 | 2.56e-4 | 8.30e-4 | 2.57e-3 | 9.63e-3 |
| **full + data Q autos (mono, ×2)** | 40 | 0.35 | **1.84e-4** | **6.72e-4** | **2.48e-3** | **7.28e-3** |

- **The ASTRA thesis holds in the marginalised regime.** Full sample ⊕ four
  quantile autos is the best vector and tightens *every* HOD-marginalised
  cosmology error vs the full auto: **n_s 2.4×**, ω_c 1.5×, ω_b 1.35×, σ₈ 1.28×.
  Environment splits respond differently to cosmology vs the HOD, breaking
  degeneracies the full-auto 2PCF cannot — and the gain survives the Hartlap cost
  (nb 30→40, correction 0.35).
- **Quantile autos alone already beat the full auto** on all four params (not just
  redundant signal). **Crosses (full×Q)** help n_s/ω_c but are a wash on ω_b/σ₈ —
  the autos are the workhorse, consistent with the earlier full-vs-cross findings.

### Pooled covariance unlocks the quadrupole (2026-06-18)

The quadrupole was previously untestable on the quantile vectors: with only the
64 c000 subboxes, adding ℓ=2 pushed nb past 64 and the Hartlap factor went
negative. `fisher_joint.py` now estimates the 500 Mpc/h subbox covariance from
the **mean-subtracted subboxes pooled across all 9 cosmologies** (`POOL_COV=True`):
each cosmology's own mean is removed so only cosmic-variance fluctuations remain
(taken cosmology-independent over this grid), giving 9×64 = **576 covariance
samples** instead of 64. nb=80 vectors now sit at Hartlap 0.86. The VECTORS set
was changed to mono-vs-mono+quad pairs that share the monopole binning, isolating
what the quadrupole buys. HOD-marginalised σ (physical units, 576-sample cov):

| Data vector | nb | hart | ω_b | ω_c | n_s | σ₈ |
|-------------|----|------|-----|-----|-----|----|
| full auto (mono+quad) | 30 | 0.95 | 2.18e-4 | 9.31e-4 | 5.23e-3 | 8.04e-3 |
| data Q autos (mono) | 32 | 0.94 | 1.94e-4 | 7.53e-4 | 2.99e-3 | 8.61e-3 |
| data Q autos (mono+quad) | 64 | 0.89 | 1.46e-4 | 5.51e-4 | 1.81e-3 | 5.45e-3 |
| full+dataQ autos (mono) | 40 | 0.93 | 1.68e-4 | 5.80e-4 | 2.40e-3 | 6.83e-3 |
| **full+dataQ autos (mono+quad)** | 80 | 0.86 | **1.14e-4** | **4.47e-4** | **1.55e-3** | **4.60e-3** |

- **The earlier "quadrupole adds almost nothing" was a Hartlap-budget artifact.**
  At a fixed nb the quad competed with monopole resolution and lost; with the
  pooled covariance it fits at native binning and **tightens every parameter by
  ~1.3–1.65×** (data-Q autos: ω_b 1.33×, ω_c 1.37×, n_s 1.65×, σ₈ 1.58×;
  full+dataQ autos: 1.47×/1.30×/1.55×/1.48×). The quadrupole (RSD) carries real,
  largely independent information once it isn't crowded out.
- **Best vector = full + data-Q autos, mono+quad** — ~1.5× tighter on *every*
  parameter than the previous monopole-only best, from reanalysis alone.
- **Pooling alone is a mild gain** (~1.1× on the full-auto baseline: Hartlap was
  already benign there); the real win is that 576 samples make the quadrupole
  *affordable*.
- **Caveats:** assumes the subbox covariance is cosmology-independent over the
  ±2–3% grid (mild); subboxes tile one box per cosmology so they share
  large-scale modes — the effective independent-sample count is below 576 (this
  approximation was already implicit in the 64-subbox estimate). Exploiting the
  50 same-phase c000 HOD runs as a direct empirical HOD covariance (C = C_CV +
  C_HOD) remains an open, more-honest alternative to the gradient+prior block.

### Global response model — clean derivatives + systematic vectors (2026-06-18)

**Decision:** exploit the **500 existing HOD draws per cosmology** (all 9 Fisher
cosmologies, same yuan23 prior, ph000) rather than the single mismatched catalog
per cosmology. Running ~50 maximin draws per cosmology and fitting one **global
linear response** ξ(θ) ≈ ξ₀ + Σ a_p θ_cosmo,p + Σ b_q θ_HOD,q across all runs
gives cosmology derivatives that are **HOD-clean by construction** (the HOD term
absorbs the cross-cosmology HOD mismatch) plus the HOD gradient, in one fit. All
runs share ph000 so cosmic variance cancels in the cosmology coefficients.

**This supersedes the Tier-0/Tier-1 derivative path** (`compute_derivatives_fullbox.py`
+ `compute_hod_derivatives.py` → `derivative_hodcorr_*`), which is demoted to a
finite-difference cross-check. The Tier-0/Tier-1 sections above are kept for
history; the contamination-*subtraction* is no longer the method.

New/changed pipeline:
- `scripts/select_hod_ensemble.py` — maximin-pick 50 HOD draws per cosmology
  (seeded on each cosmology's existing Fisher pick so prior runs are reused; the
  c000 set is identical to the old calibration list) → `data/hod_ensemble/`.
- `queue/launch_hod_ensemble.sh [cosmo|all] [iters]` — submit full-box runs for
  the selection, skipping completed ones.
- `scripts/compute_response_global.py` — the global regression →
  `derivative_global_{param}.npz` (drop-in for `derivative_hodcorr_*`) +
  `hod_gradient_global.npz`; prints condition number, σ₈ sanity (∂ξ/∂lnσ₈≈2ξ),
  and a global-vs-FD figure per parameter.
- `scripts/compute_hod_covariance.py` — empirical `C_HOD` from the same-phase
  c000 ensemble → `hod_covariance.npz`.
- `scripts/fisher_joint.py` — auto-selects the global derivatives; VECTORS now
  span {ℓ0, ℓ2} × {data, random} × {full, Q autos, full×Q}; reports **two
  marginalisation routes**: (a) joint 16-param + yuan23 prior, (b)
  C_total = C_CV + C_HOD (4 params, prior-free cross-check).

**Smoke test on the 58 runs already on disk** (50 c000 + 1 each other cosmology;
the ± cosmologies still single-HOD until the campaign fills in). Design-matrix
condition number **1.89**; ∂ξ/∂lnσ₈ vs 2ξ ratio 0.68 (order-unity, will tighten).
HOD-marginalised σ, route (a) / route (b):

| Data vector | nb | hart | ω_b | ω_c | n_s | σ₈ |
|-------------|----|------|-----|-----|-----|----|
| full auto (mono+quad) | 30 | 0.95 | 2.1e-4 / 2.6e-4 | 9.7e-4 / 1.4e-3 | 5.3e-3 / 6.8e-3 | 8.2e-3 / 1.0e-2 |
| data Q autos (mono+quad) | 64 | 0.89 | 1.5e-4 / 1.9e-4 | 5.6e-4 / 9.4e-4 | 1.8e-3 / 3.1e-3 | 5.4e-3 / 5.9e-3 |
| rand Q autos (mono+quad) | 64 | 0.89 | 1.4e-4 / 1.5e-4 | 5.8e-4 / 7.6e-4 | 2.1e-3 / 3.1e-3 | 6.3e-3 / 7.2e-3 |
| data+rand Q autos (mono+quad) | 128 | 0.78 | 8.6e-5 / 1.1e-4 | 3.1e-4 / 5.0e-4 | 1.1e-3 / 2.0e-3 | 4.1e-3 / 4.7e-3 |
| **full+data+rand Q autos (mono+quad)** | 144 | 0.75 | **6.9e-5 / 9.9e-5** | **2.7e-4 / 4.3e-4** | **9.8e-4 / 1.8e-3** | **3.7e-3 / 4.3e-3** |

- **The two routes broadly agree** (~1.3–1.6×; route b looser — prior-free, and
  C_HOD is from only 50 draws), validating the marginalisation.
- **Adding the random quantiles and the quadrupole both help**, consistent with
  the pre-redesign single-vector tests: the workhorse data+random Q-autos
  mono+quad vector is the tightest on every parameter.
- **Status:** the analysis runs on partial data and improves as the per-cosmology
  ensembles fill in (the ± cosmologies are still single-HOD in this smoke test, so
  the σ₈ sanity ratio is 0.68 not ~1). Budget: 50 draws/cosmology (~392 new runs)
  via `launch_hod_ensemble.sh`; scripts ingest more draws later with no change.

### HOD-response cosmology-independence test (2026-06-21)

`scripts/test_hod_response_cosmo_independence.py` checks the assumption the whole
global-response design rests on: that ∂ξ/∂θ_HOD is the same in every cosmology. It
fits the 12-param HOD gradient *separately within* each cosmology that has ≥15
same-phase draws (intercept + standardised HOD regressors; cosmology params are
constant within a cosmology) and compares per-cosmology gradients with propagated
fit errors → `data/derivatives/hod_gradient_percosmo.npz` +
`plots/derivatives/hod_response_cosmo_independence_ell{0,2}.png`. Auto-detects ready
cosmologies and prints the axes covered.

- **With c000/c100/c101 at 50 draws each, the ω_b axis passes:** screening χ²/dof =
  0.32–0.49 (ℓ=0 and ℓ=2), zero bins beyond 2σ — the per-cosmology HOD gradients are
  statistically indistinguishable. The single-gradient assumption holds where
  testable. χ²/dof is approximate (s-bins correlated); the "all gradient curves
  coincide" figure is the robust read.
- **Scope is limited to ω_b today** — σ₈/ω_c/n_s come online as c112/c113, c102/c103,
  c104/c105 reach ≥15 draws (re-run; it auto-expands). **σ₈ is the one to watch**
  (amplitude-like, most degenerate with HOD bias).
- Refreshed global derivatives on the 164-run grid: condition number 1.50,
  ∂ξ/∂lnσ₈-vs-2ξ ratio 0.68→1.35 (order-unity now the ω_b leg is fully
  HOD-marginalised; settles toward 1 as the σ₈ pair fills in).

### Tier-2 HOD emulator prototype + iteration experiment (2026-06-21)

Emulator feasibility study at fixed cosmology (the genuinely space-filling axis).
The source catalog has **85 cosmologies** (`c000–c181`, the full AbacusSummit
emulator grid incl. the space-filling `c130–c181` LH) × 500 HODs each — so a real
emulator is a *compute*, not a *data*, question; only the 9 Fisher cosmologies have
been processed.

- `scripts/emulator_hod_c000.py` — trains f: θ_HOD(12-D) → data vector on the 50
  c000 runs; standardise → PCA → per-component {Ridge-linear, GP Matern-5/2}, with
  leave-one-out CV. `--all-stems` does all 17 stems (510-D); default is full-auto.
  Outputs `data/emulator/` + `plots/emulator/emulator_hod_c000_{loo,pred}.png`.
  - **GP ≫ linear** (LOO RMS/spread 0.40 vs 0.84): the HOD response is substantially
    **nonlinear** — a real emulator buys a lot over the linear response model. But
    50 points in 12-D is thin (GP captures ~75%, not production-grade) → the lever is
    more draws (500 exist).
  - **Per-environment:** data & random **monopoles** emulate well in every quantile;
    **random quadrupoles fail** (RMS/spread ≈ 1.0) because their targets are
    noise-dominated, not because the model is too weak.
  - At ≳1000s of training vectors (tier-3 = process `c130–c181`) switch to an MLP
    (SUNBIRD-style; torch/jax/tf all in the env); GP is correct for ≤ few-hundred.
- **Iteration experiment** (`queue/launch_iter_experiment.sh [niter] [ndraws]`,
  pipeline `--outroot` keeps it isolated in `data/fullbox_iter10/`;
  `scripts/analyze_iter_experiment.py`). Noise-floor analysis showed data quantiles
  are signal-rich (noise/spread 0.04–0.22) but **random quadrupoles are
  noise-limited** (rand_q2/q3 ℓ2 ≈ 0.72–0.78). Reran 10 c000 draws at 10 iterations:
  noise dropped 1.44–1.77× (≈√(10/3), validates 1/√N); rand_q2/q3 ℓ2 moved
  NOISE-LIMITED→marginal (0.47), rand_q3 ℓ0 / rand_q4 ℓ2 became signal-rich.
  - **Two independent levers, now separated:** random quadrupoles need more
    **iterations** (~25 total to reach signal-rich <0.3: `launch_iter_experiment.sh
    25 10`); data quantiles need more **draws** (signal-rich but 50-point emulator
    underfits, e.g. data_q4 ℓ2).

### Data-vector search, random-leg vindication, scale finding (2026-06-20→22)

Full writeup: `notes/vector_search/vector_search_note.{tex,pdf}`. Fisher search
over the ASTRA data vectors (data/random × env-quantile autos & full-crosses ×
mono/quad), HOD-marginalised, pooled 576-subbox covariance.

- **Random legs are real signal, not an artefact** — tested two ways and they
  survive both: noise-aware Fisher (`fisher_noise_aware.py`, #1: subtract
  Tr(C⁻¹·Covδ) using the random ASTRA-iteration noise) deflates them only ~25–40%;
  a quadratic-in-HOD response (`fisher_nonlinear_response.py`, #3) shifts all
  derivatives ~30–60% equally. The random-quantile *monopoles* carry real
  cosmic-web signal; only the velocity-free *quadrupoles* are weak. **In data the
  randoms come from the LSS random catalog matched to the survey window → survey-
  robust.** (Earlier "random-quadrupole trap / artefact" framing was wrong.)
- **Optimal noise-aware vector** (`fisher_greedy_chain_all.py`): full + xrQ4 + xrQ1
  + xdQ3 + xdQ4 + dQ1 + xrQ3 → **FoM3 210×** vs 31× data-legs-only; per-param
  ω_b 5.2×, ω_c 7.6×, n_s 4.8×, σ8 2.7×. Just the **two void/knot random crosses**
  full+xrQ4+xrQ1 already give FoM3 48× (`fisher_compare_full_vs_5stem.py`).
  Workhorses = extreme-environment CROSSES. Forecast corner:
  `fisher_forecast_update.py`. **Why the two crosses work** (`fisher_crosslegs_details.py`):
  xrQ4 (knot) and xrQ1 (void) are mirror-images — opposite measured ξ0 signs
  (+0.50 vs −0.43), opposite cosmology-derivative signs, and ANTI-correlated in C
  (full–xrQ4 block +, full–xrQ1 & xrQ4–xrQ1 blocks −) → two partially-independent
  probes → strong Fisher leverage.
- **THE gain is small-scale** (`fisher_scale_environment.py`): FoM3 210→20.6→4.4×
  at s>0/20/40 Mpc/h; s<40 band alone holds 143×, BAO/large ~1× — because ASTRA
  splits by local density and all quantiles cross zero together at large s. So the
  headline gains need *simulation-based small-scale modelling*; a conservative
  s>40 cut leaves a robust but modest ~1.5×. σ8 is the laggard throughout.
- **Greedy ≠ optimal**: it is a useful interpretable heuristic; MOPED/compression
  is the optimal-but-abstract alternative.

### Tier-3 emulator: design + pilot (2026-06-21→22)

Design note: `notes/tier3_emulator/tier3_emulator_note.{tex,pdf}`. Goal: turn the
Fisher *forecast* into *measured* constraints via a simulation-based MLP emulator
ξ(θ_cosmo, θ_HOD), since the gain is small-scale (can't use perturbation theory).

- **Derivatives**: train an MLP on the space-filling block **c130–c181** (~52
  cosmologies, the AbacusSummit emulator LH; varies the broader w0waCDM+ set →
  ~8-D cosmology input) × ~120 HODs; derivatives = autodiff at the fiducial
  (nonlinear, HOD-clean by single-HOD evaluation). These halos *are* HOD-populated
  at ph000 (ntbfin) — reachable.
- **Covariance constraint (verified)**: ntbfin HOD catalogs are **ph000 only**;
  the AbacusSummit halo catalogs (25 base phases + 1883-box `small/` suite) are on
  CFS but using them = new HOD population, which is **out of scope (no new HOD
  generation)**. So the covariance stays the **subbox estimate** (~10–20% absolute)
  unless an external pre-populated galaxy mock suite is adopted — the main residual
  weakness of Tier-3.
- **Pilot (launched 2026-06-21)**: 10 cosmologies (c130 c135 … c180) × 50 maximin
  HODs × 3 iters → isolated `data/fullbox_tier3/` (`select_tier3_pilot.py`,
  `launch_tier3_pilot.sh`). Pipeline validated on c130 (sane vectors). Go/no-go:
  sub-noise emulator accuracy at s<40, leave-one-cosmology-out. Next steps: build
  the c130–c181 cosmology-param table; train a prototype MLP once ≥3–4 cosmologies
  land.

### Tier-3 MLP emulator: pilot reviewed + full campaign launched (2026-06-23)

Design note `notes/tier3_emulator/mlp_emulator_note.{tex,pdf}`. Inputs = 20-D
(8 varying cosmo params ω_b,ω_c,h,n_s,α_s,N_ur,w0,wa + 12 yuan23 HOD); targets =
void/knot legs (randoms first, then data) mono+quad; split = **leave-one-cosmology-
out** (≈76/14/10% *by cosmology*, not by run) + the 9 Fisher cosmologies as an
external anchor. `build_emulator_dataset.py` caches all 17 stems×2ℓ×15 bins (510-D)
once → `data/emulator_tier3/{dataset,dataset_anchor}.npz` so retraining masks
columns, never re-globs. `emulator_tier3_mlp.py` = torch MLP (noise-weighted MSE,
5-model ensemble), 10-fold LOCO + anchor, 8 diagnostics → `plots/emulator_tier3/`.

- **Pilot result (500 runs, 10 cosmologies): machinery validated, NOT yet accurate
  enough — no-go at pilot scale.** Predictions are physical and 7/10 folds beat the
  mean (RMS/spread 0.11–0.49), but vs **cosmic variance at the 2 Gpc/h box** the
  LOCO error is ~40–130× too large on the priority random void/knot legs. Diagnosis:
  the **sparse cosmology axis** (9 training cosmologies over 8-D → LOCO is mostly
  *extrapolation*) — hull-edge cosmologies are worse-than-mean (c160 2.8×, c155
  1.7×, c145 1.3×), hull-center fine (c165 0.11×); the anchor c000 (inside the hull)
  is predicted well = interpolation works, extrapolation fails. HODs are already
  well-sampled → **adding cosmologies, not HODs, is the lever.**
- **Yardstick caveat:** the ASTRA per-iteration noise is a far-too-stringent
  denominator on a 4M-galaxy box; the inference-relevant comparison is cosmic
  variance (subbox C/64), as above. For inference: emulator = forward model μ(θ)
  only; covariance from **simulations** (subbox C_CV) + additive C_emu from **LOCO
  residuals** (not the ensemble spread) — this is the SUNBIRD approach.
- **Full campaign launched 2026-06-23** (`select_tier3_full.py`,
  `run_fullbox_array.sh`, `launch_tier3_full.sh`): all **52 cosmologies c130–c181 ×
  100 maximin HODs × 3 iters** = 5200 runs → `data/fullbox_tier3/` (pilot's 500
  reused — maximin is prefix-stable). Submitted as one throttled **job array**
  `54892798` (1-4700%500). **Ran on the `overrun` qos** because forero's *per-user*
  desi balance (~535 node-hr) is far below the ~4700 node-hr cost (repo has 174k —
  it's a personal cap); overrun is free/low-priority/preemptible and `--requeue` +
  per-task skip-logic make it safe. Resumable: re-run `launch_tier3_full.sh`.
  Expect days–weeks. When done: rebuild cache → `run_emulator_tier3.sh` → review.

### Tier-3 diagnostics: campaign PAUSED pending floor investigation (2026-06-23)

Before letting the campaign run, two pilot diagnostics + a scale study (all on
existing data). **The campaign array `54892798` was CANCELLED** (user decision) — it
never scheduled on overrun, so 0 new runs; `fullbox_tier3/` still has the 500 pilot.
Resume with `launch_tier3_full.sh` once the floor below is understood.

- **Leg-ordering bug fixed:** `select_targets` built blocks in PRIMARY_STEMS order
  while `Y[:,mask]` is in build order → all 16 leg plots were mislabeled (headline
  per-cosmology numbers were fine). Now follows column order.
- **Learning curve vs #training cosmologies** (`emulator_tier3_learning_curve.py`):
  FLAT at ~40–70× CV from N=2..9 (robust medians; low-N has extrapolation blowups).
  But the 10 pilot cosmologies are spread every-5th, so this can't probe the DENSITY
  the full campaign adds — weak proxy.
- **Within-cosmology HOD-interp floor** (`emulator_tier3_within_cosmo.py`): 5-fold
  within each cosmology (zero cosmology extrapolation) → **~8× CV** floor (priority
  7–11×, very uniform across all 10). **Two floors:** (a) cosmology coverage IS a
  lever — the 8→40-70× gap is the extrapolation penalty the campaign would close;
  (b) a residual ~8× CV floor remains even with zero extrapolation (NOT
  iteration-limited) → campaign as configured (15 bins, 100 HODs) likely asymptotes
  ~8× CV ≈ 64× C_CV in variance = NOT inference-grade. Candidate causes: HOD-sample
  (40-50 HODs in 12-D is thin — testing via c000 +50 HODs → `emulator_c000_hod_curve.py`),
  coarse 15-bin data, or MLP architecture.
- **Most-informative scales** (`scale_information.py`, all 19 cosmologies):
  **monopole cosmology info is small-scale** (peaks ~26–55, ~50% by s≈30–55, matches
  the vector-search finding); **quadrupole info is large-scale** (~105–145, Kaiser).
  Small scales carry the highest cosmology S/N *and* the highest HOD nuisance (the two
  coincide) — which is exactly why the emulator is needed there and where the ~8×
  floor bites. Argues for **finer binning at s≲50–60 in the monopoles**. (First
  cumulative metric Tr(C⁻¹Σ) was ill-conditioned → replaced with diagonal cumulative
  Fisher; caveats: broad mixed prior, diagonal approx, 15 coarse bins.)
- **c000 +50 HODs** (`select_extra_hods.py` → 100 total, array on regular qos): the
  HOD-count test of whether the ~8× floor is HOD-sample-limited — run
  `emulator_c000_hod_curve.py` once the 100 runs land.

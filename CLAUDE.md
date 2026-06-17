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
│   └── compute_hod_derivatives.py← Tier-1: regress ∂ξ/∂θ_HOD, subtract HOD contamination
├── queue/
│   ├── run_single_box.sh         ← sbatch wrapper, single-box pipeline
│   ├── run_subboxes.sh           ← sbatch wrapper, subbox pipeline
│   ├── run_subboxes_cosmo.sh     ← sbatch wrapper taking <cosmo> <hod> arguments
│   ├── run_fullbox_cosmo.sh      ← same for the full-box pipeline (full CPU node)
│   ├── launch_fisher_subboxes.sh ← submits all nine Fisher (cosmo, hod) subbox runs
│   ├── launch_fisher_fullbox.sh  ← submits all nine full-box runs
│   └── launch_hod_calibration.sh ← Tier-1: submits the selected c000 HOD-calibration full-box runs
├── data/             ← pipeline output; subdirs {cosmo}_hod{NNN}/, fullbox/, derivatives/
├── plots/            ← figure output; mirrors the data/ layout
├── notes/            ← LaTeX technical notes (zero_crossing/)
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
| `run_fullbox_cosmo.sh`  | 8:00:00 | 256 | **0:24–0:27** (3 iterations) |

- The **subbox limit is tight** — ~30–45 min of margin against 4 h. Bump to
  `-t 5:00:00` on reruns, or raise `-c` (it uses only 8 of the node's 256 cores
  while `regular` qos charges the whole node anyway).
- The **full-box limit is hugely over-provisioned** — 8 h requested for ~25-min
  jobs; `-t 1:00:00` is plenty and schedules faster. Full-box uses all 256 cores,
  hence ~8× faster than the 8-core subbox runs despite covering the whole box.

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

### Tier 1 — HOD-contamination calibration (set up 2026-06-16)

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

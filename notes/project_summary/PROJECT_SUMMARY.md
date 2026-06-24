# ASTRA clustering — project summary

A self-contained synthesis of the work in this repository: context, the math, the
ASTRA simulations, design recommendations (with justifications) for the **Fisher**
and the **emulator**, what is most publishable, and what to do next.

---

## 1. Context

**ASTRA** classifies each galaxy by its local cosmic-web environment using a Delaunay
tessellation over the combined data+random catalog, then measures the two-point
correlation function (2PCF) *per environment quantile*. The scientific bet: splitting
clustering by environment (voids → knots) carries cosmological information **beyond the
plain 2PCF**, because different environments respond differently to cosmology vs. to
galaxy–halo (HOD) physics, breaking degeneracies.

We pursue this on two tracks:
1. **Fisher forecast** — local derivatives of the statistics → forecasted parameter errors.
2. **Emulator + inference** — a forward model ξ(θ) trained on simulations → *measured*
   constraints via MCMC (SUNBIRD-style).

The hard constraint throughout: **no new HOD generation** — we reuse existing
HOD-populated boxes (single phase, ph000).

---

## 2. Basic math

**Environment classification.** For each object, from data/random neighbour counts in
the Delaunay tessellation,
> r = (n_data − n_rand) / (n_data + n_rand),  r ∈ [−1, 1],

binned into quantiles Q1 (void) … Q4 (knot). The 2PCF multipoles ξ_ℓ(s) (ℓ = 0 monopole,
2 quadrupole) are measured for each *leg*: full-sample auto, quantile autos, and
full×quantile crosses, for data and for randoms.

**Fisher matrix.**
> F_ab = Σ_bins (∂ξ/∂θ_a) C⁻¹ (∂ξ/∂θ_b),  σ(θ_a) = √[(F⁻¹)_aa],  FoM = 1/√det(F⁻¹).

Quality is set entirely by the **derivatives** ∂ξ/∂θ and the **covariance** C.

**Emulator + inference.** A neural emulator μ(θ) = ξ(θ_cosmo, θ_HOD); Gaussian likelihood
> χ²(θ) = (d − μ(θ))ᵀ C_tot⁻¹ (d − μ(θ)),  C_tot = C_CV + C_emu (+ C_label),

with C_CV the cosmic variance, C_emu the **emulator-error covariance**, C_label the
measurement (iteration) noise of the data vector. MCMC over θ gives the posterior.

**Key diagnostic metrics.**
- **RMS/CV** — emulator prediction error in units of the cosmic-variance σ; <1 = inference-grade.
- **value-add** = σ(2PCF) / σ(2PCF + ASTRA legs) — the tightening from environment splits.
- **C_emu/C_CV** — emulator error relative to cosmic variance; sets whether a leg's signal is usable.

---

## 3. The ASTRA simulations

- **Boxes:** AbacusSummit, 2 Gpc/h per side (native), HOD-populated at z=0.5 with the
  yuan23 prior (ntbfin catalogs), **single phase ph000**. ~3–5 M galaxies/box. After
  Alcock–Paczynski rescaling to a common frame the effective side varies per cosmology
  (~1.87–2.18 Gpc/h, anisotropic); ξ(s) is measured in that common AP frame.
- **RSD & AP (how they enter the runs):** both are applied at load time, from fields
  already in the catalogs. **RSD** — the line of sight is `z`, and we use the precomputed
  redshift-space coordinate `Z_RSD` (which carries the peculiar-velocity displacement
  v_z/aH) for the z-axis while `X_PERP`/`Y_PERP` stay real-space; this populates the
  quadrupole. **AP** — we divide the transverse coordinates by the header factor `Q_PERP`
  (∝ the D_A ratio) and the line-of-sight by `Q_PAR` (∝ the H ratio), mapping each
  cosmology's box into the common fiducial frame. This anisotropic rescaling is why the
  effective box side differs per cosmology and is what makes w0, wa observable through the
  quadrupole.
- **Cosmologies.** (i) A *Fisher grid* — c000 (Planck ΛCDM) + neighbours varying the 4
  ΛCDM params (ω_b, ω_c, n_s, σ₈) at ±2–3%. (ii) An *emulator block* c130–c181 — the
  AbacusSummit emulator Latin hypercube, varying the broader **w₀wₐCDM+** set: 8 params
  (ω_b, ω_c, h, n_s, α_s, N_ur, w₀, wₐ); A_s fixed, σ₈ derived.
- **HODs.** 500 draws/cosmology exist; we use ~20–100 (maximin-selected).
- **Pipeline.** Full-box (periodic BC, the emulator target) and 500 Mpc/h sub-boxes
  (the covariance estimate, pooled → ~576 samples). Each run averages **N ASTRA
  iterations** (independent random realisations; N=3 in the pilot). Data product:
  17 stems × 2 multipoles × 15 s-bins (0–150 Mpc/h) per run.
- **Status.** All 52 emulator cosmologies × ~20–50 HODs and the 9 Fisher cosmologies ×
  50 HODs are on disk (c000 at 100); the campaign run filled the cosmology axis 10→52.

---

## 4. Recommendations — Fisher matrix design

**F1. Take derivatives from the emulator at fixed HOD, not from the ± cosmology pairs.**
*Justification:* the ±2–3% AbacusSummit pairs are not HOD-matched, so HOD mismatch
contaminated ∂ξ/∂θ at the level of the cosmology signal — the σ₈ derivative came out
~11% of the expected 2ξ and needed the whole Tier-0/1 noise-model + subtraction saga.
The emulator gives ∂ξ/∂θ at *fixed* HOD → HOD-clean by construction, and over all **8**
w₀wₐCDM+ params (the ± grid only gave 4 ΛCDM).

**F2. Use the realistic covariance C = C_CV + C_emu, not C_CV alone.**
*Justification:* with C = C_CV (perfect-emulator) the Fisher crowns the environment
crosses; but the value-add test showed that in *real* inference the environment-leg
C_emu (~3–7× CV) down-weights exactly those legs, and the value-add evaporated (×1.0).
A Fisher with C_CV only is therefore not predictive of achievable constraints.

**F3. Marginalise the HOD jointly, with emulator gradients for both blocks.**
*Justification:* ∂ξ/∂θ_HOD comes from the same emulator → a consistent (8+12)-param
Fisher. The MCMC showed HOD-marginalisation inflates ΛCDM errors only ×1.0–1.8 (ω_b, ω_c
HOD-robust; h, n_s mild) — far less than the 2–10× the old gradient+prior approach feared.

**F4. Validate the Gaussian-linear Fisher against the MCMC; use DALI where it breaks.**
*Justification:* the curated-vector MCMC recovers c000 unbiased and ~Gaussian (Fisher
reliable there), but tier3/w₀wₐ recoveries are biased 2–4σ (non-Gaussian/extrapolation)
— so Fisher σ must be cross-checked, not trusted blindly.

> Implementation: `emufisher_{lib,build,forecast,campaign,validate}.py`; note
> `notes/fisher_emulator/`. (Forecast results being finalised.)

---

## 5. Recommendations — emulator design

**E1. Train with a covariance-weighted loss (1/C_CV), not iteration-noise weighting.**
*Justification:* the arch sweep showed the iteration-noise loss *starved* the
environment legs (cross-cosmology RMS/CV ~31); CV-weighting cut that to ~2 — a **~15×**
accuracy gain on exactly the legs that gate the value-add. Per-leg networks added
nothing beyond the weighting. (Adopted in `train_emulator(cv=)`.)

**E2. Curate the data vector; don't dump all legs.**
*Justification:* the greedy ranks the **void×full-sample-cross monopole as the #1 leg**,
ahead of the plain 2PCF. The curated vector (full 2PCF + void-random-cross monopole)
gives a *real* value-add (σ ×1.2–1.7) and an unbiased c000 recovery (<0.5σ), whereas
adding all six environment legs indiscriminately gave only ×1.1 (noisy legs dilute).

**E3. Cosmology coverage is the dominant lever.**
*Justification:* filling the hull 10→52 cosmologies dropped the broad-prior monopole
LOCO from ~18× to ~3× CV; the residual failures and the tier3/w₀wₐ recovery biases are
8-D **hull-corner extrapolation**, which only denser coverage fixes.

**E4. ASTRA iterations are a *minor* lever — N≈10 is plenty, not 25.**
*Justification:* a zero-compute test (propagating the stored per-iteration scatter as
σ₁/√N) showed the **quadrupole iteration noise is already ~0.2–0.4× CV at N=3**, i.e.
~10% of the quad emulator error — so more iterations barely move C_emu. (The earlier
"quads need ~25 iters" used noise/*spread*, not the inference-relevant noise/*CV*.)

**E5. HODs are saturated by ~30/cosmology.**
*Justification:* the c000 HOD learning curve plateaus by K≈30 and is flat to K=70; the
HOD response is effectively low-dimensional and pooled across cosmologies. More HODs ≈
diminishing returns; HOD-marginalisation is already cheap.

**E6. Campaign design: coverage-first, ~10 iterations, ~30–40 HODs** — *not* the original
120-HOD/high-iteration plan. *Justification:* E3–E5; same compute, far better spent on
the cosmology axis (corners) than on saturated HODs or already-adequate iterations.

**E7. Monopoles are inference-grade now; quadrupoles (→ w₀, wₐ) are coverage-limited.**
*Justification:* the full-2PCF + data/random monopoles emulate at ~1× CV near LCDM; the
ΛCDM recovery is clean today. w₀/wₐ rely on the quadrupole, whose bias is set by
coverage, so dark energy waits on the denser campaign.

---

## 6. What is most publishable / interesting

1. **The value-add is emulator-gated, not information-limited** — a clean, general
   methodological result: ASTRA (and any environment/marked statistic) carries real
   cosmological information (Fisher: 1.3–2.4×/param), but it is only realisable in
   inference once the emulator error on those legs falls below cosmic variance. This
   reframes "does the statistic help?" as "can we emulate it accurately enough?", with a
   quantitative target (C_emu ≪ CV).
2. **A simple, curated ASTRA vector beats the 2PCF today** — full 2PCF + the single
   void×full-sample-cross monopole tightens ΛCDM by ×1.2–1.7 with an unbiased recovery
   and essentially free HOD marginalisation. Concrete, reproducible, attractive headline.
3. **The CV-weighted-loss fix** — a ~15× emulator-accuracy gain on the environment legs
   from changing the training loss to the inference metric; broadly useful for SBI emulators.
4. **The lever analysis** — coverage ≫ iterations ≈ HODs, each backed by a targeted test
   (LOCO learning curve, quad-noise 1/√N, HOD plateau). Useful guidance for any
   simulation-based-inference campaign budget.
5. **Honest emulator-aware Fisher** — replacing C_CV with C_CV + C_emu and HOD-clean
   emulator derivatives; reconciles the optimistic Fisher with real MCMC.

**Most publishable single paper:** an emulator-based ASTRA methodology paper — "*Environment
clustering for cosmology is emulator-gated: a curated void-cross + 2PCF vector and the
C_emu target*" — combining (1), (2), and (3), with the lever analysis (4) as the campaign
justification.

---

## 7. Recommended next steps

1. **Finish the emulator-based Fisher** (`emufisher_*`, running): the forecast σ table
   over data vectors, the Fisher-vs-MCMC validation, and the campaign FoM-vs-C_emu curve.
2. **Run the corrected campaign:** all 52 c130–c181 cosmologies, **densify the hull
   corners**, ~10 iterations, ~30–40 HODs — to drive the environment-leg C_emu toward
   ≲0.1× CV (the greedy shows that unlocks ~5× FoM and the full value-add).
3. **A neighbourhood-matched C_emu** so broad-prior errors are not inflated by distant
   hull-corner cosmologies (cheap, sharpens every broad recovery).
4. **An external, independent-phase mock suite** (e.g. EZmocks) for an honest covariance
   and *unbiased* real recoveries — the main residual weakness under no-new-HOD.
5. **Dark energy:** once coverage is dense, revisit w₀/wₐ with the quadrupole (and the
   higher-iteration runs) — the only directions still coverage-limited.
6. **Production emulator:** an MLP (SUNBIRD-style) on the full campaign with the curated
   vector + CV-weighted loss; MCMC on held-out mocks; compare measured σ to this Fisher.

*(Full technical detail and figures: `notes/tier3_emulator/mlp_emulator_note.pdf` (emulator
+ inference) and `notes/fisher_emulator/fisher_emulator_note.pdf` (emulator-based Fisher);
historical Fisher/vector-search in `notes/fisher/`, `notes/vector_search/`.)*

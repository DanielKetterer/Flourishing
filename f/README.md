# gfs_f: pre-registered analysis pipeline for the f construct on the GFS

Implements the full inferential machinery of "A Preregistered Structural,
Predictive, and Mechanistic Evaluation of the General Factor of Human
Flourishing" (protocol v8, two-stage lock) in Python: H1 structural
replication with permutation envelopes and pre-registered comparators, H2a
per-factor alignment with trustworthiness tiers and its Monte Carlo
validation harness, H2b cultural-distance falsification, H3 LODO predictive
validity beyond a measurement-equivalent psychopathology composite, and the
H4 latent-vs-mutualism propagator decomposition with the MIDUS-anchored
two-stage estimator and H4a calibrated cross-fitting.

## Layout

```
gfs_f/
  config.py        every threshold and the variable map (Stage 2 lock target)
  polychoric.py    weighted polychorics + asymptotic variances (DWLS weights)
  cfa.py           bifactor DWLS engine, ML/BIC/Vuong, multigroup, scores
  synthetic.py     two-channel mixture DGP (smoke tests, H2a MC, H4 MC)
  data_io.py       loading, orientation, weights, MI (m = 20), lookups
  h1_structure.py  H1 legs i-iv, envelopes A/B, SWB/S-1/MAGNA comparators
  h2_invariance.py H2a alignment + tiers + MGCFA + MC harness; H2b model
  h3_predictive.py LODO f, p composite, Croon regression, RI-CLPM, FDR
  h4_dynamics.py   B_obs identity, anchors, Stage 1/2, H4a, cross-checks
  plots.py         figures per hypothesis
  report.py        markdown summary with the flags section
  run_all.py       CLI orchestrator
```

## Running

Machinery check on synthetic GFS-format data (what the code deposit should
show it can do, before any data access):

```
python -m gfs_f.run_all --synthetic --smoke --out runs/smoke
```

Production, once the released files are on disk and the Stage 2 lock is done:

```
python -m gfs_f.run_all --data-dir /path/to/gfs --out runs/production
```

The synthetic DGP takes `--w-true` and `--mu-true`; because its
cross-sectional structure is w_L-invariant by construction, recovery of
w_true by `w_corrected` is a direct end-to-end test of the H4 identification
strategy, not a fit exercise.

## Stage 2 lock checklist (edit, commit, deposit the diff)

1. VARMAP names in `config.py` against the released codebook (osf.io/cg76b).
   COS distributes a perturbed Sweden sample data file for pre-registration
   script preparation; lock names AND file layout (`DATA_LAYOUT`,
   `WIDE_SUFFIX`, per-wave weight names) against it now rather than waiting.
   Placeholders are marked `placeholder=True` and listed by
   `validate_varmap()`.
2. Final indicator list (12-17), the H3 subjective-economic-position
   outcome variable, and the p/f item-disjointness cross-check
   (`validate_varmap` asserts it).
3. Envelope B gap sign: config.H1['envelope_gap_sign'] against the
   authoritative v8 gap definition. See the flags section of any run's
   summary.md for why the omega_h/ECV direction plausibly inverts.
4. MIDUS Lambda anchor file for H4 (currently the H1 pooled f loadings
   stand in for the anchor direction).
5. Pre-flight diagnostic bands, H4b Tier 2/3 split, H4a R2_min, and the
   H2a flag constants (run `run_h2a_mc` at full settings; that output is
   the pre-registered evidence base for the lock).
6. Lookups: `lookups/cultural_distance.csv` (Muthukrishna CFST from US),
   `iw_scores.csv`, `income_tier.csv`, `religion.csv`,
   `translation_equivalence.csv` (Cowden 2024). Templates are written by
   every run; H2b falls back (loudly) without them.

## Parity checks against R (Stage 2, on released data)

* lavaan WLSMV: point estimates should match `fit_dwls` (same estimator);
  CFI/RMSEA here are polychoric-ML approximations of the robust indices --
  SRMR, loadings, omega_h, ECV, factor scores, and all relative decision
  quantities (gaps, delta-BIC, cosines) are exact.
* Alignment: Mplus or R `sirt::invariance.alignment` cross-check; this
  implementation follows Asparouhov-Muthen 2014 (component loss
  (x^2+.01)^(1/4), sqrt(N g N g') weights, FIXED identification).
* MAGNA proper in R (psychonetrics) vs the EBIC graphical-lasso GGM here.
* RI-CLPM and the H3 joint SEM in lavaan vs the Croon-corrected regression.

## Notes

* MI is m = 20 with per-country imputation in production; envelopes run on
  imputation 1 (documented; headline statistics are Rubin-pooled).
* All survey weights are within-country normalized; PSU clustering enters
  through the H3 bootstrap and the H4 wild cluster bootstrap at country
  level per the prereg.
* Every constant carries a comment naming its prereg section. Nothing
  downstream hard-codes a threshold.

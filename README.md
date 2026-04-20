## **Flourishing**  

## **Devoted to the mathematics and science of human flourishing.**

This repository contains the complete research code, data pipeline, and manuscript for *Toward f: A General Factor of Human Flourishing* (Daniel Ketterer, 2026). The project establishes a general factor of wellbeing (*f*) — analogous to Spearman’s *g* in intelligence — using individual-level MIDUS data, then tests its predictive power and formalizes it as a dynamical system of cross-domain mutualism.

## Core Idea

Wellbeing is not a checklist of separate domains or a single summary score. It is **systemic coupling strength** — the degree to which the domains of a person’s life mutually reinforce one another and recover from disruption. When autonomy supports purpose, purpose deepens relationships, and relationships strengthen self-acceptance, the system becomes more than the sum of its parts: it becomes resilient.

This view has direct policy implications. Interventions that strengthen *cross-domain coupling* (e.g., programs that simultaneously reduce financial stress, improve health access, and preserve autonomy) generate compounding returns that single-outcome evaluations cannot detect.

## Six Domains (Ryff’s Psychological Well-Being Model)
All analysis and the Flourishing Map use the six subscales measured in MIDUS:
1. **Autonomy** — self-determination, resistance to social pressure  
2. **Environmental Mastery** — competence in managing daily life  
3. **Personal Growth** — continued development and openness  
4. **Positive Relations** — warm, trusting interpersonal bonds  
5. **Purpose in Life** — sense of direction and meaning  
6. **Self-Acceptance** — positive regard for self and past  

These yield 15 pairwise couplings that the *f* framework treats as the core structure of flourishing.

## Key Empirical Results (from MIDUS 2 & longitudinal panel)

**Measurement (Positive Manifold & Bifactor CFA)**  
- 6 Ryff subscales (N≈4,026 complete cases): all 15 correlations positive (mean *r* = 0.397). PC1 explains **50.4%** of variance.  
- **17 indicators** from 6 independent instrument families (Ryff, Keyes, Watson, Diener, Rosenberg, Pearlin, Scheier, McAdams): all 136 correlations positive (mean *r* = 0.406). PC1 explains **45.6%** of variance (PC1/PC2 ratio = 5.44).  
- Bifactor CFA (Ryff-only): ω<sub>h</sub> = **0.789**, ECV = **0.758**.  
- **Multi-instrumental bifactor** (17 indicators, 4 orthogonal group factors): ω<sub>h</sub> = **0.893**, ECV = **0.738**, FDC = **0.973** (>0.90 threshold for usable scores). Instrument-specific ω<sub>hs</sub> values are negligible (<0.04).  
- *f* is **not** merely the absence of pathology: *r*(f, negative affect) = **−0.439** (only 19.3% shared variance). Big Five explain 47.3% of *f* variance; 52.7% is independent. Clinical discrimination is large (depression *d* = 0.978, GAD *d* = 1.503).

**Prediction (Longitudinal Economic Validation)**  
Wave 2 *f* (Ryff PC1) predicts Wave 3 outcomes ~9 years later (N≈2,894 panel):  
- Log income: *r* = **0.122**, regression β = **0.083** (*p* < 0.001) controlling for baseline income, age, sex, education.  
- Employment: *r* = **0.113**, LPM β = **0.015** (*p* = 0.018).  
- Test-retest stability of *f*: *r* = **0.674**.  
- **Incremental validity**: *f* remains significant after controlling for negative affect (income β = 0.067, *p* = 0.002); negative affect does not predict beyond *f*.

**Mechanism (Dynamical Systems)**  
- Primary model: **Generalized Lotka-Volterra (GLV)** calibrated exactly via Lyapunov inversion (*J* = −½ Σ⁻¹) from individual-level covariance. Reproduces the observed positive manifold and dominant PC1 without any built-in latent factor.  
- Logistic model retained for pedagogical comparison only.  
- Coupling-resilience signature, asymmetric coupling budget, perturbation decomposition, and per-person resilience scores are all implemented and visualized.

**Capabilities Framework**  
Individual-level Alkire-Foster analysis shows 58.5% of respondents fall below the 25th-percentile threshold on at least one domain (weakest: Autonomy and Purpose in Life).

## Repository Contents
- **`Multi_Instrumental Bifactor Model of f.py`** — Full 17-indicator bifactor CFA, omega/ ECV / FDC, factor scoring (MAP + Bartlett + Thurstone), external validation.  
- **`combined_f_analysis_2.py`** — Complete pipeline: PCA, bifactor CFA, **GLV dynamical system** (Lyapunov calibration), asymmetric coupling, perturbation series, Lyapunov stability, resilience, Alkire-Foster, coupling sweep, economic validation.  
- **`longitudinal_f_prediction.py`** — Wave 2 → Wave 3 economic prediction (income, employment, disability days), incremental validity over negative affect, figures.  
- **`toward_f_v3.pdf`** — Full manuscript (14 pages).  
- **`spectral_equivalence.html`** — For any symmetric, stable Jacobian $  J  $ in a multivariate Ornstein–Uhlenbeck process, the dominant dynamical mode—the eigenvector of $  J  $ belonging to its least-negative eigenvalue—is exactly the first principal component of the stationary covariance matrix $  \Sigma  $. In other words, the statistical factor $  f  $ recovered from wellbeing data is identically the slowest-decaying collective response pattern of the underlying coupled system.

## Author
**Daniel Ketterer**  
M.S. Mathematics, Wright State University  

- [GitHub](https://github.com/DanielKetterer)  
- [LinkedIn](https://www.linkedin.com/in/daniel-ketterer-math/)

## License
Research code, data pipeline, and manuscript are shared for academic and educational purposes under a permissive license.

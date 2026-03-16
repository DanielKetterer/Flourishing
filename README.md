# Flourishing

**Devoted to the study of the mathematics and science of human flourishing.**

This repository contains the research code, paper, and public-facing tools for *Toward f: A General Factor of Human Flourishing* — a project that formalizes a general factor of wellbeing (*f*), analogous to Spearman's *g* for intelligence, and reinterprets it through dynamical systems theory.

## The core idea

Most approaches to measuring wellbeing treat it as a score on a single scale or a checklist of separate domains. *f* proposes a different view: flourishing is best understood as **systemic coupling strength** — the degree to which the domains of a person's life reinforce one another and recover from disruption.

A life where autonomy supports purpose, purpose deepens relationships, and relationships strengthen self-acceptance isn't just *better* — it's *more resilient*. The connections between domains matter more than any single domain's level.

This has direct implications for policy evaluation. An intervention that strengthens cross-domain coupling (like Medicaid expansion, which simultaneously affects health access, financial stress, and care avoidance) should produce compounding returns invisible to standard single-outcome measurement.

### Interactive tool

| File | Description |
|------|-------------|
| `index.html` | **[Flourishing Map](https://danielketterer.github.io/Flourishing/)** — a free, phone-friendly self-assessment that maps the connections between six domains of psychological well-being. No scores, no accounts — just a visual picture of what's feeding what in your life. |

## Six domains of psychological well-being

The Flourishing Map and the empirical analysis use Carol Ryff's six-dimensional model of psychological well-being, measured in the MIDUS (Midlife in the United States) study:

1. **Autonomy** — Independence of thought, self-determination, evaluating yourself by personal standards rather than external pressure
2. **Environmental Mastery** — Competence in managing everyday life, making effective use of opportunities, choosing and shaping contexts that fit your needs
3. **Personal Growth** — Continued development, openness to new experience, realizing your potential over time
4. **Positive Relations with Others** — Warm, trusting interpersonal bonds, empathy, affection, capacity for deep connection
5. **Purpose in Life** — Sense of direction, goals and beliefs that give life meaning, feeling that past and present life have purpose
6. **Self-Acceptance** — Positive attitude toward yourself and your past, acknowledging and accepting your full range of personal qualities

These six domains yield 15 pairwise connections — the coupling structure that the *f* framework treats as the core of flourishing.

## Key empirical results

- **Bifactor CFA** (MIDUS 2, Ryff PWB subscales): ω_h = 0.405, ECV = 0.417 — the general factor captures meaningful shared variance while the majority remains domain-specific
- **Mutualism model**: Coupled logistic ODEs calibrated from individual-level partial correlations reproduce the positive manifold and dominant first principal component without any latent factor built in
- **Representation coherence**: Weak coherence (cosine similarity) = 0.993, medium coherence (Jacobian alignment) = 0.961 — the statistical and dynamical models describe the same structure
- **Coupling-resilience tradeoff**: Tighter coupling → faster recovery from small shocks but catastrophic vulnerability to large ones. This is the most policy-relevant and testable prediction.
- **Dual continua validation**: *f* correlates with negative affect at only r = −0.379 (14.4% shared variance), confirming flourishing is not simply the absence of distress
- **Individual-level Alkire-Foster capabilities**: 58.5% of individuals fall below threshold on at least one domain. Weakest domains: Autonomy (24.6%), Purpose in Life (21.3%).

## Methods and tools

The analysis pipeline is implemented from scratch in Python (NumPy, SciPy, pandas, statsmodels, matplotlib). Econometric methods implemented in the companion [PracticalMath](https://github.com/DanielKetterer/PracticalMath) repository include OLS, IV/2SLS, difference-in-differences, regression discontinuity, panel fixed effects, and Callaway–Sant'Anna estimators.

## Author

**Daniel Ketterer**
M.S. Mathematics, Wright State University

- [GitHub](https://github.com/DanielKetterer)
- [LinkedIn](https://www.linkedin.com/in/daniel-ketterer/)

## License

Research code and paper are shared for academic and educational purposes. The Flourishing Map interactive tool is free to use and share.

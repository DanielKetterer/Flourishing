"""
config.py -- Single source of truth for the GFS f-construct analysis.

Everything that the preregistration ("A Preregistered Structural, Predictive,
and Mechanistic Evaluation of the General Factor of Human Flourishing",
protocol v8) fixes in advance lives here as a named constant, annotated with
the prereg section it comes from. Everything that the Stage 2 pre-data-access
addendum must confirm against the released codebook is marked STAGE2-LOCK.

Design contract:
  * No module hard-codes a GFS variable name, a threshold, or a decision
    constant. They all import from here.
  * The Stage 2 lock is performed by editing VARMAP / STAGE2 below against the
    released codebook (osf.io/cg76b) or the perturbed Sweden sample file,
    committing, and depositing the diff with the addendum.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# 1. VARIABLE MAP  (STAGE2-LOCK: confirm every name against released codebook)
# ----------------------------------------------------------------------------
# GFS ships per-wave files (or a long file with a WAVE column). data_io.py
# supports both layouts; set DATA_LAYOUT accordingly.
#
# Names below follow the GFS codebook naming style used in the published
# Wave 1 analysis code (e.g., ANNUAL_WEIGHT1, STRATA, PSU, COUNTRY). Item
# names marked '??' are best-guess placeholders and MUST be corrected at the
# Stage 2 lock. The pipeline only ever touches df[VARMAP[...]], so the lock
# is a pure config edit.

DATA_LAYOUT = "long"          # "long": one row per (ID, WAVE); "wide": suffixed
WIDE_SUFFIX = "_W{wave}"      # used only if DATA_LAYOUT == "wide"

ADMIN = dict(
    id="ID",
    wave="WAVE",
    country="COUNTRY",
    weight="ANNUAL_WEIGHT1",   # Gallup within-country annual weight (W1 name)
    weight_by_wave={1: "ANNUAL_WEIGHT1", 2: "ANNUAL_WEIGHT2",   # STAGE2-LOCK
                    3: "ANNUAL_WEIGHT3"},
    strata="STRATA",
    psu="PSU",
)

# --- f indicators -----------------------------------------------------------
# Prereg (Scale/Index Aggregation): "A bifactor model on the 12-to-17 GFS
# flourishing indicators ... one general factor f plus six group factors ...
# The two GFS hedonic items (happiness, life satisfaction) load on f only."
#
# The 12-item SFI supplies exactly 2 items per VanderWeele domain, but the
# prereg splits health into separate mental / physical group factors, which
# leaves each with a single SFI item. The extra annual-survey wellbeing items
# below bring every group factor to >= 2 indicators. Which extras exist under
# which names is a STAGE2-LOCK decision; placeholders are marked.
#
# 'reverse=True' => higher raw value is WORSE; data_io re-orients so that
# higher always means more flourishing before any model sees the data.

@dataclass(frozen=True)
class Item:
    var: str          # codebook variable name (STAGE2-LOCK)
    domain: str       # one of DOMAINS or "hedonic"
    reverse: bool = False
    n_cat: int = 11   # 0-10 response scale unless noted
    placeholder: bool = False   # True => name not yet codebook-confirmed

F_ITEMS: list[Item] = [
    # hedonic pair: loads on f only (prereg primary specification)
    Item("HAPPY",          "hedonic"),
    Item("LIFE_SAT",       "hedonic"),
    # mental health
    Item("MENTAL_HEALTH",  "mental"),
    Item("PEACE",          "mental",   placeholder=True),   # 'at peace with thoughts/feelings'
    # physical health
    Item("PHYSICAL_HLTH",  "physical"),
    Item("HEALTH_LIMIT",   "physical", reverse=True, placeholder=True),  # health limits activities
    # meaning and purpose
    Item("WORTHWHILE",     "meaning"),
    Item("LIFE_PURPOSE",   "meaning"),
    # character and virtue
    Item("PROMOTE_GOOD",   "character"),
    Item("GIVE_UP",        "character"),
    # close social relationships
    Item("SAT_RELATNSHP",  "social"),
    Item("LONELY",         "social",   reverse=True, placeholder=True),
    # financial and material stability (SFI items are worry items -> reverse)
    Item("EXPENSES",       "financial", reverse=True),
    Item("WORRY_SAFETY",   "financial", reverse=True),
]

DOMAINS = ["mental", "physical", "meaning", "character", "social", "financial"]
HEDONIC = [it.var for it in F_ITEMS if it.domain == "hedonic"]

# --- psychopathology composite p (prereg: GFS depression + anxiety items) ---
# PHQ-4 style: 2 depression + 2 anxiety, 4-category frequency response.
# Prereg: bifactor (general p + dep + anx group factors); with 2-item group
# factors the model uses within-pair tau-equivalence constraints; if < 3
# indicators available the composite reverts to single-factor (downgrade
# reported). Item-level disjointness from f's mental items is asserted in
# data_io.validate_varmap() -- part of the Stage 2 codebook cross-check.
P_ITEMS: list[Item] = [
    Item("DEPRESSED_FREQ", "dep", n_cat=4, placeholder=True),  # down/depressed/hopeless
    Item("INTEREST_LOSS",  "dep", n_cat=4, placeholder=True),  # little interest or pleasure
    Item("ANXIOUS_FREQ",   "anx", n_cat=4, placeholder=True),  # nervous/anxious/on edge
    Item("WORRY_CONTROL",  "anx", n_cat=4, placeholder=True),  # cannot stop worrying
]

# --- H3 outcomes and their LODO domain d(k) ---------------------------------
# Prereg: f_(-physical) for self-rated health change, f_(-social) for
# relationship satisfaction change, f_(-financial) for subjective economic
# position change. Wave1 -> Wave2 primary; Wave1 -> Wave3 extension.
H3_OUTCOMES = {
    "self_rated_health":    dict(var="PHYSICAL_HLTH",  lodo_domain="physical"),
    "relationship_sat":     dict(var="SAT_RELATNSHP",  lodo_domain="social"),
    "subj_econ_position":   dict(var="SUBJ_INCOME",    lodo_domain="financial",
                                 placeholder=True),    # 'getting by on income' item; STAGE2-LOCK
}
H3_CONTROLS = dict(age="AGE", sex="GENDER", education="EDUCATION_3")  # STAGE2-LOCK names

# ----------------------------------------------------------------------------
# 2. PRE-REGISTERED THRESHOLDS (Stage 1: locked now, verbatim from prereg)
# ----------------------------------------------------------------------------

H1 = dict(
    # Inference Criteria, confirmation leg (i)
    omega_h_confirm=0.80,
    omega_h_partial_lo=0.65,       # partial tier: omega_h in [0.65, 0.80)
    # NOTE: the Hypotheses section states omega_h >= 0.75 while the Inference
    # Criteria confirmation tier states >= 0.80. The decision rules here follow
    # the Inference Criteria (authoritative for verdicts); the 0.75/0.80
    # discrepancy is flagged in report.py so it can be reconciled in v8 text.
    ecv_min=0.65,                  # Hypotheses section
    cfi_min=0.90, rmsea_max=0.08, srmr_max=0.08,
    # leg (ii): permutation envelopes
    gap_omega=0.05, gap_ecv=0.05,
    envelope_reps=200,             # prereg: 200 replicates per envelope
    envelope_quantile=0.95,
    # DIRECTIONALITY NOTE (see README + h1_structure.py): under Envelope B
    # (partition permutation) misassigned domain variance can only be absorbed
    # by f, which pushes omega_h/ECV UP and group-loading magnitude/fit DOWN
    # relative to the true partition. 'Exceed the envelope by delta' therefore
    # has an unambiguous sign for fit and loading magnitude but plausibly an
    # inverted sign for omega_h/ECV. Both directions + percentile placements
    # are always computed; the sign used by the verdict is set here and is a
    # STAGE2-LOCK item to reconcile against the authoritative v8 definition.
    envelope_gap_sign=dict(omega_h=-1, ecv=-1, cfi=+1, group_load=+1),
    # leg (iii)
    hedonic_cosine_min=0.95,
    loading_sd_floor=0.10,         # Eid uniform-loading non-identification floor
    # leg (iv)
    dbic_min=10.0,
    swb_cosine_max=0.90,
    ari_confirm=0.60, ari_partial_lo=0.40,
    heywood_floor=0.001,           # prereg: Heywood cases constrained to 0.001
    magna_holdout=0.20,            # cross-validated predictive accuracy holdout
)

H2 = dict(
    alignment_r2_min=0.75,         # trustworthiness criterion 1 (avg over f-loaded items)
    noninv_max_frac=0.25,          # criterion 2
    rank_corr_min=0.98,            # criterion 3 (aligned vs configural country f means)
    min_country_n=500,             # Units of Analysis
    # alignment component loss f(x) = (x^2 + eps)^(1/4), Asparouhov-Muthen 2014
    alignment_eps=0.01,
    # effect-size flagging rule; the CONSTANT is validated/locked by the H2a
    # Monte Carlo operating-characteristics harness (Stage 2 addendum item iii)
    flag_loading_dev=0.20, flag_intercept_dev=0.30,
    # H2b domain contrast (prereg: mental-health and hedonic EXCLUDED)
    h2b_focal=("meaning", "character", "social"),
    h2b_control=("physical", "financial"),
    # Chen 2007 sequential MGCFA thresholds (sensitivity leg)
    chen_dcfi=-0.010, chen_drmsea=0.015,
    # H2a MC grid (Stage 2 addendum item iii)
    mc_violation_fracs=(0.0, 0.10, 0.25, 0.40),
    mc_magnitudes=(0.2, 0.4),
    mc_reps=40,
)

H3 = dict(
    beta_min=0.05,
    n_outcomes_min=2,              # >= 2 of 3 outcomes
    fdr_q=0.05,                    # Benjamini-Hochberg across the 3 outcomes
    reliability_match_tol=0.05,    # |omega_h(f_-d) - omega_h(p)| < 0.05
    cosine_confirm=0.85,           # residualized-construct cosine (constitutive)
    cosine_disconfirm=0.70,
    n_boot=500,                    # PSU-cluster bootstrap for SEs
)

H4 = dict(
    # MIDUS-anchored constants (Comparative Datasets section; fixed pre-access)
    phi_midus=0.957,               # AR(1) persistence, unit-year scaling
    tau_midus=6.08,                # relaxation timescale, years (95% CI [3.27, 10.42])
    c_inflate=1.50,                # Stage 2 CI inflation factor
    delta_t=1.0,                   # wave spacing in years
    tikhonov_rel_cutoff=1e-3,      # smallest/largest singular value trigger
    s_skew_max=0.20,               # symmetrized-propagator fallback threshold
    n_boot=999,                    # wild cluster bootstrap replicates (MacKinnon-Webb)
    # Stage 2 correction-map Monte Carlo (addendum items vi, vii)
    wgrid_n=15, mu_grid=(0.0, 0.5, 1.0, 2.0), mc_reps=40, mc_n=30000,
    # H4b tiers
    tier1_ci_width=0.30, tier1_r2=0.10, sparsity_gap_max=0.10,
    clip_slope_min=0.25,           # map slope below which cell is 'clipping zone'
    corner_interval=(0.90, 1.0),   # high-w_L high-tau partial-identification interval
    # H4a calibration
    h4a_alpha=0.05,                # per-truth error rate for verdict thresholds
    h4a_nsim=200,                  # parametric-bootstrap replicates per truth
    # pre-flight diagnostic bands (placeholders; Stage 2 addendum item v)
    preflight=dict(trace_frac=(0.05, 0.99), diag_range=(-0.25, 1.10),
                   spectral_radius_max=1.00, homogeneity_max=0.15,
                   collider_flip_max=0.30),
)

COMMON = dict(
    n_imputations=20,              # Missing Data: MI m = 20, Rubin's rules
    estimator="DWLS",              # WLSMV point estimates == DWLS on polychorics
    alpha=0.05, two_tailed=True,
    weight_normalization="within_country",   # Sampling Weights section
    seed=20260703,
)

# ----------------------------------------------------------------------------
# 3. COUNTRY-LEVEL LOOKUPS (H2 stratifications, H2b cultural distance)
# ----------------------------------------------------------------------------
# Prereg pre-commits stratifications by Inglehart-Welzel quadrant, World Bank
# income tier, and majority religious tradition, and H2b's primary distance is
# Muthukrishna et al. 2020 CFST-from-US. CFST values are not embedded here:
# supply lookups/cultural_distance.csv (columns: COUNTRY, CFST_US) from the
# published supplement; the IW Euclidean fallback and the median-split
# sensitivity are computed from lookups/iw_scores.csv. Template files with the
# 22 GFS countries + Hong Kong SAR are written by data_io.write_lookup_templates().
GFS_COUNTRIES = [
    "Argentina", "Australia", "Brazil", "Egypt", "Germany", "Hong Kong",
    "India", "Indonesia", "Israel", "Japan", "Kenya", "Mexico", "Nigeria",
    "Philippines", "Poland", "South Africa", "Spain", "Sweden", "Tanzania",
    "Turkiye", "United Kingdom", "United States",
]

LOOKUP_DIR = "lookups"

# ----------------------------------------------------------------------------
# 4. SMOKE-MODE OVERRIDES (never used for production runs)
# ----------------------------------------------------------------------------
SMOKE = dict(
    n_per_country=350, n_imputations=2, envelope_reps=6,
    h2_mc_cells=2, h2_mc_reps=2,
    h4_wgrid_n=5, h4_mu_grid=(0.0, 1.0), h4_mc_reps=2, h4_mc_n=2500,
    h4a_nsim=12, n_boot=60, h3_boot=60,
)


def item_vars(items: list[Item]) -> list[str]:
    return [it.var for it in items]


def domain_of(var: str) -> str:
    for it in F_ITEMS:
        if it.var == var:
            return it.domain
    raise KeyError(var)


def f_partition() -> dict[str, list[str]]:
    """Domain -> item vars, hedonic listed under 'hedonic'."""
    part: dict[str, list[str]] = {}
    for it in F_ITEMS:
        part.setdefault(it.domain, []).append(it.var)
    return part

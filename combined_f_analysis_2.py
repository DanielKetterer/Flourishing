"""
combined_f_analysis_2.py
─────────────────────────────────────────────────────────────────────────────
Combined implementation of the four models from
"Toward f: A General Factor of Human Flourishing."

KEY CHANGES FROM combined_f_analysis.py:
  • The primary dynamical system is now Generalized Lotka-Volterra (GLV),
    calibrated via Lyapunov inversion: J = -(1/2) Σ⁻¹, then
    A_glv = J / μ,  r_glv = -A · μ.
    This is exact — the GLV Jacobian at equilibrium reproduces the empirical
    covariance matrix by construction.
  • The old coupled-logistic model is retained as a pedagogical comparison
    but is no longer the primary path.
  • Asymmetric coupling analysis (coupling budget, obligate mutualists)
    from self_acceptance_analysis.py is integrated.
  • Perturbation decomposition (S/A split, commutator series) from
    perturbation_decomposition.py is integrated.

Architecture
────────────
  1. Load individual-level MIDUS 2 data (real or synthetic)
  2. Load country-level ecological data (for comparison)
  3. Individual-level PCA, bifactor CFA, demographics, dual continua
  4. MODEL 1 — Bifactor CFA (individual-level = primary test)
  5. MODEL 2 — GLV dynamical system calibrated via Lyapunov inversion
     from INDIVIDUAL-LEVEL covariance
  6. ASYMMETRIC COUPLING ANALYSIS — coupling budget, obligate spectrum,
     inhibitory interactions
  7. PERTURBATION DECOMPOSITION — S/A split, commutator correction series
  8. MODEL 3 — Lyapunov stability & resilience (on GLV system)
  9. COHERENCE — Stage 1 ↔ Stage 2 bridge
  10. MODEL 4 — Individual-level Alkire-Foster capabilities
  11. THREE-MODEL COMPARISON — GLV vs Linear (OU) vs Logistic (pedagogical)
  12. COUPLING SWEEP & RESILIENCE SIGNATURE (GLV-based)
  13. Ecological fallacy diagnostic

Data requirements
─────────────────
  Individual-level: MIDUS 2 (ICPSR 4652) .tsv file, or synthetic fallback
  Country-level:    wellbeing_data/wellbeing_merged.csv (from script 01)

Requires
────────
  pip install numpy scipy pandas matplotlib seaborn scikit-learn
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.optimize import minimize, minimize_scalar, fsolve
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov, eigvals
from numpy.linalg import inv, det, slogdet, cholesky, norm, eigh, eigvalsh

# ─── Config ───────────────────────────────────────────────────────────────────

MODE = "midus"  # "midus" for real data, "synthetic" for simulated
MIDUS_DATA_PATH = Path("04652-0001-Data.tsv")
ECOLOGICAL_DATA_PATH = Path("wellbeing_data/wellbeing_merged.csv")
OUTPUT_DIR = Path("wellbeing_figures_combined_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
BOOTSTRAP_N = 1000

# ─── Column definitions ──────────────────────────────────────────────────────

PERMA_COLS = [
    "P_positive_emotion", "E_engagement_autonomy", "R_relationships",
    "M_meaning_life_sat", "A_accomplishment_gdp", "H_health",
]
SHORT_LABELS_ECO = {
    "P_positive_emotion": "P: Pos. Emotion", "E_engagement_autonomy": "E: Engagement",
    "R_relationships": "R: Relationships", "M_meaning_life_sat": "M: Meaning",
    "A_accomplishment_gdp": "A: Accomplishment", "H_health": "H: Health",
}
PALETTE_ECO = {
    "P_positive_emotion": "#E8622A", "E_engagement_autonomy": "#3B7DD8",
    "R_relationships": "#2EAA6A", "M_meaning_life_sat": "#9B59B6",
    "A_accomplishment_gdp": "#E67E22", "H_health": "#1ABC9C",
}
BIFACTOR_GROUPS_ECO = {
    "hedonic":         ["P_positive_emotion", "M_meaning_life_sat"],
    "eudaimonic":      ["E_engagement_autonomy", "A_accomplishment_gdp"],
    "social_physical": ["R_relationships", "H_health"],
}

RYFF_COLS = [
    "B1SPWBA1", "B1SPWBE1", "B1SPWBG1",
    "B1SPWBR1", "B1SPWBU1", "B1SPWBS1",
]
RYFF_LABELS = {
    "B1SPWBA1": "Autonomy", "B1SPWBE1": "Envir. Mastery",
    "B1SPWBG1": "Personal Growth", "B1SPWBR1": "Positive Relations",
    "B1SPWBU1": "Purpose in Life", "B1SPWBS1": "Self-Acceptance",
}
RYFF_SHORT = ["Auton", "Mast", "Growth", "Relat", "Purp", "Self-A"]

BIFACTOR_GROUPS_INDIV = {
    "hedonic":    ["B1SPWBS1", "B1SPWBE1"],
    "eudaimonic": ["B1SPWBU1", "B1SPWBG1"],
    "social_aut": ["B1SPWBR1", "B1SPWBA1"],
}

AUX_COLS = ["B1SPOSPA", "B1SNEGAF", "B1SB1", "B1SA11W"]
DEMO_COLS = ["M2ID", "B1PAGE_M2", "B1PRSEX", "B1PF7A"]

# -- Economic variables (MIDUS 2) -- codebook-verified -----------------------
#
# B1PB19    : Current main status (1=working, 2=unemployed/looking,
#             3=retired, 4=homemaker, 5=disabled/other, 7=DK)
# B1STINC1  : Total personal earnings (dollars; -1=inapplicable,
#             9999998=don't know)
# B1PA24    : "Any days in past 30 totally unable to work due to health?"
#             (1=yes, 2=no, 7=DK) -- GATE for B1PA24A
# B1PA24A   : Number of such days (1-30; 97=DK, 98=refused, 99=skip)
# B1PA52    : ER/urgent-care visits past 12 months (count; 97=DK, 99=skip)
# B1PA25A   : "Health limits normal activities?"
#             (1=a lot, 2=not at all, 3=some, 4=a little, 7=DK, 9=skip)
#
# CONSTRUCTED:
#   disability_days = 0 if B1PA24 != 1, else B1PA24A (if 1-30)
#   employed        = 1 if B1PB19 == 1, else 0 (drop 7=DK)
#   health_limits   = recode B1PA25A to ordinal 0-3 (0=none..3=a lot)
#   er_visits       = B1PA52 (99 -> 0, drop 97)
ECON_RAW_COLS = ["B1PB19", "B1STINC1", "B1PA24", "B1PA24A", "B1PA52", "B1PA25A"]
ECON_DERIVED = {
    "employed":        {"label": "Employed",                   "exp_sign": +1},
    "log_income":      {"label": "ln(Personal Income)",        "exp_sign": +1},
    "disability_days": {"label": "Work Disability Days (30d)", "exp_sign": -1},
    "er_visits":       {"label": "ER Visits (12 mo)",          "exp_sign": -1},
    "health_limits":   {"label": "Health Limitation (0-3)",    "exp_sign": -1},
}

PALETTE_INDIV = {
    "B1SPWBA1": "#3B7DD8", "B1SPWBE1": "#E8622A", "B1SPWBG1": "#2EAA6A",
    "B1SPWBR1": "#9B59B6", "B1SPWBU1": "#E67E22", "B1SPWBS1": "#1ABC9C",
}

MIDUS_CORR_MATRIX = np.array([
    [1.00, 0.47, 0.47, 0.34, 0.44, 0.40],
    [0.47, 1.00, 0.56, 0.54, 0.65, 0.77],
    [0.47, 0.56, 1.00, 0.47, 0.60, 0.49],
    [0.34, 0.54, 0.47, 1.00, 0.52, 0.59],
    [0.44, 0.65, 0.60, 0.52, 1.00, 0.66],
    [0.40, 0.77, 0.49, 0.59, 0.66, 1.00],
])
MIDUS_MEANS = np.array([4.80, 5.19, 5.53, 5.36, 5.38, 5.18])
MIDUS_SDS   = np.array([0.82, 0.86, 0.82, 0.94, 0.93, 0.97])


def short_label(col):
    return RYFF_LABELS.get(col, SHORT_LABELS_ECO.get(col, col))

def palette_color(col):
    return PALETTE_INDIV.get(col, PALETTE_ECO.get(col, "#888"))

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 120,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print(f"\n{'═' * 70}\n  {title}\n{'═' * 70}")

def cosine_sim(a, b):
    return np.abs(np.dot(a, b) / (norm(a) * norm(b) + 1e-15))


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING — INDIVIDUAL LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_midus(n: int = 5000) -> pd.DataFrame:
    print(f"Generating synthetic MIDUS-like data (n={n}) …")
    L = np.linalg.cholesky(MIDUS_CORR_MATRIX)
    Z = np.random.randn(n, 6)
    X_corr = Z @ L.T
    X_scaled = X_corr * MIDUS_SDS + MIDUS_MEANS
    X_scaled = X_scaled - 0.15 * (X_scaled - MIDUS_MEANS) ** 2 / MIDUS_SDS
    X_scaled = np.clip(X_scaled, 1, 7)
    df = pd.DataFrame(X_scaled, columns=RYFF_COLS)

    ryff_mean = df[RYFF_COLS].mean(axis=1)
    df["B1SPOSPA"] = np.clip(
        0.55 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 0.65 + 3.25
        + np.random.randn(n) * 0.65 * np.sqrt(1 - 0.55**2), 1, 5)
    df["B1SNEGAF"] = np.clip(
        -0.45 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 0.58 + 1.85
        + np.random.randn(n) * 0.58 * np.sqrt(1 - 0.45**2), 1, 5)
    df["B1SB1"] = np.clip(
        0.60 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 1.8 + 7.4
        + np.random.randn(n) * 1.8 * np.sqrt(1 - 0.60**2), 0, 10)
    df["B1SA11W"] = np.clip(
        0.35 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 0.9 + 3.5
        + np.random.randn(n) * 0.9 * np.sqrt(1 - 0.35**2), 1, 5).round()
    df["M2ID"]      = np.arange(1, n + 1)
    df["B1PAGE_M2"] = np.clip(np.random.normal(55, 12, n), 35, 86).astype(int)
    df["B1PRSEX"]   = np.random.choice([1, 2], n, p=[0.47, 0.53])
    df["B1PF7A"]    = np.random.choice(range(1, 13), n,
                                        p=np.array([1,1,1,1,2,2,3,3,4,5,4,3])/30)
    age_z = (df["B1PAGE_M2"] - 55) / 12
    age_adj = -0.05 * age_z + 0.02 * age_z**2
    for col in RYFF_COLS:
        df[col] = np.clip(df[col] + age_adj * MIDUS_SDS[RYFF_COLS.index(col)], 1, 7)

    # -- Synthetic economic variables (approximate realistic distributions) --
    ryff_z = (df[RYFF_COLS].mean(axis=1) - df[RYFF_COLS].mean(axis=1).mean()) / df[RYFF_COLS].mean(axis=1).std()
    # Employment (B1PB19 coding: 1=working, 3=retired, etc.)
    age = df["B1PAGE_M2"].values
    retire_prob = np.clip((age - 55) / 30, 0, 0.8)
    work_latent = 1.5 + 0.4 * ryff_z.values - 2.0 * retire_prob + np.random.randn(n) * 0.6
    df["B1PB19"] = np.where(work_latent > 0, 1,
                   np.where(np.random.rand(n) < retire_prob, 3, 5)).astype(float)
    # Income
    log_inc = 10.2 + 0.30 * ryff_z.values + np.random.randn(n) * 0.85
    df["B1STINC1"] = np.clip(np.exp(log_inc), 0, 500000)
    df.loc[df["B1PB19"] != 1, "B1STINC1"] = np.where(
        np.random.rand((df["B1PB19"] != 1).sum()) < 0.3, 0, df.loc[df["B1PB19"] != 1, "B1STINC1"] * 0.3)
    # Disability days (gate + count)
    df["B1PA24"] = np.where(np.random.rand(n) < np.clip(0.35 - 0.12 * ryff_z.values, 0.05, 0.7), 1, 2)
    days = np.clip(np.round(np.exp(1.5 - 0.3 * ryff_z.values + np.random.randn(n) * 0.8)), 1, 30)
    df["B1PA24A"] = np.where(df["B1PA24"] == 1, days, 99).astype(float)
    # ER visits
    er_rate = np.exp(-0.8 - 0.15 * ryff_z.values + np.random.randn(n) * 0.9)
    df["B1PA52"] = np.clip(np.round(er_rate), 0, 15).astype(float)
    df.loc[df["B1PA52"] == 0, "B1PA52"] = 99  # code 0 as skip
    # Health limitation (1=a lot, 2=not at all, 3=some, 4=a little)
    lim_latent = -0.5 * ryff_z.values + np.random.randn(n) * 0.8
    df["B1PA25A"] = np.select(
        [lim_latent > 1.2, lim_latent > 0.4, lim_latent > -0.3],
        [1, 3, 4], default=2).astype(float)

    print(f"  Synthetic MIDUS data: {df.shape}")
    print(f"  ⚠  SIMULATED data — register at ICPSR for real MIDUS 2.")
    return df


def load_midus(path: Path) -> pd.DataFrame:
    print(f"Loading MIDUS 2 data from {path} …")
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {path}\n"
            f"  Download MIDUS 2 (ICPSR 4652), place main .tsv here.\n"
            f"  Or set MODE = 'synthetic' to run with simulated data.")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    for col in RYFF_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where((df[col] > 0) & (df[col] < 90))
            df[col] = df[col] / 3.0
    for col in AUX_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col] > 0)
    # Coerce economic raw columns
    for col in ECON_RAW_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    econ_found = [c for c in ECON_RAW_COLS if c in df.columns]
    print(f"  Economic vars found: {len(econ_found)}/{len(ECON_RAW_COLS)}")
    print(f"  Loaded: {len(df)} respondents, {len(df.columns)} variables")
    return df


def preprocess_midus(df: pd.DataFrame):
    df = df.copy()
    if "B1SA11W" in df.columns:
        df["B1SA11W"] = 6 - df["B1SA11W"]
    present = [c for c in RYFF_COLS if c in df.columns]
    df_clean = df.dropna(subset=present).copy()
    print(f"  Complete cases on Ryff subscales: {len(df_clean):,} / {len(df):,}")
    X_raw    = df_clean[present].values.astype(float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return df_clean, X_raw, X_scaled, present


def load_ecological_data():
    if not ECOLOGICAL_DATA_PATH.exists():
        print(f"  ⚠  No ecological data at {ECOLOGICAL_DATA_PATH}")
        return None
    df = pd.read_csv(ECOLOGICAL_DATA_PATH)
    present = [c for c in PERMA_COLS if c in df.columns]
    df_clean = df[["country"] + present].dropna(subset=present).copy()
    X_raw = df_clean[present].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    S = np.cov(X_scaled.T, ddof=1)
    N = X_scaled.shape[0]
    print(f"  Ecological data: {N} countries, {len(present)} dimensions")
    return df_clean, X_raw, X_scaled, S, present, N


# ═══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL-LEVEL PCA & CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

def report_manifold(df, cols, label=""):
    corr = df[cols].corr()
    off = corr.values[np.tril_indices_from(corr.values, k=-1)]
    print(f"\n  [{label}] Correlation summary ({len(cols)} subscales, {len(off)} pairs):")
    print(f"    Mean r : {off.mean():.3f}")
    print(f"    Min r  : {off.min():.3f}  ({'✓ all positive' if off.min() > 0 else '⚠ some negative'})")
    print(f"    Max r  : {off.max():.3f}")
    tril_rows, tril_cols = np.tril_indices(len(cols), k=-1)
    idx = np.argmax(off)
    idxmin = np.argmin(off)
    print(f"    Strongest: {short_label(cols[tril_rows[idx]])} ↔ "
          f"{short_label(cols[tril_cols[idx]])}  r={off[idx]:.3f}")
    print(f"    Weakest:   {short_label(cols[tril_rows[idxmin]])} ↔ "
          f"{short_label(cols[tril_cols[idxmin]])}  r={off[idxmin]:.3f}")
    return corr


def plot_corr_heatmap(df, cols, title_suffix=""):
    corr = df[cols].corr()
    labels = [short_label(c) for c in cols]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                xticklabels=labels, yticklabels=labels)
    off = corr.values[np.tril_indices_from(corr.values, k=-1)]
    ax.set_title(f"Positive Manifold: Subscale Correlations\n({title_suffix})", pad=14)
    ax.text(0.01, -0.10,
            f"Mean r = {off.mean():.3f}  |  All positive: {'✓' if (off > 0).all() else '✗'}",
            transform=ax.transAxes, fontsize=8.5, color="#444")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def run_pca_individual(X_scaled, cols, df_clean, suffix=""):
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    ev = pca.explained_variance_
    vr = pca.explained_variance_ratio_
    print(f"\n  PCA results [{suffix}]:")
    print(f"    Eigenvalues:   {ev.round(3)}")
    print(f"    Var explained: {(vr*100).round(1)}")
    print(f"    PC1 (f proxy): {vr[0]*100:.1f}%")
    l1 = loadings[:, 0].copy()
    n = X_scaled.shape[0]
    boot_L = np.zeros((BOOTSTRAP_N, X_scaled.shape[1]))
    for i in range(BOOTSTRAP_N):
        idx = np.random.choice(n, n, replace=True)
        pca_b = PCA(n_components=1); pca_b.fit(X_scaled[idx])
        lb = pca_b.components_[0]
        if np.dot(lb, l1) < 0: lb = -lb
        boot_L[i] = lb
    if l1.mean() < 0: l1 = -l1; boot_L = -boot_L
    ci_lo = np.percentile(boot_L, 2.5, axis=0)
    ci_hi = np.percentile(boot_L, 97.5, axis=0)
    print(f"\n  PC1 Loadings with 95% Bootstrap CI:")
    for col, lo, val, hi in zip(cols, ci_lo, l1, ci_hi):
        print(f"    {short_label(col):<22} {val:.3f}  [{min(lo,hi):.3f}, {max(lo,hi):.3f}]")
    f_raw = scores[:, 0]
    if f_raw.mean() < 0:
        f_raw = -f_raw; pca.components_[0] = -pca.components_[0]
    f_z = (f_raw - f_raw.mean()) / f_raw.std()
    df_clean = df_clean.copy()
    df_clean["f_score"] = f_z
    return df_clean, pca, vr


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 1: BIFACTOR CFA VIA ML
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sigma_bifactor(params, p, group_indices):
    lam_f = params[:p]; lam_s = params[p:2*p]; theta = np.exp(params[2*p:3*p])
    Sigma = np.outer(lam_f, lam_f)
    for group_idx in group_indices:
        lam_sg = np.zeros(p)
        for j in group_idx: lam_sg[j] = lam_s[j]
        Sigma += np.outer(lam_sg, lam_sg)
    Sigma += np.diag(theta)
    return Sigma

def _ml_discrepancy(params, S, N, p, group_indices):
    Sigma = _build_sigma_bifactor(params, p, group_indices)
    try:
        sign, logdet_sigma = slogdet(Sigma)
        if sign <= 0: return 1e10
        sign_s, logdet_s = slogdet(S)
        Sigma_inv = inv(Sigma)
        return logdet_sigma + np.trace(S @ Sigma_inv) - logdet_s - p
    except np.linalg.LinAlgError:
        return 1e10

def fit_bifactor_cfa(S, N, cols, bifactor_groups):
    p = len(cols)
    group_indices = []
    for gname, gcols in bifactor_groups.items():
        idx = [cols.index(c) for c in gcols if c in cols]
        group_indices.append(idx)
    pca = PCA(n_components=1)
    pca.fit(np.random.multivariate_normal(np.zeros(p), S, size=max(N, 200)))
    lam_f_init = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    if lam_f_init.mean() < 0: lam_f_init = -lam_f_init
    lam_s_init = np.full(p, 0.3)
    theta_init = np.log(np.diag(S) * 0.3)
    x0 = np.concatenate([lam_f_init, lam_s_init, theta_init])
    result = minimize(_ml_discrepancy, x0, args=(S, N, p, group_indices),
                      method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-10})
    lam_f = result.x[:p]; lam_s = result.x[p:2*p]; theta = np.exp(result.x[2*p:3*p])
    if lam_f.mean() < 0: lam_f = -lam_f
    Sigma_hat = _build_sigma_bifactor(result.x, p, group_indices)
    sum_lam_f_sq = np.sum(lam_f)**2
    sum_lam_s_group_sq = sum(np.sum(lam_s[gidx])**2 for gidx in group_indices)
    sum_theta = np.sum(theta)
    total_var = sum_lam_f_sq + sum_lam_s_group_sq + sum_theta
    omega_h = sum_lam_f_sq / total_var
    common_var_f = np.sum(lam_f**2); common_var_s = np.sum(lam_s**2)
    ecv = common_var_f / (common_var_f + common_var_s)
    F_min = result.fun
    n_free = 3 * p; n_data = p * (p + 1) // 2; df_model = n_data - n_free
    chi2 = max(0, (N - 1) * F_min)
    rmsea = np.sqrt(max(0, (chi2 / df_model - 1) / (N - 1))) if df_model > 0 else np.nan

    # --- SRMR: standardized root mean square residual ---
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(S)))
    R_obs = D_inv_sqrt @ S @ D_inv_sqrt          # sample correlation matrix
    D_hat_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Sigma_hat)))
    R_hat = D_hat_inv_sqrt @ Sigma_hat @ D_hat_inv_sqrt  # model-implied correlation
    resid = R_obs - R_hat
    # sum of squared lower-triangle-plus-diagonal residuals
    srmr = np.sqrt(2.0 * np.sum(np.tril(resid)**2) / (p * (p + 1)))

    # --- Null (independence) model for CFI / TLI ---
    # Sigma_null = diag(S); F_null = ln(prod s_ii / |S|)
    sign_s, logdet_s = slogdet(S)
    F_null = np.sum(np.log(np.diag(S))) - logdet_s   # = ln(prod diag / det)
    df_null = p * (p - 1) // 2
    chi2_null = max(0, (N - 1) * F_null)

    # CFI = 1 - max(chi2_m - df_m, 0) / max(chi2_null - df_null, 0)
    numer = max(chi2 - df_model, 0)
    denom = max(chi2_null - df_null, 0)
    cfi = 1.0 - numer / denom if denom > 0 else np.nan

    # TLI (NNFI) = (chi2_null/df_null - chi2_m/df_m) / (chi2_null/df_null - 1)
    if df_null > 0 and df_model > 0:
        ratio_null = chi2_null / df_null
        tli = (ratio_null - chi2 / df_model) / (ratio_null - 1) if ratio_null > 1 else np.nan
    else:
        tli = np.nan

    return {
        "lambda_f": lam_f, "lambda_s": lam_s, "theta": theta,
        "omega_h": omega_h, "ecv": ecv, "Sigma_hat": Sigma_hat,
        "F_min": F_min, "chi2": chi2, "df": df_model, "RMSEA": rmsea,
        "CFI": cfi, "TLI": tli, "SRMR": srmr,
        "chi2_null": chi2_null, "df_null": df_null,
        "converged": result.success, "group_indices": group_indices,
    }

def plot_bifactor_loadings(result, cols, bifactor_groups):
    lam_f = result["lambda_f"]; lam_s = result["lambda_s"]
    labels = [short_label(c) for c in cols]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(cols)); h = 0.35
    ax.barh(y + h/2, lam_f, height=h, color="#3B7DD8", alpha=0.85,
            label=f"General f (ω_h = {result['omega_h']:.3f})")
    ax.barh(y - h/2, np.abs(lam_s), height=h, color="#E67E22", alpha=0.85, label="Specific s_k")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Factor Loading (λ)")
    ax.set_title(f"Bifactor CFA Loadings\nECV = {result['ecv']:.3f}  |  "
                 f"CFI = {result['CFI']:.3f}  |  TLI = {result['TLI']:.3f}  |  "
                 f"SRMR = {result['SRMR']:.3f}  |  RMSEA = {result['RMSEA']:.3f}", fontsize=10)
    ax.axvline(0, color="black", lw=0.8); ax.legend(fontsize=9, loc="lower right")
    group_colors = {"hedonic": "#9B59B6", "eudaimonic": "#3B7DD8",
                    "social_physical": "#2EAA6A", "social_aut": "#2EAA6A"}
    for gname, gcols in bifactor_groups.items():
        idxs = [cols.index(c) for c in gcols if c in cols]
        if idxs:
            ymin, ymax = min(idxs) - 0.4, max(idxs) + 0.4
            ax.axhspan(ymin, ymax, alpha=0.06, color=group_colors.get(gname, "#888"))
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 2a: GENERALIZED LOTKA-VOLTERRA (PRIMARY)
# ═══════════════════════════════════════════════════════════════════════════════

class GLVSystem:
    """
    Generalized Lotka-Volterra system:

        dx_k/dt = x_k · (r_k + Σ_l A_kl · x_l)

    Calibration via Lyapunov inversion:
        Σ_obs → J = -(1/2) Σ⁻¹ → A_kl = J_kl / μ_k, r_k = -Σ_l A_kl μ_l

    This guarantees:
      • The Jacobian at x* = μ equals J by construction.
      • The stationary covariance of the OU linearisation equals Σ_obs.
      • r_k < 0 identifies obligate mutualists (no intrinsic growth).
      • Off-diagonal A_kl < 0 identifies inhibitory interactions.
    """

    def __init__(self, K, r, A, mu):
        self.K = K
        self.r = r.copy()
        self.A = A.copy()
        self.mu = mu.copy()  # calibration-point equilibrium

    def rhs(self, t, x):
        return x * (self.r + self.A @ x)

    def equilibrium(self):
        """Solve A x* = -r. Falls back to stored μ if singular."""
        try:
            x_star = -np.linalg.solve(self.A, self.r)
            if np.all(x_star > 0) and np.all(np.isfinite(x_star)):
                return x_star
        except np.linalg.LinAlgError:
            pass
        return self.mu.copy()

    def jacobian_at(self, x_star):
        return np.diag(x_star) @ self.A

    def mean_coupling_strength(self):
        """Mean off-diagonal of interaction matrix (absolute value)."""
        offdiag = self.A.copy()
        np.fill_diagonal(offdiag, 0.0)
        K = self.K
        return np.sum(np.abs(offdiag)) / (K * (K - 1))

    def simulate_population(self, n_individuals=200, r_var=0.10, A_var=0.05, seed=42):
        """Simulate population with parameter variation → positive manifold."""
        rng = np.random.RandomState(seed)
        equilibria = np.zeros((n_individuals, self.K))
        for i in range(n_individuals):
            r_i = self.r * np.exp(rng.normal(0, r_var, self.K))
            A_i = self.A * np.exp(rng.normal(0, A_var, (self.K, self.K)))
            try:
                x_star = -np.linalg.solve(A_i, r_i)
                if np.all(x_star > 0) and np.all(np.isfinite(x_star)):
                    equilibria[i] = x_star
                else:
                    equilibria[i] = self.mu
            except Exception:
                equilibria[i] = self.mu
        return equilibria


def calibrate_glv_from_lyapunov_inversion(X_raw, cols):
    """
    Calibrate GLV from individual-level data via Lyapunov inversion.

    Steps:
      1. Compute empirical covariance Σ_obs from raw data
      2. Jacobian: J = -(1/2) Σ⁻¹
      3. GLV interaction matrix: A_kl = J_kl / μ_k
      4. GLV intrinsic rates: r_k = -Σ_l A_kl μ_l

    Returns (GLVSystem, diagnostics_dict).
    """
    K = X_raw.shape[1]
    mu = X_raw.mean(axis=0)
    sds = X_raw.std(axis=0)

    # Empirical covariance (raw scale)
    Sigma_obs = np.cov(X_raw.T, ddof=1)

    # Lyapunov inversion: J = -(1/2) Σ⁻¹
    Sigma_inv = inv(Sigma_obs)
    J = -0.5 * Sigma_inv

    # Check stability
    eigs_J = eigvals(J)
    stable = np.all(np.real(eigs_J) < 0)

    # GLV parametrisation
    A_glv = J / mu[:, np.newaxis]  # A_kl = J_kl / mu_k
    r_glv = -A_glv @ mu            # r_k = -Sum_l A_kl mu_l

    glv = GLVSystem(K, r_glv, A_glv, mu)

    # Verification
    x_star = glv.equilibrium()
    J_check = glv.jacobian_at(x_star)
    Sigma_check = -0.5 * inv(J_check)

    # Coupling budget (for the old logistic comparison)
    M_log = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            if k != l:
                M_log[k, l] = -J[k, l] / J[k, k]

    diagnostics = {
        "N": X_raw.shape[0],
        "mu": mu.copy(),
        "sds": sds.copy(),
        "Sigma_obs": Sigma_obs.copy(),
        "J": J.copy(),
        "A_glv": A_glv.copy(),
        "r_glv": r_glv.copy(),
        "J_stable": stable,
        "eq_match": norm(x_star - mu) / norm(mu),
        "J_match": norm(J_check - J, 'fro') / norm(J, 'fro'),
        "Sigma_match": norm(Sigma_check - Sigma_obs, 'fro') / norm(Sigma_obs, 'fro'),
        "M_logistic": M_log,
    }
    return glv, diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 2b: COUPLED LOGISTIC (PEDAGOGICAL — retained for comparison)
# ═══════════════════════════════════════════════════════════════════════════════

class CoupledLogisticSystem:
    """
    Coupled logistic growth model (van der Maas et al. 2006):
      dx_k/dt = a_k x_k (1 - x_k/C_k + Σ_{l≠k} M_kl x_l / C_k)

    NOTE: This breaks for Self-Acceptance (C_k < 0 when coupling input > mean).
    Retained for pedagogical comparison only. Use GLV as primary model.
    """
    def __init__(self, K, a, C, M):
        self.K, self.a, self.C, self.M = K, a.copy(), C.copy(), M.copy()

    def rhs(self, t, x):
        dx = np.zeros(self.K)
        for k in range(self.K):
            mut = sum(self.M[k, l] * x[l] / self.C[k] for l in range(self.K) if l != k)
            dx[k] = self.a[k] * x[k] * (1 - x[k] / self.C[k] + mut)
        return dx


class LinearOUSystem:
    """Linear (Ornstein-Uhlenbeck) system: dz = J·(x - μ) dt + σ dW."""
    def __init__(self, J, mu):
        self.K = len(mu); self.J = J.copy(); self.mu = mu.copy()
    def rhs(self, t, x):
        return self.J @ (x - self.mu)


# ═══════════════════════════════════════════════════════════════════════════════
#  ASYMMETRIC COUPLING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_budget_analysis(glv, diag, cols):
    """
    Compute the coupling budget for each domain.

    For the logistic parameterisation:
        C_k = mu_k - Σ_{l≠k} M_kl mu_l

    Domains with C_k < 0 are OBLIGATE mutualists — they cannot sustain
    themselves without coupling input.
    """
    K = glv.K
    mu = diag["mu"]
    M_log = diag["M_logistic"]
    r_glv = glv.r
    A_glv = glv.A

    budgets = []
    for k in range(K):
        coupling_input = sum(M_log[k, l] * mu[l] for l in range(K) if l != k)
        C_k = mu[k] - coupling_input
        ratio = C_k / mu[k]
        if C_k < 0:
            status = "OBLIGATE"
        elif ratio < 0.15:
            status = "marginal"
        else:
            status = "facultative"

        # Per-source breakdown
        sources = []
        for l in range(K):
            if l == k:
                continue
            contrib = M_log[k, l] * mu[l]
            sources.append({"source_idx": l, "M_kl": M_log[k, l],
                            "mu_l": mu[l], "contrib": contrib})

        budgets.append({
            "domain_idx": k, "domain": short_label(cols[k]),
            "mu_k": mu[k], "coupling_input": coupling_input,
            "C_k": C_k, "ratio": ratio, "status": status,
            "r_glv": r_glv[k],
            "sources": sorted(sources, key=lambda s: -abs(s["contrib"])),
        })

    # Inhibitory interactions in the GLV
    offdiag = A_glv.copy()
    np.fill_diagonal(offdiag, 0)
    neg_pairs = np.argwhere(offdiag < -0.001)

    return {
        "budgets": budgets,
        "n_obligate": sum(1 for b in budgets if b["status"] == "OBLIGATE"),
        "n_marginal": sum(1 for b in budgets if b["status"] == "marginal"),
        "neg_pairs": neg_pairs,
        "diag_A": np.diag(A_glv),
        "all_diag_neg": bool(np.all(np.diag(A_glv) < 0)),
    }


def print_coupling_budget(budget_result, cols):
    """Pretty-print the coupling budget analysis."""
    print(f"\n  Coupling Budget (logistic parameterisation)")
    print(f"    C_k = mu_k - Σ_l M_kl mu_l")
    print(f"\n  {'Domain':<22} {'mu_k':>6} {'Σ M·mu':>8} {'C_k':>8} {'C/mu':>7}  Status  r_glv")
    print(f"  {'-'*80}")
    for b in budget_result["budgets"]:
        print(f"  {b['domain']:<22} {b['mu_k']:>6.2f} {b['coupling_input']:>8.3f} "
              f"{b['C_k']:>8.3f} {b['ratio']:>7.3f}  {b['status']:<12} {b['r_glv']:>8.4f}")

    if budget_result["n_obligate"] > 0:
        print(f"\n  ⚠  {budget_result['n_obligate']} obligate mutualist(s) detected (r_k < 0 in GLV).")

    if len(budget_result["neg_pairs"]) > 0:
        print(f"\n  Inhibitory GLV interactions (A_kl < 0):")
        for k, l in budget_result["neg_pairs"]:
            print(f"    {short_label(cols[k])} ← {short_label(cols[l])}: "
                  f"A = {budget_result['diag_A'][k]:.4f}")

    # Detail for the most coupling-dependent domain
    most_dep = min(budget_result["budgets"], key=lambda b: b["ratio"])
    print(f"\n  Most coupling-dependent: {most_dep['domain']} "
          f"({100*(1-most_dep['ratio']):.0f}% coupling-dependent)")
    print(f"    {'Source':<22} {'M_kl':>8} {'mu_l':>6} {'M·mu':>8} {'%':>6}")
    print(f"    {'-'*54}")
    total = most_dep["coupling_input"]
    for s in most_dep["sources"]:
        pct = 100 * s["contrib"] / total if total > 0 else 0
        print(f"    {short_label(cols[s['source_idx']]):<22} {s['M_kl']:>8.4f} "
              f"{s['mu_l']:>6.2f} {s['contrib']:>8.3f} {pct:>5.1f}%")


def plot_coupling_budget(budget_result, cols):
    """Two-panel: stacked coupling budget & obligate spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    K = len(budget_result["budgets"])
    names = [b["domain"] for b in budget_result["budgets"]]
    colors = [palette_color(c) for c in cols]

    # Panel A: Stacked bar
    ax = axes[0]
    for k, b in enumerate(budget_result["budgets"]):
        ax.barh(k, max(b["C_k"], 0), height=0.6, color='#333333', alpha=0.8,
                label='Intrinsic C_k' if k == 0 else None)
        if b["C_k"] < 0:
            ax.barh(k, b["C_k"], height=0.6, color='red', alpha=0.4,
                    label='Deficit (C<0)' if b["status"] == "OBLIGATE" else None)
        left = max(b["C_k"], 0)
        for si, s in enumerate(b["sources"][:4]):
            if s["contrib"] > 0.05:
                ax.barh(k, s["contrib"], left=left, height=0.6,
                        color=colors[s["source_idx"]], alpha=0.7)
                left += s["contrib"]
        ax.plot(b["mu_k"], k, 'k*', ms=12, zorder=5)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Level at equilibrium', fontsize=10)
    ax.set_title('(a) Coupling Budget\n* = observed mean', fontsize=11, fontweight='bold')

    # Panel B: Obligate spectrum
    ax = axes[1]
    ratios = np.array([b["ratio"] for b in budget_result["budgets"]])
    si = np.argsort(ratios)[::-1]
    clrs = ['#d7191c' if ratios[si[i]] < 0 else '#fdae61' if ratios[si[i]] < 0.15
            else '#2c7bb6' for i in range(K)]
    ax.barh(range(K), ratios[si], height=0.6, color=clrs, edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=1.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels([names[i] for i in si], fontsize=9)
    ax.set_xlabel('C_k / mu_k (fraction self-sustaining)', fontsize=10)
    ax.set_title('(b) Obligate ↔ Facultative Spectrum', fontsize=11, fontweight='bold')
    for i, idx in enumerate(si):
        pct = 100 * (1 - ratios[idx])
        ax.text(max(ratios[idx], 0) + 0.01, i, f'{pct:.0f}% coupling-dep.',
                va='center', fontsize=8)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PERTURBATION DECOMPOSITION: J = S + A
# ═══════════════════════════════════════════════════════════════════════════════

def perturbation_decomposition(J, sigma=1.0, n_orders=3):
    """
    Decompose the Lyapunov equation J Σ + Σ J' = -σ²I via perturbation series.

    J = S + A where S = (J+J')/2 (symmetric), A = (J-J')/2 (antisymmetric).

    Returns dict with S, A, asymmetry index, Σ orders, alignment metrics.
    """
    K = J.shape[0]
    S = (J + J.T) / 2
    A = (J - J.T) / 2

    asym_index = norm(A, 'fro') / norm(J, 'fro')

    # Check S stability
    s_eigs = eigvalsh(S)
    if np.any(s_eigs >= 0):
        return {"S": S, "A": A, "asymmetry_index": asym_index,
                "S_stable": False, "error": "S has non-negative eigenvalue"}

    # Full solution
    Q_full = -sigma**2 * np.eye(K)
    try:
        Sigma_full = solve_continuous_lyapunov(J, Q_full)
    except Exception:
        return {"S": S, "A": A, "asymmetry_index": asym_index,
                "S_stable": True, "error": "Full Lyapunov solve failed"}

    # Order 0: S Σ₀ + Σ₀ S = -σ²I
    Sigma_0 = solve_continuous_lyapunov(S, Q_full)
    orders = [Sigma_0]
    cumulative = [Sigma_0.copy()]

    Sigma_prev = Sigma_0
    for k in range(1, n_orders):
        commutator = A @ Sigma_prev - Sigma_prev @ A
        Sigma_k = solve_continuous_lyapunov(S, -commutator)
        orders.append(Sigma_k)
        cumulative.append(cumulative[-1] + Sigma_k)
        Sigma_prev = Sigma_k

    # PC1 at each order
    def pc1(Sigma_mat):
        vals, vecs = np.linalg.eigh(Sigma_mat)
        idx = np.argsort(-vals)
        v = vecs[:, idx[0]]
        if v.mean() < 0: v = -v
        return v

    pc1_full = pc1(Sigma_full)
    pc1_orders = [pc1(c) for c in cumulative]

    # Jacobian dominant eigenvector
    jac_vals, jac_vecs = np.linalg.eig(J)
    dom_idx = np.argmax(np.real(jac_vals))
    v_J = np.real(jac_vecs[:, dom_idx])
    if v_J.mean() < 0: v_J = -v_J

    # S dominant eigenvector
    s_vals, s_vecs = np.linalg.eig(S)
    dom_idx_s = np.argmax(np.real(s_vals))
    v_S = np.real(s_vecs[:, dom_idx_s])
    if v_S.mean() < 0: v_S = -v_S

    # Alignment metrics
    cos_J_pc1 = cosine_sim(v_J, pc1_full)
    cos_S_Sig0 = cosine_sim(v_S, pc1_orders[0])
    cos_J_S = cosine_sim(v_J, v_S)
    cos_orders = [cosine_sim(pc1_o, pc1_full) for pc1_o in pc1_orders]

    # Order magnitudes
    order_norms = [norm(o, 'fro') / norm(orders[0], 'fro') for o in orders]

    return {
        "S": S, "A": A, "asymmetry_index": asym_index, "S_stable": True,
        "Sigma_full": Sigma_full, "Sigma_0": Sigma_0,
        "orders": orders, "cumulative": cumulative,
        "pc1_full": pc1_full, "pc1_orders": pc1_orders,
        "v_J": v_J, "v_S": v_S,
        "cos_J_pc1": cos_J_pc1, "cos_S_Sig0": cos_S_Sig0, "cos_J_S": cos_J_S,
        "cos_orders_vs_full": cos_orders,
        "order_norms": order_norms,
        "error": None,
    }


def print_perturbation_decomposition(decomp, cols):
    """Pretty-print perturbation decomposition results."""
    print(f"\n  Jacobian decomposition: J = S + A")
    print(f"    Asymmetry index ‖A‖/‖J‖ = {decomp['asymmetry_index']:.4f}")
    print(f"    S stable: {decomp['S_stable']}")

    if decomp.get("error"):
        print(f"    ⚠  {decomp['error']}")
        return

    print(f"\n  Alignment chain:")
    print(f"    cos(J dom.vec, Σ PC1)     = {decomp['cos_J_pc1']:.5f}  ← what we want to explain")
    print(f"    cos(S dom.vec, Σ₀ PC1)    = {decomp['cos_S_Sig0']:.5f}  ← symmetric baseline (~1)")
    print(f"    cos(J dom.vec, S dom.vec)  = {decomp['cos_J_S']:.5f}  ← Source 1")

    for i, cos_val in enumerate(decomp["cos_orders_vs_full"]):
        label = "Σ₀" if i == 0 else "+".join(f"Σ_{j}" for j in range(i+1))
        print(f"    cos({label} PC1, Σ PC1) = {cos_val:.5f}")

    print(f"\n  Perturbation order magnitudes (‖Σₖ‖/‖Σ₀‖):")
    for i, on in enumerate(decomp["order_norms"]):
        print(f"    Order {i}: {on:.6f}")

    # Two-source decomposition
    gap_source1 = 1 - decomp["cos_J_S"]
    gap_source2 = 1 - decomp["cos_orders_vs_full"][0]
    gap_total   = 1 - decomp["cos_J_pc1"]

    print(f"\n  Misalignment decomposition (1 - cos):")
    print(f"    Source 1 (J vs S eigenvec):     {gap_source1:.6f}")
    print(f"    Source 2 (Σ₀ vs Σ rotation):    {gap_source2:.6f}")
    print(f"    Total (J vs full PC1):           {gap_total:.6f}")
    print(f"    Sum:                             {gap_source1 + gap_source2:.6f}")
    additive = abs((gap_source1 + gap_source2) - gap_total) < 0.005
    print(f"    → {'Approximately additive' if additive else 'Interactive (nonlinear)'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 3: LYAPUNOV STABILITY & RESILIENCE (on GLV)
# ═══════════════════════════════════════════════════════════════════════════════

def lyapunov_analysis(glv, x_star):
    """Full Lyapunov stability analysis at equilibrium x*."""
    J = glv.jacobian_at(x_star)
    eigs = eigvals(J)
    idx = np.argsort(np.real(eigs))[::-1]
    eigs_sorted = eigs[idx]
    lambda_1 = eigs_sorted[0]
    stable = np.all(np.real(eigs) < 0)
    tau = 1.0 / np.abs(np.real(lambda_1)) if np.real(lambda_1) != 0 else np.inf
    recovery_speed = np.abs(np.real(lambda_1)) if stable else 0.0

    Q = np.eye(glv.K)
    try:
        P = solve_continuous_lyapunov(J.T, -Q)
        P_eigvals = np.linalg.eigvalsh(P)
        basin_radius = 1.0 / np.sqrt(np.max(P_eigvals)) if np.all(P_eigvals > 0) else 0.0
    except Exception:
        P = np.eye(glv.K); basin_radius = 0.0

    resilience = np.sqrt(recovery_speed * basin_radius) if (recovery_speed > 0 and basin_radius > 0) else 0.0

    return {
        "J": J, "eigenvalues": eigs_sorted, "lambda_1": lambda_1,
        "tau": tau, "P": P, "resilience": resilience,
        "recovery_speed": recovery_speed, "basin_radius": basin_radius, "stable": stable,
    }


def plot_lyapunov_analysis(lyap_result, x_star, glv, cols):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    eigs = lyap_result["eigenvalues"]
    labels_s = [short_label(c) for c in cols]
    colors = [palette_color(c) for c in cols]

    ax = axes[0]
    ax.scatter(np.real(eigs), np.imag(eigs), s=120, c="#3B7DD8", edgecolors="black", zorder=5)
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.6, label="Stability boundary")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
    ax.set_title(f"(a) Jacobian Eigenvalues\nStable: {lyap_result['stable']}  |  τ = {lyap_result['tau']:.2f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    delta = np.zeros(glv.K); delta[0] = -x_star[0] * 0.3
    x_perturbed = np.maximum(x_star + delta, 0.01)
    sol = solve_ivp(glv.rhs, (0, 40), x_perturbed,
                    method="RK45", max_step=0.5, t_eval=np.linspace(0, 40, 200))
    for k in range(glv.K):
        deviation = (sol.y[k] - x_star[k]) / x_star[k] * 100
        ax.plot(sol.t, deviation, color=colors[k], lw=1.5, label=labels_s[k])
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Time after perturbation"); ax.set_ylabel("% deviation from equilibrium")
    ax.set_title(f"(b) Recovery from 30% Shock (GLV)\nτ ≈ {lyap_result['tau']:.1f}")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    P = lyap_result["P"]
    try:
        P_inv = inv(P)
        eigvals_p, eigvecs_p = np.linalg.eigh(P_inv)
        idx = np.argsort(eigvals_p)[::-1]
        v1, v2 = eigvecs_p[:, idx[0]], eigvecs_p[:, idx[1]]
        proj = np.column_stack([v1, v2])
        P_2d = proj.T @ P @ proj
        theta = np.linspace(0, 2*np.pi, 200)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        eigvals_2d, eigvecs_2d = np.linalg.eigh(P_2d)
        transform = eigvecs_2d @ np.diag(1.0 / np.sqrt(eigvals_2d)) @ eigvecs_2d.T
        ellipse = circle @ transform.T
        c_level = lyap_result["resilience"]**2
        ellipse *= np.sqrt(c_level)
        ax.plot(ellipse[:, 0], ellipse[:, 1], color="#3B7DD8", lw=2)
        ax.fill(ellipse[:, 0], ellipse[:, 1], alpha=0.15, color="#3B7DD8")
        ax.scatter([0], [0], marker="*", s=200, color="#E8622A", zorder=5, label="Equilibrium x*")
        ax.set_xlabel("δ along eigenvector 1"); ax.set_ylabel("δ along eigenvector 2")
        ax.set_aspect("equal")
    except Exception:
        ax.text(0.5, 0.5, "Basin projection\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title(f"(c) Basin of Attraction\nR(x*) ≈ {lyap_result['resilience']:.4f}")
    ax.legend(fontsize=8)

    fig.suptitle("Model 3: Lyapunov Stability & Resilience (GLV, Lyapunov Inversion)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  REPRESENTATION COHERENCE: Stage 1 ↔ Stage 2 Bridge
# ═══════════════════════════════════════════════════════════════════════════════

def compute_coherence_metrics(pca_result, glv, x_star, S_obs, cols):
    K = len(cols)
    w_empirical = pca_result.components_[0]
    if w_empirical.mean() < 0: w_empirical = -w_empirical

    obs_eigvals, obs_eigvecs = np.linalg.eigh(S_obs)
    idx = np.argsort(obs_eigvals)[::-1]
    v1_obs = obs_eigvecs[:, idx[0]]
    if v1_obs.mean() < 0: v1_obs = -v1_obs

    J = glv.jacobian_at(x_star)
    eigs_J, evecs_J = np.linalg.eig(J)
    dom_idx = np.argmax(np.real(eigs_J))
    v_dominant_J = np.real(evecs_J[:, dom_idx])
    if v_dominant_J.mean() < 0: v_dominant_J = -v_dominant_J

    equilibria = glv.simulate_population(n_individuals=500, seed=42)
    eq_std = (equilibria - equilibria.mean(axis=0)) / (equilibria.std(axis=0) + 1e-8)
    Sigma_model = np.cov(eq_std.T)
    mod_eigvals, mod_eigvecs = np.linalg.eigh(Sigma_model)
    idx_m = np.argsort(mod_eigvals)[::-1]
    v1_model = mod_eigvecs[:, idx_m[0]]
    if v1_model.mean() < 0: v1_model = -v1_model

    weak_coherence = cosine_sim(v1_obs, v1_model)
    medium_coherence = cosine_sim(w_empirical, v_dominant_J)

    obs_ratio = obs_eigvals[idx[0]] / obs_eigvals[idx[1]] if obs_eigvals[idx[1]] > 0 else np.inf
    mod_ratio = mod_eigvals[idx_m[0]] / mod_eigvals[idx_m[1]] if mod_eigvals[idx_m[1]] > 0 else np.inf
    sign_discipline = bool(np.all(w_empirical > 0))

    coupling_totals = np.abs(glv.A).sum(axis=0) - np.abs(np.diag(glv.A))
    cs_corr, cs_p = stats.pearsonr(np.abs(w_empirical), coupling_totals)
    model_pc1_var = mod_eigvals[idx_m[0]] / mod_eigvals.sum()

    return {
        "weak_coherence": weak_coherence, "medium_coherence": medium_coherence,
        "obs_eigenratio": obs_ratio, "model_eigenratio": mod_ratio,
        "sign_discipline": sign_discipline, "cs_correlation": cs_corr,
        "cs_p_value": cs_p, "model_pc1_var": model_pc1_var,
        "v1_obs": v1_obs, "v1_model": v1_model,
        "v_dominant_J": v_dominant_J, "w_empirical": w_empirical,
        "Sigma_model": Sigma_model,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 4: INDIVIDUAL-LEVEL CAPABILITIES-ADJUSTED FLOURISHING
# ═══════════════════════════════════════════════════════════════════════════════

def individual_alkire_foster(df, cols, threshold_quantile=0.25, af_cutoff=1.0/3.0):
    X = df[cols].values.astype(float)
    N, K = X.shape
    thresholds = np.quantile(X, threshold_quantile, axis=0)
    above_threshold = X >= thresholds[np.newaxis, :]
    deprived = (~above_threshold).astype(float)
    nussbaum_pass = above_threshold.all(axis=1).astype(float)
    w = np.ones(K) / K
    deprivation_score = deprived @ w
    identified = (deprivation_score >= af_cutoff).astype(float)
    censored_score = deprivation_score * identified
    H = identified.mean()
    A = censored_score[identified == 1].mean() if identified.sum() > 0 else 0.0
    M0 = H * A
    dim_contribution = (deprived * identified[:, np.newaxis]).mean(axis=0)
    dim_contrib_pct = dim_contribution / (dim_contribution.sum() + 1e-8)
    row_means = X.mean(axis=1); row_stds = X.std(axis=1)
    cv = row_stds / (row_means + 1e-8)
    h = 1.0 - cv / (cv.max() + 1e-8); h = np.clip(h, 0.01, 1.0)
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    f_raw = pca.fit_transform(X_std).ravel()
    if f_raw.mean() < 0: f_raw = -f_raw
    f_z = (f_raw - f_raw.mean()) / (f_raw.std() + 1e-8)
    f_positive = f_raw - f_raw.min() + 0.01
    f_cap = f_positive * nussbaum_pass * h
    f_cap_z = (f_cap - f_cap.mean()) / (f_cap.std() + 1e-8)
    compensation_gap = f_z - f_cap_z

    result_df = pd.DataFrame({
        "f_z": f_z, "f_cap_z": f_cap_z, "compensation_gap": compensation_gap,
        "nussbaum_pass": nussbaum_pass, "n_deprived_dims": deprived.sum(axis=1),
        "deprivation_score": deprivation_score, "af_identified": identified,
        "capability_h": h,
    })
    for j, col in enumerate(cols):
        result_df[f"deprived_{col}"] = deprived[:, j]

    summary = {
        "thresholds": dict(zip(cols, thresholds)),
        "H": H, "A": A, "M0": M0,
        "dim_contribution": dict(zip(cols, dim_contrib_pct)),
        "n_pass": int(nussbaum_pass.sum()),
        "n_identified": int(identified.sum()),
        "mean_gap_pass": compensation_gap[nussbaum_pass == 1].mean(),
        "mean_gap_fail": compensation_gap[nussbaum_pass == 0].mean(),
        "f_fcap_corr": float(np.corrcoef(f_z, f_cap_z)[0, 1]),
    }
    return result_df, summary


def plot_individual_alkire_foster(af_df, af_summary, cols):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    labels = [short_label(c) for c in cols]
    dep_colors = [palette_color(c) for c in cols]

    ax = axes[0, 0]
    pm = af_df["nussbaum_pass"] == 1
    ax.scatter(af_df.loc[pm, "f_z"], af_df.loc[pm, "f_cap_z"],
               alpha=0.15, s=8, color="#2EAA6A", label="Pass all", edgecolors="none")
    ax.scatter(af_df.loc[~pm, "f_z"], af_df.loc[~pm, "f_cap_z"],
               alpha=0.15, s=8, color="#E8622A", label="Below ≥1", edgecolors="none")
    ax.plot([-4, 4], [-4, 4], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("f (compensatory)"); ax.set_ylabel("f_cap (capability-adjusted)")
    ax.set_title(f"(a) f vs f_cap  |  r = {af_summary['f_fcap_corr']:.3f}")
    ax.legend(fontsize=7, markerscale=3)

    ax = axes[0, 1]
    n_dep_vals = sorted(af_df["n_deprived_dims"].unique())
    gap_means = [af_df.loc[af_df["n_deprived_dims"] == nd, "compensation_gap"].mean() for nd in n_dep_vals]
    gap_sems = [af_df.loc[af_df["n_deprived_dims"] == nd, "compensation_gap"].sem() for nd in n_dep_vals]
    colors_b = ["#2EAA6A" if nd == 0 else "#E8622A" for nd in n_dep_vals]
    ax.bar([str(int(nd)) for nd in n_dep_vals], gap_means, yerr=gap_sems,
           color=colors_b, alpha=0.8, capsize=4)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Number of dimensions below threshold")
    ax.set_ylabel("Compensation gap (f − f_cap)")
    ax.set_title("(b) Compensation Gap by Deprivation Breadth")

    ax = axes[1, 0]
    dep_rates = [af_df[f"deprived_{c}"].mean() * 100 for c in cols]
    ax.barh(labels, dep_rates, color=dep_colors, alpha=0.8)
    ax.set_xlabel("% below threshold"); ax.set_title("(c) Per-Dimension Deprivation Rates")

    ax = axes[1, 1]
    domain_gaps = []
    for c in cols:
        dep_mask = af_df[f"deprived_{c}"] == 1
        gd = af_df.loc[dep_mask, "compensation_gap"].mean() if dep_mask.sum() > 0 else 0
        gn = af_df.loc[~dep_mask, "compensation_gap"].mean()
        domain_gaps.append(gd - gn)
    ax.barh(labels, domain_gaps, color=dep_colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Excess compensation gap when deprived")
    ax.set_title("(d) Which Domains Drive f ↔ f_cap Divergence?")

    fig.suptitle(f"Individual-Level Alkire-Foster Capabilities\n"
                 f"N = {len(af_df):,}  |  M₀ = {af_summary['M0']:.3f}  |  "
                 f"H = {af_summary['H']:.3f}", fontsize=12, y=1.03)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  THREE-MODEL COMPARISON: GLV vs LINEAR vs LOGISTIC
# ═══════════════════════════════════════════════════════════════════════════════

def three_model_shock_comparison(glv, diag, cols, shock_domain=5,
                                  shock_fracs=[0.3, 0.5, 0.7]):
    """
    Compare GLV, Linear (OU), and Coupled Logistic recovery from shocks
    to a specified domain (default: Self-Acceptance = index 5).
    """
    K = glv.K
    mu = diag["mu"]
    J = diag["J"]
    M_log = diag["M_logistic"]

    # Build the three systems
    sys_glv = glv
    sys_lin = LinearOUSystem(J, mu)

    # Logistic parametrisation
    C_log = np.array([mu[k] - sum(M_log[k, l] * mu[l] for l in range(K) if l != k)
                       for k in range(K)])
    a_log = -np.diag(J) * C_log / mu
    sys_logistic = CoupledLogisticSystem(K, a_log, C_log, M_log)

    systems = [
        (sys_logistic, "Logistic (pedagogical)"),
        (sys_glv, "GLV (primary)"),
        (sys_lin, "Linear (OU)"),
    ]

    t_span = (0, 40)
    t_eval = np.linspace(0, 40, 400)
    colors = [palette_color(c) for c in cols]
    labels_s = [short_label(c) for c in cols]

    fig, axes = plt.subplots(len(shock_fracs), 3, figsize=(16, 4 * len(shock_fracs)),
                              sharex=True)

    for si, sf in enumerate(shock_fracs):
        x0 = mu.copy()
        x0[shock_domain] = mu[shock_domain] * (1 - sf)
        for sj, (sys, sn) in enumerate(systems):
            ax = axes[si, sj]
            try:
                sol = solve_ivp(sys.rhs, t_span, x0, method='RK45',
                                max_step=0.2, t_eval=t_eval, rtol=1e-8, atol=1e-10)
                diverged = np.any(np.abs(sol.y) > 100) or np.any(~np.isfinite(sol.y))
                for k in range(K):
                    dev = (sol.y[k] - mu[k]) / mu[k] * 100
                    ax.plot(sol.t, dev, color=colors[k], lw=1.5,
                            label=labels_s[k] if si == 0 and sj == 0 else None)
                ax.axhline(0, color='gray', ls=':', lw=0.5)
                ax.set_ylim(-80, 30)
                if diverged:
                    ax.text(0.5, 0.9, 'DIVERGED', transform=ax.transAxes,
                            fontsize=12, color='red', ha='center', fontweight='bold')
            except Exception:
                ax.text(0.5, 0.5, 'FAILED', transform=ax.transAxes,
                        fontsize=14, color='red', ha='center')
            if si == 0: ax.set_title(sn, fontsize=10, fontweight='bold')
            if sj == 0: ax.set_ylabel(f'{int(sf*100)}% shock\n% deviation', fontsize=9)
            if si == len(shock_fracs) - 1: ax.set_xlabel('Time', fontsize=10)

    axes[0, 0].legend(fontsize=7, ncol=3, loc='lower right')
    fig.suptitle(f'{short_label(cols[shock_domain])} Shock Recovery: '
                 f'Logistic vs GLV vs Linear',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig, {"C_logistic": C_log, "a_logistic": a_log}


# ═══════════════════════════════════════════════════════════════════════════════
#  MUTUALISM DEMO PLOT (GLV)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_glv_demo(glv, equilibria, cols):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels_s = [short_label(c) for c in cols]
    colors = [palette_color(c) for c in cols]

    # (a) Trajectory convergence
    ax = axes[0]
    x_star = glv.equilibrium()
    x0_traj = x_star * 0.3
    sol = solve_ivp(glv.rhs, (0, 60), x0_traj, method="RK45", max_step=1.0,
                    t_eval=np.linspace(0, 60, 300), rtol=1e-8, atol=1e-10)
    for k in range(glv.K):
        ax.plot(sol.t, sol.y[k], color=colors[k], lw=1.8, label=labels_s[k])
        ax.axhline(x_star[k], color=colors[k], ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Time (t)"); ax.set_ylabel("Domain level x_k(t)")
    ax.set_title("(a) GLV Convergence to Equilibrium")
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    # (b) Population correlation heatmap
    ax = axes[1]
    eq_std = (equilibria - equilibria.mean(axis=0)) / (equilibria.std(axis=0) + 1e-8)
    cov_pop = np.corrcoef(eq_std.T)
    im = ax.imshow(cov_pop, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cov_pop[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(cov_pop[i,j]) > 0.6 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels_s, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels_s, fontsize=8)
    ax.set_title("(b) Simulated Population Correlations\nPositive manifold, no latent variable")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (c) Eigenvalue spectrum
    ax = axes[2]
    cov_raw = np.cov(equilibria.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_raw))[::-1]
    var_explained = eigenvalues / eigenvalues.sum() * 100
    comps = np.arange(1, len(eigenvalues) + 1)
    bars = ax.bar(comps, var_explained, color="#3B7DD8", alpha=0.7, edgecolor="white")
    bars[0].set_color("#E8622A")
    ax.set_xlabel("Component"); ax.set_ylabel("% Variance Explained")
    ax.set_title(f"(c) Eigenvalue Spectrum\nPC1 = {var_explained[0]:.1f}%")
    ax.set_xticks(comps); ax.axhline(100/len(cols), color="gray", ls="--", lw=1)

    fig.suptitle("Model 2: GLV Dynamical System (Lyapunov Inversion Calibration)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  COUPLING SWEEP & RESILIENCE SIGNATURE (GLV-based)
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_sweep(glv, n_points=15, n_individuals=80):
    """Sweep off-diagonal coupling while holding self-limitation fixed.

    This is the correct parameterization for the resilience tradeoff:
    - Diagonal A_kk (self-limitation) stays fixed
    - Off-diagonal A_kl (inter-domain coupling) scales by multiplier α
    - r(α) = -A(α) · μ recomputed at each step to pin x* = μ

    At low α: domains are nearly independent → large basin, slow cross-domain
              recovery, low PC1 variance
    At high α: domains tightly coupled → fast recovery from small shocks,
               small basin (fragile to large shocks), high PC1 variance
    """
    K = glv.K
    D = np.diag(np.diag(glv.A))  # diagonal (self-limitation, all negative)
    O = glv.A - D                 # off-diagonal (coupling)

    # Sweep multiplier: dense near α=1, include exact calibrated system
    # The stability boundary varies by dataset; sample densely in [0.1, 2.0]
    alphas = np.unique(np.sort(np.concatenate([
        np.linspace(0.1, 0.9, 5),       # low coupling
        [1.0],                            # exact calibrated system
        np.linspace(1.1, 2.0, 5),        # near/past boundary
        np.linspace(2.5, 5.0, 4),        # high coupling
    ])))
    n_points = len(alphas)  # override
    results = {"m_bar": [], "alpha": [], "pct_var_pc1": [], "mean_eq": [],
               "resilience": [], "recovery_speed": [], "basin_radius": [],
               "stable": []}

    for alpha in alphas:
        A_scaled = D + alpha * O
        r_scaled = -A_scaled @ glv.mu

        glv_test = GLVSystem(K, r_scaled, A_scaled, glv.mu)

        # Population simulation
        equilibria = glv_test.simulate_population(n_individuals=n_individuals, seed=42)
        cov = np.cov(equilibria.T)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        pct_pc1 = evals[0] / evals.sum() * 100

        # Stability analysis
        try:
            x_star = glv_test.equilibrium()
            J_test = glv_test.jacobian_at(x_star)
            test_eigs = np.real(eigvals(J_test))
            stable = bool(np.all(test_eigs < 0))

            if stable:
                lyap = lyapunov_analysis(glv_test, x_star)
                resilience = lyap["resilience"]
                recovery_speed = lyap["recovery_speed"]
                basin_radius = lyap["basin_radius"]
            else:
                resilience = recovery_speed = basin_radius = 0.0
        except Exception:
            stable = False
            resilience = recovery_speed = basin_radius = 0.0

        m_bar = glv_test.mean_coupling_strength()
        results["m_bar"].append(m_bar)
        results["alpha"].append(alpha)
        results["pct_var_pc1"].append(pct_pc1)
        results["mean_eq"].append(equilibria.mean())
        results["resilience"].append(resilience)
        results["recovery_speed"].append(recovery_speed)
        results["basin_radius"].append(basin_radius)
        results["stable"].append(stable)

    return pd.DataFrame(results)


def coupling_resilience_signature(glv, n_individuals=50, shock_magnitudes=None, seed=42):
    if shock_magnitudes is None:
        shock_magnitudes = np.array([0.05, 0.10, 0.20, 0.35, 0.50])
    # Use multipliers that span from weak to past stability boundary
    coupling_alphas = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
    rng = np.random.RandomState(seed)
    K = glv.K; T_MAX = 100.0; RECOVERY_THRESHOLD = 0.05

    D = np.diag(np.diag(glv.A))
    O = glv.A - D

    trial_records = []
    for alpha in coupling_alphas:
        A_scaled = D + alpha * O
        r_scaled = -A_scaled @ glv.mu
        m_bar = GLVSystem(K, r_scaled, A_scaled, glv.mu).mean_coupling_strength()

        for shock_frac in shock_magnitudes:
            for trial in range(n_individuals):
                r_i = r_scaled * np.exp(rng.normal(0, 0.10, K))
                A_i = A_scaled * np.exp(rng.normal(0, 0.03, (K, K)))

                glv_i = GLVSystem(K, r_i, A_i, glv.mu)
                try:
                    x_star_i = glv_i.equilibrium()
                except Exception:
                    continue
                if not (np.all(x_star_i > 0) and np.all(np.isfinite(x_star_i))):
                    continue

                shock_domain = rng.randint(K)
                x_perturbed = x_star_i.copy()
                x_perturbed[shock_domain] *= (1.0 - shock_frac)
                x_perturbed = np.maximum(x_perturbed, 0.01)

                try:
                    sol = solve_ivp(glv_i.rhs, (0, T_MAX), x_perturbed,
                                    method="RK45", max_step=1.0,
                                    t_eval=np.linspace(0, T_MAX, 500), rtol=1e-6, atol=1e-8)
                    deviations = np.abs(sol.y.T - x_star_i) / (x_star_i + 1e-8)
                    max_dev = deviations.max(axis=1)
                    recovered_idx = np.where(max_dev < RECOVERY_THRESHOLD)[0]
                    if len(recovered_idx) > 0:
                        recovery_time = sol.t[recovered_idx[0]]; recovered = True
                    else:
                        recovery_time = T_MAX; recovered = False
                    final_dev = max_dev[-1]
                except Exception:
                    recovery_time = T_MAX; recovered = False; final_dev = 1.0

                trial_records.append({
                    "coupling": m_bar, "alpha": alpha, "shock_frac": shock_frac,
                    "recovery_time": recovery_time, "recovered": recovered,
                    "final_deviation": final_dev,
                })

    trial_df = pd.DataFrame(trial_records)
    summary = trial_df.groupby(["coupling", "shock_frac"]).agg(
        mean_recovery=("recovery_time", "mean"),
        var_recovery=("recovery_time", "var"),
        recovery_rate=("recovered", "mean"),
        n_trials=("recovered", "count"),
    ).reset_index()
    return trial_df, summary


def plot_coupling_sweep(sweep_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Identify stable region
    stable = sweep_df["stable"] if "stable" in sweep_df.columns else [True]*len(sweep_df)
    s_df = sweep_df[sweep_df["stable"]] if "stable" in sweep_df.columns else sweep_df
    u_df = sweep_df[~sweep_df["stable"]] if "stable" in sweep_df.columns else pd.DataFrame()

    # Mark calibrated system (α ≈ 1)
    alpha_col = "alpha" if "alpha" in sweep_df.columns else None

    ax = axes[0, 0]
    ax.plot(s_df["m_bar"], s_df["pct_var_pc1"], "o-", color="#3B7DD8", lw=2, label="Stable")
    if len(u_df) > 0:
        ax.plot(u_df["m_bar"], u_df["pct_var_pc1"], "x", color="#d7191c", ms=8, label="Unstable")
    ax.set_xlabel("Mean coupling |A_offdiag|"); ax.set_ylabel("PC1 variance (%)")
    ax.set_title("(a) Coupling → f Dominance")
    if alpha_col:
        cal_row = sweep_df.iloc[(sweep_df["alpha"] - 1.0).abs().argmin()]
        ax.axvline(cal_row["m_bar"], color="gray", ls="--", lw=1, alpha=0.6, label="Calibrated (α=1)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(s_df["m_bar"], s_df["mean_eq"], "o-", color="#2EAA6A", lw=2)
    if len(u_df) > 0:
        ax.plot(u_df["m_bar"], u_df["mean_eq"], "x", color="#d7191c", ms=8)
    ax.set_xlabel("Mean coupling |A_offdiag|"); ax.set_ylabel("Mean equilibrium")
    ax.set_title("(b) Coupling → Wellbeing Level")
    if alpha_col:
        ax.axvline(cal_row["m_bar"], color="gray", ls="--", lw=1, alpha=0.6)

    ax = axes[1, 0]
    ax.plot(s_df["m_bar"], s_df["resilience"], "o-", color="#E67E22", lw=2)
    if len(u_df) > 0:
        ax.plot(u_df["m_bar"], [0]*len(u_df), "x", color="#d7191c", ms=8, label="Unstable")
    ax.set_xlabel("Mean coupling |A_offdiag|"); ax.set_ylabel("Resilience R(x*)")
    ax.set_title("(c) Coupling → Resilience")
    if s_df["resilience"].max() > 0:
        peak_idx = s_df["resilience"].idxmax()
        ax.axvline(s_df.loc[peak_idx, "m_bar"], color="red", ls=":", lw=1.5, label="Peak")
    if alpha_col:
        ax.axvline(cal_row["m_bar"], color="gray", ls="--", lw=1, alpha=0.6)
    ax.legend(fontsize=7)

    ax = axes[1, 1]; ax2 = ax.twinx()
    l1, = ax.plot(s_df["m_bar"], s_df["recovery_speed"], "o-", color="#3B7DD8", lw=2, label="Recovery speed")
    l2, = ax2.plot(s_df["m_bar"], s_df["basin_radius"], "s-", color="#E8622A", lw=2, label="Basin radius")
    ax.set_xlabel("Mean coupling |A_offdiag|"); ax.set_ylabel("Recovery speed", color="#3B7DD8")
    ax2.set_ylabel("Basin radius", color="#E8622A")
    ax.set_title("(d) The Resilience Trade-off"); ax.legend(handles=[l1, l2], fontsize=8)
    if alpha_col:
        ax.axvline(cal_row["m_bar"], color="gray", ls="--", lw=1, alpha=0.6)

    fig.suptitle("Coupling Sweep — Off-Diagonal Scaling (GLV, Fixed Self-Limitation)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_coupling_resilience_signature(summary_df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    shock_vals = sorted(summary_df["shock_frac"].unique())
    cmap = plt.cm.RdYlBu_r
    norm_v = plt.Normalize(min(shock_vals), max(shock_vals))
    for sf in shock_vals:
        sub = summary_df[summary_df["shock_frac"] == sf]
        axes[0].plot(sub["coupling"], sub["mean_recovery"], "o-",
                     color=cmap(norm_v(sf)), lw=1.8, label=f"{sf*100:.0f}%")
        axes[1].plot(sub["coupling"], sub["var_recovery"], "s-",
                     color=cmap(norm_v(sf)), lw=1.8, label=f"{sf*100:.0f}%")
        axes[2].plot(sub["coupling"], sub["recovery_rate"] * 100, "^-",
                     color=cmap(norm_v(sf)), lw=1.8, label=f"{sf*100:.0f}%")
    axes[0].set_title("(a) Mean Recovery Time")
    axes[1].set_title("(b) Recovery Variance — Testable Signature")
    axes[2].set_title("(c) Recovery Success Rate"); axes[2].set_ylim(-5, 105)
    for ax in axes:
        ax.set_xlabel("Coupling M̄"); ax.legend(fontsize=7, ncol=2, title="Shock")
    fig.suptitle("Coupling–Resilience Signature (GLV, Lyapunov Inversion)",
                 fontsize=11, y=1.04)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-PERSON RESILIENCE (GLV-based)
# ═══════════════════════════════════════════════════════════════════════════════

def person_resilience_scores(X_raw, glv, cols):
    """Compute per-person resilience by back-solving r from observed scores."""
    N, K = X_raw.shape
    results = []
    for i in range(N):
        x_i = X_raw[i]
        # Person's A is same as population; r_i solves r_i = -A @ x_i
        r_i = -glv.A @ x_i
        glv_i = GLVSystem(K, r_i, glv.A.copy(), x_i)
        try:
            lyap = lyapunov_analysis(glv_i, x_i)
            results.append({
                "lambda_1_real": np.real(lyap["lambda_1"]),
                "tau": lyap["tau"], "recovery_speed": lyap["recovery_speed"],
                "basin_radius": lyap["basin_radius"], "resilience": lyap["resilience"],
                "stable": lyap["stable"], "mean_coupling": glv_i.mean_coupling_strength(),
            })
        except Exception:
            results.append({
                "lambda_1_real": np.nan, "tau": np.nan, "recovery_speed": np.nan,
                "basin_radius": np.nan, "resilience": np.nan,
                "stable": False, "mean_coupling": np.nan,
            })
    return pd.DataFrame(results)


def plot_resilience_vs_f(df_scored, resilience_df, cols):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    valid = resilience_df["resilience"].notna() & (resilience_df["resilience"] > 0)
    if valid.sum() > 5:
        ax = axes[0]
        ax.scatter(df_scored.loc[valid.values, "f_score"],
                   resilience_df.loc[valid, "resilience"],
                   alpha=0.1, s=8, color="#3B7DD8", edgecolors="none")
        r, p = stats.pearsonr(df_scored.loc[valid.values, "f_score"],
                               resilience_df.loc[valid, "resilience"])
        ax.set_xlabel("f score (z)"); ax.set_ylabel("Resilience R(x*)")
        ax.set_title(f"(a) f vs Resilience\nr = {r:.3f}, p = {p:.3e}")
    valid_tau = resilience_df["tau"].notna() & (resilience_df["tau"] < 100)
    if valid_tau.sum() > 5:
        ax = axes[1]
        ax.scatter(df_scored.loc[valid_tau.values, "f_score"],
                   resilience_df.loc[valid_tau, "tau"],
                   alpha=0.1, s=8, color="#E67E22", edgecolors="none")
        ax.set_xlabel("f score (z)"); ax.set_ylabel("Return time τ")
        ax.set_title("(b) f vs Return Time τ")
    valid_lam = resilience_df["lambda_1_real"].notna()
    if valid_lam.sum() > 5:
        ax = axes[2]
        ax.scatter(df_scored.loc[valid_lam.values, "f_score"],
                   resilience_df.loc[valid_lam, "lambda_1_real"],
                   alpha=0.1, s=8, color="#2EAA6A", edgecolors="none")
        r_l, p_l = stats.pearsonr(df_scored.loc[valid_lam.values, "f_score"],
                                   resilience_df.loc[valid_lam, "lambda_1_real"])
        ax.set_xlabel("f score (z)"); ax.set_ylabel("Dominant λ₁ (most negative = slowest)")
        ax.set_title(f"(c) f vs Dominant Eigenvalue λ₁\nr = {r_l:.3f}")
        ax.axhline(0, color="red", ls="--", lw=0.8, alpha=0.5)
    fig.suptitle("Predicted Relationships — Coupling, Resilience, f\n"
                 "(GLV, Lyapunov Inversion)", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  ECONOMIC VALIDATION: f predicts economic outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def construct_economic_variables(df):
    """
    Build derived economic variables from raw MIDUS 2 columns.
    Returns a new DataFrame aligned to df.index with columns:
        employed, log_income, disability_days, er_visits, health_limits, age
    All coded so that HIGHER = BETTER OUTCOME for employed/income and
    HIGHER = WORSE OUTCOME for disability/ER/health_limits.
    """
    out = pd.DataFrame(index=df.index)

    # -- Employment (B1PB19: 1=working) --
    if "B1PB19" in df.columns:
        v = df["B1PB19"]
        out["employed"] = np.where(v == 1, 1.0, np.where(v.isin([2,3,4,5]), 0.0, np.nan))

    # -- Log income (B1STINC1; -1 and >=9999997 are sentinels) --
    if "B1STINC1" in df.columns:
        v = df["B1STINC1"].copy()
        v = v.where((v >= 0) & (v < 9999997))
        # Keep zeros (real zero income), log(1+x)
        out["log_income"] = np.log1p(v)

    # -- Disability days (gate B1PA24 + count B1PA24A) --
    if "B1PA24" in df.columns and "B1PA24A" in df.columns:
        gate = df["B1PA24"]
        days = df["B1PA24A"]
        # If gate=2 (no), days=0; if gate=1 (yes) and days in 1-30, use count
        d = np.full(len(df), np.nan)
        d[gate == 2] = 0.0
        valid_days = (gate == 1) & (days >= 1) & (days <= 30)
        d[valid_days] = days[valid_days].values
        out["disability_days"] = d

    # -- ER visits (B1PA52; 99=skip/none, 97=DK) --
    if "B1PA52" in df.columns:
        v = df["B1PA52"].copy()
        er = np.full(len(df), np.nan)
        er[v == 99] = 0.0          # skip = no ER visits
        valid = (v >= 1) & (v <= 50)
        er[valid] = v[valid].values
        er[v == 97] = np.nan       # DK -> missing
        out["er_visits"] = er

    # -- Health limitations (B1PA25A: 1=a lot, 2=not at all, 3=some, 4=a little) --
    if "B1PA25A" in df.columns:
        v = df["B1PA25A"]
        recode = {2: 0, 4: 1, 3: 2, 1: 3}  # 0=none .. 3=a lot
        out["health_limits"] = v.map(recode)

    # -- Age for controls --
    if "B1PAGE_M2" in df.columns:
        out["age"] = pd.to_numeric(df["B1PAGE_M2"], errors="coerce")

    return out


def economic_validation(df, econ_df):
    """
    Correlations, partial correlations (age-controlled), incremental R^2,
    employment Cohen's d, and working-age subsample analysis.
    """
    from sklearn.linear_model import LinearRegression

    results = {"full": {}, "working_age": {}}
    has_f = "f_score" in df.columns
    has_negaf = "B1SNEGAF" in df.columns and df["B1SNEGAF"].notna().sum() > 50
    if not has_f:
        print("  f_score not found -- skipping.")
        return None

    merged = econ_df.copy()
    merged["f_score"] = df["f_score"].values
    if has_negaf:
        merged["negaf"] = df["B1SNEGAF"].values

    def _run_correlations(sub, label):
        """Run full correlation + incremental analysis on a subsample."""
        rows = []
        for evar, meta in ECON_DERIVED.items():
            if evar not in sub.columns:
                continue
            pair = sub[["f_score", evar]].dropna()
            n = len(pair)
            if n < 50:
                continue

            r_f, p_f = stats.pearsonr(pair["f_score"], pair[evar])
            rho_f, p_rho = stats.spearmanr(pair["f_score"], pair[evar])

            # Partial correlation controlling for age
            r_partial, p_partial = np.nan, np.nan
            if "age" in sub.columns:
                trip = sub[["f_score", evar, "age"]].dropna()
                if len(trip) > 50:
                    from numpy.linalg import lstsq
                    X_age = trip[["age"]].values
                    ones = np.ones((len(trip), 1))
                    Xa = np.hstack([X_age, ones])
                    res_f = trip["f_score"].values - Xa @ lstsq(Xa, trip["f_score"].values, rcond=None)[0]
                    res_y = trip[evar].values - Xa @ lstsq(Xa, trip[evar].values, rcond=None)[0]
                    r_partial, p_partial = stats.pearsonr(res_f, res_y)

            # Negative affect correlation
            r_neg, p_neg = np.nan, np.nan
            if has_negaf:
                neg_pair = sub[["negaf", evar]].dropna()
                if len(neg_pair) > 50:
                    r_neg, p_neg = stats.pearsonr(neg_pair["negaf"], neg_pair[evar])

            rows.append({
                "variable": evar, "label": meta["label"],
                "expected_sign": meta["exp_sign"],
                "n": n,
                "r_f": r_f, "p_f": p_f,
                "rho_f": rho_f,
                "r_partial": r_partial, "p_partial": p_partial,
                "r_negaf": r_neg, "p_negaf": p_neg,
                "sign_ok": np.sign(r_f) == meta["exp_sign"],
            })

        corr_df = pd.DataFrame(rows)

        # Incremental R^2
        incr_rows = []
        if has_negaf:
            for evar, meta in ECON_DERIVED.items():
                if evar == "employed" or evar not in sub.columns:
                    continue
                trip = sub[["f_score", "negaf", evar]].dropna()
                if len(trip) < 50:
                    continue
                y = trip[evar].values
                X_neg = trip[["negaf"]].values
                X_both = trip[["negaf", "f_score"]].values
                r2_a = LinearRegression().fit(X_neg, y).score(X_neg, y)
                reg_b = LinearRegression().fit(X_both, y)
                r2_b = reg_b.score(X_both, y)
                incr = r2_b - r2_a
                n_obs = len(trip)
                df_den = n_obs - 3
                f_stat = (incr / 1) / ((1 - r2_b) / df_den) if r2_b < 1 and df_den > 0 else np.nan
                p_incr = 1 - stats.f.cdf(f_stat, 1, df_den) if not np.isnan(f_stat) else np.nan
                incr_rows.append({
                    "variable": evar, "label": meta["label"], "n": n_obs,
                    "R2_negaf": r2_a, "R2_both": r2_b,
                    "incr_R2": incr, "F": f_stat, "p": p_incr,
                    "beta_f": reg_b.coef_[1],
                })

                # Also: f alone vs negaf alone
                r2_f_only = LinearRegression().fit(trip[["f_score"]].values, y).score(trip[["f_score"]].values, y)
                incr_rows[-1]["R2_f_only"] = r2_f_only

        incr_df = pd.DataFrame(incr_rows)

        # Employment effect size
        emp_info = None
        if "employed" in sub.columns:
            e_sub = sub[["f_score", "employed"]].dropna()
            if len(e_sub) > 50 and e_sub["employed"].nunique() == 2:
                f_yes = e_sub.loc[e_sub["employed"] == 1, "f_score"]
                f_no = e_sub.loc[e_sub["employed"] == 0, "f_score"]
                psd = np.sqrt(((len(f_yes)-1)*f_yes.var() + (len(f_no)-1)*f_no.var()) /
                              (len(f_yes)+len(f_no)-2))
                d_f = (f_yes.mean() - f_no.mean()) / (psd + 1e-10)
                d_neg = np.nan
                if has_negaf:
                    en = sub[["negaf", "employed"]].dropna()
                    if len(en) > 50:
                        ny, nn = en.loc[en["employed"]==1,"negaf"], en.loc[en["employed"]==0,"negaf"]
                        psd_n = np.sqrt(((len(ny)-1)*ny.var()+(len(nn)-1)*nn.var())/(len(ny)+len(nn)-2))
                        d_neg = (ny.mean()-nn.mean()) / (psd_n+1e-10)
                emp_info = {
                    "n": len(e_sub),
                    "n_employed": int(e_sub["employed"].sum()),
                    "n_not": int((1-e_sub["employed"]).sum()),
                    "d_f": d_f, "d_negaf": d_neg,
                    "f_mean_emp": f_yes.mean(),
                    "f_mean_not": f_no.mean(),
                }

        return {"correlations": corr_df, "incremental": incr_df, "employment": emp_info}

    # Full sample
    results["full"] = _run_correlations(merged, "Full sample")

    # Working-age subsample (25-64)
    if "age" in merged.columns:
        wa = merged[(merged["age"] >= 25) & (merged["age"] <= 64)].copy()
        if len(wa) > 200:
            results["working_age"] = _run_correlations(wa, "Working age (25-64)")
            results["n_working_age"] = len(wa)

    return results


def plot_economic_validation(df, econ_df, econ_results):
    """
    Four-panel figure:
      (a) Bivariate r(f, econ) vs r(negaf, econ) -- working-age if available
      (b) Incremental R^2 bar chart
      (c) f by employment status (working-age)
      (d) f vs log(income) scatter
    """
    if econ_results is None:
        return None

    # Prefer working-age results where available
    res = econ_results.get("working_age", econ_results.get("full", {}))
    full_res = econ_results.get("full", {})
    if not res:
        return None

    corr_df = res.get("correlations", pd.DataFrame())
    incr_df = res.get("incremental", pd.DataFrame())
    emp_info = res.get("employment", None)

    merged = econ_df.copy()
    merged["f_score"] = df["f_score"].values

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # -- (a) Correlation comparison --
    ax = axes[0]
    if len(corr_df) > 0:
        labels = corr_df["label"].values
        r_f = corr_df["r_f"].values
        r_neg = corr_df["r_negaf"].values
        r_par = corr_df["r_partial"].values

        y = np.arange(len(labels))
        bh = 0.25
        ax.barh(y - bh, r_f, bh, label="r(f, y)", color="#2EAA6A", alpha=0.85)
        if not np.all(np.isnan(r_neg)):
            ax.barh(y, r_neg, bh, label="r(neg.aff, y)", color="#E8622A", alpha=0.85)
        if not np.all(np.isnan(r_par)):
            ax.barh(y + bh, r_par, bh, label="r(f, y | age)", color="#3B7DD8", alpha=0.85)

        for i, row in corr_df.iterrows():
            sig = "***" if row["p_f"]<.001 else "**" if row["p_f"]<.01 else "*" if row["p_f"]<.05 else ""
            x = row["r_f"]
            ax.text(x + 0.008*np.sign(x), i-bh, sig, va="center",
                    ha="left" if x>0 else "right", fontsize=7, color="#2EAA6A", fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.legend(fontsize=7, loc="best")
        ax.invert_yaxis()
        wa_tag = " (25-64)" if "working_age" in econ_results else ""
        ax.set_title(f"(a) Economic Correlates{wa_tag}", fontsize=10)
        ax.set_xlabel("Pearson r")

    # -- (b) Incremental R^2 --
    ax = axes[1]
    if len(incr_df) > 0:
        labels = incr_df["label"].values
        y = np.arange(len(labels))
        bh = 0.55
        ax.barh(y, incr_df["R2_negaf"].values, bh, label=r"$R^2$ neg.aff only",
                color="#E8622A", alpha=0.55)
        ax.barh(y, incr_df["incr_R2"].values, bh, left=incr_df["R2_negaf"].values,
                label=r"$\Delta R^2$ adding f", color="#2EAA6A", alpha=0.85)
        for i, row in incr_df.iterrows():
            total = row["R2_both"]
            sig = "***" if row["p"]<.001 else "**" if row["p"]<.01 else "*" if row["p"]<.05 else ""
            ax.text(total+0.003, i, f"+{row['incr_R2']:.3f}{sig}", va="center",
                    fontsize=7, color="#2EAA6A", fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.legend(fontsize=7); ax.invert_yaxis()
        ax.set_title(r"(b) Incremental $R^2$: f Beyond Neg. Affect", fontsize=10)
        ax.set_xlabel(r"$R^2$")
    else:
        ax.text(0.5, 0.5, "No incremental\ndata available", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    # -- (c) f by employment (working-age preferred) --
    ax = axes[2]
    if emp_info and emp_info["n"] > 50:
        # Use working-age subsample
        if "age" in merged.columns:
            wa = merged[(merged["age"]>=25)&(merged["age"]<=64)].copy()
        else:
            wa = merged.copy()
        emask = wa[["f_score","employed"]].dropna().index
        emp = wa.loc[emask, "employed"]
        fs = wa.loc[emask, "f_score"]
        ax.hist(fs[emp==1], bins=40, alpha=0.65, density=True, color="#2EAA6A",
                label="Employed", edgecolor="white", lw=0.3)
        ax.hist(fs[emp==0], bins=40, alpha=0.65, density=True, color="#E8622A",
                label="Not employed", edgecolor="white", lw=0.3)
        ax.axvline(fs[emp==1].mean(), color="#2EAA6A", ls="--", lw=1.5)
        ax.axvline(fs[emp==0].mean(), color="#E8622A", ls="--", lw=1.5)
        d = emp_info["d_f"]
        ax.set_title(f"(c) f by Employment (25-64)\nCohen's d = {d:.3f}", fontsize=10)
        ax.set_xlabel("f score (z)"); ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Insufficient\nemployment data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    # -- (d) f vs log(income) --
    ax = axes[3]
    if "log_income" in merged.columns:
        if "age" in merged.columns:
            wa = merged[(merged["age"]>=25)&(merged["age"]<=64)].copy()
        else:
            wa = merged.copy()
        pair = wa[["f_score","log_income"]].dropna()
        # exclude zero income for scatter clarity
        pair = pair[pair["log_income"] > 0]
        if len(pair) > 50:
            r, p = stats.pearsonr(pair["f_score"], pair["log_income"])
            ax.hexbin(pair["f_score"], pair["log_income"], gridsize=35, cmap="YlGn", mincnt=1)
            sl, ic, *_ = stats.linregress(pair["f_score"], pair["log_income"])
            xr = np.linspace(pair["f_score"].min(), pair["f_score"].max(), 100)
            ax.plot(xr, sl*xr+ic, "k--", lw=1.5, alpha=0.8)
            ax.set_xlabel("f score (z)"); ax.set_ylabel("ln(1 + Income)")
            ax.set_title(f"(d) f vs. Income (25-64)\nr = {r:.3f}, p = {p:.2e}", fontsize=10)

    fig.suptitle("Economic Validation: f Predicts Outcomes Economists Care About",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL-LEVEL COUPLING PROXY & DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════════

def individual_coupling_resilience(df, cols):
    X = df[cols].values.astype(float)
    person_means = X.mean(axis=1); person_stds = X.std(axis=1)
    cv = person_stds / (person_means + 1e-8)
    coupling_proxy = 1.0 - (cv - cv.min()) / (cv.max() - cv.min() + 1e-8)
    weakest_domain = [cols[i] for i in X.argmin(axis=1)]
    return pd.DataFrame({
        "coupling_proxy": coupling_proxy, "person_mean": person_means,
        "person_std": person_stds, "cv": cv,
        "min_domain_score": X.min(axis=1), "max_domain_score": X.max(axis=1),
        "domain_range": X.max(axis=1) - X.min(axis=1), "weakest_domain": weakest_domain,
    })


def plot_demographic_f(df):
    if "f_score" not in df.columns: return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    if "B1PAGE_M2" in df.columns:
        df = df.copy()
        df["age_group"] = pd.cut(df["B1PAGE_M2"], bins=[34,44,54,64,74,90],
                                  labels=["35-44","45-54","55-64","65-74","75+"])
        sns.violinplot(data=df, x="age_group", y="f_score",
                       order=["35-44","45-54","55-64","65-74","75+"],
                       ax=axes[0], palette="Blues", inner="quartile", density_norm="width")
        axes[0].set_title("f by Age"); axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    if "B1PRSEX" in df.columns:
        df["sex_label"] = df["B1PRSEX"].map({1: "Male", 2: "Female"})
        sns.violinplot(data=df.dropna(subset=["sex_label"]),
                       x="sex_label", y="f_score", ax=axes[1],
                       palette={"Male": "#4A90D9", "Female": "#E05C8A"},
                       inner="quartile", density_norm="width")
        axes[1].set_title("f by Sex")
    if "B1PF7A" in df.columns:
        edu_bins = {range(1,5): "< HS", range(5,7): "HS", range(7,9): "Some college",
                    range(9,11): "BA", range(11,13): "Grad"}
        def edu_label(v):
            for r, lbl in edu_bins.items():
                if int(v) in r: return lbl
            return "Other"
        df["edu_group"] = df["B1PF7A"].dropna().apply(edu_label)
        sns.violinplot(data=df.dropna(subset=["edu_group"]),
                       x="edu_group", y="f_score",
                       order=["< HS","HS","Some college","BA","Grad"],
                       ax=axes[2], palette="Greens", inner="quartile", density_norm="width")
        axes[2].set_title("f by Education")
        plt.setp(axes[2].get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("f Scores by Demographic Group", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_f_vs_p(df):
    if "B1SNEGAF" not in df.columns or "f_score" not in df.columns: return None
    df_plot = df[["f_score", "B1SNEGAF"]].dropna()
    r, p_val = stats.pearsonr(df_plot["f_score"], df_plot["B1SNEGAF"])
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1:, :-1])
    ax_top   = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_main.hexbin(df_plot["f_score"], df_plot["B1SNEGAF"], gridsize=40, cmap="YlOrRd", mincnt=1)
    xs = np.linspace(df_plot["f_score"].min(), df_plot["f_score"].max(), 100)
    slope, intercept, *_ = stats.linregress(df_plot["f_score"], df_plot["B1SNEGAF"])
    ax_main.plot(xs, intercept + slope * xs, color="#3B7DD8", lw=2)
    ax_main.set_xlabel("f score (flourishing)"); ax_main.set_ylabel("Negative Affect (p proxy)")
    ax_main.text(0.97, 0.95,
                 f"r = {r:.3f}\nShared var: {r**2*100:.1f}%\nIndependent: {(1-r**2)*100:.1f}%",
                 transform=ax_main.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.85))
    ax_top.hist(df_plot["f_score"], bins=50, color="#3B7DD8", alpha=0.7); ax_top.axis("off")
    ax_right.hist(df_plot["B1SNEGAF"], bins=30, orientation="horizontal",
                  color="#E8622A", alpha=0.7); ax_right.axis("off")
    fig.suptitle("f vs p: Dual Continua Test", fontsize=12, y=1.0)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("  COMBINED f-FACTOR ANALYSIS v2")
    print("  GLV via Lyapunov Inversion · Asymmetric Coupling · Perturbation Decomposition")
    print("  'Toward f: A General Factor of Human Flourishing'")
    print("═" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    #  LOAD INDIVIDUAL-LEVEL DATA
    # ══════════════════════════════════════════════════════════════════════════
    print_section("Loading Individual-Level Data")
    if MODE == "synthetic":
        df_raw = generate_synthetic_midus(n=5000)
        data_label = "Synthetic MIDUS-like"
    elif MODE == "midus":
        try:
            df_raw = load_midus(MIDUS_DATA_PATH)
            data_label = "MIDUS 2 (ICPSR 4652)"
        except FileNotFoundError:
            print("  MIDUS data not found — falling back to synthetic.")
            df_raw = generate_synthetic_midus(n=5000)
            data_label = "Synthetic MIDUS-like (fallback)"
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")

    df_indiv, X_raw_indiv, X_scaled_indiv, indiv_cols = preprocess_midus(df_raw)
    N_indiv = len(df_indiv)
    S_indiv = np.cov(X_scaled_indiv.T, ddof=1)

    # ══════════════════════════════════════════════════════════════════════════
    #  POSITIVE MANIFOLD & PCA
    # ══════════════════════════════════════════════════════════════════════════
    print_section("Positive Manifold — Individual Level")
    corr = report_manifold(df_indiv, indiv_cols, label=data_label)
    fig_corr = plot_corr_heatmap(df_indiv, indiv_cols, title_suffix=data_label)
    fig_corr.savefig(OUTPUT_DIR / "01_correlation_matrix.png")
    print(f"  → {OUTPUT_DIR}/01_correlation_matrix.png")

    print_section("PCA — Extracting f (Individual Level)")
    df_scored, pca_indiv, var_ratio = run_pca_individual(
        X_scaled_indiv, indiv_cols, df_indiv, suffix=data_label)

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 1: BIFACTOR CFA
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 1: Bifactor CFA (Individual Level)")
    bif_result = fit_bifactor_cfa(S_indiv, N_indiv, indiv_cols, BIFACTOR_GROUPS_INDIV)
    print(f"  Converged: {bif_result['converged']}")
    print(f"  ω_h = {bif_result['omega_h']:.3f}")
    print(f"  ECV = {bif_result['ecv']:.3f}")
    print(f"  df = {bif_result['df']}, Chi² = {bif_result['chi2']:.2f}")
    if bif_result['df'] > 0:
        print(f"  RMSEA = {bif_result['RMSEA']:.4f}  (marginal at low df)")
        print(f"  CFI   = {bif_result['CFI']:.4f}")
        print(f"  TLI   = {bif_result['TLI']:.4f}")
        print(f"  SRMR  = {bif_result['SRMR']:.4f}")
        print(f"  Null model: Chi² = {bif_result['chi2_null']:.2f}, df = {bif_result['df_null']}")
    print(f"\n  General factor loadings (λ^f):")
    for c, lf in zip(indiv_cols, bif_result["lambda_f"]):
        print(f"    {short_label(c):<22} {lf:.3f}")
    print(f"\n  Specific factor loadings (λ^s):")
    for c, ls in zip(indiv_cols, bif_result["lambda_s"]):
        gname = next((g for g, gc in BIFACTOR_GROUPS_INDIV.items() if c in gc), "?")
        print(f"    {short_label(c):<22} {ls:.3f}  [{gname}]")
    fig_bif = plot_bifactor_loadings(bif_result, indiv_cols, BIFACTOR_GROUPS_INDIV)
    fig_bif.savefig(OUTPUT_DIR / "02_bifactor_cfa.png")
    print(f"\n  → {OUTPUT_DIR}/02_bifactor_cfa.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 2: GLV VIA LYAPUNOV INVERSION (PRIMARY)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 2: GLV Dynamical System (Lyapunov Inversion)")
    print("  Calibrating from INDIVIDUAL-LEVEL covariance via J = -(1/2) Σ⁻¹.")
    print("  This is exact: GLV Jacobian at x* = μ reproduces Σ_obs by construction.")

    glv, glv_diag = calibrate_glv_from_lyapunov_inversion(X_raw_indiv, indiv_cols)
    x_star = glv.equilibrium()

    print(f"\n  Calibration summary (N = {glv_diag['N']:,}):")
    print(f"    Jacobian stable: {glv_diag['J_stable']}")
    print(f"    Equilibrium match: {glv_diag['eq_match']:.2e}")
    print(f"    Jacobian match:    {glv_diag['J_match']:.2e}")
    print(f"    Sigma match:       {glv_diag['Sigma_match']:.2e}")

    print(f"\n  GLV intrinsic growth rates r_k:")
    for k, c in enumerate(indiv_cols):
        tag = "OBLIGATE" if glv.r[k] < 0 else "facultative"
        print(f"    {short_label(c):<22} r = {glv.r[k]:>8.4f}  [{tag}]")

    print(f"\n  GLV interaction matrix A (diag = self-limit, off-diag = coupling):")
    print(f"  {'':>16}", end="")
    for j in range(glv.K):
        print(f" {RYFF_SHORT[j]:>7}", end="")
    print()
    for i in range(glv.K):
        print(f"  {RYFF_SHORT[i]:<16}", end="")
        for j in range(glv.K):
            print(f" {glv.A[i, j]:>7.4f}", end="")
        print()

    print(f"\n  Equilibrium x* = {x_star.round(3)}")
    print(f"  Mean |A_offdiag| = {glv.mean_coupling_strength():.4f}")

    # Simulate population
    print(f"\n  Simulating GLV population (200 individuals) …", end=" ", flush=True)
    equilibria = glv.simulate_population(n_individuals=200)
    print("done.")
    sim_corr = np.corrcoef(equilibria.T)
    off_diag = sim_corr[np.triu_indices(len(indiv_cols), k=1)]
    print(f"  Simulated positive manifold:")
    print(f"    All positive: {np.all(off_diag > 0)}")
    print(f"    Mean r = {off_diag.mean():.3f}")
    sim_std = StandardScaler().fit_transform(equilibria)
    sim_pca = PCA(); sim_pca.fit(sim_std)
    print(f"    Simulated PC1: {sim_pca.explained_variance_ratio_[0]*100:.1f}% "
          f"(vs {var_ratio[0]*100:.1f}% observed)")

    fig_glv = plot_glv_demo(glv, equilibria, indiv_cols)
    fig_glv.savefig(OUTPUT_DIR / "03_glv_model.png")
    print(f"\n  → {OUTPUT_DIR}/03_glv_model.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  ASYMMETRIC COUPLING ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print_section("ASYMMETRIC COUPLING ANALYSIS")
    budget = coupling_budget_analysis(glv, glv_diag, indiv_cols)
    print_coupling_budget(budget, indiv_cols)

    fig_budget = plot_coupling_budget(budget, indiv_cols)
    fig_budget.savefig(OUTPUT_DIR / "03b_coupling_budget.png")
    print(f"\n  → {OUTPUT_DIR}/03b_coupling_budget.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  PERTURBATION DECOMPOSITION: J = S + A
    # ══════════════════════════════════════════════════════════════════════════
    print_section("PERTURBATION DECOMPOSITION: J = S + A")

    # J from Lyapunov inversion is symmetric by construction (J = -½Σ⁻¹).
    # The perturbation series is trivially zero. But the coupling budget
    # reveals genuine structural asymmetry: M_log[k,l] = -J[k,l]/J[k,k]
    # differs from M_log[l,k] = -J[l,k]/J[l,l] because J[k,k] varies.
    #
    # To demonstrate the perturbation machinery and provide a sensitivity
    # analysis, we construct J_directed from the asymmetric coupling matrix
    # M_log with a uniform row-scaling γ (the harmonic mean of -J[k,k]),
    # preserving the directional coupling asymmetry that the Lyapunov
    # inversion cancels:
    #
    #   J_directed[k,l] = γ · M_log[k,l]     for k ≠ l
    #   J_directed[k,k] = J[k,k]             (unchanged diagonal)
    #
    # This answers: "If the coupling were truly directional (as longitudinal
    # data might show), how much would the eigenstructure rotate?"

    J_sym = glv_diag["J"]
    print(f"  Lyapunov-inverted J is symmetric: ‖A‖/‖J‖ = "
          f"{norm(J_sym - J_sym.T, 'fro') / (2 * norm(J_sym, 'fro')):.2e}")

    M_log = glv_diag["M_logistic"]
    K = glv.K
    gamma = np.mean(-np.diag(J_sym))  # uniform row scale
    J_directed = np.diag(np.diag(J_sym)).copy()
    for k in range(K):
        for l in range(K):
            if k != l:
                J_directed[k, l] = gamma * M_log[k, l]

    asym_check = norm(J_directed - J_directed.T, 'fro') / (2 * norm(J_directed, 'fro'))
    print(f"  Directed-coupling J asymmetry:    ‖A‖/‖J‖ = {asym_check:.4f}")

    decomp = perturbation_decomposition(J_directed)
    print_perturbation_decomposition(decomp, indiv_cols)

    # ══════════════════════════════════════════════════════════════════════════
    #  THREE-MODEL COMPARISON: GLV vs LINEAR vs LOGISTIC
    # ══════════════════════════════════════════════════════════════════════════
    print_section("THREE-MODEL COMPARISON: Self-Acceptance Shock")

    k_sa = indiv_cols.index("B1SPWBS1") if "B1SPWBS1" in indiv_cols else 5
    fig_3model, logistic_params = three_model_shock_comparison(
        glv, glv_diag, indiv_cols, shock_domain=k_sa)
    fig_3model.savefig(OUTPUT_DIR / "03c_three_model_shock.png")
    print(f"  → {OUTPUT_DIR}/03c_three_model_shock.png")

    # Report logistic pathology for Self-Acceptance
    C_log = logistic_params["C_logistic"]
    print(f"\n  Logistic carrying capacities C_k:")
    for k, c in enumerate(indiv_cols):
        flag = " ← NEGATIVE (logistic form breaks)" if C_log[k] < 0 else ""
        print(f"    {short_label(c):<22} C = {C_log[k]:>8.3f}{flag}")

    print(f"\n  SUMMARY: The three-model architecture")
    print(f"    1. LINEAR (OU):      Exact covariance match. Use for eigenspectrum, return times.")
    print(f"    2. GLV (primary):    Handles obligate mutualism (r_k < 0). Same J at equilibrium.")
    print(f"                         Use for large perturbations, basin of attraction, coupling sweep.")
    print(f"    3. LOGISTIC (ped.):  Breaks for {budget['n_obligate']} domain(s) where C_k < 0.")
    print(f"                         Use for exposition only (5/{glv.K} domains).")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 3: LYAPUNOV STABILITY & RESILIENCE (GLV)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 3: Lyapunov Stability & Resilience (GLV)")
    lyap = lyapunov_analysis(glv, x_star)
    print(f"  Jacobian eigenvalues: {lyap['eigenvalues'].round(4)}")
    print(f"  Stable: {lyap['stable']}")
    print(f"  Dominant eigenvalue λ₁ = {lyap['lambda_1']:.4f}")
    print(f"  Return time τ = {lyap['tau']:.2f}")
    print(f"  Recovery speed = {lyap['recovery_speed']:.4f}")
    print(f"  Basin radius = {lyap['basin_radius']:.4f}")
    print(f"  Resilience R(x*) = {lyap['resilience']:.4f}")

    fig_lyap = plot_lyapunov_analysis(lyap, x_star, glv, indiv_cols)
    fig_lyap.savefig(OUTPUT_DIR / "04_lyapunov_stability.png")
    print(f"\n  → {OUTPUT_DIR}/04_lyapunov_stability.png")

    # Per-person resilience
    print(f"\n  Computing per-person resilience …", end=" ", flush=True)
    resilience_df = person_resilience_scores(X_raw_indiv, glv, indiv_cols)
    print("done.")
    valid_res = resilience_df["resilience"].notna() & (resilience_df["resilience"] > 0)
    if valid_res.sum() > 10:
        r_corr, r_p = stats.pearsonr(
            df_scored.loc[valid_res.values, "f_score"],
            resilience_df.loc[valid_res, "resilience"])
        print(f"  f–resilience correlation: r = {r_corr:.3f}, p = {r_p:.3e}")
    else:
        r_corr, r_p = np.nan, np.nan
        print(f"  f–resilience: insufficient valid scores ({valid_res.sum()})")

    fig_resil = plot_resilience_vs_f(df_scored, resilience_df, indiv_cols)
    fig_resil.savefig(OUTPUT_DIR / "05_resilience_vs_f.png")
    print(f"  → {OUTPUT_DIR}/05_resilience_vs_f.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  COHERENCE: Stage 1 ↔ Stage 2
    # ══════════════════════════════════════════════════════════════════════════
    print_section("REPRESENTATION COHERENCE: Stage 1 ↔ Stage 2")
    coherence = compute_coherence_metrics(pca_indiv, glv, x_star, S_indiv, indiv_cols)
    print(f"  Weak coherence (cov match): {coherence['weak_coherence']:.4f}")
    print(f"  Medium coherence (Jacobian): {coherence['medium_coherence']:.4f}")
    print(f"  Sign discipline: {'✓ PASS' if coherence['sign_discipline'] else '✗ FAIL'}")
    print(f"  Rank discipline:")
    print(f"    Observed λ₁/λ₂ = {coherence['obs_eigenratio']:.2f}")
    print(f"    Model   λ₁/λ₂ = {coherence['model_eigenratio']:.2f}")
    print(f"  CS discipline: r = {coherence['cs_correlation']:.3f} (p = {coherence['cs_p_value']:.3f})")
    print(f"  Model PC1 var: {coherence['model_pc1_var']*100:.1f}% "
          f"(observed: {var_ratio[0]*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 4: INDIVIDUAL-LEVEL CAPABILITIES
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 4: Individual-Level Alkire-Foster Capabilities")
    af_df, af_summary = individual_alkire_foster(df_scored, indiv_cols)
    print(f"  Thresholds (25th percentile):")
    for c, z in af_summary["thresholds"].items():
        print(f"    {short_label(c):<22} z = {z:.3f}")
    print(f"\n  Pass all: {af_summary['n_pass']:,} / {N_indiv:,} "
          f"({100*af_summary['n_pass']/N_indiv:.1f}%)")
    print(f"  H = {af_summary['H']:.3f},  A = {af_summary['A']:.3f},  M₀ = {af_summary['M0']:.3f}")
    print(f"  f ↔ f_cap correlation: {af_summary['f_fcap_corr']:.3f}")
    fig_af = plot_individual_alkire_foster(af_df, af_summary, indiv_cols)
    fig_af.savefig(OUTPUT_DIR / "06_alkire_foster.png")
    print(f"\n  → {OUTPUT_DIR}/06_alkire_foster.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  COUPLING SWEEP (GLV-based)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("COUPLING SWEEP (GLV)")
    print("  Sweeping off-diagonal coupling (fixed self-limitation) …", end=" ", flush=True)
    sweep_df = coupling_sweep(glv, n_points=15, n_individuals=80)
    print("done.")

    stable_df = sweep_df[sweep_df["stable"]]
    n_stable = stable_df.shape[0]
    n_total = sweep_df.shape[0]
    print(f"  Stable: {n_stable}/{n_total} points")

    if n_stable > 0:
        peak = stable_df.loc[stable_df["resilience"].idxmax()]
        print(f"  Peak resilience at α = {peak['alpha']:.2f} (M̄ = {peak['m_bar']:.4f})")
        print(f"    PC1 variance: {peak['pct_var_pc1']:.1f}%")
        print(f"    Resilience: {peak['resilience']:.4f}")
        # Calibrated system (α ≈ 1)
        cal = sweep_df.iloc[(sweep_df["alpha"] - 1.0).abs().argmin()]
        print(f"  Calibrated (α=1): M̄ = {cal['m_bar']:.4f}, R = {cal['resilience']:.4f}")
    if n_stable < n_total:
        first_unstable = sweep_df[~sweep_df["stable"]].iloc[0]
        print(f"  Stability boundary at α ≈ {first_unstable['alpha']:.2f}")

    fig_sweep = plot_coupling_sweep(sweep_df)
    fig_sweep.savefig(OUTPUT_DIR / "07_coupling_sweep.png")
    print(f"\n  → {OUTPUT_DIR}/07_coupling_sweep.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  COUPLING–RESILIENCE SIGNATURE
    # ══════════════════════════════════════════════════════════════════════════
    print_section("COUPLING–RESILIENCE SIGNATURE")
    print("  Simulating coupling × shock interaction …", end=" ", flush=True)
    sig_trials, sig_summary = coupling_resilience_signature(glv, n_individuals=50, seed=42)
    print("done.")
    high_coupling = sig_summary["coupling"] > sig_summary["coupling"].median()
    large_shock = sig_summary["shock_frac"] > sig_summary["shock_frac"].median()
    var_hc_ls = sig_summary.loc[high_coupling & large_shock, "var_recovery"].mean()
    var_lc_ls = sig_summary.loc[~high_coupling & large_shock, "var_recovery"].mean()
    print(f"\n  Recovery variance (coupling × shock):")
    print(f"    High coupling, large shock:  {var_hc_ls:.1f}")
    print(f"    Low coupling, large shock:   {var_lc_ls:.1f}")
    print(f"    Interaction ratio: {var_hc_ls / (var_lc_ls + 1e-8):.2f}×")
    fig_sig = plot_coupling_resilience_signature(sig_summary)
    fig_sig.savefig(OUTPUT_DIR / "08_coupling_resilience_signature.png")
    print(f"\n  → {OUTPUT_DIR}/08_coupling_resilience_signature.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  INDIVIDUAL-LEVEL COUPLING PROXY & DEMOGRAPHICS
    # ══════════════════════════════════════════════════════════════════════════
    print_section("Individual-Level Coupling Proxy & Demographics")
    coupling_df = individual_coupling_resilience(df_scored, indiv_cols)
    print(f"  Coupling proxy — mean: {coupling_df['coupling_proxy'].mean():.3f}, "
          f"SD: {coupling_df['coupling_proxy'].std():.3f}")
    r_cp, p_cp = stats.pearsonr(coupling_df["coupling_proxy"], af_df["compensation_gap"])
    print(f"  Coupling ↔ compensation gap: r = {r_cp:.3f}, p = {p_cp:.2e}")
    weak_counts = coupling_df["weakest_domain"].value_counts()
    print(f"\n  Weakest domain distribution:")
    for dom, count in weak_counts.items():
        print(f"    {short_label(dom):<22} {count:,} ({100*count/N_indiv:.1f}%)")

    fig_demo = plot_demographic_f(df_scored)
    if fig_demo:
        fig_demo.savefig(OUTPUT_DIR / "09_demographic_f.png")
        print(f"\n  → {OUTPUT_DIR}/09_demographic_f.png")
    fig_fp = plot_f_vs_p(df_scored)
    if fig_fp:
        fig_fp.savefig(OUTPUT_DIR / "10_f_vs_p.png")
        print(f"\n  -> {OUTPUT_DIR}/10_f_vs_p.png")

    # ======================================================================
    #  ECONOMIC VALIDATION: f predicts economic outcomes
    # ======================================================================
    print_section("ECONOMIC VALIDATION: f and Economic Outcomes")
    econ_df = construct_economic_variables(df_scored)
    econ_present = [c for c in ECON_DERIVED if c in econ_df.columns and econ_df[c].notna().sum() > 50]
    print(f"  Derived economic variables: {', '.join(econ_present)}")
    for c in econ_present:
        v = econ_df[c].dropna()
        print(f"    {c:<20} N={len(v):>5}, mean={v.mean():.3f}, SD={v.std():.3f}")
    if "age" in econ_df.columns:
        age = econ_df["age"].dropna()
        wa = ((age >= 25) & (age <= 64)).sum()
        print(f"  Working-age subsample (25-64): {wa:,}")

    econ_results = economic_validation(df_scored, econ_df)
    if econ_results is not None:
        for sample_key, sample_label in [("full", "Full Sample"), ("working_age", "Working Age (25-64)")]:
            res = econ_results.get(sample_key)
            if not res:
                continue
            corr_df = res.get("correlations", pd.DataFrame())
            incr_df = res.get("incremental", pd.DataFrame())
            emp_info = res.get("employment")

            n_tag = f" (N={econ_results['n_working_age']:,})" if sample_key == "working_age" else ""
            print(f"\n  === {sample_label}{n_tag} ===")
            if len(corr_df) > 0:
                print(f"  {'Variable':<28} {'N':>5}  {'r(f)':>7}  {'p':>10}  "
                      f"{'r|age':>7}  {'r(neg)':>7}  {'Sign':>4}")
                print(f"  {'-'*82}")
                for _, row in corr_df.iterrows():
                    s = "ok" if row["sign_ok"] else "FLIP"
                    rp = f"{row['r_partial']:>7.3f}" if not np.isnan(row["r_partial"]) else "    n/a"
                    rn = f"{row['r_negaf']:>7.3f}" if not np.isnan(row["r_negaf"]) else "    n/a"
                    print(f"  {row['label']:<28} {row['n']:>5}  {row['r_f']:>7.3f}  "
                          f"{row['p_f']:>10.2e}  {rp}  {rn}  {s:>4}")

            if len(incr_df) > 0:
                print(f"\n  Incremental R^2 (f beyond neg. affect):")
                print(f"  {'Variable':<28} {'N':>5}  {'R2_neg':>6}  {'R2_f':>6}  "
                      f"{'R2_both':>7}  {'dR2':>6}  {'F':>7}  {'p':>10}")
                print(f"  {'-'*88}")
                for _, row in incr_df.iterrows():
                    r2f = f"{row['R2_f_only']:>6.3f}" if "R2_f_only" in row else "   n/a"
                    print(f"  {row['label']:<28} {row['n']:>5}  "
                          f"{row['R2_negaf']:>6.3f}  {r2f}  "
                          f"{row['R2_both']:>7.3f}  {row['incr_R2']:>6.3f}  "
                          f"{row['F']:>7.1f}  {row['p']:>10.2e}")

            if emp_info:
                print(f"\n  Employment discrimination:")
                print(f"    N = {emp_info['n']:,} ({emp_info['n_employed']:,} employed, "
                      f"{emp_info['n_not']:,} not)")
                print(f"    Cohen's d (f):       {emp_info['d_f']:.3f}")
                d_neg_str = f"{emp_info['d_negaf']:.3f}" if not np.isnan(emp_info['d_negaf']) else "n/a"
                print(f"    Cohen's d (neg.aff): {d_neg_str}")
                print(f"    f mean employed:     {emp_info['f_mean_emp']:.3f}")
                print(f"    f mean not employed: {emp_info['f_mean_not']:.3f}")

        fig_econ = plot_economic_validation(df_scored, econ_df, econ_results)
        if fig_econ:
            fig_econ.savefig(OUTPUT_DIR / "11_economic_validation.png")
            print(f"\n  -> {OUTPUT_DIR}/11_economic_validation.png")
    else:
        econ_results = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVE & SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    # Merge economic variables into scored output
    for col in econ_df.columns:
        if col not in df_scored.columns:
            df_scored[f"econ_{col}"] = econ_df[col].values
    for col in af_df.columns:
        if col not in df_scored.columns:
            df_scored[col] = af_df[col].values
    for col in coupling_df.columns:
        df_scored[f"coupling_{col}"] = coupling_df[col].values
    for col in resilience_df.columns:
        df_scored[f"lyap_{col}"] = resilience_df[col].values

    out_path = Path("wellbeing_data/midus_scored_v2.csv")
    out_path.parent.mkdir(exist_ok=True)
    df_scored.to_csv(out_path, index=False)
    print(f"\n  ✓ Scored data saved → {out_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FULL IMPLEMENTATION COMPLETE (v2 — GLV via Lyapunov Inversion)")
    print("═" * 70)
    print(f"  Data:                    {data_label}")
    print(f"  N (individuals):         {N_indiv:,}")
    print(f"  Calibration:             GLV via Lyapunov inversion (J = -½ Σ⁻¹)")
    print(f"  PC1 var explained:       {var_ratio[0]*100:.1f}%")
    print(f"  Model 1 — ω_h:          {bif_result['omega_h']:.3f}")
    print(f"  Model 1 — ECV:          {bif_result['ecv']:.3f}")
    print(f"  Model 2 — GLV stable:   {glv_diag['J_stable']}")
    print(f"  Model 2 — Obligate:     {budget['n_obligate']} domain(s)")
    print(f"  Model 2 — Inhibitory:   {len(budget['neg_pairs'])} pair(s)")
    print(f"  Model 2 — Sim PC1:      {sim_pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  Model 2 — Asym index:   {decomp['asymmetry_index']:.4f}")
    if decomp.get("cos_orders_vs_full"):
        print(f"  Perturbation Σ₀ cos:    {decomp['cos_orders_vs_full'][0]:.5f}")
        if len(decomp["cos_orders_vs_full"]) > 1:
            print(f"  Perturbation Σ₀+₁ cos:  {decomp['cos_orders_vs_full'][1]:.5f}")
    print(f"  Model 3 — τ:            {lyap['tau']:.2f}")
    print(f"  Model 3 — R(x*):        {lyap['resilience']:.4f}")
    print(f"  Coherence — Weak:       {coherence['weak_coherence']:.4f}")
    print(f"  Coherence — Medium:     {coherence['medium_coherence']:.4f}")
    print(f"  Coherence — Sign:       {'PASS' if coherence['sign_discipline'] else 'FAIL'}")
    print(f"  Model 4 — M₀:          {af_summary['M0']:.3f}")
    print(f"  Model 4 — f↔f_cap:     {af_summary['f_fcap_corr']:.3f}")
    if valid_res.sum() > 10:
        print(f"  Cross-model — f↔R:     {r_corr:.3f}")
    print(f"  Coupling proxy <-> gap:   {r_cp:.3f}")
    print(f"  Resilience sig HC*LS:   {var_hc_ls:.1f}")
    print(f"  Resilience sig LC*LS:   {var_lc_ls:.1f}")
    # Economic validation summary
    if econ_results:
        print(f"  --- Economic Validation ---")
        for sk, sl in [("full","Full"),("working_age","Working Age")]:
            res = econ_results.get(sk)
            if not res: continue
            cdf = res.get("correlations", pd.DataFrame())
            if len(cdf) == 0: continue
            print(f"  [{sl}]")
            for _, row in cdf.iterrows():
                sig = "***" if row["p_f"]<.001 else "**" if row["p_f"]<.01 else "*" if row["p_f"]<.05 else ""
                print(f"    f <-> {row['label']:<26} r={row['r_f']:>7.3f}{sig}")
            emp = res.get("employment")
            if emp:
                print(f"    Employment d(f) = {emp['d_f']:.3f}")
            idf = res.get("incremental", pd.DataFrame())
            if len(idf) > 0:
                print(f"    Sum incr. R2 = {idf['incr_R2'].sum():.4f}")
    print(f"\n  Figures: {OUTPUT_DIR}/01-11_*.png")
    print(f"  Scored data: {out_path}")
    print()


if __name__ == "__main__":
    main()
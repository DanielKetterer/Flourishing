"""
combined_f_analysis.py
─────────────────────────────────────────────────────────────────────────────
Combined implementation of the four models from
"Toward f: A General Factor of Human Flourishing."

KEY CHANGE: The mutualism dynamical system (Model 2) is now calibrated at
the individual level from MIDUS Ryff subscale partial correlations, directly
addressing the ecological fallacy critique. Country-level calibration is
retained as a comparison/diagnostic only.

Architecture
────────────
  1. Load individual-level MIDUS 2 data (real or synthetic)
  2. Load country-level ecological data (for comparison)
  3. Individual-level PCA, bifactor CFA, demographics, dual continua
  4. MODEL 1 — Bifactor CFA (individual-level = primary test)
  5. MODEL 2 — Mutualism dynamical system calibrated from INDIVIDUAL-LEVEL
               partial correlations (not country-level aggregates)
  6. MODEL 3 — Lyapunov stability & resilience (on individual-calibrated system)
  7. COHERENCE — Stage 1 ↔ Stage 2 bridge
  8. MODEL 4 — Individual-level Alkire-Foster capabilities
  9. COUPLING SWEEP & RESILIENCE SIGNATURE (individual-calibrated)
  10. Ecological fallacy diagnostic (compare individual vs country calibration)

Data requirements
─────────────────
  Individual-level: MIDUS 2 (ICPSR 4652) .tsv file, or synthetic fallback
  Country-level:    wellbeing_data/wellbeing_merged.csv (from script 01)

Requires
────────
  pip install numpy scipy pandas matplotlib seaborn scikit-learn
  Optional: pip install semopy  (for bifactor CFA via SEM)
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
from numpy.linalg import inv, det, slogdet, cholesky

# ─── Config ───────────────────────────────────────────────────────────────────

# Individual-level data
MODE = "midus"  # "midus" for real data, "synthetic" for simulated
MIDUS_DATA_PATH = Path("04652-0001-Data.tsv")

# Country-level data (for ecological comparison)
ECOLOGICAL_DATA_PATH = Path("wellbeing_data/wellbeing_merged.csv")

OUTPUT_DIR = Path("wellbeing_figures_combined")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
BOOTSTRAP_N = 1000

# ─── Country-level column definitions ────────────────────────────────────────

PERMA_COLS = [
    "P_positive_emotion",
    "E_engagement_autonomy",
    "R_relationships",
    "M_meaning_life_sat",
    "A_accomplishment_gdp",
    "H_health",
]

SHORT_LABELS_ECO = {
    "P_positive_emotion":    "P: Pos. Emotion",
    "E_engagement_autonomy": "E: Engagement",
    "R_relationships":       "R: Relationships",
    "M_meaning_life_sat":    "M: Meaning",
    "A_accomplishment_gdp":  "A: Accomplishment",
    "H_health":              "H: Health",
}

PALETTE_ECO = {
    "P_positive_emotion":    "#E8622A",
    "E_engagement_autonomy": "#3B7DD8",
    "R_relationships":       "#2EAA6A",
    "M_meaning_life_sat":    "#9B59B6",
    "A_accomplishment_gdp":  "#E67E22",
    "H_health":              "#1ABC9C",
}

BIFACTOR_GROUPS_ECO = {
    "hedonic":         ["P_positive_emotion", "M_meaning_life_sat"],
    "eudaimonic":      ["E_engagement_autonomy", "A_accomplishment_gdp"],
    "social_physical": ["R_relationships", "H_health"],
}

# ─── Individual-level (MIDUS) column definitions ─────────────────────────────

RYFF_COLS = [
    "B1SPWBA1",  # Autonomy
    "B1SPWBE1",  # Environmental Mastery
    "B1SPWBG1",  # Personal Growth
    "B1SPWBR1",  # Positive Relations
    "B1SPWBU1",  # Purpose in Life
    "B1SPWBS1",  # Self-Acceptance
]

RYFF_LABELS = {
    "B1SPWBA1": "Autonomy",
    "B1SPWBE1": "Envir. Mastery",
    "B1SPWBG1": "Personal Growth",
    "B1SPWBR1": "Positive Relations",
    "B1SPWBU1": "Purpose in Life",
    "B1SPWBS1": "Self-Acceptance",
}

RYFF_PERMA = {
    "B1SPWBA1": "E",
    "B1SPWBE1": "A",
    "B1SPWBG1": "E",
    "B1SPWBR1": "R",
    "B1SPWBU1": "M",
    "B1SPWBS1": "P",
}

# Bifactor groups for individual-level CFA
BIFACTOR_GROUPS_INDIV = {
    "hedonic":    ["B1SPWBS1", "B1SPWBE1"],       # Self-Acceptance, Mastery
    "eudaimonic": ["B1SPWBU1", "B1SPWBG1"],       # Purpose, Growth
    "social_aut": ["B1SPWBR1", "B1SPWBA1"],       # Relations, Autonomy
}

AUX_COLS = ["B1SPOSPA", "B1SNEGAF", "B1SB1", "B1SA11W"]
DEMO_COLS = ["M2ID", "B1PAGE_M2", "B1PRSEX", "B1PF7A"]

PALETTE_INDIV = {
    "B1SPWBA1": "#3B7DD8",
    "B1SPWBE1": "#E8622A",
    "B1SPWBG1": "#2EAA6A",
    "B1SPWBR1": "#9B59B6",
    "B1SPWBU1": "#E67E22",
    "B1SPWBS1": "#1ABC9C",
}

# Generic short labels (resolve from either dict)
def short_label(col):
    return RYFF_LABELS.get(col, SHORT_LABELS_ECO.get(col, col))

def palette_color(col):
    return PALETTE_INDIV.get(col, PALETTE_ECO.get(col, "#888"))

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        120,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})


# ═══════════════════════════════════════════════════════════════════════════════
#  MIDUS 2 KNOWN PSYCHOMETRIC PROPERTIES (for synthetic generation)
# ═══════════════════════════════════════════════════════════════════════════════

MIDUS_CORR_MATRIX = np.array([
    # Auton   Mastery  Growth   PoRel   Purpose  Self-Acc
    [1.00,    0.47,    0.47,    0.34,    0.44,    0.40],
    [0.47,    1.00,    0.56,    0.54,    0.65,    0.77],
    [0.47,    0.56,    1.00,    0.47,    0.60,    0.49],
    [0.34,    0.54,    0.47,    1.00,    0.52,    0.59],
    [0.44,    0.65,    0.60,    0.52,    1.00,    0.66],
    [0.40,    0.77,    0.49,    0.59,    0.66,    1.00],
])
MIDUS_MEANS = np.array([4.80, 5.19, 5.53, 5.36, 5.38, 5.18])
MIDUS_SDS   = np.array([0.82, 0.86, 0.82, 0.94, 0.93, 0.97])


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING — INDIVIDUAL LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_midus(n: int = 5000) -> pd.DataFrame:
    """Simulate individual-level MIDUS 2 data matching published psychometrics."""
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

    print(f"  ✓ Synthetic MIDUS data: {df.shape}")
    print(f"  ⚠  SIMULATED data — register at ICPSR for real MIDUS 2.")
    return df


def load_midus(path: Path) -> pd.DataFrame:
    """Load MIDUS 2 DS0001 (tab-delimited, ICPSR format)."""
    print(f"Loading MIDUS 2 data from {path} …")
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {path}\n"
            f"  Download MIDUS 2 (ICPSR 4652), place main .tsv here.\n"
            f"  Or set MODE = 'synthetic' to run with simulated data.")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    missing = [c for c in RYFF_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: columns not found: {missing}")

    for col in RYFF_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where((df[col] > 0) & (df[col] < 90))
            df[col] = df[col] / 3.0   # 3-item sum → 1-7 mean

    for col in AUX_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col] > 0)

    print(f"  Loaded: {len(df)} respondents, {len(df.columns)} variables")
    return df


def preprocess_midus(df: pd.DataFrame):
    """Preprocess MIDUS data. Returns (df_clean, X_raw, X_scaled, present_cols)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING — COUNTRY LEVEL (ecological comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ecological_data():
    """Load country-level merged dataset. Returns (df, X_raw, X_scaled, S, cols, N) or None."""
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
    """Report correlation summary for positive manifold check."""
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
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0.0, vmax=1.0,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                xticklabels=labels, yticklabels=labels)
    off = corr.values[np.tril_indices_from(corr.values, k=-1)]
    ax.set_title(f"Positive Manifold: Subscale Correlations\n({title_suffix})", pad=14)
    ax.text(0.01, -0.10,
            f"Mean r = {off.mean():.3f}  |  All positive: "
            f"{'✓' if (off > 0).all() else '✗'}",
            transform=ax.transAxes, fontsize=8.5, color="#444")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def run_pca_individual(X_scaled, cols, df_clean, suffix=""):
    """PCA with bootstrap CIs on individual-level data. Returns (df_scored, pca, var_ratio)."""
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
        pca_b = PCA(n_components=1)
        pca_b.fit(X_scaled[idx])
        lb = pca_b.components_[0]
        if np.dot(lb, l1) < 0:
            lb = -lb
        boot_L[i] = lb
    if l1.mean() < 0:
        l1 = -l1
        boot_L = -boot_L

    ci_lo = np.percentile(boot_L, 2.5, axis=0)
    ci_hi = np.percentile(boot_L, 97.5, axis=0)

    print(f"\n  PC1 Loadings with 95% Bootstrap CI:")
    for col, lo, val, hi in zip(cols, ci_lo, l1, ci_hi):
        print(f"    {short_label(col):<22} {val:.3f}  "
              f"[{min(lo,hi):.3f}, {max(lo,hi):.3f}]")

    f_raw = scores[:, 0]
    if f_raw.mean() < 0:
        f_raw = -f_raw
        pca.components_[0] = -pca.components_[0]
    f_z = (f_raw - f_raw.mean()) / f_raw.std()
    df_clean = df_clean.copy()
    df_clean["f_score"] = f_z

    return df_clean, pca, vr


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 1: BIFACTOR CFA VIA ML (works on any covariance matrix)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sigma_bifactor(params, p, group_indices):
    """Construct model-implied covariance Σ(θ) for bifactor model."""
    lam_f = params[:p]
    lam_s = params[p:2*p]
    theta  = np.exp(params[2*p:3*p])
    Sigma = np.outer(lam_f, lam_f)
    for group_idx in group_indices:
        lam_sg = np.zeros(p)
        for j in group_idx:
            lam_sg[j] = lam_s[j]
        Sigma += np.outer(lam_sg, lam_sg)
    Sigma += np.diag(theta)
    return Sigma


def _ml_discrepancy(params, S, N, p, group_indices):
    """ML fitting function: F_ML = log|Σ| + tr(S·Σ⁻¹) - log|S| - p"""
    Sigma = _build_sigma_bifactor(params, p, group_indices)
    try:
        sign, logdet_sigma = slogdet(Sigma)
        if sign <= 0:
            return 1e10
        sign_s, logdet_s = slogdet(S)
        Sigma_inv = inv(Sigma)
        F = logdet_sigma + np.trace(S @ Sigma_inv) - logdet_s - p
        return F
    except np.linalg.LinAlgError:
        return 1e10


def fit_bifactor_cfa(S, N, cols, bifactor_groups):
    """Fit bifactor CFA via ML. Returns dict with loadings, omega_h, ECV, fit indices."""
    p = len(cols)

    group_indices = []
    for gname, gcols in bifactor_groups.items():
        idx = [cols.index(c) for c in gcols if c in cols]
        group_indices.append(idx)

    pca = PCA(n_components=1)
    pca.fit(np.random.multivariate_normal(np.zeros(p), S, size=max(N, 200)))
    lam_f_init = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    if lam_f_init.mean() < 0:
        lam_f_init = -lam_f_init

    lam_s_init = np.full(p, 0.3)
    theta_init = np.log(np.diag(S) * 0.3)
    x0 = np.concatenate([lam_f_init, lam_s_init, theta_init])

    result = minimize(
        _ml_discrepancy, x0, args=(S, N, p, group_indices),
        method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-10})

    lam_f = result.x[:p]
    lam_s = result.x[p:2*p]
    theta  = np.exp(result.x[2*p:3*p])
    if lam_f.mean() < 0:
        lam_f = -lam_f

    Sigma_hat = _build_sigma_bifactor(result.x, p, group_indices)

    sum_lam_f_sq = np.sum(lam_f)**2
    sum_lam_s_group_sq = 0.0
    for gidx in group_indices:
        sum_lam_s_group_sq += np.sum(lam_s[gidx])**2
    sum_theta = np.sum(theta)
    total_var = sum_lam_f_sq + sum_lam_s_group_sq + sum_theta
    omega_h = sum_lam_f_sq / total_var

    common_var_f = np.sum(lam_f**2)
    common_var_s = np.sum(lam_s**2)
    ecv = common_var_f / (common_var_f + common_var_s)

    F_min = result.fun
    n_free_params = p + p + p
    n_data_points = p * (p + 1) // 2
    df_model = n_data_points - n_free_params
    chi2 = max(0, (N - 1) * F_min)
    rmsea = np.sqrt(max(0, (chi2 / df_model - 1) / (N - 1))) if df_model > 0 else np.nan

    return {
        "lambda_f": lam_f, "lambda_s": lam_s, "theta": theta,
        "omega_h": omega_h, "ecv": ecv, "Sigma_hat": Sigma_hat,
        "F_min": F_min, "chi2": chi2, "df": df_model, "RMSEA": rmsea,
        "converged": result.success, "group_indices": group_indices,
    }


def plot_bifactor_loadings(result, cols, bifactor_groups):
    """Paired bar chart: general (f) vs specific (s) loadings."""
    lam_f = result["lambda_f"]
    lam_s = result["lambda_s"]
    labels = [short_label(c) for c in cols]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(cols))
    h = 0.35
    ax.barh(y + h/2, lam_f, height=h, color="#3B7DD8", alpha=0.85,
            label=f"General f (ω_h = {result['omega_h']:.3f})")
    ax.barh(y - h/2, np.abs(lam_s), height=h, color="#E67E22", alpha=0.85,
            label="Specific s_k")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Factor Loading (λ)")
    ax.set_title(f"Bifactor CFA Loadings\nECV = {result['ecv']:.3f}  |  "
                 f"RMSEA = {result['RMSEA']:.3f}", fontsize=11)
    ax.axvline(0, color="black", lw=0.8)
    ax.legend(fontsize=9, loc="lower right")

    group_colors = {"hedonic": "#9B59B6", "eudaimonic": "#3B7DD8",
                    "social_physical": "#2EAA6A", "social_aut": "#2EAA6A"}
    for gname, gcols in bifactor_groups.items():
        idxs = [cols.index(c) for c in gcols if c in cols]
        if idxs:
            ymin, ymax = min(idxs) - 0.4, max(idxs) + 0.4
            ax.axhspan(ymin, ymax, alpha=0.06,
                       color=group_colors.get(gname, "#888"))
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 2: MUTUALISM DYNAMICAL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class MutualismSystem:
    """
    Coupled logistic growth model (van der Maas et al. 2006, adapted).

    dx_k/dt = a_k · x_k · (1 - x_k/C_k + Σ_{l≠k} M_kl · x_l / C_k) + σ_k · ξ_k(t)
    """

    def __init__(self, K, a=None, C=None, M=None, sigma=None):
        self.K = K
        self.a = a if a is not None else np.ones(K) * 0.5
        self.C = C if C is not None else np.ones(K)
        self.M = M if M is not None else np.ones((K, K)) * 0.1
        np.fill_diagonal(self.M, 0.0)
        self.sigma = sigma if sigma is not None else np.ones(K) * 0.05

    def deterministic_rhs(self, t, x):
        dx = np.zeros(self.K)
        for k in range(self.K):
            mutualism_term = sum(
                self.M[k, l] * x[l] / self.C[k]
                for l in range(self.K) if l != k
            )
            dx[k] = self.a[k] * x[k] * (1.0 - x[k] / self.C[k] + mutualism_term)
        return dx

    def find_equilibrium(self, x0=None, t_span=(0, 200), t_eval_n=2000):
        x_star = self.analytical_equilibrium()
        x0_traj = self.C * 0.3
        sol = solve_ivp(
            self.deterministic_rhs, (0, min(t_span[1], 60)), x0_traj,
            method="RK45", max_step=1.0,
            t_eval=np.linspace(0, min(t_span[1], 60), min(t_eval_n, 300)),
            rtol=1e-8, atol=1e-10)
        return x_star, sol

    def jacobian_at_equilibrium(self, x_star):
        J = np.zeros((self.K, self.K))
        for k in range(self.K):
            mut_sum = sum(
                self.M[k, l] * x_star[l] / self.C[k]
                for l in range(self.K) if l != k)
            J[k, k] = self.a[k] * (1.0 - 2.0 * x_star[k] / self.C[k] + mut_sum)
            for l in range(self.K):
                if l != k:
                    J[k, l] = self.a[k] * x_star[k] * self.M[k, l] / self.C[k]
        return J

    def analytical_equilibrium(self):
        """x* = (I - M̃)⁻¹ C"""
        M_tilde = self.M.copy()
        np.fill_diagonal(M_tilde, 0.0)
        A = np.eye(self.K) - M_tilde
        try:
            x_star = np.linalg.solve(A, self.C)
            if np.all(x_star > 0):
                return x_star
        except np.linalg.LinAlgError:
            pass
        return self.C * 1.1

    def fast_equilibrium(self, x0=None):
        return self.analytical_equilibrium()

    def simulate_population(self, n_individuals=200, C_var=0.15, M_var=0.05,
                            a_var=0.05, seed=42):
        """Simulate population with parameter variation → positive manifold."""
        rng = np.random.RandomState(seed)
        equilibria = np.zeros((n_individuals, self.K))
        for i in range(n_individuals):
            C_i = self.C * np.exp(rng.normal(0, C_var, self.K))
            a_i = self.a * np.exp(rng.normal(0, a_var, self.K))
            M_i = self.M * np.exp(rng.normal(0, M_var, (self.K, self.K)))
            np.fill_diagonal(M_i, 0.0)
            env_factor = rng.normal(0, 0.15)
            C_i *= np.exp(env_factor * 0.3)
            sys_i = MutualismSystem(self.K, a=a_i, C=C_i, M=M_i)
            try:
                x_star = sys_i.fast_equilibrium()
                if np.all(x_star > 0) and np.all(np.isfinite(x_star)):
                    equilibria[i] = x_star
                else:
                    equilibria[i] = C_i
            except Exception:
                equilibria[i] = C_i
        return equilibria

    def mean_coupling_strength(self):
        K = self.K
        upper_tri = self.M[np.triu_indices(K, k=1)]
        return 2.0 / (K * (K - 1)) * np.sum(upper_tri)


def calibrate_mutualism_from_individual_data(X_raw, cols):
    """
    Calibrate mutualism system from INDIVIDUAL-LEVEL data.

    This is the preferred calibration path — within-person partial correlations
    avoid the ecological fallacy (Robinson 1950) inherent in country-level data.

    C_k ← individual-level column means × 0.7
    M_kl ← scaled absolute partial correlations
    a_k ← 0.5 (not identifiable from cross-sectional data)
    """
    K = X_raw.shape[1]
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    C = means * 0.7

    X_std = (X_raw - means) / stds
    R = np.corrcoef(X_std.T)
    try:
        P = inv(R)
        D = np.sqrt(np.diag(P))
        partial_corr = -P / np.outer(D, D)
        np.fill_diagonal(partial_corr, 0.0)
    except np.linalg.LinAlgError:
        partial_corr = R.copy()
        np.fill_diagonal(partial_corr, 0.0)

    M_abs = np.abs(partial_corr)
    M = 0.05 + 0.25 * (M_abs - M_abs.min()) / (M_abs.max() - M_abs.min() + 1e-8)
    np.fill_diagonal(M, 0.0)

    a = np.ones(K) * 0.5
    sys = MutualismSystem(K, a=a, C=C, M=M)

    diagnostics = {
        "N_individuals":    X_raw.shape[0],
        "partial_corr":     partial_corr,
        "bivariate_corr":   R,
        "coupling_matrix":  M.copy(),
        "mean_coupling":    sys.mean_coupling_strength(),
        "carrying_cap":     C.copy(),
    }
    return sys, diagnostics


def calibrate_mutualism_from_ecological_data(X_raw, cols):
    """Calibrate from country-level data (for comparison only)."""
    K = len(cols)
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    C = means * 0.7

    X_std = (X_raw - means) / stds
    R = np.corrcoef(X_std.T)
    try:
        P = inv(R)
        D = np.sqrt(np.diag(P))
        partial_corr = -P / np.outer(D, D)
        np.fill_diagonal(partial_corr, 0.0)
    except np.linalg.LinAlgError:
        partial_corr = R.copy()
        np.fill_diagonal(partial_corr, 0.0)

    M = np.abs(partial_corr)
    M = 0.05 + 0.25 * (M - M.min()) / (M.max() - M.min() + 1e-8)
    np.fill_diagonal(M, 0.0)

    a = np.ones(K) * 0.5
    return MutualismSystem(K, a=a, C=C, M=M)


def compare_calibrations(eco_sys, indiv_sys, cols):
    """Quantify ecological fallacy gap between calibration levels."""
    K = min(eco_sys.K, indiv_sys.K)

    M_eco_upper = eco_sys.M[:K, :K][np.triu_indices(K, k=1)]
    M_ind_upper = indiv_sys.M[:K, :K][np.triu_indices(K, k=1)]
    coupling_corr, coupling_p = stats.pearsonr(M_eco_upper, M_ind_upper)
    frobenius_diff = np.linalg.norm(eco_sys.M[:K, :K] - indiv_sys.M[:K, :K], 'fro')

    x_star_eco = eco_sys.analytical_equilibrium()
    x_star_ind = indiv_sys.analytical_equilibrium()
    eq_corr, eq_p = stats.pearsonr(x_star_eco[:K], x_star_ind[:K])

    lyap_eco = lyapunov_analysis(eco_sys, x_star_eco)
    lyap_ind = lyapunov_analysis(indiv_sys, x_star_ind)

    return {
        "coupling_correlation": coupling_corr, "coupling_p": coupling_p,
        "frobenius_diff": frobenius_diff,
        "equilibrium_correlation": eq_corr,
        "eco_resilience": lyap_eco["resilience"], "indiv_resilience": lyap_ind["resilience"],
        "eco_tau": lyap_eco["tau"], "indiv_tau": lyap_ind["tau"],
    }


def plot_mutualism_demo(sys, equilibria, cols):
    """Three-panel: trajectory, population covariance, eigenvalue spectrum."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels_s = [short_label(c) for c in cols]
    colors = [palette_color(c) for c in cols]

    # (a) Trajectory convergence
    ax = axes[0]
    x_star, sol = sys.find_equilibrium()
    for k in range(sys.K):
        ax.plot(sol.t, sol.y[k], color=colors[k], lw=1.8, label=labels_s[k])
        ax.axhline(x_star[k], color=colors[k], ls=":", lw=0.8, alpha=0.5)
        ax.axhline(sys.C[k], color=colors[k], ls="--", lw=0.6, alpha=0.3)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Domain level x_k(t)")
    ax.set_title("(a) Convergence to Equilibrium\nDashed=C; dotted=x*")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.text(0.03, 0.97, "x* > C: mutualism lifts equilibrium",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # (b) Population correlation heatmap
    ax = axes[1]
    eq_std = (equilibria - equilibria.mean(axis=0)) / equilibria.std(axis=0)
    cov_pop = np.corrcoef(eq_std.T)
    im = ax.imshow(cov_pop, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cov_pop[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(cov_pop[i,j]) > 0.6 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels_s, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels_s, fontsize=8)
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
    ax.set_xlabel("Component")
    ax.set_ylabel("% Variance Explained")
    ax.set_title(f"(c) Eigenvalue Spectrum\nPC1 = {var_explained[0]:.1f}%")
    ax.set_xticks(comps)
    ax.axhline(100/len(cols), color="gray", ls="--", lw=1)

    fig.suptitle("Model 2: Mutualism Dynamical System\n"
                 "(Individual-Level Calibration)", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 3: LYAPUNOV STABILITY & RESILIENCE
# ═══════════════════════════════════════════════════════════════════════════════

def lyapunov_analysis(sys, x_star):
    """Full Lyapunov stability analysis at equilibrium x*."""
    J = sys.jacobian_at_equilibrium(x_star)
    eigs = eigvals(J)
    idx = np.argsort(np.real(eigs))[::-1]
    eigs_sorted = eigs[idx]
    lambda_1 = eigs_sorted[0]
    stable = np.all(np.real(eigs) < 0)
    tau = 1.0 / np.abs(np.real(lambda_1)) if np.real(lambda_1) != 0 else np.inf
    recovery_speed = np.abs(np.real(lambda_1)) if stable else 0.0

    Q = np.eye(sys.K)
    try:
        P = solve_continuous_lyapunov(J.T, -Q)
        P_eigvals = np.linalg.eigvalsh(P)
        basin_radius = 1.0 / np.sqrt(np.max(P_eigvals)) if np.all(P_eigvals > 0) else 0.0
    except Exception:
        P = np.eye(sys.K)
        basin_radius = 0.0

    resilience = np.sqrt(recovery_speed * basin_radius) if (recovery_speed > 0 and basin_radius > 0) else 0.0

    return {
        "J": J, "eigenvalues": eigs_sorted, "lambda_1": lambda_1,
        "tau": tau, "P": P, "resilience": resilience,
        "recovery_speed": recovery_speed, "basin_radius": basin_radius, "stable": stable,
    }


def plot_lyapunov_analysis(lyap_result, x_star, sys, cols):
    """Three-panel: eigenvalues, perturbation response, basin ellipsoid."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    eigs = lyap_result["eigenvalues"]
    labels_s = [short_label(c) for c in cols]
    colors = [palette_color(c) for c in cols]

    # (a) Eigenvalues
    ax = axes[0]
    ax.scatter(np.real(eigs), np.imag(eigs), s=120, c="#3B7DD8", edgecolors="black", zorder=5)
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.6, label="Stability boundary")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"(a) Jacobian Eigenvalues\nStable: {lyap_result['stable']}  |  τ = {lyap_result['tau']:.2f}")
    ax.legend(fontsize=8)
    lam1 = lyap_result["lambda_1"]
    ax.annotate(f"λ₁ = {np.real(lam1):.3f}", xy=(np.real(lam1), np.imag(lam1)),
                xytext=(np.real(lam1) + 0.05, np.imag(lam1) + 0.05),
                fontsize=9, color="#E8622A",
                arrowprops=dict(arrowstyle="->", color="#E8622A"))

    # (b) Perturbation response
    ax = axes[1]
    delta = np.zeros(sys.K)
    delta[0] = -x_star[0] * 0.3
    x_perturbed = np.maximum(x_star + delta, 0.01)
    sol = solve_ivp(sys.deterministic_rhs, (0, 40), x_perturbed,
                    method="RK45", max_step=0.5, t_eval=np.linspace(0, 40, 200))
    for k in range(sys.K):
        deviation = (sol.y[k] - x_star[k]) / x_star[k] * 100
        ax.plot(sol.t, deviation, color=colors[k], lw=1.5, label=labels_s[k])
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Time after perturbation")
    ax.set_ylabel("% deviation from equilibrium")
    ax.set_title(f"(b) Recovery from 30% Shock\nτ ≈ {lyap_result['tau']:.1f}")
    ax.legend(fontsize=7, ncol=2)

    # (c) Basin ellipsoid
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
        ax.set_xlabel("δ along eigenvector 1")
        ax.set_ylabel("δ along eigenvector 2")
        ax.set_aspect("equal")
    except Exception:
        ax.text(0.5, 0.5, "Basin projection\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title(f"(c) Basin of Attraction\nR(x*) ≈ {lyap_result['resilience']:.4f}")
    ax.legend(fontsize=8)

    fig.suptitle("Model 3: Lyapunov Stability & Resilience (Individual-Level Calibration)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  REPRESENTATION COHERENCE: Stage 1 ↔ Stage 2 Bridge
# ═══════════════════════════════════════════════════════════════════════════════

def compute_coherence_metrics(pca_result, sys, x_star, S_obs, cols):
    """Measure      between reduced-form PCA (Stage 1) and dynamical model (Stage 2)."""
    K = len(cols)

    w_empirical = pca_result.components_[0]
    if w_empirical.mean() < 0:
        w_empirical = -w_empirical

    obs_eigvals, obs_eigvecs = np.linalg.eigh(S_obs)
    idx = np.argsort(obs_eigvals)[::-1]
    v1_obs = obs_eigvecs[:, idx[0]]
    if v1_obs.mean() < 0:
        v1_obs = -v1_obs

    J = sys.jacobian_at_equilibrium(x_star)
    eigs_J, evecs_J = np.linalg.eig(J)
    dom_idx = np.argmax(np.real(eigs_J))
    v_dominant_J = np.real(evecs_J[:, dom_idx])
    if v_dominant_J.mean() < 0:
        v_dominant_J = -v_dominant_J

    equilibria = sys.simulate_population(n_individuals=500, seed=42)
    eq_std = (equilibria - equilibria.mean(axis=0)) / (equilibria.std(axis=0) + 1e-8)
    Sigma_model = np.cov(eq_std.T)
    mod_eigvals, mod_eigvecs = np.linalg.eigh(Sigma_model)
    idx_m = np.argsort(mod_eigvals)[::-1]
    v1_model = mod_eigvecs[:, idx_m[0]]
    if v1_model.mean() < 0:
        v1_model = -v1_model

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    weak_coherence = cosine_sim(v1_obs, v1_model)
    medium_coherence = cosine_sim(w_empirical, v_dominant_J)

    obs_ratio = obs_eigvals[idx[0]] / obs_eigvals[idx[1]] if obs_eigvals[idx[1]] > 0 else np.inf
    mod_ratio = mod_eigvals[idx_m[0]] / mod_eigvals[idx_m[1]] if mod_eigvals[idx_m[1]] > 0 else np.inf
    sign_discipline = bool(np.all(w_empirical > 0) and np.all(sys.M >= 0))
    coupling_totals = sys.M.sum(axis=0)
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
    """
    Individual-level Alkire-Foster capabilities analysis on Ryff subscales.
    Returns (result_df, summary_dict).
    """
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

    censored_deprived = deprived * identified[:, np.newaxis]
    dim_contribution = censored_deprived.mean(axis=0)
    dim_contrib_pct = dim_contribution / (dim_contribution.sum() + 1e-8)

    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1)
    cv = row_stds / (row_means + 1e-8)
    h = 1.0 - cv / (cv.max() + 1e-8)
    h = np.clip(h, 0.01, 1.0)

    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    f_raw = pca.fit_transform(X_std).ravel()
    if f_raw.mean() < 0:
        f_raw = -f_raw
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
    """Four-panel individual-level capabilities figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    labels = [short_label(c) for c in cols]
    dep_colors = [palette_color(c) for c in cols]

    # (a) f vs f_cap
    ax = axes[0, 0]
    pm = af_df["nussbaum_pass"] == 1
    ax.scatter(af_df.loc[pm, "f_z"], af_df.loc[pm, "f_cap_z"],
               alpha=0.15, s=8, color="#2EAA6A", label="Pass all", edgecolors="none")
    ax.scatter(af_df.loc[~pm, "f_z"], af_df.loc[~pm, "f_cap_z"],
               alpha=0.15, s=8, color="#E8622A", label="Below ≥1", edgecolors="none")
    ax.plot([-4, 4], [-4, 4], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("f (compensatory)")
    ax.set_ylabel("f_cap (capability-adjusted)")
    ax.set_title(f"(a) f vs f_cap  |  r = {af_summary['f_fcap_corr']:.3f}")
    ax.legend(fontsize=7, markerscale=3)

    # (b) Gap by deprivation breadth
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

    # (c) Deprivation rates
    ax = axes[1, 0]
    dep_rates = [af_df[f"deprived_{c}"].mean() * 100 for c in cols]
    ax.barh(labels, dep_rates, color=dep_colors, alpha=0.8)
    ax.set_xlabel("% below threshold")
    ax.set_title("(c) Per-Dimension Deprivation Rates")

    # (d) Domain-specific gap
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
#  INDIVIDUAL-LEVEL COUPLING & RESILIENCE PROXY
# ═══════════════════════════════════════════════════════════════════════════════

def individual_coupling_resilience(df, cols):
    """Within-person coupling proxy (cross-sectional: 1 - normalized CV)."""
    X = df[cols].values.astype(float)
    person_means = X.mean(axis=1)
    person_stds = X.std(axis=1)
    cv = person_stds / (person_means + 1e-8)
    coupling_proxy = 1.0 - (cv - cv.min()) / (cv.max() - cv.min() + 1e-8)
    person_min = X.min(axis=1)
    person_max = X.max(axis=1)
    weakest_domain = [cols[i] for i in X.argmin(axis=1)]

    return pd.DataFrame({
        "coupling_proxy": coupling_proxy, "person_mean": person_means,
        "person_std": person_stds, "cv": cv,
        "min_domain_score": person_min, "max_domain_score": person_max,
        "domain_range": person_max - person_min, "weakest_domain": weakest_domain,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  COUPLING SWEEP & RESILIENCE SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_sweep(base_sys, n_points=15, n_individuals=80):
    """Sweep M̄ to test optimal-coupling prediction."""
    m_values = np.linspace(0.01, 0.5, n_points)
    results = {"m_bar": [], "pct_var_pc1": [], "mean_eq": [],
               "resilience": [], "recovery_speed": [], "basin_radius": []}

    for m_scale in m_values:
        M_scaled = base_sys.M * (m_scale / base_sys.mean_coupling_strength())
        np.fill_diagonal(M_scaled, 0.0)
        sys_test = MutualismSystem(base_sys.K, a=base_sys.a.copy(),
                                    C=base_sys.C.copy(), M=M_scaled)
        equilibria = sys_test.simulate_population(n_individuals=n_individuals, seed=42)
        cov = np.cov(equilibria.T)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        pct_pc1 = evals[0] / evals.sum() * 100

        try:
            x_star = sys_test.fast_equilibrium()
            lyap = lyapunov_analysis(sys_test, x_star)
            resilience = lyap["resilience"]
            recovery_speed = lyap["recovery_speed"]
            basin_radius = lyap["basin_radius"]
        except Exception:
            resilience = recovery_speed = basin_radius = 0.0

        results["m_bar"].append(m_scale)
        results["pct_var_pc1"].append(pct_pc1)
        results["mean_eq"].append(equilibria.mean())
        results["resilience"].append(resilience)
        results["recovery_speed"].append(recovery_speed)
        results["basin_radius"].append(basin_radius)

    return pd.DataFrame(results)


def coupling_resilience_signature(base_sys, n_individuals=50, shock_magnitudes=None, seed=42):
    """Simulate coupling × shock interaction on recovery variance."""
    if shock_magnitudes is None:
        shock_magnitudes = np.array([0.05, 0.10, 0.20, 0.35, 0.50])

    coupling_levels = np.linspace(0.03, 0.45, 8)
    rng = np.random.RandomState(seed)
    K = base_sys.K
    T_MAX = 100.0
    RECOVERY_THRESHOLD = 0.05

    trial_records = []
    for m_scale in coupling_levels:
        M_scaled = base_sys.M * (m_scale / (base_sys.mean_coupling_strength() + 1e-8))
        np.fill_diagonal(M_scaled, 0.0)

        for shock_frac in shock_magnitudes:
            for trial in range(n_individuals):
                C_i = base_sys.C * np.exp(rng.normal(0, 0.12, K))
                a_i = base_sys.a * np.exp(rng.normal(0, 0.05, K))
                M_i = M_scaled * np.exp(rng.normal(0, 0.03, (K, K)))
                np.fill_diagonal(M_i, 0.0)

                sys_i = MutualismSystem(K, a=a_i, C=C_i, M=M_i)
                x_star_i = sys_i.analytical_equilibrium()
                if not (np.all(x_star_i > 0) and np.all(np.isfinite(x_star_i))):
                    continue

                shock_domain = rng.randint(K)
                x_perturbed = x_star_i.copy()
                x_perturbed[shock_domain] *= (1.0 - shock_frac)
                x_perturbed = np.maximum(x_perturbed, 0.01)

                try:
                    sol = solve_ivp(
                        sys_i.deterministic_rhs, (0, T_MAX), x_perturbed,
                        method="RK45", max_step=1.0,
                        t_eval=np.linspace(0, T_MAX, 500), rtol=1e-6, atol=1e-8)
                    deviations = np.abs(sol.y.T - x_star_i) / (x_star_i + 1e-8)
                    max_dev = deviations.max(axis=1)
                    recovered_idx = np.where(max_dev < RECOVERY_THRESHOLD)[0]
                    if len(recovered_idx) > 0:
                        recovery_time = sol.t[recovered_idx[0]]
                        recovered = True
                    else:
                        recovery_time = T_MAX
                        recovered = False
                    final_dev = max_dev[-1]
                except Exception:
                    recovery_time = T_MAX
                    recovered = False
                    final_dev = 1.0

                trial_records.append({
                    "coupling": m_scale, "shock_frac": shock_frac,
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
    """Four-panel coupling sweep figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(sweep_df["m_bar"], sweep_df["pct_var_pc1"], "o-", color="#3B7DD8", lw=2)
    ax.set_xlabel("Mean coupling M̄")
    ax.set_ylabel("PC1 variance (%)")
    ax.set_title("(a) Coupling → f Dominance")

    ax = axes[0, 1]
    ax.plot(sweep_df["m_bar"], sweep_df["mean_eq"], "o-", color="#2EAA6A", lw=2)
    ax.set_xlabel("Mean coupling M̄")
    ax.set_ylabel("Mean equilibrium")
    ax.set_title("(b) Coupling → Wellbeing Level")

    ax = axes[1, 0]
    ax.plot(sweep_df["m_bar"], sweep_df["resilience"], "o-", color="#E67E22", lw=2)
    ax.set_xlabel("Mean coupling M̄")
    ax.set_ylabel("Resilience R(x*)")
    ax.set_title("(c) Coupling → Resilience")
    peak_idx = sweep_df["resilience"].idxmax()
    ax.axvline(sweep_df.loc[peak_idx, "m_bar"], color="red", ls=":", lw=1.5)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(sweep_df["m_bar"], sweep_df["recovery_speed"], "o-", color="#3B7DD8", lw=2, label="Recovery speed")
    l2, = ax2.plot(sweep_df["m_bar"], sweep_df["basin_radius"], "s-", color="#E8622A", lw=2, label="Basin radius")
    ax.set_xlabel("Mean coupling M̄")
    ax.set_ylabel("Recovery speed", color="#3B7DD8")
    ax2.set_ylabel("Basin radius", color="#E8622A")
    ax.set_title("(d) The Resilience Trade-off")
    ax.legend(handles=[l1, l2], fontsize=8)

    fig.suptitle("Coupling Sweep — Optimal Regime (Individual-Level Calibration)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def plot_coupling_resilience_signature(summary_df):
    """Three-panel coupling × shock interaction figure."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    shock_vals = sorted(summary_df["shock_frac"].unique())
    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(min(shock_vals), max(shock_vals))

    for sf in shock_vals:
        sub = summary_df[summary_df["shock_frac"] == sf]
        axes[0].plot(sub["coupling"], sub["mean_recovery"], "o-",
                     color=cmap(norm(sf)), lw=1.8, label=f"{sf*100:.0f}%")
        axes[1].plot(sub["coupling"], sub["var_recovery"], "s-",
                     color=cmap(norm(sf)), lw=1.8, label=f"{sf*100:.0f}%")
        axes[2].plot(sub["coupling"], sub["recovery_rate"] * 100, "^-",
                     color=cmap(norm(sf)), lw=1.8, label=f"{sf*100:.0f}%")

    axes[0].set_title("(a) Mean Recovery Time")
    axes[1].set_title("(b) Recovery Variance — Testable Signature")
    axes[2].set_title("(c) Recovery Success Rate")
    axes[2].set_ylim(-5, 105)
    for ax in axes:
        ax.set_xlabel("Coupling M̄")
        ax.legend(fontsize=7, ncol=2, title="Shock")

    fig.suptitle("Coupling–Resilience Signature (Individual-Level Calibration)",
                 fontsize=11, y=1.04)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMOGRAPHIC & DUAL CONTINUA PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_demographic_f(df):
    """Violin plots of f by age, sex, education."""
    if "f_score" not in df.columns:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    if "B1PAGE_M2" in df.columns:
        df = df.copy()
        df["age_group"] = pd.cut(df["B1PAGE_M2"], bins=[34,44,54,64,74,90],
                                  labels=["35-44","45-54","55-64","65-74","75+"])
        sns.violinplot(data=df, x="age_group", y="f_score",
                       order=["35-44","45-54","55-64","65-74","75+"],
                       ax=axes[0], palette="Blues", inner="quartile", density_norm="width")
        axes[0].set_title("f by Age")
        axes[0].axhline(0, color="gray", ls="--", lw=0.8)

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
    """f vs negative affect (dual continua test)."""
    if "B1SNEGAF" not in df.columns or "f_score" not in df.columns:
        return None
    df_plot = df[["f_score", "B1SNEGAF"]].dropna()
    r, p_val = stats.pearsonr(df_plot["f_score"], df_plot["B1SNEGAF"])

    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1:, :-1])
    ax_top   = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    hb = ax_main.hexbin(df_plot["f_score"], df_plot["B1SNEGAF"],
                        gridsize=40, cmap="YlOrRd", mincnt=1)
    xs = np.linspace(df_plot["f_score"].min(), df_plot["f_score"].max(), 100)
    slope, intercept, *_ = stats.linregress(df_plot["f_score"], df_plot["B1SNEGAF"])
    ax_main.plot(xs, intercept + slope * xs, color="#3B7DD8", lw=2)
    ax_main.set_xlabel("f score (flourishing)")
    ax_main.set_ylabel("Negative Affect (p proxy)")
    ax_main.text(0.97, 0.95,
                 f"r = {r:.3f}\nShared var: {r**2*100:.1f}%\nIndependent: {(1-r**2)*100:.1f}%",
                 transform=ax_main.transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.85))
    ax_top.hist(df_plot["f_score"], bins=50, color="#3B7DD8", alpha=0.7)
    ax_top.axis("off")
    ax_right.hist(df_plot["B1SNEGAF"], bins=30, orientation="horizontal",
                  color="#E8622A", alpha=0.7)
    ax_right.axis("off")
    fig.suptitle("f vs p: Dual Continua Test", fontsize=12, y=1.0)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-PERSON RESILIENCE (back-solve C from observed scores)
# ═══════════════════════════════════════════════════════════════════════════════

def person_resilience_scores(X_raw, base_sys, cols):
    """Compute per-person resilience by treating observed scores as equilibrium."""
    N, K = X_raw.shape
    M_tilde = base_sys.M.copy()
    np.fill_diagonal(M_tilde, 0.0)
    A = np.eye(K) - M_tilde
    results = []

    for i in range(N):
        x_i = X_raw[i]
        C_i = A @ x_i
        C_i = np.maximum(C_i, 0.01)
        sys_i = MutualismSystem(K, a=base_sys.a.copy(), C=C_i, M=base_sys.M.copy())
        try:
            lyap = lyapunov_analysis(sys_i, x_i)
            results.append({
                "lambda_1_real": np.real(lyap["lambda_1"]),
                "tau": lyap["tau"], "recovery_speed": lyap["recovery_speed"],
                "basin_radius": lyap["basin_radius"], "resilience": lyap["resilience"],
                "stable": lyap["stable"], "mean_coupling": sys_i.mean_coupling_strength(),
            })
        except Exception:
            results.append({
                "lambda_1_real": np.nan, "tau": np.nan, "recovery_speed": np.nan,
                "basin_radius": np.nan, "resilience": np.nan,
                "stable": False, "mean_coupling": np.nan,
            })
    return pd.DataFrame(results)


def plot_resilience_vs_f(df_scored, resilience_df, cols):
    """Scatter: f vs resilience, f vs return time, coupling vs f."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    valid = resilience_df["resilience"].notna() & (resilience_df["resilience"] > 0)
    if valid.sum() > 5:
        ax = axes[0]
        ax.scatter(df_scored.loc[valid.values, "f_score"],
                   resilience_df.loc[valid, "resilience"],
                   alpha=0.1, s=8, color="#3B7DD8", edgecolors="none")
        r, p = stats.pearsonr(df_scored.loc[valid.values, "f_score"],
                               resilience_df.loc[valid, "resilience"])
        ax.set_xlabel("f score (z)")
        ax.set_ylabel("Resilience R(x*)")
        ax.set_title(f"(a) f vs Resilience\nr = {r:.3f}, p = {p:.3e}")

    valid_tau = resilience_df["tau"].notna() & (resilience_df["tau"] < 100)
    if valid_tau.sum() > 5:
        ax = axes[1]
        ax.scatter(df_scored.loc[valid_tau.values, "f_score"],
                   resilience_df.loc[valid_tau, "tau"],
                   alpha=0.1, s=8, color="#E67E22", edgecolors="none")
        ax.set_xlabel("f score (z)")
        ax.set_ylabel("Return time τ")
        ax.set_title("(b) f vs Return Time τ")

    valid_m = resilience_df["mean_coupling"].notna()
    if valid_m.sum() > 5:
        ax = axes[2]
        ax.scatter(resilience_df.loc[valid_m, "mean_coupling"],
                   df_scored.loc[valid_m.values, "f_score"],
                   alpha=0.1, s=8, color="#2EAA6A", edgecolors="none")
        ax.set_xlabel("Mean coupling M̄")
        ax.set_ylabel("f score (z)")
        ax.set_title("(c) Coupling vs f")

    fig.suptitle("Predicted Relationships — Coupling, Resilience, f\n"
                 "(Individual-Level Calibration)", fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def main():
    print("\n" + "═" * 70)
    print("  COMBINED f-FACTOR ANALYSIS")
    print("  Models 1–4 · Individual-Level Calibration")
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
        df_raw = load_midus(MIDUS_DATA_PATH)
        data_label = "MIDUS 2 (ICPSR 4652)"
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")

    df_indiv, X_raw_indiv, X_scaled_indiv, indiv_cols = preprocess_midus(df_raw)
    N_indiv = len(df_indiv)
    S_indiv = np.cov(X_scaled_indiv.T, ddof=1)

    # ══════════════════════════════════════════════════════════════════════════
    #  INDIVIDUAL-LEVEL POSITIVE MANIFOLD & PCA
    # ══════════════════════════════════════════════════════════════════════════
    print_section("Positive Manifold — Individual Level")
    corr = report_manifold(df_indiv, indiv_cols, label=data_label)

    fig_corr = plot_corr_heatmap(df_indiv, indiv_cols, title_suffix=data_label)
    fig_corr.savefig(OUTPUT_DIR / "01_correlation_matrix.png")
    print(f"  → {OUTPUT_DIR}/01_correlation_matrix.png")

    print_section("PCA — Extracting f (Individual Level)")
    df_scored, pca_indiv, var_ratio = run_pca_individual(
        X_scaled_indiv, indiv_cols, df_indiv, suffix=data_label)
    print(f"  PC1 (f proxy): {var_ratio[0]*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 1: BIFACTOR CFA (Individual Level — the primary test)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 1: Bifactor CFA (Individual Level)")
    print("  This is the proper confirmatory test — individual-level with df > 0.")

    bif_result = fit_bifactor_cfa(S_indiv, N_indiv, indiv_cols, BIFACTOR_GROUPS_INDIV)

    print(f"  Converged: {bif_result['converged']}")
    print(f"  ω_h = {bif_result['omega_h']:.3f}")
    print(f"  ECV = {bif_result['ecv']:.3f}")
    print(f"  df = {bif_result['df']}, Chi² = {bif_result['chi2']:.2f}")
    if bif_result['df'] > 0:
        print(f"  RMSEA = {bif_result['RMSEA']:.4f}")
    else:
        print(f"  ⚠  Just-identified (df=0)")

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
    #  MODEL 2: MUTUALISM DYNAMICAL SYSTEM (Individual-Level Calibration)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 2: Mutualism Dynamical System")
    print("  Calibrating from INDIVIDUAL-LEVEL partial correlations.")
    print("  This addresses the ecological fallacy (Robinson 1950):")
    print("  coupling reflects within-person associations, not between-country aggregates.")

    sys_indiv, indiv_diag = calibrate_mutualism_from_individual_data(X_raw_indiv, indiv_cols)

    print(f"\n  Individual-level calibration (N = {indiv_diag['N_individuals']:,}):")
    print(f"    Carrying capacities C = {indiv_diag['carrying_cap'].round(3)}")
    print(f"    Mean coupling M̄ = {indiv_diag['mean_coupling']:.4f}")

    # This is the primary system for all downstream analyses
    sys = sys_indiv
    x_star = sys.analytical_equilibrium()

    print(f"\n  Equilibrium x* = {x_star.round(3)}")
    print(f"  x*/C ratio = {(x_star / sys.C).round(3)}")
    print(f"  All x* > C: {np.all(x_star > sys.C)}  (mutualism lifts equilibrium)")

    # Simulate population
    print(f"\n  Simulating population (200 individuals) …", end=" ", flush=True)
    equilibria = sys.simulate_population(n_individuals=200)
    print("done.")

    sim_corr = np.corrcoef(equilibria.T)
    off_diag = sim_corr[np.triu_indices(len(indiv_cols), k=1)]
    print(f"  Simulated positive manifold:")
    print(f"    All positive: {np.all(off_diag > 0)}")
    print(f"    Mean r = {off_diag.mean():.3f}")

    sim_std = StandardScaler().fit_transform(equilibria)
    sim_pca = PCA()
    sim_pca.fit(sim_std)
    print(f"    Simulated PC1: {sim_pca.explained_variance_ratio_[0]*100:.1f}% "
          f"(vs {var_ratio[0]*100:.1f}% observed)")

    fig_mut = plot_mutualism_demo(sys, equilibria, indiv_cols)
    fig_mut.savefig(OUTPUT_DIR / "03_mutualism_model.png")
    print(f"\n  → {OUTPUT_DIR}/03_mutualism_model.png")

    # ── Ecological comparison (if available) ──────────────────────────────────
    eco_data = load_ecological_data()
    if eco_data is not None:
        df_eco, X_raw_eco, X_scaled_eco, S_eco, eco_cols, N_eco = eco_data
        sys_eco = calibrate_mutualism_from_ecological_data(X_raw_eco, eco_cols)

        # Can only compare if dimensionality matches
        if sys_eco.K == sys_indiv.K:
            print_section("ECOLOGICAL FALLACY DIAGNOSTIC")
            eco_comp = compare_calibrations(sys_eco, sys_indiv, indiv_cols)
            print(f"  Coupling correlation (eco vs indiv): r = {eco_comp['coupling_correlation']:.3f}")
            print(f"  Coupling Frobenius distance: {eco_comp['frobenius_diff']:.4f}")
            print(f"  Equilibrium correlation: r = {eco_comp['equilibrium_correlation']:.3f}")
            print(f"  Resilience — eco: {eco_comp['eco_resilience']:.4f}, indiv: {eco_comp['indiv_resilience']:.4f}")
            print(f"  Return time — eco: {eco_comp['eco_tau']:.2f}, indiv: {eco_comp['indiv_tau']:.2f}")
            if eco_comp['coupling_correlation'] > 0.7:
                print(f"  → Structures aligned (r > 0.7). Ecological is a reasonable proxy,")
                print(f"    but individual-level calibration remains preferred.")
            else:
                print(f"  → Structures DIVERGE (r ≤ 0.7). Individual calibration is essential.")
        else:
            print(f"  ⚠  Dimension mismatch (eco: {sys_eco.K}, indiv: {sys_indiv.K}) — skipping comparison.")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 3: LYAPUNOV STABILITY & RESILIENCE
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 3: Lyapunov Stability & Resilience")

    lyap = lyapunov_analysis(sys, x_star)

    print(f"  Jacobian eigenvalues: {lyap['eigenvalues'].round(4)}")
    print(f"  Stable: {lyap['stable']}")
    print(f"  Dominant eigenvalue λ₁ = {lyap['lambda_1']:.4f}")
    print(f"  Return time τ = {lyap['tau']:.2f}")
    print(f"  Recovery speed = {lyap['recovery_speed']:.4f}")
    print(f"  Basin radius = {lyap['basin_radius']:.4f}")
    print(f"  Resilience R(x*) = {lyap['resilience']:.4f}")

    fig_lyap = plot_lyapunov_analysis(lyap, x_star, sys, indiv_cols)
    fig_lyap.savefig(OUTPUT_DIR / "04_lyapunov_stability.png")
    print(f"\n  → {OUTPUT_DIR}/04_lyapunov_stability.png")

    # Per-person resilience
    print(f"\n  Computing per-person resilience …", end=" ", flush=True)
    resilience_df = person_resilience_scores(X_raw_indiv, sys, indiv_cols)
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

    coherence = compute_coherence_metrics(pca_indiv, sys, x_star, S_indiv, indiv_cols)

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
    #  COUPLING SWEEP (Individual-Level Calibration)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("COUPLING SWEEP (Individual-Level System)")
    print("  Sweeping M̄ to test optimal-coupling prediction …", end=" ", flush=True)
    sweep_df = coupling_sweep(sys, n_points=15, n_individuals=80)
    print("done.")

    peak = sweep_df.loc[sweep_df["resilience"].idxmax()]
    print(f"  Peak resilience at M̄ = {peak['m_bar']:.3f}")
    print(f"    PC1 variance: {peak['pct_var_pc1']:.1f}%")
    print(f"    Resilience: {peak['resilience']:.4f}")

    fig_sweep = plot_coupling_sweep(sweep_df)
    fig_sweep.savefig(OUTPUT_DIR / "07_coupling_sweep.png")
    print(f"\n  → {OUTPUT_DIR}/07_coupling_sweep.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  COUPLING–RESILIENCE SIGNATURE
    # ══════════════════════════════════════════════════════════════════════════
    print_section("COUPLING–RESILIENCE SIGNATURE")
    print("  Simulating coupling × shock interaction …", end=" ", flush=True)
    sig_trials, sig_summary = coupling_resilience_signature(sys, n_individuals=50, seed=42)
    print("done.")

    high_coupling = sig_summary["coupling"] > sig_summary["coupling"].median()
    large_shock = sig_summary["shock_frac"] > sig_summary["shock_frac"].median()
    var_hc_ls = sig_summary.loc[high_coupling & large_shock, "var_recovery"].mean()
    var_lc_ls = sig_summary.loc[~high_coupling & large_shock, "var_recovery"].mean()
    var_hc_ss = sig_summary.loc[high_coupling & ~large_shock, "var_recovery"].mean()

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

    # Demographics
    fig_demo = plot_demographic_f(df_scored)
    if fig_demo:
        fig_demo.savefig(OUTPUT_DIR / "09_demographic_f.png")
        print(f"\n  → {OUTPUT_DIR}/09_demographic_f.png")

    # Dual continua
    fig_fp = plot_f_vs_p(df_scored)
    if fig_fp:
        fig_fp.savefig(OUTPUT_DIR / "10_f_vs_p.png")
        print(f"\n  → {OUTPUT_DIR}/10_f_vs_p.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVE & SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    # Merge all computed columns into scored df
    for col in af_df.columns:
        if col not in df_scored.columns:
            df_scored[col] = af_df[col].values
    for col in coupling_df.columns:
        df_scored[f"coupling_{col}"] = coupling_df[col].values
    for col in resilience_df.columns:
        df_scored[f"lyap_{col}"] = resilience_df[col].values

    out_path = Path("wellbeing_data/midus_scored.csv")
    out_path.parent.mkdir(exist_ok=True)
    df_scored.to_csv(out_path, index=False)
    print(f"\n  ✓ Scored data saved → {out_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FULL IMPLEMENTATION COMPLETE")
    print("═" * 70)
    print(f"  Data:                    {data_label}")
    print(f"  N (individuals):         {N_indiv:,}")
    print(f"  Calibration level:       INDIVIDUAL (within-person partial correlations)")
    print(f"  PC1 var explained:       {var_ratio[0]*100:.1f}%")
    print(f"  Model 1 — ω_h:          {bif_result['omega_h']:.3f}")
    print(f"  Model 1 — ECV:          {bif_result['ecv']:.3f}")
    print(f"  Model 2 — Mean M̄:       {sys.mean_coupling_strength():.4f}")
    print(f"  Model 2 — Sim PC1:      {sim_pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  Model 3 — τ:            {lyap['tau']:.2f}")
    print(f"  Model 3 — R(x*):        {lyap['resilience']:.4f}")
    print(f"  Coherence — Weak:       {coherence['weak_coherence']:.4f}")
    print(f"  Coherence — Medium:     {coherence['medium_coherence']:.4f}")
    print(f"  Coherence — Sign:       {'PASS' if coherence['sign_discipline'] else 'FAIL'}")
    print(f"  Model 4 — M₀:          {af_summary['M0']:.3f}")
    print(f"  Model 4 — f↔f_cap:     {af_summary['f_fcap_corr']:.3f}")
    if valid_res.sum() > 10:
        print(f"  Cross-model — f↔R:     {r_corr:.3f}")
    print(f"  Coupling proxy ↔ gap:   {r_cp:.3f}")
    print(f"  Resilience sig HC·LS:   {var_hc_ls:.1f}")
    print(f"  Resilience sig LC·LS:   {var_lc_ls:.1f}")
    print(f"\n  Figures: {OUTPUT_DIR}/01–10_*.png")
    print(f"  Scored data: {out_path}")
    print()


if __name__ == "__main__":
    main()

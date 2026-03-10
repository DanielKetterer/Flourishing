"""
03_individual_level_f_analysis.py
─────────────────────────────────────────────────────────────────────────────
Individual-level replication of the f-factor analysis using Ryff's
Psychological Wellbeing Scale (6 dimensions) from MIDUS 2.

WHY THIS IS A DIFFERENT BEAST THAN THE COUNTRY-LEVEL ANALYSIS
──────────────────────────────────────────────────────────────
The country-level analysis (scripts 01-02) works on population averages.
Any positive manifold found there is an *ecological correlation* — it tells
us that countries where citizens are happier also tend to have longer lives,
more social support, etc. This is real but it cannot confirm the mutualism
model, which makes claims about within-person dynamics.

Individual-level data allow us to ask the right question: does a person who
scores high on Positive Relations *also* tend to score high on Purpose in Life
and Self-Acceptance? If the answer is yes, across thousands of respondents with
wildly different country/culture/SES backgrounds, *that* is evidence for the
kind of coupled wellbeing system the paper describes.

DATA: MIDUS 2 (ICPSR 4652) — Ryff Psychological Wellbeing Scale
───────────────────────────────────────────────────────────────────
MIDUS (Midlife in the United States) is the flagship US longitudinal health
and wellbeing study run by Carol Ryff (University of Wisconsin), the same
researcher who developed the Psychological Wellbeing Scale.

MIDUS 2 (Wave 2, 2004-2006) is the recommended starting point:
  - N ≈ 4,963 (core sample)
  - 6 Ryff PWB composite scores already constructed
  - Longitudinal: linkable to MIDUS 1 (1995-96) and MIDUS 3 (2013-14) via M2ID
  - Free public access after ICPSR account registration

HOW TO ACCESS THE DATA (required steps)
─────────────────────────────────────────
 1. Create a free ICPSR account at: https://www.icpsr.umich.edu/web/pages/ICPSR/
 2. Navigate to: https://www.icpsr.umich.edu/web/NACDA/studies/4652
 3. Click "Download" → select "Delimited" format (produces .tsv files)
 4. You'll need to agree to the standard ICPSR terms (no restrictions for public use)
 5. Download "DS0001: Main Data" — this is the core survey file
 6. Unzip and place the .tsv file in:  midus_data/ICPSR_04652-0001-Data.tsv
    (or update MIDUS_DATA_PATH below)

MIDUS VARIABLE MAP (PERMA+H schema)
─────────────────────────────────────
  Ryff PWB composites (mean of 3-item subscales, 1-7 scale):
    B1SPWBA  → Autonomy               (E proxy: self-direction)
    B1SPWBE  → Environmental Mastery  (A proxy: sense of control)
    B1SPWBG  → Personal Growth        (E proxy: engagement/development)
    B1SPWBR  → Positive Relations     (R: relationships)
    B1SPWBU  → Purpose in Life        (M: meaning)
    B1SPWBS  → Self-Acceptance        (P proxy: positive self-view)

  Affect (separate module, 1-5 scale, mean of items):
    B1SPOSPA → Positive Affect         (P: positive emotion — best proxy)
    B1SPOSNA → Negative Affect         (reference / p-factor proxy)

  Life satisfaction (0-10 scale):
    B1SB1    → Overall Life Satisfaction (M supplement)

  Self-rated health (1=excellent → 5=poor, REVERSED to 1-5 good):
    B1SA11W  → Self-rated Health       (H: health)

  Demographics:
    M2ID     → respondent ID (links across all MIDUS waves)
    B1PAGE_M2→ Age
    B1PRSEX  → Sex (1=male, 2=female)
    B1PF7A   → Education (1-12 ordinal)

SCRIPT MODES
────────────
  MODE = "synthetic"  →  Generate synthetic data matching known MIDUS 2
                          psychometric properties. Runs immediately,
                          no download required. Useful for code testing.

  MODE = "midus"      →  Parse actual MIDUS 2 data from disk.
                          Requires the ICPSR download.

  MODE = "compare"    →  Run both and compare ecological vs individual-level
                          positive manifold. Requires the ICPSR download.

ANALYSES PERFORMED
───────────────────
  1. Descriptive statistics (raw and standardized)
  2. Polychoric / Pearson correlation matrix
  3. Scree plot (PCA)
  4. PC1 factor loadings (= f proxy) with 1000-bootstrap CIs
  5. Bifactor CFA via semopy (if installed), reporting:
       - omega_h  (variance in items explained by general factor)
       - omega_t  (total reliable variance)
       - ECV      (explained common variance ratio)
  6. f score distribution (histogram + KDE)
  7. Demographic subgroup f profiles (age, sex, education)
  8. Correlation between individual f and p (psychopathology proxy = neg. affect)
     → Tests the dual continua model (f and p as distinct dimensions)

KNOWN MIDUS 2 PSYCHOMETRIC BENCHMARKS (from published literature)
───────────────────────────────────────────────────────────────────
  Cronbach alphas (6 subscales): 0.70–0.85
  Inter-subscale correlations: 0.40–0.77 (highest: mastery ↔ self-acceptance)
  Autonomy is the most distinct subscale (lowest inter-correlations)
  Springer & Hauser (2006): correlations between some subscales approach .95
  after removing measurement error (latent-variable correlations)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# ─── Config ───────────────────────────────────────────────────────────────────

MODE = "midus"

MIDUS_DATA_PATH = Path("04652-0001-Data.tsv")
OUTPUT_DIR      = Path("wellbeing_figures_individual")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
BOOTSTRAP_N = 1000

# ─── Variable mappings ────────────────────────────────────────────────────────

# All 6 Ryff subscales — these ARE the theoretical building blocks of f
RYFF_COLS = [
    "B1SPWBA1",  # Autonomy            (MIDUS-1 version: 3-item sum ÷ 3 → 1-7)
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
    "B1SPWBA1": "E",   # Engagement / self-direction
    "B1SPWBE1": "A",   # Accomplishment / mastery
    "B1SPWBG1": "E",   # Engagement / growth
    "B1SPWBR1": "R",   # Relationships
    "B1SPWBU1": "M",   # Meaning / purpose
    "B1SPWBS1": "P",   # Positive self-view
}

# B1SNEGAF = negative affect (MIDUS 2 variable, ~7 items, higher=worse)
# B1SPOSPA = positive affect (already a composite mean score)
AUX_COLS = ["B1SPOSPA", "B1SNEGAF", "B1SB1", "B1SA11W"]
DEMO_COLS = ["M2ID", "B1PAGE_M2", "B1PRSEX", "B1PF7A"]

ALL_COLS = DEMO_COLS + RYFF_COLS + AUX_COLS

PALETTE = {
    "B1SPWBA1": "#3B7DD8",
    "B1SPWBE1": "#E8622A",
    "B1SPWBG1": "#2EAA6A",
    "B1SPWBR1": "#9B59B6",
    "B1SPWBU1": "#E67E22",
    "B1SPWBS1": "#1ABC9C",
}

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ─── 1. Synthetic data generator ──────────────────────────────────────────────
#
# Built from published MIDUS 2 psychometric properties:
#   - Means and SDs from Ryff et al. (2004) + MIDUS 2 documentation
#   - Correlation matrix from Springer & Hauser (2006) and Ryff & Keyes (1995)
#   - Skewness / floor effects known from MIDUS reports
#
# This is a multivariate normal approximation. Real MIDUS data have slight
# negative skew on most subscales (ceiling effects) and integer-only responses.

# MIDUS 2 known inter-subscale correlations (Pearson, total sample)
# Source: Springer & Hauser (2006) Table 2; Lachman et al. (2008)
#         Order: Auton, Mastery, Growth, PoRel, Purpose, Self-Acc
MIDUS_CORR_MATRIX = np.array([
    # Auton   Mastery  Growth   PoRel   Purpose  Self-Acc
    [1.00,    0.47,    0.47,    0.34,    0.44,    0.40],  # Autonomy
    [0.47,    1.00,    0.56,    0.54,    0.65,    0.77],  # Env. Mastery
    [0.47,    0.56,    1.00,    0.47,    0.60,    0.49],  # Personal Growth
    [0.34,    0.54,    0.47,    1.00,    0.52,    0.59],  # Positive Relations
    [0.44,    0.65,    0.60,    0.52,    1.00,    0.66],  # Purpose in Life
    [0.40,    0.77,    0.49,    0.59,    0.66,    1.00],  # Self-Acceptance
])

# Means and SDs on 1-7 scale (MIDUS 2 core sample)
MIDUS_MEANS = np.array([4.80, 5.19, 5.53, 5.36, 5.38, 5.18])
MIDUS_SDS   = np.array([0.82, 0.86, 0.82, 0.94, 0.93, 0.97])

# Aux variable distributions
# B1SPOSPA (positive affect): mean ≈ 3.25, sd ≈ 0.65  (1-5 scale)
# B1SPOSNA (negative affect): mean ≈ 1.85, sd ≈ 0.58
# B1SB1    (life satisfaction): mean ≈ 7.4, sd ≈ 1.8  (0-10)
# B1SA11W  (health, 1=excellent→5=poor, reversed): mean ≈ 3.5, sd ≈ 0.9


def generate_synthetic_midus(n: int = 5000) -> pd.DataFrame:
    """
    Simulate individual-level MIDUS 2 data matching published
    psychometric properties.

    The Cholesky decomposition of the correlation matrix is used to induce
    the known inter-subscale correlation structure. Values are then rescaled
    to match published means/SDs and clipped to valid range [1, 7].
    """
    print(f"Generating synthetic MIDUS-like data (n={n}) …")

    # Generate correlated normal variates
    L = np.linalg.cholesky(MIDUS_CORR_MATRIX)
    Z = np.random.randn(n, 6)
    X_corr = Z @ L.T

    # Rescale to MIDUS means/SDs
    X_scaled = X_corr * MIDUS_SDS + MIDUS_MEANS

    # Add slight ceiling effect (real MIDUS has left-skewed distributions)
    # by adding a small positive skew correction
    X_scaled = X_scaled - 0.15 * (X_scaled - MIDUS_MEANS) ** 2 / MIDUS_SDS
    X_scaled = np.clip(X_scaled, 1, 7)

    df = pd.DataFrame(X_scaled, columns=RYFF_COLS)

    # ── Aux variables ──
    # Positive affect correlates ~.55 with mean Ryff
    ryff_mean = df[RYFF_COLS].mean(axis=1)
    df["B1SPOSPA"] = np.clip(
        0.55 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 0.65 + 3.25
        + np.random.randn(n) * 0.65 * np.sqrt(1 - 0.55**2),
        1, 5
    )
    # Negative affect anti-correlates ~-.45 with mean Ryff
    df["B1SNEGAF"] = np.clip(
        -0.45 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 0.58 + 1.85
        + np.random.randn(n) * 0.58 * np.sqrt(1 - 0.45**2),
        1, 5
    )
    # Life satisfaction correlates ~.60 with mean Ryff
    df["B1SB1"] = np.clip(
        0.60 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 1.8 + 7.4
        + np.random.randn(n) * 1.8 * np.sqrt(1 - 0.60**2),
        0, 10
    )
    # Self-rated health correlates ~.35 with mean Ryff (1=best, 5=worst → reversed)
    df["B1SA11W"] = np.clip(
        0.35 * (ryff_mean - ryff_mean.mean()) / ryff_mean.std() * 0.9 + 3.5
        + np.random.randn(n) * 0.9 * np.sqrt(1 - 0.35**2),
        1, 5
    ).round()

    # ── Demographics ──
    df["M2ID"]      = np.arange(1, n + 1)
    df["B1PAGE_M2"] = np.clip(np.random.normal(55, 12, n), 35, 86).astype(int)
    df["B1PRSEX"]   = np.random.choice([1, 2], n, p=[0.47, 0.53])
    df["B1PF7A"]    = np.random.choice(range(1, 13), n,
                                        p=np.array([1,1,1,1,2,2,3,3,4,5,4,3])/30)

    # Simulate mild age gradient (wellbeing dips in midlife, recovers later)
    age_z = (df["B1PAGE_M2"] - 55) / 12
    age_adj = -0.05 * age_z + 0.02 * age_z**2  # U-shape
    for col in RYFF_COLS:
        df[col] = np.clip(df[col] + age_adj * MIDUS_SDS[RYFF_COLS.index(col)], 1, 7)

    print(f"  ✓ Synthetic MIDUS data: {df.shape}")
    print(f"  ⚠  This is SIMULATED data. Results will approximate but not")
    print(f"     replicate real MIDUS 2 findings. Register at ICPSR for real data.")
    return df


# ─── 2. Real MIDUS loader ─────────────────────────────────────────────────────

def load_midus(path: Path) -> pd.DataFrame:
    """
    Load MIDUS 2 DS0001 (tab-delimited, ICPSR format).
    Handles both the .tsv and SPSS-export .dat variants.
    """
    print(f"Loading MIDUS 2 data from {path} …")
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {path}\n"
            f"  Please download MIDUS 2 (ICPSR 4652) and place the main data file here.\n"
            f"  See script header for step-by-step download instructions.\n"
            f"  To run without real data, set MODE = 'synthetic' at the top of the script."
        )

    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Normalize column names (ICPSR sometimes adds leading/trailing spaces)
    df.columns = [c.strip().upper() for c in df.columns]

    # Check required columns
    missing = [c for c in RYFF_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: columns not found in data: {missing}")
        print(f"  Available columns starting with B1SPW: "
              f"{[c for c in df.columns if c.startswith('B1SPW')][:10]}")

    # Recode missing value codes and rescale to 1-7 mean.
    # The MIDUS-1 version PWB composites (suffix "1") are 3-item *sum* scores:
    #   valid range 3-21, missing coded as -1 or ≥ 90 (e.g. 98 = not calculated).
    # Divide by 3 to convert to a per-item mean on the 1-7 scale.
    for col in RYFF_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where((df[col] > 0) & (df[col] < 90))
            df[col] = df[col] / 3.0   # 3-item sum → 1-7 mean

    for col in AUX_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col] > 0)   # drop negative missing codes

    print(f"  Loaded: {len(df)} respondents, {len(df.columns)} variables")
    return df


# ─── 3. Preprocessing ─────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple:
    """
    Returns (df_clean, X_scaled, present_cols)
    Reverse-code self-rated health so higher = better.
    Drop incomplete cases on core Ryff subscales.
    """
    df = df.copy()

    # Reverse self-rated health: 1=excellent → 5, 5=poor → 1
    if "B1SA11W" in df.columns:
        df["B1SA11W"] = 6 - df["B1SA11W"]

    present = [c for c in RYFF_COLS if c in df.columns]
    df_clean = df.dropna(subset=present).copy()
    print(f"  Complete cases on Ryff subscales: {len(df_clean):,} / {len(df):,}")

    scaler   = StandardScaler()
    X        = df_clean[present].values.astype(float)
    X_scaled = scaler.fit_transform(X)

    return df_clean, X_scaled, present


# ─── 4. Correlation analysis ──────────────────────────────────────────────────

def report_manifold(df: pd.DataFrame, cols: list, label: str = ""):
    corr = df[cols].corr()
    off = corr.values[np.tril_indices_from(corr.values, k=-1)]
    print(f"\n  [{label}] Correlation summary ({len(cols)} subscales, {len(off)} pairs):")
    print(f"    Mean r : {off.mean():.3f}")
    print(f"    Min r  : {off.min():.3f}  ({'✓ all positive' if off.min() > 0 else '⚠ some negative'})")
    print(f"    Max r  : {off.max():.3f}")
    print(f"    % > 0  : {100*(off > 0).mean():.0f}%")

    # Identify most / least correlated pair
    idx = np.argmax(off)
    idxmin = np.argmin(off)
    tril_rows, tril_cols = np.tril_indices(len(cols), k=-1)
    print(f"    Strongest pair: {RYFF_LABELS.get(cols[tril_rows[idx]], cols[tril_rows[idx]])} ↔ "
          f"{RYFF_LABELS.get(cols[tril_cols[idx]], cols[tril_cols[idx]])}  r={off[idx]:.3f}")
    print(f"    Weakest pair:   {RYFF_LABELS.get(cols[tril_rows[idxmin]], cols[tril_rows[idxmin]])} ↔ "
          f"{RYFF_LABELS.get(cols[tril_cols[idxmin]], cols[tril_cols[idxmin]])}  r={off[idxmin]:.3f}")
    return corr


def plot_corr_heatmap(df: pd.DataFrame, cols: list, title_suffix: str = "") -> plt.Figure:
    corr = df[cols].corr()
    labels = [RYFF_LABELS.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        xticklabels=labels, yticklabels=labels,
    )
    off = corr.values[np.tril_indices_from(corr.values, k=-1)]
    ax.set_title(
        f"Positive Manifold: Ryff PWB Subscale Correlations\n"
        f"(Individual-level, MIDUS-like data{title_suffix})",
        pad=14,
    )
    ax.text(0.01, -0.10,
            f"Mean r = {off.mean():.3f}  |  All pairs positive: "
            f"{'✓' if (off > 0).all() else '✗'}",
            transform=ax.transAxes, fontsize=8.5, color="#444")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


# ─── 5. PCA ───────────────────────────────────────────────────────────────────

def run_pca_and_plot(
    X: np.ndarray,
    cols: list,
    df_clean: pd.DataFrame,
    suffix: str = "",
) -> tuple:
    pca = PCA()
    scores = pca.fit_transform(X)
    loadings = pca.components_.T
    ev = pca.explained_variance_
    vr = pca.explained_variance_ratio_

    print(f"\n  PCA results [{suffix}]:")
    print(f"    Eigenvalues:   {ev.round(3)}")
    print(f"    Var explained: {(vr*100).round(1)}")
    print(f"    PC1 (f proxy): {vr[0]*100:.1f}% of total variance")
    print(f"    PC1 eigenvalue: {ev[0]:.3f} (Kaiser > 1: {'✓' if ev[0] > 1 else '✗'})")

    # Bootstrap PC1 loadings
    l1 = loadings[:, 0].copy()
    n = X.shape[0]
    boot_L = np.zeros((BOOTSTRAP_N, X.shape[1]))
    for i in range(BOOTSTRAP_N):
        idx = np.random.choice(n, n, replace=True)
        pca_b = PCA(n_components=1)
        pca_b.fit(X[idx])
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
        print(f"    {RYFF_LABELS.get(col, col):<22} {val:.3f}  "
              f"[{min(lo,hi):.3f}, {max(lo,hi):.3f}]  "
              f"({RYFF_PERMA.get(col,'?')} domain)")

    # ── Scree plot ──
    n_comp = len(ev)
    fig_scree, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    comp = np.arange(1, n_comp + 1)
    ax1.plot(comp, ev, "o-", color="#3B7DD8", lw=2.2, markersize=8,
             markerfacecolor="white", markeredgewidth=2)
    ax1.axhline(1.0, color="#E8622A", ls="--", lw=1.4, label="Kaiser = 1")
    ax1.bar(1, ev[0], alpha=0.15, color="#3B7DD8", width=0.6)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title(f"Scree Plot — Individual Level [{suffix}]")
    ax1.legend(fontsize=8.5)
    ax1.annotate(f"PC1 = {vr[0]*100:.1f}%\n(f proxy)",
                 xy=(1, ev[0]), xytext=(2.3, ev[0]*0.9),
                 fontsize=8.5, color="#3B7DD8",
                 arrowprops=dict(arrowstyle="->", color="#3B7DD8"))
    ax1.set_xticks(comp)

    cumvar = np.cumsum(vr) * 100
    ax2.step(comp, cumvar, where="post", color="#2EAA6A", lw=2.2)
    ax2.fill_between(comp, cumvar, step="post", alpha=0.12, color="#2EAA6A")
    ax2.axhline(80, color="gray", ls=":", lw=1.2, label="80% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Var. (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(comp)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8.5)
    for pct, c in zip(cumvar, comp):
        ax2.text(c, pct + 1.5, f"{pct:.0f}%", ha="center", fontsize=7.5, color="#2EAA6A")
    fig_scree.suptitle(f"Individual-Level PCA — Ryff PWB [{suffix}]", fontsize=13, y=1.01)
    plt.tight_layout()

    # ── Loadings chart ──
    labels = [RYFF_LABELS.get(c, c) for c in cols]
    colors = [PALETTE.get(c, "#888") for c in cols]
    order = np.argsort(l1)[::-1]

    fig_load, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        np.arange(len(cols)),
        l1[order],
        xerr=[l1[order] - ci_lo[order], ci_hi[order] - l1[order]],
        color=[colors[i] for i in order],
        alpha=0.85, height=0.6, capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": "#555"},
    )
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(
        [f"{labels[i]}  [{RYFF_PERMA.get(cols[i],'?')}]" for i in order],
        fontsize=9.5,
    )
    ax.set_xlabel("PC1 Loading (f proxy)", fontsize=10)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-0.05, 0.65)
    ax.set_title(
        f"Factor Loadings on f (PC1)\nIndividual level [{suffix}] · "
        f"Bootstrap CIs · PERMA domain labels",
        fontsize=11,
    )
    for bar, val in zip(bars, l1[order]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8.5)
    plt.tight_layout()

    # ── Compute f scores ──
    f_raw = scores[:, 0]
    if f_raw.mean() < 0:
        f_raw = -f_raw
    f_z = (f_raw - f_raw.mean()) / f_raw.std()
    df_clean = df_clean.copy()
    df_clean["f_score"] = f_z

    return df_clean, fig_scree, fig_load, vr


# ─── 6. Bifactor CFA (omega_h) ────────────────────────────────────────────────

def run_bifactor_cfa(df: pd.DataFrame, cols: list) -> dict | None:
    """
    Run a bifactor CFA using semopy (if available).
    Returns a dict with omega_h, omega_t, ECV, and fit indices.

    The bifactor model specifies:
      - One general factor (f) loading on all 6 subscales
      - Three group factors based on conceptual clustering:
          S1: Hedonic cluster   (Self-Acceptance, Env. Mastery)
          S2: Eudaimonic core   (Purpose, Personal Growth)
          S3: Social/Autonomous (Positive Relations, Autonomy)
      - All factors orthogonal (by definition in bifactor)
    """
    try:
        import semopy
    except ImportError:
        print("\n  semopy not installed → skipping bifactor CFA.")
        print("  Install with: pip install semopy")
        print("  (PCA results already give the f proxy above.)")
        return None

    # Build model spec dynamically from actual column names.
    # RYFF_COLS order: Autonomy, Mastery, Growth, PoRel, Purpose, Self-Acc
    auton, master, growth, porel, purpose, selfacc = cols

    model_spec = (
        "f =~ " + auton + " + " + master + " + " + growth
        + " + " + porel + " + " + purpose + " + " + selfacc + "\n"
        + "S1 =~ " + selfacc + " + " + master + "\n"
        + "S2 =~ " + purpose + " + " + growth + "\n"
        + "S3 =~ " + porel + " + " + auton + "\n"
    )

    try:
        model = semopy.Model(model_spec)
        data = df[cols].dropna().copy()
        result = model.fit(data)
        params = model.inspect()

        loadings_g = params[params["lval"] == "f"]["Estimate"].values
        loadings_s = params[params["lval"].isin(["S1","S2","S3"])]["Estimate"].values
        # Residual variances: diagonal ~~ terms (lval == rval), one per observed variable
        resid_rows = params[(params["op"] == "~~") & (params["lval"] == params["rval"])
                            & (params["lval"].isin(cols))]
        residuals  = np.abs(resid_rows["Estimate"].values)

        # omega_h formula
        sum_g    = loadings_g.sum() ** 2
        sum_s_sq = (loadings_s ** 2).sum()
        sum_res  = np.abs(residuals).sum()
        omega_h  = sum_g / (sum_g + sum_s_sq + sum_res)
        omega_t  = (loadings_g.sum()**2 + sum_s_sq) / (
                    loadings_g.sum()**2 + sum_s_sq + sum_res)

        # ECV: proportion of common variance due to general factor
        common_g = loadings_g ** 2
        common_s = loadings_s ** 2
        ecv = common_g.sum() / (common_g.sum() + common_s.sum())

        print(f"\n  Bifactor CFA (semopy):")
        print(f"    omega_h  = {omega_h:.3f}  (paper range: 0.71–0.86)")
        print(f"    omega_t  = {omega_t:.3f}")
        print(f"    ECV      = {ecv:.3f}  (paper reports ~0.82 for MHC-SF)")

        # Fit indices
        try:
            stats_dict = semopy.calc_stats(model)
            cfi   = float(stats_dict.loc["CFI",   "Value"]) if "CFI"   in stats_dict.index else None
            rmsea = float(stats_dict.loc["RMSEA", "Value"]) if "RMSEA" in stats_dict.index else None
            srmr  = float(stats_dict.loc["SRMR",  "Value"]) if "SRMR"  in stats_dict.index else None
            if cfi:   print(f"    CFI      = {cfi:.3f}   (acceptable ≥ 0.95)")
            if rmsea: print(f"    RMSEA    = {rmsea:.3f}  (acceptable ≤ 0.08)")
            if srmr:  print(f"    SRMR     = {srmr:.3f}  (acceptable ≤ 0.08)")
        except Exception:
            pass  # fit indices are a bonus; don't crash if unavailable

        return {"omega_h": omega_h, "omega_t": omega_t, "ecv": ecv}
    except Exception as e:
        print(f"\n  Bifactor CFA failed: {e}")
        return None


# ─── 7. Demographic subgroup analysis ─────────────────────────────────────────

def plot_demographic_f(df: pd.DataFrame) -> plt.Figure:
    """
    Violin plots of f score by demographic subgroups.
    Tests whether the general factor varies systematically with age, sex, education.
    """
    if "f_score" not in df.columns:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Age groups
    if "B1PAGE_M2" in df.columns:
        df["age_group"] = pd.cut(
            df["B1PAGE_M2"],
            bins=[34, 44, 54, 64, 74, 90],
            labels=["35-44","45-54","55-64","65-74","75+"],
        )
        ax = axes[0]
        order = ["35-44","45-54","55-64","65-74","75+"]
        sns.violinplot(data=df, x="age_group", y="f_score",
                       order=order, ax=ax, palette="Blues",
                       inner="quartile", density_norm="width")
        ax.set_title("f Score by Age Group")
        ax.set_xlabel("Age group")
        ax.set_ylabel("f (z-standardized)")
        ax.axhline(0, color="gray", ls="--", lw=0.8)

    # Sex
    if "B1PRSEX" in df.columns:
        ax = axes[1]
        sex_labels = {1: "Male", 2: "Female"}
        df["sex_label"] = df["B1PRSEX"].map(sex_labels)
        sns.violinplot(data=df.dropna(subset=["sex_label"]),
                       x="sex_label", y="f_score",
                       ax=ax, palette={"Male": "#4A90D9", "Female": "#E05C8A"},
                       inner="quartile", density_norm="width")
        ax.set_title("f Score by Sex")
        ax.set_xlabel("Sex")
        ax.set_ylabel("")
        ax.axhline(0, color="gray", ls="--", lw=0.8)

        male_f   = df[df["B1PRSEX"] == 1]["f_score"].dropna()
        female_f = df[df["B1PRSEX"] == 2]["f_score"].dropna()
        t, p = stats.ttest_ind(male_f, female_f)
        d = (female_f.mean() - male_f.mean()) / np.sqrt(
            (female_f.std()**2 + male_f.std()**2) / 2)
        ax.text(0.05, 0.95, f"Cohen's d = {d:.3f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=8.5, va="top")

    # Education
    if "B1PF7A" in df.columns:
        edu_bins = {range(1,5): "< HS", range(5,7): "HS grad",
                    range(7,9): "Some college", range(9,11): "BA",
                    range(11,13): "Grad"}
        def edu_label(v):
            for r, lbl in edu_bins.items():
                if int(v) in r:
                    return lbl
            return "Other"
        df["edu_group"] = df["B1PF7A"].dropna().apply(edu_label)
        edu_order = ["< HS", "HS grad", "Some college", "BA", "Grad"]
        ax = axes[2]
        sns.violinplot(data=df.dropna(subset=["edu_group"]),
                       x="edu_group", y="f_score", order=edu_order,
                       ax=ax, palette="Greens", inner="quartile",
                       density_norm="width")
        ax.set_title("f Score by Education")
        ax.set_xlabel("Education level")
        ax.set_ylabel("")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.suptitle(
        "Distribution of Individual f Scores by Demographic Group\n"
        "(MIDUS-like data; violin = density, bars = quartiles)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    return fig


# ─── 8. f vs p (dual continua) ────────────────────────────────────────────────

def plot_f_vs_p(df: pd.DataFrame) -> plt.Figure | None:
    """
    Plot f score against negative affect (p-factor proxy).
    Under the dual continua model, these should be correlated but distinct —
    i.e., moderate negative r, with non-trivial scatter (not a perfect line).
    """
    if "B1SNEGAF" not in df.columns or "f_score" not in df.columns:
        return None

    df_plot = df[["f_score", "B1SNEGAF"]].dropna()
    r, p_val = stats.pearsonr(df_plot["f_score"], df_plot["B1SNEGAF"])

    # Hexbin + marginals
    fig = plt.figure(figsize=(8, 7))
    gs  = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1:, :-1])
    ax_top   = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    hb = ax_main.hexbin(df_plot["f_score"], df_plot["B1SNEGAF"],
                        gridsize=40, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax_right, shrink=0.7, label="Count")

    # Regression line
    xs = np.linspace(df_plot["f_score"].min(), df_plot["f_score"].max(), 100)
    slope, intercept, *_ = stats.linregress(df_plot["f_score"], df_plot["B1SNEGAF"])
    ax_main.plot(xs, intercept + slope * xs, color="#3B7DD8", lw=2)

    ax_main.set_xlabel("f score (general flourishing)", fontsize=10)
    ax_main.set_ylabel("Negative Affect — B1SNEGAF (p-factor proxy)", fontsize=10)

    ax_top.hist(df_plot["f_score"], bins=50, color="#3B7DD8", alpha=0.7)
    ax_top.axis("off")
    ax_right.hist(df_plot["B1SNEGAF"], bins=30, orientation="horizontal",
                  color="#E8622A", alpha=0.7)
    ax_right.axis("off")

    ax_main.text(0.97, 0.95,
                 f"r = {r:.3f}  (p = {p_val:.2e})\n"
                 f"Shared variance: {r**2*100:.1f}%\n"
                 f"Independent variance: {(1-r**2)*100:.1f}%\n\n"
                 f"Dual continua interpretation:\n"
                 f"f and p partially overlap but are\n"
                 f"meaningfully distinct dimensions.",
                 transform=ax_main.transAxes, ha="right", va="top",
                 fontsize=8.5, color="#333",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="#ccc"))

    fig.suptitle(
        "f (Flourishing) vs p (Psychopathology proxy)\n"
        "Testing the Dual Continua Model — Individual Level",
        fontsize=12, y=1.0,
    )
    return fig


# ─── 9. Ecological vs individual comparison ───────────────────────────────────

def compare_ecological_individual(
    country_vr: float,
    individual_vr: float,
) -> plt.Figure:
    """
    Bar chart comparing PC1 variance explained at country vs. individual level.
    Illustrates Simpson's paradox / ecological inflation of correlations.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    levels  = ["Country-level\n(WHR, n≈137)", "Individual-level\n(MIDUS, n≈5000)"]
    values  = [country_vr * 100, individual_vr * 100]
    colors  = ["#E8622A", "#3B7DD8"]

    bars = ax.bar(levels, values, color=colors, alpha=0.82, width=0.45)
    ax.axhline(values[1], color="#3B7DD8", ls=":", lw=1.2)
    ax.set_ylabel("Variance explained by PC1 (f proxy) %")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Ecological vs Individual-Level f\n"
        "Country averages inflate the apparent strength of f",
        fontsize=11,
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    diff = values[0] - values[1]
    ax.annotate(
        f"Ecological inflation:\n+{diff:.1f}pp\n(aggregation removes\nwithin-group variance)",
        xy=(0, values[0]/2), xytext=(0.55, values[0]*0.6),
        fontsize=8.5, color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.1),
    )
    plt.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")


def main():
    print("\n" + "═"*65)
    print("  Individual-Level f-Factor Analysis")
    print(f"  Mode: {MODE}")
    print("═"*65)

    # ── Load / generate data ───────────────────────────────────────────────────
    if MODE == "synthetic":
        df_raw = generate_synthetic_midus(n=5000)
        label  = "Synthetic MIDUS-like"
    elif MODE in ("midus", "compare"):
        df_raw = load_midus(MIDUS_DATA_PATH)
        label  = "MIDUS 2 (ICPSR 4652)"
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")

    df_clean, X, present_cols = preprocess(df_raw)

    # ── Correlation / manifold ─────────────────────────────────────────────────
    section("Positive Manifold — Individual Level")
    corr = report_manifold(df_clean, present_cols, label=label)

    fig_corr = plot_corr_heatmap(df_clean, present_cols,
                                  title_suffix=f" · {label}")
    fig_corr.savefig(OUTPUT_DIR / "indiv_01_correlation_matrix.png")
    print(f"  → {OUTPUT_DIR}/indiv_01_correlation_matrix.png")

    # ── PCA ───────────────────────────────────────────────────────────────────
    section("PCA — Extracting f at Individual Level")
    df_scored, fig_scree, fig_load, var_ratio = run_pca_and_plot(
        X, present_cols, df_clean, suffix=label
    )
    fig_scree.savefig(OUTPUT_DIR / "indiv_02_scree.png")
    fig_load.savefig(OUTPUT_DIR / "indiv_03_loadings.png")
    print(f"  → {OUTPUT_DIR}/indiv_02_scree.png")
    print(f"  → {OUTPUT_DIR}/indiv_03_loadings.png")

    # ── Bifactor CFA ──────────────────────────────────────────────────────────
    section("Bifactor CFA (omega_h)")
    cfa_results = run_bifactor_cfa(df_clean, present_cols)

    # ── f distribution ────────────────────────────────────────────────────────
    section("f Score Distribution")
    fig_fdist, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df_scored["f_score"], bins=60, color="#3B7DD8", alpha=0.7,
            edgecolor="white", density=True, label="Empirical")
    df_scored["f_score"].plot.kde(ax=ax, color="#E8622A", lw=2.2, label="KDE")
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, stats.norm.pdf(xs), color="black", lw=1.2, ls="--",
            alpha=0.6, label="N(0,1) reference")
    stat, p = stats.shapiro(df_scored["f_score"].sample(min(5000, len(df_scored))))
    ax.text(0.98, 0.93, f"Shapiro-Wilk: W={stat:.3f}, p={p:.4f}",
            transform=ax.transAxes, ha="right", fontsize=8.5)
    ax.set_xlabel("f score (z-standardized)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Individual f Scores — {label}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_fdist.savefig(OUTPUT_DIR / "indiv_04_f_distribution.png")
    print(f"  → {OUTPUT_DIR}/indiv_04_f_distribution.png")

    # ── Demographic profiles ──────────────────────────────────────────────────
    section("Demographic Subgroup f Profiles")
    fig_demo = plot_demographic_f(df_scored)
    if fig_demo:
        fig_demo.savefig(OUTPUT_DIR / "indiv_05_demographic_f.png")
        print(f"  → {OUTPUT_DIR}/indiv_05_demographic_f.png")

    # ── f vs p ────────────────────────────────────────────────────────────────
    section("f vs p: Dual Continua Test")
    fig_fp = plot_f_vs_p(df_scored)
    if fig_fp:
        fig_fp.savefig(OUTPUT_DIR / "indiv_06_f_vs_p.png")
        print(f"  → {OUTPUT_DIR}/indiv_06_f_vs_p.png")

    # ── Ecological comparison (if MODE == compare) ────────────────────────────
    if MODE == "compare":
        section("Ecological vs Individual-Level Comparison")
        ecol_path = Path("wellbeing_data/wellbeing_merged.csv")
        if ecol_path.exists():
            df_eco = pd.read_csv(ecol_path)
            eco_cols = [c for c in [
                "P_positive_emotion","E_engagement_autonomy","R_relationships",
                "M_meaning_life_sat","A_accomplishment_gdp","H_health"
            ] if c in df_eco.columns]
            if eco_cols:
                X_eco = StandardScaler().fit_transform(
                    df_eco[eco_cols].dropna().values)
                eco_vr = PCA().fit(X_eco).explained_variance_ratio_
                fig_comp = compare_ecological_individual(
                    eco_vr[0], var_ratio[0])
                fig_comp.savefig(OUTPUT_DIR / "indiv_07_ecological_comparison.png")
                print(f"  → {OUTPUT_DIR}/indiv_07_ecological_comparison.png")
        else:
            print("  Run scripts 01+02 first to generate wellbeing_merged.csv")

    # ── Save scored data ──────────────────────────────────────────────────────
    out_path = Path("wellbeing_data/midus_scored.csv")
    out_path.parent.mkdir(exist_ok=True)
    df_scored.to_csv(out_path, index=False)
    print(f"\n  ✓ Scored individual data saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  SUMMARY")
    print("═"*65)
    print(f"  Data:               {label}")
    print(f"  N (complete cases): {len(df_scored):,}")
    print(f"  PC1 var explained:  {var_ratio[0]*100:.1f}%")
    corr_off = corr.values[np.tril_indices_from(corr.values, k=-1)]
    print(f"  Mean inter-subscale r: {corr_off.mean():.3f}")
    print(f"  All correlations positive: {'✓' if (corr_off > 0).all() else '✗'}")
    if cfa_results:
        print(f"  omega_h (bifactor):    {cfa_results['omega_h']:.3f}")
        print(f"  ECV:                   {cfa_results['ecv']:.3f}")
    print(f"\n  To use real MIDUS data:")
    print(f"    1. Register at https://www.icpsr.umich.edu")
    print(f"    2. Download ICPSR 4652 ('Delimited' format)")
    print(f"    3. Place .tsv in: {MIDUS_DATA_PATH}")
    print(f"    4. Set MODE = 'midus' at top of script")
    print()


if __name__ == "__main__":
    main()
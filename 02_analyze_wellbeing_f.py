"""
02_analyze_wellbeing_f.py
─────────────────────────────────────────────────────────────────────────────
Analyzes the merged wellbeing dataset to empirically extract the general
factor of flourishing (f) and characterize the positive manifold.

What this script does, in order
────────────────────────────────
 1. Load wellbeing_data/wellbeing_merged.csv (produced by script 01)
 2. Select the 6 PERMA+H columns; drop rows with any missing values
 3. Standardize (z-score) all dimensions
 4. Compute and visualize the correlation matrix (positive manifold test)
 5. Run PCA:
    - Scree plot (eigenvalue decomposition)
    - Factor loadings on PC1 = empirical f
    - Cumulative explained variance
 6. Score each country on f (PC1 projection)
 7. Plot top-20 / bottom-20 countries by f
 8. Scatter matrix of all PERMA+H pairs
 9. Stability proxy: examine coupling asymmetry across dimensions

Methodological notes
─────────────────────
• PCA here is exploratory. A proper bifactor CFA (as in the paper) would use
  a confirmatory model (e.g., lavaan in R or semopy in Python). PCA gives an
  unrotated, consistent estimate of PC1 that serves as the f proxy.

• Standard errors on loadings are bootstrapped (n=1000, 95% CI).

• The data are country-level averages, not individual responses. The positive
  manifold observed is an *ecological correlation*. Within-country individual-
  level data (e.g., Gallup microdata) would be needed for proper replication
  of the paper's claims about individual f.

• omega_h estimation requires a bifactor CFA; we report the variance explained
  by PC1 as a direct analogue.

Requires
────────
  pip install pandas numpy matplotlib seaborn scikit-learn scipy
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_PATH  = Path("wellbeing_data/wellbeing_merged.csv")
OUTPUT_DIR = Path("wellbeing_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

PERMA_COLS = [
    "P_positive_emotion",
    "E_engagement_autonomy",
    "R_relationships",
    "M_meaning_life_sat",
    "A_accomplishment_gdp",
    "H_health",
]

PERMA_LABELS = {
    "P_positive_emotion":    "P  Positive emotion\n(Positive affect)",
    "E_engagement_autonomy": "E  Engagement\n(Autonomy / freedom)",
    "R_relationships":       "R  Relationships\n(Social support)",
    "M_meaning_life_sat":    "M  Meaning\n(Cantril ladder)",
    "A_accomplishment_gdp":  "A  Accomplishment\n(Log GDP/capita)",
    "H_health":              "H  Health\n(Healthy life expect.)",
}

SHORT_LABELS = {
    "P_positive_emotion":    "P: Positive\nEmotion",
    "E_engagement_autonomy": "E: Engagement\n(Autonomy)",
    "R_relationships":       "R: Relationships",
    "M_meaning_life_sat":    "M: Meaning\n(Life Sat.)",
    "A_accomplishment_gdp":  "A: Accomplishment\n(GDP)",
    "H_health":              "H: Health",
}

PALETTE = {
    "P_positive_emotion":    "#E8622A",
    "E_engagement_autonomy": "#3B7DD8",
    "R_relationships":       "#2EAA6A",
    "M_meaning_life_sat":    "#9B59B6",
    "A_accomplishment_gdp":  "#E67E22",
    "H_health":              "#1ABC9C",
}

BOOTSTRAP_N = 1000
np.random.seed(42)

# ─── Style ────────────────────────────────────────────────────────────────────

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

# ─── 1. Load and clean ────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Returns (df_full, df_clean, X_scaled)
    df_clean: rows with all 6 PERMA+H values present
    X_scaled: standardized PERMA+H matrix (N × 6)
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {len(df)} countries, {len(df.columns)} columns")

    present = [c for c in PERMA_COLS if c in df.columns]
    missing = [c for c in PERMA_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing}")

    df_clean = df[["country"] + present].dropna(subset=present).copy()
    print(f"Complete PERMA+H cases: {len(df_clean)} / {len(df)}")

    scaler  = StandardScaler()
    X       = df_clean[present].values
    X_scaled = scaler.fit_transform(X)

    # Store original means/stds for reporting
    df_clean["_mean_raw"] = X.mean(axis=1)

    return df, df_clean, X_scaled, present


# ─── 2. Descriptive statistics ────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, cols: list):
    print("\n" + "═" * 65)
    print("  DESCRIPTIVE STATISTICS (raw scale)")
    print("═" * 65)
    desc = df[cols].describe().T
    desc["skewness"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurtosis()
    desc = desc.round(3)
    desc.index = [PERMA_LABELS.get(c, c) for c in desc.index]
    print(desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max",
                "skewness", "kurtosis"]].to_string())


# ─── 3. Correlation matrix ────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, cols: list) -> plt.Figure:
    """
    Heatmap of Pearson correlations among PERMA+H dimensions.
    Every off-diagonal cell should be positive — the 'positive manifold.'
    """
    corr = df[cols].corr()
    labels = [SHORT_LABELS.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # show lower triangle + diagonal

    sns.heatmap(
        corr,
        mask=~(~mask),        # show full matrix
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-0.2, vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title(
        "Positive Manifold: Correlations Among PERMA+H Dimensions\n"
        "(Country-level averages, WHR 2024)",
        pad=14,
    )
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    # Annotation: average off-diagonal correlation
    off_diag = corr.values[np.tril_indices_from(corr.values, k=-1)]
    ax.text(
        0.01, -0.10,
        f"Mean off-diagonal r = {off_diag.mean():.3f}  "
        f"(min = {off_diag.min():.3f}, max = {off_diag.max():.3f})",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444",
    )
    plt.tight_layout()
    return fig


# ─── 4. PCA: scree + loadings ─────────────────────────────────────────────────

def run_pca(X: np.ndarray, cols: list) -> tuple:
    pca = PCA()
    scores = pca.fit_transform(X)
    loadings = pca.components_.T        # shape (n_features, n_components)
    eigenvalues = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_

    # ── Sign fix ──────────────────────────────────────────────────────────────
    # PCA eigenvectors have arbitrary sign. Fix it once here so that higher
    # PC1 score always means MORE flourishing.
    # Anchor: M_meaning_life_sat (life satisfaction / Cantril ladder) is the
    # most direct wellbeing measure; if its loading on PC1 is negative the
    # whole axis is pointing the wrong way → flip both scores and loadings.
    anchor_col = "M_meaning_life_sat"
    if anchor_col in cols:
        anchor_idx = cols.index(anchor_col)
    else:
        # Fallback: use the dimension with the largest absolute loading
        anchor_idx = int(np.argmax(np.abs(loadings[:, 0])))

    if loadings[anchor_idx, 0] < 0:
        loadings[:, 0] = -loadings[:, 0]
        scores[:, 0]   = -scores[:, 0]

    return pca, scores, loadings, eigenvalues, var_ratio


def bootstrap_loadings(X: np.ndarray, n_boot: int = BOOTSTRAP_N) -> np.ndarray:
    """
    Bootstrap PC1 loadings to get 95% CIs.
    Sign is fixed to match the original loading direction.
    """
    n = X.shape[0]
    boot_loadings = np.zeros((n_boot, X.shape[1]))
    ref_loadings = PCA(n_components=1).fit(X).components_[0]

    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        pca_b = PCA(n_components=1)
        pca_b.fit(X[idx])
        l = pca_b.components_[0]
        # Fix sign: align with reference
        if np.dot(l, ref_loadings) < 0:
            l = -l
        boot_loadings[i] = l

    return boot_loadings


def plot_scree(eigenvalues: np.ndarray, var_ratio: np.ndarray) -> plt.Figure:
    n = len(eigenvalues)
    cumvar = np.cumsum(var_ratio) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Scree plot
    components = np.arange(1, n + 1)
    ax1.plot(components, eigenvalues, "o-", color="#3B7DD8", lw=2.2,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax1.axhline(y=1.0, color="#E8622A", ls="--", lw=1.4,
                label="Kaiser criterion (λ = 1)")
    # Shade the dominant first component
    ax1.bar(1, eigenvalues[0], alpha=0.15, color="#3B7DD8", width=0.6)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title(
        "Scree Plot\n"
        "A dominant first eigenvalue = positive manifold",
    )
    ax1.set_xticks(components)
    ax1.legend(fontsize=8.5)

    # Annotate PC1 variance
    ax1.annotate(
        f"PC1 explains\n{var_ratio[0]*100:.1f}% of variance\n(≈ f)",
        xy=(1, eigenvalues[0]),
        xytext=(2.2, eigenvalues[0] * 0.88),
        fontsize=8.5,
        color="#3B7DD8",
        arrowprops=dict(arrowstyle="->", color="#3B7DD8", lw=1.2),
    )

    # Cumulative variance
    ax2.step(components, cumvar, where="post", color="#2EAA6A", lw=2.2)
    ax2.fill_between(components, cumvar, step="post", alpha=0.12, color="#2EAA6A")
    ax2.axhline(80, color="gray", ls=":", lw=1.2, label="80% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(components)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8.5)

    for pct, comp in zip(cumvar, components):
        ax2.text(comp, pct + 1.5, f"{pct:.0f}%", ha="center",
                 fontsize=7.5, color="#2EAA6A")

    fig.suptitle(
        "PCA Variance Structure — Evidence for a General Factor (f)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig


def plot_factor_loadings(
    loadings: np.ndarray,
    cols: list,
    boot_loadings: np.ndarray,
) -> plt.Figure:
    """
    Horizontal bar chart of PC1 loadings with 95% bootstrap CIs.
    Larger loading → dimension more strongly reflects f.
    """
    labels = [SHORT_LABELS.get(c, c) for c in cols]
    colors = [PALETTE.get(c, "#888") for c in cols]
    l1 = loadings[:, 0]

    # Sign is already fixed in run_pca, but guard here too for safety.
    # Use the anchor dimension (M_meaning_life_sat) if available.
    anchor_col = "M_meaning_life_sat"
    if anchor_col in cols:
        anchor_idx = cols.index(anchor_col)
        if l1[anchor_idx] < 0:
            l1 = -l1
            boot_loadings = -boot_loadings
    elif l1.mean() < 0:        # last-resort fallback only
        l1 = -l1
        boot_loadings = -boot_loadings

    ci_lo = np.percentile(boot_loadings, 2.5, axis=0)
    ci_hi = np.percentile(boot_loadings, 97.5, axis=0)
    # Fix sign on CI too
    if boot_loadings.mean() < 0:
        ci_lo, ci_hi = -ci_hi, -ci_lo

    order = np.argsort(l1)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(cols))

    bars = ax.barh(
        y_pos,
        l1[order],
        xerr=[l1[order] - ci_lo[order], ci_hi[order] - l1[order]],
        color=[colors[i] for i in order],
        alpha=0.85,
        height=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": "#555"},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in order], fontsize=9.5)
    ax.set_xlabel("PC1 Loading (proxy for f loading)", fontsize=10)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-0.1, 0.75)
    ax.set_title(
        "Factor Loadings on PC1 (= f)\n"
        "with 95% Bootstrap Confidence Intervals (n = 1,000)",
        fontsize=12,
    )

    # Value labels
    for bar, val, lo, hi in zip(bars, l1[order], ci_lo[order], ci_hi[order]):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8.5,
            color="#222",
        )

    # Annotation
    ax.text(
        0.98, 0.04,
        "Higher loading → dimension more\nclosely tracks the general factor",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="#555",
        style="italic",
    )
    plt.tight_layout()
    return fig


# ─── 5. Country f-scores ──────────────────────────────────────────────────────

def compute_f_scores(df: pd.DataFrame, scores: np.ndarray, cols: list) -> pd.DataFrame:
    # Standardize PC1 scores to mean=0, std=1 for interpretability.
    # Sign is already fixed in run_pca (anchored to M_meaning_life_sat).
    f_raw = scores[:, 0]
    f_z = (f_raw - f_raw.mean()) / f_raw.std()
    df = df.copy()
    df["f_score"]   = f_raw
    df["f_z"]       = f_z
    return df.sort_values("f_z", ascending=False).reset_index(drop=True)


def plot_country_f_scores(df: pd.DataFrame, n: int = 20) -> plt.Figure:
    top    = df.head(n).copy()
    bottom = df.tail(n).copy()

    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(14, 7))

    def _barplot(ax, data, title, color):
        data = data.copy().sort_values("f_z", ascending=(color == "#E8622A"))
        bars = ax.barh(
            data["country"],
            data["f_z"],
            color=color,
            alpha=0.80,
            height=0.7,
        )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("f score (z-standardized)")
        for bar, val in zip(bars, data["f_z"]):
            offset = 0.04 if val >= 0 else -0.04
            ha = "left" if val >= 0 else "right"
            ax.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha=ha,
                fontsize=7.5,
            )

    _barplot(ax_top, top,    f"Top {n} Countries by f", "#3B7DD8")
    _barplot(ax_bot, bottom, f"Bottom {n} Countries by f", "#E8622A")

    fig.suptitle(
        "Country f-Scores: General Factor of Flourishing\n"
        "(PC1 projection, z-standardized, WHR 2024 data)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    return fig


# ─── 6. Scatter matrix ────────────────────────────────────────────────────────

def plot_scatter_matrix(df: pd.DataFrame, cols: list) -> plt.Figure:
    """
    Lower-triangle scatter matrix with regression lines and Pearson r.
    Diagonal shows KDE of each dimension.
    """
    n = len(cols)
    labels = [SHORT_LABELS.get(c, c) for c in cols]
    colors_list = [PALETTE.get(c, "#888") for c in cols]

    fig, axes = plt.subplots(n, n, figsize=(13, 11))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.tick_params(labelbottom=False, labelleft=False,
                           bottom=False, left=False)
            if i == j:
                # KDE on diagonal
                data = df[cols[i]].dropna()
                data.plot.kde(ax=ax, color=colors_list[i], lw=2)
                ax.set_xlim(data.min() - data.std(), data.max() + data.std())
                ax.set_ylabel("")
            elif i > j:
                # Scatter below diagonal
                xy = df[[cols[j], cols[i]]].dropna()
                ax.scatter(
                    xy[cols[j]], xy[cols[i]],
                    alpha=0.4, s=14,
                    color=colors_list[i], edgecolors="none",
                )
                # Regression line
                slope, intercept, r, p, _ = stats.linregress(
                    xy[cols[j]], xy[cols[i]]
                )
                xs = np.linspace(xy[cols[j]].min(), xy[cols[j]].max(), 100)
                ax.plot(xs, intercept + slope * xs, color="#333", lw=1.2, alpha=0.8)
                # r annotation
                col = "#2EAA6A" if r > 0 else "#E8622A"
                ax.text(
                    0.07, 0.88, f"r={r:.2f}",
                    transform=ax.transAxes, fontsize=7.5, color=col,
                    fontweight="bold",
                )
            else:
                ax.set_visible(False)

            # Labels on edges
            if j == 0 and i > 0:
                ax.tick_params(labelleft=True, left=True)
                ax.set_ylabel(labels[i], fontsize=7.5, labelpad=3)
            if i == n - 1 and j < n - 1:
                ax.tick_params(labelbottom=True, bottom=True)
                ax.set_xlabel(labels[j], fontsize=7.5, labelpad=3)

    fig.suptitle(
        "Scatter Matrix: All Pairwise PERMA+H Relationships\n"
        "(lower triangle; r = Pearson correlation)",
        fontsize=13,
        y=1.01,
    )
    return fig


# ─── 7. Mutualism coupling heatmap ────────────────────────────────────────────

def plot_coupling_proxy(df: pd.DataFrame, cols: list) -> plt.Figure:
    """
    Partial correlation matrix as a proxy for pairwise mutualistic coupling.
    In the mutualism model, M_kl (coupling strength) determines how much each
    pair of domains mutually reinforces one another. Partial correlations
    control for all other dimensions — a closer analogue to direct coupling
    than zero-order correlations.
    """
    from numpy.linalg import inv
    X = df[cols].dropna().values
    X = StandardScaler().fit_transform(X)

    # Precision matrix = inverse of correlation matrix
    C    = np.corrcoef(X.T)
    P    = inv(C)

    # Partial correlations: r_ij|rest = -P_ij / sqrt(P_ii * P_jj)
    D    = np.sqrt(np.diag(P))
    pc   = -P / np.outer(D, D)
    np.fill_diagonal(pc, 1.0)

    labels = [SHORT_LABELS.get(c, c) for c in cols]
    pc_df  = pd.DataFrame(pc, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        pc_df,
        annot=True, fmt=".2f",
        cmap="coolwarm",
        vmin=-0.6, vmax=0.6,
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Partial r"},
    )
    ax.set_title(
        "Partial Correlation Matrix — Proxy for Mutualistic Coupling (M_kl)\n"
        "Positive values suggest direct mutual facilitation between domains",
        fontsize=11,
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


# ─── 8. f score distribution ──────────────────────────────────────────────────

def plot_f_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.hist(df["f_z"], bins=30, color="#3B7DD8", alpha=0.7, edgecolor="white",
            density=True, label="Empirical distribution")
    # KDE overlay
    df["f_z"].plot.kde(ax=ax, color="#E8622A", lw=2.2, label="KDE")

    # Normal reference
    xs = np.linspace(df["f_z"].min() - 0.5, df["f_z"].max() + 0.5, 200)
    ax.plot(xs, stats.norm.pdf(xs), color="black", lw=1.2, ls="--",
            label="N(0,1) reference", alpha=0.6)

    ax.axvline(0, color="gray", lw=0.9, ls=":")
    ax.set_xlabel("f score (z-standardized)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Country-Level f Scores", fontsize=12)
    ax.legend(fontsize=9)

    # Shapiro-Wilk normality test
    stat, p = stats.shapiro(df["f_z"])
    ax.text(
        0.98, 0.93,
        f"Shapiro-Wilk: W={stat:.3f}, p={p:.3f}",
        transform=ax.transAxes, ha="right", fontsize=8, color="#444",
    )
    plt.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def main():
    print("\n" + "═" * 65)
    print("  Wellbeing f-Factor Analysis")
    print("  Empirical extraction of the general factor of flourishing")
    print("═" * 65)

    # ── Load ──────────────────────────────────────────────────────────────────
    df_full, df, X, present_cols = load_data()

    # ── Descriptive stats ─────────────────────────────────────────────────────
    descriptive_stats(df, present_cols)

    # ── Correlations ──────────────────────────────────────────────────────────
    print_section("Correlation matrix (positive manifold test)")
    corr = df[present_cols].corr()
    off_diag = corr.values[np.tril_indices_from(corr.values, k=-1)]
    n_positive = (off_diag > 0).sum()
    print(f"  Off-diagonal correlations: {len(off_diag)} pairs")
    print(f"  Positive:  {n_positive} ({100*n_positive/len(off_diag):.0f}%)")
    print(f"  Mean r:    {off_diag.mean():.3f}")
    print(f"  Min r:     {off_diag.min():.3f}")
    print(f"  Max r:     {off_diag.max():.3f}")
    if n_positive == len(off_diag):
        print("  ✓ Full positive manifold confirmed.")
    else:
        print("  ⚠ Not all correlations positive — check for data issues.")

    fig_corr = plot_correlation_matrix(df, present_cols)
    fig_corr.savefig(OUTPUT_DIR / "01_correlation_matrix.png")
    print(f"  → Saved: {OUTPUT_DIR}/01_correlation_matrix.png")

    # ── PCA ───────────────────────────────────────────────────────────────────
    print_section("PCA: extracting f as PC1")
    pca, scores, loadings, eigenvalues, var_ratio = run_pca(X, present_cols)

    print(f"  Eigenvalues:  {eigenvalues.round(3)}")
    print(f"  Var ratio:    {(var_ratio*100).round(1)}")
    print(f"\n  PC1 (f) accounts for {var_ratio[0]*100:.1f}% of total variance")
    print(f"  PC1 eigenvalue: {eigenvalues[0]:.3f}  "
          f"(Kaiser criterion > 1: {'✓' if eigenvalues[0] > 1 else '✗'})")

    # Variance ratio analogue to omega_h
    # omega_h ≈ (sum of PC1 loadings)^2 / total variance = var_ratio[0]
    print(f"\n  omega_h analogue (PC1 var / total var): {var_ratio[0]:.3f}")
    print(f"  Paper's empirical range: 0.71 – 0.86")
    print(f"  Note: country-level aggregate data tends to inflate this.")

    fig_scree = plot_scree(eigenvalues, var_ratio)
    fig_scree.savefig(OUTPUT_DIR / "02_scree_plot.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/02_scree_plot.png")

    # Bootstrap loadings
    print(f"\n  Bootstrapping PC1 loadings (n={BOOTSTRAP_N}) …", end=" ", flush=True)
    boot_L = bootstrap_loadings(X, BOOTSTRAP_N)
    print("done.")
    l1 = loadings[:, 0]
    # Sign already fixed in run_pca; guard here too for the printout
    anchor_col = "M_meaning_life_sat"
    if anchor_col in present_cols:
        anchor_idx = present_cols.index(anchor_col)
        if l1[anchor_idx] < 0:
            l1 = -l1
    ci_lo = np.percentile(boot_L, 2.5, axis=0)
    ci_hi = np.percentile(boot_L, 97.5, axis=0)
    print(f"\n  PC1 loadings (with 95% CI):")
    for col, loading, lo, hi in zip(present_cols, l1,
                                    np.abs(ci_lo), np.abs(ci_hi)):
        print(f"    {SHORT_LABELS.get(col, col):<30} {loading:.3f}  "
              f"[{min(lo, hi):.3f}, {max(lo, hi):.3f}]")

    fig_load = plot_factor_loadings(loadings, present_cols, boot_L)
    fig_load.savefig(OUTPUT_DIR / "03_factor_loadings.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/03_factor_loadings.png")

    # ── Country f-scores ──────────────────────────────────────────────────────
    print_section("Country f-scores")
    df_scored = compute_f_scores(df, scores, present_cols)

    print(f"  Top 10 countries by f:")
    for _, row in df_scored.head(10).iterrows():
        print(f"    {row['country']:<35}  f = {row['f_z']:+.3f}")
    print(f"\n  Bottom 10 countries by f:")
    for _, row in df_scored.tail(10).iterrows():
        print(f"    {row['country']:<35}  f = {row['f_z']:+.3f}")

    fig_f = plot_country_f_scores(df_scored, n=20)
    fig_f.savefig(OUTPUT_DIR / "04_country_f_scores.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/04_country_f_scores.png")

    fig_dist = plot_f_distribution(df_scored)
    fig_dist.savefig(OUTPUT_DIR / "05_f_distribution.png")
    print(f"  → Saved: {OUTPUT_DIR}/05_f_distribution.png")

    # ── Scatter matrix ────────────────────────────────────────────────────────
    print_section("Scatter matrix")
    fig_scatter = plot_scatter_matrix(df, present_cols)
    fig_scatter.savefig(OUTPUT_DIR / "06_scatter_matrix.png")
    print(f"  → Saved: {OUTPUT_DIR}/06_scatter_matrix.png")

    # ── Coupling proxy ────────────────────────────────────────────────────────
    print_section("Mutualistic coupling proxy (partial correlations)")
    fig_couple = plot_coupling_proxy(df, present_cols)
    fig_couple.savefig(OUTPUT_DIR / "07_coupling_proxy.png")
    print(f"  → Saved: {OUTPUT_DIR}/07_coupling_proxy.png")

    # ── Save scored dataset ───────────────────────────────────────────────────
    scored_path = Path("wellbeing_data/wellbeing_scored.csv")
    df_scored.to_csv(scored_path, index=False)
    print(f"\n  ✓ Scored dataset saved → {scored_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ANALYSIS COMPLETE")
    print("═" * 65)
    print(f"  Countries analyzed:          {len(df_scored)}")
    print(f"  f variance explained (PC1):  {var_ratio[0]*100:.1f}%")
    print(f"  Positive manifold:           "
          f"{'✓ all {len(off_diag)} pairs positive' if (off_diag > 0).all() else '⚠ some negative'}")
    print(f"  Figures saved to:            {OUTPUT_DIR}/")
    print(f"  Scored data saved to:        {scored_path}")
    print()


if __name__ == "__main__":
    main()
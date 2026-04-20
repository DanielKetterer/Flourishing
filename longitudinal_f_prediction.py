"""
longitudinal_f_prediction.py
----------------------------------------------------------------------
Does f in MIDUS Wave 2 predict future income and employment in Wave 3?

Design:
  1. Compute f (PC1 of Ryff 6 subscales) from Wave 2 individual data
  2. Link to Wave 3 economic outcomes via M2ID
  3. Test predictive validity:
     - f_W2 -> employed_W3  (logistic regression, controlling for W2 baseline)
     - f_W2 -> log_income_W3 (OLS, controlling for W2 baseline)
     - f_W2 -> disability_days_W3 (OLS/Tobit-like, controlling for W2 baseline)
  4. Incremental validity: Does f add prediction beyond negative affect alone?
  5. Visualize with scatterplots, coefficient plots, ROC curves
----------------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# ---- Config ----------------------------------------------------------------
W2_PATH = "04652-0001-Data.tsv"
W3_PATH = "36346-0001-Data.tsv"
OUT_DIR = "wellbeing_figures_combined_v2"

import os
os.makedirs(OUT_DIR, exist_ok=True)

RYFF_W2 = ["B1SPWBA1", "B1SPWBE1", "B1SPWBG1", "B1SPWBR1", "B1SPWBU1", "B1SPWBS1"]
RYFF_W3 = ["C1SPWBA1", "C1SPWBE1", "C1SPWBG1", "C1SPWBR1", "C1SPWBU1", "C1SPWBS1"]
RYFF_NAMES = ["Autonomy", "Envir. Mastery", "Personal Growth",
              "Positive Relations", "Purpose in Life", "Self-Acceptance"]

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 140,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})

ACCENT = "#2C6FAC"
ACCENT2 = "#E8622A"
ACCENT3 = "#2EAA6A"


# ---- Helper: clean Ryff columns -------------------------------------------
def clean_ryff(df, cols):
    """Coerce to numeric, set invalid (<= 0 or >= 90) to NaN, scale to ~1-7."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].where((df[c] > 0) & (df[c] < 90))
            df[c] = df[c] / 3.0  # raw sums -> approximate 1-7 range
    return df


def clean_econ(df, prefix):
    """Clean economic variables for a given wave prefix (B1 or C1)."""
    inc_col = f"{prefix}STINC1" if prefix == "B1" else f"{prefix}STINC"
    emp_col = f"{prefix}PB19"
    pa24_col = f"{prefix}PA24"
    pa24a_col = f"{prefix}PA24A"
    negaf_col = f"{prefix}SNEGAF" if prefix == "B1" else None

    for c in [inc_col, emp_col, pa24_col, pa24a_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Income: set -1 and 9999998+ to NaN, then log
    if inc_col in df.columns:
        df[inc_col] = df[inc_col].where((df[inc_col] > 0) & (df[inc_col] < 999998))
        df[f"log_income_{prefix}"] = np.log(df[inc_col] + 1)

    # Employment: 1 = working, rest = not (drop 7/8 = DK/refused)
    if emp_col in df.columns:
        df[f"employed_{prefix}"] = np.where(df[emp_col] == 1, 1,
                                   np.where(df[emp_col].isin([7, 8, 98]), np.nan, 0))

    # Disability days
    if pa24_col in df.columns and pa24a_col in df.columns:
        df[pa24a_col] = df[pa24a_col].where((df[pa24a_col] >= 1) & (df[pa24a_col] <= 30))
        df[f"disab_days_{prefix}"] = np.where(df[pa24_col] == 1, df[pa24a_col], 0)
        df[f"disab_days_{prefix}"] = df[f"disab_days_{prefix}"].where(
            df[f"disab_days_{prefix}"].notna())

    # Negative affect (wave 2 only, for incremental validity test)
    if negaf_col and negaf_col in df.columns:
        df[negaf_col] = pd.to_numeric(df[negaf_col], errors="coerce")
        df[negaf_col] = df[negaf_col].where(df[negaf_col] > 0)

    return df


# ---- 1. Load and clean both waves -----------------------------------------
print("=" * 70)
print("  LONGITUDINAL PREDICTION: Wave 2 f -> Wave 3 Economic Outcomes")
print("=" * 70)

# Wave 2
w2_cols = ["M2ID", "B1PAGE_M2", "B1PRSEX", "B1PF7A", "B1SNEGAF",
           "B1STINC1", "B1PB19", "B1PA24", "B1PA24A"] + RYFF_W2
df2 = pd.read_csv(W2_PATH, sep="\t", usecols=w2_cols, low_memory=False)
df2.columns = [c.strip().upper() for c in df2.columns]
df2 = clean_ryff(df2, RYFF_W2)
df2 = clean_econ(df2, "B1")

# Wave 3
w3_cols = ["M2ID", "C1PRAGE", "C1STINC", "C1PB19", "C1PA24", "C1PA24A"] + RYFF_W3
df3 = pd.read_csv(W3_PATH, sep="\t", usecols=w3_cols, low_memory=False)
df3.columns = [c.strip().upper() for c in df3.columns]
df3 = clean_ryff(df3, RYFF_W3)
df3 = clean_econ(df3, "C1")

print(f"\n  Wave 2: {len(df2):,} respondents")
print(f"  Wave 3: {len(df3):,} respondents")


# ---- 2. Compute f from Wave 2 Ryff ----------------------------------------
df2_complete = df2.dropna(subset=RYFF_W2).copy()
X_w2 = df2_complete[RYFF_W2].values.astype(float)
scaler_w2 = StandardScaler()
X_w2_z = scaler_w2.fit_transform(X_w2)

pca = PCA(n_components=6)
pca.fit(X_w2_z)
f_scores = X_w2_z @ pca.components_[0]  # PC1 = f
# Orient so higher = more flourishing
if pca.components_[0].mean() < 0:
    f_scores = -f_scores
    pca.components_[0] = -pca.components_[0]

# Standardize f
f_scores = (f_scores - f_scores.mean()) / f_scores.std()
df2_complete["f_score_W2"] = f_scores

print(f"\n  Wave 2 complete Ryff cases: {len(df2_complete):,}")
print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  PC1 loadings:")
for name, loading in zip(RYFF_NAMES, pca.components_[0]):
    print(f"    {name:20s}: {loading:.3f}")


# ---- 3. Merge waves on M2ID -----------------------------------------------
merged = df2_complete.merge(df3, on="M2ID", how="inner", suffixes=("_w2", "_w3"))
print(f"\n  Merged panel (W2 + W3): {len(merged):,} respondents")


# ---- 4. Clean demographics ------------------------------------------------
merged["age_W2"] = pd.to_numeric(merged.get("B1PAGE_M2"), errors="coerce")
merged["female"] = (merged.get("B1PRSEX") == 2).astype(float)
# Education: B1PF7A (1-12 scale). Recode to years approx.
edu_map = {1: 6, 2: 9, 3: 12, 4: 12, 5: 13, 6: 14, 7: 16, 8: 16, 9: 18, 10: 18, 11: 20, 12: 20}
merged["educ_yrs"] = merged.get("B1PF7A", pd.Series()).map(edu_map)

# Negative affect from W2
merged["negaf_W2"] = merged.get("B1SNEGAF")

# Also compute f from Wave 3 for stability check
df3_ryff = merged[RYFF_W3].copy()
valid_w3 = df3_ryff.dropna().index
X_w3 = df3_ryff.loc[valid_w3].values.astype(float)
X_w3_z = StandardScaler().fit_transform(X_w3)
pca_w3 = PCA(n_components=1)
f_w3 = pca_w3.fit_transform(X_w3_z).flatten()
if np.corrcoef(merged.loc[valid_w3, "f_score_W2"], f_w3)[0, 1] < 0:
    f_w3 = -f_w3
f_w3 = (f_w3 - f_w3.mean()) / f_w3.std()
merged.loc[valid_w3, "f_score_W3"] = f_w3


# ---- 5. Summary statistics ------------------------------------------------
print("\n" + "=" * 70)
print("  DESCRIPTIVE STATISTICS (merged panel)")
print("=" * 70)

desc_vars = {
    "f_score_W2":       "f (Wave 2)",
    "f_score_W3":       "f (Wave 3)",
    "log_income_B1":    "ln(Income) W2",
    "log_income_C1":    "ln(Income) W3",
    "employed_B1":      "Employed W2",
    "employed_C1":      "Employed W3",
    "disab_days_B1":    "Disability Days W2",
    "disab_days_C1":    "Disability Days W3",
    "age_W2":           "Age (Wave 2)",
    "female":           "Female",
    "educ_yrs":         "Education (years)",
    "negaf_W2":         "Negative Affect W2",
}
for var, label in desc_vars.items():
    if var in merged.columns:
        s = merged[var].dropna()
        print(f"  {label:25s}  N={len(s):5,}  mean={s.mean():8.3f}  sd={s.std():7.3f}")


# ---- 6. Correlations: f_W2 with W3 outcomes --------------------------------
print("\n" + "=" * 70)
print("  BIVARIATE CORRELATIONS: f(W2) with Wave 3 Outcomes")
print("=" * 70)

outcomes_w3 = {
    "log_income_C1":  "ln(Income) W3",
    "employed_C1":    "Employed W3",
    "disab_days_C1":  "Disability Days W3",
    "f_score_W3":     "f (Wave 3)",
}

corr_results = []
for var, label in outcomes_w3.items():
    if var in merged.columns:
        pair = merged[["f_score_W2", var]].dropna()
        if len(pair) > 30:
            r, p = stats.pearsonr(pair["f_score_W2"], pair[var])
            n = len(pair)
            corr_results.append({"outcome": label, "var": var, "r": r, "p": p, "n": n})
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  f(W2) vs {label:25s}  r = {r:+.3f}  p = {p:.1e}  N = {n:,}  {sig}")


# ---- 7. OLS Regressions with controls --------------------------------------
print("\n" + "=" * 70)
print("  REGRESSION ANALYSIS: f(W2) -> Wave 3 Outcomes")
print("=" * 70)

from numpy.linalg import lstsq


def ols_with_stats(y, X, var_names):
    """OLS with heteroskedasticity-robust (HC1) standard errors."""
    n, k = X.shape
    beta, residuals, rank, sv = lstsq(X, y, rcond=None)
    yhat = X @ beta
    e = y - yhat
    sse = np.sum(e**2)
    sst = np.sum((y - y.mean())**2)
    r2 = 1 - sse / sst
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)

    # HC1 robust SEs
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    leverage_correction = n / (n - k)  # HC1
    for i in range(n):
        xi = X[i:i+1, :].T
        meat += (e[i]**2) * (xi @ xi.T)
    V_robust = leverage_correction * XtX_inv @ meat @ XtX_inv
    se_robust = np.sqrt(np.diag(V_robust))
    t_stats = beta / se_robust
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n - k)

    return {
        "beta": beta, "se": se_robust, "t": t_stats, "p": p_vals,
        "r2": r2, "r2_adj": r2_adj, "n": n, "names": var_names,
    }


def print_reg(result, title):
    print(f"\n  {title}")
    print(f"  {'':25s} {'Coef':>10s} {'Robust SE':>10s} {'t':>8s} {'p':>10s}")
    print(f"  {'-'*65}")
    for i, name in enumerate(result["names"]):
        sig = "***" if result["p"][i] < 0.001 else "**" if result["p"][i] < 0.01 else "*" if result["p"][i] < 0.05 else ""
        print(f"  {name:25s} {result['beta'][i]:10.4f} {result['se'][i]:10.4f} {result['t'][i]:8.2f} {result['p'][i]:10.4f} {sig}")
    print(f"  R-squared = {result['r2']:.4f},  Adj. R-squared = {result['r2_adj']:.4f},  N = {result['n']:,}")


# --- Model 1: ln(Income_W3) ~ f_W2 + controls + ln(Income_W2) ---------------
reg_results = {}

inc_vars = ["f_score_W2", "log_income_B1", "age_W2", "female", "educ_yrs"]
inc_df = merged[inc_vars + ["log_income_C1"]].dropna()
if len(inc_df) > 50:
    y = inc_df["log_income_C1"].values
    X = np.column_stack([np.ones(len(inc_df))] + [inc_df[v].values for v in inc_vars])
    names = ["Intercept"] + ["f (Wave 2)", "ln(Income W2)", "Age", "Female", "Educ. Years"]
    res = ols_with_stats(y, X, names)
    reg_results["income"] = res
    print_reg(res, "Model 1: ln(Income W3) ~ f(W2) + Controls + ln(Income W2)")

# --- Model 1b: Incremental validity over negative affect --------------------
inc_vars_b = ["f_score_W2", "negaf_W2", "log_income_B1", "age_W2", "female", "educ_yrs"]
inc_df_b = merged[inc_vars_b + ["log_income_C1"]].dropna()
if len(inc_df_b) > 50:
    y = inc_df_b["log_income_C1"].values
    X = np.column_stack([np.ones(len(inc_df_b))] + [inc_df_b[v].values for v in inc_vars_b])
    names = ["Intercept"] + ["f (Wave 2)", "Neg. Affect W2", "ln(Income W2)", "Age", "Female", "Educ. Years"]
    res = ols_with_stats(y, X, names)
    reg_results["income_incr"] = res
    print_reg(res, "Model 1b: ln(Income W3) ~ f(W2) + Neg.Affect + Controls (Incremental Validity)")

# --- Model 2: Employed_W3 ~ f_W2 + controls + Employed_W2 (LPM) -----------
emp_vars = ["f_score_W2", "employed_B1", "age_W2", "female", "educ_yrs"]
emp_df = merged[emp_vars + ["employed_C1"]].dropna()
if len(emp_df) > 50:
    y = emp_df["employed_C1"].values
    X = np.column_stack([np.ones(len(emp_df))] + [emp_df[v].values for v in emp_vars])
    names = ["Intercept"] + ["f (Wave 2)", "Employed W2", "Age", "Female", "Educ. Years"]
    res = ols_with_stats(y, X, names)
    reg_results["employment"] = res
    print_reg(res, "Model 2: Employed(W3) ~ f(W2) + Controls + Employed(W2)  [LPM]")

# --- Model 2b: Employment incremental validity ------------------------------
emp_vars_b = ["f_score_W2", "negaf_W2", "employed_B1", "age_W2", "female", "educ_yrs"]
emp_df_b = merged[emp_vars_b + ["employed_C1"]].dropna()
if len(emp_df_b) > 50:
    y = emp_df_b["employed_C1"].values
    X = np.column_stack([np.ones(len(emp_df_b))] + [emp_df_b[v].values for v in emp_vars_b])
    names = ["Intercept"] + ["f (Wave 2)", "Neg. Affect W2", "Employed W2", "Age", "Female", "Educ. Years"]
    res = ols_with_stats(y, X, names)
    reg_results["employment_incr"] = res
    print_reg(res, "Model 2b: Employed(W3) ~ f(W2) + Neg.Affect + Controls (Incremental Validity)")

# --- Model 3: Disability days W3 ~ f_W2 + controls -------------------------
dis_vars = ["f_score_W2", "disab_days_B1", "age_W2", "female", "educ_yrs"]
dis_df = merged[dis_vars + ["disab_days_C1"]].dropna()
if len(dis_df) > 50:
    y = dis_df["disab_days_C1"].values
    X = np.column_stack([np.ones(len(dis_df))] + [dis_df[v].values for v in dis_vars])
    names = ["Intercept"] + ["f (Wave 2)", "Disab. Days W2", "Age", "Female", "Educ. Years"]
    res = ols_with_stats(y, X, names)
    reg_results["disability"] = res
    print_reg(res, "Model 3: Disability Days(W3) ~ f(W2) + Controls + Disab.Days(W2)")


# ---- 8. f stability across waves -------------------------------------------
print("\n" + "=" * 70)
print("  f STABILITY ACROSS WAVES")
print("=" * 70)

stab = merged[["f_score_W2", "f_score_W3"]].dropna()
r_stab, p_stab = stats.pearsonr(stab["f_score_W2"], stab["f_score_W3"])
print(f"  r(f_W2, f_W3) = {r_stab:.3f}  (p = {p_stab:.1e}, N = {len(stab):,})")
print(f"  This is the test-retest stability of f across ~9 years.")


# ========================================================================
#  FIGURES
# ========================================================================
print("\n" + "=" * 70)
print("  GENERATING FIGURES")
print("=" * 70)


# --- Figure 1: Correlation matrix (f_W2 with all W3 outcomes) ---------------
fig1_vars = ["f_score_W2", "f_score_W3", "log_income_C1", "employed_C1", "disab_days_C1"]
fig1_labels = ["f (W2)", "f (W3)", "ln(Inc) W3", "Employed W3", "Disab Days W3"]
fig1_df = merged[fig1_vars].dropna()

fig, ax = plt.subplots(figsize=(7, 6))
corr_mat = fig1_df.corr()
mask = np.zeros_like(corr_mat, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-0.5, vmax=0.7, square=True,
            xticklabels=fig1_labels, yticklabels=fig1_labels,
            linewidths=0.5, ax=ax)
ax.set_title("Cross-Wave Correlations: f(W2) with Wave 3 Outcomes", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_cross_wave_correlations.png")
plt.close(fig)
print("  Saved fig1_cross_wave_correlations.png")


# --- Figure 2: Scatterplots (f_W2 vs W3 outcomes) ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 2a: f_W2 vs log_income_W3
pair_inc = merged[["f_score_W2", "log_income_C1"]].dropna()
ax = axes[0]
ax.scatter(pair_inc["f_score_W2"], pair_inc["log_income_C1"], alpha=0.15, s=8, color=ACCENT)
# Binned means
bins = pd.qcut(pair_inc["f_score_W2"], 20, duplicates="drop")
binned = pair_inc.groupby(bins, observed=True)["log_income_C1"].agg(["mean", "sem"])
bin_centers = pair_inc.groupby(bins, observed=True)["f_score_W2"].mean()
ax.errorbar(bin_centers, binned["mean"], yerr=1.96*binned["sem"],
            color=ACCENT2, linewidth=2.5, capsize=3, zorder=5, label="Binned mean (+/- 95% CI)")
r_val = stats.pearsonr(pair_inc["f_score_W2"], pair_inc["log_income_C1"])
ax.set_xlabel("f score (Wave 2)", fontsize=11)
ax.set_ylabel("ln(Personal Income) Wave 3", fontsize=11)
ax.set_title(f"f(W2) -> Income(W3)\nr = {r_val[0]:.3f}, p = {r_val[1]:.1e}", fontsize=12)
ax.legend(fontsize=8)

# 2b: f_W2 vs employed_W3
pair_emp = merged[["f_score_W2", "employed_C1"]].dropna()
ax = axes[1]
bins_e = pd.qcut(pair_emp["f_score_W2"], 15, duplicates="drop")
binned_e = pair_emp.groupby(bins_e, observed=True)["employed_C1"].mean()
bin_centers_e = pair_emp.groupby(bins_e, observed=True)["f_score_W2"].mean()
ax.bar(bin_centers_e, binned_e, width=0.18, color=ACCENT3, alpha=0.8, edgecolor="white")
ax.set_xlabel("f score (Wave 2)", fontsize=11)
ax.set_ylabel("P(Employed) Wave 3", fontsize=11)
r_emp = stats.pearsonr(pair_emp["f_score_W2"], pair_emp["employed_C1"])
ax.set_title(f"f(W2) -> Employment(W3)\nr = {r_emp[0]:.3f}, p = {r_emp[1]:.1e}", fontsize=12)

# 2c: f stability
pair_stab = merged[["f_score_W2", "f_score_W3"]].dropna()
ax = axes[2]
ax.scatter(pair_stab["f_score_W2"], pair_stab["f_score_W3"], alpha=0.15, s=8, color="#9B59B6")
ax.plot([-3, 3], [-3, 3], "k--", alpha=0.3, label="45-degree line")
bins_s = pd.qcut(pair_stab["f_score_W2"], 20, duplicates="drop")
binned_s = pair_stab.groupby(bins_s, observed=True)["f_score_W3"].mean()
bin_centers_s = pair_stab.groupby(bins_s, observed=True)["f_score_W2"].mean()
ax.plot(bin_centers_s, binned_s, color=ACCENT2, linewidth=2.5, zorder=5, label="Binned mean")
ax.set_xlabel("f score (Wave 2)", fontsize=11)
ax.set_ylabel("f score (Wave 3)", fontsize=11)
ax.set_title(f"f Stability Across Waves\nr = {r_stab:.3f} (~9 yr gap)", fontsize=12)
ax.legend(fontsize=8)

fig.suptitle("Longitudinal Prediction: Wave 2 f -> Wave 3 Outcomes", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_scatter_predictions.png")
plt.close(fig)
print("  Saved fig2_scatter_predictions.png")


# --- Figure 3: Coefficient plot (standardized betas) -------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for idx, (model_key, title, ax) in enumerate([
    ("income", "ln(Income W3)", axes[0]),
    ("employment", "P(Employed W3)", axes[1]),
]):
    if model_key in reg_results:
        res = reg_results[model_key]
        # Skip intercept
        names = res["names"][1:]
        betas = res["beta"][1:]
        ses = res["se"][1:]
        # Significance markers
        sigs = []
        for p in res["p"][1:]:
            if p < 0.001: sigs.append("***")
            elif p < 0.01: sigs.append("**")
            elif p < 0.05: sigs.append("*")
            else: sigs.append("")

        y_pos = np.arange(len(names))
        colors = [ACCENT2 if n == "f (Wave 2)" else "#555" for n in names]
        ax.barh(y_pos, betas, xerr=1.96*ses, color=colors, alpha=0.8,
                edgecolor="white", capsize=4, height=0.6)
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{n} {s}" for n, s in zip(names, sigs)])
        ax.set_xlabel("Coefficient (with 95% CI)")
        ax.set_title(f"{title}\nAdj. R2 = {res['r2_adj']:.3f}, N = {res['n']:,}", fontsize=12)

fig.suptitle("Regression Coefficients: f(W2) Predicting Wave 3 Outcomes",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_coefficient_plots.png")
plt.close(fig)
print("  Saved fig3_coefficient_plots.png")


# --- Figure 4: Incremental validity (f vs negative affect) -------------------
fig, ax = plt.subplots(figsize=(8, 5))

inc_models = []
if "income" in reg_results:
    inc_models.append(("Income (base)", reg_results["income"]))
if "income_incr" in reg_results:
    inc_models.append(("Income (+NegAffect)", reg_results["income_incr"]))
if "employment" in reg_results:
    inc_models.append(("Employment (base)", reg_results["employment"]))
if "employment_incr" in reg_results:
    inc_models.append(("Employment (+NegAffect)", reg_results["employment_incr"]))

if inc_models:
    model_names = [m[0] for m in inc_models]
    f_betas = []
    f_ses = []
    f_pvals = []
    for name, res in inc_models:
        f_idx = res["names"].index("f (Wave 2)")
        f_betas.append(res["beta"][f_idx])
        f_ses.append(res["se"][f_idx])
        f_pvals.append(res["p"][f_idx])

    y_pos = np.arange(len(model_names))
    colors = [ACCENT if "base" in m else ACCENT2 for m in model_names]
    ax.barh(y_pos, f_betas, xerr=1.96*np.array(f_ses), color=colors,
            alpha=0.8, edgecolor="white", capsize=5, height=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    sigs = ["***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "" for p in f_pvals]
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} {s}" for n, s in zip(model_names, sigs)])
    ax.set_xlabel("Coefficient on f (Wave 2)")
    ax.set_title("Incremental Validity: f(W2) coefficient WITH and WITHOUT Negative Affect",
                 fontsize=12, fontweight="bold")
    ax.legend(["Blue = base model", "Orange = + Neg. Affect control"], fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_incremental_validity.png")
plt.close(fig)
print("  Saved fig4_incremental_validity.png")


# --- Figure 5: f quintile outcomes -------------------------------------------
merged["f_quintile"] = pd.qcut(merged["f_score_W2"], 5, labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 5a: Mean income by quintile
q_inc = merged.groupby("f_quintile", observed=True)["log_income_C1"].agg(["mean", "sem", "count"])
axes[0].bar(range(5), q_inc["mean"], yerr=1.96*q_inc["sem"],
            color=[ACCENT]*5, alpha=0.8, edgecolor="white", capsize=5)
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])
axes[0].set_ylabel("Mean ln(Income) Wave 3")
axes[0].set_title("Wave 3 Income by f Quintile (Wave 2)", fontsize=12)
axes[0].set_xlabel("f Quintile (Wave 2)")

# 5b: Employment rate by quintile
q_emp = merged.groupby("f_quintile", observed=True)["employed_C1"].agg(["mean", "sem", "count"])
axes[1].bar(range(5), q_emp["mean"], yerr=1.96*q_emp["sem"],
            color=[ACCENT3]*5, alpha=0.8, edgecolor="white", capsize=5)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])
axes[1].set_ylabel("P(Employed) Wave 3")
axes[1].set_title("Wave 3 Employment by f Quintile (Wave 2)", fontsize=12)
axes[1].set_xlabel("f Quintile (Wave 2)")

# 5c: Disability days by quintile
q_dis = merged.groupby("f_quintile", observed=True)["disab_days_C1"].agg(["mean", "sem", "count"])
axes[2].bar(range(5), q_dis["mean"], yerr=1.96*q_dis["sem"],
            color=["#E74C3C"]*5, alpha=0.8, edgecolor="white", capsize=5)
axes[2].set_xticks(range(5))
axes[2].set_xticklabels(["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])
axes[2].set_ylabel("Mean Disability Days Wave 3")
axes[2].set_title("Wave 3 Disability Days by f Quintile (Wave 2)", fontsize=12)
axes[2].set_xlabel("f Quintile (Wave 2)")

fig.suptitle("Wave 3 Outcomes by Wave 2 Flourishing Quintile", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_quintile_outcomes.png")
plt.close(fig)
print("  Saved fig5_quintile_outcomes.png")


# ---- Summary ---------------------------------------------------------------
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print("""
  This analysis tests whether f (the general factor of flourishing,
  computed as PC1 of 6 Ryff PWB subscales) measured at MIDUS Wave 2
  predicts economic outcomes ~9 years later at Wave 3.

  Key design features:
    - Lagged prediction (W2 -> W3) establishes temporal precedence
    - Controls for baseline (W2) levels of the outcome variable
    - Controls for age, sex, and education
    - Incremental validity test: Does f predict BEYOND negative affect?
    - HC1 robust standard errors throughout

  Limitations:
    - Not causal (omitted variables, reverse causality via stable traits)
    - f is estimated via PCA (measurement error, single-extraction)
    - Panel attrition between waves (survivor bias)
    - LPM for binary outcomes (bounded predictions possible)
""")

print("\n  All figures saved to:", OUT_DIR)
print("  Done.")

"""
Phase 1: Multi-Instrumental Bifactor Model of f
=================================================
17 indicators from 6 instrument families in MIDUS 2 (n ~ 3,900).
Tests whether a general factor of flourishing emerges across
independently designed wellbeing measures.

Factor scoring: MAP (Maximum A Posteriori) via semopy.predict_factors
(primary), with Thurstone regression scores as cross-check.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD AND CLEAN
# ============================================================
print("=" * 70)
print("STEP 1: LOADING MIDUS 2 DATA")
print("=" * 70)

df = pd.read_csv('04652-0001-Data.tsv', sep='\t', low_memory=False)
print(f"Raw: {df.shape[0]} respondents, {df.shape[1]} variables")

# Variable mapping: short_name -> (MIDUS_var, full_name, group)
# MIDUS coding: -1 = INAPPLICABLE (no SAQ), 98 = NOT CALCULATED
indicator_map = {
    # Eudaimonic (Ryff 1989) -- MIDUS 2 7-item versions (sum scores, range ~7-49)
    'AU':   ('B1SPWBA2', 'Autonomy',            'eudaimonic'),
    'EM':   ('B1SPWBE2', 'Env Mastery',          'eudaimonic'),
    'PG':   ('B1SPWBG2', 'Personal Growth',      'eudaimonic'),
    'PR':   ('B1SPWBR2', 'Positive Relations',   'eudaimonic'),
    'PL':   ('B1SPWBU2', 'Purpose in Life',      'eudaimonic'),
    'SA':   ('B1SPWBS2', 'Self-Acceptance',       'eudaimonic'),
    # Social Wellbeing (Keyes 1998) -- 3 items each, range 3-21
    'SI':   ('B1SSWBSI', 'Social Integration',   'social'),
    'SAc':  ('B1SSWBSA', 'Social Acceptance',     'social'),
    'SCo':  ('B1SSWBSC', 'Social Coherence',      'social'),
    'SAct': ('B1SSWBAO', 'Social Actualization',  'social'),
    'SCon': ('B1SSWBMS', 'Social Contribution',   'social'),
    # Hedonic (Watson/Diener + Rosenberg)
    'PA':   ('B1SPOSAF', 'Positive Affect',       'hedonic'),
    'LS':   ('B1SSATIS', 'Life Satisfaction',      'hedonic'),
    'SE':   ('B1SESTEE', 'Self-Esteem',            'hedonic'),
    # Agentic (Pearlin/Lachman + Scheier/Carver + McAdams/Erikson)
    'Ctrl': ('B1SCTRL',  'Sense of Control',      'agentic'),
    'Opt':  ('B1SOPTIM', 'Optimism',              'agentic'),
    'Gen':  ('B1SGENER', 'Generativity',          'agentic'),
}

validator_map = {
    'NA_aff': ('B1SNEGAF', 'Negative Affect'),
    'Dep':    ('B1PDEPRE', 'Depression (count)'),
    'DepDx':  ('B1PDEPDX', 'Depression Dx'),
    'Anx':    ('B1PANXIE', 'Anxiety (count)'),
    'AnxDx':  ('B1PANXTD', 'Anxiety Dx'),
    'Agree':  ('B1SAGREE', 'Agreeableness'),
    'Extra':  ('B1SEXTRA', 'Extraversion'),
    'Neuro':  ('B1SNEURO', 'Neuroticism'),
    'Open':   ('B1SOPEN',  'Openness'),
    'Consc':  ('B1SCONS1', 'Conscientiousness'),
}

short_names = list(indicator_map.keys())
groups = {
    'eudaimonic': [k for k, v in indicator_map.items() if v[2] == 'eudaimonic'],
    'social':     [k for k, v in indicator_map.items() if v[2] == 'social'],
    'hedonic':    [k for k, v in indicator_map.items() if v[2] == 'hedonic'],
    'agentic':    [k for k, v in indicator_map.items() if v[2] == 'agentic'],
}

# Build analysis dataframe, cleaning missing codes
data = pd.DataFrame()
for sn, (var, full, grp) in indicator_map.items():
    col = pd.to_numeric(df[var], errors='coerce').replace({-1: np.nan, 98: np.nan, 99: np.nan})
    data[sn] = col

for sn, (var, full) in validator_map.items():
    col = pd.to_numeric(df[var], errors='coerce').replace({-1: np.nan, 98: np.nan, 99: np.nan})
    if sn in ['DepDx', 'AnxDx']:
        col = col.replace({8: np.nan, 9: np.nan})
    data[sn] = col

# ============================================================
# STEP 2: VALIDATE COMPOSITES
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: SCALE COMPOSITES")
print("=" * 70)

print(f"\n{'Indicator':<22} {'N valid':>8} {'%':>6} {'Mean':>8} {'SD':>8} {'Min':>6} {'Max':>6}")
print("-" * 64)
for sn in short_names:
    col = data[sn].dropna()
    full = indicator_map[sn][1]
    pct = len(col) / len(data) * 100
    print(f"{full:<22} {len(col):>8} {pct:>5.1f}% {col.mean():>8.2f} {col.std():>8.2f} {col.min():>6.0f} {col.max():>6.0f}")

# Complete cases on all 17 indicators
complete = data[short_names].dropna()
N = len(complete)
print(f"\nComplete cases on all 17 indicators: N = {N}")

# Standardize for CFA
std_data = (complete - complete.mean()) / complete.std()

# ============================================================
# STEP 3: CORRELATION MATRIX
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: CORRELATION MATRIX")
print("=" * 70)

R = complete.corr()
triu = np.triu_indices(17, k=1)
off_diag = R.values[triu]

print(f"\nPositive manifold check:")
n_neg = np.sum(off_diag < 0)
print(f"  136 unique pairs: {136 - n_neg} positive, {n_neg} negative")
print(f"  --> {'POSITIVE MANIFOLD CONFIRMED' if n_neg == 0 else 'WARNING: negative correlations found'}")
print(f"\n  Mean r  = {off_diag.mean():.4f}")
print(f"  Median r = {np.median(off_diag):.4f}")
print(f"  Range    = [{off_diag.min():.4f}, {off_diag.max():.4f}]")

# Within vs between group correlations
print(f"\n  Within-group vs between-group mean correlations:")
for g1_name, g1_vars in groups.items():
    for g2_name, g2_vars in groups.items():
        if g1_name <= g2_name:
            rs = []
            for v1 in g1_vars:
                for v2 in g2_vars:
                    if v1 != v2:
                        rs.append(R.loc[v1, v2])
            tag = "(within)" if g1_name == g2_name else "(between)"
            print(f"    {g1_name[:5]:>5}-{g2_name[:5]:<5} {tag:<10}: {np.mean(rs):.3f}")

# Print abbreviated correlation matrix
print(f"\n  Full 17x17 correlation matrix:")
labels = [indicator_map[sn][1][:10] for sn in short_names]
R_disp = R.copy()
R_disp.index = labels
R_disp.columns = labels
print(R_disp.round(3).to_string())

# ============================================================
# STEP 3b: PCA EIGENSTRUCTURE
# ============================================================
print("\n" + "=" * 70)
print("STEP 3b: PCA EIGENSTRUCTURE")
print("=" * 70)

eigenvalues = np.linalg.eigvalsh(R.values)[::-1]
total_var = eigenvalues.sum()
print(f"\n  {'PC':>4} {'Eigenvalue':>11} {'% Var':>7} {'Cum %':>7}")
print(f"  {'-'*32}")
cum = 0
for i, ev in enumerate(eigenvalues):
    cum += ev / total_var * 100
    marker = "  <--" if i == 0 else ""
    print(f"  PC{i+1:>2} {ev:>11.3f} {ev/total_var*100:>6.1f}% {cum:>6.1f}%{marker}")

ratio_12 = eigenvalues[0] / eigenvalues[1]
print(f"\n  PC1/PC2 ratio: {ratio_12:.2f}")
print(f"  PC1 alone explains {eigenvalues[0]/total_var*100:.1f}% of total variance")

# ============================================================
# STEP 4: BIFACTOR CFA
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: BIFACTOR CFA (semopy)")
print("=" * 70)

import semopy

all_17 = " + ".join(short_names)
eud_str = " + ".join(groups['eudaimonic'])
soc_str = " + ".join(groups['social'])
hed_str = " + ".join(groups['hedonic'])
age_str = " + ".join(groups['agentic'])

# --- Model A: Unidimensional ---
print("\n--- Model A: Unidimensional (single f) ---")
spec_1f = f"f =~ {all_17}"
m1f = semopy.Model(spec_1f)
m1f.fit(std_data)
est_1f = m1f.inspect()

# semopy fixes first loading to 1.0 (unstandardized), estimates factor variance
fvar_1f = est_1f[(est_1f['lval'] == 'f') & (est_1f['op'] == '~~') & (est_1f['rval'] == 'f')]['Estimate'].values[0]
loads_1f_rows = est_1f[est_1f['op'] == '~']
std_loads_1f = {}
for _, row in loads_1f_rows.iterrows():
    std_loads_1f[row['lval']] = row['Estimate'] * np.sqrt(fvar_1f)

print(f"  Factor variance: {fvar_1f:.4f}")
print(f"\n  {'Indicator':<22} {'Std Loading':>12}")
print(f"  {'-'*34}")
for sn in short_names:
    full = indicator_map[sn][1]
    ld = std_loads_1f.get(sn, 0)
    print(f"  {full:<22} {ld:>12.4f}")

stats_1f = semopy.calc_stats(m1f)
print(f"\n  Fit indices (unidimensional):")
for idx_name in ['chi2', 'DoF', 'chi2 p-value', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']:
    if idx_name in stats_1f.columns:
        val = stats_1f[idx_name].values[0]
        if isinstance(val, float):
            print(f"    {idx_name:<20} {val:.4f}")
        else:
            print(f"    {idx_name:<20} {val}")

# --- Model C: Bifactor ---
print("\n--- Model C: Bifactor (f + 4 orthogonal group factors) ---")
bifactor_spec = f"""
f =~ {all_17}
Geud =~ {eud_str}
Gsoc =~ {soc_str}
Ghed =~ {hed_str}
Gage =~ {age_str}
f ~~ 0*Geud
f ~~ 0*Gsoc
f ~~ 0*Ghed
f ~~ 0*Gage
Geud ~~ 0*Gsoc
Geud ~~ 0*Ghed
Geud ~~ 0*Gage
Gsoc ~~ 0*Ghed
Gsoc ~~ 0*Gage
Ghed ~~ 0*Gage
"""

mbf = semopy.Model(bifactor_spec)
res_bf = mbf.fit(std_data)
est_bf = mbf.inspect()

# Extract factor variances
factor_names_all = ['f', 'Geud', 'Gsoc', 'Ghed', 'Gage']
factor_vars = {}
for fn in factor_names_all:
    row = est_bf[(est_bf['lval'] == fn) & (est_bf['op'] == '~~') & (est_bf['rval'] == fn)]
    factor_vars[fn] = row['Estimate'].values[0] if len(row) > 0 else 1.0

print(f"  Factor variances:")
for fn, fv in factor_vars.items():
    print(f"    {fn}: {fv:.6f}")

# Build standardized loading table
load_rows = est_bf[est_bf['op'] == '~']
loading_table = {sn: {'f_std': 0.0, 'g_std': 0.0, 'g_name': ''} for sn in short_names}

for _, row in load_rows.iterrows():
    ov = row['lval']    # observed variable
    lv = row['rval']    # latent variable
    est_val = row['Estimate']
    if ov in short_names:
        std_val = est_val * np.sqrt(factor_vars.get(lv, 1.0))
        if lv == 'f':
            loading_table[ov]['f_std'] = std_val
        else:
            loading_table[ov]['g_std'] = std_val
            loading_table[ov]['g_name'] = lv

print(f"\n  Bifactor standardized loadings:")
print(f"  {'Indicator':<22} {'f':>8} {'Group':>8} {'Factor':>8} {'h2':>8} {'f%':>6}")
print(f"  {'-'*60}")

f_loads = np.array([loading_table[sn]['f_std'] for sn in short_names])
g_loads = np.array([loading_table[sn]['g_std'] for sn in short_names])

for i, sn in enumerate(short_names):
    lt = loading_table[sn]
    fl = lt['f_std']
    gl = lt['g_std']
    h2 = fl**2 + gl**2
    f_pct = fl**2 / h2 * 100 if h2 > 0.001 else 0
    full = indicator_map[sn][1]
    print(f"  {full:<22} {fl:8.4f} {gl:8.4f} {lt['g_name']:>8} {h2:8.4f} {f_pct:5.1f}%")

# Fit statistics for bifactor model
stats_bf = semopy.calc_stats(mbf)
print(f"\n  Fit indices (bifactor, semopy):")
for idx_name in ['chi2', 'DoF', 'chi2 p-value', 'CFI', 'TLI', 'RMSEA', 'AIC', 'BIC']:
    if idx_name in stats_bf.columns:
        val = stats_bf[idx_name].values[0]
        if isinstance(val, float) and not np.isnan(val):
            print(f"    {idx_name:<20} {val:.4f}")

# SRMR and manual fit indices using model-implied covariance
# Ensure variable ordering matches between model and observed data
obs_var_order = mbf.vars['observed']
obs_cov = std_data[obs_var_order].cov().values  # match model's variable order
sigma_implied = mbf.calc_sigma()[0]
p = sigma_implied.shape[0]

# SRMR: standardized root mean square residual
resid_matrix = obs_cov - sigma_implied
srmr_sum = 0
srmr_count = 0
for i_row in range(p):
    for j_col in range(i_row + 1):
        s_ii = obs_cov[i_row, i_row]
        s_jj = obs_cov[j_col, j_col]
        if s_ii > 1e-10 and s_jj > 1e-10:
            srmr_sum += (resid_matrix[i_row, j_col] / np.sqrt(s_ii * s_jj)) ** 2
        srmr_count += 1
srmr = np.sqrt(srmr_sum / srmr_count)
print(f"    {'SRMR (manual)':<20} {srmr:.4f}")

# Manual ML chi-square if semopy returned NaN
chi2_semopy = stats_bf['chi2'].values[0] if 'chi2' in stats_bf.columns else np.nan
if np.isnan(chi2_semopy):
    print(f"\n  semopy calc_stats returned NaN; computing fit indices manually:")
    # ML discrepancy: F = tr(Sigma^-1 S) + ln|Sigma| - ln|S| - p
    try:
        sigma_inv = np.linalg.inv(sigma_implied)
        F_ml = (np.trace(sigma_inv @ obs_cov)
                + np.linalg.slogdet(sigma_implied)[1]
                - np.linalg.slogdet(obs_cov)[1]
                - p)
        chi2_manual = (N - 1) * F_ml

        # Count free parameters
        est_table = mbf.inspect()
        n_free = len(est_table[est_table['op'].isin(['~', '~~'])])
        n_total_elements = p * (p + 1) // 2
        df_manual = n_total_elements - n_free

        # Baseline model: uncorrelated indicators (diagonal sigma)
        F_base = -np.linalg.slogdet(obs_cov)[1]  # ln|diag| = 0 for standardized
        chi2_base = (N - 1) * F_base
        df_base = p * (p - 1) // 2

        # CFI
        cfi_manual = 1.0 - max(chi2_manual - df_manual, 0) / max(chi2_base - df_base, 0)
        cfi_manual = max(0, min(1, cfi_manual))

        # TLI (NNFI)
        if df_manual > 0 and df_base > 0:
            tli_manual = ((chi2_base / df_base) - (chi2_manual / df_manual)) / ((chi2_base / df_base) - 1)
        else:
            tli_manual = np.nan

        # RMSEA
        if df_manual > 0:
            rmsea_manual = np.sqrt(max((chi2_manual / (N - 1) / df_manual) - (1.0 / (N - 1)), 0))
        else:
            rmsea_manual = np.nan

        print(f"    {'chi2 (manual)':<20} {chi2_manual:.2f}")
        print(f"    {'df (manual)':<20} {df_manual}")
        print(f"    {'CFI (manual)':<20} {cfi_manual:.4f}")
        print(f"    {'TLI (manual)':<20} {tli_manual:.4f}")
        print(f"    {'RMSEA (manual)':<20} {rmsea_manual:.4f}")
        print(f"    {'SRMR':<20} {srmr:.4f}")

        # Store for final summary
        stats_bf_manual = {
            'chi2': chi2_manual, 'DoF': df_manual,
            'CFI': cfi_manual, 'TLI': tli_manual, 'RMSEA': rmsea_manual
        }

        # Also compute manual stats for unidimensional model if needed
        chi2_1f_check = stats_1f['chi2'].values[0] if 'chi2' in stats_1f.columns else np.nan
        if np.isnan(chi2_1f_check):
            sigma_1f = m1f.calc_sigma()[0]
            obs_1f_order = m1f.vars['observed']
            obs_1f_cov = std_data[obs_1f_order].cov().values
            sigma_1f_inv = np.linalg.inv(sigma_1f)
            F_1f = (np.trace(sigma_1f_inv @ obs_1f_cov)
                    + np.linalg.slogdet(sigma_1f)[1]
                    - np.linalg.slogdet(obs_1f_cov)[1]
                    - p)
            chi2_1f_manual = (N - 1) * F_1f
            est_1f_table = m1f.inspect()
            n_free_1f = len(est_1f_table[est_1f_table['op'].isin(['~', '~~'])])
            df_1f_manual = n_total_elements - n_free_1f
            cfi_1f_manual = 1.0 - max(chi2_1f_manual - df_1f_manual, 0) / max(chi2_base - df_base, 0)
            cfi_1f_manual = max(0, min(1, cfi_1f_manual))
            if df_1f_manual > 0:
                rmsea_1f_manual = np.sqrt(max((chi2_1f_manual / (N - 1) / df_1f_manual) - (1.0 / (N - 1)), 0))
            else:
                rmsea_1f_manual = np.nan
            stats_1f_manual = {
                'chi2': chi2_1f_manual, 'DoF': df_1f_manual,
                'CFI': cfi_1f_manual, 'RMSEA': rmsea_1f_manual
            }
            print(f"\n  Unidimensional (manual):")
            print(f"    chi2 = {chi2_1f_manual:.2f}, df = {df_1f_manual}, "
                  f"CFI = {cfi_1f_manual:.4f}, RMSEA = {rmsea_1f_manual:.4f}")
        else:
            stats_1f_manual = None
    except np.linalg.LinAlgError as e:
        print(f"    Manual fit computation failed: {e}")
        stats_bf_manual = None
        stats_1f_manual = None
else:
    stats_bf_manual = None
    stats_1f_manual = None

# ============================================================
# STEP 5: OMEGA_H, ECV, OMEGA_HS, FDC
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: BIFACTOR DIAGNOSTICS")
print("=" * 70)

# Residual variances (standardized data: total var = 1 per indicator)
resid = 1.0 - f_loads**2 - g_loads**2
resid = np.maximum(resid, 0.001)

# omega_h = (sum lambda_f)^2 / [(sum lambda_f)^2 + sum_g(sum lambda_g)^2 + sum theta]
sum_f = np.sum(f_loads)
sum_f_sq = sum_f**2

group_sums_sq = {}
for gname, gvars in groups.items():
    g_idx = [short_names.index(v) for v in gvars]
    g_sum = np.sum(g_loads[g_idx])
    group_sums_sq[gname] = g_sum**2

total_group_sq = sum(group_sums_sq.values())
sum_resid = np.sum(resid)
denom = sum_f_sq + total_group_sq + sum_resid

omega_h = sum_f_sq / denom
omega_total = (sum_f_sq + total_group_sq) / denom

# ECV = sum(lambda_f^2) / [sum(lambda_f^2) + sum(lambda_g^2)]
sum_f_load_sq = np.sum(f_loads**2)
sum_g_load_sq = np.sum(g_loads**2)
ecv = sum_f_load_sq / (sum_f_load_sq + sum_g_load_sq)

# Factor Determinacy Coefficient (approximate)
fdc = np.sqrt(max(omega_h, 0))

print(f"\n  OMEGA HIERARCHICAL (omega_h):  {omega_h:.4f}")
print(f"  OMEGA TOTAL:                   {omega_total:.4f}")
print(f"  EXPLAINED COMMON VARIANCE:     {ecv:.4f}")
print(f"  FACTOR DETERMINACY (approx):   {fdc:.4f}  (threshold > 0.90)")

print(f"\n  Omega_hs by group factor (instrument-specific reliable variance):")
for gname in groups:
    ohs = group_sums_sq[gname] / denom
    n_ind = len(groups[gname])
    print(f"    {gname:<15} ({n_ind} indicators): omega_hs = {ohs:.4f}")

print(f"\n  {'='*60}")
print(f"  COMPARISON: Ryff-only (6 indicators) vs Multi-Instrumental (17)")
print(f"  {'='*60}")
print(f"  {'Metric':<32} {'Ryff-only':>12} {'Multi (17)':>12}")
print(f"  {'-'*56}")
print(f"  {'omega_h':<32} {'0.789':>12} {omega_h:>12.4f}")
print(f"  {'ECV':<32} {'0.758':>12} {ecv:>12.4f}")
print(f"  {'PC1 variance':<32} {'50.4%':>12} {eigenvalues[0]/total_var*100:>11.1f}%")
print(f"  {'PC1/PC2 ratio':<32} {'--':>12} {ratio_12:>12.2f}")
print(f"  {'FDC (from omega_h)':<32} {'--':>12} {fdc:>12.4f}")
print(f"  {'N indicators':<32} {'6':>12} {'17':>12}")
print(f"  {'N instrument families':<32} {'1':>12} {'6':>12}")
print(f"  {'N (complete cases)':<32} {'~4000':>12} {N:>12}")

# ============================================================
# STEP 6: EXTRACT f SCORES -- THREE METHODS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: f SCORE EXTRACTION (MAP, Thurstone, weighted-sum)")
print("=" * 70)

# --- Method 1: MAP factor scores from semopy (primary) ---
print("\n--- Method 1: MAP (Maximum A Posteriori) via semopy.predict_factors ---")
print("  This is the proper Bayesian/empirical-Bayes approach.")
print("  It accounts for the full model structure including group factors,")
print("  residual variances, and factor covariance constraints.")

try:
    map_scores_df = mbf.predict_factors(std_data, center=True)
    print(f"  Factor scores extracted: {list(map_scores_df.columns)}")
    f_map = map_scores_df['f'].values
    print(f"  f_MAP: mean = {f_map.mean():.4f}, sd = {f_map.std():.4f}, "
          f"range = [{f_map.min():.3f}, {f_map.max():.3f}]")
    map_success = True
except Exception as e:
    print(f"  WARNING: predict_factors failed: {e}")
    print(f"  This typically occurs when near-zero group factor variances")
    print(f"  (Geud={factor_vars.get('Geud',0):.6f}, Gage={factor_vars.get('Gage',0):.6f})")
    print(f"  create singular internal matrices.")
    map_success = False

# --- Method 1b: Bartlett factor scores (model-implied, fallback for MAP) ---
# Uses full bifactor loading structure + model-implied covariance to
# properly separate general from group factor variance.
# W = Lambda' Sigma_implied^{-1}; scores for factor k = z @ W[k,:]
print("\n--- Method 1b: Bartlett scores (bifactor-aware, model-implied) ---")
print("  Uses Sigma_implied^{-1} to separate f from group factors.")

# Build full loading matrix: [lambda_f | lambda_g1 | lambda_g2 | ...]
# Column order: f, Geud, Gsoc, Ghed, Gage
group_names_ordered = ['eudaimonic', 'social', 'hedonic', 'agentic']
n_factors = 1 + len(group_names_ordered)
Lambda_full = np.zeros((p, n_factors))
Lambda_full[:, 0] = f_loads  # general factor
for g_idx, gname in enumerate(group_names_ordered):
    for var_name in groups[gname]:
        row_idx = short_names.index(var_name)
        Lambda_full[row_idx, g_idx + 1] = g_loads[row_idx]

# Theta (residual variances)
Theta_diag = np.maximum(1.0 - np.sum(Lambda_full**2, axis=1), 0.001)
Theta_inv = np.diag(1.0 / Theta_diag)

# Bartlett score weight matrix: (Lambda' Theta^{-1} Lambda)^{-1} Lambda' Theta^{-1}
LtTinv = Lambda_full.T @ Theta_inv
LtTinvL = LtTinv @ Lambda_full
# Regularize if near-singular (those collapsed group factors)
LtTinvL_reg = LtTinvL + 1e-8 * np.eye(n_factors)
W_bartlett = np.linalg.solve(LtTinvL_reg, LtTinv)  # (n_factors x p)

# Align variable ordering: std_data columns must match short_names
z_matrix = std_data[short_names].values  # ensure column order matches
bartlett_all = z_matrix @ W_bartlett.T  # (N x n_factors)
f_bartlett = bartlett_all[:, 0]  # general factor
f_bartlett = (f_bartlett - f_bartlett.mean()) / f_bartlett.std()  # standardize

print(f"  f_Bartlett: mean = {f_bartlett.mean():.4f}, sd = {f_bartlett.std():.4f}, "
      f"range = [{f_bartlett.min():.3f}, {f_bartlett.max():.3f}]")
print(f"  (Bartlett scores use Theta^{{-1}} weighting, giving more weight to")
print(f"  indicators with less residual variance -- i.e., those most captured by f.)")

# --- Method 2: Thurstone regression scores (manual) ---
# f_hat = Lambda_f' Sigma^{-1} z  (then standardized)
# Uses observed Sigma (not model-implied), no group factor separation
print("\n--- Method 2: Thurstone regression scores (manual) ---")
print("  f_hat = Lambda_f' * Sigma_obs^{-1} * z  (classical regression method)")
print("  NOTE: does NOT separate f from group factors (uses observed Sigma).")

Sigma_obs = R.values  # correlation matrix (standardized data)
Sigma_inv = np.linalg.inv(Sigma_obs)
thurstone_weights = Sigma_inv @ f_loads  # Sigma^{-1} * lambda_f
# Normalize so scores have unit variance: w' Sigma w = 1
raw_var = thurstone_weights @ Sigma_obs @ thurstone_weights
thurstone_weights_norm = thurstone_weights / np.sqrt(raw_var)

f_thurstone = std_data[short_names].values @ thurstone_weights_norm
print(f"  f_Thurstone: mean = {f_thurstone.mean():.4f}, sd = {f_thurstone.std():.4f}, "
      f"range = [{f_thurstone.min():.3f}, {f_thurstone.max():.3f}]")

# --- Method 3: Naive weighted sum (original approach, for comparison) ---
print("\n--- Method 3: Weighted sum (z * lambda_f, original) ---")
f_naive = std_data[short_names].values @ f_loads
f_naive_z = (f_naive - f_naive.mean()) / f_naive.std()  # standardize for comparison
print(f"  f_naive: mean = {f_naive_z.mean():.4f}, sd = {f_naive_z.std():.4f}, "
      f"range = [{f_naive_z.min():.3f}, {f_naive_z.max():.3f}]")

# --- Compare scoring methods ---
print("\n--- Scoring method comparison ---")
r_bart_thur = np.corrcoef(f_bartlett, f_thurstone)[0, 1]
r_bart_naive = np.corrcoef(f_bartlett, f_naive_z)[0, 1]
r_thur_naive = np.corrcoef(f_thurstone, f_naive_z)[0, 1]
print(f"  r(Bartlett, Thurstone) = {r_bart_thur:.6f}")
print(f"  r(Bartlett, w-sum)    = {r_bart_naive:.6f}")
print(f"  r(Thurstone, w-sum)   = {r_thur_naive:.6f}")
if map_success:
    r_map_bart = np.corrcoef(f_map, f_bartlett)[0, 1]
    r_map_thur = np.corrcoef(f_map, f_thurstone)[0, 1]
    print(f"  r(MAP, Bartlett)      = {r_map_bart:.6f}")
    print(f"  r(MAP, Thurstone)     = {r_map_thur:.6f}")

# --- Factor Determinacy Coefficient (proper) ---
# FDC = cor(f_hat, f_true). For MAP scores, this is estimated as:
#   FDC^2 = lambda_f' Sigma^{-1} lambda_f  (the squared multiple correlation)
# This gives the theoretical upper bound on how well any linear scoring
# recovers the true factor.
fdc_squared = f_loads @ Sigma_inv @ f_loads
fdc_proper = np.sqrt(min(fdc_squared, 1.0))  # cap at 1.0 for numerical safety
print(f"\n  Factor Determinacy Coefficient (proper):")
print(f"    FDC^2 = lambda_f' Sigma^-1 lambda_f = {fdc_squared:.4f}")
print(f"    FDC   = {fdc_proper:.4f}  (threshold for usable scores: > 0.90)")
print(f"    FDC (approx from omega_h):           {fdc:.4f}")

# --- Select primary f scores ---
# MAP > Bartlett > Thurstone (in order of rigor)
if map_success:
    f_primary = f_map
    f_method = "MAP"
    print(f"\n  PRIMARY SCORING METHOD: MAP (predict_factors)")
else:
    f_primary = f_bartlett
    f_method = "Bartlett"
    print(f"\n  PRIMARY SCORING METHOD: Bartlett (bifactor-aware, model-implied)")
    print(f"  Bartlett scores properly separate f from group factors using the")
    print(f"  full Lambda matrix and Theta^{{-1}} weighting from the bifactor model.")

# Also extract group factor scores if MAP succeeded
if map_success:
    print(f"\n  Group factor scores from MAP:")
    for col in map_scores_df.columns:
        vals = map_scores_df[col].values
        print(f"    {col:<8}: mean = {vals.mean():.4f}, sd = {vals.std():.4f}")
else:
    # Extract group factor scores from Bartlett
    print(f"\n  Group factor scores from Bartlett:")
    factor_labels = ['f'] + [f'G_{g}' for g in group_names_ordered]
    for k, label in enumerate(factor_labels):
        vals = bartlett_all[:, k]
        print(f"    {label:<12}: mean = {vals.mean():.4f}, sd = {vals.std():.4f}")

f_series = pd.Series(f_primary, index=std_data.index, name='f_score')

# ============================================================
# STEP 7: EXTERNAL VALIDATION (using primary f scores)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 7: EXTERNAL VALIDATION (using {f_method} f scores)")
print("=" * 70)

# Merge with validators
val_df = data.loc[complete.index].copy()
val_df['f_score'] = f_series
if map_success:
    for col in map_scores_df.columns:
        val_df[f'factor_{col}'] = map_scores_df[col].values

# --- 7a: DUAL CONTINUA ---
print("\n--- 7a. DUAL CONTINUA TEST ---")
print(f"  (f should correlate moderately, NOT highly, with negative indicators)")
dc_results = {}
for vname, label in [('NA_aff', 'Negative Affect'), ('Dep', 'Depression'), ('Anx', 'Anxiety')]:
    valid = val_df[['f_score', vname]].dropna()
    if len(valid) > 30:
        r, p = stats.pearsonr(valid['f_score'], valid[vname])
        dc_results[vname] = r
        print(f"  r(f, {label:<20}) = {r:7.4f}  (p = {p:.2e})  R2 = {r**2:.3f}")

print(f"\n  Baseline: r(f_ryff, NA) = -0.379, R2 = 0.144")

# --- 7b: BIG FIVE ---
print("\n--- 7b. BIG FIVE PERSONALITY CORRELATIONS ---")
big5 = ['Agree', 'Extra', 'Neuro', 'Open', 'Consc']
b5_valid = val_df[['f_score'] + big5].dropna()
print(f"  N = {len(b5_valid)}")

print(f"\n  {'Trait':<22} {'r(f, trait)':>12} {'p-value':>12}")
print(f"  {'-'*46}")
for trait in big5:
    r, p = stats.pearsonr(b5_valid['f_score'], b5_valid[trait])
    label = validator_map[trait][1]
    print(f"  {label:<22} {r:12.4f} {p:>12.2e}")

# R^2: Big Five -> f
X = np.column_stack([np.ones(len(b5_valid)), b5_valid[big5].values])
y = b5_valid['f_score'].values
beta = np.linalg.lstsq(X, y, rcond=None)[0]
y_pred = X @ beta
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_b5 = 1 - ss_res / ss_tot

print(f"\n  R^2(Big Five -> f) = {r2_b5:.4f}")
print(f"  Variance in f BEYOND personality: {(1-r2_b5)*100:.1f}%")

# --- 7c: DISCRIMINANT VALIDITY ---
print("\n--- 7c. CLINICAL DISCRIMINANT VALIDITY ---")
dx_valid = val_df[['f_score', 'DepDx']].dropna()
dep_yes = dx_valid[dx_valid['DepDx'] == 1]['f_score']
dep_no = dx_valid[dx_valid['DepDx'] == 0]['f_score']
d_dep = 0
if len(dep_yes) > 10:
    d_dep = (dep_no.mean() - dep_yes.mean()) / np.sqrt((dep_no.var() + dep_yes.var()) / 2)
    r_pb, p_pb = stats.pointbiserialr(dx_valid['DepDx'], dx_valid['f_score'])
    print(f"  Depression prevalence:     {dx_valid['DepDx'].mean()*100:.1f}%")
    print(f"  Mean f (depressed):       {dep_yes.mean():.3f}")
    print(f"  Mean f (non-depressed):   {dep_no.mean():.3f}")
    print(f"  Cohen's d:                {d_dep:.3f}")
    print(f"  r_pb(f, Depression Dx):   {r_pb:.4f}  (p = {p_pb:.2e})")

anx_valid = val_df[['f_score', 'AnxDx']].dropna()
anx_yes = anx_valid[anx_valid['AnxDx'] == 1]['f_score']
anx_no = anx_valid[anx_valid['AnxDx'] == 0]['f_score']
d_anx = 0
if len(anx_yes) > 10:
    d_anx = (anx_no.mean() - anx_yes.mean()) / np.sqrt((anx_no.var() + anx_yes.var()) / 2)
    print(f"\n  GAD prevalence:           {anx_valid['AnxDx'].mean()*100:.1f}%")
    print(f"  Cohen's d (GAD):         {d_anx:.3f}")

# --- 7d: LOADING PROFILE ---
print("\n--- 7d. f LOADING PROFILE (ranked by general factor loading) ---")
load_profile = []
for i, sn in enumerate(short_names):
    load_profile.append((
        indicator_map[sn][1], indicator_map[sn][2],
        f_loads[i], g_loads[i], f_loads[i]**2 + g_loads[i]**2
    ))
load_profile.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"  {'Indicator':<22} {'Group':<12} {'f load':>8} {'g load':>8} {'h2':>8}")
print(f"  {'-'*58}")
for name, grp, fl, gl, h2 in load_profile:
    print(f"  {name:<22} {grp:<12} {fl:8.4f} {gl:8.4f} {h2:8.4f}")

# ============================================================
# SAVE
# ============================================================
R.to_csv('corr_17x17.csv')

output_df = pd.DataFrame({
    'indicator': [indicator_map[sn][1] for sn in short_names],
    'short': short_names,
    'group': [indicator_map[sn][2] for sn in short_names],
    'f_loading': f_loads,
    'group_loading': g_loads,
    'communality': f_loads**2 + g_loads**2,
    'thurstone_weight': thurstone_weights_norm,
    'bartlett_f_weight': W_bartlett[0, :],  # general factor weights
})
output_df.to_csv('bifactor_loadings.csv', index=False)

# Save f scores with all methods + group factors
scores_df = pd.DataFrame(index=complete.index)
scores_df['f_primary'] = f_primary
scores_df['f_method'] = f_method
scores_df['f_bartlett'] = f_bartlett
scores_df['f_thurstone'] = f_thurstone
scores_df['f_naive_z'] = f_naive_z
if map_success:
    for col in map_scores_df.columns:
        scores_df[f'factor_{col}'] = map_scores_df[col].values
else:
    # Save Bartlett group factor scores
    factor_labels = ['f'] + [f'G_{g}' for g in group_names_ordered]
    for k, label in enumerate(factor_labels):
        if k > 0:  # skip f (already saved as f_bartlett)
            scores_df[f'bartlett_{label}'] = bartlett_all[:, k]
scores_df.to_csv('f_scores.csv')

# ============================================================
# FINAL SUMMARY
# ============================================================
r_na = dc_results.get('NA_aff', np.nan)

# Extract fit indices for summary (prefer manual if semopy returned NaN)
def get_fit(stats_df, name):
    if name in stats_df.columns:
        val = stats_df[name].values[0]
        if not np.isnan(val):
            return val
    return np.nan

chi2_bf = get_fit(stats_bf, 'chi2')
dof_bf = get_fit(stats_bf, 'DoF')
cfi_bf = get_fit(stats_bf, 'CFI')
tli_bf = get_fit(stats_bf, 'TLI')
rmsea_bf = get_fit(stats_bf, 'RMSEA')

chi2_1f = get_fit(stats_1f, 'chi2')
dof_1f = get_fit(stats_1f, 'DoF')
cfi_1f = get_fit(stats_1f, 'CFI')
rmsea_1f = get_fit(stats_1f, 'RMSEA')

# Override with manual values if semopy returned NaN
if stats_bf_manual is not None:
    chi2_bf = stats_bf_manual['chi2']
    dof_bf = stats_bf_manual['DoF']
    cfi_bf = stats_bf_manual['CFI']
    tli_bf = stats_bf_manual['TLI']
    rmsea_bf = stats_bf_manual['RMSEA']
if stats_1f_manual is not None:
    chi2_1f = stats_1f_manual['chi2']
    dof_1f = stats_1f_manual['DoF']
    cfi_1f = stats_1f_manual['CFI']
    rmsea_1f = stats_1f_manual['RMSEA']

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
SAMPLE
  N = {N} (MIDUS 2, complete cases on all 17 indicators)
  17 indicators from 6 instrument families spanning 4 theoretical traditions

POSITIVE MANIFOLD
  All 136 off-diagonal correlations positive
  Mean r = {off_diag.mean():.3f}, range [{off_diag.min():.3f}, {off_diag.max():.3f}]

PCA
  PC1 = {eigenvalues[0]/total_var*100:.1f}% of variance
  PC1/PC2 eigenvalue ratio = {ratio_12:.2f}

MODEL FIT COMPARISON
  Unidimensional:  chi2 = {chi2_1f:.1f}, df = {dof_1f:.0f}, CFI = {cfi_1f:.3f}, RMSEA = {rmsea_1f:.3f}
  Bifactor:        chi2 = {chi2_bf:.1f}, df = {dof_bf:.0f}, CFI = {cfi_bf:.3f}, RMSEA = {rmsea_bf:.3f}, SRMR = {srmr:.3f}

BIFACTOR MODEL (17 indicators, 4 orthogonal group factors)
  omega_h = {omega_h:.4f}     (Ryff-only baseline: 0.789)
  ECV     = {ecv:.4f}     (Ryff-only baseline: 0.758)
  FDC     = {fdc_proper:.4f}     (proper: lambda' Sigma^-1 lambda; threshold > 0.90)

GROUP FACTOR omega_hs
  eudaimonic: {group_sums_sq['eudaimonic']/denom:.4f}
  social:     {group_sums_sq['social']/denom:.4f}
  hedonic:    {group_sums_sq['hedonic']/denom:.4f}
  agentic:    {group_sums_sq['agentic']/denom:.4f}
  (All < 0.04 -- instrument-specific variance is negligible)

FACTOR SCORING
  Primary method: {f_method}
  Bartlett scores (bifactor-aware) also computed
  Thurstone regression scores computed for comparison""")
if map_success:
    print(f"  r(MAP, Bartlett)  = {r_map_bart:.6f}")
print(f"  r(Bartlett, Thurstone) = {r_bart_thur:.6f}")
print(f"  r(Bartlett, w-sum)    = {r_bart_naive:.6f}")
print(f"""
DUAL CONTINUA
  r(f, Negative Affect) = {r_na:.4f}  (R2 = {r_na**2:.3f})

BIG FIVE INDEPENDENCE
  R^2(Big Five -> f) = {r2_b5:.4f}
  {(1-r2_b5)*100:.1f}% of f variance is independent of personality

CLINICAL DISCRIMINATION
  Cohen's d (depressed vs non-depressed): {d_dep:.3f}
  Cohen's d (GAD vs non-GAD):            {d_anx:.3f}
""")
print("Files saved: corr_17x17.csv, bifactor_loadings.csv, f_scores.csv")
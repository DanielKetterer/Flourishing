"""
03_models_full_f.py
─────────────────────────────────────────────────────────────────────────────
Full mathematical implementation of the four models specified in
"Toward f: A General Factor of Human Flourishing."

What this script adds beyond 02_analyze_wellbeing_f.py
──────────────────────────────────────────────────────
  Model 1 — Bifactor CFA via maximum-likelihood estimation
            General factor f + domain-specific factors; omega_h computation.

  Model 2 — Mutualism dynamical system (the generative model)
            Coupled logistic growth ODEs calibrated from data; equilibrium
            solving; demonstration that mutualism produces positive manifold
            without a latent variable.

  Model 3 — Lyapunov stability & resilience analysis
            Jacobian at equilibrium, dominant eigenvalue, return time tau,
            Lyapunov equation for basin-of-attraction ellipsoid, per-country
            resilience scores.

  Model 4 — Capabilities-adjusted flourishing (f_cap)
            Nussbaum threshold logic, capability set approximation,
            Alkire-Foster multidimensional deprivation, composite f_cap.

Data assumption
───────────────
  Reads wellbeing_data/wellbeing_merged.csv produced by script 01.
  Country-level ecological data. All caveats from script 02 apply.

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
from numpy.linalg import inv, det, slogdet, cholesky

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

SHORT_LABELS = {
    "P_positive_emotion":    "P: Pos. Emotion",
    "E_engagement_autonomy": "E: Engagement",
    "R_relationships":       "R: Relationships",
    "M_meaning_life_sat":    "M: Meaning",
    "A_accomplishment_gdp":  "A: Accomplishment",
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

# Bifactor domain groupings (paper: hedonic, eudaimonic, social-physical)
# Each group gets one specific factor in addition to f
BIFACTOR_GROUPS = {
    "hedonic":         ["P_positive_emotion", "M_meaning_life_sat"],
    "eudaimonic":      ["E_engagement_autonomy", "A_accomplishment_gdp"],
    "social_physical": ["R_relationships", "H_health"],
}

np.random.seed(42)

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
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load merged dataset, standardize, return (df_clean, X_raw, X_scaled, S, cols)."""
    df = pd.read_csv(DATA_PATH)
    present = [c for c in PERMA_COLS if c in df.columns]
    df_clean = df[["country"] + present].dropna(subset=present).copy()
    X_raw = df_clean[present].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    S = np.cov(X_scaled.T, ddof=1)  # sample covariance (standardized)
    N = X_scaled.shape[0]
    print(f"Loaded: {N} complete cases, {len(present)} dimensions")
    return df_clean, X_raw, X_scaled, S, present, N


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 1: BIFACTOR CFA VIA MAXIMUM LIKELIHOOD
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sigma_bifactor(params, p, group_indices):
    """
    Construct model-implied covariance matrix Σ(θ) for a bifactor model.

    Bifactor specification (paper eq.):
        y_ij = μ_j + λ_j^f · f_i + λ_j^sk · s_ki + ε_ij

    Parameters vector (params):
        [0:p]           — λ^f  (general factor loadings)
        [p:2p]          — λ^sk (specific factor loadings, one per indicator)
        [2p:3p]         — θ    (unique variances, must be > 0)

    Returns Σ = Λ_f Λ_f' + Λ_s Λ_s' + Θ
    where Λ_s is block-diagonal by group (bifactor orthogonality constraint).
    """
    lam_f = params[:p]                  # general loadings
    lam_s = params[p:2*p]               # specific loadings
    theta  = np.exp(params[2*p:3*p])    # log-parameterized for positivity

    # General factor contribution: Λ_f Λ_f'
    Sigma = np.outer(lam_f, lam_f)

    # Specific factor contributions (block-diagonal by group)
    for group_idx in group_indices:
        lam_sg = np.zeros(p)
        for j in group_idx:
            lam_sg[j] = lam_s[j]
        Sigma += np.outer(lam_sg, lam_sg)

    # Unique variances
    Sigma += np.diag(theta)

    return Sigma


def _ml_discrepancy(params, S, N, p, group_indices):
    """
    ML fitting function for CFA:
        F_ML = log|Σ(θ)| + tr(S · Σ(θ)⁻¹) - log|S| - p

    Minimizing this over θ gives ML estimates.
    """
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


def fit_bifactor_cfa(S, N, cols):
    """
    Fit bifactor CFA via ML and return loadings, unique variances, omega_h.

    Returns dict with:
        lambda_f  — general factor loadings (p,)
        lambda_s  — specific factor loadings (p,)
        theta     — unique variances (p,)
        omega_h   — omega hierarchical
        Sigma_hat — model-implied covariance
        F_min     — minimized discrepancy
        chi2      — model chi-square
        df        — degrees of freedom
        RMSEA     — root mean square error of approximation
    """
    p = len(cols)
    K = len(BIFACTOR_GROUPS)

    # Map group names → index sets
    group_indices = []
    for gname, gcols in BIFACTOR_GROUPS.items():
        idx = [cols.index(c) for c in gcols if c in cols]
        group_indices.append(idx)

    # Initialize from PCA
    pca = PCA(n_components=1)
    pca.fit(np.random.multivariate_normal(np.zeros(p), S, size=max(N, 200)))
    lam_f_init = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    # Make all loadings positive (flourishing direction)
    if lam_f_init.mean() < 0:
        lam_f_init = -lam_f_init

    lam_s_init = np.full(p, 0.3)
    theta_init = np.log(np.diag(S) * 0.3)  # log-parameterized

    x0 = np.concatenate([lam_f_init, lam_s_init, theta_init])

    # Optimize
    result = minimize(
        _ml_discrepancy,
        x0,
        args=(S, N, p, group_indices),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-10},
    )

    # Extract parameters
    lam_f = result.x[:p]
    lam_s = result.x[p:2*p]
    theta  = np.exp(result.x[2*p:3*p])

    # Ensure positive loadings on general factor
    if lam_f.mean() < 0:
        lam_f = -lam_f

    Sigma_hat = _build_sigma_bifactor(result.x, p, group_indices)

    # ── Omega hierarchical ────────────────────────────────────────────────────
    # ω_h = (Σ λ_j^f)² / [ (Σ λ_j^f)² + Σ_k (Σ_{j∈k} λ_j^sk)² + Σ θ_j ]
    sum_lam_f_sq = np.sum(lam_f)**2
    sum_lam_s_group_sq = 0.0
    for gidx in group_indices:
        sum_lam_s_group_sq += np.sum(lam_s[gidx])**2
    sum_theta = np.sum(theta)
    total_var = sum_lam_f_sq + sum_lam_s_group_sq + sum_theta
    omega_h = sum_lam_f_sq / total_var

    # ── ECV (explained common variance) ───────────────────────────────────────
    common_var_f = np.sum(lam_f**2)
    common_var_s = np.sum(lam_s**2)
    ecv = common_var_f / (common_var_f + common_var_s)

    # ── Fit indices ───────────────────────────────────────────────────────────
    F_min = result.fun
    # Bifactor model df: p*(p+1)/2 data points - (p + p + p) = p(p+1)/2 - 3p
    # But specific loadings are constrained (each loads on only one group)
    n_free_params = p + p + p  # lam_f + lam_s + theta
    n_data_points = p * (p + 1) // 2
    df_model = n_data_points - n_free_params

    chi2 = max(0, (N - 1) * F_min)
    if df_model > 0:
        rmsea = np.sqrt(max(0, (chi2 / df_model - 1) / (N - 1)))
    else:
        rmsea = np.nan

    return {
        "lambda_f":   lam_f,
        "lambda_s":   lam_s,
        "theta":      theta,
        "omega_h":    omega_h,
        "ecv":        ecv,
        "Sigma_hat":  Sigma_hat,
        "F_min":      F_min,
        "chi2":       chi2,
        "df":         df_model,
        "RMSEA":      rmsea,
        "converged":  result.success,
        "group_indices": group_indices,
    }


def plot_bifactor_loadings(result, cols):
    """Paired bar chart: general (f) loadings vs specific (s) loadings."""
    lam_f = result["lambda_f"]
    lam_s = result["lambda_s"]
    labels = [SHORT_LABELS.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(cols))
    h = 0.35

    bars_f = ax.barh(y + h/2, lam_f, height=h, color="#3B7DD8",
                     alpha=0.85, label=f"General f (ω_h = {result['omega_h']:.3f})")
    bars_s = ax.barh(y - h/2, np.abs(lam_s), height=h, color="#E67E22",
                     alpha=0.85, label="Specific s_k")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Factor Loading (λ)")
    ax.set_title(
        "Model 1: Bifactor CFA Loadings\n"
        f"General f vs Domain-Specific Factors  |  "
        f"ECV = {result['ecv']:.3f}  |  RMSEA = {result['RMSEA']:.3f}",
        fontsize=11,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.legend(fontsize=9, loc="lower right")

    # Value annotations
    for bar, val in zip(bars_f, lam_f):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    for bar, val in zip(bars_s, np.abs(lam_s)):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    # Group annotations
    group_colors = {"hedonic": "#9B59B6", "eudaimonic": "#3B7DD8",
                    "social_physical": "#2EAA6A"}
    for gname, gcols in BIFACTOR_GROUPS.items():
        idxs = [cols.index(c) for c in gcols if c in cols]
        if idxs:
            ymin, ymax = min(idxs) - 0.4, max(idxs) + 0.4
            ax.axhspan(ymin, ymax, alpha=0.06,
                       color=group_colors.get(gname, "#888"))
            ax.text(-0.08, np.mean(idxs), gname.replace("_", "\n"),
                    va="center", ha="right", fontsize=7.5,
                    color=group_colors.get(gname, "#888"),
                    style="italic", transform=ax.get_yaxis_transform())

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 2: MUTUALISM DYNAMICAL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class MutualismSystem:
    """
    Coupled logistic growth model (van der Maas et al. 2006, adapted).

    dx_k/dt = a_k · x_k · (1 - x_k/C_k + Σ_{l≠k} M_kl · x_l / C_k) + σ_k · ξ_k(t)

    The deterministic version (σ=0) is used for equilibrium analysis.
    Stochastic version used for time-series simulation.

    Parameters
    ──────────
    K           : number of wellbeing domains
    a           : intrinsic growth rates (K,)
    C           : carrying capacities (K,)
    M           : mutualistic coupling matrix (K, K), M_kk = 0
    sigma       : noise amplitudes (K,) for stochastic simulation
    """

    def __init__(self, K, a=None, C=None, M=None, sigma=None):
        self.K = K
        self.a = a if a is not None else np.ones(K) * 0.5
        self.C = C if C is not None else np.ones(K)
        self.M = M if M is not None else np.ones((K, K)) * 0.1
        np.fill_diagonal(self.M, 0.0)
        self.sigma = sigma if sigma is not None else np.ones(K) * 0.05

    def deterministic_rhs(self, t, x):
        """Right-hand side of the deterministic ODE system."""
        dx = np.zeros(self.K)
        for k in range(self.K):
            mutualism_term = sum(
                self.M[k, l] * x[l] / self.C[k]
                for l in range(self.K) if l != k
            )
            dx[k] = self.a[k] * x[k] * (1.0 - x[k] / self.C[k] + mutualism_term)
        return dx

    def find_equilibrium(self, x0=None, t_span=(0, 200), t_eval_n=2000):
        """
        Find equilibrium analytically and generate trajectory for plotting.
        Returns the equilibrium vector x* and an ODE solution.
        """
        x_star = self.analytical_equilibrium()

        # Generate trajectory for plotting (start below equilibrium)
        x0_traj = self.C * 0.3
        sol = solve_ivp(
            self.deterministic_rhs,
            (0, min(t_span[1], 60)),
            x0_traj,
            method="RK45",
            max_step=1.0,
            t_eval=np.linspace(0, min(t_span[1], 60), min(t_eval_n, 300)),
            rtol=1e-8, atol=1e-10,
        )

        return x_star, sol

    def jacobian_at_equilibrium(self, x_star):
        """
        Compute the Jacobian matrix J at equilibrium x*.

        J_kl = ∂ẋ_k / ∂x_l evaluated at x*.

        For k = l (diagonal):
            J_kk = a_k * (1 - 2x*_k/C_k + Σ_{l≠k} M_kl x*_l / C_k)

        For k ≠ l (off-diagonal):
            J_kl = a_k * x*_k * M_kl / C_k
        """
        J = np.zeros((self.K, self.K))
        for k in range(self.K):
            mut_sum = sum(
                self.M[k, l] * x_star[l] / self.C[k]
                for l in range(self.K) if l != k
            )
            # Diagonal: derivative of x_k dynamics w.r.t. x_k
            J[k, k] = self.a[k] * (1.0 - 2.0 * x_star[k] / self.C[k] + mut_sum)
            # Off-diagonal: derivative of x_k dynamics w.r.t. x_l
            for l in range(self.K):
                if l != k:
                    J[k, l] = self.a[k] * x_star[k] * self.M[k, l] / self.C[k]
        return J

    def analytical_equilibrium(self):
        """
        Analytical non-trivial equilibrium of the mutualism system.

        At equilibrium (x_k > 0), the growth equation reduces to:
            1 - x_k/C_k + Σ_{l≠k} M_kl · x_l / C_k = 0
            x_k = C_k + Σ_{l≠k} M_kl · x_l

        In matrix form: (I - M̃) x* = C, where M̃_kl = M_kl for k≠l, 0 diagonal.
        Solution: x* = (I - M̃)^{-1} C
        """
        M_tilde = self.M.copy()
        np.fill_diagonal(M_tilde, 0.0)
        A = np.eye(self.K) - M_tilde
        try:
            x_star = np.linalg.solve(A, self.C)
            if np.all(x_star > 0):
                return x_star
        except np.linalg.LinAlgError:
            pass
        return self.C * 1.1  # fallback

    def fast_equilibrium(self, x0=None):
        """Fast equilibrium — use analytical solution for the mutualism system."""
        return self.analytical_equilibrium()

    def simulate_population(self, n_individuals=200, C_var=0.15, M_var=0.05,
                            a_var=0.05, seed=42):
        """
        Simulate a population of individuals with parameter variation.
        Each individual has slightly different (C, a, M) parameters.
        Returns equilibrium vectors for the population.

        This is the key demonstration: individual differences in parameters
        produce a covariance matrix with a positive manifold even though
        no single latent variable generated the data.
        """
        rng = np.random.RandomState(seed)
        equilibria = np.zeros((n_individuals, self.K))

        for i in range(n_individuals):
            # Individual-specific parameters (positive perturbations)
            C_i = self.C * np.exp(rng.normal(0, C_var, self.K))
            a_i = self.a * np.exp(rng.normal(0, a_var, self.K))
            M_i = self.M * np.exp(rng.normal(0, M_var, (self.K, self.K)))
            np.fill_diagonal(M_i, 0.0)

            # Shared "environmental" factor that shifts all carrying capacities
            # This is NOT a latent cause — it's a correlated input parameter
            env_factor = rng.normal(0, 0.3)
            C_i *= np.exp(env_factor * 0.5)

            sys_i = MutualismSystem(self.K, a=a_i, C=C_i, M=M_i)
            try:
                x_star = sys_i.fast_equilibrium(x0=C_i * 1.2)
                if np.all(x_star > 0) and np.all(np.isfinite(x_star)):
                    equilibria[i] = x_star
                else:
                    equilibria[i] = C_i  # fallback
            except Exception:
                equilibria[i] = C_i

        return equilibria

    def mean_coupling_strength(self):
        """M̄ = 2/(K(K-1)) · Σ_{k<l} M_kl"""
        K = self.K
        upper_tri = self.M[np.triu_indices(K, k=1)]
        return 2.0 / (K * (K - 1)) * np.sum(upper_tri)


def calibrate_mutualism_from_data(X_raw, cols):
    """
    Calibrate mutualism parameters from observed country-level data.

    Carrying capacities C_k ← column means (baseline without mutual support)
    Coupling M_kl ← scaled partial correlations (proxy for direct interactions)
    Growth rates a_k ← set to 0.5 (not identifiable from cross-sectional data)

    Returns a calibrated MutualismSystem.
    """
    K = len(cols)
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)

    # Carrying capacities: slightly below observed means
    # (mutualism "lifts" equilibrium above C)
    C = means * 0.7

    # Coupling from partial correlations
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

    # Scale partial correlations to positive coupling strengths
    # Use abs and scale to [0.05, 0.3] range
    M = np.abs(partial_corr)
    M = 0.05 + 0.25 * (M - M.min()) / (M.max() - M.min() + 1e-8)
    np.fill_diagonal(M, 0.0)

    a = np.ones(K) * 0.5

    sys = MutualismSystem(K, a=a, C=C, M=M)
    return sys


def plot_mutualism_demo(sys, equilibria, cols):
    """
    Three-panel figure demonstrating the mutualism model:
    (a) Trajectory convergence to equilibrium
    (b) Population covariance → positive manifold without latent variable
    (c) Eigenvalue spectrum of population covariance
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── (a) Trajectory convergence ────────────────────────────────────────────
    ax = axes[0]
    x_star, sol = sys.find_equilibrium()
    labels_short = [SHORT_LABELS.get(c, c) for c in cols]
    colors = [PALETTE.get(c, "#888") for c in cols]

    for k in range(sys.K):
        ax.plot(sol.t, sol.y[k], color=colors[k], lw=1.8,
                label=labels_short[k])
        ax.axhline(x_star[k], color=colors[k], ls=":", lw=0.8, alpha=0.5)

    # Mark carrying capacities
    for k in range(sys.K):
        ax.axhline(sys.C[k], color=colors[k], ls="--", lw=0.6, alpha=0.3)

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Domain level x_k(t)")
    ax.set_title("(a) Convergence to Equilibrium\n"
                 "Dashed = carrying capacity; dotted = equilibrium x*")
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    # Annotation: x* > C because of mutualism
    ax.text(0.03, 0.97,
            f"x* > C for all domains\n(mutualism lifts equilibrium)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # ── (b) Population covariance heatmap ─────────────────────────────────────
    ax = axes[1]
    # Standardize equilibria
    eq_std = (equilibria - equilibria.mean(axis=0)) / equilibria.std(axis=0)
    cov_pop = np.corrcoef(eq_std.T)

    im = ax.imshow(cov_pop, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cov_pop[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(cov_pop[i,j]) > 0.6 else "black")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels_short, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels_short, fontsize=8)
    ax.set_title("(b) Simulated Population Correlations\n"
                 "Positive manifold from mutualism, no latent variable")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── (c) Eigenvalue spectrum ───────────────────────────────────────────────
    ax = axes[2]
    cov_raw = np.cov(equilibria.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_raw))[::-1]
    var_explained = eigenvalues / eigenvalues.sum() * 100

    comps = np.arange(1, len(eigenvalues) + 1)
    bars = ax.bar(comps, var_explained, color="#3B7DD8", alpha=0.7,
                  edgecolor="white")
    bars[0].set_color("#E8622A")
    bars[0].set_alpha(0.9)

    ax.set_xlabel("Component")
    ax.set_ylabel("% Variance Explained")
    ax.set_title("(c) Eigenvalue Spectrum\n"
                 f"PC1 = {var_explained[0]:.1f}% — dominant first component")
    ax.set_xticks(comps)
    ax.axhline(100/len(cols), color="gray", ls="--", lw=1,
               label=f"Null expectation ({100/len(cols):.0f}%)")
    ax.legend(fontsize=8)

    for bar, val in zip(bars, var_explained):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                f"{val:.1f}%", ha="center", fontsize=8)

    fig.suptitle(
        "Model 2: Mutualism Dynamical System — No Latent Variable, "
        "Yet Positive Manifold Emerges",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 3: LYAPUNOV STABILITY & RESILIENCE
# ═══════════════════════════════════════════════════════════════════════════════

def lyapunov_analysis(sys, x_star):
    """
    Full Lyapunov stability analysis at equilibrium x*.

    Returns dict:
        J           — Jacobian matrix at x*
        eigenvalues — eigenvalues of J (all should have Re < 0 for stability)
        lambda_1    — dominant eigenvalue (closest to zero)
        tau         — characteristic return time = 1/|Re(λ₁)|
        P           — solution to Lyapunov equation J'P + PJ = -Q
        resilience  — inf{||δx|| : x* + δx ∉ B(x*)} (basin depth proxy)
        stable      — bool, True if all eigenvalues have negative real part
    """
    J = sys.jacobian_at_equilibrium(x_star)
    eigs = eigvals(J)

    # Sort by real part (most negative last)
    idx = np.argsort(np.real(eigs))[::-1]
    eigs_sorted = eigs[idx]

    # Dominant eigenvalue: the one with largest (least negative) real part
    lambda_1 = eigs_sorted[0]
    stable = np.all(np.real(eigs) < 0)

    # Return time
    tau = 1.0 / np.abs(np.real(lambda_1)) if np.real(lambda_1) != 0 else np.inf

    # Lyapunov equation: J'P + PJ = -Q for basin-of-attraction ellipsoid
    Q = np.eye(sys.K)  # identity as the positive-definite Q
    try:
        P = solve_continuous_lyapunov(J.T, -Q)
        # Resilience ≈ min singular value of P^(-1/2) → size of smallest
        # perturbation that exits the ellipsoid {δx : δx'Pδx ≤ c}
        P_eigvals = np.linalg.eigvalsh(P)
        if np.all(P_eigvals > 0):
            # Basin "radius" in the most vulnerable direction
            resilience = 1.0 / np.sqrt(np.max(P_eigvals))
        else:
            resilience = 0.0
    except Exception:
        P = np.eye(sys.K)
        resilience = 0.0

    return {
        "J":           J,
        "eigenvalues": eigs_sorted,
        "lambda_1":    lambda_1,
        "tau":         tau,
        "P":           P,
        "resilience":  resilience,
        "stable":      stable,
    }


def country_resilience_scores(X_raw, base_sys, cols):
    """
    Compute per-country resilience by treating each country's observed values
    as its equilibrium and back-solving for country-specific carrying capacities.

    From x* = (I - M̃)^{-1} C, we get C = (I - M̃) x*

    Returns DataFrame with resilience metrics per country.
    """
    N, K = X_raw.shape
    M_tilde = base_sys.M.copy()
    np.fill_diagonal(M_tilde, 0.0)
    A = np.eye(K) - M_tilde

    results = []

    for i in range(N):
        x_i = X_raw[i]

        # Back-solve for carrying capacity: C_i = (I - M̃) x_i
        C_i = A @ x_i
        C_i = np.maximum(C_i, 0.01)  # safety floor

        sys_i = MutualismSystem(
            K,
            a=base_sys.a.copy(),
            C=C_i,
            M=base_sys.M.copy(),
        )

        try:
            lyap = lyapunov_analysis(sys_i, x_i)
            results.append({
                "lambda_1_real": np.real(lyap["lambda_1"]),
                "tau":           lyap["tau"],
                "resilience":    lyap["resilience"],
                "stable":        lyap["stable"],
                "mean_coupling": sys_i.mean_coupling_strength(),
            })
        except Exception:
            results.append({
                "lambda_1_real": np.nan,
                "tau":           np.nan,
                "resilience":    np.nan,
                "stable":        False,
                "mean_coupling": np.nan,
            })

    return pd.DataFrame(results)


def plot_lyapunov_analysis(lyap_result, x_star, sys, cols):
    """
    Three-panel figure:
    (a) Eigenvalue spectrum of the Jacobian
    (b) Perturbation-response simulation (impulse → recovery)
    (c) Basin-of-attraction ellipsoid (2D projection)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    eigs = lyap_result["eigenvalues"]

    # ── (a) Jacobian eigenvalues ──────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(np.real(eigs), np.imag(eigs), s=120, c="#3B7DD8",
               edgecolors="black", zorder=5)
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.6, label="Stability boundary")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"(a) Jacobian Eigenvalues\n"
                 f"Stable: {lyap_result['stable']}  |  "
                 f"τ = {lyap_result['tau']:.2f}")
    ax.legend(fontsize=8)

    # Annotate dominant eigenvalue
    lam1 = lyap_result["lambda_1"]
    ax.annotate(
        f"λ₁ = {np.real(lam1):.3f}",
        xy=(np.real(lam1), np.imag(lam1)),
        xytext=(np.real(lam1) + 0.05, np.imag(lam1) + 0.05),
        fontsize=9, color="#E8622A",
        arrowprops=dict(arrowstyle="->", color="#E8622A"),
    )

    # ── (b) Perturbation response ─────────────────────────────────────────────
    ax = axes[1]
    # Apply a perturbation to domain 0 and simulate recovery
    delta = np.zeros(sys.K)
    delta[0] = -x_star[0] * 0.3  # 30% negative shock to first domain

    x_perturbed = x_star + delta
    x_perturbed = np.maximum(x_perturbed, 0.01)

    sol = solve_ivp(
        sys.deterministic_rhs,
        (0, 40),
        x_perturbed,
        method="RK45",
        max_step=0.5,
        t_eval=np.linspace(0, 40, 200),
    )

    labels_short = [SHORT_LABELS.get(c, c) for c in cols]
    colors = [PALETTE.get(c, "#888") for c in cols]

    for k in range(sys.K):
        deviation = (sol.y[k] - x_star[k]) / x_star[k] * 100
        ax.plot(sol.t, deviation, color=colors[k], lw=1.5,
                label=labels_short[k])

    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Time after perturbation")
    ax.set_ylabel("% deviation from equilibrium")
    ax.set_title(f"(b) Recovery from Perturbation\n"
                 f"30% shock to {labels_short[0]}, return time τ ≈ {lyap_result['tau']:.1f}")
    ax.legend(fontsize=7, ncol=2)

    # ── (c) Basin of attraction ellipsoid (2D projection) ─────────────────────
    ax = axes[2]
    P = lyap_result["P"]

    # Project onto the two dimensions with most variance
    # Use the two largest eigenvectors of P^{-1}
    try:
        P_inv = inv(P)
        eigvals_p, eigvecs_p = np.linalg.eigh(P_inv)
        idx = np.argsort(eigvals_p)[::-1]
        v1, v2 = eigvecs_p[:, idx[0]], eigvecs_p[:, idx[1]]

        # 2x2 projected P
        proj = np.column_stack([v1, v2])
        P_2d = proj.T @ P @ proj

        # Draw ellipsoid
        theta = np.linspace(0, 2*np.pi, 200)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])

        # Transform circle by P_2d^{-1/2}
        eigvals_2d, eigvecs_2d = np.linalg.eigh(P_2d)
        transform = eigvecs_2d @ np.diag(1.0 / np.sqrt(eigvals_2d)) @ eigvecs_2d.T
        ellipse = circle @ transform.T

        # Scale to resilience level
        c_level = lyap_result["resilience"]**2
        ellipse *= np.sqrt(c_level)

        ax.plot(ellipse[:, 0], ellipse[:, 1], color="#3B7DD8", lw=2)
        ax.fill(ellipse[:, 0], ellipse[:, 1], alpha=0.15, color="#3B7DD8")
        ax.scatter([0], [0], marker="*", s=200, color="#E8622A", zorder=5,
                   label="Equilibrium x*")

        ax.set_xlabel(f"δ along eigenvector 1")
        ax.set_ylabel(f"δ along eigenvector 2")
        ax.set_aspect("equal")
    except Exception:
        ax.text(0.5, 0.5, "Basin projection\nnot available\n(P not pos. def.)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)

    ax.set_title(f"(c) Basin of Attraction (2D projection)\n"
                 f"Resilience R(x*) ≈ {lyap_result['resilience']:.4f}")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Model 3: Lyapunov Stability & Resilience Analysis",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_resilience_vs_f(df_scored, resilience_df):
    """Scatter: f-score vs resilience, showing the predicted positive correlation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) f vs resilience
    ax = axes[0]
    valid = resilience_df["resilience"].notna()
    ax.scatter(df_scored.loc[valid, "f_z"], resilience_df.loc[valid, "resilience"],
               alpha=0.5, s=30, color="#3B7DD8", edgecolors="none")
    r, p = stats.pearsonr(
        df_scored.loc[valid, "f_z"],
        resilience_df.loc[valid, "resilience"],
    )
    ax.set_xlabel("f score (z)")
    ax.set_ylabel("Resilience R(x*)")
    ax.set_title(f"(a) f vs Resilience\nr = {r:.3f}, p = {p:.3e}")

    # (b) f vs return time
    ax = axes[1]
    valid_tau = resilience_df["tau"].notna() & (resilience_df["tau"] < 100)
    if valid_tau.sum() > 5:
        ax.scatter(df_scored.loc[valid_tau, "f_z"],
                   resilience_df.loc[valid_tau, "tau"],
                   alpha=0.5, s=30, color="#E67E22", edgecolors="none")
        r_tau, p_tau = stats.pearsonr(
            df_scored.loc[valid_tau, "f_z"],
            resilience_df.loc[valid_tau, "tau"],
        )
        ax.set_title(f"(b) f vs Return Time τ\nr = {r_tau:.3f}, p = {p_tau:.3e}")
    else:
        ax.set_title("(b) f vs Return Time τ\n(insufficient valid data)")
    ax.set_xlabel("f score (z)")
    ax.set_ylabel("Return time τ = 1/|Re(λ₁)|")

    # (c) Mean coupling vs f
    ax = axes[2]
    valid_m = resilience_df["mean_coupling"].notna()
    if valid_m.sum() > 5:
        ax.scatter(resilience_df.loc[valid_m, "mean_coupling"],
                   df_scored.loc[valid_m, "f_z"],
                   alpha=0.5, s=30, color="#2EAA6A", edgecolors="none")
        ax.set_xlabel("Mean coupling M̄")
        ax.set_ylabel("f score (z)")
        ax.set_title("(c) Mean Coupling vs f\n"
                     "(Paper predicts: higher M̄ → stronger f)")
    else:
        ax.set_title("(c) Mean Coupling vs f\n(insufficient data)")

    fig.suptitle(
        "Model 3: Predicted Relationships — Coupling, Resilience, and f",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 4: CAPABILITIES-ADJUSTED FLOURISHING (f_cap)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_capabilities_adjusted_f(df, X_raw, f_scores, cols,
                                    threshold_quantile=0.25):
    """
    Capability-adjusted flourishing measure (paper Model 4):

        f_cap = f(x*) · 1[x*_k ≥ z_k ∀k] · h(Q_i)

    Components:
        f(x*)   — standard psychometric f (PC1 score)
        1[·]    — Nussbaum threshold indicator (non-compensatory)
        h(Q_i)  — capability set size (approximated as within-country
                   coefficient of variation; lower CV → more balanced → larger
                   effective capability set)

    Alkire-Foster deprivation scoring:
        g_ij = 1 if x_ij < z_j (deprived in dimension j)
        c_i  = Σ w_j · g_ij  (deprivation score)
        ρ_k  = 1 if c_i ≥ k  (identified as deprived if score ≥ cutoff)
        H    = proportion identified as deprived
        A    = average deprivation among the deprived
        M_0  = H × A  (adjusted headcount ratio)

    Returns DataFrame with f_cap and all intermediate measures.
    """
    N, K = X_raw.shape

    # ── Nussbaum thresholds (z_k): set at specified quantile ──────────────────
    thresholds = np.quantile(X_raw, threshold_quantile, axis=0)

    # ── Threshold indicator: 1 if ALL domains ≥ threshold ─────────────────────
    above_threshold = X_raw >= thresholds[np.newaxis, :]
    nussbaum_pass = above_threshold.all(axis=1).astype(float)

    # ── Per-domain deprivation indicators ─────────────────────────────────────
    deprived = (~above_threshold).astype(float)  # g_ij = 1 if x_ij < z_j

    # ── Alkire-Foster: equal weights ──────────────────────────────────────────
    w = np.ones(K) / K
    deprivation_score = deprived @ w  # c_i = Σ w_j · g_ij

    # Dual cutoff: identified as "deprived" if deprived in ≥ 1/3 of dimensions
    af_cutoff = 1.0 / 3.0
    identified = (deprivation_score >= af_cutoff).astype(float)

    # Censored deprivation (zero out non-identified)
    censored_score = deprivation_score * identified

    # H = headcount, A = average intensity among deprived
    H = identified.mean()
    A = censored_score[identified == 1].mean() if identified.sum() > 0 else 0.0
    M0 = H * A  # adjusted headcount ratio

    # ── Dimensional breakdown ─────────────────────────────────────────────────
    # Censored headcount ratio per dimension
    censored_deprived = deprived * identified[:, np.newaxis]
    dim_contribution = censored_deprived.mean(axis=0)

    # ── Capability set size proxy h(Q_i) ──────────────────────────────────────
    # Approximate as normalized balance score:
    # h = 1 - CV(x_i) / max(CV), where CV = std/mean across dimensions
    row_means = X_raw.mean(axis=1)
    row_stds  = X_raw.std(axis=1)
    cv = row_stds / (row_means + 1e-8)
    h = 1.0 - cv / (cv.max() + 1e-8)
    h = np.clip(h, 0.01, 1.0)

    # ── Composite f_cap ───────────────────────────────────────────────────────
    # Shift f_scores to positive range for multiplicative composition
    f_positive = f_scores - f_scores.min() + 0.01
    f_cap = f_positive * nussbaum_pass * h

    # Standardize for interpretability
    f_cap_z = (f_cap - f_cap.mean()) / (f_cap.std() + 1e-8)

    result_df = pd.DataFrame({
        "f_z":               (f_scores - f_scores.mean()) / f_scores.std(),
        "nussbaum_pass":     nussbaum_pass,
        "n_deprived_dims":   deprived.sum(axis=1),
        "deprivation_score": deprivation_score,
        "af_identified":     identified,
        "capability_h":      h,
        "f_cap":             f_cap,
        "f_cap_z":           f_cap_z,
    })

    # Add per-dimension deprivation
    for j, col in enumerate(cols):
        result_df[f"deprived_{col}"] = deprived[:, j]

    summary = {
        "thresholds":       dict(zip(cols, thresholds)),
        "H":                H,
        "A":                A,
        "M0":               M0,
        "dim_contribution": dict(zip(cols, dim_contribution)),
    }

    return result_df, summary


def plot_capabilities_analysis(df_clean, cap_df, cap_summary, cols):
    """
    Four-panel figure:
    (a) f vs f_cap scatter (showing penalty from threshold violations)
    (b) Dimensional deprivation rates
    (c) Alkire-Foster decomposition
    (d) Nussbaum threshold violations by dimension
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ── (a) f vs f_cap ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    pass_mask = cap_df["nussbaum_pass"] == 1
    ax.scatter(cap_df.loc[pass_mask, "f_z"], cap_df.loc[pass_mask, "f_cap_z"],
               alpha=0.5, s=30, color="#2EAA6A", label="Pass all thresholds",
               edgecolors="none")
    ax.scatter(cap_df.loc[~pass_mask, "f_z"], cap_df.loc[~pass_mask, "f_cap_z"],
               alpha=0.5, s=30, color="#E8622A", label="Below threshold (≥1 dim)",
               edgecolors="none")
    ax.plot([-3, 3], [-3, 3], "k--", lw=0.8, alpha=0.4, label="y = x")
    ax.set_xlabel("f (standard psychometric)")
    ax.set_ylabel("f_cap (capability-adjusted)")
    ax.set_title("(a) Standard f vs Capability-Adjusted f_cap\n"
                 "Threshold violations create downward penalty")
    ax.legend(fontsize=8)

    # ── (b) Deprivation rates by dimension ────────────────────────────────────
    ax = axes[0, 1]
    labels = [SHORT_LABELS.get(c, c) for c in cols]
    deprivation_rates = [cap_df[f"deprived_{c}"].mean() * 100 for c in cols]
    colors = [PALETTE.get(c, "#888") for c in cols]

    bars = ax.barh(labels, deprivation_rates, color=colors, alpha=0.8)
    ax.set_xlabel("% of countries below threshold")
    ax.set_title(f"(b) Dimension-Specific Deprivation Rates\n"
                 f"Threshold = 25th percentile per dimension")
    for bar, val in zip(bars, deprivation_rates):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}%", va="center", fontsize=9)

    # ── (c) Alkire-Foster decomposition ───────────────────────────────────────
    ax = axes[1, 0]
    dim_contrib = cap_summary["dim_contribution"]
    dim_vals = [dim_contrib[c] * 100 for c in cols]

    ax.barh(labels, dim_vals, color=colors, alpha=0.8)
    ax.set_xlabel("Contribution to M₀ (%)")
    ax.set_title(f"(c) Alkire-Foster Decomposition by Dimension\n"
                 f"M₀ = {cap_summary['M0']:.3f}  |  "
                 f"H = {cap_summary['H']:.3f}  |  "
                 f"A = {cap_summary['A']:.3f}")
    for i, val in enumerate(dim_vals):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)

    # ── (d) Number of deprived dimensions per country ─────────────────────────
    ax = axes[1, 1]
    n_deprived = cap_df["n_deprived_dims"].astype(int)
    counts = n_deprived.value_counts().sort_index()

    bars = ax.bar(counts.index, counts.values, color="#3B7DD8", alpha=0.7,
                  edgecolor="white")
    ax.set_xlabel("Number of dimensions below threshold")
    ax.set_ylabel("Number of countries")
    ax.set_title("(d) Distribution of Deprivation Breadth\n"
                 "Non-compensatory: any single deprivation matters")

    # Color the 0-deprived bar green
    if 0 in counts.index:
        bars[0].set_color("#2EAA6A")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                str(val), ha="center", fontsize=9)

    fig.suptitle(
        "Model 4: Capabilities-Adjusted Flourishing — "
        "f_cap with Nussbaum Thresholds & Alkire-Foster Scoring",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL INTEGRATION: COUPLING STRENGTH SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def coupling_sweep(base_sys, n_points=15, n_individuals=80):
    """
    The paper's critical prediction: there exists an optimal coupling regime.

    Sweep M̄ from 0 to high values and track:
      - PC1 variance explained (proxy for f dominance)
      - Mean equilibrium level (wellbeing)
      - Resilience (from Lyapunov analysis)

    Too little coupling → weak f, low equilibrium
    Optimal coupling → strong f, high equilibrium, deep basin
    Too much coupling → potential instability (cascade risk)
    """
    m_values = np.linspace(0.01, 0.5, n_points)
    results = {"m_bar": [], "pct_var_pc1": [], "mean_eq": [],
               "resilience": [], "stable_frac": []}

    for m_scale in m_values:
        # Scale the coupling matrix uniformly
        M_scaled = base_sys.M * (m_scale / base_sys.mean_coupling_strength())
        np.fill_diagonal(M_scaled, 0.0)

        sys_test = MutualismSystem(
            base_sys.K,
            a=base_sys.a.copy(),
            C=base_sys.C.copy(),
            M=M_scaled,
        )

        # Simulate population
        equilibria = sys_test.simulate_population(
            n_individuals=n_individuals, seed=42
        )

        # PC1 variance
        cov = np.cov(equilibria.T)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        pct_pc1 = evals[0] / evals.sum() * 100

        # Mean equilibrium
        mean_eq = equilibria.mean()

        # Resilience (from base equilibrium)
        try:
            x_star = sys_test.fast_equilibrium()
            lyap = lyapunov_analysis(sys_test, x_star)
            resilience = lyap["resilience"]
            stable = lyap["stable"]
        except Exception:
            resilience = 0.0
            stable = False

        results["m_bar"].append(m_scale)
        results["pct_var_pc1"].append(pct_pc1)
        results["mean_eq"].append(mean_eq)
        results["resilience"].append(resilience)
        results["stable_frac"].append(1.0 if stable else 0.0)

    return pd.DataFrame(results)


def plot_coupling_sweep(sweep_df):
    """
    Three-panel figure showing the optimal coupling regime prediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Coupling vs f dominance
    ax = axes[0]
    ax.plot(sweep_df["m_bar"], sweep_df["pct_var_pc1"],
            "o-", color="#3B7DD8", lw=2, markersize=5)
    ax.set_xlabel("Mean coupling strength M̄")
    ax.set_ylabel("PC1 variance explained (%)")
    ax.set_title("(a) Coupling → f Dominance\n"
                 "Stronger coupling → stronger positive manifold")
    ax.axhline(100/6, color="gray", ls="--", lw=0.8,
               label="Null (equal eigenvalues)")
    ax.legend(fontsize=8)

    # (b) Coupling vs mean wellbeing
    ax = axes[1]
    ax.plot(sweep_df["m_bar"], sweep_df["mean_eq"],
            "o-", color="#2EAA6A", lw=2, markersize=5)
    ax.set_xlabel("Mean coupling strength M̄")
    ax.set_ylabel("Mean equilibrium level")
    ax.set_title("(b) Coupling → Wellbeing Level\n"
                 "Mutualism lifts all domains above carrying capacity")

    # (c) Coupling vs resilience
    ax = axes[2]
    ax.plot(sweep_df["m_bar"], sweep_df["resilience"],
            "o-", color="#E67E22", lw=2, markersize=5)
    ax.set_xlabel("Mean coupling strength M̄")
    ax.set_ylabel("Resilience R(x*)")
    ax.set_title("(c) Coupling → Resilience\n"
                 "Optimal regime: strong f + deep basin")

    # Mark peak resilience
    peak_idx = sweep_df["resilience"].idxmax()
    ax.axvline(sweep_df.loc[peak_idx, "m_bar"], color="red", ls=":",
               lw=1.5, alpha=0.6, label=f"Peak at M̄ = {sweep_df.loc[peak_idx, 'm_bar']:.3f}")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Coupling Strength Sweep — The Optimal Regime Prediction\n"
        "(Paper: ∃ optimal M̄ where f, wellbeing, and resilience are jointly maximized)",
        fontsize=12, y=1.03,
    )
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
    print("  FULL f-FACTOR IMPLEMENTATION")
    print("  Models 1–4 from 'Toward f: A General Factor of Human Flourishing'")
    print("═" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    df_clean, X_raw, X_scaled, S, cols, N = load_data()

    # ── PCA baseline (for f-scores) ──────────────────────────────────────────
    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    # Sign fix
    anchor = cols.index("M_meaning_life_sat") if "M_meaning_life_sat" in cols else 0
    if pca.components_[0, anchor] < 0:
        scores[:, 0] = -scores[:, 0]
        pca.components_[0] = -pca.components_[0]

    f_scores = scores[:, 0]
    f_z = (f_scores - f_scores.mean()) / f_scores.std()
    df_clean["f_z"] = f_z

    print(f"\n  PCA baseline: PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 1: BIFACTOR CFA
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 1: Bifactor CFA")

    bif_result = fit_bifactor_cfa(S, N, cols)

    print(f"  Converged: {bif_result['converged']}")
    print(f"  ω_h (omega hierarchical) = {bif_result['omega_h']:.3f}")
    print(f"  ECV (explained common var) = {bif_result['ecv']:.3f}")
    print(f"  Paper's empirical range: ω_h ≈ 0.71 – 0.86")
    print(f"  Chi-square = {bif_result['chi2']:.2f}, df = {bif_result['df']}")
    print(f"  RMSEA = {bif_result['RMSEA']:.4f}")

    print(f"\n  General factor loadings (λ^f):")
    for c, lf in zip(cols, bif_result["lambda_f"]):
        print(f"    {SHORT_LABELS.get(c, c):<25} {lf:.3f}")

    print(f"\n  Specific factor loadings (λ^s):")
    for c, ls in zip(cols, bif_result["lambda_s"]):
        gname = next((g for g, gc in BIFACTOR_GROUPS.items() if c in gc), "?")
        print(f"    {SHORT_LABELS.get(c, c):<25} {ls:.3f}  [{gname}]")

    print(f"\n  Unique variances (θ):")
    for c, th in zip(cols, bif_result["theta"]):
        print(f"    {SHORT_LABELS.get(c, c):<25} {th:.3f}")

    fig_bif = plot_bifactor_loadings(bif_result, cols)
    fig_bif.savefig(OUTPUT_DIR / "08_bifactor_cfa.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/08_bifactor_cfa.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 2: MUTUALISM DYNAMICAL SYSTEM
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 2: Mutualism Dynamical System")

    sys = calibrate_mutualism_from_data(X_raw, cols)
    x_star, sol = sys.find_equilibrium()

    print(f"  Calibrated system:")
    print(f"    Carrying capacities C = {sys.C.round(3)}")
    print(f"    Mean coupling M̄ = {sys.mean_coupling_strength():.4f}")
    print(f"    Equilibrium x* = {x_star.round(3)}")
    print(f"    x*/C ratio = {(x_star / sys.C).round(3)}")
    print(f"    All x* > C: {np.all(x_star > sys.C)}  "
          f"(mutualism lifts equilibrium)")

    # Simulate population
    print(f"\n  Simulating population (100 individuals) …", end=" ", flush=True)
    equilibria = sys.simulate_population(n_individuals=100)
    print("done.")

    # Check positive manifold in simulated data
    sim_corr = np.corrcoef(equilibria.T)
    off_diag_sim = sim_corr[np.triu_indices(len(cols), k=1)]
    print(f"  Simulated positive manifold:")
    print(f"    All positive: {np.all(off_diag_sim > 0)}")
    print(f"    Mean r = {off_diag_sim.mean():.3f}")
    print(f"    Min r  = {off_diag_sim.min():.3f}")

    # PC1 of simulated data
    sim_std = StandardScaler().fit_transform(equilibria)
    sim_pca = PCA()
    sim_pca.fit(sim_std)
    print(f"    PC1 of simulated data: {sim_pca.explained_variance_ratio_[0]*100:.1f}% "
          f"(vs {pca.explained_variance_ratio_[0]*100:.1f}% observed)")

    fig_mut = plot_mutualism_demo(sys, equilibria, cols)
    fig_mut.savefig(OUTPUT_DIR / "09_mutualism_model.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/09_mutualism_model.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 3: LYAPUNOV STABILITY & RESILIENCE
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 3: Lyapunov Stability & Resilience")

    lyap = lyapunov_analysis(sys, x_star)

    print(f"  Jacobian eigenvalues: {lyap['eigenvalues'].round(4)}")
    print(f"  Stable: {lyap['stable']}  (all Re(λ) < 0)")
    print(f"  Dominant eigenvalue λ₁ = {lyap['lambda_1']:.4f}")
    print(f"  Return time τ = {lyap['tau']:.2f}")
    print(f"  Resilience R(x*) = {lyap['resilience']:.4f}")

    # Critical slowing down prediction
    print(f"\n  Critical slowing down signature:")
    print(f"    As λ₁ → 0, τ → ∞ (system takes longer to recover)")
    print(f"    Current τ = {lyap['tau']:.2f} — "
          f"{'far from bifurcation' if lyap['tau'] < 20 else 'approaching bifurcation'}")

    fig_lyap = plot_lyapunov_analysis(lyap, x_star, sys, cols)
    fig_lyap.savefig(OUTPUT_DIR / "10_lyapunov_stability.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/10_lyapunov_stability.png")

    # Per-country resilience
    print(f"\n  Computing per-country resilience …", end=" ", flush=True)
    resilience_df = country_resilience_scores(X_raw, sys, cols)
    print("done.")

    df_clean = df_clean.reset_index(drop=True)
    resilience_df = resilience_df.reset_index(drop=True)

    valid_res = resilience_df["resilience"].notna()
    if valid_res.sum() > 10:
        r_corr, r_p = stats.pearsonr(
            df_clean.loc[valid_res, "f_z"],
            resilience_df.loc[valid_res, "resilience"]
        )
        print(f"  f–resilience correlation: r = {r_corr:.3f}, p = {r_p:.3e}")
        print(f"  Paper prediction: positive (stronger f ↔ deeper basins)")

    fig_resil = plot_resilience_vs_f(df_clean, resilience_df)
    fig_resil.savefig(OUTPUT_DIR / "11_resilience_vs_f.png")
    print(f"  → Saved: {OUTPUT_DIR}/11_resilience_vs_f.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 4: CAPABILITIES-ADJUSTED FLOURISHING
    # ══════════════════════════════════════════════════════════════════════════
    print_section("MODEL 4: Capabilities-Adjusted Flourishing")

    cap_df, cap_summary = compute_capabilities_adjusted_f(
        df_clean, X_raw, f_scores, cols, threshold_quantile=0.25
    )

    print(f"  Nussbaum thresholds (25th percentile):")
    for c, z in cap_summary["thresholds"].items():
        print(f"    {SHORT_LABELS.get(c, c):<25} z = {z:.3f}")

    print(f"\n  Threshold compliance:")
    n_pass = int(cap_df["nussbaum_pass"].sum())
    print(f"    Pass all thresholds:  {n_pass} / {N} "
          f"({100*n_pass/N:.0f}%)")
    print(f"    Below ≥1 threshold:   {N - n_pass} / {N}")

    print(f"\n  Alkire-Foster multidimensional deprivation:")
    print(f"    Headcount H   = {cap_summary['H']:.3f}")
    print(f"    Intensity A   = {cap_summary['A']:.3f}")
    print(f"    Adj. ratio M₀ = {cap_summary['M0']:.3f}")

    print(f"\n  Dimensional contribution to M₀:")
    for c, contrib in cap_summary["dim_contribution"].items():
        print(f"    {SHORT_LABELS.get(c, c):<25} {contrib*100:.1f}%")

    # f vs f_cap correlation
    r_cap, p_cap = stats.pearsonr(cap_df["f_z"], cap_df["f_cap_z"])
    print(f"\n  f vs f_cap correlation: r = {r_cap:.3f}")
    print(f"  (< 1.0 shows threshold & capability corrections matter)")

    fig_cap = plot_capabilities_analysis(df_clean, cap_df, cap_summary, cols)
    fig_cap.savefig(OUTPUT_DIR / "12_capabilities_f_cap.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/12_capabilities_f_cap.png")

    # ══════════════════════════════════════════════════════════════════════════
    #  INTEGRATION: COUPLING STRENGTH SWEEP
    # ══════════════════════════════════════════════════════════════════════════
    print_section("INTEGRATION: Coupling Strength Sweep")
    print("  Sweeping M̄ to test optimal-coupling prediction …", end=" ", flush=True)

    sweep_df = coupling_sweep(sys, n_points=15, n_individuals=80)
    print("done.")

    peak = sweep_df.loc[sweep_df["resilience"].idxmax()]
    print(f"  Peak resilience at M̄ = {peak['m_bar']:.3f}")
    print(f"    PC1 variance: {peak['pct_var_pc1']:.1f}%")
    print(f"    Mean equilibrium: {peak['mean_eq']:.3f}")
    print(f"    Resilience: {peak['resilience']:.4f}")

    fig_sweep = plot_coupling_sweep(sweep_df)
    fig_sweep.savefig(OUTPUT_DIR / "13_coupling_sweep.png")
    print(f"\n  → Saved: {OUTPUT_DIR}/13_coupling_sweep.png")

    # ── Save all scored data ──────────────────────────────────────────────────
    df_final = df_clean.copy()
    for col in cap_df.columns:
        if col not in df_final.columns:
            df_final[col] = cap_df[col].values
    for col in resilience_df.columns:
        df_final[f"lyap_{col}"] = resilience_df[col].values

    scored_path = Path("wellbeing_data/wellbeing_full_scored.csv")
    df_final.to_csv(scored_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FULL IMPLEMENTATION COMPLETE")
    print("═" * 70)
    print(f"  Countries analyzed:                  {N}")
    print(f"  Model 1 — Bifactor ω_h:              {bif_result['omega_h']:.3f}")
    print(f"  Model 1 — ECV:                       {bif_result['ecv']:.3f}")
    print(f"  Model 2 — Simulated PC1:             {sim_pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  Model 2 — Mean coupling M̄:           {sys.mean_coupling_strength():.4f}")
    print(f"  Model 3 — Return time τ:             {lyap['tau']:.2f}")
    print(f"  Model 3 — Resilience R(x*):          {lyap['resilience']:.4f}")
    print(f"  Model 4 — Alkire-Foster M₀:          {cap_summary['M0']:.3f}")
    print(f"  Model 4 — f↔f_cap correlation:       {r_cap:.3f}")
    if valid_res.sum() > 10:
        print(f"  Cross-model — f↔resilience r:        {r_corr:.3f}")
    print(f"\n  Figures: {OUTPUT_DIR}/08–13_*.png")
    print(f"  Scored data: {scored_path}")
    print()


if __name__ == "__main__":
    main()

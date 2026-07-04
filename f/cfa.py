"""
cfa.py -- Confirmatory factor analysis engine for the f-construct pipeline.

Two estimation regimes, matching the prereg's estimator plan:

  * DWLS on polychoric correlations ("dwls"): the WLSMV point-estimate
    equivalent. Delta parameterization on the correlation metric: item
    uniquenesses are determined as theta_j = 1 - communality_j (floored at the
    prereg Heywood constant), so only loadings and free factor correlations
    are optimized, against the off-diagonal polychoric moments with inverse
    asymptotic-variance weights.
  * Normal-theory ML on Pearson correlations ("ml"): the prereg's
    'MLR continuous-approximation sensitivity' regime; also the source of
    likelihood-based quantities (AIC/BIC deltas, Vuong casewise test) that
    DWLS cannot supply, and of the multigroup machinery for the sequential
    MGCFA sensitivity in H2a.

PARITY NOTE (report.py repeats this): chi-square-derived indices (CFI, RMSEA)
under WLSMV proper use Muthen's mean-and-variance-adjusted statistic, which
requires the full asymptotic covariance of the polychorics. Here CFI/RMSEA for
DWLS solutions are computed from the ML discrepancy of the implied matrix
against the polychoric matrix -- a labeled approximation. SRMR, loadings,
omega_h, ECV, factor scores, and every *relative* decision quantity
(envelope gaps, delta-BIC, cosines) are exact under this implementation.
Stage 2 includes a lavaan WLSMV parity check on the released data.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize

RNG = np.random.default_rng


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

@dataclass
class Spec:
    """Loading pattern with parameter tying via string keys.

    loadings: (item, factor, key) -- entries sharing a key are constrained
    equal (used for the tau-equivalent hedonic group factor, Alt B, and the
    within-pair ties of the p composite). fixed: (item, factor, value).
    phi_free: (f1, f2, key) free factor correlations; unlisted pairs are 0
    (bifactor orthogonality), diagonal fixed at 1.
    """
    items: list[str]
    factors: list[str]
    loadings: list[tuple[str, str, str]] = field(default_factory=list)
    fixed: list[tuple[str, str, float]] = field(default_factory=list)
    phi_free: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, item: str, factor: str, key: str | None = None):
        self.loadings.append((item, factor, key or f"l|{item}|{factor}"))

    def keys(self) -> list[str]:
        seen: list[str] = []
        for _, _, k in self.loadings:
            if k not in seen:
                seen.append(k)
        for _, _, k in self.phi_free:
            if k not in seen:
                seen.append(k)
        return seen

    def n_free(self) -> int:
        return len(self.keys())


def bifactor_spec(partition: dict[str, list[str]], general: str = "f",
                  hedonic_key: str | None = None,
                  drop_domains: tuple[str, ...] = (),
                  s1_reference: str | None = None) -> Spec:
    """Build the prereg bifactor: f loads everything; each non-hedonic domain
    gets an orthogonal group factor.

    hedonic items load on f only (primary spec). hedonic_key='tau' ties the
    two hedonic items to a common loading on an added 'hedonic' group factor
    (Alt B). drop_domains removes domains and their items (LODO refits, Alt A
    via drop of hedonic handled by caller filtering). s1_reference: name of
    the domain whose group factor is removed (Eid S-1 comparator).
    """
    items = [v for d, vs in partition.items() for v in vs
             if d not in drop_domains]
    domains = [d for d in partition if d not in ("hedonic",) + drop_domains]
    factors = [general] + [d for d in domains if d != s1_reference]
    if hedonic_key == "tau" and "hedonic" in partition and "hedonic" not in drop_domains:
        factors.append("hedonic")
    sp = Spec(items=items, factors=factors)
    for d, vs in partition.items():
        if d in drop_domains:
            continue
        for v in vs:
            sp.add(v, general)
            if d == "hedonic":
                if hedonic_key == "tau":
                    sp.add(v, "hedonic", key="l|hedonic|tau")
            elif d != s1_reference:
                sp.add(v, d)
    return sp


# ---------------------------------------------------------------------------
# Internal parameter plumbing
# ---------------------------------------------------------------------------

class _Map:
    def __init__(self, spec: Spec):
        self.spec = spec
        self.items = spec.items
        self.factors = spec.factors
        self.ii = {v: i for i, v in enumerate(spec.items)}
        self.fi = {f: i for i, f in enumerate(spec.factors)}
        self.keys = spec.keys()
        self.ki = {k: i for i, k in enumerate(self.keys)}
        self.L_entries = [(self.ii[it], self.fi[fa], self.ki[k])
                          for it, fa, k in spec.loadings]
        self.L_fixed = [(self.ii[it], self.fi[fa], v)
                        for it, fa, v in spec.fixed]
        self.P_entries = [(self.fi[a], self.fi[b], self.ki[k])
                          for a, b, k in spec.phi_free]

    def build(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p, q = len(self.items), len(self.factors)
        L = np.zeros((p, q))
        for i, j, k in self.L_entries:
            L[i, j] = theta[k]
        for i, j, v in self.L_fixed:
            L[i, j] = v
        Phi = np.eye(q)
        for a, b, k in self.P_entries:
            Phi[a, b] = Phi[b, a] = theta[k]
        return L, Phi

    def grad_collect(self, dL: np.ndarray, dPhi: np.ndarray) -> np.ndarray:
        g = np.zeros(len(self.keys))
        for i, j, k in self.L_entries:
            g[k] += dL[i, j]
        for a, b, k in self.P_entries:
            g[k] += 2.0 * dPhi[a, b]
        return g


@dataclass
class FitResult:
    spec: Spec
    params: dict[str, float]
    L: np.ndarray
    Phi: np.ndarray
    theta: np.ndarray            # uniquenesses on the analysis metric
    F: float
    converged: bool
    N: float
    heywood: list[str]
    estimator: str
    S: np.ndarray                # matrix the model was fit to

    def implied(self) -> np.ndarray:
        return self.L @ self.Phi @ self.L.T + np.diag(self.theta)

    def loading_vector(self, factor: str) -> np.ndarray:
        j = self.spec.factors.index(factor)
        return self.L[:, j].copy()


# ---------------------------------------------------------------------------
# DWLS / ULS on the correlation metric
# ---------------------------------------------------------------------------

def dwls_weights(V: np.ndarray) -> np.ndarray:
    """Inverse asymptotic-variance weights, normalized to mean 1 (off-diag)."""
    W = 1.0 / np.where(np.isfinite(V) & (V > 0), V, np.nan)
    off = W[np.triu_indices_from(W, 1)]
    W = W / np.nanmean(off)
    W = np.where(np.isfinite(W), W, 1.0)
    np.fill_diagonal(W, 0.0)
    return W


def fit_dwls(spec: Spec, R: np.ndarray, N: float,
             V: np.ndarray | None = None, n_starts: int = 3,
             heywood_floor: float = 0.001, seed: int = 0) -> FitResult:
    """Weighted least squares on off-diagonal polychoric moments.

    F(theta) = 0.5 * sum_{i != j} W_ij (R_ij - Sigma_ij)^2 + communality penalty
    Sigma = L Phi L' ; uniquenesses determined by the unit-diagonal constraint.
    """
    m = _Map(spec)
    idx = [m.ii[v] for v in spec.items]
    S = R[np.ix_(idx, idx)] if R.shape[0] != len(idx) else R
    W = np.ones_like(S) if V is None else dwls_weights(
        V[np.ix_(idx, idx)] if V.shape[0] != len(idx) else V)
    np.fill_diagonal(W, 0.0)
    PEN = 500.0

    def fg(th: np.ndarray):
        L, Phi = m.build(th)
        Sig = L @ Phi @ L.T
        Dres = W * (S - Sig)
        np.fill_diagonal(Dres, 0.0)
        F = 0.25 * float(np.sum(Dres * (S - Sig)))       # 0.5 * sum_{i<j}*2
        dL = -Dres @ L @ Phi
        dPhi = -0.5 * (L.T @ Dres @ L)
        comm = np.diag(Sig)
        over = np.clip(comm - 0.999, 0.0, None)
        F += PEN * float(np.sum(over ** 2))
        if over.any():
            dL += PEN * 4.0 * (over[:, None] * (L @ Phi))
        return F, m.grad_collect(dL, dPhi)

    best = None
    rng = RNG(seed)
    for s in range(n_starts):
        x0 = np.array([0.55 if k.endswith("|f") else 0.40 for k in m.keys])
        x0 = x0 + (0.0 if s == 0 else rng.normal(0, 0.08, x0.size))
        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       bounds=[(-0.999, 0.999)] * x0.size,
                       options=dict(maxiter=2000, ftol=1e-12))
        if best is None or res.fun < best.fun:
            best = res
    L, Phi = m.build(best.x)
    comm = np.diag(L @ Phi @ L.T)
    theta = 1.0 - comm
    heywood = [spec.items[i] for i in np.where(theta < heywood_floor)[0]]
    theta = np.clip(theta, heywood_floor, None)
    return FitResult(spec, dict(zip(m.keys, best.x)), L, Phi, theta,
                     float(best.fun), bool(best.success), N, heywood,
                     "dwls", S)


# ---------------------------------------------------------------------------
# Normal-theory ML (single group)
# ---------------------------------------------------------------------------

def fit_ml(spec: Spec, S: np.ndarray, N: float, n_starts: int = 2,
           seed: int = 0) -> FitResult:
    """ML CFA on a covariance/correlation matrix with free uniquenesses."""
    m = _Map(spec)
    p = len(spec.items)
    nk = len(m.keys)

    def unpack(x):
        L, Phi = m.build(x[:nk])
        theta = x[nk:]
        return L, Phi, theta

    logdetS = np.linalg.slogdet(S)[1]

    def fg(x: np.ndarray):
        L, Phi, theta = unpack(x)
        Sig = L @ Phi @ L.T + np.diag(theta)
        try:
            Si = np.linalg.inv(Sig)
        except np.linalg.LinAlgError:
            return 1e8, np.zeros_like(x)
        sgn, logdet = np.linalg.slogdet(Sig)
        if sgn <= 0:
            return 1e8, np.zeros_like(x)
        F = logdet + float(np.sum(Si * S)) - logdetS - p
        G = Si @ (Sig - S) @ Si
        dL = 2.0 * G @ L @ Phi
        dPhi = L.T @ G @ L
        g = np.concatenate([m.grad_collect(dL, dPhi), np.diag(G)])
        return F, g

    best = None
    rng = RNG(seed)
    sd = np.sqrt(np.diag(S))
    for s in range(n_starts):
        x0 = np.concatenate([
            np.array([0.5 for _ in m.keys]) * sd.mean(),
            0.5 * np.diag(S)])
        if s:
            x0 = x0 * (1 + rng.normal(0, 0.05, x0.size))
        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       bounds=[(-5, 5)] * nk + [(1e-4, None)] * p,
                       options=dict(maxiter=3000, ftol=1e-12))
        if best is None or res.fun < best.fun:
            best = res
    L, Phi, theta = unpack(best.x)
    return FitResult(spec, dict(zip(m.keys, best.x[:nk])), L, Phi, theta,
                     float(best.fun), bool(best.success), N, [], "ml", S)


def ml_loglik(fit: FitResult) -> float:
    """-2 lnL up to the model-independent constant N*p*log(2pi) INCLUDED,
    so AIC/BIC differences across models on the same data are exact."""
    p = len(fit.spec.items)
    Sig = fit.implied()
    logdet = np.linalg.slogdet(Sig)[1]
    tr = float(np.sum(np.linalg.inv(Sig) * fit.S))
    return fit.N * (p * np.log(2 * np.pi) + logdet + tr)


def aic_bic(fit: FitResult) -> tuple[float, float]:
    k = fit.spec.n_free() + len(fit.spec.items)      # loadings/phi + thetas
    m2ll = ml_loglik(fit)
    return m2ll + 2 * k, m2ll + k * np.log(fit.N)


def vuong_test(fitA: FitResult, fitB: FitResult, Z: np.ndarray,
               w: np.ndarray | None = None) -> dict[str, float]:
    """Vuong (1989) non-nested LR z from casewise Gaussian log-densities.

    Z: (N, p) standardized complete-case data both models were fit to.
    w: optional survey weights (prereg: comparators on the same weights);
    the casewise mean/variance of the log-density differences are then
    weight-averaged with effective n = sum(w). Positive z favors model A.
    BIC-adjusted variant also returned.
    """
    def case_ll(fit):
        Sig = fit.implied()
        Si = np.linalg.inv(Sig)
        logdet = np.linalg.slogdet(Sig)[1]
        quad = np.einsum("ij,jk,ik->i", Z, Si, Z)
        return -0.5 * (Z.shape[1] * np.log(2 * np.pi) + logdet + quad)

    d = case_ll(fitA) - case_ll(fitB)
    ww = np.ones(d.size) if w is None else np.asarray(w, float)
    n = float(ww.sum())
    kA = fitA.spec.n_free() + len(fitA.spec.items)
    kB = fitB.spec.n_free() + len(fitB.spec.items)
    mean = float(np.average(d, weights=ww))
    var = float(np.average((d - mean) ** 2, weights=ww))
    sd = np.sqrt(var * d.size / max(d.size - 1, 1))
    z = np.sqrt(n) * mean / sd if sd > 0 else 0.0
    z_bic = np.sqrt(n) * (mean - (kA - kB) * np.log(n) / (2 * n)) / sd \
        if sd > 0 else 0.0
    return dict(z=float(z), z_bic=float(z_bic), mean_diff=mean)


# ---------------------------------------------------------------------------
# Fit indices, reliability, scores
# ---------------------------------------------------------------------------

def fit_indices(fit: FitResult) -> dict[str, float]:
    """CFI / RMSEA (labeled ML-approx for DWLS solutions -- see module note),
    SRMR (exact), chi-square analog T = (N-1) * F_ML(implied vs sample)."""
    S = fit.S
    p = S.shape[0]
    Sig = fit.implied()
    # ML discrepancy of the implied matrix (theta as fitted/determined)
    Si = np.linalg.inv(Sig)
    F_ml = np.linalg.slogdet(Sig)[1] + float(np.sum(Si * S)) \
        - np.linalg.slogdet(S)[1] - p
    F_ml = max(F_ml, 0.0)
    T = (fit.N - 1) * F_ml
    df = p * (p + 1) / 2 - (fit.spec.n_free() + p)   # thetas counted free/determined
    F_base = max(-np.linalg.slogdet(S)[1] + float(np.trace(S)) - p, 1e-12)
    Tb = (fit.N - 1) * F_base
    dfb = p * (p - 1) / 2
    cfi = 1.0 - max(T - df, 0.0) / max(Tb - dfb, T - df, 1e-12)
    rmsea = np.sqrt(max(T - df, 0.0) / max(df * (fit.N - 1), 1e-12))
    res = S - Sig
    iu = np.triu_indices(p, 0)
    srmr = float(np.sqrt(np.mean(res[iu] ** 2)))
    return dict(chisq=float(T), df=float(df), cfi=float(cfi),
                rmsea=float(rmsea), srmr=srmr,
                index_regime="ml_approx" if fit.estimator == "dwls" else "ml")


def omega_ecv(fit: FitResult, general: str = "f") -> dict[str, float]:
    """omega_h, omega_total, ECV and the per-block f-loading SD (Eid floor).

    Computed on the standardized (correlation-metric) solution -- exact for
    DWLS fits; ML fits are standardized first.
    """
    L = fit.L.copy()
    theta = fit.theta.copy()
    d = np.sqrt(np.diag(fit.implied()))
    L = L / d[:, None]
    theta = theta / d ** 2
    gj = fit.spec.factors.index(general)
    lam_g = L[:, gj]
    grp_cols = [j for j in range(L.shape[1]) if j != gj]
    sum_g2 = float(np.sum(lam_g) ** 2)
    sum_s2 = float(sum(np.sum(L[:, j]) ** 2 for j in grp_cols))
    denom = sum_g2 + sum_s2 + float(np.sum(theta))
    ecv_num = float(np.sum(lam_g ** 2))
    ecv_den = ecv_num + float(sum(np.sum(L[:, j] ** 2) for j in grp_cols))
    # per-block mean |f loading| SD across blocks (blocks = group factors + hedonic)
    blocks: dict[str, list[float]] = {}
    for it, fa, _ in fit.spec.loadings:
        if fa != general:
            blocks.setdefault(fa, []).append(abs(lam_g[fit.spec.items.index(it)]))
    loaded = {it for it, fa, _ in fit.spec.loadings if fa != general}
    hed = [abs(lam_g[i]) for i, it in enumerate(fit.spec.items) if it not in loaded]
    if hed:
        blocks["hedonic_only"] = hed
    bmeans = np.array([np.mean(v) for v in blocks.values()])
    return dict(
        omega_h=sum_g2 / denom,
        omega_total=(sum_g2 + sum_s2) / denom,
        ecv=ecv_num / ecv_den,
        f_loading_block_sd=float(np.std(bmeans, ddof=0)),
        mean_abs_group_loading=float(np.mean(
            [abs(L[fit.spec.items.index(it), fit.spec.factors.index(fa)])
             for it, fa, _ in fit.spec.loadings if fa != general])) if grp_cols else 0.0,
    )


def bartlett(fit: FitResult, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bartlett factor scores and their error covariance (L'Th^-1 L)^-1.

    Z: (N, p) items standardized to the analysis metric (continuous
    approximation at the scoring step -- named in the prereg and applied
    deliberately for MIDUS comparability). Returns (scores (N, q), err_cov).
    """
    Ti = np.diag(1.0 / fit.theta)
    M = np.linalg.inv(fit.L.T @ Ti @ fit.L)
    A = M @ fit.L.T @ Ti
    return Z @ A.T, M


def determinacy(fit: FitResult) -> np.ndarray:
    M = np.linalg.inv(fit.L.T @ np.diag(1.0 / fit.theta) @ fit.L)
    return 1.0 / np.sqrt(1.0 + np.diag(M))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


# ---------------------------------------------------------------------------
# Multigroup ML (sequential MGCFA sensitivity for H2a)
# ---------------------------------------------------------------------------

def fit_ml_multigroup(spec: Spec, S_list: list[np.ndarray],
                      N_list: list[float], means_list: list[np.ndarray] | None,
                      level: str = "metric",
                      free_intercepts: list[tuple[str, int]] = ()) -> dict:
    """Metric (equal loadings) or partial-scalar (equal loadings + intercepts,
    with (item, group) exceptions) multigroup ML.

    Continuous-approximation regime by design: this is the prereg's second
    H2a sensitivity, evaluated under Chen (2007) / Rutkowski-Svetina
    thresholds; the categorical Svetina-Rutkowski variant is a Stage 2
    lavaan parity item (see README). Returns pooled F, per-group fits, df.
    """
    G = len(S_list)
    m = _Map(spec)
    p, nk = len(spec.items), len(m.keys)
    scalar = level.startswith("scalar")
    q = len(spec.factors)
    free_set = {(spec.items.index(it), g) for it, g in free_intercepts}
    Ntot = float(sum(N_list))

    # layout: [shared loadings/phi (nk)] [theta g0..gG-1 (G*p)]
    #         [+ scalar: nu shared (p)] [alpha per group>0 (q*(G-1))]
    #         [+ free intercept deviations (len(free_set))]
    n_dev = len(free_set)
    ntot = nk + G * p + (p + q * (G - 1) + n_dev if scalar else 0)
    dev_index = {ig: i for i, ig in enumerate(sorted(free_set))}

    def unpack(x):
        L, Phi = m.build(x[:nk])
        thetas = x[nk:nk + G * p].reshape(G, p)
        nu = alpha = dev = None
        if scalar:
            o = nk + G * p
            nu = x[o:o + p]; o += p
            alpha = np.zeros((G, q))
            alpha[1:] = x[o:o + q * (G - 1)].reshape(G - 1, q); o += q * (G - 1)
            dev = x[o:o + n_dev]
        return L, Phi, thetas, nu, alpha, dev

    def fg(x):
        L, Phi, thetas, nu, alpha, dev = unpack(x)
        F = 0.0
        g_load = np.zeros(nk); g_th = np.zeros((G, p))
        g_nu = np.zeros(p); g_al = np.zeros((G, q)); g_dev = np.zeros(n_dev)
        for g in range(G):
            Sg = S_list[g]
            Sig = L @ Phi @ L.T + np.diag(thetas[g])
            try:
                Si = np.linalg.inv(Sig)
            except np.linalg.LinAlgError:
                return 1e8, np.zeros_like(x)
            sgn, logdet = np.linalg.slogdet(Sig)
            if sgn <= 0:
                return 1e8, np.zeros_like(x)
            wg = N_list[g] / Ntot
            Fg = logdet + float(np.sum(Si * Sg)) \
                - np.linalg.slogdet(Sg)[1] - p
            Gmat = Si @ (Sig - Sg) @ Si
            if scalar:
                mu = nu.copy()
                for (i, gg), di in dev_index.items():
                    if gg == g:
                        mu[i] = mu[i] + dev[di]
                mu = mu + L @ alpha[g]
                delta = means_list[g] - mu
                Fg += float(delta @ Si @ delta)
                Gmat = Gmat - Si @ np.outer(delta, delta) @ Si
                Sid = Si @ delta
                g_nu += wg * (-2.0 * Sid)
                g_al[g] += wg * (-2.0 * (L.T @ Sid))
                for (i, gg), di in dev_index.items():
                    if gg == g:
                        g_dev[di] += wg * (-2.0 * Sid[i])
                dL_mean = -2.0 * np.outer(Sid, alpha[g])
            else:
                dL_mean = 0.0
            F += wg * Fg
            g_load += wg * m.grad_collect(2.0 * Gmat @ L @ Phi + dL_mean,
                                          L.T @ Gmat @ L)
            g_th[g] += wg * np.diag(Gmat)
        grad = [g_load, g_th.ravel()]
        if scalar:
            grad += [g_nu, g_al[1:].ravel(), g_dev]
        return F, np.concatenate(grad)

    x0 = np.zeros(ntot)
    x0[:nk] = 0.5
    for g in range(G):
        x0[nk + g * p: nk + (g + 1) * p] = 0.5 * np.diag(S_list[g])
    if scalar:
        x0[nk + G * p: nk + G * p + p] = np.concatenate(
            [means_list[g][None] for g in range(G)]).mean(0)
    bounds = [(-5, 5)] * nk + [(1e-4, None)] * (G * p)
    if scalar:
        bounds += [(-10, 10)] * (p + q * (G - 1) + n_dev)
    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=4000, ftol=1e-11))
    L, Phi, thetas, nu, alpha, dev = unpack(res.x)
    # pooled chi-square analog and indices
    T = (Ntot - G) * res.fun
    n_mom = G * p * (p + 1) / 2 + (G * p if scalar else 0)
    df = n_mom - (ntot if scalar else nk + G * p)
    Fb = 0.0
    for g in range(G):
        Fb += (N_list[g] / Ntot) * max(
            -np.linalg.slogdet(S_list[g])[1] + np.trace(S_list[g]) - p, 1e-12)
    Tb = (Ntot - G) * Fb
    dfb = G * p * (p - 1) / 2
    cfi = 1 - max(T - df, 0) / max(Tb - dfb, T - df, 1e-12)
    rmsea = np.sqrt(G) * np.sqrt(max(T - df, 0) / max(df * (Ntot - G), 1e-12))
    return dict(F=float(res.fun), converged=bool(res.success), L=L, Phi=Phi,
                thetas=thetas, nu=nu, alpha=alpha, cfi=float(cfi),
                rmsea=float(rmsea), df=float(df), T=float(T))

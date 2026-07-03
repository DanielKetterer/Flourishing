"""
polychoric.py -- Weighted polychoric correlations for ordinal GFS items.

Two-step estimator (Olsson 1979): thresholds fixed at inverse-normal marginal
cumulative proportions, then rho maximizes the bivariate-normal likelihood of
the (survey-weighted) contingency table. Because the likelihood depends on the
data only through the collapsed K x K table, cost is independent of N -- the
GFS Wave 1 cross-section (~200k) is as cheap as the smoke sample.

Also provides the per-pair asymptotic variance of rho (inverse observed
information from the pairwise likelihood at the optimum), used as inverse
DWLS weights in cfa.py -- this is what makes 'DWLS on polychorics' the
WLSMV point-estimate equivalent named in the prereg.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

_CLIP = 8.0  # |threshold| clip: Phi(8) is 1 to ~6e-16, safe for +-inf handling
_GL_N = 24
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_gl_x = (_gl_x + 1.0) / 2.0   # map to [0, 1]
_gl_w = _gl_w / 2.0


def bvn_cdf(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """P(X <= h, Y <= k) for standard bivariate normal, vectorized over h, k.

    Uses the single-integral identity
        Phi2(h,k,rho) = Phi(h)Phi(k)
                        + (1/2pi) * int_0^rho exp(-(h^2 - 2 r h k + k^2)
                                                   / (2(1-r^2))) / sqrt(1-r^2) dr
    evaluated by Gauss-Legendre quadrature (24 nodes; abs error ~1e-10 for
    |rho| <= 0.999). h, k broadcast; rho scalar.
    """
    h = np.clip(np.asarray(h, float), -_CLIP, _CLIP)
    k = np.clip(np.asarray(k, float), -_CLIP, _CLIP)
    rho = float(np.clip(rho, -0.9999, 0.9999))
    base = norm.cdf(h) * norm.cdf(k)
    if rho == 0.0:
        return base
    r = rho * _gl_x                                  # (Q,)
    one_m = 1.0 - r * r
    hh = h[..., None]; kk = k[..., None]
    expo = -(hh * hh - 2.0 * r * hh * kk + kk * kk) / (2.0 * one_m)
    integ = np.exp(expo) / np.sqrt(one_m)
    return base + rho * (integ @ _gl_w) / (2.0 * np.pi)


def thresholds_from_margin(counts: np.ndarray) -> np.ndarray:
    """Inverse-normal thresholds (length K-1) from weighted category counts."""
    p = counts / counts.sum()
    cum = np.cumsum(p)[:-1]
    return norm.ppf(np.clip(cum, 1e-10, 1 - 1e-10))


def _cell_probs(tx: np.ndarray, ty: np.ndarray, rho: float) -> np.ndarray:
    """K1 x K2 rectangle probabilities from thresholds via CDF differencing."""
    gx = np.concatenate(([-_CLIP], tx, [_CLIP]))
    gy = np.concatenate(([-_CLIP], ty, [_CLIP]))
    H, K = np.meshgrid(gx, gy, indexing="ij")
    C = bvn_cdf(H, K, rho)
    P = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    return np.clip(P, 1e-12, None)


def polychoric_pair(table: np.ndarray) -> tuple[float, float]:
    """(rho_hat, asy_var) from a weighted contingency table.

    asy_var is 1 / observed information of the pairwise log-likelihood in rho
    at the optimum (thresholds fixed), by central second differences. Relative
    magnitudes are what DWLS consumes, so the fixed-threshold approximation
    (Olsson two-step) is the standard one.
    """
    table = np.asarray(table, float)
    rows = table.sum(1) > 0
    cols = table.sum(0) > 0
    table = table[rows][:, cols]                    # drop empty categories
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, np.inf
    tx = thresholds_from_margin(table.sum(1))
    ty = thresholds_from_margin(table.sum(0))

    def nll(r: float) -> float:
        return -float(np.sum(table * np.log(_cell_probs(tx, ty, r))))

    res = minimize_scalar(nll, bounds=(-0.995, 0.995), method="bounded",
                          options=dict(xatol=1e-6))
    rho = float(res.x)
    h = 1e-4
    d2 = (nll(min(rho + h, 0.995)) - 2 * nll(rho) + nll(max(rho - h, -0.995))) / h**2
    var = 1.0 / d2 if d2 > 0 else np.inf
    return rho, max(var, 1e-12)


def weighted_table(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                   kx: int, ky: int) -> np.ndarray:
    """Survey-weighted K x K contingency table; NaNs pairwise-deleted.

    NOTE (Missing Data section of the prereg): pairwise-present polychorics
    are NOT the registered missing-data model -- run this on multiply-imputed
    data (data_io.multiply_impute) and pool. The pairwise path exists only so
    single-imputation and diagnostic calls are well-defined.
    """
    m = ~(np.isnan(x) | np.isnan(y))
    xi = x[m].astype(int); yi = y[m].astype(int); wi = w[m]
    tab = np.zeros((kx, ky))
    np.add.at(tab, (xi, yi), wi)
    return tab


def polychoric_matrix(data: np.ndarray, w: np.ndarray, n_cats: list[int]
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(R, V): polychoric correlation matrix and matching asy-variance matrix.

    data: (N, p) integer-coded ordinal (NaN allowed), w: (N,) weights.
    """
    p = data.shape[1]
    R = np.eye(p)
    V = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(i + 1, p):
            tab = weighted_table(data[:, i], data[:, j], w, n_cats[i], n_cats[j])
            r, v = polychoric_pair(tab)
            R[i, j] = R[j, i] = r
            V[i, j] = V[j, i] = v
    return nearest_pd_corr(R), V


def nearest_pd_corr(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Eigenvalue-clipped positive-definite repair, rescaled to unit diagonal."""
    vals, vecs = np.linalg.eigh((R + R.T) / 2)
    if vals.min() > eps:
        return R
    vals = np.clip(vals, eps, None)
    A = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


def pearson_corr_weighted(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted Pearson correlation with pairwise-complete handling."""
    p = X.shape[1]
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            m = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if m.sum() < 10:
                continue
            wi = w[m]
            xi = X[m, i]; yj = X[m, j]
            mx = np.average(xi, weights=wi); my = np.average(yj, weights=wi)
            sx = np.sqrt(np.average((xi - mx) ** 2, weights=wi))
            sy = np.sqrt(np.average((yj - my) ** 2, weights=wi))
            if sx <= 0 or sy <= 0:
                continue
            R[i, j] = R[j, i] = np.average((xi - mx) * (yj - my), weights=wi) / (sx * sy)
    return nearest_pd_corr(R)

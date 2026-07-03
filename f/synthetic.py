"""
synthetic.py -- GFS-format synthetic data under the two-channel mixture DGP.

Three consumers:
  1. End-to-end smoke tests of the pipeline before any GFS data access
     (the code-deposit verification run).
  2. The H2a Monte Carlo operating-characteristics validation of per-factor
     alignment (Stage 2 addendum item iii): invariance violations injected at
     pre-registered fractions/magnitudes.
  3. The H4 Stage 2 machinery: the indirect-inference correction map for w_L
     and the H4a parametric-bootstrap cross-fitting both simulate from this
     DGP (items vi, vii).

DGP (documented so the Stage 2 addendum can either adopt or replace it with
the authoritative v8 specification -- the module boundary is exactly here):

  latent channel   : f_t AR(1) with persistence phi (stationary var 1);
                     domain specifics s_{d,t} iid over waves.
  mutualism channel: 6-vector d_t, VAR(1) with A = expm(c J dt),
                     J = -(1/2) Sigma_d^{-1} (GLV linearization at the fixed
                     point, matching the MIDUS Lyapunov-inversion form),
                     c scaled so the slowest relaxation timescale is tau.
                     Sigma_d = (1-r) I + r 11', r = 0.5 (positive manifold).
  trait channel    : scalar tau_i per person, item weight sqrt(mu_ratio*h2_j)
                     -- mu_ratio is the trait/dynamic common-variance ratio,
                     the quantity the third anchor B^T = I absorbs.
  mixture          : common_jt = sqrt(w_L) latent_jt + sqrt(1-w_L) mutual_jt
                     with both channels normalized to per-item communality
                     h2_j, so cross-sectional structure is w_L-invariant --
                     the observational-equivalence property H4 is built to
                     break with longitudinal information.
  measurement      : + unique N(0, 1-h2_j) iid; discretized to the item's
                     category count through skewed normal cutpoints.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import expm
from . import config as C

RNG = np.random.default_rng


def _mutualism_operator(tau: float, dt: float, r: float = 0.5,
                        k: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, Sigma_d, Sigma_eps) for the domain VAR; A = expm(c J dt)."""
    Sigma_d = (1 - r) * np.eye(k) + r * np.ones((k, k))
    J = -0.5 * np.linalg.inv(Sigma_d)
    lam_max = np.linalg.eigvalsh(J).max()          # least negative
    c = -1.0 / (tau * lam_max)
    A = expm(c * J * dt)
    Sigma_eps = Sigma_d - A @ Sigma_d @ A.T        # PD: J, Sigma_d commute
    return A, Sigma_d, Sigma_eps


def true_loadings(items: list[C.Item], seed: int = 7) -> dict:
    rng = RNG(seed)
    lam_f, lam_g = {}, {}
    for it in items:
        lam_f[it.var] = rng.uniform(0.55, 0.75)
        lam_g[it.var] = 0.0 if it.domain == "hedonic" else rng.uniform(0.35, 0.55)
    return dict(lam_f=lam_f, lam_g=lam_g)


def _discretize(y: np.ndarray, n_cat: int, rng) -> np.ndarray:
    """Skewed cutpoints on the standardized scale (wellbeing-style skew)."""
    ks = np.arange(1, n_cat)
    cum = (ks / n_cat) ** 0.75
    from scipy.stats import norm
    cuts = norm.ppf(cum)
    return np.digitize(y, cuts).astype(float)


def simulate_panel(n_per_country: int = 350, countries: list[str] | None = None,
                   waves: int = 3, w_L: float = 0.6, mu_ratio: float = 0.5,
                   phi: float | None = None, tau: float | None = None,
                   invariance: dict | None = None, dropout: bool = True,
                   item_missing: float = 0.02, seed: int = 1
                   ) -> tuple[pd.DataFrame, dict]:
    """Full GFS-format long dataframe + ground-truth dict.

    invariance: dict(frac=..., magnitude=..., factor='f'|'group') injects
    loading + intercept violations in a random frac of (item, country) cells
    (H2a MC harness). Violated cells are returned in truth['violated'].
    """
    rng = RNG(seed)
    phi = C.H4["phi_midus"] if phi is None else phi
    tau = C.H4["tau_midus"] if tau is None else tau
    countries = countries or C.GFS_COUNTRIES
    Gn = len(countries)
    items = C.F_ITEMS
    p = len(items)
    tl = true_loadings(items)
    lam_f = np.array([tl["lam_f"][it.var] for it in items])
    lam_g = np.array([tl["lam_g"][it.var] for it in items])
    h2 = lam_f ** 2 + lam_g ** 2
    dom_idx = {d: i for i, d in enumerate(C.DOMAINS)}
    A, Sd, Se = _mutualism_operator(tau, C.H4["delta_t"])
    Le = np.linalg.cholesky(Se)
    dbar_sd = np.sqrt(np.ones(6) @ Sd @ np.ones(6)) / 6.0

    # country effects and (optional) invariance violations
    alpha_c = rng.normal(0, 0.30, Gn)                       # country f means
    delta_cj = rng.normal(0, 0.12, (Gn, p))                 # item intercept shifts
    lam_mult = np.ones((Gn, p)); nu_shift = np.zeros((Gn, p))
    violated = []
    if invariance and invariance.get("frac", 0) > 0:
        vr = RNG(invariance.get("seed", seed + 99))
        cells = [(g, j) for g in range(Gn) for j in range(p)]
        n_v = int(round(invariance["frac"] * len(cells)))
        for g, j in [cells[i] for i in vr.choice(len(cells), n_v, replace=False)]:
            s = vr.choice([-1, 1])
            lam_mult[g, j] = 1.0 + s * invariance["magnitude"]
            nu_shift[g, j] = s * invariance["magnitude"]
            violated.append((countries[g], items[j].var))

    rows = []
    truth_scores = []
    for g, ctry in enumerate(countries):
        n = n_per_country
        # trait, demographics
        tau_i = rng.normal(0, 1, n)
        age = rng.integers(18, 81, n)
        sex = rng.integers(1, 3, n)
        edu = rng.integers(1, 4, n)
        trait_shift = 0.10 * (edu - 2) - 0.005 * (age - 49)
        tau_i = tau_i + trait_shift
        # latent channel
        f = np.empty((waves, n)); f[0] = rng.normal(0, 1, n)
        for t in range(1, waves):
            f[t] = phi * f[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, 1, n)
        s = rng.normal(0, 1, (waves, 6, n))                 # domain specifics, iid t
        # mutualism channel
        d = np.empty((waves, 6, n))
        d[0] = np.linalg.cholesky(Sd) @ rng.normal(0, 1, (6, n))
        for t in range(1, waves):
            d[t] = A @ d[t - 1] + Le @ rng.normal(0, 1, (6, n))
        dbar = d.mean(1) / dbar_sd                          # (waves, n)
        # p latent (for the psychopathology composite + realism of dropout)
        gbar = np.sqrt(w_L) * f + np.sqrt(1 - w_L) * dbar
        v = np.empty((waves, n)); v[0] = rng.normal(0, 1, n)
        for t in range(1, waves):
            v[t] = 0.8 * v[t - 1] + np.sqrt(1 - 0.64) * rng.normal(0, 1, n)
        p_lat = -0.6 * gbar + 0.8 * v

        # survey design
        strata = rng.integers(0, 10, n)
        psu = strata * 5 + rng.integers(0, 5, n)
        w = np.exp(rng.normal(0, 0.35, n)); w = w / w.mean()

        present = np.ones((waves, n), bool)
        if dropout:
            keep2 = rng.random(n) < 1 / (1 + np.exp(-(1.30 + 0.15 * gbar[0])))
            keep3 = keep2 & (rng.random(n) < 1 / (1 + np.exp(-(0.80 + 0.15 * gbar[0]))))
            present[1] = keep2; present[2 % waves] = keep3 if waves > 2 else present[-1]

        for t in range(waves):
            X = np.empty((n, p))
            for j, it in enumerate(items):
                if it.domain == "hedonic":
                    lat = lam_f[j] * f[t]
                    mut = np.sqrt(h2[j]) * dbar[t]
                else:
                    di = dom_idx[it.domain]
                    lat = lam_f[j] * f[t] + lam_g[j] * s[t, di]
                    mut = np.sqrt(h2[j]) * d[t, di]
                common = np.sqrt(w_L) * lat + np.sqrt(1 - w_L) * mut
                yj = (lam_mult[g, j] * common
                      + np.sqrt(mu_ratio * h2[j]) * tau_i
                      + alpha_c[g] * lam_f[j] + delta_cj[g, j] + nu_shift[g, j]
                      + rng.normal(0, np.sqrt(1 - h2[j]), n))
                X[:, j] = _discretize(yj, it.n_cat, rng)
            # H3 outcome pseudo-item (financial domain, not an f indicator)
            fin = dom_idx["financial"]
            se_lat = 0.60 * f[t] + 0.45 * s[t, fin]
            se_mut = np.sqrt(0.60 ** 2 + 0.45 ** 2) * d[t, fin]
            subj = (np.sqrt(w_L) * se_lat + np.sqrt(1 - w_L) * se_mut
                    + np.sqrt(mu_ratio * 0.56) * tau_i
                    + rng.normal(0, np.sqrt(1 - 0.56), n))
            SUBJ = _discretize(subj, 11, rng)
            # p items (4-category PHQ-4 style; reversed later: high = worse)
            P = np.empty((n, 4))
            for jj, itp in enumerate(C.P_ITEMS):
                grp = rng.normal(0, 1, n)
                yp = 0.70 * p_lat[t] + 0.35 * grp + rng.normal(0, 0.62, n)
                P[:, jj] = _discretize(yp, itp.n_cat, rng)
            m = present[t]
            df = pd.DataFrame({
                C.ADMIN["id"]: np.arange(n) + g * 10 ** 6,
                C.ADMIN["wave"]: t + 1,
                C.ADMIN["country"]: ctry,
                C.ADMIN["weight"]: w,
                C.ADMIN["strata"]: strata + g * 100,
                C.ADMIN["psu"]: psu + g * 1000,
                C.H3_CONTROLS["age"]: age,
                C.H3_CONTROLS["sex"]: sex,
                C.H3_CONTROLS["education"]: edu,
                C.H3_OUTCOMES["subj_econ_position"]["var"]: SUBJ,
            })
            for j, it in enumerate(items):
                col = X[:, j]
                if it.reverse:                              # ship codebook-oriented
                    col = (it.n_cat - 1) - col
                df[it.var] = col
            for jj, itp in enumerate(C.P_ITEMS):
                df[itp.var] = P[:, jj]
            if item_missing > 0:
                for cvar in [it.var for it in items] + [it.var for it in C.P_ITEMS]:
                    miss = rng.random(n) < item_missing
                    df.loc[miss, cvar] = np.nan
            rows.append(df[m])
            truth_scores.append(pd.DataFrame({
                C.ADMIN["id"]: df[C.ADMIN["id"]], C.ADMIN["wave"]: t + 1,
                "f_true": np.sqrt(w_L) * f[t] + np.sqrt(1 - w_L) * dbar[t]}))

    long = pd.concat(rows, ignore_index=True)
    truth = dict(w_L=w_L, mu_ratio=mu_ratio, phi=phi, tau=tau,
                 lam_f=lam_f, lam_g=lam_g, h2=h2, violated=violated,
                 alpha_c=dict(zip(countries, alpha_c)),
                 scores=pd.concat(truth_scores, ignore_index=True))
    return long, truth


def simulate_items_fast(N: int, w_L: float, mu_ratio: float,
                        phi: float, tau: float, waves: int = 3,
                        seed: int = 0, discretize: bool = True) -> np.ndarray:
    """(waves, N, p) item panel only -- fast path for the H4 Monte Carlo
    (correction map, H4a parametric bootstrap). Single pooled population,
    v8 discretized measurement layer applied when discretize=True."""
    rng = RNG(seed)
    items = C.F_ITEMS
    p = len(items)
    tl = true_loadings(items)
    lam_f = np.array([tl["lam_f"][it.var] for it in items])
    lam_g = np.array([tl["lam_g"][it.var] for it in items])
    h2 = lam_f ** 2 + lam_g ** 2
    dom_idx = {d: i for i, d in enumerate(C.DOMAINS)}
    A, Sd, Se = _mutualism_operator(tau, C.H4["delta_t"])
    Le = np.linalg.cholesky(Se)
    dbar_sd = np.sqrt(np.ones(6) @ Sd @ np.ones(6)) / 6.0
    tau_i = rng.normal(0, 1, N)
    f = np.empty((waves, N)); f[0] = rng.normal(0, 1, N)
    for t in range(1, waves):
        f[t] = phi * f[t - 1] + np.sqrt(max(1 - phi ** 2, 1e-9)) * rng.normal(0, 1, N)
    d = np.empty((waves, 6, N))
    d[0] = np.linalg.cholesky(Sd) @ rng.normal(0, 1, (6, N))
    for t in range(1, waves):
        d[t] = A @ d[t - 1] + Le @ rng.normal(0, 1, (6, N))
    dbar = d.mean(1) / dbar_sd
    s = rng.normal(0, 1, (waves, 6, N))
    X = np.empty((waves, N, p))
    for t in range(waves):
        for j, it in enumerate(items):
            if it.domain == "hedonic":
                lat = lam_f[j] * f[t]
                mut = np.sqrt(h2[j]) * dbar[t]
            else:
                di = dom_idx[it.domain]
                lat = lam_f[j] * f[t] + lam_g[j] * s[t, di]
                mut = np.sqrt(h2[j]) * d[t, di]
            y = (np.sqrt(w_L) * lat + np.sqrt(1 - w_L) * mut
                 + np.sqrt(mu_ratio * h2[j]) * tau_i
                 + rng.normal(0, np.sqrt(1 - h2[j]), N))
            X[t, :, j] = _discretize(y, it.n_cat, rng) if discretize else y
    return X

"""
h4_dynamics.py -- H4 block: latent-factor vs mutualism channel for the GFS
within-person propagator.

Identity (Statistical Models, H4): with x_t = mu_i + z_t and z_{t+1} = B z_t
+ innovation, Cov(x_{t+k}, x_t) = V_mu + B^k Sigma_z, so person random
intercepts cancel algebraically in the differences:

    B_obs = (Sigma_2 - Sigma_1)(Sigma_1 - Sigma_0)^(-1)

computed on wave-centered indicator panels; Tikhonov-regularized inversion
with GCV, flagged, when the smallest singular value of Sigma_1 - Sigma_0
falls below 1e-3 of the largest.

Stage 1: inverse-variance-weighted off-diagonal Frobenius mixture of the
symmetrized B_obs against the MIDUS-anchored latent channel
B^L = phi Lambda Lambda' / (Lambda' Lambda), phi = 0.957, and the mutualism
channel B^M = expm(c J dt), J = -(1/2) Sigma_W1^(-1) rescaled to tau = 6.08y;
sparse-J graphical-lasso variant; three-anchor extension adding B^T = I on
the simplex. Entry variances (hence weights and the w_hat distribution) come
from the country-level wild cluster bootstrap with Rademacher perturbations
of centered country moment contributions -- the moment-estimator analogue of
the MacKinnon-Webb (2018) restricted-residual wild cluster scheme; the naive
person-level cluster bootstrap is the pre-registered sensitivity.

Stage 2: indirect-inference bias-correction map w_corrected =
f^-1(w_hat, mu_hat) from the MIDUS-calibrated Monte Carlo over the
(w_L x mu_ratio) grid under the discretized measurement layer, with
wild-cluster-bootstrap CIs scaled by c_inflate = 1.50; clipping-zone and
high-w_L/high-tau results reported as partial-identification intervals.

H4a: parametric-bootstrap cross-fitting (Wagenmakers et al. 2004) between
the fitted pure-latent and pure-mutualism single-channel models; verdicts
(Latent / Mutualism / Indistinguishable) from (Delta_fit, kappa) at
thresholds calibrated to fixed per-truth error rates; confusion matrix over
the calibration grid. Indistinguishable is a live outcome.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.isotonic import IsotonicRegression

from . import config as C
from . import cfa
from .synthetic import simulate_items_fast

RNG = np.random.default_rng
FN = [it.var for it in C.F_ITEMS]


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def panel_moments(panel: pd.DataFrame) -> dict:
    """Weighted, wave-centered cross-covariances on the complete 3-wave
    panel, with per-country contributions for the wild cluster bootstrap.
    Base-t convention: Sigma_k = Cov(x_{1+k}, x_1); C22 kept for the
    cross-wave homogeneity diagnostic."""
    cols = {t: [f"{v}_w{t}" for v in FN] for t in (1, 2, 3)}
    X = {t: panel[cols[t]].to_numpy(float) for t in (1, 2, 3)}
    w = panel[C.ADMIN["weight"]].to_numpy(float)
    m = np.ones(len(panel), bool)
    for t in (1, 2, 3):
        m &= ~np.isnan(X[t]).any(1)
    w = w[m]
    ctry = panel[C.ADMIN["country"]].to_numpy()[m]
    W = w.sum()
    Xc = {}
    for t in (1, 2, 3):
        Xt = X[t][m]
        Xc[t] = Xt - np.average(Xt, 0, weights=w)      # pooled wave centering

    def xcov(A, Bm, wi):
        return (A * wi[:, None]).T @ Bm / wi.sum()

    per_country = {}
    for c in np.unique(ctry):
        i = ctry == c
        wc = w[i]
        per_country[c] = dict(
            pi=float(wc.sum() / W),
            S0=xcov(Xc[1][i], Xc[1][i], wc),
            S1=xcov(Xc[2][i], Xc[1][i], wc),
            S2=xcov(Xc[3][i], Xc[1][i], wc),
            S22=xcov(Xc[2][i], Xc[2][i], wc))

    def pool(key):
        return sum(d["pi"] * d[key] for d in per_country.values())

    return dict(S0=pool("S0"), S1=pool("S1"), S2=pool("S2"), S22=pool("S22"),
                per_country=per_country, n=int(m.sum()), W=float(W),
                Xc=Xc, mask=m, weights=w, country=ctry)


def b_obs_from(S0, S1, S2, cutoff=None) -> tuple[np.ndarray, dict]:
    cutoff = cutoff or C.H4["tikhonov_rel_cutoff"]
    D1, D2 = S1 - S0, S2 - S1
    sv = np.linalg.svd(D1, compute_uv=False)
    flag = sv.min() / sv.max() < cutoff
    if not flag:
        B = D2 @ np.linalg.inv(D1)
        lam = 0.0
    else:
        p = D1.shape[0]
        lams = np.logspace(-6, 0, 25) * sv.max() ** 2
        best = None
        for lam in lams:
            Minv = np.linalg.inv(D1 @ D1.T + lam * np.eye(p))
            Bl = D2 @ D1.T @ Minv
            H = D1.T @ Minv @ D1
            gcv = np.linalg.norm(D2 - Bl @ D1, "fro") ** 2 \
                / max(1 - np.trace(H) / p, 1e-6) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, Bl, lam)
        _, B, lam = best
    return B, dict(tikhonov=bool(flag), lam=float(lam),
                   cond=float(sv.max() / sv.min()))


def s_skew(B: np.ndarray) -> float:
    sym = (B + B.T) / 2
    skw = (B - B.T) / 2
    return float(np.linalg.norm(skw, "fro") / max(np.linalg.norm(sym, "fro"),
                                                  1e-12))


# ---------------------------------------------------------------------------
# Pre-flight diagnostics (bands: Stage 2 addendum item v; defaults flagged)
# ---------------------------------------------------------------------------

def preflight(B, mom) -> dict:
    bands = C.H4["preflight"]
    p = B.shape[0]
    S0, S22 = mom["S0"], mom["S22"]
    tr_frac = float(np.trace(B) / p)
    diag = np.diag(B)
    rho = float(np.abs(np.linalg.eigvals(B)).max())
    homog = float(np.linalg.norm(S22 - S0, "fro") / np.linalg.norm(S0, "fro"))
    K = np.linalg.inv(S0)
    pc = -K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
    np.fill_diagonal(pc, 0)
    marg = S0 / np.sqrt(np.outer(np.diag(S0), np.diag(S0)))
    big = np.abs(pc) > 0.10
    flips = float(np.mean(np.sign(pc[big]) != np.sign(marg[big]))) \
        if big.any() else 0.0
    checks = dict(
        trace=(bands["trace_frac"][0] <= tr_frac <= bands["trace_frac"][1],
               tr_frac),
        diag_range=(bands["diag_range"][0] <= diag.min()
                    and diag.max() <= bands["diag_range"][1],
                    (float(diag.min()), float(diag.max()))),
        spectral_radius=(rho < bands["spectral_radius_max"], rho),
        homogeneity=(homog < bands["homogeneity_max"], homog),
        collider=(flips < bands["collider_flip_max"], flips))
    return dict(all_pass=all(v[0] for v in checks.values()), checks=checks)


# ---------------------------------------------------------------------------
# Anchors and the Stage 1 mixture
# ---------------------------------------------------------------------------

def anchor_latent(lam: np.ndarray, phi: float | None = None) -> np.ndarray:
    phi = C.H4["phi_midus"] if phi is None else phi
    return phi * np.outer(lam, lam) / float(lam @ lam)


def anchor_mutualism(S_w1: np.ndarray, tau: float | None = None,
                     K: np.ndarray | None = None) -> np.ndarray:
    tau = C.H4["tau_midus"] if tau is None else tau
    J = -0.5 * (np.linalg.inv(S_w1) if K is None else K)
    lam_max = np.linalg.eigvalsh((J + J.T) / 2).max()
    c = -1.0 / (tau * lam_max)
    return expm(c * J * C.H4["delta_t"])


def _off(M):
    p = M.shape[0]
    mask = ~np.eye(p, dtype=bool)
    return M[mask]


def stage1_mixture(Bsym, BL, BM, om=None, BT=None) -> dict:
    b, l, m = _off(Bsym), _off(BL), _off(BM)
    om = np.ones_like(b) if om is None else om
    d = l - m
    w_raw = float(np.sum(om * (b - m) * d) / max(np.sum(om * d * d), 1e-12))
    w = float(np.clip(w_raw, 0, 1))
    fitted = w * l + (1 - w) * m
    ss_res = float(np.sum(om * (b - fitted) ** 2))
    bbar = float(np.sum(om * b) / np.sum(om))
    ss_tot = float(np.sum(om * (b - bbar) ** 2))
    out = dict(w_raw=w_raw, w=w, clipped=bool(w != w_raw),
               r2_offdiag=1 - ss_res / max(ss_tot, 1e-12))
    if BT is not None:
        from scipy.optimize import minimize
        t = _off(BT)
        A = np.column_stack([l, m, t])

        def obj(x):
            return float(np.sum(om * (b - A @ x) ** 2))
        cons = [dict(type="eq", fun=lambda x: x.sum() - 1)]
        res = minimize(obj, np.array([w, 1 - w, 0.0]).clip(0.01, 0.98),
                       method="SLSQP", bounds=[(0, 1)] * 3, constraints=cons)
        out["three_anchor"] = dict(w_L=float(res.x[0]), w_M=float(res.x[1]),
                                   w_T=float(res.x[2]))
    return out


def wild_cluster_bootstrap(mom, lam_anchor, B_reps, seed) -> dict:
    """Rademacher perturbation of centered country moment contributions
    (same draw across Sigma_0/1/2); returns per-entry variance of the
    symmetrized B and the bootstrap distribution of the Stage 1 w_hat."""
    rng = RNG(seed)
    pc = mom["per_country"]
    cs = list(pc)
    S0, S1, S2 = mom["S0"], mom["S1"], mom["S2"]
    Bs_list, w_list = [], []
    BM0 = anchor_mutualism(S0)
    BL0 = anchor_latent(lam_anchor)
    for b in range(B_reps):
        e = rng.choice([-1.0, 1.0], len(cs))
        S0b = S0 + sum(e[i] * pc[c]["pi"] * (pc[c]["S0"] - S0)
                       for i, c in enumerate(cs))
        S1b = S1 + sum(e[i] * pc[c]["pi"] * (pc[c]["S1"] - S1)
                       for i, c in enumerate(cs))
        S2b = S2 + sum(e[i] * pc[c]["pi"] * (pc[c]["S2"] - S2)
                       for i, c in enumerate(cs))
        try:
            Bb, _ = b_obs_from(S0b, S1b, S2b)
        except np.linalg.LinAlgError:
            continue
        Bsb = (Bb + Bb.T) / 2
        Bs_list.append(Bsb)
        w_list.append(stage1_mixture(Bsb, BL0, anchor_mutualism(S0b), None)
                      ["w_raw"])
    arr = np.array(Bs_list)
    var = arr.var(0)
    return dict(entry_var=var, w_raw_boot=np.array(w_list))


def person_cluster_bootstrap(mom, lam_anchor, B_reps, seed) -> np.ndarray:
    """Naive person-level bootstrap sensitivity (prereg)."""
    rng = RNG(seed)
    Xc, w = mom["Xc"], mom["weights"]
    n = w.size
    BL0 = anchor_latent(lam_anchor)
    out = []
    for b in range(B_reps):
        idx = rng.integers(0, n, n)
        wi = w[idx]
        S0 = (Xc[1][mom["mask"]][idx] * wi[:, None]).T @ Xc[1][mom["mask"]][idx] / wi.sum()
        S1 = (Xc[2][mom["mask"]][idx] * wi[:, None]).T @ Xc[1][mom["mask"]][idx] / wi.sum()
        S2 = (Xc[3][mom["mask"]][idx] * wi[:, None]).T @ Xc[1][mom["mask"]][idx] / wi.sum()
        try:
            Bb, _ = b_obs_from(S0, S1, S2)
        except np.linalg.LinAlgError:
            continue
        out.append(stage1_mixture((Bb + Bb.T) / 2, BL0,
                                  anchor_mutualism(S0), None)["w_raw"])
    return np.array(out)


def mu_hat_from(S0, S1, B) -> float:
    p = S0.shape[0]
    try:
        Vmu = (S1 - B @ S0) @ np.linalg.inv(np.eye(p) - B)
    except np.linalg.LinAlgError:
        return np.nan
    Vmu = (Vmu + Vmu.T) / 2
    num, den = float(np.trace(Vmu)), float(np.trace(S0 - Vmu))
    return float(np.clip(num / max(den, 1e-9), 0.0, 5.0))


# ---------------------------------------------------------------------------
# Simulation-side estimator (shared by the map and H4a)
# ---------------------------------------------------------------------------

def _sim_moments(X):
    Xc = [X[t] - X[t].mean(0) for t in range(3)]
    n = X.shape[1]
    S0 = Xc[0].T @ Xc[0] / n
    S1 = Xc[1].T @ Xc[0] / n
    S2 = Xc[2].T @ Xc[0] / n
    return S0, S1, S2


def _estimate_w(X, lam_anchor) -> float:
    S0, S1, S2 = _sim_moments(X)
    B, _ = b_obs_from(S0, S1, S2)
    Bs = (B + B.T) / 2
    return stage1_mixture(Bs, anchor_latent(lam_anchor),
                          anchor_mutualism(S0), None)["w_raw"]


def stage2_map(lam_anchor, seed, smoke=False) -> dict:
    """Correction map E[w_hat | w_true, mu] on the pre-registered grid under
    the discretized measurement layer; isotonic-monotonized per mu."""
    cfg = C.H4
    ng = C.SMOKE["h4_wgrid_n"] if smoke else cfg["wgrid_n"]
    mus = C.SMOKE["h4_mu_grid"] if smoke else cfg["mu_grid"]
    reps = C.SMOKE["h4_mc_reps"] if smoke else cfg["mc_reps"]
    Nmc = C.SMOKE["h4_mc_n"] if smoke else cfg["mc_n"]
    wgrid = np.linspace(0, 1, ng)
    table = {}
    for mu in mus:
        means = []
        for wt in wgrid:
            ws = [_estimate_w(simulate_items_fast(
                Nmc, wt, mu, cfg["phi_midus"], cfg["tau_midus"],
                seed=seed + int(1e4 * wt) + int(100 * mu) + r), lam_anchor)
                for r in range(reps)]
            means.append(float(np.mean(ws)))
        iso = IsotonicRegression(increasing=True).fit(wgrid, means)
        table[mu] = iso.predict(wgrid)
    return dict(wgrid=wgrid, mus=list(mus), table=table)


def invert_map(map_out, w_raw, mu_hat) -> dict:
    wg = map_out["wgrid"]
    mus = np.array(map_out["mus"])
    mu_hat = float(np.clip(mu_hat, mus.min(), mus.max()))
    j = np.searchsorted(mus, mu_hat)
    if j == 0:
        curve = map_out["table"][mus[0]]
    elif j >= len(mus):
        curve = map_out["table"][mus[-1]]
    else:
        a = (mu_hat - mus[j - 1]) / (mus[j] - mus[j - 1])
        curve = (1 - a) * map_out["table"][mus[j - 1]] \
            + a * map_out["table"][mus[j]]
    w_corr = float(np.interp(w_raw, curve, wg))
    slope = float(np.interp(w_corr, wg[:-1] + np.diff(wg) / 2,
                            np.diff(curve) / np.diff(wg)))
    clipping = slope < C.H4["clip_slope_min"]
    return dict(w_corrected=w_corr, curve=curve, slope=slope,
                clipping=bool(clipping))


# ---------------------------------------------------------------------------
# H4a: calibrated single-channel model selection
# ---------------------------------------------------------------------------

def _channel_fits(Bs, S0, lam_anchor, om=None):
    b = _off(Bs)
    om = np.ones_like(b) if om is None else om
    Ml = _off(anchor_latent(lam_anchor, 1.0))
    phi = float(np.clip(np.sum(om * b * Ml) / max(np.sum(om * Ml * Ml), 1e-12),
                        0.01, 0.999))
    fitL = float(np.sum(om * (b - phi * Ml) ** 2))
    best = None
    for tau in np.geomspace(0.5, 40, 25):
        m = _off(anchor_mutualism(S0, tau))
        f = float(np.sum(om * (b - m) ** 2))
        if best is None or f < best[0]:
            best = (f, tau)
    fitM, tau_hat = best
    denom = max(float(np.sum(om * b * b)), 1e-12)
    return dict(phi_hat=phi, tau_hat=float(tau_hat),
                fitL=fitL / denom, fitM=fitM / denom,
                d=(fitL - fitM) / denom)


def run_h4a(mom, B, lam_anchor, mu_hat, om, seed, smoke=False,
            r2_stage1=np.nan) -> dict:
    cfg = C.H4
    nsim = C.SMOKE["h4a_nsim"] if smoke else cfg["h4a_nsim"]
    Nsim = min(mom["n"], C.SMOKE["h4_mc_n"] if smoke else cfg["mc_n"])
    Bs = (B + B.T) / 2
    obs = _channel_fits(Bs, mom["S0"], lam_anchor, om)
    dL, dM = [], []
    for r in range(nsim):
        XL = simulate_items_fast(Nsim, 1.0, mu_hat, obs["phi_hat"],
                                 cfg["tau_midus"], seed=seed + r)
        S0, S1, S2 = _sim_moments(XL)
        Bl, _ = b_obs_from(S0, S1, S2)
        dL.append(_channel_fits((Bl + Bl.T) / 2, S0, lam_anchor)["d"])
        XM = simulate_items_fast(Nsim, 0.0, mu_hat, cfg["phi_midus"],
                                 obs["tau_hat"], seed=seed + 5000 + r)
        S0, S1, S2 = _sim_moments(XM)
        Bm, _ = b_obs_from(S0, S1, S2)
        dM.append(_channel_fits((Bm + Bm.T) / 2, S0, lam_anchor)["d"])
    dL, dM = np.array(dL), np.array(dM)
    a = cfg["h4a_alpha"]
    tL = float(np.quantile(dL, 1 - a))    # reject Latent if d_obs > tL
    tM = float(np.quantile(dM, a))        # reject Mutualism if d_obs < tM
    rejL, rejM = obs["d"] > tL, obs["d"] < tM
    if not np.isnan(r2_stage1) and r2_stage1 < C.H4["tier1_r2"]:
        verdict = "Indistinguishable"     # R2_min gate (Stage 2 locks value)
    elif rejM and not rejL:
        verdict = "Latent"
    elif rejL and not rejM:
        verdict = "Mutualism"
    else:
        verdict = "Indistinguishable"
    lo, hi = min(dL.min(), dM.min()), max(dL.max(), dM.max())
    bins = np.linspace(lo, hi, 25)
    hL, _ = np.histogram(dL, bins, density=True)
    hM, _ = np.histogram(dM, bins, density=True)
    kappa = float(1 - np.sum(np.minimum(hL, hM)) * np.diff(bins)[0])
    return dict(verdict=verdict, d_obs=float(obs["d"]), thresholds=(tM, tL),
                kappa=kappa, dist_L=dL, dist_M=dM,
                phi_hat=obs["phi_hat"], tau_hat=obs["tau_hat"],
                fitL=obs["fitL"], fitM=obs["fitM"])


def confusion_matrix(map_seed, lam_anchor, tM, tL, mu_hat, seed,
                     smoke=False) -> pd.DataFrame:
    cfg = C.H4
    ng = C.SMOKE["h4_wgrid_n"] if smoke else 5
    reps = C.SMOKE["h4_mc_reps"] if smoke else 10
    Nmc = C.SMOKE["h4_mc_n"] if smoke else cfg["mc_n"]
    rows = []
    for wt in np.linspace(0, 1, ng):
        counts = dict(Latent=0, Mutualism=0, Indistinguishable=0)
        for r in range(reps):
            X = simulate_items_fast(Nmc, wt, mu_hat, cfg["phi_midus"],
                                    cfg["tau_midus"], seed=seed + r + int(wt * 999))
            S0, S1, S2 = _sim_moments(X)
            B, _ = b_obs_from(S0, S1, S2)
            d = _channel_fits((B + B.T) / 2, S0, lam_anchor)["d"]
            if d < tM:
                counts["Latent"] += 1
            elif d > tL:
                counts["Mutualism"] += 1
            else:
                counts["Indistinguishable"] += 1
        rows.append(dict(w_true=float(wt),
                         **{k: v / reps for k, v in counts.items()}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def naive_demeaned_lag1(mom) -> dict:
    Xc = [mom["Xc"][t][mom["mask"]] for t in (1, 2, 3)]
    pm = sum(Xc) / 3.0
    Xd = [x - pm for x in Xc]
    Xlag = np.vstack([Xd[0], Xd[1]])
    Xlead = np.vstack([Xd[1], Xd[2]])
    S00 = Xlag.T @ Xlag / Xlag.shape[0]
    S10 = Xlead.T @ Xlag / Xlag.shape[0]
    B = S10 @ np.linalg.inv(S00 + 1e-8 * np.eye(S00.shape[0]))
    return dict(trace_frac=float(np.trace(B) / B.shape[0]),
                spectral_radius=float(np.abs(np.linalg.eigvals(B)).max()),
                note="Nickell-biased by construction; cross-check only")


def separability_test(panel, h1_fit, mom) -> dict:
    """Scope condition Cov(eta, network residual) = 0: regress Bartlett f at
    t+1 on the GGM nodewise residuals at t (pooled transitions), Wald F."""
    import statsmodels.api as sm
    from sklearn.covariance import GraphicalLassoCV
    mask = mom["mask"]
    Xc = {t: mom["Xc"][t][mask] for t in (1, 2, 3)}
    sd = Xc[1].std(0)
    Z1 = Xc[1] / np.where(sd > 0, sd, 1)
    gl = GraphicalLassoCV(cv=3).fit(Z1)
    K = gl.precision_
    Bnw = -K / np.diag(K)[:, None]
    np.fill_diagonal(Bnw, 0.0)
    fs = {}
    for t in (1, 2, 3):
        Zt = Xc[t] / np.where(sd > 0, sd, 1)
        fs[t], _ = cfa.bartlett(h1_fit, Zt)
    resid = {t: (Xc[t] / np.where(sd > 0, sd, 1))
             - (Xc[t] / np.where(sd > 0, sd, 1)) @ Bnw.T for t in (1, 2)}
    y = np.concatenate([fs[2][:, 0], fs[3][:, 0]])
    Xr = np.vstack([resid[1], resid[2]])
    mdl = sm.OLS(y, sm.add_constant(Xr)).fit()
    fres = mdl.f_test(np.hstack([np.zeros((Xr.shape[1], 1)),
                                 np.eye(Xr.shape[1])]))
    return dict(F=float(fres.fvalue), p=float(fres.pvalue),
                holds=bool(fres.pvalue > 0.05))


def elasticnet_gmm(mom, lam_anchor) -> dict:
    """De Paula-Rasul-Souza-style cross-check: adaptive elastic net on the
    stacked per-country moment systems vec(D2_c) = (D1_c' kron I) vec(B).
    Simplified implementation, clearly a cross-check, not the primary."""
    from sklearn.linear_model import Ridge, ElasticNetCV
    pcs = mom["per_country"]
    p = mom["S0"].shape[0]
    rowsX, rowsY = [], []
    for c, d in pcs.items():
        D1 = d["S1"] - d["S0"]
        D2 = d["S2"] - d["S1"]
        rowsX.append(np.kron(D1.T, np.eye(p)) * np.sqrt(d["pi"]))
        rowsY.append(D2.flatten() * np.sqrt(d["pi"]))
    Xd, yd = np.vstack(rowsX), np.concatenate(rowsY)
    ridge = Ridge(alpha=1e-3).fit(Xd, yd)
    wadapt = 1.0 / np.maximum(np.abs(ridge.coef_), 1e-3)
    Xs = Xd / wadapt[None, :]
    en = ElasticNetCV(l1_ratio=0.5, cv=3, max_iter=5000).fit(Xs, yd)
    B = (en.coef_ / wadapt).reshape(p, p)
    Bs = (B + B.T) / 2
    mix = stage1_mixture(Bs, anchor_latent(lam_anchor),
                         anchor_mutualism(mom["S0"]), None)
    return dict(w=mix["w"], r2=mix["r2_offdiag"],
                nonzero=int((np.abs(en.coef_) > 1e-8).sum()))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_h4(panel: pd.DataFrame, h1_fit, seed: int, smoke: bool = False,
           lam_anchor: np.ndarray | None = None) -> dict:
    cfg = C.H4
    nboot = C.SMOKE["n_boot"] if smoke else cfg["n_boot"]
    if lam_anchor is None:
        # ANCHOR NOTE: registered anchor direction is the MIDUS Lambda mapped
        # onto the GFS indicator set, fixed in the Stage 2 addendum; until
        # that file is supplied the H1 pooled f-loading direction stands in.
        lam_anchor = h1_fit.loading_vector("f")
    mom = panel_moments(panel)
    B, tik = b_obs_from(mom["S0"], mom["S1"], mom["S2"])
    Bs = (B + B.T) / 2
    sk = s_skew(B)
    use_sym = sk > cfg["s_skew_max"]
    pf = preflight(B, mom)
    boot = wild_cluster_bootstrap(mom, lam_anchor, nboot, seed)
    om = 1.0 / np.maximum(_off(boot["entry_var"]), 1e-10)
    om = om / om.mean()
    BL = anchor_latent(lam_anchor)
    BM = anchor_mutualism(mom["S0"])
    BT = np.eye(len(FN))
    mix = stage1_mixture(Bs, BL, BM, om, BT=BT)
    # sparse-J variant
    from sklearn.covariance import GraphicalLassoCV
    sd = mom["Xc"][1][mom["mask"]].std(0)
    Z1 = mom["Xc"][1][mom["mask"]] / np.where(sd > 0, sd, 1)
    K_sparse = GraphicalLassoCV(cv=3).fit(Z1).precision_
    Dsd = np.diag(sd)
    K_sparse_cov = np.linalg.inv(Dsd) @ K_sparse @ np.linalg.inv(Dsd)
    BM_sparse = anchor_mutualism(mom["S0"], K=K_sparse_cov)
    mix_sparse = stage1_mixture(Bs, BL, BM_sparse, om)
    sparsity_gap = abs(mix["w"] - mix_sparse["w"])

    mu_hat = mu_hat_from(mom["S0"], mom["S1"], B)
    map_out = stage2_map(lam_anchor, seed + 11, smoke=smoke)
    inv = invert_map(map_out, mix["w_raw"], mu_hat)
    boot_corr = np.array([invert_map(map_out, wb, mu_hat)["w_corrected"]
                          for wb in boot["w_raw_boot"]])
    lo, hi = np.percentile(boot_corr, [2.5, 97.5])
    half = (hi - lo) / 2 * cfg["c_inflate"]
    ci = (max(inv["w_corrected"] - half, 0.0),
          min(inv["w_corrected"] + half, 1.0))
    ci_width = float(ci[1] - ci[0])

    corner = inv["w_corrected"] >= cfg["corner_interval"][0]
    if inv["clipping"]:
        interval = (0.0, float(np.interp(C.H4["clip_slope_min"],
                                         [0, 1], [0, 1]) or 0.25))
        partial_id = ("clipping_zone", (0.0, float(max(inv["w_corrected"],
                                                       0.25))))
    elif corner:
        partial_id = ("high_wL_high_tau_corner", cfg["corner_interval"])
    else:
        partial_id = None

    if partial_id is not None:
        tier = "partial_identification"
    else:
        checks = [ci_width < cfg["tier1_ci_width"],
                  mix["r2_offdiag"] > cfg["tier1_r2"],
                  pf["all_pass"], sparsity_gap < cfg["sparsity_gap_max"],
                  sk < cfg["s_skew_max"]]
        tier = 1 if all(checks) else (2 if sum(checks) >= 3 else 3)

    h4a = run_h4a(mom, B, lam_anchor, mu_hat, om, seed + 33, smoke=smoke,
                  r2_stage1=mix["r2_offdiag"])
    conf = confusion_matrix(map_out, lam_anchor, h4a["thresholds"][0],
                            h4a["thresholds"][1], mu_hat, seed + 44,
                            smoke=smoke)
    person_boot = person_cluster_bootstrap(mom, lam_anchor,
                                           max(nboot // 3, 20), seed + 55)
    checks_x = dict(naive=naive_demeaned_lag1(mom),
                    separability=separability_test(panel, h1_fit, mom),
                    elasticnet=elasticnet_gmm(mom, lam_anchor))

    dirl = directional_commitment(h4a["verdict"], tier, inv["w_corrected"],
                                  partial_id)
    return dict(B_obs=B, B_sym=Bs, s_skew=sk, symmetrized_fallback=use_sym,
                tikhonov=tik, preflight=pf, anchors=dict(BL=BL, BM=BM),
                stage1=mix, sparse=dict(w=mix_sparse["w"], gap=sparsity_gap),
                mu_hat=mu_hat, map=map_out, inversion=inv,
                w_corrected=inv["w_corrected"], ci=ci, ci_width=ci_width,
                partial_identification=partial_id, tier=tier,
                h4a=h4a, confusion=conf,
                boot=dict(w_raw=boot["w_raw_boot"], w_corr=boot_corr,
                          person_w_raw=person_boot),
                cross_checks=checks_x, directional=dirl,
                n_panel=mom["n"])


def directional_commitment(h4a_verdict: str, tier, w_corr: float,
                           partial_id) -> dict:
    """Prereg: the coupling-strength interpretation predicts the propagator
    is NOT latent-dominated. Default joint configuration implemented here
    (v8's exact configuration to be transcribed at Stage 2): falsified when
    H4a returns Latent and H4b places w_L at/above the corner interval."""
    latent_dominated = (h4a_verdict == "Latent"
                        and ((partial_id and partial_id[0].startswith("high"))
                             or (tier in (1, 2) and w_corr >= 0.90)))
    return dict(coupling_interpretation_falsified=bool(latent_dominated),
                basis=f"H4a={h4a_verdict}, w_corrected={w_corr:.3f}, "
                      f"tier={tier}, partial_id={partial_id}",
                scope="Falsification targets the coupling-strength "
                      "interpretation only; H1's structural claim and H3's "
                      "predictive claim are untouched (prereg).")

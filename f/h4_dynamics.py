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
from .data_io import rubin_pool
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


def run_h4a(obs: dict, n_panel: int, lam_anchor, mu_hat, seed, smoke=False,
            r2_stage1=np.nan) -> dict:
    """Calibrated cross-fitting. obs holds the (MI-pooled) observed decision
    statistics: d, phi_hat, tau_hat, fitL, fitM. The verdict consumes BOTH
    decision statistics (Delta_fit, kappa) per the prereg: kappa below the
    config floor (Stage 2 addendum item vii locks it) means the two truths
    are not separable at this design and the verdict is Indistinguishable."""
    cfg = C.H4
    nsim = C.SMOKE["h4a_nsim"] if smoke else cfg["h4a_nsim"]
    Nsim = min(n_panel, C.SMOKE["h4_mc_n"] if smoke else cfg["mc_n"])
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
    lo, hi = min(dL.min(), dM.min()), max(dL.max(), dM.max())
    bins = np.linspace(lo, hi, 25)
    hL, _ = np.histogram(dL, bins, density=True)
    hM, _ = np.histogram(dM, bins, density=True)
    kappa = float(1 - np.sum(np.minimum(hL, hM)) * np.diff(bins)[0])
    if kappa < cfg["h4a_kappa_min"]:
        verdict = "Indistinguishable"     # kappa gate (Stage 2 locks value)
    elif not np.isnan(r2_stage1) and r2_stage1 < cfg["tier1_r2"]:
        verdict = "Indistinguishable"     # R2_min gate (Stage 2 locks value)
    elif rejM and not rejL:
        verdict = "Latent"
    elif rejL and not rejM:
        verdict = "Mutualism"
    else:
        verdict = "Indistinguishable"
    return dict(verdict=verdict, d_obs=float(obs["d"]), thresholds=(tM, tL),
                kappa=kappa, kappa_min=cfg["h4a_kappa_min"],
                dist_L=dL, dist_M=dM,
                phi_hat=obs["phi_hat"], tau_hat=obs["tau_hat"],
                fitL=obs["fitL"], fitM=obs["fitM"])


def confusion_matrix(lam_anchor, tM, tL, seed, smoke=False) -> pd.DataFrame:
    """P(verdict | truth) across the pre-registered w_L x mu_ratio grid
    (Stage 2 addendum item vii), not just the w_L margin."""
    cfg = C.H4
    ng = C.SMOKE["h4_wgrid_n"] if smoke else 5
    reps = C.SMOKE["h4_mc_reps"] if smoke else 10
    Nmc = C.SMOKE["h4_mc_n"] if smoke else cfg["mc_n"]
    mus = C.SMOKE["h4_mu_grid"] if smoke else cfg["mu_grid"]
    rows = []
    for mu in mus:
        for wt in np.linspace(0, 1, ng):
            counts = dict(Latent=0, Mutualism=0, Indistinguishable=0)
            for r in range(reps):
                X = simulate_items_fast(Nmc, wt, mu, cfg["phi_midus"],
                                        cfg["tau_midus"],
                                        seed=seed + r + int(wt * 999)
                                        + int(mu * 7717))
                S0, S1, S2 = _sim_moments(X)
                B, _ = b_obs_from(S0, S1, S2)
                d = _channel_fits((B + B.T) / 2, S0, lam_anchor)["d"]
                if d < tM:
                    counts["Latent"] += 1
                elif d > tL:
                    counts["Mutualism"] += 1
                else:
                    counts["Indistinguishable"] += 1
            rows.append(dict(w_true=float(wt), mu_ratio=float(mu),
                             **{k: v / reps for k, v in counts.items()}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def f_wave_scores(panel: pd.DataFrame, h1_fit) -> np.ndarray:
    """Bartlett f scores at each wave under the pooled W1 loadings and W1
    standardization (wave-invariant scoring rule, as in H3)."""
    cols1 = [f"{v}_w1" for v in FN]
    X1 = panel[cols1].to_numpy(float)
    mu1 = np.nanmean(X1, 0)
    sd1 = np.nanstd(X1, 0)
    out = []
    for wv in (1, 2, 3):
        X = panel[[f"{v}_w{wv}" for v in FN]].to_numpy(float)
        Z = np.nan_to_num((X - mu1) / np.where(sd1 > 0, sd1, 1))
        fs, _ = cfa.bartlett(h1_fit, Z)
        out.append(fs[:, 0])
    return np.column_stack(out)


def ri_ar1_f(panel: pd.DataFrame, h1_fit) -> dict:
    """Univariate random-intercept AR(1) on the 3-wave f scores: the RI-CLPM
    cross-check named in the prereg's H4 list (the bivariate per-outcome
    RI-CLPM lives in H3). Separates the trait (random-intercept) variance
    share from the within-person lag-1 persistence of f."""
    from scipy.optimize import minimize
    F = f_wave_scores(panel, h1_fit)
    w = panel[C.ADMIN["weight"]].to_numpy(float)
    m = ~np.isnan(F).any(1)
    Z, ww = F[m], w[m]
    mu_s = np.average(Z, 0, weights=ww)
    S = np.cov(Z, rowvar=False, aweights=ww)

    def build(th):
        vI, v1, a, ve = np.exp(th[0]), np.exp(th[1]), th[2], np.exp(th[3])
        W = {1: v1}
        W[2] = a * a * W[1] + ve
        W[3] = a * a * W[2] + ve
        Sig = np.full((3, 3), vI)
        for t in range(3):
            for s in range(3):
                if t == s:
                    Sig[t, s] += W[t + 1]
                elif t > s:
                    Sig[t, s] += a ** (t - s) * W[s + 1]
                else:
                    Sig[t, s] += a ** (s - t) * W[t + 1]
        return Sig

    def nll(th):
        Sig = build(th)
        sgn, ld = np.linalg.slogdet(Sig)
        if sgn <= 0:
            return 1e9
        return float(ld + np.sum(np.linalg.inv(Sig) * S))

    res = minimize(nll, np.array([np.log(0.3), np.log(0.5), 0.5, np.log(0.3)]),
                   method="L-BFGS-B",
                   bounds=[(-6, 3), (-6, 3), (-0.99, 0.99), (-6, 3)],
                   options=dict(maxiter=2000))
    vI, v1 = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    return dict(a_within=float(res.x[2]),
                ri_variance_share=vI / (vI + v1),
                converged=bool(res.success), n=int(m.sum()))


def _pool_preflight(pf_list: list[dict]) -> dict:
    """Rubin-style pooling of the pre-flight diagnostics: band checks are
    re-evaluated on the imputation-mean statistic; per-imputation pass
    fraction reported alongside."""
    bands = C.H4["preflight"]
    names = ["trace", "diag_range", "spectral_radius", "homogeneity",
             "collider"]
    checks = {}
    for nm in names:
        vals = [pf["checks"][nm][1] for pf in pf_list]
        if nm == "diag_range":
            v = (float(np.mean([x[0] for x in vals])),
                 float(np.mean([x[1] for x in vals])))
            ok = bands["diag_range"][0] <= v[0] and v[1] <= bands["diag_range"][1]
        else:
            v = float(np.mean(vals))
            if nm == "trace":
                ok = bands["trace_frac"][0] <= v <= bands["trace_frac"][1]
            elif nm == "spectral_radius":
                ok = v < bands["spectral_radius_max"]
            elif nm == "homogeneity":
                ok = v < bands["homogeneity_max"]
            else:
                ok = v < bands["collider_flip_max"]
        checks[nm] = (bool(ok), v)
    return dict(all_pass=all(v[0] for v in checks.values()), checks=checks,
                pass_fraction=float(np.mean([pf["all_pass"]
                                             for pf in pf_list])))


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

def run_h4(imputed_panels, h1_fit, seed: int, smoke: bool = False,
           lam_anchor: np.ndarray | None = None) -> dict:
    """H4 across the full MI set (prereg, Missing Data: the panel
    auto-covariance matrices are estimated per imputation and pooled by
    Rubin's rules). The Stage 2 correction map and the H4a calibration
    simulations are data-independent given the anchors and run once; every
    data-dependent statistic is computed per imputation and pooled. The
    heavier cross-checks and the reported bootstrap distributions use
    imputation 1 (documented)."""
    cfg = C.H4
    if isinstance(imputed_panels, pd.DataFrame):
        imputed_panels = [imputed_panels]
    m_imp = len(imputed_panels)
    nboot = C.SMOKE["n_boot"] if smoke else cfg["n_boot"]
    if lam_anchor is None:
        # ANCHOR NOTE: registered anchor direction is the MIDUS Lambda mapped
        # onto the GFS indicator set, fixed in the Stage 2 addendum; until
        # that file is supplied the H1 pooled f-loading direction stands in.
        lam_anchor = h1_fit.loading_vector("f")
    BL = anchor_latent(lam_anchor)
    BT = np.eye(len(FN))
    map_out = stage2_map(lam_anchor, seed + 11, smoke=smoke)

    from sklearn.covariance import GraphicalLassoCV
    per, pf_list = [], []
    mom1 = None
    for k, panel in enumerate(imputed_panels):
        mom = panel_moments(panel)
        if k == 0:
            mom1 = mom
        B, tik = b_obs_from(mom["S0"], mom["S1"], mom["S2"])
        Bs = (B + B.T) / 2
        pf_list.append(preflight(B, mom))
        boot = wild_cluster_bootstrap(mom, lam_anchor, nboot, seed + 17 * k)
        om = 1.0 / np.maximum(_off(boot["entry_var"]), 1e-10)
        om = om / om.mean()
        BM = anchor_mutualism(mom["S0"])
        mix = stage1_mixture(Bs, BL, BM, om, BT=BT)
        # sparse-J variant
        sd = mom["Xc"][1][mom["mask"]].std(0)
        Z1 = mom["Xc"][1][mom["mask"]] / np.where(sd > 0, sd, 1)
        K_sparse = GraphicalLassoCV(cv=3).fit(Z1).precision_
        Dsd = np.diag(sd)
        K_sparse_cov = np.linalg.inv(Dsd) @ K_sparse @ np.linalg.inv(Dsd)
        mix_sparse = stage1_mixture(Bs, BL,
                                    anchor_mutualism(mom["S0"],
                                                     K=K_sparse_cov), om)
        mu_hat = mu_hat_from(mom["S0"], mom["S1"], B)
        inv = invert_map(map_out, mix["w_raw"], mu_hat)
        boot_corr = np.array([invert_map(map_out, wb, mu_hat)["w_corrected"]
                              for wb in boot["w_raw_boot"]])
        per.append(dict(B=B, Bs=Bs, tik=tik, sk=s_skew(B), mix=mix,
                        sparse_w=mix_sparse["w"],
                        sparsity_gap=abs(mix["w"] - mix_sparse["w"]),
                        mu_hat=mu_hat, inv=inv, boot=boot,
                        boot_corr=boot_corr,
                        obs_cf=_channel_fits(Bs, mom["S0"], lam_anchor, om)))

    # ---- Rubin pooling of the headline statistics ----
    pool_w = rubin_pool([p["inv"]["w_corrected"] for p in per])
    w_corrected = pool_w["point"]
    W_within = float(np.mean([np.var(p["boot_corr"], ddof=1)
                              if p["boot_corr"].size > 1 else 0.0
                              for p in per]))
    T_var = W_within + (1 + 1 / m_imp) * pool_w["between_var"]
    half = 1.96 * np.sqrt(T_var) * cfg["c_inflate"]
    ci = (max(w_corrected - half, 0.0), min(w_corrected + half, 1.0))
    ci_width = float(ci[1] - ci[0])
    w_raw_pool = float(np.mean([p["mix"]["w_raw"] for p in per]))
    mu_hat_pool = float(np.mean([p["mu_hat"] for p in per]))
    r2_pool = float(np.mean([p["mix"]["r2_offdiag"] for p in per]))
    sparsity_gap = float(np.mean([p["sparsity_gap"] for p in per]))
    sk_pool = float(np.mean([p["sk"] for p in per]))
    use_sym = sk_pool > cfg["s_skew_max"]
    pf = _pool_preflight(pf_list)
    inv_pooled = invert_map(map_out, w_raw_pool, mu_hat_pool)
    ta = [p["mix"].get("three_anchor") for p in per
          if p["mix"].get("three_anchor")]
    three_anchor = ({k: float(np.mean([t[k] for t in ta]))
                     for k in ("w_L", "w_M", "w_T")} if ta else None)

    corner = w_corrected >= cfg["corner_interval"][0]
    if inv_pooled["clipping"]:
        partial_id = ("clipping_zone", (0.0, float(max(w_corrected, 0.25))))
    elif corner:
        partial_id = ("high_wL_high_tau_corner", cfg["corner_interval"])
    else:
        partial_id = None

    if partial_id is not None:
        tier = "partial_identification"
    else:
        checks = [ci_width < cfg["tier1_ci_width"],
                  r2_pool > cfg["tier1_r2"],
                  pf["all_pass"], sparsity_gap < cfg["sparsity_gap_max"],
                  sk_pool < cfg["s_skew_max"]]
        tier = 1 if all(checks) else (2 if sum(checks) >= 3 else 3)

    obs_pool = dict(
        d=float(np.mean([p["obs_cf"]["d"] for p in per])),
        phi_hat=float(np.mean([p["obs_cf"]["phi_hat"] for p in per])),
        tau_hat=float(np.mean([p["obs_cf"]["tau_hat"] for p in per])),
        fitL=float(np.mean([p["obs_cf"]["fitL"] for p in per])),
        fitM=float(np.mean([p["obs_cf"]["fitM"] for p in per])))
    h4a = run_h4a(obs_pool, mom1["n"], lam_anchor, mu_hat_pool, seed + 33,
                  smoke=smoke, r2_stage1=r2_pool)
    conf = confusion_matrix(lam_anchor, h4a["thresholds"][0],
                            h4a["thresholds"][1], seed + 44, smoke=smoke)
    # cross-checks on imputation 1 (documented)
    person_boot = person_cluster_bootstrap(mom1, lam_anchor,
                                           max(nboot // 3, 20), seed + 55)
    checks_x = dict(naive=naive_demeaned_lag1(mom1),
                    separability=separability_test(imputed_panels[0], h1_fit,
                                                   mom1),
                    elasticnet=elasticnet_gmm(mom1, lam_anchor),
                    ri_ar1=ri_ar1_f(imputed_panels[0], h1_fit))

    dirl = directional_commitment(h4a["verdict"], tier, w_corrected,
                                  partial_id)
    p1 = per[0]
    return dict(B_obs=p1["B"], B_sym=p1["Bs"], s_skew=sk_pool,
                symmetrized_fallback=use_sym,
                tikhonov=p1["tik"], preflight=pf,
                anchors=dict(BL=BL, BM=anchor_mutualism(mom1["S0"])),
                stage1=dict(w_raw=w_raw_pool,
                            w=float(np.mean([p["mix"]["w"] for p in per])),
                            r2_offdiag=r2_pool,
                            clipped=bool(inv_pooled["clipping"]),
                            three_anchor=three_anchor),
                sparse=dict(w=float(np.mean([p["sparse_w"] for p in per])),
                            gap=sparsity_gap),
                mu_hat=mu_hat_pool, map=map_out, inversion=inv_pooled,
                w_corrected=w_corrected, ci=ci, ci_width=ci_width,
                mi=dict(m=m_imp, between_var=pool_w["between_var"],
                        within_var=W_within,
                        per_imputation_w=[p["inv"]["w_corrected"]
                                          for p in per]),
                partial_identification=partial_id, tier=tier,
                h4a=h4a, confusion=conf,
                boot=dict(w_raw=p1["boot"]["w_raw_boot"],
                          w_corr=p1["boot_corr"],
                          person_w_raw=person_boot),
                cross_checks=checks_x, directional=dirl,
                n_panel=mom1["n"])


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

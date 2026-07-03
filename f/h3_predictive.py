"""
h3_predictive.py -- H3: predictive validity of leave-one-domain-out f beyond
a measurement-equivalent psychopathology composite.

Primary estimand (v8): per outcome k with own domain d(k), f_(-d(k)) from a
re-fit of the H1 bifactor excluding domain d(k)'s indicators and group
factor; outcome's Wave 1 -> Wave 2 change (baseline controlled) regressed on
latent f_(-d(k)), latent p, and demographic controls, pooled SEM with country
fixed effects. Wave 1 -> Wave 3 as pre-registered extension; full-f as the
secondary, part-whole-contaminated, MIDUS-comparable estimand.

'Latent' implementation: Bartlett scores + Croon correction. Bartlett scores
are conditionally unbiased (FS = eta + e, e orthogonal to eta and, with
item-disjoint f and p measurement models, e_f orthogonal to e_p), so the
score-outcome covariances identify the latent covariances and only the score
variances need the (L' Theta^-1 L)^-1 error-variance subtraction. The
regression is then solved on the corrected covariance matrix -- structurally
equivalent to keeping f and p latent in the SEM, with a lavaan joint-SEM
parity check listed for Stage 2. Country FE via weighted within-country
demeaning; SEs by PSU-cluster bootstrap (prereg: weights at country level,
PSU clustering); Rubin pooling across imputations.

Confirmation requires (constitutively): standardized beta >= 0.05 on >= 2 of
3 outcomes at BH-FDR q = .05, the reliability-matching precondition
|omega_h(f_-d) - omega_h(p)| < .05 (or the matched indicator subset), and
residualized-construct cosine >= .85 on confirming outcomes (Hoyle et al.
2023 partialing caveat carried in report text). RI-CLPM within-person effects
of opposite sign are a disconfirmation leg.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from . import config as C
from . import cfa
from .polychoric import polychoric_matrix
from .data_io import rubin_pool

RNG = np.random.default_rng


# ---------------------------------------------------------------------------
# Measurement pieces
# ---------------------------------------------------------------------------

def p_spec(single: bool = False) -> cfa.Spec:
    items = [it.var for it in C.P_ITEMS]
    if single:
        sp = cfa.Spec(items=items, factors=["p"])
        for v in items:
            sp.add(v, "p")
        return sp
    sp = cfa.Spec(items=items, factors=["p", "dep", "anx"])
    for it in C.P_ITEMS:
        sp.add(it.var, "p")
        sp.add(it.var, it.domain, key=f"l|{it.domain}|tau")  # within-pair tie
    return sp


def fit_p(R18, V18, names18, N, seed) -> tuple[cfa.FitResult, dict]:
    pn = [it.var for it in C.P_ITEMS]
    idx = [names18.index(v) for v in pn]
    Rp, Vp = R18[np.ix_(idx, idx)], V18[np.ix_(idx, idx)]
    fit = cfa.fit_dwls(p_spec(False), Rp, N, Vp, seed=seed)
    downgraded = False
    if fit.heywood or not fit.converged:
        fit = cfa.fit_dwls(p_spec(True), Rp, N, Vp, seed=seed)
        downgraded = True
    oe = cfa.omega_ecv(fit, general="p")
    omega = oe["omega_h"] if not downgraded else oe["omega_total"]
    return fit, dict(downgraded=downgraded, omega=omega)


def lodo_fit(R18, V18, names18, N, drop_domain, seed) -> cfa.FitResult:
    part = {d: v for d, v in C.f_partition().items() if d != drop_domain}
    keep = [v for vs in part.values() for v in vs]
    idx = [names18.index(v) for v in keep]
    fit = cfa.fit_dwls(cfa.bifactor_spec(part), R18[np.ix_(idx, idx)], N,
                       V18[np.ix_(idx, idx)],
                       heywood_floor=C.H1["heywood_floor"], seed=seed)
    return fit


def joint_f_p_spec(drop_domain) -> cfa.Spec:
    part = {d: v for d, v in C.f_partition().items() if d != drop_domain}
    sp = cfa.bifactor_spec(part)
    for it in C.P_ITEMS:
        sp.items.append(it.var)
        sp.add(it.var, "p")
    sp.factors.append("p")
    sp.phi_free.append(("f", "p", "phi|f|p"))
    return sp


def residualized_cosine(R18, V18, names18, N, drop_domain, seed) -> float:
    """cosine(f loadings with p in the model, f loadings without p): the
    implemented residualized-construct check -- does partialing latent p
    reshape what f measures? Verified against the v8 definition at Stage 2."""
    solo = lodo_fit(R18, V18, names18, N, drop_domain, seed)
    spj = joint_f_p_spec(drop_domain)
    idx = [names18.index(v) for v in spj.items]
    joint = cfa.fit_dwls(spj, R18[np.ix_(idx, idx)], N,
                         V18[np.ix_(idx, idx)], seed=seed)
    lam_solo = solo.loading_vector("f")
    lam_joint = joint.loading_vector("f")[:len(lam_solo)]
    return cfa.cosine(lam_solo, lam_joint)


# ---------------------------------------------------------------------------
# Croon-corrected structural regression
# ---------------------------------------------------------------------------

def _weighted_demean(df, cols, w, by):
    X = df[cols].to_numpy(float)
    out = X.copy()
    for _, idx in df.groupby(by).indices.items():
        ww = w[idx][:, None]
        mu = np.nansum(ww * X[idx], 0) / np.nansum(
            ww * ~np.isnan(X[idx]), 0).clip(1e-9)
        out[idx] = X[idx] - mu
    return out


def croon_regression(frame: pd.DataFrame, ycol2: str, ycol1: str,
                     fs_f: np.ndarray, err_f: float,
                     fs_p: np.ndarray, err_p: float) -> dict:
    """Country-FE weighted regression of Y_w2 on [Y_w1, f, p, controls] with
    the Bartlett error variances subtracted from the score variances."""
    w = frame[C.ADMIN["weight"]].to_numpy(float)
    ctrl = list(C.H3_CONTROLS.values())
    d = frame.copy()
    d["_f"], d["_p"] = fs_f, fs_p
    cols = [ycol2, ycol1, "_f", "_p"] + ctrl
    Xd = _weighted_demean(d, cols, w, C.ADMIN["country"])
    m = ~np.isnan(Xd).any(1)
    Xd, ww = Xd[m], w[m]
    Cm = np.cov(Xd, rowvar=False, aweights=ww)
    Cc = Cm.copy()
    i_f, i_p = 2, 3
    Cc[i_f, i_f] = max(Cc[i_f, i_f] - err_f, 0.05 * Cm[i_f, i_f])
    Cc[i_p, i_p] = max(Cc[i_p, i_p] - err_p, 0.05 * Cm[i_p, i_p])
    xi = list(range(1, Cm.shape[0]))
    b = np.linalg.solve(Cc[np.ix_(xi, xi)], Cc[xi, 0])
    sdx = np.sqrt(np.diag(Cc)[xi])
    beta = b * sdx / np.sqrt(Cc[0, 0])
    return dict(beta_f=float(beta[1]), beta_p=float(beta[2]),
                beta_baseline=float(beta[0]), n=int(m.sum()),
                betas=dict(zip(["baseline", "f", "p"] + ctrl, map(float, beta))),
                mask=m, weights=ww, X=Xd, err=(err_f, err_p))


def psu_bootstrap(frame, res, B, seed) -> float:
    """Cluster bootstrap over PSUs (fixed measurement stage -- documented
    approximation; full-refit variant available via h3(..., full_refit))."""
    rng = RNG(seed)
    psus = frame.loc[res["mask"], C.ADMIN["psu"]].to_numpy()
    X, w = res["X"], res["weights"]
    uniq = np.unique(psus)
    groups = {u: np.where(psus == u)[0] for u in uniq}
    err_f, err_p = res["err"]
    out = []
    for b in range(B):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in take])
        Cm = np.cov(X[idx], rowvar=False, aweights=w[idx])
        Cc = Cm.copy()
        Cc[2, 2] = max(Cc[2, 2] - err_f, 0.05 * Cm[2, 2])
        Cc[3, 3] = max(Cc[3, 3] - err_p, 0.05 * Cm[3, 3])
        xi = list(range(1, Cm.shape[0]))
        try:
            bb = np.linalg.solve(Cc[np.ix_(xi, xi)], Cc[xi, 0])
        except np.linalg.LinAlgError:
            continue
        # ``Cc[xi, xi]`` uses NumPy advanced indexing and returns the
        # diagonal as a 1-D array, so index the f predictor variance directly.
        var_f = Cc[xi[1], xi[1]]
        if Cc[0, 0] <= 0 or var_f < 0:
            continue
        out.append(bb[1] * np.sqrt(var_f) / np.sqrt(Cc[0, 0]))
    return float(np.std(out, ddof=1)) if len(out) > 3 else np.nan


def bh_fdr(pvals: dict[str, float], q: float) -> dict[str, bool]:
    keys = list(pvals)
    ps = np.array([pvals[k] for k in keys])
    order = np.argsort(ps)
    m = len(ps)
    passed = np.zeros(m, bool)
    thresh = q * (np.arange(1, m + 1)) / m
    ok = np.where(ps[order] <= thresh)[0]
    if ok.size:
        passed[order[:ok.max() + 1]] = True
    return dict(zip(keys, map(bool, passed)))


# ---------------------------------------------------------------------------
# RI-CLPM sensitivity (3 waves, lag paths tied across intervals)
# ---------------------------------------------------------------------------

def riclpm(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> dict:
    """Bivariate random-intercept cross-lagged panel model, T = 3.
    Returns the within-person cross-lag c: f_t-1 -> y_t (and b: y -> f).
    ML on the 6 x 6 moment matrix + means; lag matrix and innovation
    covariance tied across the two intervals (T = 3 identification)."""
    Zm = np.column_stack([X[:, 0], Y[:, 0], X[:, 1], Y[:, 1], X[:, 2], Y[:, 2]])
    m = ~np.isnan(Zm).any(1)
    Z, ww = Zm[m], w[m]
    N = ww.sum()
    mu_s = np.average(Z, 0, weights=ww)
    S = np.cov(Z, rowvar=False, aweights=ww)

    def build(th):
        mu = th[0:6]
        vIx, vIy = np.exp(th[6]), np.exp(th[7])
        cI = th[8] * np.sqrt(vIx * vIy)
        SigI = np.array([[vIx, cI], [cI, vIy]])
        vX1, vY1 = np.exp(th[9]), np.exp(th[10])
        c1 = th[11] * np.sqrt(vX1 * vY1)
        W1 = np.array([[vX1, c1], [c1, vY1]])
        A = np.array([[th[12], th[13]], [th[14], th[15]]])
        vex, vey = np.exp(th[16]), np.exp(th[17])
        ce = th[18] * np.sqrt(vex * vey)
        E = np.array([[vex, ce], [ce, vey]])
        W = {1: W1}
        W[2] = A @ W1 @ A.T + E
        W[3] = A @ W[2] @ A.T + E
        Sig = np.zeros((6, 6))
        for t in range(3):
            for s in range(3):
                blk = SigI.copy()
                if t == s:
                    blk = blk + W[t + 1]
                elif t > s:
                    blk = blk + np.linalg.matrix_power(A, t - s) @ W[s + 1]
                else:
                    blk = blk + (np.linalg.matrix_power(A, s - t) @ W[t + 1]).T
                Sig[2 * t:2 * t + 2, 2 * s:2 * s + 2] = blk
        return mu, Sig

    def nll(th):
        mu, Sig = build(th)
        try:
            Si = np.linalg.inv(Sig)
            sgn, ld = np.linalg.slogdet(Sig)
            if sgn <= 0:
                return 1e9
        except np.linalg.LinAlgError:
            return 1e9
        d = mu_s - mu
        return float(ld + np.sum(Si * S) + d @ Si @ d)

    th0 = np.concatenate([mu_s, np.log([0.3, 0.3]), [0.2],
                          np.log([0.5, 0.5]), [0.2],
                          [0.3, 0.05, 0.05, 0.3],
                          np.log([0.4, 0.4]), [0.1]])
    res = minimize(nll, th0, method="L-BFGS-B",
                   bounds=[(None, None)] * 6 + [(-6, 3)] * 2 + [(-0.95, 0.95)]
                   + [(-6, 3)] * 2 + [(-0.95, 0.95)] + [(-1.5, 1.5)] * 4
                   + [(-6, 3)] * 2 + [(-0.95, 0.95)],
                   options=dict(maxiter=3000))
    A = np.array([[res.x[12], res.x[13]], [res.x[14], res.x[15]]])
    return dict(a_xx=float(A[0, 0]), b_xy=float(A[0, 1]),
                c_yx=float(A[1, 0]), d_yy=float(A[1, 1]),
                converged=bool(res.success), n=int(m.sum()))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_h3(imputed_panels: list[pd.DataFrame], seed: int,
           smoke: bool = False) -> dict:
    cfg = C.H3
    B = C.SMOKE["h3_boot"] if smoke else cfg["n_boot"]
    fnames = [it.var for it in C.F_ITEMS]
    pnames = [it.var for it in C.P_ITEMS]
    names18 = fnames + pnames
    ncats = [it.n_cat for it in C.F_ITEMS] + [it.n_cat for it in C.P_ITEMS]
    w1cols = [f"{v}_w1" for v in names18]

    per_outcome: dict[str, dict] = {k: dict(beta=[], beta_p=[], beta_full=[],
                                            beta_w3=[]) for k in C.H3_OUTCOMES}
    match_report, cosines, se_boot = {}, {}, {}
    riclpm_out, p_meta = {}, None

    for k_imp, panel in enumerate(imputed_panels):
        w = panel[C.ADMIN["weight"]].to_numpy(float)
        X18 = panel[w1cols].to_numpy(float)
        N = float(w.sum())
        R18, V18 = polychoric_matrix(X18, w, ncats)
        mu = np.nanmean(X18, 0); sd = np.nanstd(X18, 0)
        Z18 = (X18 - mu) / np.where(sd > 0, sd, 1)
        Z18 = np.nan_to_num(Z18)

        pfit, p_meta = fit_p(R18, V18, names18, N, seed + k_imp)
        z_p = Z18[:, [names18.index(v) for v in pnames]]
        fs_p_all, errP = cfa.bartlett(pfit, z_p)
        fs_p = fs_p_all[:, pfit.spec.factors.index("p")]
        err_p = errP[pfit.spec.factors.index("p")][pfit.spec.factors.index("p")]

        full_part = C.f_partition()
        full_fit = cfa.fit_dwls(cfa.bifactor_spec(full_part),
                                R18[np.ix_(range(len(fnames)),
                                           range(len(fnames)))], N,
                                V18[np.ix_(range(len(fnames)),
                                           range(len(fnames)))],
                                seed=seed + k_imp)
        fs_full, errF_full = cfa.bartlett(
            full_fit, Z18[:, [names18.index(v) for v in fnames]])
        fs_full_f = fs_full[:, 0]
        err_full = errF_full[0, 0]

        for out_name, spec_o in C.H3_OUTCOMES.items():
            dd = spec_o["lodo_domain"]
            lfit = lodo_fit(R18, V18, names18, N, dd, seed + k_imp)
            keep = lfit.spec.items
            z_l = Z18[:, [names18.index(v) for v in keep]]
            fs_l, errL = cfa.bartlett(lfit, z_l)
            fs_f = fs_l[:, 0]; err_f = errL[0, 0]
            om_f = cfa.omega_ecv(lfit)["omega_h"]
            match = abs(om_f - p_meta["omega"]) < cfg["reliability_match_tol"]
            match_report[out_name] = dict(omega_f=float(om_f),
                                          omega_p=float(p_meta["omega"]),
                                          matched=bool(match))
            y2, y1 = f"{spec_o['var']}_w2", f"{spec_o['var']}_w1"
            res = croon_regression(panel, y2, y1, fs_f, err_f, fs_p, err_p)
            per_outcome[out_name]["beta"].append(res["beta_f"])
            per_outcome[out_name]["beta_p"].append(res["beta_p"])
            resF = croon_regression(panel, y2, y1, fs_full_f, err_full,
                                    fs_p, err_p)
            per_outcome[out_name]["beta_full"].append(resF["beta_f"])
            y3 = f"{spec_o['var']}_w3"
            if y3 in panel.columns:
                res3 = croon_regression(panel, y3, y1, fs_f, err_f, fs_p, err_p)
                per_outcome[out_name]["beta_w3"].append(res3["beta_f"])
            if k_imp == 0:
                se_boot[out_name] = psu_bootstrap(panel, res, B, seed)
                cosines[out_name] = residualized_cosine(
                    R18, V18, names18, N, dd, seed)
                Yw = panel[[y1, y2, f"{spec_o['var']}_w3"]].to_numpy(float)
                Fw = np.column_stack([fs_full_f, fs_full_f, fs_full_f])
                # RI-CLPM needs wave-specific f scores: rebuild from wave cols
                Fw = _wave_scores(panel, full_fit, fnames, mu[:len(fnames)],
                                  sd[:len(fnames)])
                riclpm_out[out_name] = riclpm(Fw, Yw, w)

    # pooled inference
    results, pvals = {}, {}
    for out_name in C.H3_OUTCOMES:
        pool = rubin_pool(per_outcome[out_name]["beta"])
        se_w = se_boot.get(out_name, np.nan)
        m = len(per_outcome[out_name]["beta"])
        se_tot = float(np.sqrt(se_w ** 2 + (1 + 1 / m) * pool["between_var"])) \
            if np.isfinite(se_w) else np.nan
        z = pool["point"] / se_tot if se_tot and np.isfinite(se_tot) else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        pvals[out_name] = p
        results[out_name] = dict(
            beta=pool["point"], se=se_tot, p=p,
            beta_p_composite=float(np.mean(per_outcome[out_name]["beta_p"])),
            beta_full_f=float(np.mean(per_outcome[out_name]["beta_full"])),
            beta_w3=float(np.mean(per_outcome[out_name]["beta_w3"]))
            if per_outcome[out_name]["beta_w3"] else np.nan,
            cosine=cosines.get(out_name, np.nan),
            match=match_report.get(out_name),
            riclpm=riclpm_out.get(out_name))
    fdr = bh_fdr(pvals, cfg["fdr_q"])
    confirming = [k for k, r in results.items()
                  if r["beta"] >= cfg["beta_min"] and fdr[k]
                  and r["match"]["matched"]
                  and r["cosine"] >= cfg["cosine_confirm"]]
    any_neg = any(r["beta"] <= -cfg["beta_min"] and fdr[k]
                  for k, r in results.items())
    opp_riclpm = [k for k, r in results.items()
                  if r["riclpm"] and r["riclpm"]["converged"]
                  and np.sign(r["riclpm"]["c_yx"]) < 0 and r["beta"] > 0]
    if len(confirming) >= cfg["n_outcomes_min"] and not any_neg:
        verdict = "confirmed"
    elif any_neg or all(not fdr[k] or results[k]["beta"] < cfg["beta_min"]
                        for k in results) \
            or all(results[k]["cosine"] < cfg["cosine_disconfirm"]
                   for k in confirming or results):
        verdict = "disconfirmed"
    else:
        verdict = "partial"
    return dict(verdict=verdict, results=results, fdr=fdr,
                confirming=confirming, riclpm_opposite=opp_riclpm,
                p_composite=p_meta, between_country=None,
                note="full-f betas are secondary, part-whole contaminated "
                     "by construction (prereg language).")


def _wave_scores(panel, fit, fnames, mu1, sd1):
    """Bartlett f scores at each wave under W1 loadings (wave-invariant
    scoring rule -- named approximation for the RI-CLPM sensitivity)."""
    out = []
    for wv in (1, 2, 3):
        cols = [f"{v}_w{wv}" for v in fnames]
        X = panel[cols].to_numpy(float)
        Z = (X - mu1) / np.where(sd1 > 0, sd1, 1)
        Z = np.nan_to_num(Z)
        fs, _ = cfa.bartlett(fit, Z)
        out.append(fs[:, 0])
    return np.column_stack(out)

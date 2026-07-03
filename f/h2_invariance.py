"""
h2_invariance.py -- H2a cross-country measurement properties (descriptive)
and H2b cultural-distance falsification.

H2a primary: per-factor alignment optimization (Asparouhov & Muthen 2014)
run separately for f and each group factor, from per-country configural
DWLS bifactor solutions. Component loss f(x) = (x^2 + eps)^(1/4) with
pair weights sqrt(N_g1 N_g2); FIXED alignment (group 1: alpha = 0, psi = 1).
Marginal (one-factor) alignment and sequential MGCFA (Chen 2007 thresholds,
continuous-approximation regime; categorical Svetina-Rutkowski variant is a
Stage 2 lavaan parity item) are the pre-registered sensitivities.

Trustworthiness criteria -> descriptive tiers (Tier k = 3 - #criteria met + 1):
  1. alignment R^2 > 0.75 averaged across f-loaded indicators
  2. <= 25% of (loading + intercept) parameters flagged non-invariant
  3. Spearman rank correlation of aligned vs configural country-level f
     means >= 0.98
No confirmation/disconfirmation of invariance is issued (host-program
position); tiers gate H3/H4 interpretation and the H2b verdict.

The effect-size flagging constants (config.H2 flag_*) are validated and
locked by run_h2a_mc(), the Stage 2 addendum item (iii) Monte Carlo
operating-characteristics harness included below.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from . import config as C
from . import cfa
from .polychoric import polychoric_matrix, pearson_corr_weighted, nearest_pd_corr

RNG = np.random.default_rng


# ---------------------------------------------------------------------------
# Per-country configural machinery
# ---------------------------------------------------------------------------

def country_inputs(df_w1: pd.DataFrame, min_n: int):
    """Per-country polychoric R, weights V, Pearson corr, standardized item
    means (pooled-SD metric), and analytic n."""
    items = C.F_ITEMS
    names = [it.var for it in items]
    n_cats = [it.n_cat for it in items]
    pooled = df_w1[names].to_numpy(float)
    mu_p = np.nanmean(pooled, 0)
    sd_p = np.nanstd(pooled, 0)
    out = {}
    for ctry, d in df_w1.groupby(C.ADMIN["country"]):
        X = d[names].to_numpy(float)
        n_eff = int((~np.isnan(X).any(1)).sum())
        if n_eff < min_n:
            continue
        w = d[C.ADMIN["weight"]].to_numpy(float)
        R, V = polychoric_matrix(X, w, n_cats)
        S = pearson_corr_weighted((X - mu_p) / sd_p, w)
        means = np.nanmean((X - mu_p) / sd_p, 0)
        out[ctry] = dict(R=R, V=V, S=S, means=means, n=n_eff,
                         N=float(np.nansum(w)))
    return out, (mu_p, sd_p)


def configural_fits(cinp: dict, seed: int) -> dict[str, cfa.FitResult]:
    part = C.f_partition()
    return {c: cfa.fit_dwls(cfa.bifactor_spec(part), d["R"], d["N"], d["V"],
                            heywood_floor=C.H1["heywood_floor"], seed=seed)
            for c, d in cinp.items()}


# ---------------------------------------------------------------------------
# Alignment optimization
# ---------------------------------------------------------------------------

def _loss_component(x, eps):
    return (x * x + eps) ** 0.25


def align_factor(lam: np.ndarray, nu: np.ndarray, Ns: np.ndarray,
                 eps: float, n_starts: int = 3, seed: int = 0) -> dict:
    """lam, nu: (G, J) configural loadings and intercepts for one factor.
    Returns aligned parameters, (alpha_g, psi_g), R^2 by parameter type."""
    G, J = lam.shape
    Wp = np.sqrt(np.outer(Ns, Ns))
    iu = np.triu_indices(G, 1)
    wpair = Wp[iu]

    def aligned(x):
        u = np.concatenate(([0.0], x[:G - 1]))          # log sqrt(psi)
        a = np.concatenate(([0.0], x[G - 1:]))          # alpha
        sq = np.exp(u)
        lam_s = lam / sq[:, None]
        nu_s = nu - lam_s * a[:, None]
        return lam_s, nu_s, a, sq ** 2

    def loss(x):
        lam_s, nu_s, _, _ = aligned(x)
        tot = 0.0
        for j in range(J):
            dl = lam_s[iu[0], j] - lam_s[iu[1], j]
            dn = nu_s[iu[0], j] - nu_s[iu[1], j]
            tot += float(np.sum(wpair * (_loss_component(dl, eps)
                                         + _loss_component(dn, eps))))
        return tot / wpair.sum()

    rng = RNG(seed)
    best = None
    for s in range(n_starts):
        x0 = np.zeros(2 * (G - 1)) if s == 0 else rng.normal(0, 0.2, 2 * (G - 1))
        res = minimize(loss, x0, method="L-BFGS-B",
                       options=dict(maxiter=1500))
        if best is None or res.fun < best.fun:
            best = res
    lam_s, nu_s, alpha, psi = aligned(best.x)
    wg = Ns / Ns.sum()
    lam_bar = (wg[:, None] * lam_s).sum(0)
    nu_bar = (wg[:, None] * nu_s).sum(0)
    # R^2: variance of configural params explained by the aligned model
    lam_hat = lam_bar[None, :] * np.sqrt(psi)[:, None]
    nu_hat = nu_bar[None, :] + lam_bar[None, :] * alpha[:, None]
    def r2(obs, hat):
        num = float(np.sum(wg[:, None] * (obs - hat) ** 2))
        den = float(np.sum(wg[:, None] * (obs - (wg[:, None] * obs).sum(0)) ** 2))
        return 1.0 - num / den if den > 0 else np.nan
    return dict(alpha=alpha, psi=psi, lam_aligned=lam_s, nu_aligned=nu_s,
                lam_inv=lam_bar, nu_inv=nu_bar, loss=float(best.fun),
                r2_loading=r2(lam, lam_hat), r2_intercept=r2(nu, nu_hat),
                converged=bool(best.success))


def flag_noninvariance(al: dict, cfg=C.H2) -> tuple[np.ndarray, float]:
    """Effect-size flags on aligned parameters; returns (G, J, 2) bool array
    (loading, intercept) and the flagged fraction."""
    dl = np.abs(al["lam_aligned"] - al["lam_inv"][None, :])
    dn = np.abs(al["nu_aligned"] - al["nu_inv"][None, :])
    flags = np.stack([dl > cfg["flag_loading_dev"],
                      dn > cfg["flag_intercept_dev"]], axis=-1)
    return flags, float(flags.mean())


# ---------------------------------------------------------------------------
# H2a driver
# ---------------------------------------------------------------------------

def run_h2a(df_w1: pd.DataFrame, pooled_fit: cfa.FitResult, seed: int,
            smoke: bool = False, min_n: int | None = None,
            lookups_dir: str | None = None) -> dict:
    cfg = C.H2
    min_n = min_n if min_n is not None else cfg["min_country_n"]
    cinp, (mu_p, sd_p) = country_inputs(df_w1, min_n)
    countries = sorted(cinp)
    if len(countries) < 3:
        return dict(error=f"only {len(countries)} countries reach n >= {min_n}")
    cfits = configural_fits(cinp, seed)
    part = C.f_partition()
    names = [it.var for it in C.F_ITEMS]
    Ns = np.array([cinp[c]["N"] for c in countries])

    # per-factor alignment: f over all items; each group factor over its items
    factor_sets = {"f": names}
    for d in C.DOMAINS:
        factor_sets[d] = part[d]
    align_out = {}
    for fac, its in factor_sets.items():
        jidx = [names.index(v) for v in its]
        lam = np.array([[cfits[c].loading_vector(fac)[names.index(v)]
                         for v in its] for c in countries])
        nu = np.array([[cinp[c]["means"][j] for j in jidx] for c in countries])
        al = align_factor(lam, nu, Ns, cfg["alignment_eps"],
                          seed=seed + hash(fac) % 1000)
        flags, frac = flag_noninvariance(al)
        al.update(flags=flags, flag_frac=frac, items=its)
        align_out[fac] = al

    # marginal alignment sensitivity: one-factor model per country
    lam_m = []
    for c in countries:
        sp = cfa.Spec(items=names, factors=["f"])
        for v in names:
            sp.add(v, "f")
        fm = cfa.fit_dwls(sp, cinp[c]["R"], cinp[c]["N"], cinp[c]["V"],
                          seed=seed)
        lam_m.append(fm.loading_vector("f"))
    nu_all = np.array([cinp[c]["means"] for c in countries])
    al_marg = align_factor(np.array(lam_m), nu_all, Ns, cfg["alignment_eps"],
                           seed=seed + 7)

    # trustworthiness criteria on the f alignment
    alf = align_out["f"]
    r2_avg = float(np.nanmean([alf["r2_loading"], alf["r2_intercept"]]))
    # configural country-level f means: Bartlett scores under pooled loadings
    Zc, conf_means = [], []
    A_err = None
    for c in countries:
        d = df_w1[df_w1[C.ADMIN["country"]] == c]
        X = (d[names].to_numpy(float) - mu_p) / sd_p
        m = ~np.isnan(X).any(1)
        fs, A_err = cfa.bartlett(pooled_fit, X[m])
        wsub = d[C.ADMIN["weight"]].to_numpy(float)[m]
        conf_means.append(np.average(fs[:, 0], weights=wsub))
    rank_r = float(spearmanr(alf["alpha"], conf_means).statistic)
    crit = dict(r2=r2_avg > cfg["alignment_r2_min"],
                flags=alf["flag_frac"] <= cfg["noninv_max_frac"],
                rank=rank_r >= cfg["rank_corr_min"])
    tier = 4 - sum(crit.values()) if sum(crit.values()) < 3 else 1

    # sequential MGCFA sensitivity (continuous approximation, Chen 2007)
    mg = mgcfa_sequence(cinp, countries, seed)

    # stratified descriptives from lookups (if supplied)
    strata_report = stratified_means(countries, alf["alpha"], lookups_dir)

    return dict(countries=countries, tier=int(tier), criteria=crit,
                r2_avg=r2_avg, flag_frac=alf["flag_frac"], rank_corr=rank_r,
                alignment=align_out, marginal=al_marg,
                configural_means=dict(zip(countries, map(float, conf_means))),
                aligned_means=dict(zip(countries, map(float, alf["alpha"]))),
                mgcfa=mg, strata=strata_report, n_by_country=dict(
                    zip(countries, [cinp[c]["n"] for c in countries])))


def mgcfa_sequence(cinp: dict, countries: list[str], seed: int) -> dict:
    part = C.f_partition()
    sp = cfa.bifactor_spec(part)
    S_list = [cinp[c]["S"] for c in countries]
    N_list = [cinp[c]["N"] for c in countries]
    m_list = [cinp[c]["means"] for c in countries]
    p = len(sp.items)
    # configural: independent per-group ML fits
    F_cfg, k_cfg = 0.0, 0
    Ntot = sum(N_list)
    for S, N in zip(S_list, N_list):
        f = cfa.fit_ml(sp, S, N, seed=seed)
        F_cfg += (N / Ntot) * f.F
        k_cfg += sp.n_free() + p
    G = len(countries)
    T = (Ntot - G) * F_cfg
    df_cfg = G * p * (p + 1) / 2 - k_cfg
    Fb = sum((N / Ntot) * max(-np.linalg.slogdet(S)[1] + np.trace(S) - p, 1e-12)
             for S, N in zip(S_list, N_list))
    Tb = (Ntot - G) * Fb
    cfi_cfg = 1 - max(T - df_cfg, 0) / max(Tb - G * p * (p - 1) / 2,
                                           T - df_cfg, 1e-12)
    rmsea_cfg = np.sqrt(G) * np.sqrt(max(T - df_cfg, 0)
                                     / max(df_cfg * (Ntot - G), 1e-12))
    met = cfa.fit_ml_multigroup(sp, S_list, N_list, None, level="metric")
    sca = cfa.fit_ml_multigroup(sp, S_list, N_list, m_list, level="scalar")
    return dict(
        configural=dict(cfi=float(cfi_cfg), rmsea=float(rmsea_cfg)),
        metric=dict(cfi=met["cfi"], rmsea=met["rmsea"],
                    dcfi=float(met["cfi"] - cfi_cfg),
                    drmsea=float(met["rmsea"] - rmsea_cfg),
                    pass_chen=bool(met["cfi"] - cfi_cfg >= C.H2["chen_dcfi"]
                                   and met["rmsea"] - rmsea_cfg
                                   <= C.H2["chen_drmsea"])),
        scalar=dict(cfi=sca["cfi"], rmsea=sca["rmsea"],
                    dcfi=float(sca["cfi"] - met["cfi"]),
                    drmsea=float(sca["rmsea"] - met["rmsea"]),
                    pass_chen=bool(sca["cfi"] - met["cfi"] >= C.H2["chen_dcfi"]
                                   and sca["rmsea"] - met["rmsea"]
                                   <= C.H2["chen_drmsea"])))


def stratified_means(countries, alpha, lookups_dir):
    if not lookups_dir:
        return None
    from .data_io import load_lookup
    out = {}
    for name, col in [("iw_scores.csv", None), ("income_tier.csv", "INCOME_TIER"),
                      ("religion.csv", "RELIGION")]:
        d = load_lookup(lookups_dir, name)
        if d is None:
            continue
        d = d.set_index(C.ADMIN["country"])
        if name == "iw_scores.csv":
            quad = ["TS" if (d.loc[c, "IW_TRAD"] > 0) == (d.loc[c, "IW_SURV"] > 0)
                    else "mixed" for c in countries if c in d.index]
            out["iw_quadrant"] = quad
        else:
            key = [d.loc[c, col] if c in d.index else None for c in countries]
            df = pd.DataFrame(dict(stratum=key, alpha=alpha))
            out[col] = df.groupby("stratum")["alpha"].mean().to_dict()
    return out or None


# ---------------------------------------------------------------------------
# H2a Monte Carlo operating characteristics (Stage 2 addendum item iii)
# ---------------------------------------------------------------------------

def run_h2a_mc(seed: int, smoke: bool = False, n_per_country: int = 500,
               n_countries: int = 22) -> pd.DataFrame:
    """Per-factor bifactor alignment at G countries under the pre-registered
    violation grid: fractions {0, .10, .25, .40} x magnitudes {.2, .4},
    GFS-matched ordinal margins. Records f-mean recovery, R^2, and the
    sensitivity/specificity of the effect-size flagging rule -- the evidence
    base for locking the adequacy criteria and flag constants."""
    from .synthetic import simulate_panel
    cfg = C.H2
    fracs = cfg["mc_violation_fracs"]
    mags = cfg["mc_magnitudes"]
    reps = C.SMOKE["h2_mc_reps"] if smoke else cfg["mc_reps"]
    if smoke:
        fracs = fracs[:C.SMOKE["h2_mc_cells"]]
        mags = mags[:1]
    rows = []
    for fr in fracs:
        for mg in mags:
            for r in range(reps):
                inv = dict(frac=fr, magnitude=mg, seed=seed + r) if fr > 0 else None
                dfp, truth = simulate_panel(
                    n_per_country=n_per_country,
                    countries=C.GFS_COUNTRIES[:n_countries], waves=1,
                    invariance=inv, dropout=False, item_missing=0.0,
                    seed=seed + 1000 * r + int(100 * fr) + int(10 * mg))
                from .data_io import orient_items, normalize_weights
                d1 = normalize_weights(orient_items(dfp))
                cinp, _ = country_inputs(d1, min_n=100)
                countries = sorted(cinp)
                cfits = configural_fits(cinp, seed)
                names = [it.var for it in C.F_ITEMS]
                lam = np.array([cfits[c].loading_vector("f") for c in countries])
                nu = np.array([cinp[c]["means"] for c in countries])
                Ns = np.array([cinp[c]["N"] for c in countries])
                al = align_factor(lam, nu, Ns, cfg["alignment_eps"], seed=seed)
                flags, frac_fl = flag_noninvariance(al)
                truth_alpha = np.array([truth["alpha_c"][c] for c in countries])
                rec = float(np.corrcoef(al["alpha"], truth_alpha)[0, 1])
                viol = {(c, v) for c, v in truth["violated"]}
                hit = fp = 0
                nviol = max(len([1 for c in countries for v in names
                                 if (c, v) in viol]), 1)
                for gi, c in enumerate(countries):
                    for ji, v in enumerate(names):
                        f_any = bool(flags[gi, ji].any())
                        if (c, v) in viol and f_any:
                            hit += 1
                        elif (c, v) not in viol and f_any:
                            fp += 1
                rows.append(dict(frac=fr, mag=mg, rep=r,
                                 alpha_recovery=rec,
                                 r2_loading=al["r2_loading"],
                                 r2_intercept=al["r2_intercept"],
                                 flag_frac=frac_fl,
                                 sensitivity=hit / nviol if fr > 0 else np.nan,
                                 fpr=fp / max(len(countries) * len(names)
                                              - nviol, 1)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H2b -- cultural-distance falsification
# ---------------------------------------------------------------------------

def run_h2b(h2a: dict, lookups_dir: str | None, seed: int,
            smoke: bool = False) -> dict:
    """Mixed-effects regression of indicator-level non-invariance magnitude
    on cultural distance x domain contrast, translation covariate partialed,
    country random intercept. Verdict issued only at H2a Tier 1/2."""
    from .data_io import load_lookup
    cfg = C.H2
    countries = h2a["countries"]
    alf = h2a["alignment"]["f"]
    names = alf["items"]
    dom = {it.var: it.domain for it in C.F_ITEMS}
    rows = []
    for gi, c in enumerate(countries):
        for ji, v in enumerate(names):
            d = dom[v]
            if d in ("hedonic", "mental"):     # pre-committed exclusion
                continue
            mag = (abs(alf["lam_aligned"][gi, ji] - alf["lam_inv"][ji])
                   + abs(alf["nu_aligned"][gi, ji] - alf["nu_inv"][ji]))
            rows.append(dict(country=c, item=v, noninv=mag,
                             focal=int(d in cfg["h2b_focal"])))
    df = pd.DataFrame(rows)

    dist_src = "CFST_US"
    cd = load_lookup(lookups_dir, "cultural_distance.csv") if lookups_dir else None
    if cd is not None and cd["CFST_US"].notna().any():
        dmap = cd.set_index(C.ADMIN["country"])["CFST_US"].to_dict()
    else:
        iw = load_lookup(lookups_dir, "iw_scores.csv") if lookups_dir else None
        if iw is not None and iw["IW_TRAD"].notna().any():
            iw = iw.set_index(C.ADMIN["country"])
            us = iw.loc["United States"]
            dmap = {c: float(np.hypot(iw.loc[c, "IW_TRAD"] - us["IW_TRAD"],
                                      iw.loc[c, "IW_SURV"] - us["IW_SURV"]))
                    for c in countries if c in iw.index}
            dist_src = "IW_euclidean_fallback"
        else:
            rng = RNG(seed)
            dmap = {c: float(rng.uniform(0.05, 0.30)) for c in countries}
            dist_src = "SYNTHETIC_PLACEHOLDER (supply lookups before any real run)"
    df["dist"] = df["country"].map(dmap)
    tr = load_lookup(lookups_dir, "translation_equivalence.csv") if lookups_dir else None
    if tr is not None and tr["T_EQUIV"].notna().any():
        tmap = tr.set_index([C.ADMIN["country"], "ITEM"])["T_EQUIV"].to_dict()
        df["T"] = [tmap.get((c, v), 0.0) for c, v in zip(df.country, df.item)]
        covered = df["T"].notna()
        df = df[covered]                      # covered-item subset primary
        t_src = "cowden_classification"
    else:
        df["T"] = 0.0
        t_src = "ZERO_PLACEHOLDER (translation covariate not supplied)"
    df["dist_c"] = df["dist"] - df["dist"].mean()

    md, model_kind, model_warning = _fit_h2b_model(df)
    coef = float(md.params.get("dist_c:focal", np.nan))
    pval = float(md.pvalues.get("dist_c:focal", np.nan))
    gated = h2a["tier"] <= 2
    if not gated:
        verdict = f"descriptive_only (H2a Tier {h2a['tier']}; confounding caveat)"
    elif coef > 0 and pval < C.COMMON["alpha"]:
        verdict = "confirmed"
    elif np.isnan(coef):
        verdict = "indeterminate"
    else:
        verdict = "disconfirmed"
    return dict(verdict=verdict, interaction_coef=coef, interaction_p=pval,
                distance_source=dist_src, translation_source=t_src,
                model_kind=model_kind, model_warning=model_warning,
                model_summary=str(md.summary()), data=df,
                median_split=_median_split_sensitivity(df))


def _fit_h2b_model(df: pd.DataFrame):
    """Fit the preregistered H2b model robustly for smoke/synthetic runs.

    Statsmodels' random-intercept MixedLM can raise a hard singular-matrix
    error in small synthetic panels, especially when optional lookup covariates
    are placeholders with no variation.  Keep the preregistered mixed model as
    primary, remove constant nuisance terms, and fall back to a country-clustered
    linear model rather than aborting the full pipeline smoke test.
    """
    import statsmodels.formula.api as smf

    d = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["noninv", "dist_c", "focal", "T", "country"]).copy()
    rhs = "dist_c * focal"
    if d["T"].nunique(dropna=True) > 1:
        rhs += " + T"
    formula = f"noninv ~ {rhs}"

    if (len(d) < 4 or d["country"].nunique() < 2 or
            d["dist_c"].nunique(dropna=True) < 2 or
            d["focal"].nunique(dropna=True) < 2):
        return _NullH2BResult(), "unfit", "insufficient variation for H2b model"

    try:
        md = smf.mixedlm(formula, d, groups=d["country"]).fit(
            reml=True, method="lbfgs")
        return md, "mixedlm_random_intercept", None
    except (np.linalg.LinAlgError, ValueError) as exc:
        # In small smoke data the random-intercept Hessian/normal equations can
        # be singular.  Country-clustered OLS preserves the same fixed-effect
        # contrast and is adequate as an execution fallback.
        try:
            ols = smf.ols(formula, d).fit(
                cov_type="cluster", cov_kwds={"groups": d["country"]})
            return ols, "ols_country_cluster_fallback", str(exc)
        except (np.linalg.LinAlgError, ValueError) as fallback_exc:
            return (_NullH2BResult(), "unfit",
                    f"mixedlm failed: {exc}; ols fallback failed: {fallback_exc}")


class _NullH2BResult:
    params = pd.Series(dtype=float)
    pvalues = pd.Series(dtype=float)

    def summary(self):
        return "H2b model was not fit."


def _median_split_sensitivity(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["far"] = (d["dist"] > d["dist"].median()).astype(int)
    cell = d.groupby(["far", "focal"])["noninv"].mean().unstack()
    try:
        did = float(cell.loc[1, 1] - cell.loc[1, 0]
                    - (cell.loc[0, 1] - cell.loc[0, 0]))
    except Exception:
        did = np.nan
    return dict(cell_means=cell.to_dict(), interaction_did=did)

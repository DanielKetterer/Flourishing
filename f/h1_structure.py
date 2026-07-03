"""
h1_structure.py -- H1: structural replication of the bifactor beyond fitting
propensity, against the pre-registered comparators.

Legs (Inference Criteria, all four must clear for confirmation):
  (i)   omega_h >= 0.80 with CFI >= 0.90, RMSEA <= 0.08, SRMR <= 0.08
  (ii)  partition-specificity gaps (Envelope B) >= 0.05 on omega_h and ECV,
        loading-magnitude check, value-permutation criterion (Envelope A)
  (iii) f-loading direction stable across hedonic specifications
        (cosine > 0.95) and standardized f-loading SD across blocks >= 0.10
  (iv)  delta-BIC >= 10 over the SWB-extended one-factor model with loading
        cosine to SWB < 0.90, and delta-BIC >= 10 over MAGNA or ARI >= 0.60

ENVELOPE DIRECTION (flagged in config and report): permuting the item->domain
assignment on intact data forces true domain covariance into the only common
channel left, f. In this implementation's DGP checks that RAISES omega_h/ECV
and LOWERS fit and group-loading magnitude relative to the true partition.
Both gap directions and percentile placements are therefore always recorded;
the verdict consumes config.H1['envelope_gap_sign'], a Stage-2 reconciliation
item against the authoritative v8 gap definitions.
"""

from __future__ import annotations
import numpy as np
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.metrics import adjusted_rand_score

from . import config as C
from . import cfa
from .polychoric import polychoric_matrix, nearest_pd_corr
from .data_io import rubin_pool

RNG = np.random.default_rng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def standardized_items(df, items, w=None):
    """Weighted-standardized continuous approximation of the ordinal items
    (complete cases) -- used for scores, Vuong, and the MAGNA likelihood."""
    X = df[[it.var for it in items]].to_numpy(float)
    m = ~np.isnan(X).any(1)
    X = X[m]
    ww = np.ones(X.shape[0]) if w is None else np.asarray(w)[m]
    mu = np.average(X, axis=0, weights=ww)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=ww))
    return (X - mu) / np.where(sd > 0, sd, 1.0), ww


def fit_bifactor(R, V, N, partition, hedonic_key=None, drop=(), s1=None,
                 seed=0):
    sp = cfa.bifactor_spec(partition, hedonic_key=hedonic_key,
                           drop_domains=drop, s1_reference=s1)
    idx = None  # spec order == R order handled by caller subsetting
    fit = cfa.fit_dwls(sp, R, N, V, heywood_floor=C.H1["heywood_floor"],
                       seed=seed)
    return fit


def subset(R, V, items_all, items_keep):
    idx = [items_all.index(v) for v in items_keep]
    return R[np.ix_(idx, idx)], (V[np.ix_(idx, idx)] if V is not None else None)


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------

STATS = ("omega_h", "ecv", "cfi", "rmsea", "srmr", "mean_abs_group_loading")


def _collect(fit):
    oe = cfa.omega_ecv(fit)
    fi = cfa.fit_indices(fit)
    return dict(omega_h=oe["omega_h"], ecv=oe["ecv"], cfi=fi["cfi"],
                rmsea=fi["rmsea"], srmr=fi["srmr"],
                mean_abs_group_loading=oe["mean_abs_group_loading"],
                f_loading_block_sd=oe["f_loading_block_sd"])


def envelope_A(df_w1, items, w, N, partition, reps, seed=0):
    """Value permutation of each indicator across persons (Bonifay-Cai
    fitting-propensity envelope): destroys all inter-item structure, keeps
    margins. Polychorics recomputed each replicate."""
    rng = RNG(seed)
    X = df_w1[[it.var for it in items]].to_numpy(float)
    n_cats = [it.n_cat for it in items]
    recs = []
    for r in range(reps):
        Xp = X.copy()
        for j in range(Xp.shape[1]):
            rng.shuffle(Xp[:, j])
        R, V = polychoric_matrix(Xp, w, n_cats)
        fit = fit_bifactor(R, V, N, partition, seed=seed + r)
        recs.append(_collect(fit))
    return recs


def envelope_B(R, V, N, partition, reps, seed=0, permute_hedonic=False):
    """Partition-specificity envelope: random item->domain reassignment on
    intact data (block-size multiset preserved). Hedonic pair stays f-only
    unless permute_hedonic=True (config decision, documented)."""
    rng = RNG(seed)
    fixed_hed = partition.get("hedonic", []) if not permute_hedonic else []
    pool = [v for d, vs in partition.items() for v in vs
            if d != "hedonic" or permute_hedonic]
    sizes = {d: len(vs) for d, vs in partition.items() if d != "hedonic"}
    recs = []
    for r in range(reps):
        perm = list(pool)
        rng.shuffle(perm)
        part = {"hedonic": list(fixed_hed)}
        k = 0
        for d, s in sizes.items():
            part[d] = perm[k:k + s]
            k += s
        if permute_hedonic:
            part["hedonic"] = perm[k:]
        fit = fit_bifactor(R, V, N, part, seed=seed + r)
        recs.append(_collect(fit))
    return recs


def gap_report(observed, env_records, q):
    """Both-direction gaps + percentile placement for every recorded stat."""
    out = {}
    for s in STATS:
        env = np.array([r[s] for r in env_records])
        lo, hi = np.quantile(env, [1 - q, q])
        out[s] = dict(
            observed=float(observed[s]), env_mean=float(env.mean()),
            env_q_lo=float(lo), env_q_hi=float(hi),
            gap_above=float(observed[s] - hi),   # obs exceeds upper envelope
            gap_below=float(lo - observed[s]),   # obs sits below lower envelope
            percentile=float((env < observed[s]).mean()),
        )
    return out


# ---------------------------------------------------------------------------
# MAGNA / network comparator
# ---------------------------------------------------------------------------

def gaussian_m2ll(Z, K):
    N, p = Z.shape
    S = (Z.T @ Z) / N
    sgn, logdetK = np.linalg.slogdet(K)
    return N * (p * np.log(2 * np.pi) - logdetK + float(np.sum(K * S)))


def fit_magna(Z, seed=0):
    """Regularized Gaussian network on the same items: GraphicalLasso with
    EBIC (gamma = 0.5) selection over the CV alpha path -- the standard
    psychonetrics-style GGM estimate this Python deposit uses in place of
    MAGNA proper; the R cross-check is a Stage 2 parity item."""
    gl = GraphicalLassoCV(cv=3).fit(Z)
    alphas = np.unique(np.concatenate([gl.cv_results_["alphas"],
                                       [gl.alpha_]]))
    N, p = Z.shape
    best = None
    for a in alphas:
        try:
            m = GraphicalLasso(alpha=a, max_iter=200).fit(Z)
        except Exception:
            continue
        K = m.precision_
        E = int((np.abs(K[np.triu_indices(p, 1)]) > 1e-8).sum())
        ebic = gaussian_m2ll(Z, K) + E * np.log(N) + 4 * 0.5 * E * np.log(p)
        if best is None or ebic < best[0]:
            best = (ebic, K, E, a)
    ebic, K, E, a = best
    return dict(K=K, edges=E, alpha=float(a), ebic=float(ebic),
                m2ll=float(gaussian_m2ll(Z, K)), n_params=E + p)


def network_communities(K, items, seed=0):
    pc = -K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
    np.fill_diagonal(pc, 0.0)
    W = np.abs(pc) * (np.abs(pc) > 1e-8)
    method = None
    labels = None
    try:
        import igraph as ig
        g = ig.Graph.Weighted_Adjacency(W.tolist(), mode="undirected",
                                        attr="weight", loops=False)
        comm = g.community_spinglass(weights="weight", spins=8)
        labels = np.array(comm.membership)
        method = "spinglass"
    except Exception:
        try:
            import networkx as nx
            g = nx.from_numpy_array(W)
            comms = nx.community.greedy_modularity_communities(g, weight="weight")
            labels = np.zeros(len(items), int)
            for ci, cset in enumerate(comms):
                for node in cset:
                    labels[node] = ci
            method = "greedy_modularity_fallback"
        except Exception:
            from sklearn.cluster import SpectralClustering
            labels = SpectralClustering(n_clusters=7, affinity="precomputed",
                                        random_state=seed).fit_predict(W + 1e-6)
            method = "spectral_fallback"
    return labels, method


def magna_cv(Z, holdout, seed=0):
    rng = RNG(seed)
    n = Z.shape[0]
    te = rng.random(n) < holdout
    tr = ~te
    net = fit_magna(Z[tr], seed=seed)
    K = net["K"]
    # heldout Gaussian log-likelihood + nodewise predictive R2
    ll_te = -0.5 * gaussian_m2ll(Z[te], K) / te.sum()
    B = -K / np.diag(K)[:, None]
    np.fill_diagonal(B, 0.0)
    pred = Z[te] @ B.T
    r2 = 1 - np.mean((Z[te] - pred) ** 2, 0) / np.maximum(Z[te].var(0), 1e-9)
    return dict(heldout_ll_per_case=float(ll_te),
                nodewise_r2_mean=float(np.mean(r2)), train_net=net, test=te)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_h1(imputed_w1: list, seed: int, smoke: bool = False) -> dict:
    cfg = C.H1
    reps = C.SMOKE["envelope_reps"] if smoke else cfg["envelope_reps"]
    part = C.f_partition()
    items = C.F_ITEMS
    names = [it.var for it in items]
    n_cats = [it.n_cat for it in items]
    w_all = [d[C.ADMIN["weight"]].to_numpy(float) for d in imputed_w1]

    # --- polychorics per imputation, primary + alternatives, Rubin pooling
    prim_stats, altA_stats, altB_stats, R_list, V_list, fits = [], [], [], [], [], []
    for k, d in enumerate(imputed_w1):
        X = d[names].to_numpy(float)
        R, V = polychoric_matrix(X, w_all[k], n_cats)
        R_list.append(R); V_list.append(V)
        N = float(w_all[k].sum())
        f0 = fit_bifactor(R, V, N, part, seed=seed + k)
        fits.append(f0)
        prim_stats.append({**_collect(f0), "heywood": len(f0.heywood)})
        fA = fit_bifactor(*subset(R, V, names,
                                  [v for v in names if v not in C.HEDONIC]),
                          N, {d_: v for d_, v in part.items()
                              if d_ != "hedonic"}, seed=seed + k)
        altA_stats.append({**_collect(fA), "lam_f": fA.loading_vector("f")})
        fB = fit_bifactor(R, V, N, part, hedonic_key="tau", seed=seed + k)
        altB_stats.append({**_collect(fB), "lam_f": fB.loading_vector("f")})

    fit0 = fits[0]
    pooled = {s: rubin_pool([r[s] for r in prim_stats]) for s in STATS}
    pooled["f_loading_block_sd"] = rubin_pool(
        [r["f_loading_block_sd"] for r in prim_stats])
    lam_f0 = fit0.loading_vector("f")
    nonhed = [i for i, v in enumerate(names) if v not in C.HEDONIC]
    cosA = cfa.cosine(lam_f0[nonhed],
                      np.mean([r["lam_f"] for r in altA_stats], 0))
    cosB = cfa.cosine(lam_f0, np.mean([r["lam_f"] for r in altB_stats], 0))
    hed_agree = (abs(pooled["omega_h"]["point"]
                     - np.mean([r["omega_h"] for r in altA_stats])) < 0.05 and
                 abs(pooled["omega_h"]["point"]
                     - np.mean([r["omega_h"] for r in altB_stats])) < 0.05)

    # --- comparators on the ML continuous-approximation regime
    Z, wz = standardized_items(imputed_w1[0], items, w_all[0])
    S_p = nearest_pd_corr(np.corrcoef(Z, rowvar=False))
    N0 = Z.shape[0]
    sp_bif = cfa.bifactor_spec(part)
    ml_bif = cfa.fit_ml(sp_bif, S_p, N0, seed=seed)
    sp_s1 = cfa.bifactor_spec(part, s1_reference="mental")
    ml_s1 = cfa.fit_ml(sp_s1, S_p, N0, seed=seed)
    swb_items_sub = C.HEDONIC + part["meaning"]           # hedonic + eudaimonic
    sp_swb1 = Spec_onefactor(swb_items_sub)
    sub_idx = [names.index(v) for v in swb_items_sub]
    ml_swb1 = cfa.fit_ml(sp_swb1, S_p[np.ix_(sub_idx, sub_idx)],
                         N0, seed=seed)
    sp_swb_all = Spec_onefactor(names)                    # SWB-extended: all items
    ml_swb = cfa.fit_ml(sp_swb_all, S_p, N0, seed=seed)
    aic_b, bic_b = cfa.aic_bic(ml_bif)
    aic_s1, bic_s1 = cfa.aic_bic(ml_s1)
    aic_sw, bic_sw = cfa.aic_bic(ml_swb)
    vuong = cfa.vuong_test(ml_bif, ml_swb, Z)
    cos_swb = cfa.cosine(np.abs(ml_bif.loading_vector("f")),
                         np.abs(ml_swb.loading_vector("f")))

    net = fit_magna(Z, seed=seed)
    bic_net = net["m2ll"] + net["n_params"] * np.log(N0)
    aic_net = net["m2ll"] + 2 * net["n_params"]
    labels, comm_method = network_communities(net["K"], names, seed=seed)
    block_labels = [it.domain for it in items]            # hedonic = own block
    ari = float(adjusted_rand_score(block_labels, labels))
    cv = magna_cv(Z, cfg["magna_holdout"], seed=seed)
    ml_bif_tr = cfa.fit_ml(sp_bif, nearest_pd_corr(
        np.corrcoef(Z[~cv["test"]], rowvar=False)), int((~cv["test"]).sum()),
        seed=seed)
    Sig_tr = ml_bif_tr.implied()
    Kb = np.linalg.inv(Sig_tr)
    cv_bif_ll = -0.5 * gaussian_m2ll(Z[cv["test"]], Kb) / cv["test"].sum()

    # --- envelopes (imputation 1; documented in README)
    N1 = float(w_all[0].sum())
    envA = envelope_A(imputed_w1[0], items, w_all[0], N1, part, reps,
                      seed=seed + 101)
    envB = envelope_B(R_list[0], V_list[0], N1, part, reps, seed=seed + 202)
    obs = prim_stats[0]
    gaps = dict(A=gap_report(obs, envA, cfg["envelope_quantile"]),
                B=gap_report(obs, envB, cfg["envelope_quantile"]))

    signed_gap = {}
    for s in ("omega_h", "ecv", "cfi"):
        sgn = cfg["envelope_gap_sign"].get(s if s != "cfi" else "cfi", 1)
        g = gaps["B"][s]
        signed_gap[s] = g["gap_above"] if sgn > 0 else g["gap_below"]
    load_check = gaps["B"]["mean_abs_group_loading"]["gap_above"] > 0

    # --- tier verdict
    oh = pooled["omega_h"]["point"]
    fitok = (pooled["cfi"]["point"] >= cfg["cfi_min"]
             and pooled["rmsea"]["point"] <= cfg["rmsea_max"]
             and pooled["srmr"]["point"] <= cfg["srmr_max"])
    leg1 = oh >= cfg["omega_h_confirm"] and fitok
    leg2 = (signed_gap["omega_h"] >= cfg["gap_omega"]
            and signed_gap["ecv"] >= cfg["gap_ecv"] and load_check
            and gaps["A"]["omega_h"]["percentile"] >= cfg["envelope_quantile"])
    leg3 = (min(cosA, cosB) > cfg["hedonic_cosine_min"]
            and pooled["f_loading_block_sd"]["point"] >= cfg["loading_sd_floor"])
    dbic_swb = bic_sw - bic_b
    dbic_net = bic_net - bic_b
    leg4 = (dbic_swb >= cfg["dbic_min"] and cos_swb < cfg["swb_cosine_max"]
            and (dbic_net >= cfg["dbic_min"] or ari >= cfg["ari_confirm"]))
    if leg1 and leg2 and leg3 and leg4:
        verdict = "confirmed"
    elif (oh < cfg["omega_h_partial_lo"] or not fitok
          or min(signed_gap["omega_h"], signed_gap["ecv"]) < 0
          or (dbic_net < -cfg["dbic_min"] and ari < cfg["ari_partial_lo"])):
        verdict = "disconfirmed"
    else:
        verdict = "partial"

    return dict(
        verdict=verdict, legs=dict(leg1=leg1, leg2=leg2, leg3=leg3, leg4=leg4),
        pooled=pooled, heywood=fit0.heywood,
        loadings=dict(items=names, L=fit0.L, factors=fit0.spec.factors),
        alternatives=dict(cos_altA=cosA, cos_altB=cosB, agree=hed_agree,
                          omega_altA=float(np.mean([r["omega_h"] for r in altA_stats])),
                          omega_altB=float(np.mean([r["omega_h"] for r in altB_stats]))),
        eid_s1=dict(bic=bic_s1, dbic_vs_primary=float(bic_s1 - bic_b)),
        swb=dict(aic=aic_sw, bic=bic_sw, dbic=float(dbic_swb),
                 vuong_z=vuong["z"], vuong_z_bic=vuong["z_bic"],
                 cosine=cos_swb, bic_swb_subset=cfa.aic_bic(ml_swb1)[1]),
        magna=dict(aic=aic_net, bic=bic_net, dbic=float(dbic_net), ari=ari,
                   edges=net["edges"], communities=labels.tolist(),
                   comm_method=comm_method,
                   cv_heldout_ll_net=cv["heldout_ll_per_case"],
                   cv_heldout_ll_bifactor=float(cv_bif_ll),
                   nodewise_r2=cv["nodewise_r2_mean"]),
        envelopes=gaps, signed_gap=signed_gap,
        env_records=dict(A=envA, B=envB), observed=obs,
        bic_primary=float(bic_b),
        R_list=R_list, V_list=V_list, fits=fits,
        note_thresholds="0.75 (Hypotheses) vs 0.80 (Inference Criteria) "
                        "omega_h discrepancy encoded per Inference Criteria; "
                        "reconcile in v8 text.",
    )


def Spec_onefactor(items: list[str]) -> cfa.Spec:
    sp = cfa.Spec(items=list(items), factors=["f"])
    for v in items:
        sp.add(v, "f")
    return sp

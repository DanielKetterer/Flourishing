"""
plots.py -- Figures for each hypothesis block. All matplotlib, PNG output.
Every function is defensive: a plotting failure never kills the pipeline.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 140


def _save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=DPI)
    plt.close(fig)


def h1_loadings(h1, outdir):
    L = np.asarray(h1["loadings"]["L"])
    items = h1["loadings"]["items"]
    factors = h1["loadings"]["factors"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(factors)), factors, rotation=45, ha="right")
    ax.set_yticks(range(len(items)), items, fontsize=8)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if abs(L[i, j]) > 1e-6:
                ax.text(j, i, f"{L[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5)
    ax.set_title("H1 primary bifactor: standardized loadings (DWLS/polychoric)")
    fig.colorbar(im, shrink=0.8)
    _save(fig, outdir, "h1_loadings.png")


def h1_envelopes(h1, outdir):
    stats = ["omega_h", "ecv", "cfi", "mean_abs_group_loading"]
    labels = ["omega_h", "ECV", "CFI", "mean |group loading|"]
    fig, axes = plt.subplots(2, len(stats), figsize=(14, 6), sharey=False)
    for row, (env, envname) in enumerate([("A", "Envelope A: value permutation"),
                                          ("B", "Envelope B: partition permutation")]):
        recs = h1["env_records"][env]
        for col, (s, lab) in enumerate(zip(stats, labels)):
            ax = axes[row, col]
            vals = [r[s] for r in recs]
            ax.hist(vals, bins=max(6, len(vals) // 3), color="#8899aa",
                    alpha=0.8)
            obs = h1["observed"][s]
            ax.axvline(obs, color="crimson", lw=2, label=f"observed {obs:.3f}")
            ax.set_title(f"{envname.split(':')[0]}: {lab}", fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle("H1 permutation envelopes (red = observed on the registered "
                 "partition)", fontsize=11)
    _save(fig, outdir, "h1_envelopes.png")


def h1_comparators(h1, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ["SWB one-factor", "Eid S-1", "MAGNA network"]
    dbics = [h1["swb"]["dbic"], h1["eid_s1"]["dbic_vs_primary"],
             h1["magna"]["dbic"]]
    axes[0].barh(names, dbics, color=["#4477aa"] * 3)
    axes[0].axvline(10, color="crimson", ls="--", lw=1,
                    label="delta-BIC = 10 criterion")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("BIC(comparator) - BIC(bifactor)  [> 0 favors bifactor]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("H1 leg (iv): information criteria")
    comm = np.asarray(h1["magna"]["communities"])
    items = h1["loadings"]["items"]
    axes[1].scatter(range(len(items)), comm, c=comm, cmap="tab10", s=40)
    axes[1].set_xticks(range(len(items)), items, rotation=90, fontsize=7)
    axes[1].set_ylabel("network community")
    axes[1].set_title(f"MAGNA communities ({h1['magna']['comm_method']}), "
                      f"ARI vs domains = {h1['magna']['ari']:.2f}")
    _save(fig, outdir, "h1_comparators.png")


def h2_alignment(h2a, outdir):
    if "error" in h2a:
        return
    countries = h2a["countries"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    facs = list(h2a["alignment"])
    r2l = [h2a["alignment"][f]["r2_loading"] for f in facs]
    r2n = [h2a["alignment"][f]["r2_intercept"] for f in facs]
    x = np.arange(len(facs))
    axes[0].bar(x - 0.2, r2l, 0.4, label="loadings")
    axes[0].bar(x + 0.2, r2n, 0.4, label="intercepts")
    axes[0].axhline(0.75, color="crimson", ls="--", lw=1, label="R2 = 0.75")
    axes[0].set_xticks(x, facs, rotation=45)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].set_title("H2a alignment R2 by factor")
    am = [h2a["aligned_means"][c] for c in countries]
    cm = [h2a["configural_means"][c] for c in countries]
    order = np.argsort(am)
    axes[1].plot(np.array(am)[order], range(len(countries)), "o-", ms=4,
                 label="aligned alpha_g")
    axes[1].plot(np.array(cm)[order], range(len(countries)), "s--", ms=4,
                 alpha=0.7, label="configural (Bartlett)")
    axes[1].set_yticks(range(len(countries)),
                       [countries[i] for i in order], fontsize=7)
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"country f means, rank r = {h2a['rank_corr']:.3f}")
    alf = h2a["alignment"]["f"]
    flags = np.asarray(alf["flags"]).any(-1).astype(float)
    im = axes[2].imshow(flags, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(alf["items"])), alf["items"], rotation=90,
                       fontsize=7)
    axes[2].set_yticks(range(len(countries)), countries, fontsize=6)
    axes[2].set_title(f"non-invariance flags "
                      f"({100 * h2a['flag_frac']:.1f}% of parameters)")
    _save(fig, outdir, "h2_alignment.png")


def h2b_scatter(h2b, outdir):
    df = h2b.get("data")
    if df is None or len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for focal, col, lab in [(0, "#4477aa", "control (physical/financial)"),
                            (1, "#cc6677", "focal (meaning/character/social)")]:
        d = df[df.focal == focal]
        g = d.groupby("country").agg(dist=("dist", "first"),
                                     noninv=("noninv", "mean"))
        ax.scatter(g.dist, g.noninv, color=col, label=lab, s=35)
        if len(g) > 2:
            b = np.polyfit(g.dist, g.noninv, 1)
            xs = np.linspace(g.dist.min(), g.dist.max(), 10)
            ax.plot(xs, np.polyval(b, xs), color=col, lw=1.5)
    ax.set_xlabel(f"cultural distance ({h2b['distance_source']})")
    ax.set_ylabel("mean non-invariance magnitude")
    ax.legend(fontsize=8)
    ax.set_title(f"H2b: interaction = {h2b['interaction_coef']:.3f} "
                 f"(p = {h2b['interaction_p']:.3f}) -> {h2b['verdict']}")
    _save(fig, outdir, "h2b_distance.png")


def h3_forest(h3, outdir):
    res = h3["results"]
    names = list(res)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, k in enumerate(names):
        r = res[k]
        se = r["se"] if np.isfinite(r.get("se", np.nan)) else 0
        ax.errorbar(r["beta"], i, xerr=1.96 * se, fmt="o", color="#4477aa",
                    capsize=4, ms=7,
                    label="f_(-d) primary (W1->W2)" if i == 0 else None)
        ax.plot(r["beta_full_f"], i - 0.18, "s", color="#999999", ms=5,
                label="full f (secondary)" if i == 0 else None)
        if np.isfinite(r.get("beta_w3", np.nan)):
            ax.plot(r["beta_w3"], i + 0.18, "^", color="#44aa77", ms=5,
                    label="W1->W3 extension" if i == 0 else None)
        star = " *FDR" if h3["fdr"].get(k) else ""
        ax.text(ax.get_xlim()[1], i, f" cos={r['cosine']:.2f}{star}",
                fontsize=7, va="center")
    ax.axvline(0.05, color="crimson", ls="--", lw=1, label="beta = 0.05")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("standardized beta on outcome change (baseline controlled)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"H3 predictive validity beyond p -> {h3['verdict']}")
    _save(fig, outdir, "h3_forest.png")


def h4_matrices(h4, outdir):
    mats = [("B_obs (symmetrized)", h4["B_sym"]),
            ("B^L latent anchor", h4["anchors"]["BL"]),
            ("B^M mutualism anchor", h4["anchors"]["BM"])]
    vmax = max(np.abs(m).max() for _, m in mats)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (t, M) in zip(axes, mats):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"H4 propagator vs anchors: w_hat_raw = "
                 f"{h4['stage1']['w_raw']:.3f}, off-diag R2 = "
                 f"{h4['stage1']['r2_offdiag']:.3f}, s_skew = {h4['s_skew']:.3f}")
    fig.savefig(os.path.join(outdir, "h4_matrices.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def h4_map(h4, outdir):
    m = h4["map"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mu in m["mus"]:
        axes[0].plot(m["wgrid"], m["table"][mu], "o-", ms=3,
                     label=f"mu_ratio = {mu}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].axhline(h4["stage1"]["w_raw"], color="crimson", lw=1.2,
                    label=f"observed w_raw = {h4['stage1']['w_raw']:.3f}")
    axes[0].set_xlabel("w_L (truth)"); axes[0].set_ylabel("E[w_hat]")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Stage 2 correction map (indirect inference)")
    axes[1].hist(h4["boot"]["w_corr"], bins=20, color="#8899aa", alpha=0.85)
    axes[1].axvline(h4["w_corrected"], color="crimson", lw=2,
                    label=f"w_corrected = {h4['w_corrected']:.3f}")
    axes[1].axvspan(h4["ci"][0], h4["ci"][1], color="crimson", alpha=0.12,
                    label=f"CI x{1.5} [{h4['ci'][0]:.2f}, {h4['ci'][1]:.2f}]")
    axes[1].set_xlabel("bootstrap w_corrected")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"H4b tier: {h4['tier']}")
    _save(fig, outdir, "h4_map.png")


def h4a_dists(h4, outdir):
    h4a = h4["h4a"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(h4a["dist_L"], bins=12, alpha=0.6, color="#4477aa",
                 density=True, label="d | Latent truth")
    axes[0].hist(h4a["dist_M"], bins=12, alpha=0.6, color="#cc6677",
                 density=True, label="d | Mutualism truth")
    axes[0].axvline(h4a["d_obs"], color="k", lw=2,
                    label=f"d_obs = {h4a['d_obs']:.3f}")
    for t, lab in zip(h4a["thresholds"], ["t_M", "t_L"]):
        axes[0].axvline(t, ls="--", lw=1, color="gray")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Delta_fit = fit_L - fit_M")
    axes[0].set_title(f"H4a cross-fitting -> {h4a['verdict']} "
                      f"(kappa = {h4a['kappa']:.2f})")
    conf = h4["confusion"]
    bottom = np.zeros(len(conf))
    for k, col in [("Latent", "#4477aa"), ("Indistinguishable", "#dddddd"),
                   ("Mutualism", "#cc6677")]:
        axes[1].bar(conf["w_true"], conf[k], width=0.16, bottom=bottom,
                    color=col, label=k)
        bottom += conf[k].to_numpy()
    axes[1].set_xlabel("w_L (truth)"); axes[1].set_ylabel("verdict share")
    axes[1].legend(fontsize=8)
    axes[1].set_title("H4a confusion profile over the calibration grid")
    _save(fig, outdir, "h4a_crossfit.png")


def make_all(results: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    jobs = []
    if "h1" in results:
        jobs += [(h1_loadings, results["h1"]), (h1_envelopes, results["h1"]),
                 (h1_comparators, results["h1"])]
    if "h2a" in results and "error" not in results["h2a"]:
        jobs.append((h2_alignment, results["h2a"]))
    if "h2b" in results:
        jobs.append((h2b_scatter, results["h2b"]))
    if "h3" in results:
        jobs.append((h3_forest, results["h3"]))
    if "h4" in results:
        jobs += [(h4_matrices, results["h4"]), (h4_map, results["h4"]),
                 (h4a_dists, results["h4"])]
    made, failed = [], []
    for fn, arg in jobs:
        try:
            fn(arg, outdir)
            made.append(fn.__name__)
        except Exception as e:      # noqa: BLE001 -- plots must not kill runs
            failed.append(f"{fn.__name__}: {e}")
    return dict(made=made, failed=failed)

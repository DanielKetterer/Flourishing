"""
report.py -- Markdown summary of a pipeline run: verdicts, key statistics,
and the flags a Stage 2 reviewer (or Daniel) needs in one place.
"""

from __future__ import annotations
import numpy as np
from . import config as C


def _f(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_markdown(results: dict, smoke: bool, truth: dict | None = None) -> str:
    L = []
    L.append("# GFS f-construct pipeline run summary\n")
    mode = ("SMOKE (synthetic, reduced settings; verdicts are machinery "
            "checks, not evidence)") if smoke else "production"
    L.append(f"Mode: {mode}\n")

    h1 = results.get("h1")
    if h1:
        p = h1["pooled"]
        L.append("## H1 structural replication\n")
        L.append(f"Verdict: **{h1['verdict']}**  (legs: "
                 + ", ".join(f"{k}={'pass' if v else 'FAIL'}"
                             for k, v in h1["legs"].items()) + ")\n")
        L.append(f"omega_h = {_f(p['omega_h']['point'])}, "
                 f"ECV = {_f(p['ecv']['point'])}, "
                 f"CFI = {_f(p['cfi']['point'])}, "
                 f"RMSEA = {_f(p['rmsea']['point'])}, "
                 f"SRMR = {_f(p['srmr']['point'])}, "
                 f"f-loading block SD = {_f(p['f_loading_block_sd']['point'])}\n")
        L.append(f"Hedonic stability: cos(AltA) = {_f(h1['alternatives']['cos_altA'])}, "
                 f"cos(AltB) = {_f(h1['alternatives']['cos_altB'])}\n")
        L.append(f"Comparators: dBIC(SWB) = {_f(h1['swb']['dbic'], 1)} "
                 f"(cos = {_f(h1['swb']['cosine'])}, Vuong z = "
                 f"{_f(h1['swb']['vuong_z'], 2)}); dBIC(S-1) = "
                 f"{_f(h1['eid_s1']['dbic_vs_primary'], 1)}; dBIC(MAGNA) = "
                 f"{_f(h1['magna']['dbic'], 1)}, ARI = {_f(h1['magna']['ari'])}\n")
        gb = h1["envelopes"]["B"]
        L.append("Envelope B placements (percentile of observed in the "
                 "permuted-partition distribution): "
                 + ", ".join(f"{s} = {_f(gb[s]['percentile'], 2)}"
                             for s in ("omega_h", "ecv", "cfi",
                                       "mean_abs_group_loading")) + "\n")

    h2a = results.get("h2a")
    if h2a and "error" not in h2a:
        L.append("## H2a measurement properties (descriptive)\n")
        L.append(f"Tier: **{h2a['tier']}**  "
                 f"(R2 = {_f(h2a['r2_avg'])} vs > 0.75; flagged = "
                 f"{_f(100 * h2a['flag_frac'], 1)}% vs <= 25%; rank corr = "
                 f"{_f(h2a['rank_corr'])} vs >= 0.98)\n")
        mg = h2a["mgcfa"]
        L.append(f"MGCFA sensitivity (Chen): metric dCFI = "
                 f"{_f(mg['metric']['dcfi'])} "
                 f"({'pass' if mg['metric']['pass_chen'] else 'fail'}); "
                 f"scalar dCFI = {_f(mg['scalar']['dcfi'])} "
                 f"({'pass' if mg['scalar']['pass_chen'] else 'fail'})\n")

    h2mc = results.get("h2a_mc")
    if h2mc is not None and len(h2mc):
        L.append("H2a Monte Carlo operating characteristics (flag-rule "
                 "evidence base): mean alpha recovery = "
                 f"{_f(np.nanmean(h2mc['alpha_recovery']))}, mean flag "
                 f"sensitivity = {_f(np.nanmean(h2mc['sensitivity']))}, mean "
                 f"FPR = {_f(np.nanmean(h2mc['fpr']))}\n")

    h2b = results.get("h2b")
    if h2b:
        L.append("## H2b cultural-distance falsification\n")
        L.append(f"Verdict: **{h2b['verdict']}**; interaction = "
                 f"{_f(h2b['interaction_coef'])} (p = "
                 f"{_f(h2b['interaction_p'])}); distance source: "
                 f"{h2b['distance_source']}; translation covariate: "
                 f"{h2b['translation_source']}\n")
        if h2b.get("model_kind"):
            L.append(f"Model fit: {h2b['model_kind']}\n")

    h3 = results.get("h3")
    if h3:
        L.append("## H3 predictive validity beyond p\n")
        L.append(f"Verdict: **{h3['verdict']}** (confirming outcomes: "
                 f"{h3['confirming'] or 'none'})\n")
        L.append("| outcome | beta f_(-d) | se | FDR pass | beta p | full-f "
                 "(secondary) | W1->W3 | cosine | omega match |\n")
        L.append("|---|---|---|---|---|---|---|---|---|\n")
        for k, r in h3["results"].items():
            L.append(f"| {k} | {_f(r['beta'])} | {_f(r['se'])} | "
                     f"{h3['fdr'].get(k)} | {_f(r['beta_p_composite'])} | "
                     f"{_f(r['beta_full_f'])} | {_f(r['beta_w3'])} | "
                     f"{_f(r['cosine'])} | {r['match']['matched']} |\n")
        if h3["riclpm_opposite"]:
            L.append(f"RI-CLPM opposite-sign within-person effects "
                     f"(disconfirmation leg): {h3['riclpm_opposite']}\n")

    h4 = results.get("h4")
    if h4:
        L.append("## H4 latent vs mutualism channel\n")
        L.append(f"w_raw = {_f(h4['stage1']['w_raw'])}, w_corrected = "
                 f"**{_f(h4['w_corrected'])}** "
                 f"(CI x1.5: [{_f(h4['ci'][0])}, {_f(h4['ci'][1])}]), "
                 f"mu_hat = {_f(h4['mu_hat'])}, tier: **{h4['tier']}**\n")
        L.append(f"Three-anchor simplex: {h4['stage1'].get('three_anchor')}\n")
        L.append(f"Stage 1 off-diagonal R2 = {_f(h4['stage1']['r2_offdiag'])}; "
                 f"sparse-J gap = {_f(h4['sparse']['gap'])}; s_skew = "
                 f"{_f(h4['s_skew'])}; Tikhonov = {h4['tikhonov']['tikhonov']}; "
                 f"pre-flight all-pass = {h4['preflight']['all_pass']}\n")
        L.append(f"H4a: **{h4['h4a']['verdict']}** (d_obs = "
                 f"{_f(h4['h4a']['d_obs'])}, kappa = {_f(h4['h4a']['kappa'])}, "
                 f"phi_hat = {_f(h4['h4a']['phi_hat'])}, tau_hat = "
                 f"{_f(h4['h4a']['tau_hat'], 2)})\n")
        L.append(f"Directional commitment: falsified = "
                 f"{h4['directional']['coupling_interpretation_falsified']} "
                 f"({h4['directional']['basis']})\n")
        cx = h4["cross_checks"]
        L.append(f"Cross-checks: separability holds = "
                 f"{cx['separability']['holds']} (p = "
                 f"{_f(cx['separability']['p'])}); elastic-net w = "
                 f"{_f(cx['elasticnet']['w'])}; naive lag-1 trace/p = "
                 f"{_f(cx['naive']['trace_frac'])}\n")

    if truth is not None:
        L.append("## Synthetic ground truth (smoke validation)\n")
        L.append(f"True w_L = {truth['w_L']}, mu_ratio = {truth['mu_ratio']}, "
                 f"phi = {truth['phi']}, tau = {truth['tau']}. Compare with "
                 f"w_corrected and mu_hat above; the DGP's cross-sectional "
                 f"structure is w_L-invariant by construction, so recovery of "
                 f"w_L here is the direct test of the H4 identification "
                 f"strategy.\n")

    L.append("## Flags (read before Stage 2 lock)\n")
    flags = [
        "omega_h threshold: Hypotheses text says 0.75, Inference Criteria "
        "say 0.80. Verdicts here use 0.80 (Inference Criteria). Reconcile "
        "in v8.",
        "Envelope B gap DIRECTION: partition permutation forces domain "
        "covariance into f, which raises omega_h/ECV and lowers fit and "
        "group-loading magnitude. 'Exceed the envelope by 0.05' as written "
        "likely inverts for omega_h/ECV. Both directions and percentiles are "
        "computed; the verdict sign lives in config.H1['envelope_gap_sign'] "
        "and must be reconciled against the authoritative v8 definition.",
        "CFI/RMSEA under DWLS are polychoric-ML approximations of the WLSMV "
        "robust indices (SRMR, loadings, omega, ECV, gaps, cosines are "
        "exact). Run the lavaan WLSMV parity check at Stage 2.",
        "Placeholder variable names pending codebook lock: "
        + ", ".join(v for v in
                    [it.var for it in C.F_ITEMS + C.P_ITEMS if it.placeholder]
                    + [o["var"] for o in C.H3_OUTCOMES.values()
                       if o.get("placeholder")]) + ". "
        "COS distributes a perturbed Sweden sample file for exactly this "
        "purpose; lock names and layout against it before deposit.",
        "Alignment is implemented from scratch in Python (loss, weights, and "
        "FIXED identification per Asparouhov-Muthen 2014); the included MC "
        "harness is the prereg's own validation route, but an Mplus/sirt "
        "cross-check is recommended at Stage 2.",
        "H4 latent-anchor direction currently uses the H1 pooled f loadings; "
        "the registered anchor is the MIDUS Lambda mapped to GFS items, to "
        "be fixed in the Stage 2 addendum file.",
        "H2a tier ladder implemented as Tier = 4 - (#criteria met), Tier 1 = "
        "all three; confirm the v8 mapping. H4b Tier 2/3 split and the "
        "pre-flight bands are defaults pending v8 transcription.",
    ]
    for fl in flags:
        L.append(f"- {fl}\n")
    return "\n".join(L)

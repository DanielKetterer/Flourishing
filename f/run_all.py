"""
run_all.py -- Orchestrates the full pre-registered pipeline.

    python -m gfs_f.run_all --synthetic --smoke --out runs/smoke
    python -m gfs_f.run_all --data-dir /path/to/gfs --out runs/production

Smoke mode: synthetic GFS-format data at reduced scale and reduced Monte
Carlo settings (config.SMOKE). Purpose: verify every registered procedure
executes end-to-end and recovers a known DGP before any data access.
Production mode: full pre-registered constants from config.
"""

from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import config as C
from .data_io import (load_gfs, orient_items, normalize_weights,
                      validate_varmap, wave_frame, panel_frame,
                      multiply_impute, n_cats_map, write_lookup_templates)
from .synthetic import simulate_panel
from .h1_structure import run_h1
from .h2_invariance import run_h2a, run_h2b, run_h2a_mc
from .h3_predictive import run_h3
from .h4_dynamics import run_h4
from . import plots, report

DROP_KEYS = {"R_list", "V_list", "fits", "Xc", "per_country", "mask",
             "weights", "country", "data", "X", "train_net", "test",
             "env_records", "dist_L", "dist_M", "boot", "map", "anchors",
             "flags", "lam_aligned", "nu_aligned", "model_summary"}


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()
                if k not in DROP_KEYS}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 500 else f"<array {o.shape}>"
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records") if len(o) <= 100 else f"<df {o.shape}>"
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)[:300]


def main(argv=None):
    ap = argparse.ArgumentParser(description="GFS f-construct pipeline")
    ap.add_argument("--synthetic", action="store_true",
                    help="simulate GFS-format data instead of loading files")
    ap.add_argument("--data-dir", default=None, help="released GFS data dir")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced settings for the end-to-end machinery check")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--w-true", type=float, default=0.6,
                    help="synthetic DGP mixture weight w_L")
    ap.add_argument("--mu-true", type=float, default=0.5,
                    help="synthetic DGP trait/dynamic variance ratio")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["h1", "h2", "h2mc", "h3", "h4"],
                    help="blocks to skip (debugging aid)")
    args = ap.parse_args(argv)
    t0 = time.time()
    seed = C.COMMON["seed"]
    smoke = args.smoke
    os.makedirs(args.out, exist_ok=True)
    figdir = os.path.join(args.out, "figures")
    lookups = os.path.join(args.out, C.LOOKUP_DIR)
    write_lookup_templates(lookups)

    truth = None
    if args.synthetic or not args.data_dir:
        npc = C.SMOKE["n_per_country"] if smoke else 1000
        print(f"[run_all] simulating GFS-format panel: {npc}/country x "
              f"{len(C.GFS_COUNTRIES)} countries, w_L = {args.w_true}, "
              f"mu_ratio = {args.mu_true}")
        long_df, truth = simulate_panel(n_per_country=npc, waves=3,
                                        w_L=args.w_true,
                                        mu_ratio=args.mu_true, seed=seed)
    else:
        long_df = load_gfs(args.data_dir)
    long_df = normalize_weights(orient_items(long_df))
    varmap_report = validate_varmap(long_df)

    m = C.SMOKE["n_imputations"] if smoke else C.COMMON["n_imputations"]
    ncmap = n_cats_map()
    results: dict = dict(varmap=varmap_report,
                         mode="smoke" if smoke else "production")

    # ---- Wave 1 imputations (H1, H2) ----
    w1 = wave_frame(long_df, 1)
    fcols = [it.var for it in C.F_ITEMS]
    print(f"[run_all] multiple imputation, wave 1 (m = {m}) ...")
    imputed_w1 = multiply_impute(w1, fcols, m, ncmap,
                                 per_country=not smoke, seed=seed)

    if "h1" not in args.skip:
        print("[run_all] H1 structural block ...")
        results["h1"] = run_h1(imputed_w1, seed, smoke=smoke)
        print(f"          H1 verdict: {results['h1']['verdict']} "
              f"(omega_h = {results['h1']['pooled']['omega_h']['point']:.3f})")

    if "h2" not in args.skip and "h1" in results:
        print("[run_all] H2a alignment block ...")
        results["h2a"] = run_h2a(imputed_w1[0], results["h1"]["fits"][0],
                                 seed, smoke=smoke,
                                 min_n=150 if smoke else None,
                                 lookups_dir=lookups)
        if "error" not in results["h2a"]:
            print(f"          H2a tier: {results['h2a']['tier']}")
            results["h2b"] = run_h2b(results["h2a"], lookups, seed, smoke)
            print(f"          H2b verdict: {results['h2b']['verdict']}")

    if "h2mc" not in args.skip:
        print("[run_all] H2a Monte Carlo operating characteristics ...")
        results["h2a_mc"] = run_h2a_mc(
            seed, smoke=smoke,
            n_per_country=250 if smoke else 500,
            n_countries=8 if smoke else len(C.GFS_COUNTRIES))

    # ---- Panel imputations (H3, H4) ----
    panel = panel_frame(long_df)
    stems = [it.var for it in C.F_ITEMS + C.P_ITEMS] + \
            [o["var"] for o in C.H3_OUTCOMES.values()]
    pcols = [f"{v}_w{t}" for v in stems for t in (1, 2, 3)
             if f"{v}_w{t}" in panel.columns]
    ncmap_panel = {f"{k}_w{t}": v for k, v in ncmap.items() for t in (1, 2, 3)}
    print(f"[run_all] multiple imputation, 3-wave panel "
          f"(n = {len(panel)}, m = {m}) ...")
    imputed_panels = multiply_impute(panel, pcols, m, ncmap_panel,
                                     per_country=not smoke, seed=seed + 50)

    if "h3" not in args.skip:
        print("[run_all] H3 predictive block ...")
        results["h3"] = run_h3(imputed_panels, seed, smoke=smoke)
        print(f"          H3 verdict: {results['h3']['verdict']}")

    if "h4" not in args.skip and "h1" in results:
        print("[run_all] H4 dynamics block ...")
        results["h4"] = run_h4(imputed_panels[0], results["h1"]["fits"][0],
                               seed, smoke=smoke)
        print(f"          H4 w_corrected = {results['h4']['w_corrected']:.3f}"
              f" (tier {results['h4']['tier']}), H4a: "
              f"{results['h4']['h4a']['verdict']}")

    # ---- outputs ----
    print("[run_all] figures ...")
    plot_report = plots.make_all(results, figdir)
    results["plots"] = plot_report
    md = report.build_markdown(results, smoke,
                               truth=dict(w_L=truth["w_L"],
                                          mu_ratio=truth["mu_ratio"],
                                          phi=truth["phi"], tau=truth["tau"])
                               if truth else None)
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(sanitize(results), fh, indent=1)
    print(f"[run_all] done in {time.time() - t0:.0f}s -> {args.out}")
    return results


if __name__ == "__main__":
    main()

"""
data_io.py -- Loading, orientation, weighting, and multiple imputation.

Supports both plausible release layouts (Stage 2 lock pins the real one):
  * long : one file (or per-wave files) with a WAVE column
  * wide : one row per ID with wave-suffixed columns (config.WIDE_SUFFIX)

Missing data follows the prereg: multiple imputation (m = 20 in production)
with all H1-H4 indicators and demographic auxiliaries in the imputation
model, sampling weights included as a predictor (GFS methodology convention),
imputation run separately per country, pooled by Rubin's rules. WLSMV/DWLS
has no FIML; without this pipeline the polychorics silently become
pairwise-present, which the prereg explicitly rejects as the missing-data
model.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from . import config as C


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_gfs(data_dir: str, waves=(1, 2, 3), file_pattern="gfs_wave{w}.csv"
             ) -> pd.DataFrame:
    """Read released GFS files into the canonical long format.

    Accepts either a single long file 'gfs_all_waves.csv' with a WAVE column
    or per-wave files matching file_pattern. Wide files are melted using
    config.WIDE_SUFFIX. Everything downstream consumes the long format.
    """
    single = os.path.join(data_dir, "gfs_all_waves.csv")
    if os.path.exists(single):
        df = pd.read_csv(single)
        if C.ADMIN["wave"] in df.columns:
            return df
        return _wide_to_long(df, waves)
    frames = []
    for w in waves:
        path = os.path.join(data_dir, file_pattern.format(w=w))
        if not os.path.exists(path):
            print(f"[data_io] wave {w} file not found ({path}); skipping")
            continue
        d = pd.read_csv(path)
        if C.ADMIN["wave"] not in d.columns:
            d[C.ADMIN["wave"]] = w
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No GFS files found under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def _wide_to_long(df: pd.DataFrame, waves) -> pd.DataFrame:
    out = []
    stems = [it.var for it in C.F_ITEMS + C.P_ITEMS] + \
            [o["var"] for o in C.H3_OUTCOMES.values()]
    for w in waves:
        suf = C.WIDE_SUFFIX.format(wave=w)
        cols = {s + suf: s for s in stems if s + suf in df.columns}
        keep = [C.ADMIN["id"], C.ADMIN["country"], C.ADMIN["strata"],
                C.ADMIN["psu"]] + list(C.H3_CONTROLS.values())
        keep = [k for k in keep if k in df.columns]
        d = df[keep + list(cols)].rename(columns=cols).copy()
        wcol = C.ADMIN["weight_by_wave"].get(w, C.ADMIN["weight"])
        d[C.ADMIN["weight"]] = df[wcol] if wcol in df.columns else 1.0
        d[C.ADMIN["wave"]] = w
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Orientation, weights, validation
# ---------------------------------------------------------------------------

def orient_items(df: pd.DataFrame) -> pd.DataFrame:
    """Re-code reversed items so higher always = more flourishing / more
    symptomatic-free. Idempotence guard via attrs flag."""
    if df.attrs.get("_oriented"):
        return df
    df = df.copy()
    for it in C.F_ITEMS:
        if it.reverse and it.var in df.columns:
            df[it.var] = (it.n_cat - 1) - df[it.var]
    df.attrs["_oriented"] = True
    return df


def normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Prereg Sampling Weights: Gallup weights applied at the country level.
    Within-country normalization to mean 1 so each country contributes in
    proportion to its analytic n in pooled analyses."""
    df = df.copy()
    wcol, ccol = C.ADMIN["weight"], C.ADMIN["country"]
    if wcol not in df.columns:
        df[wcol] = 1.0
    df[wcol] = df[wcol].fillna(1.0)
    df[wcol] = df.groupby(ccol)[wcol].transform(lambda s: s / s.mean())
    return df


def validate_varmap(df: pd.DataFrame) -> dict:
    """Stage 2 pre-flight: presence of every mapped variable, placeholder
    inventory, and the prereg's p/f item-disjointness codebook cross-check
    (p items must be disjoint from f's mental-health indicators)."""
    missing = [it.var for it in C.F_ITEMS + C.P_ITEMS
               if it.var not in df.columns]
    missing += [o["var"] for o in C.H3_OUTCOMES.values()
                if o["var"] not in df.columns]
    placeholders = [it.var for it in C.F_ITEMS + C.P_ITEMS if it.placeholder]
    f_mental = {it.var for it in C.F_ITEMS if it.domain == "mental"}
    overlap = f_mental & {it.var for it in C.P_ITEMS}
    report = dict(missing=missing, placeholders=placeholders,
                  p_f_disjoint=len(overlap) == 0, overlap=sorted(overlap))
    if missing:
        print(f"[validate_varmap] MISSING variables (fix VARMAP at Stage 2 "
              f"lock): {missing}")
    if placeholders:
        print(f"[validate_varmap] placeholder names pending codebook "
              f"confirmation: {placeholders}")
    if overlap:
        print(f"[validate_varmap] VIOLATION: p items overlap f mental items: "
              f"{overlap}")
    return report


# ---------------------------------------------------------------------------
# Analysis frames
# ---------------------------------------------------------------------------

def wave_frame(long_df: pd.DataFrame, wave: int) -> pd.DataFrame:
    return long_df[long_df[C.ADMIN["wave"]] == wave].reset_index(drop=True)


def panel_frame(long_df: pd.DataFrame, waves=(1, 2, 3)) -> pd.DataFrame:
    """Wide panel of respondents observed at all requested waves; item and
    outcome columns suffixed _w{t}; W1 admin/demographics carried."""
    idc, wc = C.ADMIN["id"], C.ADMIN["wave"]
    stems = [it.var for it in C.F_ITEMS + C.P_ITEMS] + \
            [o["var"] for o in C.H3_OUTCOMES.values()]
    stems = [s for s in stems if s in long_df.columns]
    base = wave_frame(long_df, waves[0])
    admin = [idc, C.ADMIN["country"], C.ADMIN["weight"], C.ADMIN["strata"],
             C.ADMIN["psu"]] + [v for v in C.H3_CONTROLS.values()
                                if v in base.columns]
    out = base[admin + stems].rename(columns={s: f"{s}_w{waves[0]}"
                                              for s in stems})
    for w in waves[1:]:
        d = wave_frame(long_df, w)[[idc] + stems]
        d = d.rename(columns={s: f"{s}_w{w}" for s in stems})
        out = out.merge(d, on=idc, how="inner")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multiple imputation (m = 20 production) + Rubin pooling
# ---------------------------------------------------------------------------

def multiply_impute(df: pd.DataFrame, cols: list[str], m: int,
                    n_cats: dict[str, int], per_country: bool = True,
                    seed: int = 0) -> list[pd.DataFrame]:
    """MICE-style MI via sklearn IterativeImputer with posterior sampling.

    Weights enter the imputation model as a predictor (GFS methodology);
    imputation is per-country by default (GFS convention; per_country=False
    is the pooled fast path used only in smoke runs). Ordinal columns are
    rounded and clipped back to their category range after each draw.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    cols = [c for c in cols if c in df.columns]
    aux = [C.ADMIN["weight"]] + [v for v in C.H3_CONTROLS.values()
                                 if v in df.columns]
    use = cols + [a for a in aux if a not in cols]
    outs = []
    for k in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=6,
                               random_state=seed + k)
        d = df.copy()
        if per_country:
            for _, idx in df.groupby(C.ADMIN["country"]).groups.items():
                block = df.loc[idx, use].astype(float)
                if block[cols].isna().values.any():
                    d.loc[idx, use] = imp.fit_transform(block)
        else:
            d[use] = imp.fit_transform(df[use].astype(float))
        for c in cols:
            k_c = n_cats.get(c, 11)
            d[c] = np.clip(np.round(d[c]), 0, k_c - 1)
        outs.append(d)
    return outs


def rubin_pool(points: list[float], variances: list[float] | None = None
               ) -> dict:
    """Rubin's rules. If within-imputation variances are unavailable
    (envelope-style statistics), the between-imputation spread is reported
    as the MI uncertainty."""
    q = np.asarray(points, float)
    m = len(q)
    qbar = float(q.mean())
    B = float(q.var(ddof=1)) if m > 1 else 0.0
    if variances is None:
        return dict(point=qbar, between_var=B, m=m)
    W = float(np.mean(variances))
    T = W + (1 + 1 / m) * B
    df = (m - 1) * (1 + W / ((1 + 1 / m) * B)) ** 2 if B > 0 else np.inf
    return dict(point=qbar, var=T, se=float(np.sqrt(T)), df=df,
                within=W, between=B, m=m)


def n_cats_map() -> dict[str, int]:
    d = {it.var: it.n_cat for it in C.F_ITEMS + C.P_ITEMS}
    for o in C.H3_OUTCOMES.values():
        d[o["var"]] = 11
    return d


# ---------------------------------------------------------------------------
# Lookup templates (H2 stratifications, H2b covariates)
# ---------------------------------------------------------------------------

def write_lookup_templates(out_dir: str):
    """Emit CSV templates for the H2/H2b country-level lookups. Fill CFST_US
    from Muthukrishna et al. (2020) supplement, IW_TRAD/IW_SURV from the
    published Inglehart-Welzel country scores, INCOME_TIER from the World
    Bank classification for the survey year, RELIGION from the codebook's
    majority-tradition variable, and the item x country translation
    covariate from the Cowden et al. (2024) cognitive-interview
    classification."""
    os.makedirs(out_dir, exist_ok=True)
    base = pd.DataFrame({C.ADMIN["country"]: C.GFS_COUNTRIES})
    for name, cols in [
        ("cultural_distance.csv", dict(CFST_US=np.nan)),
        ("iw_scores.csv", dict(IW_TRAD=np.nan, IW_SURV=np.nan)),
        ("income_tier.csv", dict(INCOME_TIER="")),
        ("religion.csv", dict(RELIGION="")),
    ]:
        d = base.copy()
        for c, v in cols.items():
            d[c] = v
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            d.to_csv(p, index=False)
    tr = os.path.join(out_dir, "translation_equivalence.csv")
    if not os.path.exists(tr):
        rows = [(c, it.var, np.nan) for c in C.GFS_COUNTRIES
                for it in C.F_ITEMS]
        pd.DataFrame(rows, columns=[C.ADMIN["country"], "ITEM", "T_EQUIV"]
                     ).to_csv(tr, index=False)


def load_lookup(out_dir: str, name: str) -> pd.DataFrame | None:
    p = os.path.join(out_dir, name)
    if os.path.exists(p):
        d = pd.read_csv(p)
        if d.select_dtypes(include=[np.number]).isna().all().all():
            return None
        return d
    return None

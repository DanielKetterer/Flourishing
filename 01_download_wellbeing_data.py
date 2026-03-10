"""
01_download_wellbeing_data.py
─────────────────────────────────────────────────────────────────────────────
Downloads and harmonizes cross-national wellbeing data for f-factor analysis.

Data sources
────────────
1. World Happiness Report 2024 (WHR) — GitHub CSV mirror (Escavine/World-Happiness)
   Items: Positive affect, Freedom to make life choices, Social support,
          Ladder score (Cantril 0-10), Log GDP per capita,
          Healthy life expectancy, Generosity, Perceptions of corruption

2. OECD Better Life Index (BLI) — via datasets/oecd-bli GitHub mirror
   URL  : https://raw.githubusercontent.com/datasets/oecd-bli/master/data/oecd-bli.csv
   Items: Community, Health, Life Satisfaction, Work-Life Balance,
          Jobs, Income, Education, Environment, Safety, Civic Engagement, Housing

3. World Bank Open Data API
   Items: Life expectancy, Unemployment rate, School enrollment,
          Health expenditure % GDP

PERMA+H mapping (WHR primary columns)
──────────────────────────────────────
  P  Positive emotion   → Positive affect        (not in this mirror; skipped gracefully)
  E  Engagement         → Freedom to make life choices  (autonomy proxy)
  R  Relationships      → Social support
  M  Meaning            → Ladder score  (Cantril life evaluation)
  A  Accomplishment     → Log GDP per capita  (material achievement proxy)
  H  Health             → Healthy life expectancy

Note: WHR data are country-level Gallup averages (~3-year rolling windows).
They are population-mean estimates, not individual-level survey responses.
The positive manifold we expect to observe is therefore an ecological
correlation; individual-level structure may differ.

Outputs
───────
  wellbeing_data/WHR2024_raw.csv
  wellbeing_data/oecd_bli_raw.csv
  wellbeing_data/worldbank_raw.csv
  wellbeing_data/wellbeing_merged.csv   ← main output for analysis script
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("wellbeing_data")
DATA_DIR.mkdir(exist_ok=True)

WHR_URL = (
    "https://raw.githubusercontent.com/Escavine/World-Happiness/main/"
    "World-happiness-report-2024.csv"
)
OECD_BLI_URL = (
    "https://raw.githubusercontent.com/datasets/oecd-bli/master/data/oecd-bli.csv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# WHR raw column (lowercase) → PERMA+H label
# Covers naming variations across WHR editions and mirrors
WHR_COLUMN_MAP = {
    # PERMA+H core
    "positive affect":                    "P_positive_emotion",
    "freedom to make life choices":       "E_engagement_autonomy",
    "social support":                     "R_relationships",
    "ladder score":                       "M_meaning_life_sat",
    "life ladder":                        "M_meaning_life_sat",       # panel edition alt
    "log gdp per capita":                 "A_accomplishment_gdp",
    "healthy life expectancy at birth":   "H_health",
    "healthy life expectancy":            "H_health",                 # ← this mirror's name
    # Reference / robustness columns
    "negative affect":                    "ref_negative_affect",
    "generosity":                         "ref_generosity",
    "perceptions of corruption":          "ref_corruption",
}

# Possible country column names (case-sensitive candidates tried in order)
COUNTRY_COL_CANDIDATES = ["Country name", "country name", "Country", "country"]

# Possible year column names (only present in panel-format files)
YEAR_COL_CANDIDATES = ["year", "Year"]

# OECD BLI indicator name → wellbeing domain label (supplement)
OECD_INDICATOR_MAP = {
    "Life satisfaction":      "oecd_life_satisfaction",
    "Community":              "oecd_community",
    "Work-life balance":      "oecd_work_life_balance",
    "Health status":          "oecd_health_status",
    "Jobs and earnings":      "oecd_jobs",
    "Education and skills":   "oecd_education",
    "Income and wealth":      "oecd_income",
    "Safety":                 "oecd_safety",
}

# World Bank indicators { WB code : output label }
WB_INDICATORS = {
    "SP.DYN.LE00.IN":     "wb_life_expectancy",
    "SL.UEM.TOTL.ZS":     "wb_unemployment_pct",
    "SE.PRM.NENR":        "wb_primary_enrollment",
    "SH.XPD.CHEX.GD.ZS": "wb_health_expenditure_pct_gdp",
}

PERMA_COLS = [
    "P_positive_emotion",
    "E_engagement_autonomy",
    "R_relationships",
    "M_meaning_life_sat",
    "A_accomplishment_gdp",
    "H_health",
]

# ─── WHR ──────────────────────────────────────────────────────────────────────

def download_whr() -> pd.DataFrame:
    print("⬇  World Happiness Report 2024 …", end=" ", flush=True)
    r = requests.get(WHR_URL, timeout=60, headers=HEADERS)
    r.raise_for_status()
    path = DATA_DIR / "WHR2024_raw.csv"
    path.write_bytes(r.content)
    df = pd.read_csv(path)
    print(f"done. ({len(df)} rows, {len(df.columns)} cols)")
    return df


def parse_whr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to PERMA+H schema.

    Handles two file shapes:
      • Cross-section (this mirror): no year column, one row per country.
        We use the data as-is.
      • Panel (official WHR .xls): has a year column; we take the most
        recent observation per country.
    """
    # ── Identify country column ──────────────────────────────────────────────
    country_col = next((c for c in COUNTRY_COL_CANDIDATES if c in df.columns), None)
    if country_col is None:
        # Fallback: any column whose lowered name contains "country"
        country_col = next(
            (c for c in df.columns if "country" in c.lower()), None
        )
    if country_col is None:
        raise ValueError(
            f"Cannot identify country column. Found: {list(df.columns)}"
        )

    # ── Identify year column (optional) ─────────────────────────────────────
    year_col = next((c for c in YEAR_COL_CANDIDATES if c in df.columns), None)

    # ── Subset to most recent year if panel ─────────────────────────────────
    if year_col:
        df_cs = (
            df.sort_values(year_col)
              .groupby(country_col, as_index=False)
              .last()
        )
    else:
        # Already a cross-section — just copy
        df_cs = df.copy()

    # ── Rename columns to PERMA+H schema (case-insensitive match) ───────────
    rename = {}
    for c in df_cs.columns:
        key = c.strip().lower()
        if key in WHR_COLUMN_MAP:
            rename[c] = WHR_COLUMN_MAP[key]

    df_cs = df_cs.rename(columns=rename)
    df_cs = df_cs.rename(columns={country_col: "country"})
    if year_col:
        df_cs = df_cs.rename(columns={year_col: "year"})

    # ── Keep only mapped columns + identifiers ──────────────────────────────
    id_cols = [col for col in ["country", "year"] if col in df_cs.columns]
    keep = id_cols + [
        lbl for lbl in WHR_COLUMN_MAP.values() if lbl in df_cs.columns
    ]
    # De-duplicate while preserving order (a label can appear via two aliases)
    seen = set()
    keep_dedup = [c for c in keep if not (c in seen or seen.add(c))]
    df_cs = df_cs[keep_dedup].copy()

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"   WHR cross-section: {len(df_cs)} countries")
    _report_missingness(df_cs, PERMA_COLS, label="WHR")
    return df_cs.reset_index(drop=True)


# ─── OECD BLI ─────────────────────────────────────────────────────────────────

def download_oecd_bli() -> pd.DataFrame:
    print("\n⬇  OECD Better Life Index …", end=" ", flush=True)
    try:
        r = requests.get(OECD_BLI_URL, timeout=30, headers=HEADERS)
        r.raise_for_status()
        path = DATA_DIR / "oecd_bli_raw.csv"
        path.write_bytes(r.content)
        df = pd.read_csv(path, low_memory=False)
        print(f"done. ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"failed ({e}). Skipping OECD BLI.")
        return pd.DataFrame()


def parse_oecd_bli(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot BLI long → wide: one row per country, one column per indicator.
    We keep the average score across genders (Total) where applicable.
    """
    if df.empty:
        return df

    df.columns = [c.strip() for c in df.columns]

    possible_country    = ["Country", "COUNTRY", "country"]
    possible_indicator  = ["Indicator", "INDICATOR", "indicator"]
    possible_value      = ["Value", "VALUE", "value"]
    possible_inequality = ["Inequality", "INEQUALITY", "inequality"]

    country_col    = next((c for c in possible_country    if c in df.columns), None)
    indicator_col  = next((c for c in possible_indicator  if c in df.columns), None)
    value_col      = next((c for c in possible_value      if c in df.columns), None)
    inequality_col = next((c for c in possible_inequality if c in df.columns), None)

    if not all([country_col, indicator_col, value_col]):
        print(f"   OECD BLI columns not recognized: {list(df.columns)[:8]}")
        return pd.DataFrame()

    if inequality_col:
        df = df[df[inequality_col].str.strip().str.lower() == "total"].copy()

    df_sub = df[df[indicator_col].isin(OECD_INDICATOR_MAP.keys())].copy()
    df_sub["_label"] = df_sub[indicator_col].map(OECD_INDICATOR_MAP)

    df_wide = (
        df_sub
        .groupby([country_col, "_label"])[value_col]
        .mean()
        .unstack("_label")
        .reset_index()
        .rename(columns={country_col: "country_oecd"})
    )

    print(f"   OECD BLI: {len(df_wide)} countries, "
          f"{[c for c in df_wide.columns if c != 'country_oecd']}")
    return df_wide


# ─── World Bank ───────────────────────────────────────────────────────────────

WB_API = "https://api.worldbank.org/v2/country/all/indicator"


def _fetch_wb_indicator(indicator: str, label: str) -> pd.DataFrame:
    url = f"{WB_API}/{indicator}?format=json&per_page=300&mrv=1"
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        r.raise_for_status()
        payload = r.json()
        records = payload[1] if len(payload) > 1 else []
        rows = [
            {
                "country":           rec["country"]["value"],
                "iso3":              rec["countryiso3code"],
                label:               rec["value"],
                f"{label}_year":     rec.get("date"),
            }
            for rec in records
            if rec.get("value") is not None
            and rec.get("countryiso3code")
        ]
        print(f"     {label:<40} {len(rows)} countries")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"     {label:<40} failed ({e})")
        return pd.DataFrame(columns=["country", label])


def download_world_bank() -> pd.DataFrame:
    print("\n⬇  World Bank supplemental indicators …")
    frames = []
    for indicator, label in WB_INDICATORS.items():
        frame = _fetch_wb_indicator(indicator, label)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        join_on = ["country", "iso3"] if "iso3" in f.columns else ["country"]
        merged = pd.merge(merged, f, on=join_on, how="outer")

    path = DATA_DIR / "worldbank_raw.csv"
    merged.to_csv(path, index=False)
    return merged


# ─── Merge ────────────────────────────────────────────────────────────────────

def _normalise_country(s: pd.Series) -> pd.Series:
    """Lowercase + strip for fuzzy country name matching."""
    return s.str.lower().str.strip()


def merge_sources(
    whr: pd.DataFrame,
    bli: pd.DataFrame,
    wb:  pd.DataFrame,
) -> pd.DataFrame:
    merged = whr.copy()
    merged["_key"] = _normalise_country(merged["country"])

    if not bli.empty:
        bli = bli.copy()
        bli["_key"] = _normalise_country(bli["country_oecd"])
        drop_cols = [c for c in bli.columns if c in merged.columns and c != "_key"]
        bli = bli.drop(columns=drop_cols + ["country_oecd"], errors="ignore")
        merged = pd.merge(merged, bli, on="_key", how="left")

    if not wb.empty and len(wb) > 0:
        wb = wb.copy()
        wb["_key"] = _normalise_country(wb["country"])
        drop_cols = ["country", "iso3"] + [
            c for c in wb.columns if c in merged.columns and c != "_key"
        ]
        wb = wb.drop(columns=drop_cols, errors="ignore")
        merged = pd.merge(merged, wb, on="_key", how="left")

    merged = merged.drop(columns=["_key"], errors="ignore")
    return merged


# ─── Utilities ────────────────────────────────────────────────────────────────

def _report_missingness(df: pd.DataFrame, cols: list, label: str = ""):
    present = [c for c in cols if c in df.columns]
    if not present:
        print(f"   [{label}] None of the target columns found.")
        return
    print(f"   [{label}] Missingness in core PERMA+H columns:")
    for c in present:
        n = df[c].isna().sum()
        pct = 100 * n / len(df)
        print(f"     {c:<35} {n:>3} missing ({pct:.0f}%)")


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PERMA_COLS if c in df.columns]
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    desc["n_valid"] = df[cols].notna().sum()
    return desc.round(3)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  Wellbeing Data Downloader")
    print("  Target: f-factor analysis (PERMA+H schema)")
    print("═" * 60 + "\n")

    # 1. WHR (primary)
    whr_raw = download_whr()
    whr     = parse_whr(whr_raw)

    # 2. OECD BLI (supplemental)
    bli_raw = download_oecd_bli()
    bli     = parse_oecd_bli(bli_raw)

    # 3. World Bank (supplemental)
    wb = download_world_bank()

    # 4. Merge
    print("\n🔀 Merging sources …")
    merged = merge_sources(whr, bli, wb)

    # 5. Save
    out_path = DATA_DIR / "wellbeing_merged.csv"
    merged.to_csv(out_path, index=False)

    # 6. Report
    print(f"\n✓  Merged dataset saved → {out_path}")
    print(f"   Shape : {merged.shape[0]} countries × {merged.shape[1]} variables")
    print(f"\n   Columns:\n   " + "\n   ".join(merged.columns.tolist()))

    print("\n" + "─" * 60)
    print("  Summary statistics for PERMA+H columns")
    print("─" * 60)
    summ = _summary_table(merged)
    if not summ.empty:
        print(summ.to_string())

    # Count complete PERMA+H cases (no missing in any of the 6 dimensions)
    perma_present = [c for c in PERMA_COLS if c in merged.columns]
    complete = merged[perma_present].dropna().shape[0]
    print(f"\n   Complete PERMA+H observations: {complete} / {len(merged)}")
    print("\n   → Run 02_analyze_wellbeing_f.py to extract f and generate plots.\n")


if __name__ == "__main__":
    main()
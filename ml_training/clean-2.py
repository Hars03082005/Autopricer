"""
ml_training/clean-1.py
AutoPricer Preprocessing Pipeline â€” clean-1
Source  : overall.csv
Output  : processed_overall.csv  +  processed_overall_report.json
Design principles
-----------------
* Maximize data quality, consistency, and generalization.
* Remove impossible records; never silently clip/correct them.
* Retain meaningful location features (locality, rto, pincode).
* Drop redundant derived columns (city, age_bucket, make_model_trim).
* Generate only km_per_year as an engineered feature; let tree models
  learn everything else natively.
* Distinguish truly unknown values from valid categories (certified,
  pincode stay as NaN when genuinely missing).
* Produce a full JSON audit report alongside every CSV.
    if pd.isna(value):
        return default
    s = re.sub(r"\s+", " ", str(value).strip().lower())
    return s if s not in {"", "nan", "none", "null"} else default
def _log_step(audit: list, step: str, before: int, after: int, detail: str = "") -> None:
    dropped = before - after
    msg = f"  [{step:<38}]  {before:>7,} â†’ {after:>7,}  (dropped {dropped:>6,})"
    if detail:
        msg += f"  | {detail}"
    print(msg)
    audit.append({"step": step, "rows_before": before, "rows_after": after,
                  "rows_dropped": dropped, "detail": detail})
def _fingerprint(path: Path) -> dict:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    s = path.stat()
    return {"filename": path.name, "sha256": sha.hexdigest(),
            "size_bytes": s.st_size, "size_mb": round(s.st_size / 1_048_576, 3)}
# â”€â”€ STAGE 1: LOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_raw(path: Path) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 1 â€” LOAD  |  {path.name}\n{DIV}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Names   : {list(df.columns)}")
    return df
def validate_schema(df: pd.DataFrame) -> None:
    print(f"\n{DIV}\nSTAGE 2 â€” SCHEMA VALIDATION\n{DIV}")
    missing_required = REQUIRED_COLS - set(df.columns)
    if missing_required:
        raise ValueError(f"Required source columns missing: {missing_required}")
    print(f"  All {len(REQUIRED_COLS)} required columns present âœ“")
    optional_present = [c for c in ["trim", "locality", "rto", "pincode", "color",
                                    "owner", "certified", "age", "year"] if c in df.columns]
    print(f"  Optional columns found: {optional_present}")
def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 3 â€” SELECT & RENAME\n{DIV}")
    available = [c for c in RAW_KEEP if c in df.columns]
    df = df[available].copy()
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    print(f"  Columns selected : {len(available)}  â†’  {list(df.columns)}")
    return df
def validate_price(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 4 â€” VALIDATE PRICE\n{DIV}")
    b = len(df)
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    n_non_num = int(df["selling_price"].isna().sum())
    df = df[df["selling_price"].notna()]
    n_range = int((~df["selling_price"].between(PRICE_MIN, PRICE_MAX)).sum())
    df = df[df["selling_price"].between(PRICE_MIN, PRICE_MAX)]
    _log_step(audit, "price_invalid", b, len(df),
              f"non-numeric={n_non_num}, out-of-range=[{PRICE_MIN:,}â€“{PRICE_MAX:,}]:{n_range}")
    print(f"  Range : â‚¹{df['selling_price'].min():,.0f} â€“ â‚¹{df['selling_price'].max():,.0f}")
    return df
def validate_age(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 5 â€” VALIDATE VEHICLE AGE\n{DIV}")
    b = len(df)
    if "age" in df.columns:
        df["vehicle_age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["vehicle_age"] = np.nan
    if "year" in df.columns:
        df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
        mask = df["vehicle_age"].isna() & df["year_num"].between(1990, CURRENT_YEAR)
        df.loc[mask, "vehicle_age"] = (CURRENT_YEAR - df.loc[mask, "year_num"]).clip(lower=0)
    n_missing = int(df["vehicle_age"].isna().sum())
    df = df[df["vehicle_age"].notna()]
    n_range = int((~df["vehicle_age"].between(AGE_MIN, AGE_MAX)).sum())
    df = df[df["vehicle_age"].between(AGE_MIN, AGE_MAX)]
    _log_step(audit, "age_invalid", b, len(df),
              f"missing={n_missing}, impossible=[0â€“{AGE_MAX}yrs]:{n_range}")
    print(f"  Range : {df['vehicle_age'].min():.0f} â€“ {df['vehicle_age'].max():.0f} years")
    return df
def validate_odometer(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 6 â€” VALIDATE ODOMETER\n{DIV}")
    b = len(df)
    df["odometer_reading"] = pd.to_numeric(df["odometer_raw"], errors="coerce")
    n_non_num = int(df["odometer_reading"].isna().sum())
    df = df[df["odometer_reading"].notna()]
    n_range = int((~df["odometer_reading"].between(ODO_MIN, ODO_MAX)).sum())
    df = df[df["odometer_reading"].between(ODO_MIN, ODO_MAX)]
    _log_step(audit, "odometer_invalid", b, len(df),
              f"non-numeric={n_non_num}, out-of-range=[{ODO_MIN:,}â€“{ODO_MAX:,}]:{n_range}")
    return df
def normalize_brand(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 7 â€” NORMALIZE BRAND\n{DIV}")
    b = len(df)
    df["brand"] = df["brand_raw"].apply(_norm).replace(BRAND_ALIAS)
    n_unknown = int((~df["brand"].isin(OEM_ALLOWLIST)).sum())
    df = df[df["brand"].isin(OEM_ALLOWLIST)]
    _log_step(audit, "brand_not_in_allowlist", b, len(df),
              f"not-in-allowlist={n_unknown}")
    print(f"  Unique brands : {df['brand'].nunique()}")
    print(df["brand"].value_counts().head(10).to_string())
    return df
def normalize_model(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 8 â€” NORMALIZE MODEL\n{DIV}")
    b = len(df)
    df["model"] = df["model_raw"].apply(_norm)
    n_unknown = int((df["model"] == "unknown").sum())
    df = df[df["model"] != "unknown"]
    _log_step(audit, "model_blank", b, len(df), f"blank/unknown={n_unknown}")
    print(f"  Unique models : {df['model'].nunique()}")
    return df
def normalize_variant(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 9 â€” NORMALIZE VARIANT\n{DIV}")
    if "variant_raw" in df.columns:
        df["variant"] = df["variant_raw"].apply(_norm)
    else:
        df["variant"] = "unknown"
    n_unk = (df["variant"] == "unknown").sum()
    print(f"  Unique variants   : {df['variant'].nunique()}")
    print(f"  Unknown/missing   : {n_unk:,} ({n_unk / len(df) * 100:.1f}%)  â€” retained")
    return df
def normalize_fuel(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 10 â€” NORMALIZE FUEL TYPE\n{DIV}")
    b = len(df)
    if "fuel_raw" in df.columns:
        df["fuel_type"] = df["fuel_raw"].apply(_norm).replace(FUEL_ALIAS)
    else:
        df["fuel_type"] = "petrol"
    n_invalid = int((~df["fuel_type"].isin(VALID_FUEL)).sum())
    df = df[df["fuel_type"].isin(VALID_FUEL)]
    _log_step(audit, "fuel_invalid", b, len(df), f"invalid={n_invalid}")
    print(df["fuel_type"].value_counts().to_string())
    return df
def normalize_transmission(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 11 â€” NORMALIZE TRANSMISSION\n{DIV}")
    b = len(df)
    if "trans_raw" in df.columns:
        df["transmission"] = df["trans_raw"].apply(_norm).replace(TRANS_ALIAS)
    else:
        df["transmission"] = "manual"
    n_invalid = int((~df["transmission"].isin(VALID_TRANS)).sum())
    df = df[df["transmission"].isin(VALID_TRANS)]
    _log_step(audit, "transmission_invalid", b, len(df), f"invalid={n_invalid}")
    print(df["transmission"].value_counts().to_string())
    return df
def normalize_seller_type(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 12 â€” NORMALIZE SELLER TYPE\n{DIV}")
    def _seller(value) -> str:
        s = _norm(value)
        if any(k in s for k in _DEALER_KEYWORDS):
            return "dealer"
        if any(k in s for k in _INDIVIDUAL_KEYWORDS):
            return "individual"
        return "unknown"
    if "seller_type_raw" in df.columns:
        df["seller_type"] = df["seller_type_raw"].apply(_seller)
    else:
        df["seller_type"] = "unknown"
    print(df["seller_type"].value_counts(dropna=False).to_string())
    return df
def _normalize_locality(raw) -> str:
    if pd.isna(raw):
        return "unknown"
    s = re.sub(r"\s+", " ", str(raw).strip().lower())
    if s in {"", "nan", "none", "null"}:
        return "unknown"
    # 1. Exact alias match
    if s in LOCALITY_ALIAS:
        return LOCALITY_ALIAS[s]
    # 2. Keyword substring match
    for keyword, canonical in _KEYWORD_MAP:
        if keyword in s:
            return canonical
    return s
def normalize_locality(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 13 â€” NORMALIZE LOCALITY\n{DIV}")
    if "locality" in df.columns:
        df["locality"] = df["locality"].apply(_normalize_locality)
    else:
        df["locality"] = "unknown"
    n_unk = (df["locality"] == "unknown").sum()
    print(f"  Unique localities : {df['locality'].nunique()}")
    print(f"  Unknown           : {n_unk:,} ({n_unk / len(df) * 100:.1f}%)")
    print("  Top 15:")
    print(df["locality"].value_counts().head(15).to_string())
    return df
def normalize_rto(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 14 â€” NORMALIZE RTO\n{DIV}")
    if "rto" in df.columns:
        df["rto"] = df["rto"].apply(lambda x: _norm(x, "unknown"))
    else:
        df["rto"] = "unknown"
    n_unk = (df["rto"] == "unknown").sum()
    print(f"  Unique RTOs : {df['rto'].nunique()}")
    print(f"  Unknown     : {n_unk:,}")
    print("  Top 10:")
    print(df["rto"].value_counts().head(10).to_string())
    return df
def normalize_color(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 15 â€” NORMALIZE COLOR\n{DIV}")
    def _color(value) -> str:
        s = _norm(value, "unknown")
        if s == "unknown":
            return "unknown"
        if s in VALID_COLORS:
            return s
        if s in COLOR_ALIAS:
            return COLOR_ALIAS[s]
        for base in VALID_COLORS:
            if base in s:
                return base
        return "unknown"
    if "color" in df.columns:
        df["color"] = df["color"].apply(_color)
    else:
        df["color"] = "unknown"
    print(df["color"].value_counts(dropna=False).to_string())
    return df
def normalize_certified(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 16 â€” NORMALIZE CERTIFIED\n{DIV}")
    _MAP = {"yes": 1.0, "1": 1.0, "true": 1.0, "y": 1.0,
            "no": 0.0,  "0": 0.0, "false": 0.0, "n": 0.0}
    if "certified" in df.columns:
        df["certified"] = df["certified"].apply(_norm).map(_MAP)
    else:
        df["certified"] = np.nan
    c1  = int((df["certified"] == 1.0).sum())
    c0  = int((df["certified"] == 0.0).sum())
    nan = int(df["certified"].isna().sum())
    print(f"  Certified = 1     : {c1:,}")
    print(f"  Certified = 0     : {c0:,}")
    print(f"  Truly unknownâ†’NaN : {nan:,}  (tree models handle NaN natively)")
    return df
def normalize_pincode(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 17 â€” NORMALIZE PINCODE\n{DIV}")
    if "pincode" in df.columns:
        df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
        bad = df["pincode"].notna() & ~df["pincode"].between(100_000, 999_999)
        n_bad = int(bad.sum())
        df.loc[bad, "pincode"] = np.nan
        n_valid = int(df["pincode"].notna().sum())
        n_nan   = int(df["pincode"].isna().sum())
        print(f"  Valid pincodes  : {n_valid:,}")
        print(f"  Invalid â†’ NaN   : {n_bad:,}")
        print(f"  Total NaN       : {n_nan:,}  (tree models handle NaN natively)")
    else:
        df["pincode"] = np.nan
        print("  pincode column not present â€” set to NaN")
    return df
def normalize_owner_count(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 18 â€” NORMALIZE OWNER COUNT\n{DIV}")
    def _parse(value) -> int:
        s = _norm(value)
        if s == "unknown":
            return 1
        try:
            return max(1, min(int(float(s)), 6))
        except Exception:
            return 1
    if "owner_raw" in df.columns:
        df["owner_count"] = df["owner_raw"].apply(_parse)
    else:
        df["owner_count"] = 1
    print(df["owner_count"].value_counts().sort_index().to_string())
    return df
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 19 â€” FEATURE ENGINEERING\n{DIV}")
    safe_age = df["vehicle_age"].clip(lower=0.5)
    df["km_per_year"] = (df["odometer_reading"] / safe_age).clip(0, 100_000).round(1)
    print(f"  km_per_year â€” min={df['km_per_year'].min():,.0f}  "
          f"mean={df['km_per_year'].mean():,.0f}  "
          f"max={df['km_per_year'].max():,.0f}")
    print("  NOTE: age_bucket not created â€” tree models learn splits natively")
    print("  NOTE: make_model_trim not created â€” brand+model+variant already present")
    return df
# â”€â”€ STAGE 20: DEDUPLICATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DUPLICATE_COLUMNS = [
    "brand",
    "model",
    "variant",
    "vehicle_age",
    "odometer_reading",
    "owner_count",
    "fuel_type",
    "transmission",
    "seller_type",
    "locality",
    "selling_price",
]
def deduplicate(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    print(f"\n{DIV}\nSTAGE 20 â€” DEDUPLICATION\n{DIV}")
    b = len(df)
    dedup_cols = [c for c in DUPLICATE_COLUMNS if c in df.columns]
    print(f"  Deduplicating on {len(dedup_cols)} columns: {dedup_cols}")
    print(f"  Rows BEFORE deduplication : {b:,}")
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    a = len(df)
    print(f"  Rows AFTER deduplication  : {a:,}")
    _log_step(audit, "exact_duplicates_removed", b, a,
              f"duplicates={b - a}")
    return df
def generate_report(df: pd.DataFrame, src: Path, audit: list, duration: float) -> dict:
    print(f"\n{DIV}\nSTAGE 21 â€” AUDIT REPORT\n{DIV}")
    miss = {}
    for col in ML_FEATURES:
        if col in df.columns:
            n = int(df[col].isna().sum())
            miss[col] = {"missing_count": n, "missing_pct": round(n / len(df) * 100, 2)}
    print("  MISSING VALUE SUMMARY")
    for col, info in miss.items():
        if info["missing_count"] > 0:
            print(f"    {col:<25} {info['missing_count']:>7,}  ({info['missing_pct']:.1f}%)")
    cat_cols = ["brand", "model", "variant", "locality", "rto",
                "fuel_type", "transmission", "seller_type", "color"]
    cats = {}
    for col in cat_cols:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            cats[col] = {"unique": int(df[col].nunique()),
                         "top_10": {str(k): int(v) for k, v in vc.head(10).items()}}
    num_cols = ["vehicle_age", "odometer_reading", "km_per_year",
                "owner_count", "certified", "pincode", "selling_price"]
    nums = {}
    for col in num_cols:
        if col in df.columns:
            s = df[col].describe()
            nums[col] = {k: round(float(v), 2) for k, v in s.items()}
    print("\n  NUMERIC FEATURE STATISTICS")
    for col in ["vehicle_age", "odometer_reading", "km_per_year", "selling_price"]:
        if col in nums:
            s = nums[col]
            print(f"    {col:<25}  mean={s.get('mean', 0):>10,.0f}  "
                  f"std={s.get('std', 0):>10,.0f}  "
                  f"min={s.get('min', 0):>8,.0f}  "
                  f"max={s.get('max', 0):>12,.0f}")
    return {
        "script":                    SCRIPT_NAME,
        "source_file":               src.name,
        "processed_at":              datetime.now().isoformat(),
        "current_year_used":         CURRENT_YEAR,
        "processing_duration_sec":   round(duration, 2),
        "dataset_fingerprint":       _fingerprint(src),
        "rows_raw":                  audit[0]["rows_after"] if audit else 0,
        "rows_output":               len(df),
        "columns_output":            len(ML_FEATURES),
        "features":                  ML_FEATURES,
        "removed_features":          ["city", "age_bucket", "make_model_trim", "year"],
        "pipeline_steps":            audit,
        "missing_value_summary":     miss,
        "categorical_distributions": cats,
        "numeric_statistics":        nums,
    }
def save_outputs(df: pd.DataFrame, report: dict, out_name: str) -> None:
    print(f"\n{DIV}\nSTAGE 22 â€” SAVE OUTPUTS\n{DIV}")
    csv_path    = DATA_DIR / f"processed_{out_name}.csv"
    report_path = DATA_DIR / f"processed_{out_name}_report.json"
    out_cols = [c for c in ML_FEATURES if c in df.columns]
    df[out_cols].to_csv(csv_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  CSV    : {csv_path}  ({len(df):,} rows Ã— {len(out_cols)} cols)")
    print(f"  Report : {report_path}")
    print(f"\n  Output columns: {out_cols}")
# â”€â”€ PIPELINE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def process_file(out_name: str, in_path: Path) -> None:
    t0    = time.perf_counter()
    audit: list[dict] = []
    print(f"\n{'#' * 80}")
    print(f"  AutoPricer Preprocessing Pipeline  ({SCRIPT_NAME})")
    print(f"  Source  : {in_path.name}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}")
    df = load_raw(in_path)
    validate_schema(df)
    df = select_and_rename(df)
    # Seed audit with raw count
    audit.append({"step": "raw_loaded", "rows_before": 0, "rows_after": len(df),
                  "rows_dropped": 0, "detail": in_path.name})
    df = validate_price(df, audit)
    df = validate_age(df, audit)
    df = validate_odometer(df, audit)
    df = normalize_brand(df, audit)
    df = normalize_model(df, audit)
    df = normalize_variant(df)
    df = normalize_fuel(df, audit)
    df = normalize_transmission(df, audit)
    df = normalize_seller_type(df)
    df = normalize_locality(df)
    df = normalize_rto(df)
    df = normalize_color(df)
    df = normalize_certified(df)
    df = normalize_pincode(df)
    df = normalize_owner_count(df)
    df = engineer_features(df)
    df = deduplicate(df, audit)
    duration = time.perf_counter() - t0
    report   = generate_report(df, in_path, audit, duration)
    save_outputs(df, report, out_name)
    print(f"\n{'#' * 80}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output : {len(df):,} records  Ã—  {len(ML_FEATURES)} features")
    print(f"  Time   : {duration:.2f}s")
    print(f"{'#' * 80}\n")
def main() -> None:
    for out_name, in_path in INPUT_FILES.items():
        if not in_path.exists():
            print(f"WARNING: {in_path} not found â€” skipping")
            continue
        process_file(out_name, in_path)
if __name__ == "__main__":
    main()

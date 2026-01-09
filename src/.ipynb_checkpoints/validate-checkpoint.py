from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# This module serves as a data validation gate before modeling - a reproducible governance step
# Compared to validate_old.py, it has a modular design, it separates validation logic
# ('validate_application_train') from execution ('main')

# Typed and documented - uses type hints and constants
REQUIRED_COLS = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED"]
LEAKAGE_KEYWORDS = ["TARGET", "DEFAULT", "OVERDUE", "DELINQ", "DPD", "LATE", "PAST_DUE"]

def validate_application_train(df: pd.DataFrame, null_warn_threshold: float = 0.80) -> dict:
    report: dict = {"checks": {}, "warnings": []}

    # Required columns (Assert target exists + schema checks for the other required columns; of validate_old.py)
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    report["checks"]["missing_required_cols"] = missing_required
    if missing_required:
        report["warnings"].append(f"Missing required columns: {missing_required}")

    # Target distribution (NEW)
    if "TARGET" in df.columns:
        vc = df["TARGET"].value_counts(dropna=False).to_dict()
        report["checks"]["target_value_counts"] = vc
        report["checks"]["default_rate"] = float(df["TARGET"].mean())

    # Duplicate keys (NEW)
    if "SK_ID_CURR" in df.columns:
        dup_keys = int(df["SK_ID_CURR"].duplicated().sum())
        report["checks"]["duplicate_SK_ID_CURR"] = dup_keys
        if dup_keys > 0:
            report["warnings"].append(f"Found {dup_keys} duplicate SK_ID_CURR values")

    # Missingness (Null threshold warnings of validate_old.py)
    miss = df.isna().mean().sort_values(ascending=False)
    high_missing_cols = miss[miss >= null_warn_threshold].index.tolist()
    report["checks"]["high_missing_cols_ge_threshold"] = {
        "threshold": null_warn_threshold,
        "count": len(high_missing_cols),
        "cols": high_missing_cols[:100],  # cap
    }

    # Simple numeric sanity checks (numeric range sanity checks - e.g., AMT_INCOME_TOTAL >=0, DAYS_BIRTH < 0; of validate_old.py)
    if "AMT_INCOME_TOTAL" in df.columns:
        neg_income = int((df["AMT_INCOME_TOTAL"] < 0).sum())
        report["checks"]["negative_income_count"] = neg_income
        if neg_income > 0:
            report["warnings"].append(f"Negative income values: {neg_income}")

    if "DAYS_BIRTH" in df.columns:
        # In Home Credit, days are negative (days before application)
        nonneg_birth = int((df["DAYS_BIRTH"] >= 0).sum())
        report["checks"]["non_negative_DAYS_BIRTH_count"] = nonneg_birth
        if nonneg_birth > 0:
            report["warnings"].append(f"Unexpected non-negative DAYS_BIRTH values: {nonneg_birth}")

    # Leakage scan (heuristic) - NEW
    leakage_cols = []
    for c in df.columns:
        uc = c.upper()
        if any(k in uc for k in LEAKAGE_KEYWORDS) and c != "TARGET":
            leakage_cols.append(c)
    report["checks"]["potential_leakage_cols_heuristic"] = leakage_cols

    return report

def main():
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / "application_train.csv"
    out_path = root / "reports" / "data_quality.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    report = validate_application_train(df)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report to {out_path}")
    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f" - {w}")

if __name__ == "__main__":
    main()

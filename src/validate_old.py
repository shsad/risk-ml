import pandas as pd
import json
import os

# This module serves as a data validation gate before modeling - a reproducible governance step

# load application_train.csv

def validate_data(path="../data/raw/application_train.csv", report_path="../reports/data_quality.json"):
    report = {"status": "OK", "checks": {}}

    # Load
    try:
        df = pd.read_csv(path)
        report["checks"]["load"] = f"Loaded {df.shape[0]} rows, {df.shape[1]} columns"
    except Exception as e:
        report["status"] = "FAIL"
        report["checks"]["load"] = f"Error loading file: {e}"
        save_report(report, report_path)
        return

    # Assert target existence
    if "TARGET" not in df.columns:
        report["status"] = "FAIL"
        report["checks"]["target"] = "TARGET column missing"
    else:
        report["checks"]["target"] = "TARGET column present"

    # Schema checks (a few required column)
    required_cols = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "DAYS_BIRTH"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report["status"] = "FAIL"
        report["checks"]["schema"] = f"Missing required columns: {missing_cols}"
    else:
        report["checks"]["schema"] = "All required columns present"

    # Null threshold warnings
    null_pct = df.isnull().mean() * 100
    high_null = null_pct[null_pct > 80].to_dict()
    report["checks"]["nulls"] = {"columns_over_80pct": high_null}

    # Numeric range sanity checks (e.g., AMT_INCOME_TOTAL >= 0, DAYS_BIRTH < 0)
    numeric_issues = {}
    if "AMT_INCOME_TOTAL" in df.columns:
        bad_income = (df["AMT_INCOME_TOTAL"] < 0).sum()
        if bad_income > 0:
            numeric_issues["AMT_INCOME_TOTAL"] = f"{bad_income} negative values"
    if "DAYS_BIRTH" in df.columns:
        bad_birth = (df["DAYS_BIRTH"] >= 0).sum()
        if bad_birth > 0:
            numeric_issues["DAYS_BIRTH"] = f"{bad_birth} non-negative values"
    report["checks"]["numeric_ranges"] = numeric_issues or "All numeric ranges OK"

    # Save a small JSON report to reports/data_quality.json
    save_report(report, report_path)
    print(f"Validation complete. Report saved to {report_path}")

def save_report(report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    validate_data()
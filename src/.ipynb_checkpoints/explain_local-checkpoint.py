from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from src.data_load import load_application_train

def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "application_train.csv"
    model_path = root / "reports" / "xgb_model.joblib"
    out_path = root / "reports" / "local_examples.csv"

    df = load_application_train(str(data_path))

    pipe = joblib.load(model_path)

    # Pick 3 examples: high, medium, low predicted risk
    X = df.drop(columns=["target"], errors="ignore")
    probs = pipe.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["pred_pd"] = probs

    examples = pd.concat([
        df_out.sort_values("pred_pd", ascending=False).head(1),
        df_out.sort_values("pred_pd", ascending=True).head(1),
        df_out.iloc[[df_out["pred_pd"].sub(df_out["pred_pd"].median()).abs().idxmin()]],
    ])

    cols_to_show = ["sk_id_curr", "target", "pred_pd",
                    "amt_income_total", "amt_credit", "amt_annuity",
                    "days_birth", "days_employed",
                    "name_income_type", "name_education_type", "occupation_type"]

    cols_to_show = [c for c in cols_to_show if c in examples.columns]
    examples[cols_to_show].to_csv(out_path, index=False)

    print(examples[cols_to_show].to_string(index=False))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

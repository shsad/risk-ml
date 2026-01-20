from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_load import load_application_train
from src.features import train_val_split

def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "application_train.csv"
    model_path = root / "reports" / "xgb_model.joblib"

    df = load_application_train(str(data_path))
    X_train, X_val, y_train, y_val = train_val_split(df)

    pipe = joblib.load(model_path)
    y_prob = pipe.predict_proba(X_val)[:, 1]

    # Group fairness proxy (illustrative)
    if "code_gender" not in X_val.columns:
        print("code_gender not in validation set columns.")
        return

    tmp = X_val.copy()
    tmp["y_true"] = y_val.values
    tmp["y_prob"] = y_prob

    for g, grp in tmp.groupby("code_gender"):
        if grp["y_true"].nunique() < 2:
            continue
        roc = roc_auc_score(grp["y_true"], grp["y_prob"])
        pr = average_precision_score(grp["y_true"], grp["y_prob"])
        print(f"{g}: n={len(grp)} | ROC-AUC={roc:.3f} | PR-AUC={pr:.3f}")

if __name__ == "__main__":
    main()

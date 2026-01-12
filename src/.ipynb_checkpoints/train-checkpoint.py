from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data_load import load_application_train
from src.features import build_preprocessor, train_val_split


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "application_train.csv"
    model_path = root / "reports" / "baseline_logreg.joblib"

    df = load_application_train(data_path)
    X_train, X_val, y_train, y_val = train_val_split(df)

    preprocessor = build_preprocessor()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    y_val_pred = pipe.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, y_val_pred)
    pr = average_precision_score(y_val, y_val_pred)

    print(f"Validation ROC-AUC: {roc:.4f}")
    print(f"Validation PR-AUC:  {pr:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

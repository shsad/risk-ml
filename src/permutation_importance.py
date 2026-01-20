from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, make_scorer

from src.data_load import load_application_train
from src.features import train_val_split

def pr_auc_scorer(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, y_prob)

def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "application_train.csv"
    model_path = root / "reports" / "xgb_model.joblib"
    out_csv = root / "reports" / "permutation_importance_pr_auc.csv"

    df = load_application_train(str(data_path))
    X_train, X_val, y_train, y_val = train_val_split(df)

    pipe = joblib.load(model_path)

    result = permutation_importance(
        pipe,
        X_val,
        y_val,
        scoring=pr_auc_scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=2,
    )

    # Feature names again
    # pre = pipe.named_steps["preprocess"]
    # num_names = pre.transformers_[0][2]
    # ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    # cat_names = pre.transformers_[1][2]
    # ohe_names = ohe.get_feature_names_out(cat_names).tolist()
    # feature_names = list(num_names) + list(ohe_names)

    # feature_names = pipe[:-1].get_feature_names_out()

    # print(len(feature_names))
    # print(len(result.importances_mean))

    feature_names = X_val.columns

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(out_csv, index=False)

    print("Top 20 permutation importances (PR-AUC drop):")
    print(df_imp.head(20).to_string(index=False))
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
